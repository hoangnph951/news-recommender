"""Mô hình gợi ý tin tức: news encoder, user encoder, click predictor (slide L9).

Mọi biến thể dùng chung một lớp `NewsRecModel`, bật/tắt từng thành phần bằng config
để chạy baseline và ablation trong cùng một khung:

  model.text_encoder : glove (NRMS gốc: word embedding + multi-head self-attention)
                       plm   (mô hình ngôn ngữ pretrain, fine-tune vài tầng trên cùng)
  model.views        : các view của tin (text, category, subcategory, entity),
                       gộp bằng attention giữa các view như NAML
  model.user_encoder : attn (NRMS) | mean (trung bình, late fusion) | cand (candidate-aware)
  model.popularity   : cộng thêm điểm phổ biến/độ mới của tin, trọng số do user quyết định
                       (ý tưởng PP-Rec; hybrid có trọng số, slide L7)
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from newsrec.plm_cache import PLMCache, build_cache, cache_dir, load_plm, transformer_layers

NEG = -1e4  # giá trị che (mask) an toàn cả khi chạy fp16


class AdditiveAttention(nn.Module):
    """Gộp một chuỗi vector thành 1 vector bằng trọng số attention học được."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):  # x: [B, L, D], mask: [B, L] True = hợp lệ
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, NEG)
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1), weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.qkv = nn.Linear(in_dim, 3 * num_heads * head_dim)

    def forward(self, x, mask=None):  # [B, L, D] -> [B, L, heads * head_dim]
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_mask = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.transpose(1, 2).reshape(B, L, -1)


# ----------------------------------------------------------------------------- news encoder
class GloveTextEncoder(nn.Module):
    """News encoder của NRMS: word embedding -> self-attention -> additive attention."""

    def __init__(self, title_tokens, vocab_size, out_dim, cfg, pretrained=None):
        super().__init__()
        self.register_buffer("tokens", torch.as_tensor(title_tokens, dtype=torch.long), persistent=False)
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained))
        self.dropout = nn.Dropout(cfg.get("dropout", 0.2))
        self.self_attn = MultiHeadSelfAttention(300, cfg["num_heads"], cfg["head_dim"])
        dim = cfg["num_heads"] * cfg["head_dim"]
        self.attn = AdditiveAttention(dim, cfg.get("attn_hidden", 200))
        self.out = nn.Identity() if dim == out_dim else nn.Linear(dim, out_dim)

    def forward(self, ids):
        tokens = self.tokens[ids]
        mask = tokens != 0
        mask[:, 0] = True  # tin padding: tránh hàng toàn bị che
        x = self.dropout(self.embedding(tokens))
        x = self.dropout(self.self_attn(x, mask))
        return self.out(self.attn(x, mask)[0])


class PLMTextEncoder(nn.Module):
    """Chạy `finetune_layers` tầng trên cùng của PLM trên hidden state đã cache, rồi gộp bằng attention."""

    def __init__(self, cache, out_dim, cfg, plm_layers=None):
        super().__init__()
        self.cache = cache  # không phải nn.Module nên không nằm trong state_dict
        hidden = cache.meta["hidden_size"] if cache else cfg["hidden_size"]
        self.layers = nn.ModuleList(plm_layers or [])
        if cfg.get("freeze"):  # ablation PLM đóng băng, dùng lại đúng cache của mô hình chính
            self.layers.requires_grad_(False)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.1))
        self.attn = AdditiveAttention(hidden, cfg.get("attn_hidden", 200))
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, ids):
        device = self.out.weight.device
        h, mask = self.cache.lookup(ids, device)
        h = h.float()
        mask = mask.clone()
        mask[:, 0] = True
        ext = (~mask)[:, None, None, :].to(h.dtype) * NEG
        for layer in self.layers:
            out = layer(h, attention_mask=ext)
            h = out[0] if isinstance(out, tuple) else out
        return self.out(self.attn(self.dropout(h), mask)[0])


class NewsEncoder(nn.Module):
    def __init__(self, text, news_arrays, entity_emb, dim, views):
        super().__init__()
        self.views = list(views)
        self.text = text
        if "category" in self.views:
            self.register_buffer("category", torch.as_tensor(news_arrays["category"], dtype=torch.long), persistent=False)
            self.cat = nn.Sequential(nn.Embedding(int(news_arrays["category"].max()) + 1, 100), nn.Linear(100, dim), nn.ReLU())
        if "subcategory" in self.views:
            self.register_buffer("subcategory", torch.as_tensor(news_arrays["subcategory"], dtype=torch.long), persistent=False)
            self.subcat = nn.Sequential(nn.Embedding(int(news_arrays["subcategory"].max()) + 1, 100), nn.Linear(100, dim), nn.ReLU())
        if "entity" in self.views:
            self.register_buffer("entities", torch.as_tensor(news_arrays["entities"], dtype=torch.long), persistent=False)
            self.entity_emb = nn.Embedding.from_pretrained(torch.as_tensor(entity_emb), freeze=True, padding_idx=0)
            self.entity = nn.Sequential(nn.Linear(entity_emb.shape[1], dim), nn.ReLU())
        self.view_attn = AdditiveAttention(dim, 100) if len(self.views) > 1 else None

    def forward(self, ids):
        out = []
        for view in self.views:
            if view == "text":
                out.append(self.text(ids))
            elif view == "category":
                out.append(self.cat(self.category[ids]))
            elif view == "subcategory":
                out.append(self.subcat(self.subcategory[ids]))
            elif view == "entity":
                ent = self.entities[ids]
                m = (ent != 0).unsqueeze(-1).float()
                mean = (self.entity_emb(ent) * m).sum(1) / m.sum(1).clamp(min=1)
                out.append(self.entity(mean))
        if self.view_attn is None:
            return out[0]
        return self.view_attn(torch.stack([o.float() for o in out], dim=1))[0]


# ----------------------------------------------------------------------------- user encoder
class UserEncoder(nn.Module):
    """Trả về (vector user [B, D], trọng số trên lịch sử [B, H]) - trọng số dùng để giải thích gợi ý."""

    def __init__(self, kind, dim, cfg):
        super().__init__()
        self.kind = kind
        if kind in ("attn", "cand"):
            self.self_attn = MultiHeadSelfAttention(dim, cfg["num_heads"], cfg["head_dim"])
            self.proj = nn.Identity() if cfg["num_heads"] * cfg["head_dim"] == dim else nn.Linear(cfg["num_heads"] * cfg["head_dim"], dim)
            self.attn = AdditiveAttention(dim, cfg.get("attn_hidden", 200))
        if kind == "cand":
            self.cand_proj = nn.Linear(dim, dim, bias=False)
        if kind not in ("attn", "mean", "cand"):
            raise ValueError(f"user_encoder không hợp lệ: {kind}")

    def contextualize(self, hist, mask):
        if self.kind == "mean":
            return hist
        return self.proj(self.self_attn(hist, mask))

    def forward(self, hist, mask):
        mask = mask.clone()
        mask[:, -1] = True  # user không có lịch sử: dùng vector của tin padding (học được)
        if self.kind == "mean":
            w = mask.float() / mask.float().sum(1, keepdim=True)
            return torch.bmm(w.unsqueeze(1), hist).squeeze(1), w
        return self.attn(self.contextualize(hist, mask), mask)

    def candidate_aware(self, hist, mask, cands):
        """Vector user riêng cho từng ứng viên: attention trên lịch sử theo độ liên quan với ứng viên."""
        mask = mask.clone()
        mask[:, -1] = True
        ctx = self.contextualize(hist, mask)                                   # [B, H, D]
        scores = torch.einsum("bcd,bhd->bch", self.cand_proj(cands), ctx) / math.sqrt(ctx.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], NEG)
        w = torch.softmax(scores.float(), dim=-1).to(ctx.dtype)                 # [B, C, H]
        return torch.bmm(w, ctx), w


# ----------------------------------------------------------------------------- popularity
class Popularity(nn.Module):
    """Đặc trưng phổ biến/độ mới của tin tại thời điểm impression.

    Chỉ dùng số click/hiển thị ở các giờ TRƯỚC giờ của impression (không rò rỉ tương lai):
    với cửa sổ w giờ, đếm trong [t - w, t). Độ mới = số giờ kể từ lần đầu tin được hiển thị.
    """

    WINDOWS = (1, 6, 24, 72)

    def __init__(self, clicks, shows, hidden=64):
        super().__init__()
        zeros = np.zeros((clicks.shape[0], 1), dtype=np.int32)
        cum = lambda a: torch.as_tensor(np.hstack([zeros, a.cumsum(1, dtype=np.int32)]))  # noqa: E731
        self.register_buffer("cum_clicks", cum(clicks), persistent=False)
        self.register_buffer("cum_shows", cum(shows), persistent=False)
        seen = shows > 0
        first = np.where(seen.any(1), seen.argmax(1), shows.shape[1])
        self.register_buffer("first_seen", torch.as_tensor(first, dtype=torch.long), persistent=False)
        n_feat = 3 * len(self.WINDOWS) + 2
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def features(self, ids, time):  # ids [B, C], time [B]
        t = time.clamp(max=self.cum_clicks.shape[1] - 1)[:, None].expand_as(ids)
        feats = []
        for w in self.WINDOWS:
            lo = (t - w).clamp(min=0)
            c = (self.cum_clicks[ids, t] - self.cum_clicks[ids, lo]).float()
            s = (self.cum_shows[ids, t] - self.cum_shows[ids, lo]).float()
            feats += [torch.log1p(c), torch.log1p(s), (c + 0.1) / (s + 10.0) * 10]
        first = self.first_seen[ids]
        age = (t - first).clamp(min=0).float()
        feats += [torch.log1p(age) / 5, (first >= t).float()]  # tin chưa từng hiển thị trước t
        return torch.stack(feats, dim=-1)

    def forward(self, ids, time):
        return self.mlp(self.features(ids, time)).squeeze(-1)


# ----------------------------------------------------------------------------- full model
class NewsRecModel(nn.Module):
    def __init__(self, cfg, data, plm_cache=None, with_text=True):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        dim = m["news_dim"]
        text = None
        if with_text and m["text_encoder"] == "glove":
            text = GloveTextEncoder(data.news["title"], data.info["vocab_size"], dim, m["glove"], data.glove)
        elif with_text and m["text_encoder"] == "plm":
            plm = m["plm"]
            layers = None
            if plm.get("finetune_layers", 0) > 0:
                full = load_plm(plm["name"])
                layers = list(transformer_layers(full))[-plm["finetune_layers"]:]
            text = PLMTextEncoder(plm_cache, dim, plm, layers)
        elif with_text:
            raise ValueError(f"text_encoder không hợp lệ: {m['text_encoder']}")
        views = m["views"] if with_text else [v for v in m["views"] if v != "text"]
        self.news_encoder = NewsEncoder(text, data.news, data.entity_emb, dim, views) if with_text else None
        self.user_encoder = UserEncoder(m["user_encoder"], dim, m["user"])
        self.popularity = Popularity(*data.popularity) if m.get("popularity") else None
        self.gate = nn.Linear(dim + 1, 1) if self.popularity is not None else None

    # --- news
    def encode_news(self, ids):
        """Encode các tin (loại trùng trong batch để đỡ tốn tính toán)."""
        uniq, inv = torch.unique(ids, return_inverse=True)
        return self.news_encoder(uniq)[inv]

    # --- chấm điểm khi đã có vector tin
    def score_vectors(self, hist_vecs, hist_ids, cand_vecs, cand_ids, time, return_weights=False):
        hist_mask = hist_ids != 0
        user, weights = self.user_encoder(hist_vecs, hist_mask)
        if self.user_encoder.kind == "cand":
            cand_user, weights = self.user_encoder.candidate_aware(hist_vecs, hist_mask, cand_vecs)
            content = ((user.unsqueeze(1) + cand_user) * cand_vecs).sum(-1)
        else:
            content = torch.einsum("bd,bcd->bc", user, cand_vecs)
        scores = content.float()
        if self.popularity is not None:
            hist_len = hist_mask.sum(1, keepdim=True).float()
            eta = torch.sigmoid(self.gate(torch.cat([user.float(), torch.log1p(hist_len)], dim=-1)))
            scores = (1 - eta) * scores + eta * self.popularity(cand_ids, time)
        out = {"scores": scores, "user": user}
        if return_weights:
            out["weights"] = weights
        return out

    def forward(self, batch):
        hist, cands = batch["history"], batch["candidates"]
        B, H = hist.shape
        vecs = self.encode_news(torch.cat([hist.reshape(-1), cands.reshape(-1)]))
        hist_vecs, cand_vecs = vecs[: B * H].view(B, H, -1), vecs[B * H :].view(B, cands.shape[1], -1)
        out = self.score_vectors(hist_vecs, hist, cand_vecs, cands, batch["time"])
        out["hist_vecs"] = hist_vecs
        return out


def build_model(cfg, data, with_text=True):
    cache = None
    m = cfg["model"]
    if with_text and m["text_encoder"] == "plm" and "text" in m["views"]:
        directory = cache_dir(data.dir, m["plm"])
        if not (directory / "meta.json").exists():  # ví dụ ablation chỉ dùng tiêu đề cần cache riêng
            build_cache(data, m["plm"], "cuda" if torch.cuda.is_available() else "cpu")
        cache = PLMCache(directory, m["plm"].get("cache_device", "cpu"))
    return NewsRecModel(cfg, data, cache, with_text)
