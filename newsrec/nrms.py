"""NRMS: Neural News Recommendation with Multi-Head Self-Attention (Wu et al., 2019).

Kiến trúc 3 thành phần (slide L9, trang 34-38):
  * News encoder : word embedding -> multi-head self-attention -> additive attention
  * User encoder : multi-head self-attention trên các tin đã đọc -> additive attention
  * Click predictor : tích vô hướng giữa vector user và vector tin
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class NRMSConfig:
    embed_dim: int = 300
    num_heads: int = 16
    head_dim: int = 16
    attn_hidden: int = 200
    dropout: float = 0.2
    max_history: int = 50
    npratio: int = 4


class AdditiveAttention(nn.Module):
    """Gộp một chuỗi vector thành 1 vector bằng trọng số attention học được."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):  # x: [B, L, D], mask: [B, L], True = vị trí hợp lệ
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        out_dim = num_heads * head_dim
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask=None):  # x: [B, L, D] -> [B, L, heads * head_dim]
        B, L, _ = x.shape

        def split(t):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-1, -2) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        out = torch.softmax(scores, dim=-1) @ v
        return out.transpose(1, 2).reshape(B, L, -1)


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cfg, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained))
        self.dropout = nn.Dropout(cfg.dropout)
        self.self_attn = MultiHeadSelfAttention(cfg.embed_dim, cfg.num_heads, cfg.head_dim)
        self.attn = AdditiveAttention(cfg.num_heads * cfg.head_dim, cfg.attn_hidden)

    def forward(self, tokens):  # tokens: [N, title_len]
        mask = tokens != 0
        x = self.dropout(self.embedding(tokens))
        x = self.dropout(self.self_attn(x, mask))
        return self.attn(x, mask)


class UserEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        news_dim = cfg.num_heads * cfg.head_dim
        self.self_attn = MultiHeadSelfAttention(news_dim, cfg.num_heads, cfg.head_dim)
        self.attn = AdditiveAttention(news_dim, cfg.attn_hidden)

    def forward(self, news_vecs, mask):  # news_vecs: [B, H, D], mask: [B, H]
        return self.attn(self.self_attn(news_vecs, mask), mask)


class NRMS(nn.Module):
    def __init__(self, title_tokens, vocab_size, cfg=None, pretrained=None):
        super().__init__()
        self.cfg = cfg or NRMSConfig()
        # Bảng token tiêu đề của mọi tin; không lưu vào checkpoint vì đã có trong dữ liệu tiền xử lý.
        self.register_buffer("title_tokens", torch.as_tensor(np.asarray(title_tokens)), persistent=False)
        self.news_encoder = NewsEncoder(vocab_size, self.cfg, pretrained)
        self.user_encoder = UserEncoder(self.cfg)

    def encode_news(self, news_ids):  # [...] -> [..., D]
        flat = news_ids.reshape(-1)
        return self.news_encoder(self.title_tokens[flat]).view(*news_ids.shape, -1)

    def encode_user(self, history_vecs, history_ids):
        return self.user_encoder(history_vecs, history_ids != 0)

    def forward(self, history, candidates):  # [B, H], [B, C] -> điểm [B, C]
        user = self.encode_user(self.encode_news(history), history)
        return (self.encode_news(candidates) * user.unsqueeze(1)).sum(-1)
