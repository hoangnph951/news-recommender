"""Hàm loss.

  * ranking_loss     : softmax trên tập ứng viên, nhãn là phân phối click (1+K hoặc listwise)
  * contrastive_loss : học tương phản cho user encoder (CL4Rec, slide L10). Hai "view" của cùng
                       một lịch sử đọc (mask / crop / reorder) phải cho vector user gần nhau,
                       còn vector của các user khác trong batch phải xa nhau.
"""
import random

import torch
from torch.nn import functional as F

from newsrec.model import NEG


def ranking_loss(scores, target, cand_mask):
    logp = torch.log_softmax(scores.float().masked_fill(~cand_mask, NEG), dim=-1)
    return -(target * logp).sum(-1).mean()


def _valid_window(hist_ids, ratio):
    """Chọn một đoạn liên tiếp dài ~ratio * độ dài lịch sử (lịch sử được pad bên trái)."""
    B, H = hist_ids.shape
    length = (hist_ids != 0).sum(1)
    win = (length.float() * ratio).ceil().long().clamp(min=1)
    start_min = H - length
    span = (length - win).clamp(min=0)
    start = start_min + (torch.rand(B, device=hist_ids.device) * (span + 1).float()).long().clamp(max=span)
    pos = torch.arange(H, device=hist_ids.device)
    return (pos >= start[:, None]) & (pos < (start + win)[:, None])


def augment(hist_vecs, hist_ids, kind, ratio):
    valid = hist_ids != 0
    if kind == "mask":
        keep = valid & (torch.rand(hist_ids.shape, device=hist_ids.device) > ratio)
        keep[:, -1] |= valid[:, -1]  # luôn giữ tin gần nhất
        return hist_vecs, hist_ids * keep
    if kind == "crop":
        return hist_vecs, hist_ids * (_valid_window(hist_ids, ratio) & valid)
    if kind == "reorder":
        window = _valid_window(hist_ids, ratio) & valid
        index = torch.arange(hist_ids.shape[1], device=hist_ids.device).repeat(hist_ids.shape[0], 1)
        for b in range(hist_ids.shape[0]):
            pos = window[b].nonzero().squeeze(1)
            index[b, pos] = pos[torch.randperm(len(pos), device=pos.device)]
        return torch.gather(hist_vecs, 1, index.unsqueeze(-1).expand_as(hist_vecs)), torch.gather(hist_ids, 1, index)
    raise ValueError(f"augment không hợp lệ: {kind}")


RATIOS = {"mask": 0.3, "crop": 0.6, "reorder": 0.3}


def contrastive_loss(user_encoder, hist_vecs, hist_ids, augments, temperature):
    rows = (hist_ids != 0).sum(1) >= 2  # cần ít nhất 2 tin mới tạo được 2 view khác nhau
    if rows.sum() < 2:
        return hist_vecs.new_zeros(())
    hist_vecs, hist_ids = hist_vecs[rows], hist_ids[rows]
    users = []
    for _ in range(2):
        kind = random.choice(augments)
        v, ids = augment(hist_vecs, hist_ids, kind, RATIOS[kind])
        users.append(F.normalize(user_encoder(v, ids != 0)[0].float(), dim=-1))
    logits = users[0] @ users[1].T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
