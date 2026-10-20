"""Dataset huấn luyện với negative sampling (slide L9, trang 15 và 25).

Mỗi mẫu gồm 1 tin được click (positive) và K tin không được click (negative)
lấy ngẫu nhiên trong cùng impression. Positive luôn nằm ở vị trí 0, nên nhãn
của bài toán phân loại K+1 lớp luôn là 0.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_history(history, max_len):
    """Giữ max_len tin gần nhất, pad 0 vào bên trái."""
    out = np.zeros(max_len, dtype=np.int64)
    history = history[-max_len:]
    if len(history):
        out[-len(history):] = history
    return out


class TrainDataset(Dataset):
    def __init__(self, behaviors, max_history=50, npratio=4, seed=0):
        self.behaviors = behaviors
        self.max_history = max_history
        self.npratio = npratio
        self.rng = np.random.default_rng(seed)
        self.samples = []
        for bi, b in enumerate(behaviors):
            if (b["labels"] == 0).any():
                self.samples.extend((bi, p) for p in b["candidates"][b["labels"] == 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        bi, pos = self.samples[i]
        b = self.behaviors[bi]
        neg = b["candidates"][b["labels"] == 0]
        negs = self.rng.choice(neg, self.npratio, replace=len(neg) < self.npratio)
        candidates = np.concatenate([[pos], negs]).astype(np.int64)
        history = pad_history(b["history"], self.max_history)
        return torch.from_numpy(history), torch.from_numpy(candidates), 0
