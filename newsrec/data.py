"""Nạp dữ liệu đã tiền xử lý và tạo batch huấn luyện.

Hai cách tạo mẫu huấn luyện, đều chỉ dùng tin nhãn 0 trong CÙNG impression làm mẫu âm:
  * sampled  : 1 tin được click + K tin không click (NRMS, slide L9 trang 25)
  * listwise : toàn bộ tin của impression, nhãn click chuẩn hoá thành phân phối
"""
import json
import pickle
import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class Behaviors:
    impression_id: np.ndarray
    user: np.ndarray
    time: np.ndarray
    hist_ptr: np.ndarray
    hist: np.ndarray
    cand_ptr: np.ndarray
    cand: np.ndarray
    label: np.ndarray

    def __len__(self):
        return len(self.impression_id)

    def history(self, i):
        return self.hist[self.hist_ptr[i] : self.hist_ptr[i + 1]]

    def candidates(self, i):
        return self.cand[self.cand_ptr[i] : self.cand_ptr[i + 1]]

    def labels(self, i):
        return self.label[self.cand_ptr[i] : self.cand_ptr[i + 1]]

    @cached_property
    def impression_of_candidate(self):
        return np.repeat(np.arange(len(self), dtype=np.int32), np.diff(self.cand_ptr))

    @property
    def has_labels(self):
        return len(self.label) > 0 and self.label.min() >= 0


class MindData:
    """Truy cập lười (lazy) các file trong thư mục tiền xử lý."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.info = json.loads((self.dir / "info.json").read_text(encoding="utf-8"))
        self._splits = {}

    @cached_property
    def news(self):
        with np.load(self.dir / "news.npz") as f:
            return {k: f[k] for k in f.files}

    @cached_property
    def meta(self):
        with open(self.dir / "meta.pkl", "rb") as f:
            return pickle.load(f)

    @cached_property
    def entity_emb(self):
        return np.load(self.dir / "entity_emb.npy")

    @cached_property
    def glove(self):
        path = self.dir / "glove.npy"
        return np.load(path) if path.exists() else None

    @cached_property
    def popularity(self):
        with np.load(self.dir / "popularity.npz") as f:
            return f["clicks"], f["shows"]

    def has_split(self, name):
        return (self.dir / f"behaviors_{name}.npz").exists()

    def split(self, name):
        if name not in self._splits:
            with np.load(self.dir / f"behaviors_{name}.npz") as f:
                self._splits[name] = Behaviors(**{k: f[k] for k in f.files})
        return self._splits[name]


def pad_history(history, max_len):
    """Giữ max_len tin gần nhất, pad 0 vào bên trái."""
    out = np.zeros(max_len, dtype=np.int64)
    history = history[-max_len:]
    if len(history):
        out[-len(history):] = history
    return out


class TrainDataset(Dataset):
    def __init__(self, beh, max_history=50, loss="sampled", npratio=4, max_candidates=100, max_samples=None, seed=0):
        self.beh, self.max_history, self.loss = beh, max_history, loss
        self.npratio, self.max_candidates = npratio, max_candidates
        imp = beh.impression_of_candidate
        npos = np.bincount(imp[beh.label == 1], minlength=len(beh))
        nneg = np.bincount(imp[beh.label == 0], minlength=len(beh))
        valid = (npos > 0) & (nneg > 0)
        if loss == "sampled":  # mỗi mẫu là vị trí (trong mảng cand) của một tin được click
            samples = np.nonzero((beh.label == 1) & valid[imp])[0]
        elif loss == "listwise":  # mỗi mẫu là một impression
            samples = np.nonzero(valid)[0]
        else:
            raise ValueError(f"loss không hợp lệ: {loss}")
        if max_samples and len(samples) > max_samples:
            samples = np.sort(np.random.default_rng(seed).choice(samples, max_samples, replace=False))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        beh = self.beh
        if self.loss == "sampled":
            pos = self.samples[i]
            imp = beh.impression_of_candidate[pos]
            cands, labels = beh.candidates(imp), beh.labels(imp)
            neg = cands[labels == 0].tolist()
            negs = random.sample(neg, self.npratio) if len(neg) >= self.npratio else random.choices(neg, k=self.npratio)
            candidates = np.array([beh.cand[pos]] + negs, dtype=np.int64)
            target = np.zeros(len(candidates), dtype=np.float32)
            target[0] = 1
        else:
            imp = self.samples[i]
            candidates, labels = beh.candidates(imp).astype(np.int64), beh.labels(imp)
            if len(candidates) > self.max_candidates:  # giữ mọi tin được click, lấy ngẫu nhiên tin còn lại
                pos, neg = np.nonzero(labels == 1)[0], np.nonzero(labels == 0)[0]
                k = min(len(neg), max(1, self.max_candidates - len(pos)))
                keep = np.concatenate([pos, random.sample(neg.tolist(), k)])
                candidates, labels = candidates[keep], labels[keep]
            target = (labels == 1).astype(np.float32)
            target /= target.sum()
        history = beh.history(imp)
        return {
            "history": pad_history(history, self.max_history),
            "hist_len": min(len(history), self.max_history),
            "candidates": candidates,
            "target": target,
            "time": int(beh.time[imp]),
        }


def collate(items):
    """Pad danh sách ứng viên về cùng độ dài trong batch (cần cho listwise)."""
    n = max(len(it["candidates"]) for it in items)
    batch = {
        "history": torch.from_numpy(np.stack([it["history"] for it in items])),
        "hist_len": torch.tensor([it["hist_len"] for it in items]),
        "time": torch.tensor([it["time"] for it in items]),
        "candidates": torch.zeros(len(items), n, dtype=torch.long),
        "cand_mask": torch.zeros(len(items), n, dtype=torch.bool),
        "target": torch.zeros(len(items), n),
    }
    for b, it in enumerate(items):
        k = len(it["candidates"])
        batch["candidates"][b, :k] = torch.from_numpy(it["candidates"])
        batch["cand_mask"][b, :k] = True
        batch["target"][b, :k] = torch.from_numpy(it["target"])
    return batch


class ResumableSampler(Sampler):
    """Hoán vị cố định theo (seed, epoch), có thể bắt đầu từ giữa epoch khi resume."""

    def __init__(self, n, seed, epoch, start=0):
        self.order = np.random.default_rng(seed + epoch).permutation(n)[start:]

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order.tolist())
