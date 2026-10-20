"""Độ đo đánh giá theo từng impression rồi lấy trung bình (slide L8).

Cách tính giống script đánh giá chính thức của MIND: AUC, MRR, nDCG@5, nDCG@10,
xếp hạng TOÀN BỘ ứng viên của impression gốc (không lấy mẫu lại).
"""
import numpy as np

NAMES = ("AUC", "MRR", "nDCG@5", "nDCG@10")


def auc_score(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def mrr_score(labels, scores):
    y = labels[np.argsort(-scores, kind="stable")]
    if y.sum() == 0:
        return np.nan
    return (y / np.arange(1, len(y) + 1)).sum() / y.sum()


def dcg_score(labels, scores, k):
    y = labels[np.argsort(-scores, kind="stable")[:k]]
    return ((2 ** y - 1) / np.log2(np.arange(len(y)) + 2)).sum()


def ndcg_score(labels, scores, k):
    best = dcg_score(labels, labels.astype(float), k)
    if best == 0:
        return np.nan
    return dcg_score(labels, scores, k) / best


def impression_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    return [auc_score(labels, scores), mrr_score(labels, scores),
            ndcg_score(labels, scores, 5), ndcg_score(labels, scores, 10)]


def per_impression(labels, scores, ptr):
    """labels/scores phẳng theo mảng cand, ptr là cand_ptr -> mảng [số impression, 4]."""
    return np.array([impression_metrics(labels[a:b], scores[a:b]) for a, b in zip(ptr[:-1], ptr[1:])])


def summarize(rows, mask=None):
    rows = rows if mask is None else rows[mask]
    return dict(zip(NAMES, np.nanmean(rows, axis=0).round(4).tolist()))


def evaluate(all_labels, all_scores):
    """Dạng list (mỗi phần tử một impression), tiện cho test và baseline nhỏ."""
    return summarize(np.array([impression_metrics(l, s) for l, s in zip(all_labels, all_scores)]))


def format_metrics(metrics):
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
