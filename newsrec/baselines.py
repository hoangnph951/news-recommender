"""Các baseline không cần GPU, ghi kết quả cùng định dạng với mô hình (runs/<tên>/).

  * popularity : tin được click nhiều nhất trong tập train (đơn giản nhất)
  * recent_pop : tin được click nhiều trong 24 giờ trước impression (chỉ dùng quá khứ)
  * tfidf      : gợi ý dựa trên nội dung (slide L4). Profile user là trung bình vector TF-IDF
                 (tiêu đề + tóm tắt) của các tin đã đọc, điểm là độ tương đồng cosine.

Chạy:
    python -m newsrec.baselines --data data/processed/mind_large
"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from newsrec.data import MindData
from newsrec.evaluate import metrics_report, print_report, save_run


def popularity_scores(data, beh):
    train = data.split("train")
    clicks = np.bincount(train.cand[train.label == 1], minlength=data.info["num_news"])
    return np.log1p(clicks[beh.cand]).astype(np.float32)


def recent_popularity_scores(data, beh, window=24):
    clicks, _ = data.popularity
    cum = np.hstack([np.zeros((len(clicks), 1), np.int64), clicks.cumsum(1)])
    t = np.minimum(np.repeat(beh.time, np.diff(beh.cand_ptr)), cum.shape[1] - 1)
    recent = cum[beh.cand, t] - cum[beh.cand, np.maximum(t - window, 0)]
    total = np.log1p(cum[beh.cand, t]) * 1e-3  # phá thế hoà bằng độ phổ biến tích luỹ
    return (np.log1p(recent) + total).astype(np.float32)


def tfidf_scores(data, beh):
    texts = [f"{t} {a}" for t, a in zip(data.meta["title"], data.meta["abstract"])]
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_features=100000).fit_transform(texts)
    X = normalize(X).tocsr().astype(np.float32)
    out = np.zeros(len(beh.cand), dtype=np.float32)
    for i in tqdm(range(len(beh)), desc="tfidf", mininterval=10):
        hist = beh.history(i)
        if len(hist):  # user không có lịch sử: mọi tin điểm 0 (AUC = 0.5)
            profile = np.asarray(X[hist].mean(axis=0)).ravel()
            out[beh.cand_ptr[i] : beh.cand_ptr[i + 1]] = X[beh.candidates(i)] @ profile
    return out


BASELINES = {
    "popularity": ("Popularity (train)", popularity_scores),
    "recent_pop": ("Popularity 24h gần nhất", recent_popularity_scores),
    "tfidf": ("Content-based TF-IDF", tfidf_scores),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/mind_large")
    parser.add_argument("--only", nargs="*", choices=list(BASELINES))
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    data = MindData(args.data)
    dev = data.split("dev")
    for name in args.only or BASELINES:
        method, fn = BASELINES[name]
        scores = fn(data, dev)
        report = metrics_report(dev, scores)
        print_report(name, report)
        save_run(f"{args.runs_dir}/{name}", name, method, report, scores)


if __name__ == "__main__":
    main()
