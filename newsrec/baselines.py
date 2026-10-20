"""Các baseline để so sánh với NRMS.

  * popularity : gợi ý tin được click nhiều nhất trong tập train (lọc cộng tác đơn giản nhất).
  * tfidf      : gợi ý dựa trên nội dung (slide L4). Profile của user là trung bình
                 vector TF-IDF của các tin đã đọc, điểm của tin là độ tương đồng cosine.
  * hybrid     : lai có trọng số (slide L7): w * tfidf + (1 - w) * popularity,
                 trên điểm đã chuẩn hoá về thứ hạng trong từng impression.

Chạy:
    python -m newsrec.baselines --model tfidf
"""
import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

from newsrec import preprocess
from newsrec.metrics import evaluate, format_metrics


class PopularityRecommender:
    def fit(self, data):
        self.clicks = np.zeros(len(data["news_ids"]))
        for b in data["train"]:
            np.add.at(self.clicks, b["candidates"][b["labels"] == 1], 1)
        return self

    def score(self, history, candidates):
        return np.log1p(self.clicks[candidates])


class TfidfRecommender:
    def fit(self, data):
        texts = [f"{t} {a}" for t, a in zip(data["title_text"], data["abstract_text"])]
        self.vectors = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=50000).fit_transform(texts)
        return self

    def score(self, history, candidates):
        if len(history) == 0:
            return np.zeros(len(candidates))
        profile = np.asarray(self.vectors[history].mean(axis=0)).ravel()
        return self.vectors[candidates] @ profile


class HybridRecommender:
    def __init__(self, weight=0.5):
        self.weight = weight

    def fit(self, data):
        self.parts = [TfidfRecommender().fit(data), PopularityRecommender().fit(data)]
        return self

    def score(self, history, candidates):
        content, popular = (rankdata(p.score(history, candidates)) / len(candidates) for p in self.parts)
        return self.weight * content + (1 - self.weight) * popular


MODELS = {"popularity": PopularityRecommender, "tfidf": TfidfRecommender, "hybrid": HybridRecommender}


def evaluate_recommender(rec, behaviors):
    scores = [rec.score(b["history"], b["candidates"]) for b in behaviors]
    return evaluate([b["labels"] for b in behaviors], scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/mind_small.pkl")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    args = parser.parse_args()

    data = preprocess.load(args.data)
    for name in MODELS if args.model == "all" else [args.model]:
        rec = MODELS[name]().fit(data)
        print(f"{name:<11} {format_metrics(evaluate_recommender(rec, data['dev']))}")


if __name__ == "__main__":
    main()
