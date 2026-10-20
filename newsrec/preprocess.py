"""Tiền xử lý MIND thành một file pickle dùng chung cho baseline, NRMS và demo.

Các bước (slide L9, trang 15):
  * Gộp tin của tập train và dev, gán chỉ số cho mỗi tin (0 = padding).
  * Tách từ tiêu đề, xây từ điển từ tập train, cắt/pad tiêu đề về độ dài cố định.
  * Chuyển UserID thành số thứ tự.
  * (Tuỳ chọn) nạp GloVe làm word embedding khởi tạo.

Chạy:
    python -m newsrec.preprocess --train-dir data/MINDsmall_train \
        --dev-dir data/MINDsmall_dev --out data/processed/mind_small.pkl
"""
import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from newsrec.mind import read_behaviors, read_news, tokenize

PAD, UNK = 0, 1


def build_vocab(token_lists, min_freq):
    counter = Counter(tok for tokens in token_lists for tok in tokens)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_titles(token_lists, vocab, max_len):
    out = np.zeros((len(token_lists) + 1, max_len), dtype=np.int64)  # dòng 0 = tin padding
    for i, tokens in enumerate(token_lists, start=1):
        ids = [vocab.get(t, UNK) for t in tokens[:max_len]]
        out[i, : len(ids)] = ids
    return out


def encode_behaviors(df, news_index, user_index):
    behaviors = []
    for row in df.itertuples(index=False):
        candidates = [(news_index[n], label) for n, label in row.impressions if n in news_index]
        behaviors.append({
            "impression_id": row.impression_id,
            "user_id": row.user_id,
            "user": user_index.get(row.user_id, 0),
            "time": row.time,
            "history": np.array([news_index[n] for n in row.history if n in news_index], dtype=np.int64),
            "candidates": np.array([c for c, _ in candidates], dtype=np.int64),
            "labels": np.array([l for _, l in candidates], dtype=np.int64),
        })
    return behaviors


def load_glove(path, vocab, dim, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[PAD] = 0
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            idx = vocab.get(word)
            if idx is not None and len(values) == dim:
                matrix[idx] = np.asarray(values, dtype=np.float32)
                found += 1
    print(f"GloVe phủ {found}/{len(vocab)} từ ({found / len(vocab):.1%})")
    return matrix


def build(train_dir, dev_dir, max_title_len=30, min_word_freq=1, glove_path=None, embed_dim=300):
    train_dir, dev_dir = Path(train_dir), Path(dev_dir)
    train_news = read_news(train_dir / "news.tsv")
    news = pd.concat([train_news, read_news(dev_dir / "news.tsv")]).drop_duplicates("news_id")
    news = news.reset_index(drop=True)
    news_index = {nid: i for i, nid in enumerate(news["news_id"], start=1)}

    title_tokens = [tokenize(t) for t in news["title"]]
    train_ids = set(train_news["news_id"])
    vocab = build_vocab(
        [toks for toks, nid in zip(title_tokens, news["news_id"]) if nid in train_ids], min_word_freq
    )

    categories = {c: i for i, c in enumerate(sorted(news["category"].unique()), start=1)}
    subcategories = {c: i for i, c in enumerate(sorted(news["subcategory"].unique()), start=1)}

    train_df = read_behaviors(train_dir / "behaviors.tsv")
    dev_df = read_behaviors(dev_dir / "behaviors.tsv")
    user_index = {u: i for i, u in enumerate(train_df["user_id"].unique(), start=1)}

    def with_pad(values):  # vị trí 0 ứng với tin padding
        return [""] + list(values)

    return {
        "news_ids": with_pad(news["news_id"]),
        "news_index": news_index,
        "title_text": with_pad(news["title"]),
        "abstract_text": with_pad(news["abstract"]),
        "category_text": with_pad(news["category"]),
        "subcategory_text": with_pad(news["subcategory"]),
        "url": with_pad(news["url"]),
        "title": encode_titles(title_tokens, vocab, max_title_len),
        "category": np.array([0] + [categories[c] for c in news["category"]], dtype=np.int64),
        "subcategory": np.array([0] + [subcategories[c] for c in news["subcategory"]], dtype=np.int64),
        "vocab": vocab,
        "categories": categories,
        "subcategories": subcategories,
        "user_index": user_index,
        "train": encode_behaviors(train_df, news_index, user_index),
        "dev": encode_behaviors(dev_df, news_index, user_index),
        "embedding": load_glove(glove_path, vocab, embed_dim) if glove_path else None,
        "max_title_len": max_title_len,
    }


def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", default="data/MINDsmall_train")
    parser.add_argument("--dev-dir", default="data/MINDsmall_dev")
    parser.add_argument("--out", default="data/processed/mind_small.pkl")
    parser.add_argument("--max-title-len", type=int, default=30)
    parser.add_argument("--min-word-freq", type=int, default=1)
    parser.add_argument("--glove", help="đường dẫn glove.*.300d.txt (tuỳ chọn)")
    parser.add_argument("--embed-dim", type=int, default=300)
    args = parser.parse_args()

    data = build(args.train_dir, args.dev_dir, args.max_title_len, args.min_word_freq, args.glove, args.embed_dim)
    save(data, args.out)
    print(
        f"{len(data['news_ids']) - 1} tin, {len(data['vocab'])} từ, {len(data['user_index'])} user, "
        f"{len(data['train'])} impression train, {len(data['dev'])} impression dev -> {args.out}"
    )


if __name__ == "__main__":
    main()
