"""Tiền xử lý MIND thành các mảng numpy phẳng, đủ gọn để chạy với MIND-large.

Kết quả ghi vào một thư mục (mặc định data/processed/mind_large):
  info.json              thống kê, mốc thời gian, kích thước
  meta.pkl               văn bản của tin (để hiển thị, TF-IDF, PLM), các bảng ánh xạ id
  news.npz               title (token id cho GloVe), category, subcategory, entities
  entity_emb.npy         vector entity 100 chiều (từ entity_embedding.vec)
  glove.npy              (tuỳ chọn) word embedding khởi tạo cho GloVe
  behaviors_<split>.npz  impression dạng CSR: hist_ptr/hist, cand_ptr/cand/label, user, time
  popularity.npz         số click / số lần hiển thị của mỗi tin theo từng giờ

Chỉ số tin 0 là padding. User chỉ số 0 là user không xuất hiện trong tập train.
Thời gian được quy về số giờ kể từ 0h ngày đầu tiên của dữ liệu.

Chạy:
    python -m newsrec.preprocess --raw data/raw --size large --out data/processed/mind_large
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from newsrec.mind import parse_entities, read_behaviors, read_entity_embeddings, read_news, tokenize

PAD, UNK = 0, 1
SPLITS = ("train", "dev", "test")


def build_vocab(token_lists, min_freq):
    counter = Counter(tok for tokens in token_lists for tok in tokens)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def pad_rows(rows, width, dtype=np.int32):
    """rows là list các list chỉ số; dòng 0 của kết quả là padding."""
    out = np.zeros((len(rows) + 1, width), dtype=dtype)
    for i, row in enumerate(rows, start=1):
        row = row[:width]
        out[i, : len(row)] = row
    return out


def flatten(lists, index=None, dtype=np.int32):
    lengths = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])
    items = (x for l in lists for x in l) if index is None else (index[x] for l in lists for x in l)
    return ptr, np.fromiter(items, dtype=dtype, count=int(ptr[-1]))


def load_glove(path, vocab, dim=300, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[PAD] = 0
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.rstrip().partition(" ")
            idx = vocab.get(word)
            if idx is not None:
                values = np.asarray(rest.split(" "), dtype=np.float32)
                if len(values) == dim:
                    matrix[idx] = values
                    found += 1
    print(f"GloVe phủ {found}/{len(vocab)} từ ({found / len(vocab):.1%})")
    return matrix


def build(split_dirs, out_dir, max_title_len=30, min_word_freq=1, max_entities=8, glove_path=None):
    """split_dirs: dict {"train": dir, "dev": dir, "test": dir (tuỳ chọn)}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_dirs = {s: Path(d) for s, d in split_dirs.items() if d}

    # ---- tin tức ----
    news_frames = {s: read_news(d / "news.tsv") for s, d in split_dirs.items()}
    news = pd.concat(news_frames.values()).drop_duplicates("news_id").reset_index(drop=True)
    news_index = {nid: i for i, nid in enumerate(news["news_id"], start=1)}
    print(f"{len(news)} tin")

    title_tokens = [tokenize(t) for t in news["title"]]
    train_ids = set(news_frames["train"]["news_id"])
    vocab = build_vocab([t for t, n in zip(title_tokens, news["news_id"]) if n in train_ids], min_word_freq)
    titles = pad_rows([[vocab.get(t, UNK) for t in toks] for toks in title_tokens], max_title_len)

    categories = {c: i for i, c in enumerate(sorted(news["category"].unique()), start=1)}
    subcategories = {c: i for i, c in enumerate(sorted(news["subcategory"].unique()), start=1)}

    entity_vectors = {}
    for d in split_dirs.values():
        if (d / "entity_embedding.vec").exists():
            entity_vectors.update(read_entity_embeddings(d / "entity_embedding.vec"))
    entity_index = {q: i for i, q in enumerate(entity_vectors, start=1)}
    entity_dim = len(next(iter(entity_vectors.values()))) if entity_vectors else 100
    entity_emb = np.zeros((len(entity_index) + 1, entity_dim), dtype=np.float32)
    for q, i in entity_index.items():
        entity_emb[i] = entity_vectors[q]
    news_entities = []
    for te, ae in zip(news["title_entities"], news["abstract_entities"]):
        qids = dict.fromkeys(q for q in parse_entities(te) + parse_entities(ae) if q in entity_index)
        news_entities.append([entity_index[q] for q in qids])
    entities = pad_rows(news_entities, max_entities)

    np.savez_compressed(
        out / "news.npz", title=titles, entities=entities,
        category=np.array([0] + [categories[c] for c in news["category"]], dtype=np.int32),
        subcategory=np.array([0] + [subcategories[c] for c in news["subcategory"]], dtype=np.int32),
    )
    np.save(out / "entity_emb.npy", entity_emb)
    if glove_path:
        np.save(out / "glove.npy", load_glove(glove_path, vocab))

    # ---- hành vi ----
    behaviors = {s: read_behaviors(d / "behaviors.tsv") for s, d in split_dirs.items()}
    base = min(df["time"].min() for df in behaviors.values()).normalize()
    user_index = {u: i for i, u in enumerate(behaviors["train"]["user_id"].unique(), start=1)}
    num_hours = 1 + max(int((df["time"].max() - base) / pd.Timedelta(hours=1)) for df in behaviors.values())
    clicks = np.zeros((len(news_index) + 1, num_hours), dtype=np.int32)
    shows = np.zeros_like(clicks)

    stats = {}
    for split, df in behaviors.items():
        hist_ptr, hist = flatten(df["history"].tolist(), news_index)
        cand_ptr, cand = flatten(df["candidates"].tolist(), news_index)
        _, label = flatten(df["labels"].tolist(), dtype=np.int8)
        hours = ((df["time"] - base) / pd.Timedelta(hours=1)).astype(np.int32).to_numpy()
        users = np.array([user_index.get(u, 0) for u in df["user_id"]], dtype=np.int32)
        np.savez(
            out / f"behaviors_{split}.npz",
            impression_id=df["impression_id"].to_numpy(), user=users, time=hours,
            hist_ptr=hist_ptr, hist=hist, cand_ptr=cand_ptr, cand=cand, label=label,
        )
        cell = cand.astype(np.int64) * num_hours + np.repeat(hours, np.diff(cand_ptr))
        shows += np.bincount(cell, minlength=shows.size).reshape(shows.shape).astype(np.int32)
        clicked = label == 1  # tập test không có nhãn nên chỉ góp vào số lần hiển thị
        clicks += np.bincount(cell[clicked], minlength=clicks.size).reshape(clicks.shape).astype(np.int32)
        stats[split] = {
            "impressions": len(df), "users": int(df["user_id"].nunique()),
            "candidates": int(len(cand)), "clicks": int(clicked.sum()),
            "hours": [int(hours.min()), int(hours.max())],
        }
        print(f"{split}: {stats[split]}")
    np.savez_compressed(out / "popularity.npz", clicks=clicks, shows=shows)

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({
            "news_ids": [""] + news["news_id"].tolist(),
            "title": [""] + news["title"].tolist(),
            "abstract": [""] + news["abstract"].tolist(),
            "category": [""] + news["category"].tolist(),
            "subcategory": [""] + news["subcategory"].tolist(),
            "url": [""] + news["url"].tolist(),
            "news_index": news_index,
            "vocab": vocab,
            "categories": categories,
            "subcategories": subcategories,
            "users": [""] + list(user_index),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    info = {
        "num_news": len(news_index) + 1, "vocab_size": len(vocab), "num_users": len(user_index) + 1,
        "num_categories": len(categories) + 1, "num_subcategories": len(subcategories) + 1,
        "num_entities": len(entity_index) + 1, "entity_dim": entity_dim,
        "num_hours": num_hours, "base_time": str(base), "max_title_len": max_title_len,
        "splits": list(behaviors), "glove": bool(glove_path), "stats": stats,
    }
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw", help="thư mục chứa MIND<size>_train, _dev, _test")
    parser.add_argument("--size", default="large", choices=["large", "small"])
    parser.add_argument("--out", help="mặc định data/processed/mind_<size>")
    parser.add_argument("--no-test", action="store_true", help="bỏ qua tập test")
    parser.add_argument("--max-title-len", type=int, default=30)
    parser.add_argument("--min-word-freq", type=int, default=1)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--glove", help="đường dẫn glove.840B.300d.txt (cho baseline NRMS-GloVe)")
    args = parser.parse_args()

    raw = Path(args.raw)
    dirs = {s: raw / f"MIND{args.size}_{s}" for s in SPLITS}
    if args.no_test or not (dirs["test"] / "behaviors.tsv").exists():
        dirs.pop("test")
    build(dirs, args.out or f"data/processed/mind_{args.size}", args.max_title_len,
          args.min_word_freq, args.max_entities, args.glove)


if __name__ == "__main__":
    main()
