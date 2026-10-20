"""Đọc các file của bộ dữ liệu MIND (slide L9, trang 6-9)."""
import csv
import json
import re

import numpy as np
import pandas as pd

NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

_TOKEN_RE = re.compile(r"\w+")


def _read_tsv(path, columns):
    return pd.read_csv(
        path, sep="\t", header=None, names=columns,
        quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
    )


def read_news(path):
    """news.tsv: mỗi dòng là một bản tin."""
    return _read_tsv(path, NEWS_COLUMNS)


def read_behaviors(path):
    """behaviors.tsv: mỗi dòng là một lần hiển thị (impression).

    Trả về DataFrame với các cột đã tách sẵn:
      history     : list news_id đã click trước đó (chỉ có thứ tự, không có thời điểm)
      candidates  : list news_id được hiển thị
      labels      : list nhãn 1 = click, 0 = không click, -1 = không có nhãn (tập test)
      time        : pandas Timestamp
    """
    df = _read_tsv(path, BEHAVIOR_COLUMNS)
    df["history"] = df["history"].str.split()
    pairs = df.pop("impressions").str.split().map(lambda ts: [parse_impression(t) for t in ts])
    df["candidates"] = pairs.map(lambda p: [n for n, _ in p])
    df["labels"] = pairs.map(lambda p: [l for _, l in p])
    df["time"] = pd.to_datetime(df["time"], format=TIME_FORMAT)
    df["impression_id"] = df["impression_id"].astype(np.int64)
    return df


def parse_impression(token):
    if len(token) > 2 and token[-2] == "-":
        return token[:-2], int(token[-1])
    return token, -1


def parse_entities(field, min_confidence=0.0):
    """Lấy danh sách WikidataId từ cột title_entities / abstract_entities."""
    if not field or field == "[]":
        return []
    try:
        items = json.loads(field)
    except json.JSONDecodeError:
        return []
    return [e["WikidataId"] for e in items if e.get("Confidence", 1.0) >= min_confidence and "WikidataId" in e]


def read_entity_embeddings(path):
    """entity_embedding.vec: mỗi dòng là WikidataId + vector 100 chiều (TransE)."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())
