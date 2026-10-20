"""Đọc các file của bộ dữ liệu MIND (slide L9, trang 6-9)."""
import csv
import re

import pandas as pd

NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]

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

    Cột `history` là danh sách news_id đã click trước đó, cột `impressions`
    là danh sách (news_id, label) với label 1 = click, 0 = không click.
    Tập test của MIND không có nhãn, khi đó label = -1.
    """
    df = _read_tsv(path, BEHAVIOR_COLUMNS)
    df["history"] = df["history"].str.split()
    df["impressions"] = df["impressions"].str.split().map(
        lambda tokens: [parse_impression(t) for t in tokens]
    )
    return df


def parse_impression(token):
    if "-" in token:
        news_id, label = token.rsplit("-", 1)
        return news_id, int(label)
    return token, -1


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())
