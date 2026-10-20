"""Sinh một bộ dữ liệu MIND giả lập rất nhỏ để test nhanh toàn bộ pipeline."""
import random

import pytest

from newsrec import preprocess

TOPICS = {
    "sports": ["football", "match", "goal", "team", "league", "coach"],
    "finance": ["stock", "market", "bank", "price", "investor", "economy"],
    "tech": ["phone", "apple", "software", "chip", "robot", "internet"],
}


def write_split(folder, news, num_users, num_impressions, seed):
    rnd = random.Random(seed)
    folder.mkdir()
    with open(folder / "news.tsv", "w", encoding="utf-8") as f:
        for nid, cat, title in news:
            f.write(f"{nid}\t{cat}\t{cat}_sub\t{title}\tabstract about {title}\thttps://example.com/{nid}\t[]\t[]\n")
    by_topic = {c: [n for n, cat, _ in news if cat == c] for c in TOPICS}
    with open(folder / "behaviors.tsv", "w", encoding="utf-8") as f:
        for i in range(num_impressions):
            user = rnd.randrange(num_users)
            liked = list(TOPICS)[user % len(TOPICS)]  # mỗi user thích một chủ đề
            history = rnd.sample(by_topic[liked], 3)
            pos = rnd.choice(by_topic[liked])
            negs = rnd.sample([n for n, cat, _ in news if cat != liked], 4)
            imps = [f"{pos}-1"] + [f"{n}-0" for n in negs]
            rnd.shuffle(imps)
            f.write(f"{i + 1}\tU{user}\t11/15/2019 8:55:22 AM\t{' '.join(history)}\t{' '.join(imps)}\n")


@pytest.fixture(scope="session")
def mind_dirs(tmp_path_factory):
    rnd = random.Random(0)
    news = []
    for cat, words in TOPICS.items():
        for _ in range(10):
            news.append((f"N{len(news) + 1}", cat, " ".join(rnd.sample(words, 4))))
    root = tmp_path_factory.mktemp("mind")
    write_split(root / "train", news, num_users=12, num_impressions=200, seed=1)
    write_split(root / "dev", news, num_users=12, num_impressions=60, seed=2)
    return root / "train", root / "dev"


@pytest.fixture(scope="session")
def data(mind_dirs):
    return preprocess.build(*mind_dirs, max_title_len=8)


@pytest.fixture(scope="session")
def data_path(data, tmp_path_factory):
    path = tmp_path_factory.mktemp("processed") / "mind.pkl"
    preprocess.save(data, path)
    return path
