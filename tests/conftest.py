"""Bộ dữ liệu MIND giả lập rất nhỏ (cùng định dạng file) và một BERT tí hon để test nhanh toàn bộ pipeline."""
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from newsrec.config import load_config
from newsrec.data import MindData
from newsrec.preprocess import build

TOPICS = {
    "sports": ["football", "match", "goal", "team", "league", "coach"],
    "finance": ["stock", "market", "bank", "price", "investor", "economy"],
    "tech": ["phone", "apple", "software", "chip", "robot", "internet"],
}
ENTITIES = {"sports": "Q2736", "finance": "Q8134", "tech": "Q11016"}
START = datetime(2019, 11, 9)


def make_news(rnd):
    news = []
    for cat, words in TOPICS.items():
        for _ in range(10):
            news.append((f"N{len(news) + 1}", cat, " ".join(rnd.sample(words, 4))))
    return news


def write_split(folder, news, num_users, num_impressions, day, seed, labelled=True):
    rnd = random.Random(seed)
    folder.mkdir()
    with open(folder / "news.tsv", "w", encoding="utf-8") as f:
        for nid, cat, title in news:
            ent = json.dumps([{"Label": cat, "Type": "O", "WikidataId": ENTITIES[cat], "Confidence": 0.9,
                               "OccurrenceOffsets": [0], "SurfaceForms": [cat]}])
            f.write(f"{nid}\t{cat}\t{cat}_sub\t{title}\tabstract about {title}\thttps://example.com/{nid}\t{ent}\t[]\n")
    with open(folder / "entity_embedding.vec", "w", encoding="utf-8") as f:
        for i, q in enumerate(ENTITIES.values()):
            vec = np.eye(len(ENTITIES))[i].tolist() + [0.0] * 97
            f.write(q + "\t" + "\t".join(f"{v:.3f}" for v in vec) + "\n")
    by_topic = {c: [n for n, cat, _ in news if cat == c] for c in TOPICS}
    with open(folder / "behaviors.tsv", "w", encoding="utf-8") as f:
        for i in range(num_impressions):
            user = rnd.randrange(num_users)
            liked = list(TOPICS)[user % len(TOPICS)]  # mỗi user thích một chủ đề
            history = rnd.sample(by_topic[liked], rnd.randint(0, 4))
            pos = rnd.choice(by_topic[liked])
            negs = rnd.sample([n for n, cat, _ in news if cat != liked], 4)
            imps = [f"{pos}-1"] + [f"{n}-0" for n in negs]
            if not labelled:
                imps = [t[:-2] for t in imps]
            rnd.shuffle(imps)
            t = START + timedelta(days=day, seconds=int(i * 86400 / num_impressions))
            stamp = t.strftime("%m/%d/%Y %I:%M:%S %p").lstrip("0")
            f.write(f"{i + 1}\tU{user}\t{stamp}\t{' '.join(history)}\t{' '.join(imps)}\n")


@pytest.fixture(scope="session")
def raw_dirs(tmp_path_factory):
    news = make_news(random.Random(0))
    root = tmp_path_factory.mktemp("mind")
    write_split(root / "train", news, num_users=12, num_impressions=240, day=0, seed=1)
    write_split(root / "dev", news, num_users=15, num_impressions=60, day=1, seed=2)
    write_split(root / "test", news, num_users=15, num_impressions=30, day=2, seed=3, labelled=False)
    return {s: root / s for s in ("train", "dev", "test")}


@pytest.fixture(scope="session")
def data_dir(raw_dirs, tmp_path_factory):
    out = tmp_path_factory.mktemp("processed") / "mind"
    build(raw_dirs, out, max_title_len=8)
    return out


@pytest.fixture(scope="session")
def data(data_dir):
    return MindData(data_dir)


@pytest.fixture(scope="session")
def tiny_plm(tmp_path_factory):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    d = tmp_path_factory.mktemp("tiny_bert")
    words = sorted({w for ws in TOPICS.values() for w in ws} | {"abstract", "about"})
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]) + "\n")
    BertTokenizerFast(str(d / "vocab.txt")).save_pretrained(d)
    config = BertConfig(vocab_size=5 + len(words), hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(d)
    return str(d)


@pytest.fixture
def tiny_sets(data_dir, tiny_plm):
    """Ghi đè để thu nhỏ mọi config về kích thước chạy được trong vài giây."""
    return [
        f"data.dir={data_dir.as_posix()}", f"model.plm.name={tiny_plm.replace(chr(92), '/')}",
        "model.plm.max_len=16", "model.news_dim=32",
        "model.user.num_heads=4", "model.user.head_dim=8", "model.glove.num_heads=4", "model.glove.head_dim=8",
        "model.max_history=6", "train.epochs=4", "train.batch_size=16", "train.lr=0.003", "train.plm_lr=0.001",
        "train.warmup_ratio=0", "train.eval_every=0", "train.eval_subset=null", "train.amp=false",
        "eval.batch_size=32",
    ]


@pytest.fixture
def cfg(tiny_sets):
    return load_config("configs/full.yaml", [*tiny_sets, "model.plm.finetune_layers=1"])
