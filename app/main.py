"""Demo web: chọn user, xem tin được gợi ý kèm lý do, bấm đọc để gợi ý cập nhật; trang kết quả thực nghiệm.

Chạy (từ thư mục gốc của repo):
    uvicorn app.main:app
rồi mở http://127.0.0.1:8000

Biến môi trường:
    NEWSREC_DATA  thư mục tiền xử lý (mặc định data/processed/mind_large)
    NEWSREC_RUN   thư mục run đã huấn luyện, có config.yaml, best.pt, news_vecs.npy
                  (mặc định runs/full_seed42). Không có thì demo dùng độ phổ biến 24h.
    NEWSREC_RUNS  thư mục chứa các run để hiển thị bảng kết quả (mặc định runs)
"""
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from newsrec.analysis import GROUPS, results_table
from newsrec.config import load_config
from newsrec.data import MindData
from newsrec.model import build_model

STATIC = Path(__file__).parent / "static"


class Recommender:
    def __init__(self, data_dir, run_dir, feedback_db):
        self.data = MindData(data_dir)
        self.meta = self.data.meta
        self.user_ids = self.meta["users"]
        self.user_index = {u: i for i, u in enumerate(self.user_ids) if u}
        self.now = self.data.info["stats"]["dev"]["hours"][1] + 1  # "bây giờ" = ngay sau giờ cuối của dev
        self._index_last_impressions()

        run = Path(run_dir)
        self.model = None
        if (run / "news_vecs.npy").exists():
            cfg = load_config(run / "config.yaml")
            self.model = build_model(cfg, self.data, with_text=False).eval()
            state = torch.load(run / "best.pt", map_location="cpu")["model"]
            self.model.load_state_dict({k: v for k, v in state.items() if not k.startswith("news_encoder.")})
            self.vecs = torch.from_numpy(np.load(run / "news_vecs.npy").astype(np.float32))
            self.max_history = cfg["model"]["max_history"]
            self.method = cfg.get("method", cfg["name"])
        else:
            clicks, _ = self.data.popularity
            hi = min(self.now, clicks.shape[1])
            self.recent = clicks[:, max(hi - 24, 0) : hi].sum(1) + 1e-3 * clicks[:, :hi].sum(1)
            self.method = "Độ phổ biến 24h gần nhất (chưa có mô hình)"

        self.db = sqlite3.connect(feedback_db, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS clicks (user TEXT, news_id TEXT, ts REAL)")
        self.extra = {}  # click trong demo, cộng thêm vào lịch sử gốc
        for user, news_id in self.db.execute("SELECT user, news_id FROM clicks ORDER BY ts"):
            if news_id in self.meta["news_index"]:
                self.extra.setdefault(user, []).append(self.meta["news_index"][news_id])

    def _index_last_impressions(self):
        """Impression gần nhất của mỗi user (train hoặc dev) để lấy lịch sử đọc mới nhất."""
        self.splits = [s for s in ("train", "dev") if self.data.has_split(s)]
        users, times, refs = [], [], []
        for k, s in enumerate(self.splits):
            beh = self.data.split(s)
            users.append(beh.user)
            times.append(beh.time)
            refs.append(np.stack([np.full(len(beh), k), np.arange(len(beh))], 1))
        users, times, refs = np.concatenate(users), np.concatenate(times), np.concatenate(refs)
        order = np.lexsort((times, users))
        last = np.r_[users[order][1:] != users[order][:-1], True]
        self.last_ref = {int(u): tuple(r) for u, r in zip(users[order][last], refs[order][last]) if u > 0}

    def history(self, user):
        hist = []
        if user in self.user_index and self.user_index[user] in self.last_ref:
            split, i = self.last_ref[self.user_index[user]]
            beh = self.data.split(self.splits[split])
            hist = beh.history(i).tolist() + beh.candidates(i)[beh.labels(i) == 1].tolist()
        return hist + self.extra.get(user, [])

    def news(self, idx, **extra):
        m = self.meta
        return {"news_id": m["news_ids"][idx], "title": m["title"][idx], "abstract": m["abstract"][idx],
                "category": m["category"][idx], "subcategory": m["subcategory"][idx], "url": m["url"][idx], **extra}

    @torch.no_grad()
    def recommend(self, user, k=10, chunk=20000):
        history = self.history(user)
        read = set(history)
        n = self.data.info["num_news"]
        k = min(k, n - 1 - len(read))
        if self.model is None:
            scores = self.recent.astype(np.float64).copy()
            scores[0] = -np.inf
            scores[list(read)] = -np.inf
            top = np.argsort(-scores)[:k]
            return {"items": [self.news(int(i), score=round(float(scores[i]), 3)) for i in top], "popularity_weight": 1.0}

        hist = torch.zeros(1, self.max_history, dtype=torch.long)
        recent = history[-self.max_history:]
        if recent:
            hist[0, -len(recent):] = torch.tensor(recent)
        hist_vecs, time_ = self.vecs[hist], torch.tensor([self.now])
        scores = np.empty(n, dtype=np.float32)
        for start in range(1, n, chunk):
            cands = torch.arange(start, min(start + chunk, n)).unsqueeze(0)
            scores[start : start + cands.shape[1]] = self.model.score_vectors(
                hist_vecs, hist, self.vecs[cands], cands, time_)["scores"][0].numpy()
        scores[0] = -np.inf
        scores[list(read)] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]

        # Giải thích: các tin trong lịch sử có trọng số attention cao nhất (L1: thuyết phục người dùng)
        cands = torch.from_numpy(top.astype(np.int64)).unsqueeze(0)
        weights = self.model.score_vectors(hist_vecs, hist, self.vecs[cands], cands, time_, return_weights=True)["weights"][0]
        items = []
        for j, i in enumerate(top):
            w = weights[j] if weights.dim() == 2 else weights  # candidate-aware: trọng số riêng cho từng tin
            because = [self.meta["title"][int(hist[0, p])] for p in torch.argsort(w, descending=True)[:3].tolist()
                       if hist[0, p] != 0] if recent else []
            items.append(self.news(int(i), score=round(float(scores[i]), 3), because=because))

        eta = None
        if self.model.popularity is not None:
            user_vec = self.model.user_encoder(hist_vecs, hist != 0)[0]
            hist_len = torch.tensor([[float(len(recent))]])
            eta = float(torch.sigmoid(self.model.gate(torch.cat([user_vec, torch.log1p(hist_len)], -1))))
        return {"items": items, "popularity_weight": eta}

    def click(self, user, news_id):
        idx = self.meta["news_index"].get(news_id)
        if idx is None:
            raise KeyError(news_id)
        self.extra.setdefault(user, []).append(idx)
        self.db.execute("INSERT INTO clicks VALUES (?, ?, ?)", (user, news_id, time.time()))
        self.db.commit()

    def sample_users(self, limit, seed=0):
        """Một số user có lịch sử dài để demo (lấy ngẫu nhiên rồi chọn người đọc nhiều nhất)."""
        rng = np.random.default_rng(seed)
        idx = list(self.last_ref)
        pick = rng.choice(idx, min(len(idx), 3000), replace=False)
        lengths = []
        for u in pick:
            split, i = self.last_ref[int(u)]
            beh = self.data.split(self.splits[split])
            lengths.append(int(beh.hist_ptr[i + 1] - beh.hist_ptr[i]))
        order = np.argsort(lengths)[::-1][:limit]
        return [{"user_id": self.user_ids[int(pick[j])], "history_len": lengths[j]} for j in order]


@lru_cache
def get_recommender():
    return Recommender(
        os.environ.get("NEWSREC_DATA", "data/processed/mind_large"),
        os.environ.get("NEWSREC_RUN", "runs/full_seed42"),
        os.environ.get("NEWSREC_FEEDBACK", "data/feedback.db"),
    )


app = FastAPI(title="News Recommender")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Click(BaseModel):
    user_id: str
    news_id: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/results")
def results_page():
    return FileResponse(STATIC / "results.html")


@app.get("/api/info")
def info():
    rec = get_recommender()
    return {"method": rec.method, "num_news": rec.data.info["num_news"] - 1, "num_users": len(rec.last_ref)}


@app.get("/api/users")
def users(limit: int = 30):
    return get_recommender().sample_users(limit)


@app.get("/api/history")
def history(user_id: str, limit: int = 20):
    rec = get_recommender()
    return [rec.news(int(i)) for i in reversed(rec.history(user_id)[-limit:])]


@app.get("/api/recommend")
def recommend(user_id: str, k: int = 10):
    return get_recommender().recommend(user_id, k)


@app.post("/api/click")
def click(c: Click):
    try:
        get_recommender().click(c.user_id, c.news_id)
    except KeyError:
        raise HTTPException(404, "không có tin này")
    return {"ok": True}


@app.get("/api/results")
def results():
    rows = results_table(os.environ.get("NEWSREC_RUNS", "runs"))
    return {"groups": GROUPS, "rows": rows}
