"""Demo web: chọn một user, xem tin được gợi ý, bấm đọc tin để lịch sử và gợi ý thay đổi.

Chạy (từ thư mục gốc của repo):
    uvicorn app.main:app --reload
rồi mở http://127.0.0.1:8000

Biến môi trường:
    NEWSREC_DATA  file tiền xử lý (mặc định data/processed/mind_small.pkl)
    NEWSREC_CKPT  checkpoint NRMS (mặc định checkpoints/nrms.pt). Không có checkpoint
                  thì demo dùng baseline TF-IDF + popularity.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from newsrec import preprocess
from newsrec.baselines import HybridRecommender
from newsrec.train import load_model, news_vectors, user_vectors

STATIC = Path(__file__).parent / "static"


class Recommender:
    def __init__(self, data_path, ckpt_path):
        self.data = preprocess.load(data_path)
        self.histories = {}
        for b in self.data["train"] + self.data["dev"]:  # lịch sử mới nhất của mỗi user
            clicked = b["candidates"][b["labels"] == 1]
            self.histories[b["user_id"]] = list(b["history"]) + list(clicked)

        if Path(ckpt_path).exists():
            self.model = load_model(ckpt_path, self.data)
            self.vecs = news_vectors(self.model, len(self.data["news_ids"]))
            self.method = "NRMS"
        else:
            self.model = None
            self.fallback = HybridRecommender().fit(self.data)
            self.method = "TF-IDF + popularity"

    def scores(self, history):
        all_news = np.arange(1, len(self.data["news_ids"]))
        if self.model is None:
            return all_news, self.fallback.score(np.array(history, dtype=np.int64), all_news)
        user = user_vectors(self.model, self.vecs, [np.array(history, dtype=np.int64)])[0]
        return all_news, (self.vecs[1:] @ user).cpu().numpy()

    def recommend(self, user_id, k=10):
        history = self.histories.get(user_id, [])
        news, scores = self.scores(history)
        scores[np.isin(news, history)] = -np.inf  # không gợi ý lại tin đã đọc
        k = min(k, len(news) - 1)
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.news(int(news[i]), float(scores[i])) for i in top]

    def news(self, idx, score=None):
        d = self.data
        item = {
            "news_id": d["news_ids"][idx], "title": d["title_text"][idx],
            "abstract": d["abstract_text"][idx], "category": d["category_text"][idx],
            "subcategory": d["subcategory_text"][idx], "url": d["url"][idx],
        }
        if score is not None:
            item["score"] = round(score, 4)
        return item


@lru_cache
def get_recommender():
    return Recommender(
        os.environ.get("NEWSREC_DATA", "data/processed/mind_small.pkl"),
        os.environ.get("NEWSREC_CKPT", "checkpoints/nrms.pt"),
    )


app = FastAPI(title="News Recommender")


class Click(BaseModel):
    user_id: str
    news_id: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    rec = get_recommender()
    return {"method": rec.method, "num_news": len(rec.data["news_ids"]) - 1, "num_users": len(rec.histories)}


@app.get("/api/users")
def users(limit: int = 50):
    rec = get_recommender()
    ranked = sorted(rec.histories.items(), key=lambda kv: -len(kv[1]))[:limit]
    return [{"user_id": u, "history_len": len(h)} for u, h in ranked]


@app.get("/api/history")
def history(user_id: str, limit: int = 20):
    rec = get_recommender()
    return [rec.news(int(i)) for i in reversed(rec.histories.get(user_id, [])[-limit:])]


@app.get("/api/recommend")
def recommend(user_id: str, k: int = 10):
    with torch.no_grad():
        return get_recommender().recommend(user_id, k)


@app.post("/api/click")
def click(c: Click):
    rec = get_recommender()
    idx = rec.data["news_index"].get(c.news_id)
    if idx is None:
        raise HTTPException(404, "không có tin này")
    rec.histories.setdefault(c.user_id, []).append(idx)
    return {"ok": True, "history_len": len(rec.histories[c.user_id])}
