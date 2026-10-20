from fastapi.testclient import TestClient

from newsrec.baselines import MODELS, evaluate_recommender
from newsrec.nrms import NRMSConfig
from newsrec.train import evaluate_model, load_model, train

SMALL = NRMSConfig(embed_dim=32, num_heads=4, head_dim=8, attn_hidden=16, dropout=0.0, max_history=10)


def test_preprocess(data):
    assert data["title"].shape == (31, 8)
    assert (data["title"][0] == 0).all()  # tin padding
    assert len(data["train"]) == 200 and len(data["dev"]) == 60
    b = data["train"][0]
    assert b["labels"].sum() == 1 and len(b["candidates"]) == 5


def test_baselines(data):
    for name, cls in MODELS.items():
        metrics = evaluate_recommender(cls().fit(data), data["dev"])
        assert 0 <= metrics["AUC"] <= 1, name
    # mỗi user giả lập chỉ thích một chủ đề nên TF-IDF phải gần như hoàn hảo
    assert evaluate_recommender(MODELS["tfidf"]().fit(data), data["dev"])["AUC"] > 0.9


def test_nrms_learns(data, tmp_path):
    ckpt = tmp_path / "nrms.pt"
    _, best = train(data, SMALL, epochs=3, batch_size=16, lr=3e-3, out=ckpt)
    assert best["AUC"] > 0.8
    # checkpoint lưu mô hình tốt nhất: nạp lại phải cho đúng các độ đo đó
    assert evaluate_model(load_model(ckpt, data), data, data["dev"]) == best


def test_app(data_path, tmp_path, monkeypatch):
    import app.main as main

    monkeypatch.setenv("NEWSREC_DATA", str(data_path))
    monkeypatch.setenv("NEWSREC_CKPT", str(tmp_path / "missing.pt"))  # dùng baseline
    main.get_recommender.cache_clear()
    client = TestClient(main.app)

    assert client.get("/api/info").json()["method"] == "TF-IDF + popularity"
    user = client.get("/api/users").json()[0]["user_id"]
    recs = client.get("/api/recommend", params={"user_id": user, "k": 5}).json()
    assert len(recs) == 5
    before = len(client.get("/api/history", params={"user_id": user, "limit": 100}).json())
    assert client.post("/api/click", json={"user_id": user, "news_id": recs[0]["news_id"]}).json()["ok"]
    after = client.get("/api/history", params={"user_id": user, "limit": 100}).json()
    assert len(after) == before + 1 and after[0]["news_id"] == recs[0]["news_id"]
    assert client.get("/").status_code == 200
    main.get_recommender.cache_clear()
