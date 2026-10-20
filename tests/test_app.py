import pytest
from fastapi.testclient import TestClient

from newsrec.train import train


@pytest.fixture
def client_factory(data_dir, tmp_path, monkeypatch):
    import app.main as main

    def make(run_dir):
        monkeypatch.setenv("NEWSREC_DATA", str(data_dir))
        monkeypatch.setenv("NEWSREC_RUN", str(run_dir))
        monkeypatch.setenv("NEWSREC_RUNS", str(tmp_path / "runs"))
        monkeypatch.setenv("NEWSREC_FEEDBACK", str(tmp_path / "feedback.db"))
        main.get_recommender.cache_clear()
        return TestClient(main.app)

    yield make
    main.get_recommender.cache_clear()


def check_demo(client):
    user = client.get("/api/users").json()[0]["user_id"]
    rec = client.get("/api/recommend", params={"user_id": user, "k": 5}).json()
    assert len(rec["items"]) == 5
    history = client.get("/api/history", params={"user_id": user, "limit": 100}).json()
    read = {h["news_id"] for h in history}
    assert not read & {r["news_id"] for r in rec["items"]}  # không gợi ý lại tin đã đọc
    clicked = rec["items"][0]["news_id"]
    assert client.post("/api/click", json={"user_id": user, "news_id": clicked}).json()["ok"]
    after = client.get("/api/history", params={"user_id": user, "limit": 100}).json()
    assert after[0]["news_id"] == clicked
    assert clicked not in {r["news_id"] for r in client.get("/api/recommend", params={"user_id": user}).json()["items"]}
    assert len(client.get("/api/recommend", params={"user_id": "__new__"}).json()["items"]) == 10
    for page in ("/", "/results", "/static/style.css"):
        assert client.get(page).status_code == 200
    return rec


def test_app_without_model(client_factory, tmp_path):
    client = client_factory(tmp_path / "missing")
    assert "phổ biến" in client.get("/api/info").json()["method"]
    check_demo(client)
    assert client.get("/api/results").json()["rows"] == []


def test_app_with_trained_model(client_factory, cfg, tmp_path):
    cfg["train"]["epochs"] = 2
    cfg["model"]["user_encoder"] = "cand"
    run = tmp_path / "runs" / "full_seed42"
    train(cfg, run)
    client = client_factory(run)
    assert client.get("/api/info").json()["method"] == cfg["method"]
    rec = check_demo(client)
    assert 0 <= rec["popularity_weight"] <= 1
    assert any(item["because"] for item in rec["items"])  # có giải thích từ lịch sử đọc
    rows = client.get("/api/results").json()["rows"]
    assert rows[0]["name"] == "full" and rows[0]["overall"]["AUC"][0] > 0.5
