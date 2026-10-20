import json

import numpy as np
import pytest
import torch

from newsrec.analysis import bootstrap, ensemble, results_table
from newsrec.baselines import BASELINES
from newsrec.config import load_config, save_config
from newsrec.data import TrainDataset, collate
from newsrec.evaluate import metrics_report, save_run
from newsrec.losses import augment, contrastive_loss, ranking_loss
from newsrec.model import Popularity, build_model
from newsrec.train import train


def test_preprocess(data):
    info = data.info
    assert info["num_news"] == 31 and set(info["splits"]) == {"train", "dev", "test"}
    assert data.news["title"].shape == (31, 8) and (data.news["title"][0] == 0).all()
    assert (data.news["entities"][1:, 0] > 0).all()  # mỗi tin có 1 entity
    train = data.split("train")
    assert len(train) == 240 and train.cand_ptr[-1] == len(train.cand) == 240 * 5
    assert (np.bincount(train.impression_of_candidate[train.label == 1]) == 1).all()
    assert data.split("test").label.min() == -1 and not data.split("test").has_labels
    clicks, shows = data.popularity
    assert clicks.sum() == 240 + 60 and shows.sum() == (240 + 60 + 30) * 5  # test chỉ góp số lần hiển thị
    assert train.time.min() == 0 and data.split("dev").time.min() == 24


def test_datasets(data):
    beh = data.split("train")
    sampled = TrainDataset(beh, max_history=6, loss="sampled", npratio=4)
    assert len(sampled) == 240
    item = sampled[0]
    assert item["target"][0] == 1 and len(item["candidates"]) == 5
    assert beh.label[sampled.samples[0]] == 1
    listwise = TrainDataset(beh, max_history=6, loss="listwise", max_candidates=3)
    item = listwise[0]
    assert len(item["candidates"]) == 3 and item["target"].sum() == pytest.approx(1)
    batch = collate([listwise[i] for i in range(4)])
    assert batch["candidates"].shape == (4, 3) and batch["cand_mask"].all()


def test_popularity_has_no_future_leakage(data):
    clicks, shows = data.popularity
    pop = Popularity(clicks, shows)
    news = int(np.argmax(clicks.sum(1)))
    hour = int(np.nonzero(clicks[news])[0][-1])  # giờ cuối cùng tin này có click
    ids, t = torch.tensor([[news]]), torch.tensor([hour])
    feats = pop.features(ids, t)[0, 0]
    expected = clicks[news, max(hour - 24, 0) : hour].sum()  # cửa sổ 24h, không gồm giờ hiện tại
    assert torch.expm1(feats[6]).round().item() == expected


def test_augment_keeps_valid_history():
    vecs = torch.randn(3, 6, 4)
    ids = torch.tensor([[0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 0, 5], [6, 7, 8, 9, 10, 11]])
    for kind, ratio in [("mask", 0.5), ("crop", 0.5), ("reorder", 0.5)]:
        v, new_ids = augment(vecs, ids, kind, ratio)
        assert ((new_ids != 0) <= (ids != 0)).all()  # không tạo ra tin từ vị trí padding
        assert (new_ids != 0).sum(1).min() >= 1
        if kind == "reorder":
            assert sorted(new_ids[2].tolist()) == sorted(ids[2].tolist())


@pytest.mark.parametrize("sets", [
    [],
    ["model.user_encoder=cand"],
    ["model.user_encoder=mean", "model.popularity=false"],
    ["train.loss=listwise"],
    ["model.text_encoder=glove", "model.views=[text]"],
    ["model.plm.finetune_layers=0", "model.plm.fields=title"],
])
def test_variants_forward_backward(cfg, data, sets, tmp_path):
    save_config(cfg, tmp_path / "c.yaml")
    cfg = load_config(tmp_path / "c.yaml", sets)
    model = build_model(cfg, data)
    ds = TrainDataset(data.split("train"), cfg["model"]["max_history"], cfg["train"]["loss"], 4, 10)
    batch = collate([ds[i] for i in range(8)])
    out = model(batch)
    loss = ranking_loss(out["scores"], batch["target"], batch["cand_mask"])
    loss = loss + 0.1 * contrastive_loss(model.user_encoder, out["hist_vecs"], batch["history"], ["mask", "crop", "reorder"], 0.2)
    loss.backward()
    assert torch.isfinite(loss)
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert all(g is not None for n, g in grads.items() if n.startswith("news_encoder.text.")), "text encoder không nhận gradient"


def test_train_full_model_and_resume(cfg, data, tmp_path):
    run = tmp_path / "full"
    cfg["train"]["time_limit_hours"] = 1e-9  # dừng ngay sau bước đầu tiên
    assert train(cfg, run) is None and (run / "INCOMPLETE").exists()
    cfg["train"]["time_limit_hours"] = None
    report = train(cfg, run, resume=True)
    assert report["overall"]["AUC"] > 0.8, report
    assert not (run / "INCOMPLETE").exists()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_impressions"] == 60 and set(metrics) >= {"overall", "overlap", "unseen"}
    assert np.load(run / "dev_scores.npy").shape == (60 * 5,)
    assert np.load(run / "news_vecs.npy").shape == (31, 32)


def test_glove_baseline_trains(cfg, data, tmp_path):
    cfg["model"].update(text_encoder="glove", views=["text"], popularity=False)
    cfg["train"]["contrastive"]["weight"] = 0
    assert train(cfg, tmp_path / "glove")["overall"]["AUC"] > 0.8


def test_baselines_and_analysis(data, data_dir, tmp_path):
    dev = data.split("dev")
    for name, (method, fn) in BASELINES.items():
        scores = fn(data, dev)
        assert scores.shape == dev.cand.shape
        save_run(tmp_path / name, name, method, metrics_report(dev, scores), scores)
    # mỗi user giả lập chỉ thích một chủ đề nên TF-IDF phải gần như hoàn hảo
    assert json.loads((tmp_path / "tfidf" / "metrics.json").read_text(encoding="utf-8"))["overall"]["AUC"] > 0.9
    ens = ensemble([tmp_path / "tfidf", tmp_path / "recent_pop"], data_dir, "ens", runs_dir=tmp_path)
    assert ens["overall"]["AUC"] > 0.5
    rows = results_table(tmp_path)
    assert {r["name"] for r in rows} == {"popularity", "recent_pop", "tfidf", "ens"}
    test = bootstrap(tmp_path / "tfidf", tmp_path / "popularity", data_dir, n=200)
    assert test["AUC"]["diff"] > 0 and test["AUC"]["p_value"] < 0.05


def test_every_experiment_config_builds(tiny_sets, data):
    from newsrec.experiments import EXPERIMENTS

    for group in EXPERIMENTS.values():
        for name, config, sets, _ in group:
            build_model(load_config(config, [*sets, *tiny_sets]), data)
