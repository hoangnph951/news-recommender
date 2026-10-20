"""Huấn luyện một cấu hình mô hình, đánh giá trên dev và lưu kết quả vào thư mục run.

Chạy:
    python -m newsrec.train --config configs/full.yaml
    python -m newsrec.train --config configs/full.yaml --set train.seed=2 --name full_seed2
    python -m newsrec.train --config configs/full.yaml --resume      # chạy tiếp sau khi bị ngắt

Kaggle giới hạn mỗi phiên 12 giờ: khi chạm train.time_limit_hours, script lưu last.pt rồi
dừng. Chạy lại cùng lệnh với --resume để huấn luyện tiếp từ đúng vị trí đó.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from newsrec.config import load_config, save_config
from newsrec.data import MindData, ResumableSampler, TrainDataset, collate
from newsrec.evaluate import metrics_report, news_vectors, print_report, save_run, score_impressions, write_prediction
from newsrec.losses import contrastive_loss, ranking_loss
from newsrec.metrics import format_metrics
from newsrec.model import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, cfg):
    plm, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (plm if ".text.layers." in name else rest).append(p)
    groups = [{"params": rest, "lr": cfg["lr"]}]
    if plm:
        groups.append({"params": plm, "lr": cfg["plm_lr"]})
    return torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.0))


def evaluate_split(model, data, cfg, split, impressions, device):
    beh = data.split(split)
    vecs = news_vectors(model, data.info["num_news"], cfg["eval"].get("news_batch_size", 1024), device)
    scores = score_impressions(model, beh, vecs, impressions, cfg["model"]["max_history"], cfg["eval"]["batch_size"])
    return scores, vecs


def train(cfg, out_dir, device="cpu", resume=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    t_cfg = cfg["train"]
    seed_everything(t_cfg["seed"])
    data = MindData(cfg["data"]["dir"])
    dataset = TrainDataset(data.split("train"), cfg["model"]["max_history"], t_cfg["loss"], t_cfg["npratio"],
                           t_cfg["max_candidates"], t_cfg.get("max_train_samples"), t_cfg["seed"])
    dev = data.split("dev")
    rng = np.random.default_rng(0)  # tập con dev cố định để chọn checkpoint trong lúc huấn luyện
    n_sub = t_cfg.get("eval_subset") or len(dev)
    dev_subset = np.sort(rng.choice(len(dev), min(n_sub, len(dev)), replace=False))

    model = build_model(cfg, data).to(device)
    optimizer = make_optimizer(model, t_cfg)
    steps_per_epoch = -(-len(dataset) // t_cfg["batch_size"])
    total_steps = steps_per_epoch * t_cfg["epochs"]
    warmup = int(total_steps * t_cfg.get("warmup_ratio", 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(warmup, 1), max(0.0, (total_steps - s) / max(total_steps - warmup, 1))))
    use_amp = t_cfg.get("amp", True) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best_auc": -1.0, "history": []}
    if resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        random.setstate(ckpt["py_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        print(f"Tiếp tục từ epoch {state['epoch'] + 1}, bước {state['step_in_epoch']}")

    def save_last():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state,
                    "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}, out / "last.pt")

    def validate():
        scores, _ = evaluate_split(model, data, cfg, "dev", dev_subset, device)
        report = metrics_report(dev, scores, dev_subset)
        state["history"].append({"step": state["global_step"], **report["overall"]})
        print(f"[bước {state['global_step']}] dev-subset: {format_metrics(report['overall'])}")
        if report["overall"]["AUC"] > state["best_auc"]:
            state["best_auc"] = report["overall"]["AUC"]
            torch.save({"model": model.state_dict()}, out / "best.pt")
            print("  -> lưu best.pt")
        model.train()

    cl = t_cfg.get("contrastive") or {}
    started, limit = time.time(), t_cfg.get("time_limit_hours") or float("inf")
    eval_every = t_cfg.get("eval_every") or 0
    num_workers = t_cfg.get("num_workers", 0)
    while state["epoch"] < t_cfg["epochs"]:
        sampler = ResumableSampler(len(dataset), t_cfg["seed"], state["epoch"], state["step_in_epoch"] * t_cfg["batch_size"])
        loader = DataLoader(dataset, batch_size=t_cfg["batch_size"], sampler=sampler, collate_fn=collate,
                            num_workers=num_workers, pin_memory=use_amp, persistent_workers=False)
        model.train()
        bar = tqdm(loader, desc=f"epoch {state['epoch'] + 1}/{t_cfg['epochs']}", initial=state["step_in_epoch"],
                   total=steps_per_epoch, mininterval=10)
        running = None
        for batch in bar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=use_amp):
                out_ = model(batch)
                loss = ranking_loss(out_["scores"], batch["target"], batch["cand_mask"])
                if cl.get("weight", 0) > 0:
                    loss = loss + cl["weight"] * contrastive_loss(model.user_encoder, out_["hist_vecs"], batch["history"],
                                                                  cl.get("augment", ["mask", "crop", "reorder"]),
                                                                  cl.get("temperature", 0.2))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state["step_in_epoch"] += 1
            state["global_step"] += 1
            running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
            bar.set_postfix(loss=f"{running:.4f}", refresh=False)
            if eval_every and state["global_step"] % eval_every == 0:
                validate()
            if (time.time() - started) / 3600 > limit:
                save_last()
                (out / "INCOMPLETE").write_text("chạy lại với --resume", encoding="utf-8")
                print(f"Hết thời gian cho phép, đã lưu {out / 'last.pt'}. Chạy lại với --resume.")
                return None
        state["epoch"] += 1
        state["step_in_epoch"] = 0
        validate()
        save_last()

    # ---- đánh giá đầy đủ với checkpoint tốt nhất
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    scores, vecs = evaluate_split(model, data, cfg, "dev", np.arange(len(dev)), device)
    report = metrics_report(dev, scores)
    print_report("dev (đầy đủ)", report)
    save_run(out, cfg["name"], cfg.get("method", cfg["name"]), report, scores,
             {"seed": t_cfg["seed"], "train_history": state["history"],
              "params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    np.save(out / "news_vecs.npy", vecs.cpu().numpy().astype(np.float16))  # demo web dùng trực tiếp
    if data.has_split("test") and cfg["eval"].get("predict_test", False):
        test = data.split("test")
        write_prediction(test, score_impressions(model, test, vecs, np.arange(len(test)), cfg["model"]["max_history"],
                                                 cfg["eval"]["batch_size"]), out / "prediction")
    (out / "INCOMPLETE").unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="ghi đè config, ví dụ model.popularity=false")
    parser.add_argument("--name", help="tên run (mặc định: <name trong config>_seed<seed>)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run_name = args.name or f"{cfg['name']}_seed{cfg['train']['seed']}"
    cfg["name"] = re.sub(r"_seed\d+$", "", run_name)  # các seed của cùng cấu hình được gộp trong bảng kết quả
    report = train(cfg, Path(args.runs_dir) / run_name, args.device, args.resume)
    if report:
        print(json.dumps(report["overall"]))


if __name__ == "__main__":
    main()
