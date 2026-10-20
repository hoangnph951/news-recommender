"""Huấn luyện và đánh giá NRMS.

Chạy:
    python -m newsrec.train --epochs 3
    python -m newsrec.train --epochs 1 --max-train 5000 --max-dev 2000   # chạy thử nhanh
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from newsrec import preprocess
from newsrec.dataset import TrainDataset, pad_history
from newsrec.metrics import evaluate, format_metrics
from newsrec.nrms import NRMS, NRMSConfig


@torch.no_grad()
def news_vectors(model, num_news, batch_size=1024):
    model.eval()
    device = model.title_tokens.device
    ids = torch.arange(num_news, device=device)
    return torch.cat([model.encode_news(ids[i : i + batch_size]) for i in range(0, num_news, batch_size)])


@torch.no_grad()
def user_vectors(model, vecs, histories, batch_size=512):
    model.eval()
    device = vecs.device
    out = []
    for i in range(0, len(histories), batch_size):
        hist = np.stack([pad_history(h, model.cfg.max_history) for h in histories[i : i + batch_size]])
        hist = torch.from_numpy(hist).to(device)
        out.append(model.encode_user(vecs[hist], hist))
    return torch.cat(out)


def evaluate_model(model, data, behaviors):
    vecs = news_vectors(model, len(data["news_ids"]))
    users = user_vectors(model, vecs, [b["history"] for b in behaviors]).cpu().numpy()
    vecs = vecs.cpu().numpy()
    scores = [vecs[b["candidates"]] @ u for b, u in zip(behaviors, users)]
    return evaluate([b["labels"] for b in behaviors], scores)


def build_model(data, cfg):
    return NRMS(data["title"], len(data["vocab"]), cfg, data.get("embedding"))


def load_model(checkpoint, data, device="cpu"):
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(data, NRMSConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def train(data, cfg, epochs=3, batch_size=64, lr=1e-4, device="cpu", out=None,
          max_train=None, max_dev=None, seed=0):
    torch.manual_seed(seed)
    train_behaviors = data["train"][:max_train]
    dev_behaviors = data["dev"][:max_dev]
    loader = DataLoader(TrainDataset(train_behaviors, cfg.max_history, cfg.npratio, seed),
                        batch_size=batch_size, shuffle=True)
    model = build_model(data, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        total, bar = 0.0, tqdm(loader, desc=f"epoch {epoch}")
        for step, (history, candidates, labels) in enumerate(bar, start=1):
            scores = model(history.to(device), candidates.to(device))
            loss = loss_fn(scores, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            bar.set_postfix(loss=f"{total / step:.4f}")

        metrics = evaluate_model(model, data, dev_behaviors)
        print(f"epoch {epoch} dev: {format_metrics(metrics)}")
        if best is None or metrics["AUC"] > best["AUC"]:
            best = metrics
            if out:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "metrics": metrics}, out)
                print(f"  lưu checkpoint tốt nhất -> {out}")
    return model, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/mind_small.pkl")
    parser.add_argument("--out", default="checkpoints/nrms.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train", type=int, help="chỉ dùng N impression đầu của tập train")
    parser.add_argument("--max-dev", type=int, help="chỉ đánh giá trên N impression đầu của tập dev")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = preprocess.load(args.data)
    cfg = NRMSConfig()
    if data.get("embedding") is not None:
        cfg.embed_dim = data["embedding"].shape[1]
    _, best = train(data, cfg, args.epochs, args.batch_size, args.lr, args.device, args.out,
                    args.max_train, args.max_dev)
    print(f"tốt nhất: {format_metrics(best)}")


if __name__ == "__main__":
    main()
