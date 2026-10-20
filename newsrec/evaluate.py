"""Chấm điểm toàn bộ ứng viên của mọi impression và tính độ đo.

Mọi lần chạy (baseline hay mô hình) đều ghi ra cùng một định dạng trong thư mục run:
  metrics.json     độ đo trên tập dev: tổng thể, user đã có trong train (overlap), user mới (unseen)
  dev_scores.npy   điểm của từng ứng viên, thẳng hàng với mảng cand của behaviors_dev
nhờ đó các bước ensemble, kiểm định thống kê và bảng kết quả dùng chung cho mọi phương pháp.

Chạy lại đánh giá cho một run đã huấn luyện:
    python -m newsrec.evaluate --run runs/full_seed42
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from newsrec.data import pad_history
from newsrec.metrics import format_metrics, per_impression, summarize


@torch.no_grad()
def news_vectors(model, num_news, batch_size=1024, device="cpu"):
    model.eval()
    out = []
    for start in tqdm(range(0, num_news, batch_size), desc="news vectors", leave=False):
        ids = torch.arange(start, min(start + batch_size, num_news), device=device)
        out.append(model.news_encoder(ids).float())
    return torch.cat(out)


@torch.no_grad()
def score_impressions(model, beh, vecs, impressions, max_history, batch_size=256):
    """Điểm của mọi ứng viên trong các impression đã chọn (trả về mảng phẳng theo thứ tự đó)."""
    model.eval()
    device = vecs.device
    out = []
    for start in tqdm(range(0, len(impressions), batch_size), desc="scoring", leave=False):
        imps = impressions[start : start + batch_size]
        hist = torch.from_numpy(np.stack([pad_history(beh.history(i), max_history) for i in imps])).to(device)
        lengths = [int(beh.cand_ptr[i + 1] - beh.cand_ptr[i]) for i in imps]
        cands = torch.zeros(len(imps), max(lengths), dtype=torch.long)
        for b, i in enumerate(imps):
            cands[b, : lengths[b]] = torch.from_numpy(beh.candidates(i).astype(np.int64))
        cands = cands.to(device)
        time = torch.from_numpy(beh.time[imps].astype(np.int64)).to(device)
        scores = model.score_vectors(vecs[hist], hist, vecs[cands], cands, time)["scores"].cpu().numpy()
        out.extend(scores[b, : lengths[b]] for b in range(len(imps)))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def subset_ptr(beh, impressions):
    lengths = beh.cand_ptr[impressions + 1] - beh.cand_ptr[impressions]
    return np.concatenate([[0], np.cumsum(lengths)])


def subset_labels(beh, impressions):
    return np.concatenate([beh.labels(i) for i in impressions])


def metrics_report(beh, scores, impressions=None):
    """Độ đo tổng thể và theo nhóm user (giống bảng L9: overlap / unseen users)."""
    if impressions is None:
        rows = per_impression(beh.label, scores, beh.cand_ptr)
        users = beh.user
    else:
        rows = per_impression(subset_labels(beh, impressions), scores, subset_ptr(beh, impressions))
        users = beh.user[impressions]
    return {
        "overall": summarize(rows),
        "overlap": summarize(rows, users > 0),
        "unseen": summarize(rows, users == 0),
        "num_impressions": int(len(rows)),
    }


def write_prediction(beh, scores, path):
    """File nộp leaderboard MIND: mỗi dòng 'impression_id [thứ hạng của từng ứng viên]'."""
    path = Path(path)
    with open(path.with_suffix(".txt"), "w") as f:
        for i in range(len(beh)):
            s = scores[beh.cand_ptr[i] : beh.cand_ptr[i + 1]]
            ranks = np.empty(len(s), dtype=np.int64)
            ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
            f.write(f"{beh.impression_id[i]} [{','.join(map(str, ranks))}]\n")
    with zipfile.ZipFile(path.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path.with_suffix(".txt"), arcname="prediction.txt")


def save_run(run_dir, name, method, metrics, dev_scores, extra=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "dev_scores.npy", dev_scores.astype(np.float32))
    (run_dir / "metrics.json").write_text(json.dumps({"name": name, "method": method, **metrics, **(extra or {})},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")


def print_report(title, report):
    print(f"{title}  [{report['num_impressions']} impression]")
    for group in ("overall", "overlap", "unseen"):
        print(f"  {group:<8} {format_metrics(report[group])}")


def main():
    from newsrec.config import load_config
    from newsrec.data import MindData
    from newsrec.model import build_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--test", action="store_true", help="ghi thêm prediction.zip cho tập test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    data = MindData(cfg["data"]["dir"])
    model = build_model(cfg, data).to(args.device)
    model.load_state_dict(torch.load(run / args.checkpoint, map_location=args.device)["model"])
    vecs = news_vectors(model, data.info["num_news"], device=args.device)
    dev = data.split("dev")
    scores = score_impressions(model, dev, vecs, np.arange(len(dev)), cfg["model"]["max_history"], cfg["eval"]["batch_size"])
    report = metrics_report(dev, scores)
    print_report("dev", report)
    save_run(run, cfg["name"], cfg.get("method", cfg["name"]), report, scores)
    if args.test and data.has_split("test"):
        test = data.split("test")
        write_prediction(test, score_impressions(model, test, vecs, np.arange(len(test)), cfg["model"]["max_history"],
                                                 cfg["eval"]["batch_size"]), run / "prediction")


if __name__ == "__main__":
    main()
