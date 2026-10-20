"""Tổng hợp kết quả các run: bảng so sánh, ensemble, kiểm định paired bootstrap.

    python -m newsrec.analysis table                                  # bảng mean ± std theo cấu hình
    python -m newsrec.analysis ensemble runs/full_seed42 runs/full_seed43 --name ens_full
    python -m newsrec.analysis bootstrap runs/full_seed42 runs/nrms_glove_seed42
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from newsrec.data import MindData
from newsrec.evaluate import metrics_report, print_report, save_run
from newsrec.metrics import NAMES, per_impression

GROUPS = ("overall", "overlap", "unseen")


def load_runs(runs_dir="runs"):
    runs = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        m["dir"] = str(path.parent)
        runs.append(m)
    return runs


def results_table(runs_dir="runs"):
    """Gộp các seed của cùng cấu hình: trả về list dòng {name, method, n, group: {metric: (mean, std)}}."""
    by_name = defaultdict(list)
    for r in load_runs(runs_dir):
        by_name[r["name"]].append(r)
    rows = []
    for name, rs in by_name.items():
        row = {"name": name, "method": rs[0].get("method", name), "n": len(rs)}
        for g in GROUPS:
            vals = np.array([[r[g][k] for k in NAMES] for r in rs], dtype=float)
            row[g] = {k: (float(np.nanmean(vals[:, i])), float(np.nanstd(vals[:, i]))) for i, k in enumerate(NAMES)}
        rows.append(row)
    return sorted(rows, key=lambda r: r["overall"]["AUC"][0])


def markdown_table(rows, group="overall"):
    lines = [f"| Mô hình | seed | {' | '.join(NAMES)} |", "|---|---|" + "---|" * len(NAMES)]
    for r in rows:
        cells = [f"{100 * m:.2f} ± {100 * s:.2f}" if r["n"] > 1 else f"{100 * m:.2f}" for m, s in r[group].values()]
        lines.append(f"| {r['method']} | {r['n']} | {' | '.join(cells)} |")
    return "\n".join(lines)


def standardize(scores, ptr):
    """Chuẩn hoá z-score trong từng impression để cộng điểm của các mô hình khác thang đo."""
    out = np.empty_like(scores, dtype=np.float64)
    for a, b in zip(ptr[:-1], ptr[1:]):
        s = scores[a:b].astype(np.float64)
        out[a:b] = (s - s.mean()) / (s.std() + 1e-8)
    return out


def ensemble(run_dirs, data_dir, name, weights=None, runs_dir="runs"):
    dev = MindData(data_dir).split("dev")
    weights = weights or [1.0] * len(run_dirs)
    total = sum(w * standardize(np.load(Path(d) / "dev_scores.npy"), dev.cand_ptr) for w, d in zip(weights, run_dirs))
    report = metrics_report(dev, total.astype(np.float32))
    print_report(name, report)
    save_run(Path(runs_dir) / name, name, f"Ensemble ({len(run_dirs)} mô hình)", report, total.astype(np.float32),
             {"members": [str(d) for d in run_dirs], "weights": weights})
    return report


def bootstrap(run_a, run_b, data_dir, n=1000, seed=0):
    """Paired bootstrap theo impression: A có tốt hơn B một cách có ý nghĩa thống kê không?"""
    dev = MindData(data_dir).split("dev")
    rows_a = per_impression(dev.label, np.load(Path(run_a) / "dev_scores.npy"), dev.cand_ptr)
    rows_b = per_impression(dev.label, np.load(Path(run_b) / "dev_scores.npy"), dev.cand_ptr)
    ok = ~(np.isnan(rows_a).any(1) | np.isnan(rows_b).any(1))
    diff = rows_a[ok] - rows_b[ok]
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean(0) for _ in range(n)])
    result = {}
    for i, k in enumerate(NAMES):
        lo, hi = np.percentile(means[:, i], [2.5, 97.5])
        result[k] = {"diff": float(diff[:, i].mean()), "ci95": [float(lo), float(hi)],
                     "p_value": float((means[:, i] <= 0).mean())}  # một phía: H0 là A không tốt hơn B
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--runs-dir", default="runs")
    t.add_argument("--out", default="results.md")
    e = sub.add_parser("ensemble")
    e.add_argument("runs", nargs="+")
    e.add_argument("--name", required=True)
    e.add_argument("--weights", nargs="*", type=float)
    b = sub.add_parser("bootstrap")
    b.add_argument("run_a")
    b.add_argument("run_b")
    b.add_argument("-n", type=int, default=1000)
    for p in (e, b):
        p.add_argument("--data", default="data/processed/mind_large")
    args = parser.parse_args()

    if args.cmd == "table":
        rows = results_table(args.runs_dir)
        text = "\n\n".join(f"### {g}\n\n{markdown_table(rows, g)}" for g in GROUPS)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
    elif args.cmd == "ensemble":
        ensemble(args.runs, args.data, args.name, args.weights)
    else:
        for k, v in bootstrap(args.run_a, args.run_b, args.data, args.n).items():
            print(f"{k:<8} chênh lệch {100 * v['diff']:+.2f}  CI95 [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}]"
                  f"  p = {v['p_value']:.4f}")


if __name__ == "__main__":
    main()
