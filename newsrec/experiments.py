"""Danh sách thực nghiệm của đồ án và trình chạy tuần tự.

Chạy lại cùng một lệnh nhiều lần (ví dụ qua nhiều phiên Kaggle) là an toàn: run đã xong
(có metrics.json) được bỏ qua, run đang dở (có last.pt) được chạy tiếp.

    python -m newsrec.experiments --list
    python -m newsrec.experiments main
    python -m newsrec.experiments ablation --set train.max_train_samples=1000000
"""
import argparse
from pathlib import Path

import torch

from newsrec.config import load_config
from newsrec.train import train

# tên -> (config, các ghi đè, mô tả)
EXPERIMENTS = {
    "main": [
        ("nrms_glove", "configs/nrms_glove.yaml", [], "NRMS (GloVe, tái lập)"),
        ("full", "configs/full.yaml", [], "Đề xuất đầy đủ"),
    ],
    "baselines_extra": [
        ("naml_glove", "configs/naml_glove.yaml", [], "NAML-style (GloVe)"),
        ("plm_nrms", "configs/full.yaml",
         ["model.views=[text]", "model.popularity=false", "train.contrastive.weight=0"], "NRMS với PLM (chỉ đổi news encoder)"),
    ],
    "ablation": [  # thường chạy với ít dữ liệu hơn (--set), nên có dòng tham chiếu cùng thiết lập
        ("abl_reference", "configs/full.yaml", [], "Đầy đủ (cùng thiết lập ablation)"),
        ("abl_no_views", "configs/full.yaml", ["model.views=[text]"], "Bỏ category/subcategory/entity"),
        ("abl_title_only", "configs/full.yaml", ["model.plm.fields=title", "model.plm.max_len=32"], "Chỉ dùng tiêu đề"),
        ("abl_no_pop", "configs/full.yaml", ["model.popularity=false"], "Bỏ popularity"),
        ("abl_no_cl", "configs/full.yaml", ["train.contrastive.weight=0"], "Bỏ học tương phản"),
        ("abl_frozen_plm", "configs/full.yaml", ["model.plm.freeze=true"], "PLM đóng băng"),
        ("abl_user_mean", "configs/full.yaml", ["model.user_encoder=mean"], "User encoder trung bình"),
        ("abl_user_cand", "configs/full.yaml", ["model.user_encoder=cand"], "User encoder candidate-aware"),
        ("abl_listwise", "configs/full.yaml", ["train.loss=listwise", "train.batch_size=16"], "Loss listwise"),
    ],
}


def run_group(group, seeds, extra_sets, runs_dir, device, only=None):
    for name, config, sets, desc in EXPERIMENTS[group]:
        if only and name not in only:
            continue
        for seed in seeds:
            run = Path(runs_dir) / f"{name}_seed{seed}"
            if (run / "metrics.json").exists() and not (run / "INCOMPLETE").exists():
                print(f"[bỏ qua] {run} đã xong")
                continue
            cfg = load_config(config, [*sets, f"train.seed={seed}", *extra_sets])
            cfg["name"], cfg["method"] = name, desc
            print(f"\n===== {run} : {desc} =====")
            if train(cfg, run, device, resume=True) is None:
                print("Dừng do hết thời gian của phiên. Chạy lại lệnh này để tiếp tục.")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", nargs="?", choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--only", nargs="*", help="chỉ chạy các thực nghiệm có tên này")
    parser.add_argument("--set", nargs="*", default=[], help="ghi đè thêm cho mọi thực nghiệm")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.list or not args.group:
        for group, items in EXPERIMENTS.items():
            print(group)
            for name, config, sets, desc in items:
                print(f"  {name:<16} {desc:<40} {config} {' '.join(sets)}")
        return
    run_group(args.group, args.seeds, args.set, args.runs_dir, args.device, args.only)


if __name__ == "__main__":
    main()
