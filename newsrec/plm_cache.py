"""Tính trước hidden state của mô hình ngôn ngữ (PLM) cho mọi tin.

Fine-tune toàn bộ PLM trên MIND-large (khoảng 3,4 triệu click, mỗi mẫu cần encode khoảng
55 tin) quá chậm với GPU miễn phí. Vì vậy ta chạy một lần các tầng dưới của PLM (đóng băng)
rồi lưu lại output, và khi huấn luyện chỉ fine-tune `finetune_layers` tầng trên cùng.
Với finetune_layers = 0, PLM được đóng băng hoàn toàn.

Chạy:
    python -m newsrec.plm_cache --data data/processed/mind_large --device cuda
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from newsrec.data import MindData

DEFAULTS = {"name": "sentence-transformers/all-MiniLM-L12-v2", "fields": "title_abstract", "max_len": 48, "finetune_layers": 2}


def cache_dir(data_dir, plm_cfg):
    cfg = {**DEFAULTS, **plm_cfg}
    slug = re.sub(r"[^A-Za-z0-9]+", "-", cfg["name"]).strip("-")
    return Path(data_dir) / "plm" / f"{slug}_{cfg['fields']}_{cfg['max_len']}_top{cfg['finetune_layers']}"


def load_plm(name, **kwargs):
    from transformers import AutoModel
    return AutoModel.from_pretrained(name, attn_implementation="eager", **kwargs)


def transformer_layers(model):
    """Danh sách các tầng transformer của mô hình kiểu BERT/RoBERTa (MiniLM, BERT, RoBERTa...)."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    raise ValueError("Chỉ hỗ trợ PLM kiểu BERT/RoBERTa (có model.encoder.layer)")


@torch.no_grad()
def build_cache(data, plm_cfg, device="cpu", batch_size=256):
    from transformers import AutoTokenizer

    cfg = {**DEFAULTS, **plm_cfg}
    out = cache_dir(data.dir, cfg)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = load_plm(cfg["name"]).to(device).eval()
    num_layers = len(transformer_layers(model))
    keep = cfg["finetune_layers"]
    if not 0 <= keep <= num_layers:
        raise ValueError(f"finetune_layers phải trong [0, {num_layers}]")

    titles, abstracts = data.meta["title"], data.meta["abstract"]
    n, L, H = len(titles), cfg["max_len"], model.config.hidden_size
    states = np.lib.format.open_memmap(out / "states.npy", mode="w+", dtype=np.float16, shape=(n, L, H))
    mask = np.zeros((n, L), dtype=bool)
    use_amp = device.startswith("cuda")
    for start in tqdm(range(1, n, batch_size), desc="PLM cache"):  # chỉ số 0 là tin padding, giữ toàn 0
        idx = range(start, min(start + batch_size, n))
        pair = [abstracts[i] for i in idx] if cfg["fields"] == "title_abstract" else None
        enc = tokenizer([titles[i] for i in idx], pair, max_length=L, truncation=True,
                        padding="max_length", return_tensors="pt").to(device)
        with torch.autocast("cuda", enabled=use_amp):
            hidden = model(**enc, output_hidden_states=True).hidden_states[num_layers - keep]
        states[idx.start : idx.stop] = hidden.float().cpu().numpy().astype(np.float16)
        mask[idx.start : idx.stop] = enc["attention_mask"].bool().cpu().numpy()
    states.flush()
    np.save(out / "mask.npy", mask)
    (out / "meta.json").write_text(json.dumps({**cfg, "hidden_size": H, "num_layers": num_layers}, indent=2), encoding="utf-8")
    print(f"Đã lưu cache {states.shape} vào {out}")
    return out


class PLMCache:
    """Hidden state đã tính sẵn, đặt trên CPU (mặc định) hoặc GPU nếu đủ bộ nhớ."""

    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"Chưa có PLM cache ở {directory}. Chạy: python -m newsrec.plm_cache")
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.states = torch.from_numpy(np.load(directory / "states.npy"))
        self.mask = torch.from_numpy(np.load(directory / "mask.npy"))
        self.device = torch.device(device)
        if self.device.type == "cuda":
            self.states, self.mask = self.states.to(self.device), self.mask.to(self.device)
        elif torch.cuda.is_available():
            self.states = self.states.pin_memory()

    def lookup(self, ids, device):
        ids = ids.to(self.device)
        return self.states[ids].to(device, non_blocking=True), self.mask[ids].to(device, non_blocking=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/mind_large")
    parser.add_argument("--model", default=DEFAULTS["name"])
    parser.add_argument("--fields", default=DEFAULTS["fields"], choices=["title", "title_abstract"])
    parser.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    parser.add_argument("--finetune-layers", type=int, default=DEFAULTS["finetune_layers"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    build_cache(MindData(args.data), {"name": args.model, "fields": args.fields, "max_len": args.max_len,
                                      "finetune_layers": args.finetune_layers}, args.device, args.batch_size)


if __name__ == "__main__":
    main()
