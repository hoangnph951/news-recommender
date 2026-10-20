"""Tải và giải nén MIND từ HuggingFace (link chính thức trên https://msnews.github.io/).

Bộ dữ liệu là gated: cần đăng nhập HuggingFace, đồng ý điều khoản ở
https://huggingface.co/datasets/yjw1029/MIND rồi đăng nhập trên máy bằng `hf auth login`
(hoặc đặt biến môi trường HF_TOKEN, ví dụ Kaggle Secrets).

Chạy:
    python scripts/download_mind.py --size large            # train + dev + test vào data/raw/
    python scripts/download_mind.py --size large --glove    # thêm GloVe 840B (khoảng 2 GB) cho baseline NRMS
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "yjw1029/MIND"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"


def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)


def download_mind(size, splits, raw):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    for split in splits:
        target = raw / f"MIND{size}_{split}"
        if (target / "behaviors.tsv").exists():
            print(f"{target} đã có, bỏ qua")
            continue
        filename = f"MIND{size}_{split}.zip"
        print(f"Tải {REPO}/{filename}")
        try:
            archive = hf_hub_download(REPO, filename, repo_type="dataset", cache_dir=raw / ".hf_cache")
        except GatedRepoError:
            sys.exit("Chưa có quyền tải: hãy đồng ý điều khoản ở https://huggingface.co/datasets/yjw1029/MIND "
                     "rồi chạy `hf auth login` (hoặc đặt HF_TOKEN).")
        extract(archive, target)
        print(f"  -> {target}")
    shutil.rmtree(raw / ".hf_cache", ignore_errors=True)  # xoá file zip đã giải nén cho đỡ tốn chỗ


def download_glove(raw):
    target = raw / "glove" / "glove.840B.300d.txt"
    if target.exists():
        print(f"{target} đã có, bỏ qua")
        return
    archive = raw / "glove.840B.300d.zip"
    print(f"Tải {GLOVE_URL}")
    urllib.request.urlretrieve(GLOVE_URL, archive)
    extract(archive, target.parent)
    archive.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="large", choices=["large", "small"])
    parser.add_argument("--splits", nargs="*", default=None, help="mặc định: train dev (+ test nếu large)")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--glove", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw)
    splits = args.splits or (["train", "dev", "test"] if args.size == "large" else ["train", "dev"])
    download_mind(args.size, splits, raw)
    if args.glove:
        download_glove(raw)


if __name__ == "__main__":
    main()
