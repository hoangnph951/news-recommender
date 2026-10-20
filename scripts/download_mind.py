"""Tải và giải nén MIND small (train + dev) vào thư mục data/.

Chạy:
    python scripts/download_mind.py

Nếu link không tải được, hãy tải thủ công từ https://msnews.github.io/ rồi giải nén thành
data/MINDsmall_train/ và data/MINDsmall_dev/ (mỗi thư mục có news.tsv, behaviors.tsv).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://recodatasets.z20.web.core.windows.net/newsrec"
FILES = {"MINDsmall_train": "MINDsmall_train.zip", "MINDsmall_dev": "MINDsmall_dev.zip"}


def progress(blocks, block_size, total):
    if total > 0:
        print(f"\r  {min(blocks * block_size / total, 1):.0%}", end="", flush=True)


def main(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    for folder, filename in FILES.items():
        target = data_dir / folder
        if (target / "behaviors.tsv").exists():
            print(f"{target} đã có, bỏ qua")
            continue
        archive = data_dir / filename
        print(f"Tải {BASE_URL}/{filename}")
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", archive, progress)
        except OSError as e:
            sys.exit(f"\nKhông tải được ({e}). Hãy tải thủ công từ https://msnews.github.io/")
        print()
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        archive.unlink()
        print(f"Đã giải nén vào {target}")


if __name__ == "__main__":
    main(*sys.argv[1:])
