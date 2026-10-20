"""Hệ gợi ý tin tức trên bộ dữ liệu MIND."""
import sys

# Windows mặc định in ra bằng cp1252, không in được tiếng Việt khi chuyển hướng output ra file.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
