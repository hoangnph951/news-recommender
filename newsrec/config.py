"""Cấu hình thực nghiệm bằng YAML.

Một file config có thể kế thừa file khác bằng khoá `base:` và ghi đè từng khoá.
Từ dòng lệnh có thể ghi đè thêm: --set model.popularity=false train.seed=2
"""
import copy
from pathlib import Path

import yaml


def merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base" in cfg:
        cfg = merge(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
