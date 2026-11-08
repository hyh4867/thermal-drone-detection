"""Shared helpers: config, seeding, run metadata, JSON IO."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]  # repository root


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p.relative_to(ROOT))
    cfg["_config_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return cfg


def path(rel: str) -> Path:
    return ROOT / rel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


_SHA_CACHE: dict[str, str] = {}


def file_sha256(p: Path) -> str:
    key = str(p)
    if key not in _SHA_CACHE:
        sha_file = ROOT / "data" / "dataset_sha256.txt"
        # Reuse the hash recorded at download time when available (6 GB zip).
        if sha_file.exists():
            for line in sha_file.read_text().splitlines():
                h, _, name = line.partition("  ")
                if name and p.name == Path(name).name:
                    _SHA_CACHE[key] = h
                    return h
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def run_metadata(cfg: dict, extra: dict | None = None) -> dict:
    meta = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".")),
        "config": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "seed": cfg.get("seed"),
        "datasets": {
            "anti_uav300": {
                "source": "https://github.com/ZhaoJ9014/Anti-UAV (Google Drive id 1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr)",
                "file": "Anti-UAV-RGBT.zip",
                "sha256": _safe_sha(cfg["data"]["antiuav_zip"]),
            },
            "svanstrom2021": {
                "source": "https://doi.org/10.5281/zenodo.5500576 (v1.0.0)",
                "file": "Drone-detection-dataset-v1.0.0.zip",
                "sha256": _safe_sha(cfg["data"]["svanstrom_zip"]),
            },
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": _cpu_name(),
            "memory_bytes": _memory_bytes(),
            "numpy": np.__version__,
        },
    }
    try:
        import cv2
        import torch
        import torchvision

        meta["env"].update(
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            opencv=cv2.__version__,
            mps_available=torch.backends.mps.is_available(),
        )
    except ImportError:
        pass
    if extra:
        meta.update(extra)
    return meta


def _safe_sha(rel: str) -> str | None:
    p = path(rel)
    return file_sha256(p) if p.exists() else None


def _memory_bytes() -> int | None:
    """Physical memory from the OS (macOS: sysctl hw.memsize), never hard-coded."""
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None


def _cpu_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")


def read_json(p: Path):
    with open(p) as f:
        return json.load(f)
