"""Stage 3 inference: one model, one frame set, reference magnification m (whole image, no TTA, no tiles).

m is the magnification of 640x512 data (Anti-UAV, CST); Svanström (320x256) is enlarged 2m times.
Frames with ``pre_scale`` (the reduced Anti-UAV val set) are first reduced with INTER_AREA.

Usage (always through tools/heavy-run.sh):
  PYTHONPATH=src .venv/bin/python -m tdd.stage3.infer --tag b_m2 --set cst_val --model B --m 2

Models are named in MODELS (checkpoint + anchor sizes). Running a track 1 model on a CST train
set, or anything on cst_val before results/stage3/frozen_config.json exists, is refused.
The cache is appended line by line and resumes after an interruption (same rule as stage 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import cv2
import numpy as np
import torch

from ..common import ROOT, load_config, set_seed
from ..evaluate import load_image
from ..stage2.infer import Predictor
from .data import CST_TRAIN_SETS, OUT, cache_path, frame_set, is_svanstrom

ANCHORS_DEFAULT = (8, 16, 32, 64, 128)
ANCHORS_SMALL = (4, 8, 16, 32, 64)
MODELS = {   # name -> (checkpoint, anchor sizes, uses CST train)
    "S1": ("runs/full_frcnn/best.pt", ANCHORS_DEFAULT, False),
    "B": ("runs/stage2/r1/epoch1.pt", ANCHORS_DEFAULT, False),
    "C37": ("runs/stage2/r2_n37/last.pt", ANCHORS_DEFAULT, False),
    "T1a": ("runs/stage3/t1a/epoch1.pt", ANCHORS_SMALL, False),
    "T1b": ("runs/stage3/t1b/last.pt", ANCHORS_DEFAULT, False),
    "T2_5": ("runs/stage3/t2_5/last.pt", None, True),      # anchors follow the track 1 final model (frozen_track1.json)
    "T2_20": ("runs/stage3/t2_20/last.pt", None, True),
    "T1s1": ("runs/stage3/t1_seed1/last.pt", None, False),  # optional extra run (addendum 1, item 4)
}


def model_spec(name: str):
    ckpt, anchors, cst = MODELS[name]
    if anchors is None:
        t1 = json.loads((OUT / "frozen_track1.json").read_text())
        anchors = tuple(t1["final"]["anchor_sizes"])
        if name == "T1s1":
            ckpt = t1["seed1_checkpoint"]
    return ckpt, tuple(anchors), cst


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def make_predictor(model: str, m: float, set_name: str, config="configs/full.yaml"):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    ckpt, anchors, _ = model_spec(model)
    cfg["model"]["anchor_sizes"] = list(anchors)
    scale = m * (2.0 if is_svanstrom(set_name) else 1.0)
    p = Predictor(cfg, ROOT / ckpt, scale, 1, False)

    def predict(frame):
        g = load_image(frame["image"])
        ps = frame.get("pre_scale")
        if ps:
            g = cv2.resize(g, None, fx=ps, fy=ps, interpolation=cv2.INTER_AREA)
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        d = p._single(g).copy()
        d = d[: p.max_dets]
        d[:, :4] /= scale
        return [[round(float(v), 3) for v in row[:4]] + [float(row[4])] for row in d]

    return predict, {"ckpt": ckpt, "anchor_sizes": list(anchors), "scale": scale}


def guard(model: str, set_name: str):
    uses_cst = MODELS[model][2]
    if set_name in CST_TRAIN_SETS and not uses_cst and not (OUT / "frozen_track1.json").exists():
        raise SystemExit(f"refused: {model} on {set_name} before track 1 is frozen (track 1 must not see CST data)")
    if set_name == "cst_val" and not (OUT / "frozen_config.json").exists():
        raise SystemExit("refused: cst_val before results/stage3/frozen_config.json exists")


def run(tag, set_name, model, m, limit_s_per_frame=None):
    guard(model, set_name)
    frames = frame_set(set_name)
    out = cache_path(tag, set_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    if done >= len(frames):
        print(f"{out.name}: already complete ({done})")
        return
    if done:
        raw = out.read_bytes()
        assert raw.endswith(b"\n"), f"{out.name}: last line is cut; remove the partial line before resuming"
    predict, spec = make_predictor(model, m, set_name)
    t0 = time.time()
    with open(out, "a") as f:
        for i in range(done, len(frames)):
            f.write(json.dumps(predict(frames[i])) + "\n")
            n = i - done + 1
            if n % 64 == 0:
                f.flush()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            if n % 500 == 0 or n == 30:
                spf = (time.time() - t0) / n
                mem = torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0
                print(f"{tag} {set_name}: {i + 1}/{len(frames)} {spf:.3f} s/frame, mps driver {mem:.2f} GB", flush=True)
                if limit_s_per_frame and n >= 30 and spf > limit_s_per_frame:
                    print(f"ABORT {tag}: {spf:.2f} s/frame exceeds the limit {limit_s_per_frame}", flush=True)
                    raise SystemExit(3)
    meta = {"tag": tag, "set": set_name, "model": model, "m": m, **spec, "tile": 1, "tta": False,
            "ckpt_sha256": sha256_file(ROOT / spec["ckpt"]), "frames": len(frames), "resumed_from": done,
            "seconds_this_run": round(time.time() - t0, 1),
            "s_per_frame": round((time.time() - t0) / max(1, len(frames) - done), 4)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--limit-s-per-frame", type=float, default=None)
    a = ap.parse_args()
    run(a.tag, a.set, a.model, a.m, a.limit_s_per_frame)
