"""Stage 2 inference with input scale, tiling and flip TTA; streamed to a JSONL cache.

Usage:
  PYTHONPATH=src .venv/bin/python -m tdd.stage2.infer --tag s1_x3 --set svan_dev10 --scale 3
  ... --tile 2 (2x2 tiles of the upscaled image, 20% overlap)   --tta (horizontal flip)
  ... --ckpt runs/stage2/r1/best.pt (default: stage 1 model runs/full_frcnn/best.pt)

One image per forward pass (low-memory setting of configs/full.yaml). A killed run
resumes from the last written line. Boxes are stored in native pixel coordinates.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from ..common import ROOT, load_config, set_seed
from ..evaluate import load_image
from ..model import build_model, set_input_size, to_tensor
from .data import cache_path, frame_set

TILE_OVERLAP = 0.2
MERGE_IOU = 0.5


def tile_windows(w: int, h: int, n: int):
    """n x n windows covering (w, h) with TILE_OVERLAP overlap between neighbours."""
    if n == 1:
        return [(0, 0, w, h)]
    tw = int(np.ceil(w / (n - (n - 1) * TILE_OVERLAP)))
    th = int(np.ceil(h / (n - (n - 1) * TILE_OVERLAP)))
    xs = [round(i * (w - tw) / (n - 1)) for i in range(n)]
    ys = [round(i * (h - th) / (n - 1)) for i in range(n)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def merge_tta(a: np.ndarray, b: np.ndarray, iou: float = MERGE_IOU) -> np.ndarray:
    """Pre-registered flip-TTA merge: NMS over both passes; the kept box's score is
    the mean of the two passes' scores for that object (a pass that has no box
    overlapping it contributes 0, i.e. the score is halved)."""
    from ..metrics import iou_matrix

    if len(a) == 0 and len(b) == 0:
        return np.zeros((0, 5))
    allb = np.concatenate([a, b]) if len(a) and len(b) else (a if len(a) else b)
    src = np.concatenate([np.zeros(len(a), int), np.ones(len(b), int)])
    order = np.argsort(-allb[:, 4], kind="stable")
    allb, src = allb[order], src[order]
    ious = iou_matrix(allb[:, :4], allb[:, :4])
    alive = np.ones(len(allb), bool)
    out = []
    for i in range(len(allb)):
        if not alive[i]:
            continue
        cluster = alive & (ious[i] >= iou)
        cluster[i] = True
        other = cluster & (src != src[i])
        s_other = allb[other, 4].max() if other.any() else 0.0
        out.append([*allb[i, :4], 0.5 * (allb[i, 4] + s_other)])
        alive &= ~cluster
    out = np.asarray(out)
    return out[np.argsort(-out[:, 4], kind="stable")]


class Predictor:
    def __init__(self, cfg, ckpt, scale=1.0, tile=1, tta=False):
        self.device = torch.device(cfg["train"]["device"] if torch.backends.mps.is_available() else "cpu")
        m = build_model(cfg["model"])
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.m = m.to(self.device).eval()
        self.scale, self.tile, self.tta = scale, tile, tta
        self.max_dets = cfg["model"]["detections_per_img"]

    @torch.no_grad()
    def _forward(self, g: np.ndarray) -> np.ndarray:
        h, w = g.shape
        set_input_size(self.m, min(h, w), max(h, w))
        r = self.m([to_tensor(g, self.device)])[0]
        b = r["boxes"].float().cpu().numpy()
        s = r["scores"].float().cpu().numpy()[:, None]
        return np.concatenate([b, s], 1).reshape(-1, 5)

    def _single(self, g: np.ndarray) -> np.ndarray:
        """Detections on an (already scaled) image, in that image's coordinates."""
        h, w = g.shape
        if self.tile == 1:
            return self._forward(g)
        parts = []
        for x1, y1, x2, y2 in tile_windows(w, h, self.tile):
            d = self._forward(np.ascontiguousarray(g[y1:y2, x1:x2]))
            d[:, [0, 2]] += x1
            d[:, [1, 3]] += y1
            parts.append(d)
        d = np.concatenate(parts)
        if len(d):
            keep = nms(torch.from_numpy(d[:, :4]).float(), torch.from_numpy(d[:, 4]).float(), MERGE_IOU).numpy()
            d = d[keep][: self.max_dets]
        return d

    def __call__(self, frame: dict) -> list:
        g = load_image(frame["image"])
        if self.scale != 1.0:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        d = self._single(g)
        if self.tta:
            f = self._single(np.ascontiguousarray(g[:, ::-1]))
            w = g.shape[1]
            f[:, [0, 2]] = w - f[:, [2, 0]]
            d = merge_tta(d, f)[: self.max_dets]
        d = d.copy()
        d[:, :4] /= self.scale
        return [[round(float(v), 3) for v in row[:4]] + [float(row[4])] for row in d]


def run(tag, set_name, ckpt, scale, tile, tta, config="configs/full.yaml", limit_s_per_frame=None):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    frames = frame_set(set_name)
    out = cache_path(tag, set_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    if done >= len(frames):
        print(f"{out.name}: already complete ({done})")
        return
    pred = Predictor(cfg, ROOT / ckpt, scale, tile, tta)
    t0 = time.time()
    with open(out, "a") as f:
        for i in range(done, len(frames)):
            f.write(json.dumps(pred(frames[i])) + "\n")
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
                    print(f"ABORT {tag}: {spf:.2f} s/frame exceeds the pre-registered limit {limit_s_per_frame}", flush=True)
                    raise SystemExit(3)
    meta = {"tag": tag, "set": set_name, "ckpt": ckpt, "scale": scale, "tile": tile, "tta": tta,
            "frames": len(frames), "seconds_this_run": round(time.time() - t0, 1),
            "s_per_frame": round((time.time() - t0) / max(1, len(frames) - done), 4)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("DONE", json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--ckpt", default="runs/full_frcnn/best.pt")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit-s-per-frame", type=float, default=None)
    a = ap.parse_args()
    run(a.tag, a.set, a.ckpt, a.scale, a.tile, a.tta, limit_s_per_frame=a.limit_s_per_frame)


if __name__ == "__main__":
    main()
