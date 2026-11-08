"""Training-free baseline: morphological top-hat spot detector.

For each frame: response = max(white top-hat, black top-hat) (hot or cold point
targets against a smooth background), z-score it, threshold at ``k`` std, take
8-connected components as candidate boxes, score = peak z-score.

Usage:
  PYTHONPATH=src .venv/bin/python -m tdd.baseline            # grid search on val + full evaluation
"""
from __future__ import annotations

import argparse
import itertools
import time

import cv2
import numpy as np

from .common import ROOT, load_config, path, read_json, run_metadata, set_seed, write_json
from .evaluate import evaluate_all, load_image, ap_on



class TopHatDetector:
    def __init__(self, kernel: int, k: float, polarity: str = "both", max_dets: int = 10,
                 min_area: int = 2, pad: int = 1):
        self.kernel, self.k, self.polarity = kernel, k, polarity
        self.max_dets, self.min_area, self.pad = max_dets, min_area, pad
        self._se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))

    def response(self, gray: np.ndarray) -> np.ndarray:
        """Top-hat contrast map (uint8 morphology: exact for 8-bit input, ~10x faster)."""
        resp = None
        if self.polarity in ("both", "white"):
            resp = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self._se)
        if self.polarity in ("both", "black"):
            b = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._se)
            resp = b if resp is None else np.maximum(resp, b)
        return resp.astype(np.float32)

    def detect(self, resp: np.ndarray, ignore=(), k: float | None = None) -> np.ndarray:
        k = self.k if k is None else k
        if ignore:
            resp = resp.copy()
            for x1, y1, x2, y2 in ignore:  # e.g. burned-in OSD text
                resp[y1:y2, x1:x2] = 0.0
        mu, sd = float(resp.mean()), float(resp.std())
        z = (resp - mu) / max(sd, 1e-6)
        mask = (z > k).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        H, W = resp.shape
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            zz = z[y:y + h, x:x + w]
            peak = float(zz[lab[y:y + h, x:x + w] == i].max())
            p = self.pad
            dets.append([max(x - p, 0), max(y - p, 0), min(x + w + p, W), min(y + h + p, H), peak])
        if not dets:
            return np.zeros((0, 5), np.float32)
        d = np.asarray(dets, np.float32)
        return d[np.argsort(-d[:, 4])[: self.max_dets]]

    def __call__(self, gray: np.ndarray, ignore=()) -> np.ndarray:
        return self.detect(self.response(gray), ignore)


def make_predict_fn(det: TopHatDetector, osd_rects=()):
    """``osd_rects`` are ignored only on Anti-UAV frames (the OSD text is burned
    into those videos; Svanström frames have none)."""
    def predict(frames, scale: float = 1.0):
        out = []
        for fr in frames:
            g = load_image(fr["image"])
            if scale != 1.0:
                g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            d = det(g, osd_rects if fr["image"].startswith("antiuav/") else ())
            if scale != 1.0 and len(d):
                d[:, :4] /= scale
            out.append(d.tolist())
        return out
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--reuse-grid", action="store_true", help="reuse the val selection in <out>/val_grid_search.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    bc = cfg["baseline"]
    val = read_json(path(cfg["data"]["manifests_dir"]) / "antiuav_val.json")

    if args.reuse_grid:
        best = read_json(ROOT / args.out / "val_grid_search.json")["selected"]
        det = TopHatDetector(best["kernel"], best["k"], bc["polarity"], bc["max_dets"], bc["min_area"], bc["pad"])
        evaluate_all(cfg, make_predict_fn(det, bc["osd_rects"] if best["osd_mask"] else ()), name="baseline_tophat",
                     out_dir=ROOT / args.out,
                     method_info={"method": "top-hat spot detector (training-free)",
                                  "params": {**bc, "kernel": best["kernel"], "k": best["k"], "osd_mask": best["osd_mask"]},
                                  "selection": "val_grid_search.json (full val, stride 25)"})
        return
    # --- hyper-parameter selection on val only ---
    grid = []
    t0 = time.time()
    # One top-hat response per (frame, kernel); every (k, osd_mask) reuses it.
    combos = list(itertools.product(bc["kernel_grid"], bc["k_grid"], bc["osd_mask_grid"]))
    preds = {c: [] for c in combos}
    dets_by_kernel = {kern: TopHatDetector(kern, 0.0, bc["polarity"], bc["max_dets"], bc["min_area"], bc["pad"])
                      for kern in bc["kernel_grid"]}
    for fr in val:
        g = load_image(fr["image"])
        for kern, det in dets_by_kernel.items():
            resp = det.response(g)
            for k in bc["k_grid"]:
                for osd in bc["osd_mask_grid"]:
                    preds[(kern, k, osd)].append(det.detect(resp, bc["osd_rects"] if osd else (), k).tolist())
    for (kernel, k, osd) in combos:
        r = ap_on(val, preds[(kernel, k, osd)], cfg["eval"]["iou_thr"])
        grid.append({"kernel": kernel, "k": k, "osd_mask": osd, "val_ap50": r})
        print(f"kernel={kernel:>2} k={k:<4} osd_mask={osd!s:<5} val AP50={r:.4f}", flush=True)
    best = max(grid, key=lambda g: g["val_ap50"])
    print("best", best, f"grid search {time.time() - t0:.0f}s")
    det = TopHatDetector(best["kernel"], best["k"], bc["polarity"], bc["max_dets"], bc["min_area"], bc["pad"])

    out = ROOT / args.out
    write_json({"grid": grid, "selected": best, "selection": "max AP@0.5 on val (Anti-UAV, source-video split)",
                "meta": run_metadata(cfg)}, out / "val_grid_search.json")
    evaluate_all(cfg, make_predict_fn(det, bc["osd_rects"] if best["osd_mask"] else ()), name="baseline_tophat", out_dir=out,
                 method_info={"method": "top-hat spot detector (training-free)",
                              "params": {**bc, "kernel": best["kernel"], "k": best["k"],
                                         "osd_mask": best["osd_mask"]}})


if __name__ == "__main__":
    main()
