"""Failure-case gallery from saved predictions (deterministic selection).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.failures --run frcnn --metrics results/model

Categories (at the val-selected operating threshold, IoU 0.5):
  missed_small_drone      Anti-UAV test drone < 16 px with no matching detection
  missed_distant_drone    Svanström DRONE / DISTANT frame, drone not detected
  bird_as_drone           Svanström BIRD frame with a detection overlapping the bird (else any detection)
  airplane_as_drone       Svanström AIRPLANE frame, same rule
  helicopter_as_drone     Svanström HELICOPTER frame, same rule
  false_alarm_no_drone    Anti-UAV test drone-free frame with a detection
  missed_large_drone      Anti-UAV test drone >= 32 px missed (clutter / low contrast)
Within a category, at most one frame per video/sequence; candidates are ranked by
the most confident wrong detection (for false alarms) or by smallest best IoU (for
misses), ties broken by name — no manual picking.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .common import ROOT, path, read_json, load_config, write_json
from .evaluate import load_eval_sets, load_image
from .metrics import iou_matrix, match_frame, size_of

PER_CAT = 3


def _dets(p, thr):
    d = np.asarray(p, dtype=np.float64).reshape(-1, 5)
    return d[d[:, 4] >= thr]


def render(fr, dets, scale, title, out_png):
    g = load_image(fr["image"])
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    full = img.copy()
    focus = None
    for b in fr["gt"]:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        cv2.rectangle(full, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 220, 0), 1)
        focus = ((x1 + x2) // 2, (y1 + y2) // 2)
    for d in dets:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in d[:4]]
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(full, f"{d[4]:.2f}", (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if focus is None:
            focus = ((x1 + x2) // 2, (y1 + y2) // 2)
    H, W = g.shape
    focus = focus or (W // 2, H // 2)
    r = 40
    cx, cy = min(max(focus[0], r), W - r), min(max(focus[1], r), H - r)
    crop = full[cy - r:cy + r, cx - r:cx + r]
    crop = cv2.resize(crop, (H, H), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack([full, crop])
    bar = np.zeros((22, canvas.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack([bar, canvas]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="frcnn")
    ap.add_argument("--metrics", default="results/model")
    ap.add_argument("--out", default="results/failures")
    args = ap.parse_args()
    cfg = load_config(args.config)
    iou_thr = cfg["eval"]["iou_thr"]
    thr = read_json(ROOT / args.metrics / "antiuav_metrics.json")["threshold"]
    man, runs = path(cfg["data"]["manifests_dir"]), ROOT / "runs" / args.run
    _, test_main, sv, supp = load_eval_sets(cfg)
    test = test_main + supp
    test_p = read_json(runs / "preds_antiuav_test.json") + read_json(runs / "preds_antiuav_test_small.json")
    sv_p = read_json(runs / "preds_svanstrom_ir.json")
    assert len(test) == len(test_p) and len(sv) == len(sv_p), "predictions do not match evaluation sets"
    del man
    sv_scale = cfg["eval"]["svanstrom_scale"]

    cands = {k: [] for k in ["missed_small_drone", "missed_distant_drone", "bird_as_drone", "airplane_as_drone",
                             "helicopter_as_drone", "false_alarm_no_drone", "missed_large_drone"]}

    def best_iou(fr, d):
        if not len(d) or not fr["gt"]:
            return 0.0
        return float(iou_matrix(d[:, :4], np.asarray(fr["gt"])).max())

    for fr, p in zip(test, test_p):
        d = _dets(p, thr)
        if not fr["gt"]:
            if len(d):
                cands["false_alarm_no_drone"].append((-float(d[:, 4].max()), fr["seq"], fr, d, 1.0))
            continue
        tp, gm, _ = match_frame(fr["gt"], d, iou_thr)
        if gm.all():
            continue
        s = size_of(fr["gt"][0])
        key = "missed_small_drone" if s < 16 else ("missed_large_drone" if s >= 32 else None)
        if key:
            cands[key].append((best_iou(fr, d), fr["seq"], fr, d, 1.0))
    for fr, p in zip(sv, sv_p):
        d = _dets(p, thr)
        if fr["cls"] == "DRONE":
            if fr["bin"] == "DISTANT" and fr["gt"]:
                _, gm, _ = match_frame(fr["gt"], d, iou_thr)
                if not gm.all():
                    cands["missed_distant_drone"].append((best_iou(fr, d), fr["video"], fr, d, sv_scale))
        elif len(d):
            key = f"{fr['cls'].lower()}_as_drone"
            # prefer frames where the detection lands on the labelled object itself
            on_obj = best_iou(fr, d) >= 0.1
            cands[key].append((-(float(d[:, 4].max()) + (1.0 if on_obj else 0.0)), fr["video"], fr, d, sv_scale))

    index = []
    for cat, lst in cands.items():
        lst.sort(key=lambda t: (t[0], t[1], t[2]["frame"]))
        used, picked = set(), 0
        for rank, (_, vid, fr, d, sc) in enumerate(lst):
            if vid in used:
                continue
            used.add(vid)
            gt_sz = round(size_of(fr["gt"][0]), 1) if fr["gt"] else None
            top = f"{d[:, 4].max():.2f}" if len(d) else "none"
            title = f"{cat} | {vid} f{fr['frame']} | gt {gt_sz}px | top det {top} (thr {thr:.2f})"
            png = ROOT / args.out / f"{cat}_{picked + 1:02d}.png"
            render(fr, d, sc, title, png)
            index.append({"category": cat, "file": str(png.relative_to(ROOT)), "video": vid, "frame": fr["frame"],
                          "gt_size_px_native": gt_sz, "n_dets_above_thr": int(len(d)),
                          "top_score": None if not len(d) else float(d[:, 4].max()),
                          "candidates_in_category": len(lst)})
            picked += 1
            if picked >= PER_CAT:
                break
    write_json({"threshold": thr, "per_category": PER_CAT, "items": index}, ROOT / args.out / "index.json")
    for it in index:
        print(it["category"], it["file"], it["candidates_in_category"])


if __name__ == "__main__":
    main()
