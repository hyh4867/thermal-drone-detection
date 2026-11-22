"""Per-GT and per-frame tables shared by every stage 2 analysis.

Both datasets have at most one labelled object per frame, so "GT g is found at
score threshold t under criterion c" is simply ``best_score_c(g) >= t`` where
``best_score_c`` is the highest score among detections that satisfy c with g.
(With one GT per frame this equals the greedy matching used in stage 1; the
equality is checked in tests/test_stage2.py and against the stage 1 numbers.)
"""
from __future__ import annotations

import math

import numpy as np

from ..metrics import iou_matrix
from .data import ref_scale

CRITERIA = ("iou50", "iou30", "iou10", "center")
SIZE_EDGES = [0, 8, 12, 16, 24, 32, 48, 64, 1e9]          # sqrt(w*h) in 640x512 reference pixels
SIZE_NAMES = ["<8", "8-12", "12-16", "16-24", "24-32", "32-48", "48-64", ">=64"]


def _det(p):
    return np.asarray(p, dtype=np.float64).reshape(-1, 5)


def build(frames, preds, unit_key, positive=lambda f: True):
    """Returns (gt_table, frame_table) as dicts of numpy arrays.

    gt_table: one row per GT box of frames where ``positive(frame)`` (the object is a drone).
      unit, size (reference px), best score per criterion (0 if none), max IoU over
      detections >= 0.954 / over all stored detections, and the best loose match
      (highest-score detection with IoU >= 0.1) for box-convention statistics.
    frame_table: one row per frame: unit, has_gt, positive, max score (0 if no detection).
    """
    g = {k: [] for k in ("unit", "frame_idx", "size", "gw", "gh", *CRITERIA, "max_iou_all", "loose_score",
                         "loose_iou", "loose_rw", "loose_rh", "loose_dx", "loose_dy")}
    fr = {k: [] for k in ("unit", "has_gt", "positive", "max_score", "n_gt")}
    for i, (f, p) in enumerate(zip(frames, preds)):
        d = _det(p)
        pos = bool(positive(f))
        fr["unit"].append(unit_key(f))
        fr["has_gt"].append(bool(f["gt"]))
        fr["positive"].append(pos)
        fr["max_score"].append(float(d[:, 4].max()) if len(d) else 0.0)
        fr["n_gt"].append(len(f["gt"]))
        if not pos:
            continue
        rs = ref_scale(f)
        for b in f["gt"]:
            b = np.asarray(b, dtype=np.float64)
            w, h = b[2] - b[0], b[3] - b[1]
            g["unit"].append(unit_key(f))
            g["frame_idx"].append(i)
            g["size"].append(math.sqrt(max(w, 0) * max(h, 0)) * rs)
            g["gw"].append(w * rs)
            g["gh"].append(h * rs)
            if len(d) == 0:
                for c in CRITERIA:
                    g[c].append(0.0)
                for k in ("max_iou_all", "loose_score", "loose_iou"):
                    g[k].append(0.0)
                for k in ("loose_rw", "loose_rh", "loose_dx", "loose_dy"):
                    g[k].append(np.nan)
                continue
            iou = iou_matrix(d[:, :4], b[None])[:, 0]
            cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
            inside = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
            s = d[:, 4]
            g["iou50"].append(float(s[iou >= 0.5].max()) if (iou >= 0.5).any() else 0.0)
            g["iou30"].append(float(s[iou >= 0.3].max()) if (iou >= 0.3).any() else 0.0)
            g["iou10"].append(float(s[iou >= 0.1].max()) if (iou >= 0.1).any() else 0.0)
            g["center"].append(float(s[inside].max()) if inside.any() else 0.0)
            g["max_iou_all"].append(float(iou.max()))
            loose = iou >= 0.1
            if loose.any():
                j = int(np.argmax(np.where(loose, s, -1)))
                g["loose_score"].append(float(s[j]))
                g["loose_iou"].append(float(iou[j]))
                g["loose_rw"].append(float((d[j, 2] - d[j, 0]) / max(w, 1e-9)))
                g["loose_rh"].append(float((d[j, 3] - d[j, 1]) / max(h, 1e-9)))
                g["loose_dx"].append(float((cx[j] - (b[0] + b[2]) / 2) * rs))
                g["loose_dy"].append(float((cy[j] - (b[1] + b[3]) / 2) * rs))
            else:
                g["loose_score"].append(0.0)
                g["loose_iou"].append(0.0)
                for k in ("loose_rw", "loose_rh", "loose_dx", "loose_dy"):
                    g[k].append(np.nan)
    gt = {k: np.asarray(v) for k, v in g.items()}
    ft = {k: np.asarray(v) for k, v in fr.items()}
    return gt, ft


def scale_boxes(preds, k: float):
    """Box-size calibration (addendum 1): every box is scaled by k about its centre."""
    if k == 1.0:
        return preds
    out = []
    for p in preds:
        d = _det(p).copy()
        cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
        hw, hh = (d[:, 2] - d[:, 0]) * k / 2, (d[:, 3] - d[:, 1]) * k / 2
        d[:, 0], d[:, 1], d[:, 2], d[:, 3] = cx - hw, cy - hh, cx + hw, cy + hh
        out.append(d.tolist())
    return out


def recall(gt, thr, crit="iou50", mask=None):
    s = gt[crit] if mask is None else gt[crit][mask]
    return float(((s >= thr) & (s > 0)).mean()) if len(s) else float("nan")   # 0 = no detection at all


def far(ft, thr, mask):
    """Share of the masked frames with at least one detection >= thr."""
    s = ft["max_score"][mask]
    return float(((s >= thr) & (s > 0)).mean()) if len(s) else float("nan")


def threshold_for_far(ft, mask, target):
    """Lowest score threshold whose frame alarm rate on the masked frames is <= target."""
    s = np.sort(ft["max_score"][mask])[::-1]
    n = len(s)
    k = int(math.floor(target * n + 1e-9))       # at most k frames may alarm
    if k >= n:
        return 0.0
    # frames alarm when max_score >= thr; allow the k highest-scoring frames
    thr = np.nextafter(s[k], np.inf) if k < n else 0.0
    # ties: frames with the same score as s[k] must all stay below the threshold
    return float(thr)


def size_bin(sizes):
    return np.digitize(sizes, SIZE_EDGES[1:-1])


def bootstrap_units(units, n, seed):
    """Yields index arrays resampling whole units (videos / sequences)."""
    uniq, inv = np.unique(units, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick]) if len(pick) else np.zeros(0, int)


def ci(samples):
    a = np.asarray([s for s in samples if s == s], dtype=float)
    if len(a) == 0:
        return None
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
