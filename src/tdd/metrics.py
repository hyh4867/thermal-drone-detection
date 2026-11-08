"""Detection metrics for single-class (drone) evaluation.

Conventions
-----------
* Boxes are ``[x1, y1, x2, y2]`` in pixels (float).
* A frame record is ``{"gt": [[x1,y1,x2,y2], ...], "det": [[x1,y1,x2,y2,score], ...]}``.
* A detection is a true positive if it is matched to a not-yet-matched GT box in
  the same frame with IoU >= ``iou_thr``. Detections are matched in descending
  score order (VOC / COCO style greedy matching).
* AP is the all-point interpolated area under the precision/recall curve
  (PASCAL VOC 2010+ definition).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N,4) and ``b`` (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)[:, None, :4]
    b = np.asarray(b, dtype=np.float64)[None, :, :4]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_frame(gt: Sequence, det: Sequence, iou_thr: float = 0.5):
    """Greedy score-ordered matching inside one frame.

    Returns ``(det_tp, gt_matched)``: boolean arrays aligned with ``det`` sorted
    by descending score (the sort order is returned as the third element) and
    with ``gt`` respectively.
    """
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    det = np.asarray(det, dtype=np.float64).reshape(-1, 5)
    order = np.argsort(-det[:, 4], kind="stable")
    det = det[order]
    ious = iou_matrix(det[:, :4], gt)
    gt_matched = np.zeros(len(gt), dtype=bool)
    det_tp = np.zeros(len(det), dtype=bool)
    for i in range(len(det)):
        if len(gt) == 0:
            break
        cand = np.where(~gt_matched, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            gt_matched[j] = True
            det_tp[i] = True
    return det_tp, gt_matched, order


def average_precision(frames: Iterable[dict], iou_thr: float = 0.5) -> dict:
    """AP over all frames plus the full PR curve (for threshold selection)."""
    scores, tps = [], []
    n_gt = 0
    for fr in frames:
        det = np.asarray(fr["det"], dtype=np.float64).reshape(-1, 5)
        n_gt += len(fr["gt"])
        det_tp, _, order = match_frame(fr["gt"], det, iou_thr)
        scores.append(det[order, 4])
        tps.append(det_tp)
    if n_gt == 0:
        raise ValueError("AP undefined: no ground-truth boxes")
    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros(0, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    scores, tps = scores[order], tps[order]
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(~tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    # all-point interpolation
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "n_gt": int(n_gt), "scores": scores, "precision": precision, "recall": recall}


def best_f1_threshold(ap_result: dict) -> dict:
    """Score threshold that maximises F1 on a PR curve (used on *val* only)."""
    p, r, s = ap_result["precision"], ap_result["recall"], ap_result["scores"]
    if len(s) == 0:
        return {"threshold": math.inf, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    # Only thresholds at the last detection of a run of equal scores are valid
    # operating points; take the max-F1 among them.
    valid = np.ones(len(s), dtype=bool)
    valid[:-1] = s[:-1] != s[1:]
    i = int(np.argmax(np.where(valid, f1, -1)))
    return {"threshold": float(s[i]), "f1": float(f1[i]), "precision": float(p[i]), "recall": float(r[i])}


def size_of(box) -> float:
    """Target size = sqrt(w*h) in pixels (the Anti-UAV paper's scale definition)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return math.sqrt(max(w, 0.0) * max(h, 0.0))


def operating_point(frames: Sequence[dict], thr: float, iou_thr: float = 0.5,
                    size_bins: Sequence[tuple] | None = None) -> dict:
    """Precision / recall / false-alarm statistics at a fixed score threshold.

    ``size_bins`` is a list of ``(name, lo, hi)`` with lo <= size < hi.
    """
    tp = fp = n_gt = 0
    empty_frames = empty_with_alarm = empty_fp = 0
    bins = {name: [0, 0] for name, _, _ in (size_bins or [])}  # [matched, total]
    for fr in frames:
        det = np.asarray(fr["det"], dtype=np.float64).reshape(-1, 5)
        det = det[det[:, 4] >= thr]
        det_tp, gt_matched, _ = match_frame(fr["gt"], det, iou_thr)
        tp += int(det_tp.sum())
        fp += int((~det_tp).sum())
        n_gt += len(fr["gt"])
        if len(fr["gt"]) == 0:
            empty_frames += 1
            empty_fp += len(det)
            empty_with_alarm += int(len(det) > 0)
        for g, m in zip(fr["gt"], gt_matched):
            s = size_of(g)
            for name, lo, hi in size_bins or []:
                if lo <= s < hi:
                    bins[name][0] += int(m)
                    bins[name][1] += 1
    out = {
        "threshold": thr,
        "frames": len(frames),
        "gt_boxes": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": n_gt - tp,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / n_gt if n_gt else float("nan"),
        "empty_frames": empty_frames,
        "empty_frames_with_alarm": empty_with_alarm,
        "false_alarm_rate_empty_frames": empty_with_alarm / empty_frames if empty_frames else float("nan"),
        "fp_per_empty_frame": empty_fp / empty_frames if empty_frames else float("nan"),
    }
    if size_bins:
        out["recall_by_size"] = {
            name: {"matched": m, "total": t, "recall": (m / t if t else float("nan"))}
            for name, (m, t) in bins.items()
        }
    return out
