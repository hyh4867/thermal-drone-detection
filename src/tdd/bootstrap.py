"""Video-level percentile bootstrap for the headline metrics.

Frames of one video are strongly correlated, so the resampling unit is the video
(Anti-UAV: sequence; Svanström: video), not the frame. Matching inside a frame does
not depend on other frames, so per-frame match results are computed once and the
bootstrap only re-aggregates them.
"""
from __future__ import annotations

import collections

import numpy as np

from .metrics import match_frame


def _ap_from(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tps = tps[order]
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(~tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class _Unit:
    """Per-video aggregates: AP inputs plus counts at the operating threshold."""

    def __init__(self):
        self.scores, self.tps = [], []
        self.n_gt = self.tp = self.fp = 0
        self.empty = self.empty_alarm = 0
        self.frames = self.alarm = 0
        self.obj = self.obj_hit = 0

    def finish(self):
        self.scores = np.concatenate(self.scores) if self.scores else np.zeros(0)
        self.tps = np.concatenate(self.tps) if self.tps else np.zeros(0, bool)
        return self


def _units(frames, preds, key, thr, iou_thr):
    units = collections.defaultdict(_Unit)
    for f, p in zip(frames, preds):
        u = units[key(f)]
        d = np.asarray(p, dtype=np.float64).reshape(-1, 5)
        tp_all, _, order = match_frame(f["gt"], d, iou_thr)
        u.scores.append(d[order, 4])
        u.tps.append(tp_all)
        u.n_gt += len(f["gt"])
        dt = d[d[:, 4] >= thr]
        tp_thr, _, _ = match_frame(f["gt"], dt, iou_thr)
        u.tp += int(tp_thr.sum())
        u.fp += int((~tp_thr).sum())
        u.frames += 1
        u.alarm += int(len(dt) > 0)
        if not f["gt"]:
            u.empty += 1
            u.empty_alarm += int(len(dt) > 0)
        else:
            u.obj += 1
            u.obj_hit += int(tp_thr.any())
    return [u.finish() for u in units.values()]


def _ci(samples):
    a = np.asarray([s for s in samples if s == s])
    if len(a) == 0:
        return None
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def _resample(units, stats, n, rng):
    """stats: name -> fn(list[_Unit]) -> float. Returns name -> [lo, hi]."""
    out = {k: [] for k in stats}
    m = len(units)
    if m == 0:
        return {k: None for k in stats}
    for _ in range(n):
        pick = [units[i] for i in rng.integers(0, m, m)]
        for k, fn in stats.items():
            out[k].append(fn(pick))
    return {k: _ci(v) for k, v in out.items()}


def _ratio(num, den):
    def f(us):
        d = sum(den(u) for u in us)
        return sum(num(u) for u in us) / d if d else float("nan")
    return f


def _ap(us):
    return _ap_from(np.concatenate([u.scores for u in us]), np.concatenate([u.tps for u in us]),
                    sum(u.n_gt for u in us))


def bootstrap_antiuav(test, test_p, thr, iou_thr, n, seed):
    units = _units(test, test_p, lambda f: f["seq"], thr, iou_thr)
    rng = np.random.default_rng(seed)
    res = _resample(units, {
        "ap50": _ap,
        "precision": _ratio(lambda u: u.tp, lambda u: u.tp + u.fp),
        "recall": _ratio(lambda u: u.tp, lambda u: u.n_gt),
        "false_alarm_rate_drone_free_frames": _ratio(lambda u: u.empty_alarm, lambda u: u.empty),
    }, n, rng)
    return {"unit": "sequence", "n_units": len(units),
            "n_units_with_drone_free_frames": sum(1 for u in units if u.empty), "B": n, "seed": seed,
            "ci95": res}


def bootstrap_svanstrom(frames, preds, thr, iou_thr, n, seed):
    rng = np.random.default_rng(seed)
    out = {"unit": "video", "B": n, "seed": seed, "drone_recall_by_bin": {}, "non_drone": {}}
    for b in ["CLOSE", "MEDIUM", "DISTANT", "ALL"]:
        sel = [(f, p) for f, p in zip(frames, preds) if f["cls"] == "DRONE" and (b == "ALL" or f["bin"] == b)]
        units = _units([f for f, _ in sel], [p for _, p in sel], lambda f: f["video"], thr, iou_thr)
        out["drone_recall_by_bin"][b] = {
            "n_videos": len(units),
            "ci95": _resample(units, {"recall": _ratio(lambda u: u.tp, lambda u: u.n_gt)}, n, rng)["recall"],
        }
    for cls in ["BIRD", "AIRPLANE", "HELICOPTER"]:
        sel = [(f, p) for f, p in zip(frames, preds) if f["cls"] == cls]
        units = _units([f for f, _ in sel], [p for _, p in sel], lambda f: f["video"], thr, iou_thr)
        r = _resample(units, {
            "frame_alarm_rate": _ratio(lambda u: u.alarm, lambda u: u.frames),
            "object_misidentified_rate": _ratio(lambda u: u.obj_hit, lambda u: u.obj),
        }, n, rng)
        out["non_drone"][cls] = {"n_videos": len(units), "ci95": r}
    return out
