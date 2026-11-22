"""Detector evaluation on a drone-only dataset (Anti-UAV test, CST test) for stage 2.

One labelled drone at most per frame; frames flagged ``invalid`` are left out.
Reports AP@0.5, precision / recall at a fixed threshold (IoU >= 0.5 and centre
criterion), the alarm rate on drone-free frames, recall by size, sequence-level
bootstrap intervals and, against a reference, paired differences on the same frames.
"""
from __future__ import annotations

import numpy as np

from ..bootstrap import _ap, _units
from ..metrics import average_precision, operating_point
from . import tables as T

SIZE_BINS = [("tiny", 0, 8), ("small", 8, 16), ("medium", 16, 32), ("large", 32, 64), ("xlarge", 64, 100000)]


def _valid(frames, preds):
    keep = [i for i, f in enumerate(frames) if not f.get("invalid")]
    return [frames[i] for i in keep], [preds[i] for i in keep]


class DetEval:
    def __init__(self, frames, preds, thr, k: float = 1.0):
        frames, preds = _valid(frames, preds)
        preds = T.scale_boxes(preds, k)
        self.frames, self.preds, self.thr, self.k = frames, preds, thr, k
        self.gt, self.ft = T.build(frames, preds, lambda f: f["seq"])
        self.empty = ~self.ft["has_gt"]
        self.seqs = sorted({f["seq"] for f in frames})
        # stage 1 style per-sequence units (AP inputs, tp/fp at thr, empty-frame alarms)
        by = {}
        for f, p in zip(frames, preds):
            by.setdefault(f["seq"], ([], []))
            by[f["seq"]][0].append(f)
            by[f["seq"]][1].append(p)
        self.units = {s: _units(fs, ps, lambda f: f["seq"], thr, 0.5)[0] for s, (fs, ps) in by.items()}
        self.centre = {s: int(((self.gt["center"] >= thr) & (self.gt["center"] > 0) & (self.gt["unit"] == s)).sum()) for s in self.seqs}

    def metrics(self) -> dict:
        fr = [{"gt": f["gt"], "det": p} for f, p in zip(self.frames, self.preds)]
        op = operating_point(fr, self.thr, 0.5, SIZE_BINS)
        g = self.gt
        b_sz = {n: (g["size"] >= lo) & (g["size"] < hi) for n, lo, hi in SIZE_BINS}
        return {"threshold": float(self.thr), "k": self.k, "frames": len(fr), "sequences": len(self.seqs),
                "gt_boxes": op["gt_boxes"], "ap50": average_precision(fr, 0.5)["ap"],
                "precision": op["precision"], "recall_iou50": op["recall"], "tp": op["tp"], "fp": op["fp"],
                "recall_center": T.recall(g, self.thr, "center"),
                "drone_free_frames": op["empty_frames"], "drone_free_frames_with_alarm": op["empty_frames_with_alarm"],
                "far_drone_free": op["false_alarm_rate_empty_frames"],
                "recall_by_size": {n: {"total": int(m.sum()), "iou50": T.recall(g, self.thr, "iou50", m),
                                       "center": T.recall(g, self.thr, "center", m)} for n, m in b_sz.items()}}

    def _stat(self, seqs):
        us = [self.units[s] for s in seqs]
        n_gt = sum(u.n_gt for u in us)
        tp, fp = sum(u.tp for u in us), sum(u.fp for u in us)
        e, ea = sum(u.empty for u in us), sum(u.empty_alarm for u in us)
        return {"ap50": _ap(us), "recall_iou50": tp / n_gt if n_gt else np.nan,
                "recall_center": sum(self.centre[s] for s in seqs) / n_gt if n_gt else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan, "far_drone_free": ea / e if e else np.nan}


def bootstrap(ev: DetEval, ref: "DetEval | None" = None, B=1000, seed=0) -> dict:
    assert ref is None or ev.seqs == ref.seqs
    rng = np.random.default_rng(seed)
    seqs = np.asarray(ev.seqs)
    reps, diff = {}, {}
    for _ in range(B):
        pick = seqs[rng.integers(0, len(seqs), len(seqs))]
        a = ev._stat(pick)
        for k, v in a.items():
            reps.setdefault(k, []).append(v)
        if ref is not None:
            b = ref._stat(pick)
            for k in a:
                diff.setdefault(k, []).append(a[k] - b[k])
    out = {"ci95": {k: T.ci(v) for k, v in reps.items()}, "B": B, "seed": seed, "unit": "sequence", "n_units": len(seqs)}
    if ref is not None:
        pa, pb = ev._stat(seqs), ref._stat(seqs)
        out["paired_diff"] = {k: {"diff": float(pa[k] - pb[k]), "ci95": T.ci(diff[k])} for k in pa}
    return out
