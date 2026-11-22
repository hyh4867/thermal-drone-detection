"""Stage 3 evaluation of one configuration on a drone-only frame set (CST, Anti-UAV).

Adds to the stage 2 DetEval: recall of the tiny and small bins with sequence-level intervals and paired
differences, and the read-out at the 10% alarm point of the SAME frames (a point of the curve, not a
usable threshold; pre-registration 3, section 4).
"""
from __future__ import annotations

import numpy as np

from ..stage2 import tables as T
from ..stage2.deteval import DetEval

BINS = {"tiny": (0, 8), "small": (8, 16)}
KEYS = [f"recall_{b}_{c}" for b in BINS for c in ("iou50", "center")]


class Ev3(DetEval):
    def __init__(self, frames, preds, thr):
        super().__init__(frames, preds, thr, 1.0)
        g = self.gt
        ix = {s: i for i, s in enumerate(self.seqs)}
        gi = np.asarray([ix[u] for u in g["unit"]], dtype=int)
        self.ix = ix
        self.bin = {}
        for b, (lo, hi) in BINS.items():
            m = (g["size"] >= lo) & (g["size"] < hi)
            n = np.zeros(len(self.seqs))
            np.add.at(n, gi[m], 1)
            self.bin[f"n_{b}"] = n
            for c in ("iou50", "center"):
                h = np.zeros(len(self.seqs))
                np.add.at(h, gi[m], ((g[c] >= thr) & (g[c] > 0))[m])
                self.bin[f"hit_{b}_{c}"] = h

    def metrics3(self) -> dict:
        out = self.metrics()
        g, ft = self.gt, self.ft
        t10 = T.threshold_for_far(ft, self.empty, 0.10)
        r10 = {"threshold": t10, "far_drone_free": T.far(ft, t10, self.empty),
               "recall_iou50": T.recall(g, t10), "recall_center": T.recall(g, t10, "center")}
        for b, (lo, hi) in BINS.items():
            m = (g["size"] >= lo) & (g["size"] < hi)
            r10[f"recall_{b}_iou50"] = T.recall(g, t10, "iou50", m)
            r10[f"recall_{b}_center"] = T.recall(g, t10, "center", m)
        out["at_far10_same_frames"] = r10
        loose = g["loose_score"] >= self.thr
        out["box_ratio_pred_over_label"] = {"n": int(loose.sum()),
                                            "width_median": float(np.nanmedian(g["loose_rw"][loose])) if loose.any() else None,
                                            "height_median": float(np.nanmedian(g["loose_rh"][loose])) if loose.any() else None}
        return out

    def _stat3(self, seqs):
        s = self._stat(seqs)
        i = np.asarray([self.ix[q] for q in seqs], dtype=int)
        for b in BINS:
            n = self.bin[f"n_{b}"][i].sum()
            for c in ("iou50", "center"):
                s[f"recall_{b}_{c}"] = self.bin[f"hit_{b}_{c}"][i].sum() / n if n else np.nan
        return s


def bootstrap3(ev: Ev3, ref: "Ev3 | None" = None, B=1000, seed=0) -> dict:
    assert ref is None or ev.seqs == ref.seqs, "paired comparison needs the same sequences"
    rng = np.random.default_rng(seed)
    seqs = np.asarray(ev.seqs)
    reps, diff = {}, {}
    for _ in range(B):
        pick = seqs[rng.integers(0, len(seqs), len(seqs))]
        a = ev._stat3(pick)
        for k, v in a.items():
            reps.setdefault(k, []).append(v)
        if ref is not None:
            b = ref._stat3(pick)
            for k in a:
                diff.setdefault(k, []).append(a[k] - b[k])
    out = {"ci95": {k: T.ci(v) for k, v in reps.items()}, "B": B, "seed": seed, "unit": "sequence", "n_units": len(seqs)}
    if ref is not None:
        pa, pb = ev._stat3(seqs), ref._stat3(seqs)
        out["paired_diff"] = {k: {"diff": float(pa[k] - pb[k]), "ci95": T.ci(diff[k])} for k in pa}
    return out


def paired(ev: Ev3, ref: Ev3, B=1000, seed=0) -> dict:
    return bootstrap3(ev, ref, B, seed)["paired_diff"]
