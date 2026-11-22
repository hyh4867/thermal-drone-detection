"""Stage 2 evaluation of one configuration on a Svanström frame set.

A configuration = predictions (any model / input setting / tracker output), a score
threshold and a box-size calibration factor k. Recall is reported under IoU >= 0.5
(the registered primary criterion) and under the centre criterion (addendum 1).
"""
from __future__ import annotations

import numpy as np

from . import tables as T

CLASSES = ("BIRD", "AIRPLANE", "HELICOPTER")
BINS = ("CLOSE", "MEDIUM", "DISTANT")


class SvanEval:
    """Tables for one (frames, predictions, k)."""

    def __init__(self, frames, preds, k: float = 1.0):
        self.frames = frames
        preds = T.scale_boxes(preds, k)
        self.k = k
        vid = lambda f: f["video"]
        self.gt, self.ft = T.build(frames, preds, vid, positive=lambda f: f["cls"] == "DRONE")
        self.obj, _ = T.build(frames, preds, vid, positive=lambda f: f["cls"] != "DRONE")
        self.cls = np.asarray([f["cls"] for f in frames])
        self.neg = self.cls != "DRONE"
        self.gt_bin = np.asarray([frames[i]["bin"] for i in self.gt["frame_idx"]])
        self.obj_cls = np.asarray([frames[i]["cls"] for i in self.obj["frame_idx"]])

    def far(self, thr):
        return T.far(self.ft, thr, self.neg)

    def threshold_for_far(self, target):
        return T.threshold_for_far(self.ft, self.neg, target)

    def metrics(self, thr) -> dict:
        g = self.gt
        out = {"threshold": float(thr), "k": self.k, "drone_gt": int(len(g["size"])),
               "non_drone_frames": int(self.neg.sum()),
               "recall_iou50": T.recall(g, thr), "recall_center": T.recall(g, thr, "center"),
               "far": self.far(thr),
               "far_by_class": {c: T.far(self.ft, thr, self.cls == c) for c in CLASSES},
               "recall_by_distance": {b: {"gt": int((self.gt_bin == b).sum()),
                                          "iou50": T.recall(g, thr, "iou50", self.gt_bin == b),
                                          "center": T.recall(g, thr, "center", self.gt_bin == b)} for b in BINS},
               "object_flagged": {c: {"frames_with_label": int((self.obj_cls == c).sum()),
                                      "iou50": T.recall(self.obj, thr, "iou50", self.obj_cls == c),
                                      "center": T.recall(self.obj, thr, "center", self.obj_cls == c)} for c in CLASSES}}
        return out

    def _video_stats(self, thr):
        """Per-video sufficient statistics for the bootstrap."""
        vids = np.unique(self.ft["unit"])
        ix = {v: i for i, v in enumerate(vids)}
        s = {k: np.zeros(len(vids)) for k in ("n_gt", "hit50", "hitc", "n_neg", "alarm")}
        gi = np.asarray([ix[u] for u in self.gt["unit"]], dtype=int)
        np.add.at(s["n_gt"], gi, 1)
        np.add.at(s["hit50"], gi, self.gt["iou50"] >= thr)
        np.add.at(s["hitc"], gi, self.gt["center"] >= thr)
        fi = np.asarray([ix[u] for u in self.ft["unit"]], dtype=int)
        np.add.at(s["n_neg"], fi[self.neg], 1)
        np.add.at(s["alarm"], fi[self.neg], self.ft["max_score"][self.neg] >= thr)
        return vids, s


def _ratios(s, pick):
    def r(a, b):
        d = s[b][pick].sum()
        return s[a][pick].sum() / d if d else np.nan
    return r("hit50", "n_gt"), r("hitc", "n_gt"), r("alarm", "n_neg")


def bootstrap(ev: SvanEval, thr, other: "SvanEval | None" = None, other_thr=None, B=1000, seed=0) -> dict:
    """Video-level percentile intervals; with ``other`` also the paired difference
    (ev - other) on the same resampled videos."""
    vids, s = ev._video_stats(thr)
    if other is not None:
        vo, so = other._video_stats(other_thr)
        assert list(vids) == list(vo), "paired comparison needs the same videos"
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in ("recall_iou50", "recall_center", "far")}
    diff = {k: [] for k in reps}
    for _ in range(B):
        pick = rng.integers(0, len(vids), len(vids))
        a = _ratios(s, pick)
        for k, v in zip(reps, a):
            reps[k].append(v)
        if other is not None:
            b = _ratios(so, pick)
            for k, va, vb in zip(diff, a, b):
                diff[k].append(va - vb)
    out = {"ci95": {k: T.ci(v) for k, v in reps.items()}, "B": B, "seed": seed, "unit": "video", "n_videos": len(vids)}
    if other is not None:
        pa, pb = _ratios(s, np.arange(len(vids))), _ratios(so, np.arange(len(vids)))
        out["paired_diff"] = {k: {"diff": float(x - y), "ci95": T.ci(diff[k])} for k, x, y in zip(reps, pa, pb)}
    return out


def select_k(frames, preds, thr, ks=None):
    """Addendum 1 rule: k maximising IoU>=0.5 recall at ``thr`` (ties -> smaller k)."""
    ks = ks or [round(1.0 + 0.05 * i, 2) for i in range(13)]
    best = None
    rows = []
    for k in ks:
        g, _ = T.build(frames, T.scale_boxes(preds, k), lambda f: f["video"], positive=lambda f: f["cls"] == "DRONE")
        r = T.recall(g, thr)
        rows.append({"k": k, "recall_iou50": r})
        if best is None or r > best[1] + 1e-12:
            best = (k, r)
    return best[0], rows
