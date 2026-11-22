"""Stage 2 / Q2 pre-check: how badly does a constant-velocity model fit when the camera moves?

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.motion  -> results/stage2/motion.json

Uses only LABELS (target centre tracks) and, for the global-shift estimate, the dev
images. No detector output. All distances in 640x512 reference pixels at the tracker
frame rate (Svanström: every 5th frame of 30 fps = 6 Hz; Anti-UAV val: every 3rd
frame of 20 fps = 6.7 Hz, labels read from the raw per-sequence JSON).

Per step t -> t+1:
  static residual   |c[t+1] - c[t]|                  (zero-velocity prediction)
  CV residual       |c[t+1] - (2 c[t] - c[t-1])|     (constant-velocity prediction)
  global shift g    translation between the two frames by phase correlation
  compensated       the same two residuals after subtracting g from the target motion
Pre-registered rules: use the CV model if its median residual is below the static
one, else a random walk; add global-shift compensation only if it lowers the median
CV residual by at least 20%.
"""
from __future__ import annotations

import collections
import json

import cv2
import numpy as np

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from ..evaluate import load_image
from .data import MAN, frame_set, ref_scale

SVAN_DT = 5 / 30.0
AUV_STRIDE, AUV_FPS = 3, 20.0


def global_shift(a: np.ndarray, b: np.ndarray):
    """Translation (dx, dy) of image content from a to b, and the correlation response."""
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(fa, fb, win)
    return float(dx), float(dy), float(resp)


def _stats(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99))}


def residuals(centres, shifts=None):
    """centres: list of (x, y) or None per consecutive frame. Returns static / cv residual
    vectors (as arrays of 2-vectors), optionally with the global shift removed."""
    stat, cv_ = [], []
    for t in range(1, len(centres) - 1):
        if centres[t - 1] is None or centres[t] is None or centres[t + 1] is None:
            continue
        c0, c1, c2 = (np.asarray(centres[t + i]) for i in (-1, 0, 1))
        d01, d12 = c1 - c0, c2 - c1
        if shifts is not None:
            d01 = d01 - np.asarray(shifts[t][:2])
            d12 = d12 - np.asarray(shifts[t + 1][:2])
        stat.append(d12)
        cv_.append(d12 - d01)
    return np.asarray(stat).reshape(-1, 2), np.asarray(cv_).reshape(-1, 2)


def svanstrom_dev(with_images=True):
    by_vid = collections.defaultdict(list)
    for f in frame_set("svan_dev"):
        by_vid[f["video"]].append(f)
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    shift_mag, resp = [], []
    for vid, frs in by_vid.items():
        frs.sort(key=lambda f: f["frame"])
        rs = ref_scale(frs[0])
        centres = [((f["gt"][0][0] + f["gt"][0][2]) / 2 * rs, (f["gt"][0][1] + f["gt"][0][3]) / 2 * rs) if f["gt"] else None
                   for f in frs]
        shifts = None
        if with_images:
            shifts = [(0.0, 0.0, 0.0)]
            prev = load_image(frs[0]["image"])
            for f in frs[1:]:
                cur = load_image(f["image"])
                dx, dy, r = global_shift(prev, cur)
                shifts.append((dx * rs, dy * rs, r))
                shift_mag.append(np.hypot(dx, dy) * rs)
                resp.append(r)
                prev = cur
        grp = "drone" if frs[0]["cls"] == "DRONE" else "non_drone_object"
        s, c = residuals(centres)
        res[grp]["static"].extend(np.linalg.norm(s, axis=1))
        res[grp]["cv"].extend(np.linalg.norm(c, axis=1))
        res[grp]["cv_component"].extend(c.ravel())
        res[grp]["static_component"].extend(s.ravel())
        if shifts is not None:
            s2, c2 = residuals(centres, shifts)
            res[grp]["static_compensated"].extend(np.linalg.norm(s2, axis=1))
            res[grp]["cv_compensated"].extend(np.linalg.norm(c2, axis=1))
    out = {g: {k: _stats(v) for k, v in d.items() if not k.endswith("_component")} for g, d in res.items()}
    for g, d in res.items():
        out[g]["cv_component_std"] = float(np.std(d["cv_component"]))
        out[g]["static_component_std"] = float(np.std(d["static_component"]))
        out[g]["cv_component_abs_quantiles"] = {str(q): float(np.quantile(np.abs(d["cv_component"]), q)) for q in (0.5, 0.75, 0.9)}
    out["global_shift_px_per_step"] = _stats(shift_mag)
    out["phase_correlation_response"] = _stats(resp)
    return out


def antiuav_val():
    seqs = sorted({f["seq"] for f in read_json(MAN / "antiuav_val.json")})
    raw = ROOT / "data" / "raw" / "antiuav300"
    stat, cv_, comp = [], [], []
    for s in seqs:
        cand = [p for p in raw.glob(f"*/{s}/infrared.json")]
        if not cand:
            continue
        d = json.loads(cand[0].read_text())
        centres = []
        for i in range(0, len(d["exist"]), AUV_STRIDE):
            r = d["gt_rect"][i]
            ok = d["exist"][i] and r and len(r) == 4 and r[2] > 0 and r[3] > 0
            centres.append((r[0] + r[2] / 2, r[1] + r[3] / 2) if ok else None)
        a, b = residuals(centres)
        stat.extend(np.linalg.norm(a, axis=1))
        cv_.extend(np.linalg.norm(b, axis=1))
        comp.extend(b.ravel())
    return {"sequences": len(seqs), "dt_s": AUV_STRIDE / AUV_FPS, "static": _stats(stat), "cv": _stats(cv_),
            "cv_component_std": float(np.std(comp))}


def main():
    cfg = load_config("configs/full.yaml")
    sv = svanstrom_dev()
    au = antiuav_val()
    d = sv["drone"]
    use_cv = d["cv"]["median"] < d["static"]["median"]
    gain = 1 - d["cv_compensated"]["median"] / d["cv"]["median"]
    # three process-noise levels from the CV residual of the drone tracks (per-axis):
    # one-step prediction variance of an NCV model ~ q dt^3 / 3  ->  q = 3 r^2 / dt^3
    qs = {k: 3 * v ** 2 / SVAN_DT ** 3 for k, v in d["cv_component_abs_quantiles"].items()}
    out = {"svanstrom_dev": sv, "antiuav_val": au, "dt_svanstrom_s": SVAN_DT,
           "decision": {"model": "cv" if use_cv else "rw",
                        "rule_model": "cv if median CV residual < median static residual (dev drone tracks)",
                        "global_shift_compensation": bool(gain >= 0.2),
                        "cv_residual_reduction_by_compensation": gain,
                        "q_levels_px2_per_s3": qs,
                        "q_rule": "q = 3 r^2 / dt^3 with r = 50th / 75th / 90th percentile of |CV residual| per axis"}}
    out["meta"] = run_metadata(cfg, {"stage": 2})
    write_json(out, ROOT / "results" / "stage2" / "motion.json")
    print(json.dumps({"svan_drone": {k: v for k, v in d.items()}, "auv_val": au, "decision": out["decision"],
                      "shift": sv["global_shift_px_per_step"], "resp": sv["phase_correlation_response"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()
