"""Stage 2 / Q2: run the tracker on cached detections and score it.

Usage (tuning on dev, pre-registered grid):
  PYTHONPATH=src .venv/bin/python -m tdd.stage2.track_eval tune --tag s1 --k 1.35
    -> results/stage2/tracker_tuning__<tag>__<set>.json  (chosen parameters + the whole grid)

Outputs of the tracker are confirmed-track boxes with the running mean detection
score of the track; the frame-alarm rate is matched with a threshold on that score.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import time

import numpy as np

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from ..metrics import iou_matrix
from . import tables as T
from .data import frame_set, load_preds, ref_scale
from .diagnose import s1_threshold
from .evalset import SvanEval
from .tracker import TrackerParams, track_video

FPS = {"svanstrom": 30.0, "antiuav": 20.0, "cst": 25.0}


def unit_of(f):
    return f.get("video") or f["seq"]


def fps_of(f):
    img = f["image"]
    return FPS["svanstrom"] if img.startswith("svanstrom/") else FPS["cst"] if "cst_antiuav" in img else FPS["antiuav"]


def by_video(frames):
    groups = collections.defaultdict(list)
    for i, f in enumerate(frames):
        groups[unit_of(f)].append(i)
    for v in groups.values():
        v.sort(key=lambda i: frames[i]["frame"])
    return groups


def run_tracker(frames, preds, params: TrackerParams, k: float = 1.0):
    """Tracks every video. Returns (outputs, ids): outputs[i] = [[x1,y1,x2,y2,track_score], ...]
    in native pixels (boxes already k-calibrated), ids[i] = matching track ids."""
    preds = T.scale_boxes(preds, k)
    outs, ids = [None] * len(frames), [None] * len(frames)
    for vid, idx in by_video(frames).items():
        rs = ref_scale(frames[idx[0]])
        fps = fps_of(frames[idx[0]])
        times = [frames[i]["frame"] / fps for i in idx]
        dets = []
        for i in idx:
            d = np.asarray(preds[i], dtype=float).reshape(-1, 5).copy()
            d[:, :4] *= rs
            dets.append(d)
        for i, recs in zip(idx, track_video(times, dets, params)):
            outs[i] = [[r[0] / rs, r[1] / rs, r[2] / rs, r[3] / rs, r[4]] for r in recs]
            ids[i] = [r[5] for r in recs]
    return outs, ids


def sequence_metrics(frames, outs, ids, thr, positive=lambda f: f.get("cls", "DRONE") == "DRONE"):
    """Track-level metrics at output-score threshold ``thr``.

    ids=None scores a plain detector (no track identities)."""
    lat = {"iou50": [], "center": []}
    never = {"iou50": 0, "center": 0}
    kept_n = {"iou50": 0, "center": 0}
    kept_d = {"iou50": 0, "center": 0}
    n_ids = []
    drone_videos = 0
    off_target_ids, pos_minutes = 0, 0.0
    false_ids, neg_minutes, neg_alarm_frames = 0, 0.0, 0
    for vid, idx in by_video(frames).items():
        f0 = frames[idx[0]]
        fps = fps_of(f0)
        dt = (frames[idx[1]]["frame"] - f0["frame"]) / fps if len(idx) > 1 else 0.0
        if not positive(f0):
            neg_minutes += len(idx) * dt / 60.0
            seen = set()
            for i in idx:
                o = [(b, j) for j, b in enumerate(outs[i]) if b[4] >= thr]
                neg_alarm_frames += bool(o)
                if ids is not None:
                    seen.update(ids[i][j] for _, j in o)
            false_ids += len(seen)
            continue
        gt_frames = [i for i in idx if frames[i]["gt"]]
        if not gt_frames:
            continue
        drone_videos += 1
        hit = {"iou50": [], "center": []}
        tp_ids = set()
        pos_minutes += len(idx) * dt / 60.0
        if ids is not None:
            all_ids = {ids[i][j] for i in idx for j, b in enumerate(outs[i]) if b[4] >= thr}
        for i in gt_frames:
            g = np.asarray(frames[i]["gt"][0], dtype=float)
            o = np.asarray([b for b in outs[i] if b[4] >= thr], dtype=float).reshape(-1, 5)
            if len(o):
                iou = iou_matrix(o[:, :4], g[None])[:, 0]
                cx, cy = (o[:, 0] + o[:, 2]) / 2, (o[:, 1] + o[:, 3]) / 2
                inside = (cx >= g[0]) & (cx <= g[2]) & (cy >= g[1]) & (cy <= g[3])
                hit["iou50"].append(bool((iou >= 0.5).any()))
                hit["center"].append(bool(inside.any()))
                if ids is not None and inside.any():
                    kept = [j for j, b in enumerate(outs[i]) if b[4] >= thr]
                    tp_ids.update(ids[i][kept[j]] for j in np.flatnonzero(inside))
            else:
                hit["iou50"].append(False)
                hit["center"].append(False)
        t0 = frames[gt_frames[0]]["frame"] / fps
        for c in ("iou50", "center"):
            h = np.asarray(hit[c])
            if not h.any():
                never[c] += 1
                continue
            first = int(np.argmax(h))
            lat[c].append(frames[gt_frames[first]]["frame"] / fps - t0)
            kept_n[c] += int(h[first + 1:].sum())
            kept_d[c] += int(len(h) - first - 1)
        if ids is not None:
            off_target_ids += len(all_ids - tp_ids)
            if tp_ids:
                n_ids.append(len(tp_ids))
    out = {"drone_videos": drone_videos, "non_drone_minutes": neg_minutes,
           "non_drone_alarm_frames_per_minute": neg_alarm_frames / neg_minutes if neg_minutes else None}
    for c in ("iou50", "center"):
        out[c] = {"never_found_videos": never[c], "never_found_share": never[c] / drone_videos if drone_videos else None,
                  "first_found_latency_s_median": float(np.median(lat[c])) if lat[c] else None,
                  "first_found_latency_s_p90": float(np.percentile(lat[c], 90)) if lat[c] else None,
                  "maintenance_after_first": kept_n[c] / kept_d[c] if kept_d[c] else None}
    if ids is not None:
        out["false_confirmed_tracks_per_minute"] = false_ids / neg_minutes if neg_minutes else None
        out["false_confirmed_tracks"] = false_ids
        out["track_ids_per_found_drone_video_mean"] = float(np.mean(n_ids)) if n_ids else None
        out["drone_video_minutes"] = pos_minutes
        out["tracks_never_on_target_in_drone_videos_per_minute"] = off_target_ids / pos_minutes if pos_minutes else None
    return out


GRID = {"det_threshold": [0.05, 0.1, 0.2, 0.3, 0.5], "mn": [(2, 3), (3, 4), (3, 5), (4, 6)],
        "delete_misses": [2, 3, 5], "coast_outputs": [0, 1, 2]}


def tune(tag, k, set_name="svan_dev"):
    cfg = load_config("configs/full.yaml")
    motion = read_json(ROOT / "results" / "stage2" / "motion.json")["decision"]
    q_levels = sorted(motion["q_levels_px2_per_s3"].values())
    frames, preds = frame_set(set_name), load_preds(tag, set_name)
    thr0 = s1_threshold()
    t_base = SvanEval(frame_set(set_name), load_preds("s1", set_name)).far(thr0)   # stage 1 FAR on the same dev frames
    det = SvanEval(frames, preds, k)
    ref = {}
    for name, target in (("T_base", t_base), ("T10", 0.10)):
        t = det.threshold_for_far(target)
        ref[name] = {"threshold": t, "recall_iou50": T.recall(det.gt, t), "recall_center": T.recall(det.gt, t, "center"),
                     "far": det.far(t), "sequence": sequence_metrics(frames, T.scale_boxes(preds, k), None, t)}
    rows = []
    t0 = time.time()
    combos = list(itertools.product(GRID["det_threshold"], GRID["mn"], GRID["delete_misses"], q_levels, GRID["coast_outputs"]))
    for n, (dth, (m, nn), kk, q, co) in enumerate(combos):
        p = TrackerParams(model=motion["model"], q=q, confirm_m=m, confirm_n=nn, delete_misses=kk,
                          det_threshold=dth, coast_outputs=co)
        outs, ids = run_tracker(frames, preds, p, k)
        ev = SvanEval(frames, outs, 1.0)
        row = {"det_threshold": dth, "confirm_m": m, "confirm_n": nn, "delete_misses": kk, "q": q, "coast_outputs": co}
        for name, target in (("T_base", t_base), ("T10", 0.10)):
            t = ev.threshold_for_far(target)
            sm = sequence_metrics(frames, outs, ids, t)
            row[name] = {"track_score_threshold": t, "recall_iou50": T.recall(ev.gt, t), "recall_center": T.recall(ev.gt, t, "center"),
                         "far": ev.far(t), "false_tracks_per_min": sm["false_confirmed_tracks_per_minute"],
                         "never_found_share_center": sm["center"]["never_found_share"],
                         "latency_median_center": sm["center"]["first_found_latency_s_median"],
                         "maintenance_center": sm["center"]["maintenance_after_first"]}
        rows.append(row)
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(combos)} combos, {(time.time() - t0) / 60:.1f} min", flush=True)
    best = max(rows, key=lambda r: (round(r["T_base"]["recall_iou50"], 4), -r["T_base"]["false_tracks_per_min"]))
    out = {"tag": tag, "set": set_name, "k": k, "T_base": t_base, "motion_model": motion["model"],
           "grid": {**GRID, "q": q_levels}, "detector_only": ref, "selected": best,
           "selection_rule": "highest dev iso-FAR(T_base) IoU>=0.5 recall of confirmed-track outputs; ties -> fewer false tracks per minute",
           "rows": rows, "meta": run_metadata(cfg, {"stage": 2})}
    write_json(out, ROOT / "results" / "stage2" / f"tracker_tuning__{tag}__{set_name}.json")
    b = best
    print("detector only", {n: (round(r["recall_iou50"], 3), round(r["recall_center"], 3), round(r["far"], 3)) for n, r in ref.items()})
    print("selected", {k_: v for k_, v in b.items() if k_ not in ("T_base", "T10")})
    for n in ("T_base", "T10"):
        print(n, {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in b[n].items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune"])
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--set", default="svan_dev")
    a = ap.parse_args()
    tune(a.tag, a.k, a.set)


if __name__ == "__main__":
    main()
