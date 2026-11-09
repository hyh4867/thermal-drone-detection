"""One evaluation protocol for every method (baseline and fine-tuned model).

1. Anti-UAV val  -> AP@0.5 and the operating score threshold (max F1 on val).
2. Anti-UAV test -> AP@0.5, precision, recall, false-alarm rate on drone-free
   frames (all 1,153 of them), recall by target size (main sample + the
   small-target supplement).
3. Svanström IR (never trained or tuned on) -> drone recall per distance bin and
   the share of bird / airplane / helicopter frames flagged as drone, at the
   same val-chosen threshold.

A ``predict_fn(frames, scale=1.0) -> list[list[[x1,y1,x2,y2,score]]]`` returns
boxes in the frame's native pixel coordinates.
"""
from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, path, read_json, run_metadata, write_json
from .bootstrap import bootstrap_antiuav, bootstrap_svanstrom
from .metrics import average_precision, best_f1_threshold, match_frame, operating_point, size_of

_FRAMES = ROOT / "data" / "frames"


def load_image(rel: str) -> np.ndarray:
    img = cv2.imread(str(_FRAMES / rel), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(rel)
    return img


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval. NOTE: frames of one video are correlated, so the
    effective sample size is smaller and the true uncertainty is wider."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _frames_with_preds(frames, preds):
    return [{"gt": f["gt"], "det": p} for f, p in zip(frames, preds)]


def ap_on(frames, preds, iou_thr=0.5) -> float:
    return average_precision(_frames_with_preds(frames, preds), iou_thr)["ap"]


STREAM_CHUNK = 64


def _empty_cache() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _predict_cached(predict_fn, frames, cache: Path, scale=1.0, reuse=True, full_frames=None):
    """Predictions are streamed to ``<cache>.partial.jsonl`` one frame per line as
    they are made (nothing accumulates in memory), so a killed run resumes from the
    last written frame; the finished file is then written as the JSON cache."""
    if reuse and cache.exists():
        preds = read_json(cache)
        if len(preds) == len(frames):
            return preds, 0.0
        if full_frames is not None:
            sub = _cached_subset(cache, full_frames, frames)
            if sub is not None:
                return sub, 0.0
        del preds
    partial = cache.with_suffix(".partial.jsonl")
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if reuse and partial.exists():
        with open(partial) as f:
            done = sum(1 for _ in f)
    t = time.time()
    with open(partial, "a" if done else "w") as f:
        for i in range(done, len(frames), STREAM_CHUNK):
            for p in predict_fn(frames[i:i + STREAM_CHUNK], scale=scale):
                f.write(json.dumps(p) + "\n")
            f.flush()
    dt = time.time() - t
    _empty_cache()
    with open(partial) as f:
        preds = [json.loads(line) for line in f]
    if len(preds) != len(frames):
        raise RuntimeError(f"{partial}: {len(preds)} predictions for {len(frames)} frames")
    write_json(preds, cache)
    partial.unlink()
    return preds, dt


def evaluate_antiuav(cfg, val, val_p, test, test_p, supp, supp_p):
    iou = cfg["eval"]["iou_thr"]
    bins = [tuple(b) for b in cfg["eval"]["size_bins"]]
    val_ap = average_precision(_frames_with_preds(val, val_p), iou)
    op_val = best_f1_threshold(val_ap)
    thr = op_val["threshold"]
    test_frames = _frames_with_preds(test, test_p)
    test_ap = average_precision(test_frames, iou)
    op = operating_point(test_frames, thr, iou)
    # size-bin recall over main sample + small-target supplement
    size_op = operating_point(test_frames + _frames_with_preds(supp, supp_p), thr, iou, bins)
    for b in size_op["recall_by_size"].values():
        b["ci95"] = wilson(b["matched"], b["total"])
    # per-sequence recall spread (frames within a sequence are correlated)
    by_seq = collections.defaultdict(list)
    for f, p in zip(test, test_p):
        by_seq[f["seq"]].append({"gt": f["gt"], "det": p})
    seq_recall = []
    for s, frs in by_seq.items():
        o = operating_point(frs, thr, iou)
        if o["gt_boxes"]:
            seq_recall.append(o["recall"])
    return {
        "val": {"frames": len(val), "ap50": val_ap["ap"], "n_gt": val_ap["n_gt"], "selected_threshold": op_val},
        "test": {
            "frames": len(test),
            "ap50": test_ap["ap"],
            "precision": op["precision"], "precision_ci95": wilson(op["tp"], op["tp"] + op["fp"]),
            "recall": op["recall"], "recall_ci95": wilson(op["tp"], op["gt_boxes"]),
            "tp": op["tp"], "fp": op["fp"], "fn": op["fn"], "gt_boxes": op["gt_boxes"],
            "drone_free_frames": op["empty_frames"],
            "drone_free_frames_with_alarm": op["empty_frames_with_alarm"],
            "false_alarm_rate_drone_free_frames": op["false_alarm_rate_empty_frames"],
            "false_alarm_rate_ci95": wilson(op["empty_frames_with_alarm"], op["empty_frames"]),
            "fp_per_drone_free_frame": op["fp_per_empty_frame"],
            "per_sequence_recall": {
                "n_sequences": len(seq_recall),
                "median": float(np.median(seq_recall)) if seq_recall else None,
                "p10": float(np.percentile(seq_recall, 10)) if seq_recall else None,
                "p90": float(np.percentile(seq_recall, 90)) if seq_recall else None,
            },
            "recall_by_size": size_op["recall_by_size"],
            "recall_by_size_frames": size_op["frames"],
        },
        "threshold": thr,
    }


def evaluate_svanstrom(cfg, frames, preds, thr):
    iou = cfg["eval"]["iou_thr"]
    out = {"drone_recall_by_bin": {}, "non_drone_flagged_as_drone": {}}
    groups = collections.defaultdict(list)
    for f, p in zip(frames, preds):
        groups[(f["cls"], f["bin"])].append((f, p))
    order = ["CLOSE", "MEDIUM", "DISTANT"]
    # drone recall per distance bin (+ all bins)
    for b in order + ["ALL"]:
        items = [x for (c, bb), v in groups.items() if c == "DRONE" and (b == "ALL" or bb == b) for x in v]
        frs = [{"gt": f["gt"], "det": p} for f, p in items]
        o = operating_point(frs, thr, iou)
        sizes = [f["obj_size"] for f, _ in items if f["obj_size"] is not None]
        out["drone_recall_by_bin"][b] = {
            "videos": len({f["video"] for f, _ in items}),
            "frames": len(items), "drone_boxes": o["gt_boxes"], "matched": o["tp"],
            "recall": o["recall"], "recall_ci95": wilson(o["tp"], o["gt_boxes"]),
            "precision_in_drone_videos": o["precision"],
            "median_size_px_native": float(np.median(sizes)) if sizes else None,
        }
    # non-drone classes: frames where the detector fires (any box >= thr) and
    # frames where the labelled bird/plane/heli itself is boxed as a drone (IoU >= iou_thr)
    for cls in ["BIRD", "AIRPLANE", "HELICOPTER"]:
        res = {}
        for b in order + ["ALL"]:
            items = [x for (c, bb), v in groups.items() if c == cls and (b == "ALL" or bb == b) for x in v]
            n = len(items)
            any_alarm = obj_hit = n_obj = 0
            vids_alarm = set()
            for f, p in items:
                d = np.asarray(p, dtype=np.float64).reshape(-1, 5)
                d = d[d[:, 4] >= thr]
                if len(d):
                    any_alarm += 1
                    vids_alarm.add(f["video"])
                if f["gt"]:
                    n_obj += 1
                    tp, _, _ = match_frame(f["gt"], d, iou)
                    obj_hit += int(tp.any())
            vids = {f["video"] for f, _ in items}
            res[b] = {
                "videos": len(vids), "frames": n,
                "frames_with_drone_alarm": any_alarm,
                "frame_alarm_rate": any_alarm / n if n else None,
                "frame_alarm_rate_ci95": wilson(any_alarm, n),
                "frames_with_object": n_obj,
                "object_boxed_as_drone": obj_hit,
                "object_misidentified_rate": obj_hit / n_obj if n_obj else None,
                "object_misidentified_ci95": wilson(obj_hit, n_obj),
                "videos_with_any_alarm": len(vids_alarm),
            }
        out["non_drone_flagged_as_drone"][cls] = res
    return out


def load_eval_sets(cfg):
    man = path(cfg["data"]["manifests_dir"])
    sub = cfg["eval"].get("subsample", {})
    val = [f for f in read_json(man / "antiuav_val.json") if f["frame"] % sub.get("val_every", 1) == 0]
    test = [f for f in read_json(man / "antiuav_test.json")
            if (f["gt"] and f["frame"] % sub.get("test_every", 1) == 0)
            or (not f["gt"] and f["frame"] % sub.get("test_absent_every", 1) == 0)]
    sv = [f for f in read_json(man / "svanstrom_ir.json") if f["frame"] % sub.get("svanstrom_every", 1) == 0]
    supp = read_json(man / "antiuav_test_small_supplement.json")[:: sub.get("supp_every", 1)]
    return val, test, sv, supp


def _cached_subset(cache: Path, full_frames, subset):
    """Reuse predictions made earlier on the full manifest (same method) for a subset."""
    if not cache.exists():
        return None
    preds = read_json(cache)
    if len(preds) != len(full_frames):
        return None
    idx = {(f["image"]): p for f, p in zip(full_frames, preds)}
    if not all(f["image"] in idx for f in subset):
        return None
    return [idx[f["image"]] for f in subset]


def evaluate_all(cfg, predict_fn, name: str, out_dir: Path, method_info: dict, reuse: bool = True):
    man = path(cfg["data"]["manifests_dir"])
    runs = ROOT / "runs" / name
    val, test, sv, supp = load_eval_sets(cfg)
    timing = {}
    full = {k: read_json(man / f"{k}.json") for k in ("antiuav_val", "antiuav_test", "svanstrom_ir")}
    val_p, timing["val_s"] = _predict_cached(predict_fn, val, runs / "preds_antiuav_val.json", reuse=reuse,
                                             full_frames=full["antiuav_val"])
    test_p, timing["test_s"] = _predict_cached(predict_fn, test, runs / "preds_antiuav_test.json", reuse=reuse,
                                               full_frames=full["antiuav_test"])
    supp_p, timing["supp_s"] = _predict_cached(predict_fn, supp, runs / "preds_antiuav_test_small.json", reuse=reuse,
                                               full_frames=read_json(man / "antiuav_test_small_supplement.json"))
    au = evaluate_antiuav(cfg, val, val_p, test, test_p, supp, supp_p)
    bs = cfg["eval"].get("bootstrap")
    if bs:
        au["test"]["bootstrap"] = bootstrap_antiuav(test, test_p, au["threshold"], cfg["eval"]["iou_thr"],
                                                    bs["n"], bs["seed"])
    meta = run_metadata(cfg, {"method": method_info, "timing_seconds": timing})
    write_json({**au, "meta": meta}, out_dir / "antiuav_metrics.json")
    sv_p, timing["svanstrom_s"] = _predict_cached(
        predict_fn, sv, runs / "preds_svanstrom_ir.json", scale=cfg["eval"]["svanstrom_scale"], reuse=reuse,
        full_frames=full["svanstrom_ir"])
    svr = evaluate_svanstrom(cfg, sv, sv_p, au["threshold"])
    if bs:
        svr["bootstrap"] = bootstrap_svanstrom(sv, sv_p, au["threshold"], cfg["eval"]["iou_thr"], bs["n"], bs["seed"])
    write_json({"threshold_from_antiuav_val": au["threshold"], **svr,
                "meta": run_metadata(cfg, {"method": method_info, "timing_seconds": timing})},
               out_dir / "svanstrom_metrics.json")
    print_summary(au, svr)
    return au, svr


def print_summary(au, svr):
    t = au["test"]
    print(f"val AP50={au['val']['ap50']:.4f} thr={au['threshold']:.4f}")
    print(f"test AP50={t['ap50']:.4f} P={t['precision']:.4f} R={t['recall']:.4f} "
          f"FAR(drone-free)={t['false_alarm_rate_drone_free_frames']:.4f} ({t['drone_free_frames_with_alarm']}/{t['drone_free_frames']})")
    for k, v in t["recall_by_size"].items():
        print(f"  size {k}: {v['matched']}/{v['total']} recall={v['recall']:.4f}")
    for k, v in svr["drone_recall_by_bin"].items():
        print(f"  svanstrom drone {k}: {v['matched']}/{v['drone_boxes']} recall={v['recall']:.4f}")
    for c, r in svr["non_drone_flagged_as_drone"].items():
        a = r["ALL"]
        print(f"  svanstrom {c}: frame alarm {a['frames_with_drone_alarm']}/{a['frames']}={a['frame_alarm_rate']:.4f} "
              f"object-as-drone {a['object_boxed_as_drone']}/{a['frames_with_object']}")
