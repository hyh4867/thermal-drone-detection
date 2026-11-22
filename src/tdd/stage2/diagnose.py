"""Stage 2 / Q1: why does recall drop on Svanström? (no new training, stage 1 model)

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.diagnose [--svan svan_dev] [--out diagnosis]
  -> results/stage2/<out>.json

Data: Anti-UAV test (stage 1 cached predictions; explanatory only, never used for a
choice) and the Svanström dev videos (default) at the stage 1 configuration (x2).
Definitions follow docs/PREREGISTRATION_2.md section 5/Q1.
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from ..metrics import average_precision
from . import tables as T
from .data import frame_set, load_preds, ref_scale

THR_LOW = 0.05
B, SEED = 1000, 0


def s1_threshold() -> float:
    return read_json(ROOT / "results" / "full" / "model" / "antiuav_metrics.json")["threshold"]


def load_auv(tag="s1"):
    """Anti-UAV test main sample, plus the small-target supplement for size-bin recall."""
    main = frame_set("auv_test")
    mp = load_preds(tag, "auv_test")
    seen = {f["image"] for f in main}
    supp_f, supp_p = [], []
    for f, p in zip(frame_set("auv_test_small"), load_preds(tag, "auv_test_small")):
        if f["image"] not in seen:
            seen.add(f["image"])
            supp_f.append(f)
            supp_p.append(p)
    return main, mp, supp_f, supp_p


def merged_bins(n_auv_per_bin, min_n=30):
    """Pre-registered rule: an Anti-UAV size bin with fewer than ``min_n`` boxes is merged
    with the next larger bin. Returns a group index per bin."""
    groups, cur, acc = [], 0, 0
    for i, n in enumerate(n_auv_per_bin):
        groups.append(cur)
        acc += n
        if acc >= min_n and i < len(n_auv_per_bin) - 1:
            cur += 1
            acc = 0
    if acc < min_n and cur > 0:                       # tail too small -> merge down
        groups = [min(g, cur - 1) for g in groups]
    return np.asarray(groups)


def standardized_recall(a_bins, a_hit, s_bins, groups):
    """sum_b w_SV(b) * recall_AUV(b) over merged size groups."""
    ga, gs = groups[a_bins], groups[s_bins]
    tot = 0.0
    for b in np.unique(gs):
        w = (gs == b).mean()
        m = ga == b
        if not m.any():
            return float("nan")
        tot += w * a_hit[m].mean()
    return float(tot)


def decomposition(a_main, a_size, s_gt, thr, groups, ia=None, isz=None, isv=None):
    """Gap and single-factor shares. a_main: AUV main-sample GT table; a_size: main+supplement."""
    am = a_main if ia is None else {k: v[ia] for k, v in a_main.items()}
    az = a_size if isz is None else {k: v[isz] for k, v in a_size.items()}
    sv = s_gt if isv is None else {k: v[isv] for k, v in s_gt.items()}
    ab, sb = T.size_bin(az["size"]), T.size_bin(sv["size"])
    r_a, r_s = T.recall(am, thr), T.recall(sv, thr)
    g = r_a - r_s
    g_t = T.recall(am, THR_LOW) - T.recall(sv, THR_LOW)
    g_l = T.recall(am, thr, "iou10") - T.recall(sv, thr, "iou10")
    r_a_std = standardized_recall(ab, az["iou50"] >= thr, sb, groups)
    # Pre-registered share: R_AUV - R_AUV^std. Check value: the same with R_AUV replaced by
    # the bin recalls re-weighted to the Anti-UAV main sample's own sizes (removes the
    # effect of the small-target supplement on the bin recalls).
    r_a_self = standardized_recall(ab, az["iou50"] >= thr, T.size_bin(am["size"]), groups)
    share_s = r_a - r_a_std
    g_tls = standardized_recall(ab, az["iou10"] >= THR_LOW, sb, groups) - T.recall(sv, THR_LOW, "iou10")
    singles = (g - g_t) + (g - g_l) + share_s
    return {"recall_auv": r_a, "recall_svan": r_s, "gap": g,
            "share_threshold": g - g_t, "share_localisation": g - g_l, "share_size": share_s,
            "unexplained": g_tls, "overlap": singles - (g - g_tls),
            "share_size_check_self_standardised": r_a_self - r_a_std,
            "recall_auv_size_standardised": r_a_std, "recall_auv_self_standardised": r_a_self,
            "gap_low_threshold": g_t, "gap_iou10": g_l}


def by_size(gt, thr):
    out = {}
    b = T.size_bin(gt["size"])
    for i, name in enumerate(T.SIZE_NAMES):
        m = b == i
        n = int(m.sum())
        row = {"n": n}
        for c in T.CRITERIA:
            row[f"recall_{c}"] = float((gt[c][m] >= thr).mean()) if n else None
        row["recall_iou50_low_thr"] = float((gt["iou50"][m] >= THR_LOW).mean()) if n else None
        out[name] = row
    return out


def by_size_ci(gt, thr):
    reps = collections.defaultdict(list)
    b = T.size_bin(gt["size"])
    for idx in T.bootstrap_units(gt["unit"], B, SEED):
        bb, hit = b[idx], gt["iou50"][idx] >= thr
        for i, name in enumerate(T.SIZE_NAMES):
            m = bb == i
            reps[name].append(hit[m].mean() if m.any() else np.nan)
    return {k: T.ci(v) for k, v in reps.items()}


def curve(gt, ft, neg_mask, thresholds):
    return [{"thr": float(t), "far": T.far(ft, t, neg_mask), "recall": T.recall(gt, t),
             "recall_iou10": T.recall(gt, t, "iou10")} for t in thresholds]


def hist(values, edges):
    h, _ = np.histogram(values, bins=edges)
    return {"edges": [float(e) for e in edges], "counts": h.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svan", default="svan_dev")
    ap.add_argument("--out", default="diagnosis")
    a = ap.parse_args()
    cfg = load_config("configs/full.yaml")
    thr = s1_threshold()

    am_f, am_p, as_f, as_p = load_auv()
    sv_f, sv_p = frame_set(a.svan), load_preds("s1", a.svan)
    a_gt, a_ft = T.build(am_f, am_p, lambda f: f["seq"])
    a_gt2, _ = T.build(am_f + as_f, am_p + as_p, lambda f: f["seq"])
    s_gt, s_ft = T.build(sv_f, sv_p, lambda f: f["video"], positive=lambda f: f["cls"] == "DRONE")
    s_cls = np.asarray([f["cls"] for f in sv_f])
    s_neg = s_cls != "DRONE"                      # bird / airplane / helicopter frames
    a_neg = ~a_ft["has_gt"]                       # drone-free frames
    assert a_ft["n_gt"].max() <= 1 and s_ft["n_gt"].max() <= 1, "one object per frame assumed"

    out = {"threshold_stage1": thr, "threshold_low": THR_LOW, "svanstrom_set": a.svan,
           "frames": {"auv_test": len(am_f), "auv_test_supplement_extra": len(as_f), "svanstrom": len(sv_f)},
           "videos": {"svanstrom": len(set(s_ft["unit"])), "svanstrom_drone": len(set(s_gt["unit"])),
                      "auv_sequences": len(set(a_ft["unit"]))}}

    # --- threshold-free and operating-point numbers --------------------------------
    drone_idx = [i for i, f in enumerate(sv_f) if f["cls"] == "DRONE"]
    ap_drone = average_precision([{"gt": sv_f[i]["gt"], "det": sv_p[i]} for i in drone_idx])["ap"]
    ap_pooled = average_precision([{"gt": f["gt"] if f["cls"] == "DRONE" else [], "det": p}
                                   for f, p in zip(sv_f, sv_p)])["ap"]
    ap_auv = average_precision([{"gt": f["gt"], "det": p} for f, p in zip(am_f, am_p)])["ap"]
    t_base = T.far(s_ft, thr, s_neg)
    out["ap50"] = {"auv_test": ap_auv, "svanstrom_drone_videos": ap_drone, "svanstrom_all_frames": ap_pooled}
    out["operating_point_stage1"] = {
        "auv": {"recall": T.recall(a_gt, thr), "gt": int(len(a_gt["size"])), "far_drone_free": T.far(a_ft, thr, a_neg),
                "drone_free_frames": int(a_neg.sum())},
        "svanstrom": {"recall": T.recall(s_gt, thr), "gt": int(len(s_gt["size"])),
                      "far_non_drone_frames (T_base)": t_base, "non_drone_frames": int(s_neg.sum()),
                      "far_by_class": {c: T.far(s_ft, thr, s_cls == c) for c in ("BIRD", "AIRPLANE", "HELICOPTER")}}}
    out["recall_by_criterion"] = {
        name: {f"thr_{t:g}": {c: T.recall(g, t, c) for c in T.CRITERIA} for t in (thr, 0.5, THR_LOW, 0.001)}
        for name, g in (("auv_test", a_gt), ("svanstrom", s_gt))}

    # FAR-matched thresholds (alternative definition of the threshold factor)
    thr_sv_35 = T.threshold_for_far(s_ft, s_neg, T.far(a_ft, thr, a_neg))
    out["far_matched"] = {
        "auv_far_at_stage1_thr": T.far(a_ft, thr, a_neg),
        "svanstrom_thr_for_same_far": thr_sv_35, "svanstrom_far_there": T.far(s_ft, thr_sv_35, s_neg),
        "svanstrom_recall_there": T.recall(s_gt, thr_sv_35),
        "threshold_share_alt": (T.recall(a_gt, thr) - T.recall(s_gt, thr)) - (T.recall(a_gt, thr) - T.recall(s_gt, thr_sv_35)),
        "svanstrom_thr_for_far10": T.threshold_for_far(s_ft, s_neg, 0.10),
        "svanstrom_recall_at_far10": T.recall(s_gt, T.threshold_for_far(s_ft, s_neg, 0.10))}
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.05, 6), np.linspace(0.05, 0.9, 18),
                                     np.linspace(0.9, 0.999, 34), [thr]]))
    out["far_recall_curve"] = {"auv_test": curve(a_gt, a_ft, a_neg, grid), "svanstrom": curve(s_gt, s_ft, s_neg, grid)}

    # --- score distributions of true positives (IoU >= 0.5) --------------------------
    edges = np.concatenate([[0, 0.05], np.linspace(0.1, 0.9, 9), [0.95, 0.97, 0.98, 0.99, 0.995, 1.0001]])
    out["tp_score_hist"] = {n: hist(g["iou50"][g["iou50"] > 0], edges) for n, g in (("auv_test", a_gt), ("svanstrom", s_gt))}
    out["tp_score_quantiles"] = {n: {str(q): float(np.quantile(g["iou50"][g["iou50"] > 0], q)) for q in (0.1, 0.25, 0.5, 0.75)}
                                 for n, g in (("auv_test", a_gt), ("svanstrom", s_gt))}

    # --- localisation: what happened to missed GT ---------------------------------------
    def missed(g):
        m = g["iou50"] < thr                      # missed at the stage 1 operating point
        n = int(m.sum())
        cats = {
            "found_iou50_but_score_below_thr": int((g["iou50"][m] >= THR_LOW).sum()),
            "box_iou_0.3-0.5_score_ge_thr": int(((g["iou30"][m] >= thr)).sum()),
            "box_iou_0.1-0.3_score_ge_thr": int(((g["iou10"][m] >= thr) & (g["iou30"][m] < thr)).sum()),
            "centre_inside_gt_score_ge_thr": int((g["center"][m] >= thr).sum()),
            "nothing_overlapping_at_any_score(iou<0.1)": int((g["max_iou_all"][m] < 0.1).sum()),
        }
        return {"missed": n, "of_gt": int(len(m)), "categories_not_exclusive": cats,
                "max_iou_any_score_hist": hist(g["max_iou_all"][m], np.linspace(0, 1, 11))}
    out["missed_gt"] = {"auv_test": missed(a_gt), "svanstrom": missed(s_gt)}

    def box_convention(g):
        m = (g["loose_score"] >= 0.5) & np.isfinite(g["loose_rw"])
        res = {"pairs": int(m.sum())}
        for k in ("loose_rw", "loose_rh", "loose_iou", "loose_dx", "loose_dy"):
            v = g[k][m]
            res[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        res["note"] = "rw/rh = predicted / labelled box width, height; dx/dy = centre offset in reference px"
        b = T.size_bin(g["size"])
        res["by_size"] = {name: {"pairs": int((m & (b == i)).sum()),
                                 "median_rw": float(np.median(g["loose_rw"][m & (b == i)])) if (m & (b == i)).any() else None,
                                 "median_rh": float(np.median(g["loose_rh"][m & (b == i)])) if (m & (b == i)).any() else None,
                                 "median_iou": float(np.median(g["loose_iou"][m & (b == i)])) if (m & (b == i)).any() else None}
                          for i, name in enumerate(T.SIZE_NAMES)}
        return res
    out["box_convention"] = {"auv_test": box_convention(a_gt), "svanstrom": box_convention(s_gt)}
    out["gt_box_shape"] = {n: {"median_w": float(np.median(g["gw"])), "median_h": float(np.median(g["gh"])),
                               "median_aspect_w_over_h": float(np.median(g["gw"] / np.maximum(g["gh"], 1e-9)))}
                           for n, g in (("auv_test", a_gt), ("svanstrom", s_gt))}

    # --- size -----------------------------------------------------------------------------
    out["size_bins_reference_px"] = T.SIZE_NAMES
    out["recall_by_size"] = {"auv_test_plus_supplement": by_size(a_gt2, thr), "svanstrom": by_size(s_gt, thr)}
    out["recall_by_size_ci95_iou50"] = {"auv_test_plus_supplement": by_size_ci(a_gt2, thr), "svanstrom": by_size_ci(s_gt, thr)}
    out["size_quantiles_reference_px"] = {n: {str(q): float(np.quantile(g["size"], q)) for q in (0.1, 0.5, 0.9)}
                                          for n, g in (("auv_test", a_gt), ("svanstrom", s_gt))}

    # --- decomposition ------------------------------------------------------------------------
    n_auv = np.bincount(T.size_bin(a_gt2["size"]), minlength=len(T.SIZE_NAMES))
    groups = merged_bins(n_auv)
    point = decomposition(a_gt, a_gt2, s_gt, thr, groups)
    reps = collections.defaultdict(list)
    seqs = np.unique(np.concatenate([a_gt["unit"], a_gt2["unit"]]))
    mem_a = {q: np.flatnonzero(a_gt["unit"] == q) for q in seqs}
    mem_z = {q: np.flatnonzero(a_gt2["unit"] == q) for q in seqs}
    rng = np.random.default_rng(SEED)
    it_s = T.bootstrap_units(s_gt["unit"], B, SEED + 1)
    for isv in it_s:
        pick = seqs[rng.integers(0, len(seqs), len(seqs))]       # the same sequences for both tables
        ia = np.concatenate([mem_a[q] for q in pick])
        iz = np.concatenate([mem_z[q] for q in pick])
        for k, v in decomposition(a_gt, a_gt2, s_gt, thr, groups, ia, iz, isv).items():
            reps[k].append(v)
    out["decomposition"] = {"point": point, "ci95": {k: T.ci(v) for k, v in reps.items()},
                            "size_groups_after_merge": groups.tolist(), "auv_boxes_per_size_bin": n_auv.tolist(),
                            "bootstrap": {"B": B, "seed": SEED, "unit": "Anti-UAV sequence / Svanström video"}}

    # --- false alarms on non-drone videos -----------------------------------------------------
    rows = []
    gridc = np.zeros((4, 4), int)
    rel = collections.Counter()
    per_video = collections.Counter()
    for f, p in zip(sv_f, sv_p):
        if f["cls"] == "DRONE":
            continue
        d = np.asarray(p, dtype=np.float64).reshape(-1, 5)
        d = d[d[:, 4] >= thr]
        for x1, y1, x2, y2, s in d:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            gridc[min(3, int(4 * cy / f["height"])), min(3, int(4 * cx / f["width"]))] += 1
            edge = min(cx, cy, f["width"] - cx, f["height"] - cy) < 0.05 * f["width"]
            if f["gt"]:
                g = f["gt"][0]
                gw, gh = g[2] - g[0], g[3] - g[1]
                gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                on = abs(cx - gx) <= 0.75 * gw and abs(cy - gy) <= 0.75 * gh   # inside the box grown 1.5x
                from ..metrics import iou_matrix
                iou = float(iou_matrix(np.asarray([[x1, y1, x2, y2]]), np.asarray([g]))[0, 0])
                kind = "on_object_iou>=0.5" if iou >= 0.5 else "on_object_iou<0.5" if on else "elsewhere_object_labelled"
            else:
                kind = "frame_without_object_label"
            rel[kind] += 1
            per_video[f["video"]] += 1
            rows.append((f["cls"], kind, float(np.sqrt((x2 - x1) * (y2 - y1)) * ref_scale(f)), float(s), bool(edge)))
    n = len(rows)
    sizes = np.asarray([r[2] for r in rows])
    top = per_video.most_common(10)
    out["false_alarms_non_drone"] = {
        "boxes_at_stage1_thr": n, "relation_to_labelled_object": dict(rel),
        "share_on_labelled_object_1.5x": (rel["on_object_iou>=0.5"] + rel["on_object_iou<0.5"]) / n if n else None,
        "by_class": {c: dict(collections.Counter(r[1] for r in rows if r[0] == c)) for c in ("BIRD", "AIRPLANE", "HELICOPTER")},
        "position_grid_4x4_rows_top_to_bottom": gridc.tolist(),
        "near_image_edge_share": float(np.mean([r[4] for r in rows])) if n else None,
        "size_hist_reference_px": hist(sizes, T.SIZE_EDGES[:-1] + [1e4]) if n else None,
        "size_median_reference_px": float(np.median(sizes)) if n else None,
        "score_median": float(np.median([r[3] for r in rows])) if n else None,
        "videos_with_alarm": len(per_video), "non_drone_videos": len({f["video"] for f in sv_f if f["cls"] != "DRONE"}),
        "top10_videos_share_of_boxes": sum(c for _, c in top) / n if n else None,
        "top10_videos": top}
    # --- addendum 1: distance bins, box-size calibration, object confusion ------------------
    s_bin = np.asarray([sv_f[i]["bin"] for i in s_gt["frame_idx"]])
    out["svanstrom_recall_by_distance"] = {
        b: {"gt": int((s_bin == b).sum()) if b != "ALL" else int(len(s_bin)),
            **{c: T.recall(s_gt, thr, c, None if b == "ALL" else s_bin == b) for c in T.CRITERIA},
            "iou50_low_thr": T.recall(s_gt, THR_LOW, "iou50", None if b == "ALL" else s_bin == b),
            "center_low_thr": T.recall(s_gt, THR_LOW, "center", None if b == "ALL" else s_bin == b),
            "median_label_size_ref_px": float(np.median(s_gt["size"][s_bin == b])) if b != "ALL" else float(np.median(s_gt["size"]))}
        for b in ("CLOSE", "MEDIUM", "DISTANT", "ALL")}
    ks = [round(1.0 + 0.05 * i, 2) for i in range(13)]
    sweep = []
    for k in ks:
        g_k, _ = T.build(sv_f, T.scale_boxes(sv_p, k), lambda f: f["video"], positive=lambda f: f["cls"] == "DRONE")
        sweep.append({"k": k, "recall_iou50": T.recall(g_k, thr), "recall_iou50_low_thr": T.recall(g_k, THR_LOW),
                      "by_distance": {b: T.recall(g_k, thr, "iou50", s_bin == b) for b in ("CLOSE", "MEDIUM", "DISTANT")}})
    best = max(sweep, key=lambda r: (r["recall_iou50"], -r["k"]))
    out["box_calibration"] = {"rule": "k maximising IoU>=0.5 recall at the stage 1 threshold on dev (ties -> smaller k)",
                              "sweep": sweep, "selected_k": best["k"], "recall_iou50_at_selected": best["recall_iou50"]}
    kb = best["k"]
    s_corr = dict(s_gt)
    s_corr["size"] = s_gt["size"] / kb
    out["recall_by_size_svanstrom_corrected_size"] = {"k": kb, "table": by_size(s_corr, thr), "ci95_iou50": by_size_ci(s_corr, thr)}
    o_gt, _ = T.build(sv_f, sv_p, lambda f: f["video"], positive=lambda f: f["cls"] != "DRONE")
    o_cls = np.asarray([sv_f[i]["cls"] for i in o_gt["frame_idx"]])
    out["object_flagged_as_drone"] = {
        c: {"frames_with_object_label": int((o_cls == c).sum()),
            "iou50": T.recall(o_gt, thr, "iou50", o_cls == c), "center": T.recall(o_gt, thr, "center", o_cls == c),
            "iou10": T.recall(o_gt, thr, "iou10", o_cls == c)} for c in ("BIRD", "AIRPLANE", "HELICOPTER")}
    out["meta"] = run_metadata(cfg, {"stage": 2, "model": "runs/full_frcnn/best.pt (stage 1, cached predictions)"})
    write_json(out, ROOT / "results" / "stage2" / f"{a.out}.json")

    d = out["decomposition"]["point"]
    print(f"AP50 svan drone {ap_drone:.3f} pooled {ap_pooled:.3f} | auv {ap_auv:.3f}")
    print(f"recall auv {d['recall_auv']:.3f} svan {d['recall_svan']:.3f} gap {d['gap']:.3f}  T_base {t_base:.3f}")
    print({k: round(v, 3) for k, v in d.items()})
    print("ci", out["decomposition"]["ci95"])
    print("by distance", {b: {k: round(v, 3) for k, v in r.items()} for b, r in out["svanstrom_recall_by_distance"].items()})
    print("box k", kb, [(r["k"], round(r["recall_iou50"], 3)) for r in sweep])
    print("object flagged", out["object_flagged_as_drone"])
    print("corrected size", {k: (v["n"], v["recall_iou50"] and round(v["recall_iou50"], 2), v["recall_center"] and round(v["recall_center"], 2)) for k, v in out["recall_by_size_svanstrom_corrected_size"]["table"].items()})
    print("on-object share", out["false_alarms_non_drone"]["share_on_labelled_object_1.5x"], dict(rel))


if __name__ == "__main__":
    main()
