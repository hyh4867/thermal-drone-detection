"""Stage 2 final evaluation with the frozen configuration (pre-registration section 7).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.final_eval [--part svanstrom|antiuav|antiuav_track|cst|all]
  -> results/stage2/final__<part>.json

Reads results/stage2/frozen_config.json (dev-only choices) and prediction caches made
by tdd.stage2.infer. A model whose cache is missing is reported as "not run".
Reference everywhere: S1 = stage 1 model at its stage 1 settings (threshold 0.954,
x2 input on Svanström, no box calibration, no tracker).
"""
from __future__ import annotations

import argparse

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from . import tables as T
from .data import frame_set, load_preds
from .deteval import DetEval, bootstrap as det_bootstrap
from .evalset import SvanEval, bootstrap as sv_bootstrap
from .track_eval import run_tracker, sequence_metrics
from .tracker import TrackerParams

OUT = ROOT / "results" / "stage2"


def _try(tag, set_name):
    try:
        return load_preds(tag, set_name)
    except (FileNotFoundError, RuntimeError) as e:
        return f"prediction cache missing or incomplete: {tag} on {set_name} ({type(e).__name__})"


def svanstrom(fz, set_name="svan_test"):
    frames = frame_set(set_name)
    thr0 = fz["stage1_threshold"]
    s1_raw = load_preds("s1", set_name)
    ref = SvanEval(frames, s1_raw, 1.0)                       # S1 exactly as in stage 1
    out = {"set": set_name, "frames": len(frames), "videos": len({f["video"] for f in frames}),
           "reference_S1": {"detector": {**ref.metrics(thr0), **sv_bootstrap(ref, thr0)},
                            "sequence": sequence_metrics(frames, s1_raw, None, thr0)},
           "models": {}}
    # the stage 1 detections with box-size calibration only (same input, same threshold)
    cal = SvanEval(frames, s1_raw, fz["stage1_x2"]["k"])
    out["S1_box_calibrated"] = {"k": fz["stage1_x2"]["k"], "detector": {**cal.metrics(thr0), **sv_bootstrap(cal, thr0, ref, thr0)},
                                "sequence": sequence_metrics(frames, T.scale_boxes(s1_raw, fz["stage1_x2"]["k"]), None, thr0)}
    thr0_10 = fz["stage1_x2"]["thresholds"]["T10"]            # the same detections at their own dev-matched 10% threshold
    out["S1_box_calibrated_T10"] = {"k": fz["stage1_x2"]["k"], "threshold": thr0_10,
                                    "detector": {**cal.metrics(thr0_10), **sv_bootstrap(cal, thr0_10)},
                                    "sequence": sequence_metrics(frames, T.scale_boxes(s1_raw, fz["stage1_x2"]["k"]), None, thr0_10)}
    evs = {}
    for name, m in fz["models"].items():
        preds = _try(m["svanstrom_tag"], set_name)
        if isinstance(preds, str):
            out["models"][name] = {"status": "not run", "detail": preds}
            continue
        k = m["k"]
        ev = SvanEval(frames, preds, k)
        row = {"status": "ok", "tag": m["svanstrom_tag"], "k": k, "detector": {}, "tracker": {}}
        tp = TrackerParams(**m["tracker"]["params"])
        outs, ids = run_tracker(frames, preds, tp, k)
        tev = SvanEval(frames, outs, 1.0)
        for point in ("T_base", "T10"):
            thr = m["thresholds"][point]
            d = {**ev.metrics(thr), **sv_bootstrap(ev, thr, ref, thr0),
                 "sequence": sequence_metrics(frames, T.scale_boxes(preds, k), None, thr)}
            ts = m["tracker"]["track_score_thresholds"][point]
            t = {**tev.metrics(ts), **sv_bootstrap(tev, ts, ref, thr0), "sequence": sequence_metrics(frames, outs, ids, ts),
                 "paired_vs_same_model_detector": sv_bootstrap(tev, ts, ev, thr)["paired_diff"]}
            row["detector"][point] = d
            row["tracker"][point] = t
        # addendum 1 item 5: every configuration also against the box-calibrated stage 1 model
        row["paired_vs_box_calibrated_S1"] = {
            "detector": sv_bootstrap(ev, m["thresholds"]["T_base"], cal, thr0)["paired_diff"],
            "tracker": sv_bootstrap(tev, m["tracker"]["track_score_thresholds"]["T_base"], cal, thr0)["paired_diff"]}
        row["paired_vs_box_calibrated_S1_T10"] = {
            "detector": sv_bootstrap(ev, m["thresholds"]["T10"], cal, thr0_10)["paired_diff"],
            "tracker": sv_bootstrap(tev, m["tracker"]["track_score_thresholds"]["T10"], cal, thr0_10)["paired_diff"]}
        evs[name] = (ev, tev)
        if "A" in evs and name != "A":                           # trained model minus the no-training configuration
            a, (ev_a, tev_a) = fz["models"]["A"], evs["A"]
            row["paired_vs_A"] = {
                point: {"detector": sv_bootstrap(ev, m["thresholds"][point], ev_a, a["thresholds"][point])["paired_diff"],
                        "tracker": sv_bootstrap(tev, m["tracker"]["track_score_thresholds"][point], tev_a,
                                                a["tracker"]["track_score_thresholds"][point])["paired_diff"]}
                for point in ("T_base", "T10")}
        out["models"][name] = row
    return out


def _native(fz, set_name, with_tracker=True, only=None):
    frames = frame_set(set_name)
    thr0 = fz["stage1_threshold"]
    s1p = _try("s1" if set_name.startswith("auv_test") else "s1_n", set_name) if set_name != "auv_track" else _try("s1_n", set_name)
    out = {"set": set_name, "frames": len(frames), "models": {}}
    ref = None
    if not isinstance(s1p, str):
        ref = DetEval(frames, s1p, thr0)
        out["reference_S1"] = {"detector": {**ref.metrics(), **det_bootstrap(ref)},
                               "sequence": sequence_metrics(frames, s1p, None, thr0, positive=lambda f: True)}
    else:
        out["reference_S1"] = {"status": "not run", "detail": s1p}
    for name, m in fz["models"].items():
        if only and name not in only:
            continue
        tag = m["native_tag"]
        if set_name == "auv_track" and tag == "s1":
            tag = "s1_n"
        if set_name == "cst_test" and tag == "s1":
            tag = "s1_n"
        preds = _try(tag, set_name)
        if isinstance(preds, str):
            out["models"][name] = {"status": "not run", "detail": preds}
            continue
        thr = m["antiuav_val"]["threshold"]
        ev = DetEval(frames, preds, thr)
        row = {"status": "ok", "tag": tag, "detector": {**ev.metrics(), **(det_bootstrap(ev, ref) if ref else det_bootstrap(ev)),
                                                         "sequence": sequence_metrics(frames, preds, None, thr, positive=lambda f: True)}}
        if with_tracker:
            tp = TrackerParams(**m["tracker"]["params"])
            outs, ids = run_tracker(frames, preds, tp, 1.0)
            ts = m["tracker"]["track_score_thresholds"]["T_base"]
            tev = DetEval(frames, outs, ts)
            row["tracker"] = {**tev.metrics(), **det_bootstrap(tev, ev),
                              "note": "paired_diff = tracker output minus the same model's detector output",
                              "sequence": sequence_metrics(frames, outs, ids, ts, positive=lambda f: True)}
        out["models"][name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--config", default="frozen_config")
    ap.add_argument("--svan-set", default="svan_test", help="only for the pipeline smoke test on dev data")
    ap.add_argument("--out-dir", default=None, help="only for the pipeline smoke test (default results/stage2)")
    a = ap.parse_args()
    cfg = load_config("configs/full.yaml")
    fz = read_json(OUT / f"{a.config}.json")
    out_dir = (ROOT / a.out_dir) if a.out_dir else OUT
    assert a.svan_set == "svan_test" or a.out_dir, "a non-test set may only be written outside results/stage2"
    parts = ["svanstrom", "antiuav", "antiuav_track", "cst"] if a.part == "all" else [a.part]
    for part in parts:
        if part == "svanstrom":
            res = svanstrom(fz, a.svan_set)
        elif part == "antiuav":
            res = _native(fz, "auv_test", with_tracker=False)
            supp = {}
            for name, m in fz["models"].items():       # size-bin recall incl. the small-target supplement
                tag = m["native_tag"]
                p1, p2 = _try(tag, "auv_test"), _try(tag, "auv_test_small")
                if isinstance(p1, str) or isinstance(p2, str):
                    continue
                f1, f2 = frame_set("auv_test"), frame_set("auv_test_small")
                seen = {f["image"] for f in f1}
                extra = [(f, p) for f, p in zip(f2, p2) if f["image"] not in seen]
                ev = DetEval(f1 + [x[0] for x in extra], p1 + [x[1] for x in extra], m["antiuav_val"]["threshold"])
                supp[name] = ev.metrics()["recall_by_size"]
            res["recall_by_size_with_supplement"] = supp
        elif part == "antiuav_track":
            res = _native(fz, "auv_track")
        else:
            res = _native(fz, "cst_test")
        res["frozen_config"] = f"results/stage2/{a.config}.json"
        res["meta"] = run_metadata(cfg, {"stage": 2})
        write_json(res, out_dir / f"final__{part}.json")
        print("==", part)
        r = res.get("reference_S1", {}).get("detector")
        if r:
            print("  S1 ref", {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
        for name, row in res["models"].items():
            if row["status"] != "ok":
                print(" ", name, row["status"])
                continue
            if part == "svanstrom":
                d, t = row["detector"]["T_base"], row["tracker"]["T_base"]
                print(f"  {name}: det R50 {d['recall_iou50']:.3f} Rc {d['recall_center']:.3f} FAR {d['far']:.3f} "
                      f"dR50 {d['paired_diff']['recall_iou50']['diff']:+.3f} {d['paired_diff']['recall_iou50']['ci95']} | "
                      f"trk R50 {t['recall_iou50']:.3f} Rc {t['recall_center']:.3f} FAR {t['far']:.3f} "
                      f"false tracks/min {t['sequence']['false_confirmed_tracks_per_minute']:.1f}")
            else:
                d = row["detector"]
                print(f"  {name}: AP {d['ap50']:.3f} R50 {d['recall_iou50']:.3f} Rc {d['recall_center']:.3f} P {d['precision']:.3f} FAR {d['far_drone_free']:.3f}"
                      + (f" | trk R50 {row['tracker']['recall_iou50']:.3f} Rc {row['tracker']['recall_center']:.3f} FAR {row['tracker']['far_drone_free']:.3f}" if "tracker" in row else ""))


if __name__ == "__main__":
    main()
