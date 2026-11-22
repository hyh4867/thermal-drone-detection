"""Stage 2 / Q1 input experiment: input scale and tiling on the dev videos (every 10th frame).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.inputs_eval  -> results/stage2/inputs.json
Selection metric (pre-registered): IoU>=0.5 recall at the threshold that matches the
stage 1 frame alarm rate (T_base) on the same dev frames, boxes calibrated with the
configuration's own k (addendum 1). Missing caches are reported as not run / failed.
"""
from __future__ import annotations

import json

from ..common import ROOT, load_config, run_metadata, write_json
from .data import cache_path, frame_set, load_preds
from .diagnose import s1_threshold
from .evalset import SvanEval, bootstrap, select_k

SET = "svan_dev10"
CONFIGS = [("s1_x1", "x1 whole image"), ("s1", "x2 whole image (stage 1)"), ("s1_x3", "x3 whole image"),
           ("s1_x4", "x4 whole image"), ("s1_t3", "x3, 2x2 tiles"), ("s1_t4", "x4, 2x2 tiles")]


def main():
    cfg = load_config("configs/full.yaml")
    thr0 = s1_threshold()
    frames = frame_set(SET)
    base = SvanEval(frames, load_preds("s1", SET), 1.0)
    t_base = base.far(thr0)
    out = {"set": SET, "frames": len(frames), "stage1_threshold": thr0, "T_base": t_base, "configs": {}}
    evs = {}
    for tag, label in CONFIGS:
        try:
            preds = load_preds(tag, SET)
        except (FileNotFoundError, RuntimeError) as e:
            out["configs"][tag] = {"label": label, "status": "not run / failed", "detail": str(e)[:200]}
            continue
        meta_p = cache_path(tag, SET).with_suffix(".meta.json")
        k, sweep = select_k(frames, preds, thr0)
        ev = SvanEval(frames, preds, k)
        evs[tag] = ev
        row = {"label": label, "status": "ok", "k_selected": k, "k_sweep": sweep,
               "s_per_frame": json.loads(meta_p.read_text())["s_per_frame"] if meta_p.exists() else None,
               "at_stage1_threshold": ev.metrics(thr0)}
        for name, target in (("iso_far_T_base", t_base), ("iso_far_10pct", 0.10)):
            t = ev.threshold_for_far(target)
            row[name] = ev.metrics(t)
        out["configs"][tag] = row
    ref = evs["s1"]
    t_ref = ref.threshold_for_far(t_base)
    for tag, ev in evs.items():
        if tag == "s1":
            continue
        t = ev.threshold_for_far(t_base)
        out["configs"][tag]["paired_vs_x2_iso_far"] = bootstrap(ev, t, ref, t_ref)["paired_diff"]
    ok = {t: r for t, r in out["configs"].items() if r["status"] == "ok"}
    best = max(ok, key=lambda t: ok[t]["iso_far_T_base"]["recall_iou50"])
    # pre-registered tie rule: within 1 point -> the cheaper setting
    cands = [t for t in ok if ok[best]["iso_far_T_base"]["recall_iou50"] - ok[t]["iso_far_T_base"]["recall_iou50"] <= 0.01]
    cost = {"s1_x1": 0, "s1": 1, "s1_x3": 2, "s1_t3": 3, "s1_x4": 4, "s1_t4": 5}
    chosen = min(cands, key=lambda t: cost[t])
    out["selection"] = {"best_by_metric": best, "within_1_point": cands, "selected": chosen,
                        "rule": "highest dev iso-FAR(T_base) IoU>=0.5 recall; within 1 point -> cheaper"}
    out["meta"] = run_metadata(cfg, {"stage": 2})
    write_json(out, ROOT / "results" / "stage2" / "inputs.json")
    print(f"T_base {t_base:.3f}")
    for tag, r in out["configs"].items():
        if r["status"] != "ok":
            print(tag, r["status"]); continue
        a, b, c = r["at_stage1_threshold"], r["iso_far_T_base"], r["iso_far_10pct"]
        print(f"{tag:6s} k={r['k_selected']:.2f} s/f={r['s_per_frame']} | thr0: R50 {a['recall_iou50']:.3f} Rc {a['recall_center']:.3f} FAR {a['far']:.3f}"
              f" | isoFAR: thr {b['threshold']:.3f} R50 {b['recall_iou50']:.3f} Rc {b['recall_center']:.3f} dist-c {b['recall_by_distance']['DISTANT']['center']:.3f}"
              f" | FAR10: R50 {c['recall_iou50']:.3f} Rc {c['recall_center']:.3f}")
    print(out["selection"])


if __name__ == "__main__":
    main()
