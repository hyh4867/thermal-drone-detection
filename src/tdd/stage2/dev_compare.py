"""Stage 2: compare configurations on a dev frame set (selection only, never test).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.dev_compare --name tta --set svan_dev10 --tags s1_x1 s1_x1tta
  -> results/stage2/dev_compare__<name>.json
Per tag: box-size factor k (addendum 1 rule, chosen on this set), threshold matching
the stage 1 frame alarm rate on the same frames (T_base) and 10%, recalls there, and
the paired video-level bootstrap against the first tag.
"""
from __future__ import annotations

import argparse

from ..common import ROOT, load_config, run_metadata, write_json
from .data import frame_set, load_preds
from .diagnose import s1_threshold
from .evalset import SvanEval, bootstrap, select_k


def compare(set_name, tags):
    assert set_name.startswith("svan_dev"), "dev sets only"
    thr0 = s1_threshold()
    frames = frame_set(set_name)
    t_base = SvanEval(frames, load_preds("s1", set_name)).far(thr0)
    out = {"set": set_name, "frames": len(frames), "videos": len({f["video"] for f in frames}),
           "stage1_threshold": thr0, "T_base": t_base, "configs": {}}
    ref = None
    for tag in tags:
        preds = load_preds(tag, set_name)
        k, sweep = select_k(frames, preds, thr0)
        ev = SvanEval(frames, preds, k)
        tb, t10 = ev.threshold_for_far(t_base), ev.threshold_for_far(0.10)
        row = {"k": k, "k_sweep": sweep, "at_stage1_threshold": ev.metrics(thr0),
               "T_base": {**ev.metrics(tb), **bootstrap(ev, tb, *((ref[0], ref[1]) if ref else ()))},
               "T10": {**ev.metrics(t10), **bootstrap(ev, t10, *((ref[0], ref[2]) if ref else ()))}}
        if ref is None:
            ref = (ev, tb, t10)
        out["configs"][tag] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", default="svan_dev_b")
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    out = compare(a.set, a.tags)
    out["meta"] = run_metadata(load_config("configs/full.yaml"), {"stage": 2})
    write_json(out, ROOT / "results" / "stage2" / f"dev_compare__{a.name}.json")
    print(f"{a.set}: {out['frames']} frames, T_base {out['T_base']:.3f}")
    for tag, r in out["configs"].items():
        b, c = r["T_base"], r["T10"]
        d = b.get("paired_diff", {}).get("recall_iou50")
        print(f"{tag:12s} k={r['k']:.2f} | T_base thr {b['threshold']:.4f} R50 {b['recall_iou50']:.3f} {b['ci95']['recall_iou50']} Rc {b['recall_center']:.3f}"
              f" distant R50/Rc {b['recall_by_distance']['DISTANT']['iou50']:.3f}/{b['recall_by_distance']['DISTANT']['center']:.3f}"
              f" | T10 R50 {c['recall_iou50']:.3f} Rc {c['recall_center']:.3f}" + (f" | vs first: {d['diff']:+.3f} {d['ci95']}" if d else ""))


if __name__ == "__main__":
    main()
