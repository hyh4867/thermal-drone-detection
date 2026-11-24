"""Seed study, POST HOC (not pre-registered, written after the seed results were seen; descriptive only).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.seeds.posthoc_threshold   -> results/seeds/posthoc_common_threshold.json

Question: how much of the seed-to-seed spread of the threshold-dependent numbers (alarm rates, recall) comes from
the different operating thresholds the seeds got on the hold-out (0.909-0.938), and how much from the weights?
All three seeds are read at ONE common threshold, the frozen threshold of seed 0. No new inference, no verdict,
no selection; the registered numbers (each seed at its own threshold) stay the result.
"""
from __future__ import annotations

from ..common import load_config, run_metadata, write_json
from ..stage2.evalset import SvanEval
from ..stage3.evalx import Ev3
from . import data as D
from .evaluate import spread


def main():
    meta = run_metadata(load_config("configs/full.yaml"), {"stage": "seeds"})
    if meta["git_dirty"]:
        raise SystemExit("refused: the working tree is dirty (commit first; results/ does not count)")
    thr = D.frozen(0)["threshold"]
    cst, auv, sv = D.frame_set("cst_val"), D.frame_set("auv_test"), D.frame_set("svan_test_h")
    rows = {"cst_val.recall_center": {}, "cst_val.recall_tiny_center": {}, "cst_val.far_drone_free": {},
            "antiuav.recall_iou50": {}, "antiuav.far_drone_free": {}, "svanstrom.far": {}, "svanstrom.recall_center": {}}
    for s in D.SEEDS:
        t = D.tag(s)
        m = Ev3(cst, D.load_preds(t, "cst_val"), thr).metrics()
        rows["cst_val.recall_center"][s] = m["recall_center"]
        rows["cst_val.recall_tiny_center"][s] = m["recall_by_size"]["tiny"]["center"]
        rows["cst_val.far_drone_free"][s] = m["far_drone_free"]
        m = Ev3(auv, D.load_preds(t, "auv_test"), thr).metrics()
        rows["antiuav.recall_iou50"][s] = m["recall_iou50"]
        rows["antiuav.far_drone_free"][s] = m["far_drone_free"]
        m = SvanEval(sv, D.load_preds(t, "svan_test_h"), 1.0).metrics(thr)
        rows["svanstrom.far"][s] = m["far"]
        rows["svanstrom.recall_center"][s] = m["recall_center"]
    res = {"note": "POST HOC, descriptive: every seed read at the frozen threshold of seed 0 instead of its own; not pre-registered, no verdict",
           "common_threshold": thr, "own_thresholds": {str(s): D.frozen(s)["threshold"] for s in D.SEEDS},
           "metrics": {k: spread(v) for k, v in rows.items()}, "meta": meta}
    write_json(res, D.OUT / "posthoc_common_threshold.json")
    for k, v in res["metrics"].items():
        print(f"{k:28} " + " ".join(f"s{a}={b:.4f}" for a, b in v["by_seed"].items()) + f" | range {v['range']:.4f}")


if __name__ == "__main__":
    main()
