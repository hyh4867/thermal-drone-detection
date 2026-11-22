"""Stage 3 track 1 selection (pre-registration 3, section 5). No CST data is read here.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.select
  -> results/stage3/selection.json       V1, V2, V3 of every candidate and the rule's steps
  -> results/stage3/frozen_track1.json   the track 1 final configuration (model, m, threshold, checkpoint hash)

V1  Svanström dev, every 20th frame (svan_dev20; addendum 2): centre-criterion recall at the 10% frame-alarm point of the same frames
V2  AP@0.5 on Anti-UAV val reduced to 1/4
V3  AP@0.5 on Anti-UAV val
Rule 1 (parent): among (S1, B, T1a) x (m = 1, 2) with V3 >= V3(S1, m1) - 0.02, the highest
  min(V1 - V1(base), V2 - V2(base)); within 0.01 of the best -> cheaper (smaller m, then S1 > B > T1a);
  if no score is > 0 the final configuration is (S1, m = 1).
Rule 2 (child): C37 for S1, T1b for B, same m; replaces the parent only if V2 >= V2(parent) - 0.01 and
  V3 >= V3(parent) - 0.01.
"""
from __future__ import annotations

import json

from ..common import ROOT, load_config, run_metadata, write_json
from ..metrics import average_precision, best_f1_threshold
from ..stage2 import data as d2
from ..stage2.evalset import SvanEval, bootstrap as sv_bootstrap
from ..stage2.freeze import s1_threshold
from .data import OUT, frame_set, load_preds, stage2_subset
from .infer import model_spec, sha256_file

PARENTS = ("S1", "B", "T1a")
CHILD = {"S1": "C37", "B": "T1b"}
MS = (1, 2)
V3_MARGIN, TIE, CHILD_MARGIN = 0.02, 0.01, 0.01


def tag(model, m):
    return f"{model.lower()}_m{m}"


def svan_preds(model, m):
    """S1 on svan_dev10 exists from stage 2 (same settings: whole image, no TTA): x2 = stage 1 caches, x4 = s1_x4."""
    if model == "S1":
        src = "s1" if m == 1 else "s1_x4"
        return stage2_subset(src, "svan_dev10", "svan_dev20"), "stage 2 cache " + ("s1 (x2)" if m == 1 else "s1_x4") + ", frames of svan_dev20"
    return load_preds(tag(model, m), "svan_dev20"), f"stage 3 cache {tag(model, m)}__svan_dev20"


def v1(model, m, frames):
    preds, src = svan_preds(model, m)
    ev = SvanEval(frames, preds, 1.0)
    thr = ev.threshold_for_far(0.10)
    mt = ev.metrics(thr)
    return {"value": mt["recall_center"], "threshold": thr, "far": mt["far"], "recall_iou50": mt["recall_iou50"],
            "distant_center": mt["recall_by_distance"]["DISTANT"]["center"], "ci95": sv_bootstrap(ev, thr)["ci95"]["recall_center"],
            "source": src}


def ap_val(model, m, set_name, frames):
    preds = load_preds(tag(model, m), set_name)
    ap = average_precision([{"gt": f["gt"], "det": d} for f, d in zip(frames, preds)], 0.5)
    op = best_f1_threshold(ap)
    return {"value": ap["ap"], "f1_threshold": op["threshold"], "f1": op["f1"]}


def main():
    cfg = load_config("configs/full.yaml")
    svan, val, valq = frame_set("svan_dev20"), frame_set("auv_val"), frame_set("auv_val_q")
    cand = {}
    for model in (*PARENTS, *CHILD.values()):
        for m in MS:
            c = {"model": model, "m": m, "V2": ap_val(model, m, "auv_val_q", valq), "V3": ap_val(model, m, "auv_val", val)}
            if model in PARENTS:
                c["V1"] = v1(model, m, svan)
            cand[tag(model, m)] = c
    base = cand[tag("S1", 1)]
    steps = []
    rows = []
    for model in PARENTS:
        for m in MS:
            c = cand[tag(model, m)]
            c["eligible"] = c["V3"]["value"] >= base["V3"]["value"] - V3_MARGIN
            c["d1"] = c["V1"]["value"] - base["V1"]["value"]
            c["d2"] = c["V2"]["value"] - base["V2"]["value"]
            c["score"] = min(c["d1"], c["d2"])
            rows.append(c)
    elig = [c for c in rows if c["eligible"]]
    best = max(c["score"] for c in elig)
    cost = lambda c: (c["m"], PARENTS.index(c["model"]))
    if best <= 0:
        parent = base
        steps.append("rule 1: no eligible configuration has a score > 0 -> (S1, m = 1)")
    else:
        near = [c for c in elig if c["score"] >= best - TIE and c["score"] > 0]
        parent = min(near, key=cost)
        steps.append(f"rule 1: best score {best:.4f}; within {TIE}: {[tag(c['model'], c['m']) for c in near]} -> {tag(parent['model'], parent['m'])}")
    final = parent
    child_name = CHILD.get(parent["model"])
    if child_name:
        ch = cand[tag(child_name, parent["m"])]
        ok2 = ch["V2"]["value"] >= parent["V2"]["value"] - CHILD_MARGIN
        ok3 = ch["V3"]["value"] >= parent["V3"]["value"] - CHILD_MARGIN
        ch["replaces_parent"] = bool(ok2 and ok3)
        steps.append(f"rule 2: child {tag(child_name, parent['m'])}: V2 {ch['V2']['value']:.4f} vs parent {parent['V2']['value']:.4f} ({'ok' if ok2 else 'fails'}), "
                     f"V3 {ch['V3']['value']:.4f} vs {parent['V3']['value']:.4f} ({'ok' if ok3 else 'fails'}) -> {'child' if ok2 and ok3 else 'parent'}")
        if ok2 and ok3:
            final = ch
        sibling = parent if final is ch else ch
    else:
        steps.append("rule 2: the parent has no child")
        sibling = None
    for s in steps:
        print(s)
    meta = run_metadata(cfg, {"stage": 3})
    write_json({"candidates": cand, "base": tag("S1", 1), "steps": steps, "parent": tag(parent["model"], parent["m"]),
                "final": tag(final["model"], final["m"]), "sibling": tag(sibling["model"], sibling["m"]) if sibling else None,
                "margins": {"V3": V3_MARGIN, "tie": TIE, "child": CHILD_MARGIN}, "meta": meta}, OUT / "selection.json")
    ckpt, anchors, _ = model_spec(final["model"])
    is_base = final is base
    thr = s1_threshold() if is_base else final["V3"]["f1_threshold"]
    write_json({"track": 1,
                "final": {"model": final["model"], "m": final["m"], "tag": tag(final["model"], final["m"]), "checkpoint": ckpt,
                          "checkpoint_sha256": sha256_file(ROOT / ckpt), "anchor_sizes": list(anchors), "threshold": thr,
                          "threshold_source": "stage 1 value" if is_base else f"max F1 on Anti-UAV val, cache {tag(final['model'], final['m'])}__auv_val"},
                "sibling": None if sibling is None else {"model": sibling["model"], "m": sibling["m"], "tag": tag(sibling["model"], sibling["m"]),
                                                         "threshold": sibling["V3"]["f1_threshold"] if sibling is not base else s1_threshold()},
                "selection": "results/stage3/selection.json", "meta": meta}, OUT / "frozen_track1.json")
    print(json.dumps(json.loads((OUT / "frozen_track1.json").read_text())["final"], indent=1))


if __name__ == "__main__":
    main()
