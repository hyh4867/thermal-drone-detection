"""Stage 3: track 2 choices on the CST train hold-out and the frozen configuration of both tracks.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.freeze --step m      (choose m for track 2 from T2_20 on cst_tr_hold)
       PYTHONPATH=src .venv/bin/python -m tdd.stage3.freeze --step final  (thresholds, box convention, frozen_config.json)

Nothing here reads cst_val or cst_test. Track 1 comes unchanged from frozen_track1.json (no CST data).
Track 2 (pre-registration 3, section 6): m in (1, 2) by AP@0.5 of T2_20 on cst_tr_hold (within 0.01 -> m = 1),
the same m for T2_5; operating threshold = max F1 on cst_tr_hold for each model; no box calibration.
"""
from __future__ import annotations

import argparse
import json

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from ..metrics import average_precision, best_f1_threshold
from ..stage2.freeze import s1_threshold
from .data import OUT, frame_set, load_preds
from .evalx import Ev3
from .infer import model_spec, sha256_file

TIE = 0.01


def hold_ap(tag, frames):
    preds = load_preds(tag, "cst_tr_hold")
    ap = average_precision([{"gt": f["gt"], "det": d} for f, d in zip(frames, preds)], 0.5)
    op = best_f1_threshold(ap)
    return {"ap50": ap["ap"], "f1_threshold": op["threshold"], "f1": op["f1"], "precision": op["precision"], "recall": op["recall"]}


def spec(model):
    ckpt, anchors, _ = model_spec(model)
    return {"checkpoint": ckpt, "checkpoint_sha256": sha256_file(ROOT / ckpt), "anchor_sizes": list(anchors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["m", "final"])
    a = ap.parse_args()
    cfg = load_config("configs/full.yaml")
    hold = frame_set("cst_tr_hold")
    t1 = read_json(OUT / "frozen_track1.json")
    r = {m: hold_ap(f"t2_20_m{m}", hold) for m in (1, 2)}
    m2 = 2 if r[2]["ap50"] > r[1]["ap50"] + TIE else 1
    choice = {"hold_sequences": len({f["seq"] for f in hold}), "hold_frames": len(hold), "T2_20": {f"m{m}": v for m, v in r.items()},
              "m": m2, "rule": f"m = 2 only if AP@0.5 on cst_tr_hold is more than {TIE} above m = 1"}
    print(json.dumps(choice, indent=1))
    if a.step == "m":
        write_json({**choice, "meta": run_metadata(cfg, {"stage": 3})}, OUT / "track2_m.json")
        return
    assert read_json(OUT / "track2_m.json")["m"] == m2
    t2 = {}
    for name in ("T2_20",):          # addendum 2: T2-5 is not run
        tag = f"{name.lower()}_m{m2}"
        h = hold_ap(tag, hold)
        t2[name] = {"model": name, "m": m2, "tag": tag, **spec(name), "threshold": h["f1_threshold"],
                    "threshold_source": f"max F1 on cst_tr_hold, cache {tag}__cst_tr_hold", "hold": h}
    # box convention on the hold-out (reported only; no calibration is applied)
    fin = t1["final"]
    conv = {}
    for label, tag, thr in ((f"track 1 final ({fin['tag']})", fin["tag"], fin["threshold"]),
                            ("T2_20", t2["T2_20"]["tag"], t2["T2_20"]["threshold"])):
        ev = Ev3(hold, load_preds(tag, "cst_tr_hold"), thr)
        mt = ev.metrics3()
        conv[label] = {"threshold": thr, **mt["box_ratio_pred_over_label"], "recall_iou50": mt["recall_iou50"],
                       "recall_center": mt["recall_center"], "ap50": mt["ap50"]}
    configs = {"T1": {"role": "track 1 final", **fin}}
    configs["S1_m1"] = {"role": "reference and baseline of J1/J2/J5", "model": "S1", "m": 1, "tag": "s1_m1", **spec("S1"),
                        "threshold": s1_threshold(), "threshold_source": "stage 1 value"}
    for name, c in t2.items():
        configs[name] = {"role": "track 2", **c}
    # a reference that duplicates the final configuration is dropped
    for k in ("S1_m1",):
        if k in configs and configs[k]["tag"] == fin["tag"]:
            configs[k]["duplicate_of"] = "T1"
    write_json({"configs": configs, "track2_choice": choice, "box_convention_on_cst_tr_hold": conv,
                "final_sets": {"cst_val": ["S1_m1", "T1", "T2_20"],          # addendum 2: reduced final evaluation
                               "auv_test": ["T1", "T2_20"], "auv_test_small": ["T1", "T2_20"]},
                "meta": run_metadata(cfg, {"stage": 3})}, OUT / "frozen_config.json")
    print("frozen:", {k: (v["tag"], round(v["threshold"], 4)) for k, v in configs.items()})


if __name__ == "__main__":
    main()
