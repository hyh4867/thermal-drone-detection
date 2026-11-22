"""Stage 3: the cost of T2-20 on Svanström test (pre-registration 3, addendum 6). Descriptive only.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.svan_eval
  -> results/stage3/svanstrom_test.json   (refuses to run while the working tree is dirty)

Run AFTER the final results were seen; not used for any verdict. Frames: svan_test_h (every other stage 2
Svanström test frame). Configurations and thresholds are the frozen ones; no box calibration (k = 1).
The reading rule on the paired centre-recall difference (T2-20 minus S1) was fixed before the run.
"""
from __future__ import annotations

import numpy as np

from ..common import load_config, read_json, run_metadata, write_json
from ..stage2 import tables as T
from ..stage2.evalset import SvanEval, bootstrap as sv_bootstrap
from .data import OUT, frame_set, load_preds

SET = "svan_test_h"
SIZES = [("tiny", 0, 8), ("small", 8, 16), ("medium", 16, 32), ("large", 32, 64), ("xlarge", 64, 1e9)]   # reference px (640x512)


def reading(ci):
    if ci[1] < 0:
        return "T2-20은 Svanström에서도 손해를 봤다"
    if ci[0] > 0:
        return "Svanström에서는 이득이 있었다"
    return "Svanström에서는 차이를 확인하지 못했다"


def row(ev: SvanEval, thr, ref=None, ref_thr=None):
    g = ev.gt
    m = ev.metrics(thr)
    t10 = ev.threshold_for_far(0.10)
    m10 = ev.metrics(t10)
    loose = g["loose_score"] >= thr
    out = {"threshold": float(thr), "k": 1.0, **m,
           "recall_by_size": {n: {"gt": int(((g["size"] >= lo) & (g["size"] < hi)).sum()),
                                  "iou50": T.recall(g, thr, "iou50", (g["size"] >= lo) & (g["size"] < hi)),
                                  "center": T.recall(g, thr, "center", (g["size"] >= lo) & (g["size"] < hi))} for n, lo, hi in SIZES},
           "at_far10_same_frames": {"threshold": t10, "far": m10["far"], "recall_iou50": m10["recall_iou50"], "recall_center": m10["recall_center"],
                                    "note": "a point read off the curve of the same frames, not a usable threshold"},
           "box_ratio_pred_over_label": {"n": int(loose.sum()), "width_median": float(np.nanmedian(g["loose_rw"][loose])),
                                         "height_median": float(np.nanmedian(g["loose_rh"][loose]))},
           **(sv_bootstrap(ev, thr, ref, ref_thr) if ref is not None else sv_bootstrap(ev, thr))}
    return out


def main():
    cfg = load_config("configs/full.yaml")
    meta = run_metadata(cfg, {"stage": 3})
    if meta["git_dirty"]:
        raise SystemExit("refused: the working tree is dirty (commit first; results/ does not count)")
    fz = read_json(OUT / "frozen_config.json")
    frames = frame_set(SET)
    c1, c2 = fz["configs"]["T1"], fz["configs"]["T2_20"]
    e1 = SvanEval(frames, load_preds(c1["tag"], SET), 1.0)
    e2 = SvanEval(frames, load_preds(c2["tag"], SET), 1.0)
    r1 = {"tag": c1["tag"], "model": c1["model"], "m": c1["m"], "checkpoint_sha256": c1["checkpoint_sha256"], **row(e1, c1["threshold"])}
    r2 = {"tag": c2["tag"], "model": c2["model"], "m": c2["m"], "checkpoint_sha256": c2["checkpoint_sha256"],
          **row(e2, c2["threshold"], e1, c1["threshold"])}
    d = r2["paired_diff"]["recall_center"]
    p10 = d["diff"] < -0.03
    res = {"set": SET, "frames": len(frames), "videos": len({f["video"] for f in frames}),
           "note": "run after the final results were seen (addendum 6); not used for any verdict; Svanström input is enlarged 2x (m = 1); k = 1",
           "configs": {"T1_S1_m1": r1, "T2_20": r2},
           "paired": "T2_20 minus T1_S1_m1, each at its own frozen threshold (their alarm rates differ)",
           "reading_rule": {"diff_recall_center": d, "sentence": reading(d["ci95"])},
           "P10": {"prediction": "T2-20 centre recall more than 3 points below the track 1 final", "diff": d["diff"], "correct": bool(p10)},
           "frozen_config": "results/stage3/frozen_config.json", "meta": meta}
    write_json(res, OUT / "svanstrom_test.json")
    for n, r in res["configs"].items():
        print(f"{n}: Rc {r['recall_center']:.4f} R50 {r['recall_iou50']:.4f} FAR {r['far']:.4f} far10 Rc {r['at_far10_same_frames']['recall_center']:.4f}")
    print("diff Rc:", d, "->", res["reading_rule"]["sentence"], "| P10 correct:", p10)


if __name__ == "__main__":
    main()
