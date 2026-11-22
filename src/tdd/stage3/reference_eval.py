"""Stage 3 reference row (B, m = 1) on CST val (pre-registration 3, addendum 5). Descriptive only.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.reference_eval
  -> results/stage3/reference__cst_val.json   (refuses to run while the working tree is dirty)

Run AFTER the final results were seen; not used for any verdict. The operating threshold is the recorded
Anti-UAV val max-F1 threshold of this model and inference setting (selection.json, cache b_m1__auv_val);
nothing is chosen on CST. final__*.json is not touched. The reading rule on the paired AP@0.5 difference
against (S1, m = 1) was fixed before the run.
"""
from __future__ import annotations

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from .data import OUT, frame_set, load_preds
from .evalx import Ev3, bootstrap3
from .infer import model_spec, sha256_file

TAG, MODEL, M = "b_m1", "B", 1


def reading(ci):
    if ci[0] > 0:
        return "선택 규칙이 CST에서 도움이 됐을 구성을 받아들이지 않았다"
    if ci[1] < 0:
        return "B가 CST val에서 S1보다 낮았다"
    return "참조 행에서도 B가 S1보다 낫다고 확인되지 않았다"


def main():
    cfg = load_config("configs/full.yaml")
    meta = run_metadata(cfg, {"stage": 3})
    if meta["git_dirty"]:
        raise SystemExit("refused: the working tree is dirty (commit first; results/ does not count)")
    fz = read_json(OUT / "frozen_config.json")
    thr = read_json(OUT / "selection.json")["candidates"][TAG]["V3"]["f1_threshold"]
    frames = frame_set("cst_val")
    s1 = fz["configs"]["S1_m1"]
    ref = Ev3(frames, load_preds(s1["tag"], "cst_val"), s1["threshold"])
    ev = Ev3(frames, load_preds(TAG, "cst_val"), thr)
    ckpt, anchors, _ = model_spec(MODEL)
    row = {"status": "ok", "tag": TAG, "model": MODEL, "m": M, "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ROOT / ckpt),
           "anchor_sizes": list(anchors), "threshold_source": "results/stage3/selection.json b_m1.V3.f1_threshold (max F1 on Anti-UAV val, cache b_m1__auv_val)",
           **ev.metrics3(), **bootstrap3(ev, ref)}
    d = row["paired_diff"]["ap50"]
    res = {"set": "cst_val", "frames": len(frames), "sequences": len({f["seq"] for f in frames}),
           "note": "reference row run after the final results were seen (addendum 5); not used for any verdict",
           "reference": "S1_m1 (paired_diff is against it; its numbers are those of final__cst_val.json)",
           "S1_m1": {"tag": s1["tag"], "threshold": s1["threshold"], "ap50": ref.metrics()["ap50"]},
           "configs": {"B_m1": row},
           "reading_rule": {"diff_ap50": d, "sentence": reading(d["ci95"])},
           "frozen_config": "results/stage3/frozen_config.json", "meta": meta}
    write_json(res, OUT / "reference__cst_val.json")
    print(f"B_m1: AP {row['ap50']:.4f} {row['ci95']['ap50']} R50 {row['recall_iou50']:.4f} Rc {row['recall_center']:.4f} FAR {row['far_drone_free']:.4f}")
    print("diff AP vs S1:", d, "->", res["reading_rule"]["sentence"])


if __name__ == "__main__":
    main()
