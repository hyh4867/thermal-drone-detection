"""Stage 2: freeze every choice from dev data into results/stage2/frozen_config.json.

Usage:
  PYTHONPATH=src .venv/bin/python -m tdd.stage2.freeze --input x2 \
      --model A:s1:runs/full_frcnn/best.pt:svan_dev \
      --model B:r1e1:runs/stage2/r1/epoch1.pt:svan_dev \
      --model C37:r2n37:runs/stage2/r2_n37/last.pt:svan_dev_b
  (name : prediction-tag prefix : checkpoint : dev set used for calibration)

Per model, from that dev set only: box-size factor k (addendum 1), the score
thresholds that match the stage 1 frame alarm rate on the same dev frames (T_base)
and 10% (T10), the tracker parameters (pre-registered grid) with their track-score
thresholds, and the Anti-UAV-val max-F1 threshold used on Anti-UAV / CST.
Nothing here reads a test split. The file is committed before the final evaluation.
"""
from __future__ import annotations

import argparse

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from ..metrics import average_precision, best_f1_threshold
from .data import frame_set, load_preds
from .diagnose import s1_threshold
from .evalset import SvanEval, select_k
from .track_eval import tune

INPUTS = {"x1": (1, 1, False), "x2": (2, 1, False), "x3": (3, 1, False), "x4": (4, 1, False), "t3": (3, 2, False),
          "t4": (4, 2, False), "x1tta": (1, 1, True), "x2tta": (2, 1, True), "x3tta": (3, 1, True), "x4tta": (4, 1, True),
          "t3tta": (3, 2, True), "t4tta": (4, 2, True)}


def pred_tag(prefix: str, inp: str) -> str:
    """Prediction-cache tag of a model at a Svanström input configuration."""
    return "s1" if (prefix, inp) == ("s1", "x2") else f"{prefix}_{inp}"


def native_tag(prefix: str, tta: bool) -> str:
    """Tag for 640x512 data (Anti-UAV, CST): scale 1, TTA only if adopted."""
    return "s1" if (prefix == "s1" and not tta) else f"{prefix}_n{'tta' if tta else ''}"


def val_threshold(prefix: str, ckpt: str, tta: bool) -> dict:
    """Max-F1 threshold on Anti-UAV val for this model (stage 1 rule)."""
    if prefix == "s1" and not tta:
        return {"threshold": s1_threshold(), "source": "results/full/model/antiuav_metrics.json"}
    val = frame_set("auv_val")
    if tta:
        preds, src = load_preds(native_tag(prefix, True), "auv_val"), f"cache {native_tag(prefix, True)}__auv_val"
    else:
        p = ROOT / ckpt
        ep = p.stem.replace("epoch", "") if p.stem.startswith("epoch") else None
        cand = p.parent / (f"val_preds_epoch{ep}.json" if ep is not None else "val_preds_last.json")
        if not cand.exists():       # last.pt of a one-epoch run == epoch0
            cand = sorted(p.parent.glob("val_preds_epoch*.json"))[-1]
        preds, src = read_json(cand), str(cand.relative_to(ROOT))
    ap = average_precision([{"gt": f["gt"], "det": d} for f, d in zip(val, preds)], 0.5)
    op = best_f1_threshold(ap)
    return {"threshold": op["threshold"], "val_ap50": ap["ap"], "val_f1": op["f1"], "source": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, choices=sorted(INPUTS))
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", default="frozen_config")
    a = ap.parse_args()
    cfg = load_config("configs/full.yaml")
    thr0 = s1_threshold()
    scale, tile, tta = INPUTS[a.input]
    out = {"input": {"name": a.input, "scale": scale, "tile": tile, "tta": tta,
                     "note": "Svanström (320x256) only; 640x512 data is run at scale 1"},
           "stage1_threshold": thr0, "models": {}}
    # box-size factor for the stage 1 model at its stage 1 input (x2): the "box calibration only" step
    dev = frame_set("svan_dev")
    k2, sweep2 = select_k(dev, load_preds("s1", "svan_dev"), thr0)
    ev2 = SvanEval(dev, load_preds("s1", "svan_dev"), k2)
    out["stage1_x2"] = {"dev_set": "svan_dev", "k": k2, "k_sweep": sweep2, "dev_at_stage1_threshold": ev2.metrics(thr0),
                        "thresholds": {"T10": ev2.threshold_for_far(0.10)}}       # reference row of the 10% table
    for spec in a.model:
        name, prefix, ckpt, dev = spec.split(":")
        frames = frame_set(dev)
        t_base = SvanEval(frames, load_preds("s1", dev)).far(thr0)
        tag = pred_tag(prefix, a.input)
        preds = load_preds(tag, dev)
        k, sweep = select_k(frames, preds, thr0)
        ev = SvanEval(frames, preds, k)
        thr_b, thr_10 = ev.threshold_for_far(t_base), ev.threshold_for_far(0.10)
        tf = ROOT / "results" / "stage2" / f"tracker_tuning__{tag}__{dev}.json"
        tr = read_json(tf) if tf.exists() and read_json(tf)["k"] == k else tune(tag, k, dev)
        sel = tr["selected"]
        out["models"][name] = {
            "prefix": prefix, "checkpoint": ckpt, "dev_set": dev, "svanstrom_tag": tag, "native_tag": native_tag(prefix, tta),
            "T_base_on_dev_set": t_base, "k": k, "k_sweep": sweep,
            "thresholds": {"T_base": thr_b, "T10": thr_10},
            "dev_detector": {"T_base": ev.metrics(thr_b), "T10": ev.metrics(thr_10), "at_stage1_threshold": ev.metrics(thr0)},
            "tracker": {"params": {"model": tr["motion_model"], "q": sel["q"], "confirm_m": sel["confirm_m"],
                                   "confirm_n": sel["confirm_n"], "delete_misses": sel["delete_misses"],
                                   "det_threshold": sel["det_threshold"], "coast_outputs": sel["coast_outputs"]},
                        "track_score_thresholds": {"T_base": sel["T_base"]["track_score_threshold"],
                                                   "T10": sel["T10"]["track_score_threshold"]},
                        "dev": {"T_base": sel["T_base"], "T10": sel["T10"]},
                        "tuning_file": f"results/stage2/tracker_tuning__{tag}__{dev}.json"},
            "antiuav_val": val_threshold(prefix, ckpt, tta)}
        m = out["models"][name]
        print(name, "k", k, "thr", round(thr_b, 4), round(thr_10, 4), "dev R50@Tbase", round(m["dev_detector"]["T_base"]["recall_iou50"], 3),
              "val thr", round(m["antiuav_val"]["threshold"], 4))
    out["meta"] = run_metadata(cfg, {"stage": 2})
    write_json(out, ROOT / "results" / "stage2" / f"{a.out}.json")


if __name__ == "__main__":
    main()
