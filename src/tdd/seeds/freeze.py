"""Seed study: operating threshold of one seed on the CST train hold-out, then the frozen file.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.seeds.freeze --seed 1   -> results/seeds/frozen_seed1.json

Stage 3 rule unchanged: threshold = max F1 on cst_tr_hold (20 sequences never trained on). m is fixed to 1.
Nothing here reads a test set. The file is written once; test inference is refused until it is committed.
Requires a complete training run (1,075 iterations) and a passed cache check of the hold-out cache.
"""
from __future__ import annotations

import argparse

from ..common import ROOT, load_config, read_json, run_metadata, write_json
from ..metrics import average_precision, best_f1_threshold
from ..stage3.evalx import Ev3
from ..stage3.infer import sha256_file
from . import data as D


def hold_point(tag: str, frames=None) -> dict:
    frames = frames if frames is not None else D.frame_set(D.HOLD)
    preds = D.load_preds(tag, D.HOLD)
    ap = average_precision([{"gt": f["gt"], "det": d} for f, d in zip(frames, preds)], 0.5)
    op = best_f1_threshold(ap)
    return {"ap50": ap["ap"], "f1_threshold": op["threshold"], "f1": op["f1"], "precision": op["precision"], "recall": op["recall"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=D.NEW_SEEDS)
    a = ap.parse_args()
    out = D.frozen_path(a.seed)
    if out.exists():
        raise SystemExit(f"refused: {out.name} already exists (a seed is frozen once)")
    cfg = load_config("configs/full.yaml")
    tr = read_json(D.OUT / f"train_t2_20_s{a.seed}.json")
    if tr["status"] != "complete" or tr["iterations"] != D.TRAIN_ITERS or tr["meta"]["seed"] != a.seed:
        raise SystemExit(f"refused: training of seed {a.seed} is not a complete {D.TRAIN_ITERS}-iteration run with that seed")
    tag, sha = D.tag(a.seed), sha256_file(ROOT / D.ckpt(a.seed))
    chk = read_json(D.OUT / f"cache_check__seed{a.seed}_hold.json")
    row = [c for c in chk["caches"] if c["tag"] == tag and c["set"] == D.HOLD]
    if not (chk["all_ok"] and row and row[0]["status"] == "ok" and chk["checkpoints_sha256"][D.ckpt(a.seed)] == sha):
        raise SystemExit("refused: the hold-out cache check did not pass for this checkpoint")
    hold = D.frame_set(D.HOLD)
    h = hold_point(tag, hold)
    mt = Ev3(hold, D.load_preds(tag, D.HOLD), h["f1_threshold"]).metrics3()
    seed0 = D.frozen(0)
    res = {"seed": a.seed, "model": D.model(a.seed), "m": D.M, "m_source": "fixed to the value seed 0 chose (results/stage3/track2_m.json)",
           "tag": tag, "checkpoint": D.ckpt(a.seed), "checkpoint_sha256": sha, "anchor_sizes": seed0["anchor_sizes"],
           "threshold": h["f1_threshold"], "threshold_source": f"max F1 on cst_tr_hold, cache {tag}__cst_tr_hold",
           "hold": {"sequences": len({f["seq"] for f in hold}), "frames": len(hold), **h,
                    "recall_center": mt["recall_center"], "box_ratio_pred_over_label": mt["box_ratio_pred_over_label"]},
           "train": {"file": f"results/seeds/train_t2_20_s{a.seed}.json", "iterations": tr["iterations"], "minutes_used": tr["minutes_used"],
                     "train_loss": tr["history"][-1]["train_loss"], "antiuav_val_ap50": tr["history"][-1]["val_ap50"],
                     "args": tr["args"], "git_commit": tr["meta"]["git_commit"]},
           "seed0_reference": {"threshold": seed0["threshold"], "hold": seed0["hold"]},
           "meta": run_metadata(cfg, {"stage": "seeds"})}
    write_json(res, out)
    print(f"frozen seed {a.seed}: threshold {h['f1_threshold']:.4f}, hold AP {h['ap50']:.4f}, F1 {h['f1']:.4f} (seed 0: {seed0['threshold']:.4f}, {seed0['hold']['ap50']:.4f})")


if __name__ == "__main__":
    main()
