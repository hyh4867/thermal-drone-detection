"""Seed study inference: one seed, one frame set, with the stage 3 inference code itself.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.seeds.infer --seed 1 --set cst_tr_hold

The loop, the predictor and the cache format are tdd.stage3.infer.run (whole image, m = 1, no TTA, no tiles);
only the model table, the cache folder and the guard are replaced. Guard: the hold-out set is always allowed;
a test set (CST val, Anti-UAV test and supplement, Svanström test) is refused until the frozen file of that
seed is COMMITTED and its checkpoint sha256 equals the checkpoint on disk.
"""
from __future__ import annotations

import argparse

from ..common import ROOT
from ..stage3 import infer as base
from . import data as D


def register():
    for s in D.NEW_SEEDS:
        base.MODELS[D.model(s)] = (D.ckpt(s), None, True)     # anchors follow the track 1 final model, like T2_20
    base.cache_path = D.cache_path
    base.guard = guard


def guard(model: str, set_name: str):
    seeds = [s for s in D.NEW_SEEDS if D.model(s) == model]
    if not seeds:
        raise SystemExit(f"refused: {model} is not a model of the seed study")
    seed = seeds[0]
    if set_name == D.HOLD:
        return
    if set_name not in D.TEST_SETS:
        raise SystemExit(f"refused: {set_name} is not part of the seed study")
    if not D.frozen_committed(seed):
        raise SystemExit(f"refused: {set_name} before results/seeds/frozen_seed{seed}.json is committed")
    if D.frozen(seed)["checkpoint_sha256"] != base.sha256_file(ROOT / D.ckpt(seed)):
        raise SystemExit(f"refused: the checkpoint of seed {seed} differs from the frozen sha256")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=D.NEW_SEEDS)
    ap.add_argument("--set", required=True, choices=(D.HOLD, *D.TEST_SETS))
    a = ap.parse_args()
    register()
    base.run(D.tag(a.seed), a.set, D.model(a.seed), float(D.M))
