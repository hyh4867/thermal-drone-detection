"""Seed study of T2-20 (docs/PREREGISTRATION_SEEDS.md): names, paths and the freeze check.

Seed 0 is the stage 3 run itself: its checkpoint, caches, threshold and results are read from stage 3 and
never rewritten. Seeds 1 and 2 live under runs/seeds and results/seeds.
"""
from __future__ import annotations

import json
import subprocess

from ..common import ROOT, read_json
from ..stage3 import data as d3

SEEDS = (0, 1, 2)
NEW_SEEDS = (1, 2)
M = 1                                   # fixed: the magnification seed 0 chose (pre-registration, section 3)
OUT = ROOT / "results" / "seeds"
PREDS = ROOT / "runs" / "seeds" / "preds"
HOLD = "cst_tr_hold"
TEST_SETS = ("cst_val", "auv_test", "auv_test_small", "svan_test_h")
TRAIN_ITERS = 1075
frame_set = d3.frame_set


def model(seed: int) -> str:
    return "T2_20" if seed == 0 else f"T2_20s{seed}"


def tag(seed: int) -> str:
    return f"{model(seed).lower()}_m{M}"


def ckpt(seed: int) -> str:
    return "runs/stage3/t2_20/last.pt" if seed == 0 else f"runs/seeds/t2_20_s{seed}/last.pt"


def cache_path(tag_: str, set_name: str):
    """Caches of the new seeds are under runs/seeds/preds; everything else is a stage 3 cache."""
    base = PREDS if tag_.startswith("t2_20s") else d3.PREDS
    return base / f"{tag_}__{set_name}.jsonl"


def load_preds(tag_: str, set_name: str):
    frames = frame_set(set_name)
    with open(cache_path(tag_, set_name)) as f:
        preds = [json.loads(line) for line in f]
    if len(preds) != len(frames):
        raise RuntimeError(f"{tag_}__{set_name}: {len(preds)} predictions for {len(frames)} frames (unfinished?)")
    return preds


def frozen_path(seed: int):
    return OUT / f"frozen_seed{seed}.json"


def frozen_committed(seed: int) -> bool:
    """True only if the frozen file of this seed exists, is tracked and equals the committed version."""
    p = frozen_path(seed)
    if not p.exists():
        return False
    try:
        rel = str(p.relative_to(ROOT))
    except ValueError:                  # a file outside the repository is never "committed"
        return False
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if run("ls-files", "--error-unmatch", rel).returncode != 0:
        return False
    return run("diff", "--quiet", "HEAD", "--", rel).returncode == 0


def frozen(seed: int) -> dict:
    """Frozen settings of one seed. Seed 0: the stage 3 frozen configuration (read only)."""
    if seed == 0:
        c = read_json(d3.OUT / "frozen_config.json")["configs"]["T2_20"]
        return {"seed": 0, "source": "results/stage3/frozen_config.json (T2_20)", **c}
    return read_json(frozen_path(seed))


def s1() -> dict:
    return read_json(d3.OUT / "frozen_config.json")["configs"]["S1_m1"]
