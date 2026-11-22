"""Stage 2 frame sets and prediction caches.

Every frame set is a list of manifest records; predictions are lists of
[x1, y1, x2, y2, score] in the frame's NATIVE pixel coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common import ROOT, read_json
from .split import load_split

MAN = ROOT / "data" / "manifests"
PREDS = ROOT / "runs" / "stage2" / "preds"
S1_RUN = ROOT / "runs" / "full_frcnn"
REF_W = 640.0   # reference frame width; sizes and tracker units are quoted at 640x512


def ref_scale(frame: dict) -> float:
    """Factor from native pixels to the 640x512 reference frame (Svanström 320x256 -> 2)."""
    return REF_W / frame["width"]


def svanstrom(splits, every: int = 5):
    sp = load_split()
    return [f for f in read_json(MAN / "svanstrom_ir.json") if sp[f["video"]] in splits and f["frame"] % every == 0]


def frame_set(name: str):
    """Named frame sets used by stage 2."""
    if name == "svan_dev10":      # input experiment: dev videos, every 10th frame
        return svanstrom(("dev_a", "dev_b"), 10)
    if name == "svan_dev5":       # the dev frames that are not in svan_dev10 (dev = dev10 + dev5)
        return [f for f in svanstrom(("dev_a", "dev_b")) if f["frame"] % 10 == 5]
    if name == "svan_dev":
        return svanstrom(("dev_a", "dev_b"))
    if name == "svan_dev_b":
        return svanstrom(("dev_b",))
    if name == "svan_dev_a":
        return svanstrom(("dev_a",))
    if name == "svan_test":
        return svanstrom(("test",))
    if name == "svan_all":
        return read_json(MAN / "svanstrom_ir.json")
    if name in ("auv_val", "auv_test", "auv_test_small", "auv_track", "cst_test"):
        fn = {"auv_val": "antiuav_val", "auv_test": "antiuav_test", "auv_test_small": "antiuav_test_small_supplement",
              "auv_track": "antiuav_test_track", "cst_test": "cst_test"}[name]
        return read_json(MAN / f"{fn}.json")
    raise KeyError(name)


def _key(f):
    return f["image"]


def s1_preds(name: str):
    """Stage 1 model predictions (default configuration) cut from the stage 1 caches."""
    src = {"svan": ("svanstrom_ir", "preds_svanstrom_ir.json"), "auv_val": ("antiuav_val", "preds_antiuav_val.json"),
           "auv_test": ("antiuav_test", "preds_antiuav_test.json"),
           "auv_test_small": ("antiuav_test_small_supplement", "preds_antiuav_test_small.json")}
    k = "svan" if name.startswith("svan") else name
    man, cache = src[k]
    full = read_json(MAN / f"{man}.json")
    preds = read_json(S1_RUN / cache)
    assert len(full) == len(preds), (name, len(full), len(preds))
    idx = {_key(f): p for f, p in zip(full, preds)}
    return [idx[_key(f)] for f in frame_set(name)]


def cache_path(tag: str, set_name: str) -> Path:
    return PREDS / f"{tag}__{set_name}.jsonl"


def load_preds(tag: str, set_name: str):
    """Predictions of configuration ``tag`` on ``set_name`` (tag 's1' = stage 1 caches)."""
    if tag == "s1" and (set_name.startswith("svan") or set_name in ("auv_val", "auv_test", "auv_test_small")):
        return s1_preds(set_name)
    frames = frame_set(set_name)
    p = cache_path(tag, set_name)
    if not p.exists() and set_name.startswith("svan"):
        return _from_other_svan_caches(tag, set_name, frames)
    with open(p) as f:
        preds = [json.loads(line) for line in f]
    if len(preds) != len(frames):
        raise RuntimeError(f"{p}: {len(preds)} predictions for {len(frames)} frames (unfinished?)")
    return preds


SVAN_SETS = ("svan_dev", "svan_dev_a", "svan_dev_b", "svan_dev10", "svan_dev5", "svan_test")


def _from_other_svan_caches(tag, set_name, frames):
    """A Svanström frame set can be served from complete caches of the same
    configuration on other Svanström sets (e.g. dev = dev_a + dev_b), keyed by image."""
    have = {}
    for other in SVAN_SETS:
        q = cache_path(tag, other)
        if other == set_name or not q.exists():
            continue
        fr = frame_set(other)
        with open(q) as f:
            pr = [json.loads(line) for line in f]
        if len(pr) == len(fr):
            have.update({_key(a): b for a, b in zip(fr, pr)})
    missing = sum(1 for f in frames if _key(f) not in have)
    if missing:
        raise FileNotFoundError(f"{cache_path(tag, set_name)}: no cache ({missing} of {len(frames)} frames not covered by other caches)")
    return [have[_key(f)] for f in frames]
