"""Stage 3 frame sets and prediction caches (pre-registration 3, sections 3-4).

Predictions are lists of [x1, y1, x2, y2, score] in the coordinates of the frame record
(for the reduced val set: the 160x128 reduced image).
"""
from __future__ import annotations

import copy
import json

from ..common import ROOT, read_json
from ..stage2 import data as d2

PREDS = ROOT / "runs" / "stage3" / "preds"
OUT = ROOT / "results" / "stage3"
SPLIT = ROOT / "docs" / "splits" / "cst_stage3_split.json"
QUARTER = 0.25

CST_TRAIN_SETS = ("cst_tr_hold", "cst_tr_20", "cst_tr_5")
# sets a track 1 configuration may be run on BEFORE the freeze (no CST data)
TRACK1_SELECTION_SETS = ("svan_dev20", "auv_val", "auv_val_q")


def frame_set(name: str):
    if name == "auv_val_q":
        # Anti-UAV val reduced to 1/4 (160x128): the image is reduced at load time (pre_scale), boxes here
        out = copy.deepcopy(d2.frame_set("auv_val"))
        for f in out:
            f["pre_scale"] = QUARTER
            f["gt"] = [[v * QUARTER for v in b] for b in f["gt"]]
            f["width"], f["height"] = int(f["width"] * QUARTER), int(f["height"] * QUARTER)
        return out
    if name == "svan_dev20":          # addendum 2: every other frame of svan_dev10
        return [f for f in d2.frame_set("svan_dev10") if f["frame"] % 20 == 0]
    if name == "cst_val":
        return read_json(d2.MAN / "cst_val.json")
    if name in CST_TRAIN_SETS:
        seqs = set(read_json(SPLIT)[name])
        return [f for f in read_json(d2.MAN / "cst_train.json") if f["seq"] in seqs]
    if name == "cst_test_h":          # every other stage 2 CST test frame (stride 8)
        return [f for f in d2.frame_set("cst_test") if f["frame"] % 8 == 0]
    if name == "svan_test_h":         # every other stage 2 Svanström test frame (stride 10)
        return [f for f in d2.frame_set("svan_test") if f["frame"] % 10 == 0]
    return d2.frame_set(name)


def is_svanstrom(name: str) -> bool:
    return name.startswith("svan")


def cache_path(tag: str, set_name: str):
    return PREDS / f"{tag}__{set_name}.jsonl"


def load_preds(tag: str, set_name: str):
    frames = frame_set(set_name)
    with open(cache_path(tag, set_name)) as f:
        preds = [json.loads(line) for line in f]
    if len(preds) != len(frames):
        raise RuntimeError(f"{tag}__{set_name}: {len(preds)} predictions for {len(frames)} frames (unfinished?)")
    return preds


def stage2_subset(tag: str, full_set: str, sub_set: str):
    """Stage 2 predictions of ``tag`` on ``full_set`` cut down to the frames of ``sub_set`` (no new inference)."""
    full = d2.frame_set(full_set)
    preds = d2.load_preds(tag, full_set)
    idx = {f["image"]: p for f, p in zip(full, preds)}
    return [idx[f["image"]] for f in frame_set(sub_set)]
