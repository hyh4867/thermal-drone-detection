"""Stage 2: freeze the Svanström video-level split (dev-A / dev-B / test).

Stage 1 already reported results on every Svanström video, so none of them is
"unseen" any more. Stage 2 therefore splits the 360 usable IR videos BY VIDEO,
stratified by class x distance bin, into

  test   (50%)  final evaluation, each frozen configuration once
  dev_b  (25%)  selection / calibration (thresholds, input scale, tracker, checkpoints)
  dev_a  (25%)  adaptation pool (the only Svanström videos a model may be trained on);
                also usable for selection of training-free settings

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.split
Writes docs/splits/svanstrom_stage2_split.json (committed, never regenerated with
another seed). Uses only the manifest's video names, classes and bins - no labels,
no predictions.
"""
from __future__ import annotations

import collections
import hashlib
import json

import numpy as np

from ..common import ROOT, read_json

SEED = 20261119
OUT = ROOT / "docs" / "splits" / "svanstrom_stage2_split.json"


def make_split(videos: dict, seed: int = SEED) -> dict:
    """videos: name -> (cls, bin). Returns name -> split."""
    strata = collections.defaultdict(list)
    for name, key in videos.items():
        strata[key].append(name)
    rng = np.random.default_rng(seed)
    out = {}
    for key in sorted(strata):
        names = sorted(strata[key])
        order = [names[i] for i in rng.permutation(len(names))]
        n = len(order)
        n_test = int(n * 0.5 + 0.5)
        n_b = (n - n_test + 1) // 2
        for i, name in enumerate(order):
            out[name] = "test" if i < n_test else "dev_b" if i < n_test + n_b else "dev_a"
    return out


def load_split() -> dict:
    return {v["video"]: v["split"] for v in read_json(OUT)["videos"]}


def main():
    man = read_json(ROOT / "data" / "manifests" / "svanstrom_ir.json")
    videos = {}
    for f in man:
        videos[f["video"]] = (f["cls"], f["bin"])
    split = make_split(videos)
    rows = [{"video": v, "cls": videos[v][0], "bin": videos[v][1], "split": split[v]} for v in sorted(videos)]
    counts = collections.Counter((r["cls"], r["bin"], r["split"]) for r in rows)
    table = {f"{c}/{b}": {s: counts[(c, b, s)] for s in ("dev_a", "dev_b", "test")}
             for c, b in sorted({(r["cls"], r["bin"]) for r in rows})}
    listing = "\n".join(f"{r['video']} {r['split']}" for r in rows)
    doc = {
        "seed": seed_note(), "fractions": {"test": 0.5, "dev_b": 0.25, "dev_a": 0.25},
        "stratified_by": ["cls", "bin"], "n_videos": len(rows),
        "counts_by_stratum": table,
        "counts_by_split": dict(collections.Counter(r["split"] for r in rows)),
        "listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "videos": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("n_videos", "counts_by_split", "counts_by_stratum", "listing_sha256")}, indent=1))


def seed_note():
    return SEED


if __name__ == "__main__":
    main()
