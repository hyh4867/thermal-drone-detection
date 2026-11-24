"""Seed study: check that the prediction caches of one seed come from the expected model (stage 3 method).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.seeds.verify_caches --seed 1 --scope hold|test [--frames 5]
  -> results/seeds/cache_check__seed<k>_<scope>.json   (exit code 1 if a cache fails: do not aggregate)

Same four checks as tdd.stage3.verify_caches: (1) meta.json equals the expectation and its checkpoint sha256
equals the file on disk now; (2) the DONE line of the run log says the same; (3) line count == frames and every
line parses; (4) a few frames inferred again match the cache (mean |difference| of the top 5 scores < 0.01) and
are closer to it than to any other cache of the same frame set. The "other" caches include the stage 3 caches
(seed 0 and S1) and the other seed, so a cache written by the wrong checkpoint is caught.
Scope test: every test set whose cache file exists; a missing one is listed as "not run" (no failure).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ..common import ROOT
from ..stage3 import data as d3
from ..stage3 import infer as base
from ..stage3.verify_caches import TOL, TOP, read_lines, top_scores
from . import data as D
from .infer import register


def log_done(tag, set_name):
    for p in sorted((D.OUT / "logs").glob("*.log")):
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("DONE {"):
                d = json.loads(line[5:])
                if d.get("tag") == tag and d.get("set") == set_name:
                    return p.name, d
    return None, None


def check(seed: int, set_name: str, n_frames: int = 5) -> dict:
    tag, model = D.tag(seed), D.model(seed)
    r = {"tag": tag, "set": set_name, "seed": seed}
    own_p = D.cache_path(tag, set_name)
    if not own_p.exists():
        return {**r, "status": "not run"}
    frames = D.frame_set(set_name)
    n = len(frames)
    files = sorted(D.PREDS.glob(f"*__{set_name}.jsonl")) + sorted(d3.PREDS.glob(f"*__{set_name}.jsonl"))
    pool = {p.name.split("__")[0]: read_lines(p) for p in files}
    ckpt, anchors, _ = base.model_spec(model)
    scale = D.M * (2.0 if d3.is_svanstrom(set_name) else 1.0)
    sha = base.sha256_file(ROOT / ckpt)
    exp = {"model": model, "m": float(D.M), "ckpt": ckpt, "anchor_sizes": list(anchors), "scale": scale, "tile": 1, "tta": False}
    lines, bad, nl = pool[tag]
    mp = own_p.with_suffix(".meta.json")
    meta = json.loads(mp.read_text()) if mp.exists() else None
    same = lambda d: bool(d) and all(d.get(k) == v for k, v in exp.items()) and d.get("frames") == n and d.get("tag") == tag and d.get("set") == set_name
    logname, done = log_done(tag, set_name)
    idx = sorted({int(round(q * (n - 1))) for q in np.linspace(0, 1, n_frames)})
    if meta and meta.get("resumed_from"):
        rp = meta["resumed_from"]
        idx = sorted(set(idx) | {i for i in range(rp - 6, rp + 6) if 0 <= i < n})
    idx = [i for i in idx if i < len(lines) and lines[i] is not None]
    predict, _ = base.make_predictor(model, float(D.M), set_name)
    again = {i: top_scores(predict(frames[i])) for i in idx}
    del predict
    dist = {t: float(np.mean([np.abs(again[i] - top_scores(c[0][i])).mean() for i in idx]))
            for t, c in pool.items() if all(i < len(c[0]) and c[0][i] is not None for i in idx)}
    own = dist.get(tag, float("inf"))
    others = {t: d for t, d in dist.items() if t != tag}
    near = min(others.items(), key=lambda kv: kv[1]) if others else (None, None)
    frozen_sha = D.frozen(seed)["checkpoint_sha256"] if D.frozen_path(seed).exists() else None
    r.update(expected=exp, lines=len(lines), frames=n, bad_lines=bad, ends_with_newline=nl, meta_matches=same(meta),
             meta_sha_matches_file=bool(meta) and meta.get("ckpt_sha256") == sha, resumed_from=(meta or {}).get("resumed_from"),
             log=logname, log_matches=same(done), recheck_frames=len(idx), recheck_own_diff=own,
             recheck_nearest_other=near[0], recheck_nearest_other_diff=near[1], recheck_others=others,
             checkpoint_sha256=sha, sha_matches_frozen=None if frozen_sha is None else frozen_sha == sha)
    good = (len(lines) == n and bad == 0 and nl and r["meta_matches"] and r["meta_sha_matches_file"] and r["log_matches"]
            and own < TOL and (near[1] is None or own < near[1]) and r["sha_matches_frozen"] is not False)
    if set_name in D.TEST_SETS:
        good = good and r["sha_matches_frozen"] is True
    r["status"] = "ok" if good else "FAIL"
    print(f"{r['status']:4} {tag}__{set_name}: lines {len(lines)}/{n}, bad {bad}, meta {r['meta_matches']}, sha {r['meta_sha_matches_file']}, "
          f"log {r['log_matches']}, frozen sha {r['sha_matches_frozen']}, own {own:.5f}, nearest other {near[0]} {None if near[1] is None else round(near[1], 5)}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=D.NEW_SEEDS)
    ap.add_argument("--scope", required=True, choices=["hold", "test"])
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    register()
    rows = [check(a.seed, s, a.frames) for s in ((D.HOLD,) if a.scope == "hold" else D.TEST_SETS)]
    ok = all(r["status"] in ("ok", "not run") for r in rows) and any(r["status"] == "ok" for r in rows)
    out = {"scope": a.scope, "seed": a.seed, "all_ok": bool(ok), "tolerance_mean_abs_score_diff": TOL, "scores_per_frame": TOP,
           "checkpoints_sha256": {D.ckpt(a.seed): base.sha256_file(ROOT / D.ckpt(a.seed))}, "caches": rows}
    (D.OUT / f"cache_check__seed{a.seed}_{a.scope}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("ALL OK" if ok else "FAILED - do not aggregate", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
