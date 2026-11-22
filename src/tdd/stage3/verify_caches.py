"""Stage 3: check that every prediction cache comes from the expected configuration (stage 2 method).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.verify_caches --scope selection|hold|final [--frames 5]
  -> results/stage3/cache_check__<scope>.json   (exit code 1 if any cache fails: do not aggregate)

The expected configuration of a cache is fixed by its tag "<model>_m<m>" (models in tdd.stage3.infer.MODELS).
For each cache: (1) meta.json equals the expectation (model, m, checkpoint path, anchors, scale, no tiles, no TTA)
and its checkpoint sha256 equals the file on disk now; (2) the DONE line of the queue log says the same;
(3) line count == frames of the set and every line parses as a list of [x1, y1, x2, y2, score];
(4) a few frames inferred again with the expected configuration match the cache (mean |difference| of the top 5
scores per frame < 0.01) and are closer to it than to any other cache of the same frame set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from ..common import ROOT
from .data import OUT, PREDS, cache_path, frame_set, is_svanstrom
from .infer import MODELS, make_predictor, model_spec, sha256_file

LOGS = OUT / "logs"
TOP, TOL = 5, 0.01
SCOPES = {"selection": ("svan_dev20", "auv_val", "auv_val_q"), "hold": ("cst_tr_hold",),
          "final": ("cst_val", "auv_test", "auv_test_small"),
          "reference": ("cst_val",),
          "svanstrom": ("svan_test_h",)}    # addendum 6     # addendum 5: the reference row; written to its own file, the final check is kept
NAME = {m.lower(): m for m in MODELS}


def read_lines(path):
    raw = path.read_bytes()
    rows, bad = [], 0
    for line in raw.split(b"\n")[:-1] if raw.endswith(b"\n") else raw.split(b"\n"):
        try:
            r = json.loads(line)
            assert isinstance(r, list) and all(isinstance(b, list) and len(b) == 5 for b in r)
            rows.append(r)
        except Exception:
            rows.append(None)
            bad += 1
    return rows, bad, raw.endswith(b"\n")


def top_scores(row):
    s = sorted((b[4] for b in row), reverse=True)[:TOP]
    return np.array(s + [0.0] * (TOP - len(s)))


def log_done(tag, set_name):
    for p in sorted(LOGS.glob("*.log")):
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("DONE {"):
                d = json.loads(line[5:])
                if d.get("tag") == tag and d.get("set") == set_name:
                    return p.name, d
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True, choices=sorted(SCOPES))
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    rows, ok_all, shas = [], True, {}
    required = set()
    if a.scope == "final":
        fz = json.loads((OUT / "frozen_config.json").read_text())
        for s, names in fz["final_sets"].items():
            for n in names:
                c = fz["configs"].get(n)
                if c and not c.get("duplicate_of"):
                    required.add((c["tag"], s))
    if a.scope == "reference":
        required.add(("b_m1", "cst_val"))
    if a.scope == "svanstrom":
        required |= {("s1_m1", "svan_test_h"), ("t2_20_m1", "svan_test_h")}
    for s in SCOPES[a.scope]:
        files = sorted(PREDS.glob(f"*__{s}.jsonl"))
        if not files and not any(r[1] == s for r in required):
            continue
        frames = frame_set(s)
        n = len(frames)
        pool = {p.name.split("__")[0]: read_lines(p) for p in files}
        for tag in sorted(set(pool) | {t for t, ss in required if ss == s}):
            r = {"tag": tag, "set": s}
            mm = re.fullmatch(r"(.+)_m(\d+)", tag)
            if tag not in pool or not mm or mm.group(1) not in NAME:
                r["status"] = "FAIL (missing cache)" if tag not in pool else "FAIL (unknown tag)"
                rows.append(r)
                ok_all = False
                print(r["status"], tag, s, flush=True)
                continue
            model, m = NAME[mm.group(1)], float(mm.group(2))
            ckpt, anchors, _ = model_spec(model)
            scale = m * (2.0 if is_svanstrom(s) else 1.0)
            shas.setdefault(ckpt, sha256_file(ROOT / ckpt))
            exp = {"model": model, "m": m, "ckpt": ckpt, "anchor_sizes": list(anchors), "scale": scale, "tile": 1, "tta": False}
            lines, bad, nl = pool[tag]
            mp = cache_path(tag, s).with_suffix(".meta.json")
            meta = json.loads(mp.read_text()) if mp.exists() else None
            same = lambda d: bool(d) and all(d.get(k) == v for k, v in exp.items()) and d.get("frames") == n and d.get("tag") == tag and d.get("set") == s
            logname, done = log_done(tag, s)
            idx = sorted({int(round(q * (n - 1))) for q in np.linspace(0, 1, a.frames)})
            if meta and meta.get("resumed_from"):       # a resumed cache: also check the frames around the resume point
                rp = meta["resumed_from"]
                idx = sorted(set(idx) | {i for i in range(rp - 6, rp + 6) if 0 <= i < n})
            idx = [i for i in idx if i < len(lines) and lines[i] is not None]
            predict, _ = make_predictor(model, m, s)
            again = {i: top_scores(predict(frames[i])) for i in idx}
            del predict
            dist = {t: float(np.mean([np.abs(again[i] - top_scores(c[0][i])).mean() for i in idx]))
                    for t, c in pool.items() if all(i < len(c[0]) and c[0][i] is not None for i in idx)}
            own = dist.get(tag, float("inf"))
            others = {t: d for t, d in dist.items() if t != tag}
            near = min(others.items(), key=lambda kv: kv[1]) if others else (None, None)
            r.update(required=(tag, s) in required, expected=exp, lines=len(lines), frames=n, bad_lines=bad, ends_with_newline=nl,
                     meta_matches=same(meta), meta_sha_matches_file=bool(meta) and meta.get("ckpt_sha256") == shas[ckpt],
                     resumed_from=(meta or {}).get("resumed_from"), log=logname, log_matches=same(done),
                     recheck_frames=len(idx), recheck_own_diff=own, recheck_nearest_other=near[0], recheck_nearest_other_diff=near[1],
                     checkpoint_sha256=shas[ckpt][:12])
            good = (len(lines) == n and bad == 0 and nl and r["meta_matches"] and r["meta_sha_matches_file"] and r["log_matches"]
                    and own < TOL and (near[1] is None or own < near[1]))
            r["status"] = "ok" if good else "FAIL"
            ok_all &= good
            rows.append(r)
            print(f"{r['status']:4} {tag}__{s}: lines {len(lines)}/{n}, bad {bad}, meta {r['meta_matches']}, sha {r['meta_sha_matches_file']}, "
                  f"log {r['log_matches']}, own {own:.5f}, nearest other {near[0]} {None if near[1] is None else round(near[1], 5)}", flush=True)
    out = {"scope": a.scope, "all_ok": bool(ok_all), "tolerance_mean_abs_score_diff": TOL, "scores_per_frame": TOP,
           "checkpoints_sha256": shas, "caches": rows}
    (OUT / f"cache_check__{a.scope}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("ALL OK" if ok_all else "FAILED - do not aggregate", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
