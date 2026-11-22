"""Check that every prediction cache used by the final evaluation comes from the expected configuration.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.verify_caches [--frames 5]
  -> results/stage2/cache_check.json   (exit code 1 if any cache fails)

The caches are otherwise only checked by frame count. For each cache this compares
  1. the expected (checkpoint, scale, tile, TTA) from frozen_config.json with the cache's meta.json,
  2. the same with the DONE line of the queue log,
  3. line count == frames of the set, and every line parses as a list of [x1, y1, x2, y2, score],
  4. a few frames inferred again with the expected configuration against the cached lines, and against the
     other caches of the same frame set (the cache's own lines must be the closest ones),
and records the sha256 and modification time of each checkpoint (must be older than the freeze).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ..common import ROOT, load_config, set_seed
from .data import PREDS, cache_path, frame_set
from .infer import Predictor

OUT = ROOT / "results/stage2"
LOGS = OUT / "logs"
S1_CKPT = "runs/full_frcnn/best.pt"
TOP = 5            # scores compared per frame
TOL = 0.01         # mean |score difference| allowed between a cache and its re-inference (MPS is not bit-exact)


def expected():
    """(tag, set) -> (checkpoint, scale, tile, tta, used by)."""
    fz = json.loads((OUT / "frozen_config.json").read_text())
    exp = {}
    for name, m in fz["models"].items():
        ck = m["checkpoint"]
        exp[(m["svanstrom_tag"], "svan_test")] = (ck, 1.0, 1, True, f"{name}, Svanström test")
        nt = m["native_tag"]
        for s, what in (("auv_val", "Anti-UAV val 임계값"), ("auv_test", "Anti-UAV test"), ("auv_test_small", "Anti-UAV 작은 표적 보충")):
            exp[(nt, s)] = (ck, 1.0, 1, True, f"{name}, {what}")
        if name in ("A", "B", "C37"):
            for s, what in (("cst_test", "CST test"), ("auv_track", "Anti-UAV 추적 표본")):
                exp[(nt, s)] = (ck, 1.0, 1, True, f"{name}, {what}")
    for s, what in (("cst_test", "CST test"), ("auv_track", "Anti-UAV 추적 표본")):
        exp[("s1_n", s)] = (S1_CKPT, 1.0, 1, False, f"S1, {what}")
    return fz, exp


def log_done(tag, set_name):
    for p in sorted(LOGS.glob("*.log")):
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("DONE {"):
                d = json.loads(line[5:])
                if d["tag"] == tag and d["set"] == set_name:
                    return p.name, d
    return None, None


def read_lines(path):
    """All lines of a cache; returns (rows per line, number of bad lines, ends with newline)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    fz, exp = expected()
    freeze_time = dt.datetime.fromisoformat(fz["meta"]["timestamp_utc"])
    cfg = load_config("configs/full.yaml")
    set_seed(cfg["seed"])

    ckpts = {}
    for ck in sorted({v[0] for v in exp.values()}):
        p = ROOT / ck
        mt = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
        ckpts[ck] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size,
                     "modified_utc": mt.isoformat(timespec="seconds"), "older_than_freeze": mt < freeze_time}

    caches = {}       # set -> {tag: rows} for every cache file of the sets we touch (the comparison pool)
    sets = sorted({s for _, s in exp})
    frames = {s: frame_set(s) for s in sets}
    for s in sets:
        caches[s] = {p.name.split("__")[0]: read_lines(p) for p in sorted(PREDS.glob(f"*__{s}.jsonl"))}

    rows, ok_all = [], True
    by_cfg = {}
    for key, v in exp.items():
        by_cfg.setdefault(v[:4], []).append(key)
    for (ck, scale, tile, tta), keys in sorted(by_cfg.items()):
        pred = Predictor(cfg, ROOT / ck, scale, tile, tta)
        for tag, s in sorted(keys):
            r = {"tag": tag, "set": s, "used_for": exp[(tag, s)][4],
                 "expected": {"ckpt": ck, "scale": scale, "tile": tile, "tta": tta}}
            path = cache_path(tag, s)
            n = len(frames[s])
            if not path.exists():
                r.update(status="missing")
                rows.append(r)
                ok_all = False
                continue
            lines, bad, nl = caches[s][tag]
            mp = path.with_suffix(".meta.json")
            meta = json.loads(mp.read_text()) if mp.exists() else None
            same = lambda d: bool(d) and d["tag"] == tag and d["set"] == s and d["ckpt"] == ck and float(d["scale"]) == scale \
                and d["tile"] == tile and d["tta"] == tta and d["frames"] == n
            logname, done = log_done(tag, s)
            idx = sorted({int(round(q * (n - 1))) for q in np.linspace(0, 1, a.frames)})
            if (tag, s) == ("s1_ntta", "auv_track"):      # the file that was being written at the reboot
                idx = sorted(set(idx) | set(range(8612, 8624)))
            idx = [i for i in idx if i < len(lines) and lines[i] is not None]
            again = {i: top_scores(pred(frames[s][i])) for i in idx}
            dist = {t: float(np.mean([np.abs(again[i] - top_scores(c[0][i])).mean() for i in idx]))
                    for t, c in caches[s].items() if all(i < len(c[0]) and c[0][i] is not None for i in idx)}
            own = dist[tag]
            others = {t: d for t, d in dist.items() if t != tag}
            nearest_other = min(others.items(), key=lambda kv: kv[1]) if others else (None, None)
            r.update(lines=len(lines), frames=n, bad_lines=bad, ends_with_newline=nl,
                     meta_matches=same(meta), log=logname, log_matches=same(done),
                     recheck_frames=len(idx), recheck_own_diff=own,
                     recheck_nearest_other=nearest_other[0], recheck_nearest_other_diff=nearest_other[1],
                     checkpoint_sha256=ckpts[ck]["sha256"][:12])
            r["status"] = "ok" if (len(lines) == n and bad == 0 and nl and r["meta_matches"] and r["log_matches"]
                                   and own < TOL and (nearest_other[1] is None or own < nearest_other[1])
                                   and ckpts[ck]["older_than_freeze"]) else "FAIL"
            ok_all &= r["status"] == "ok"
            rows.append(r)
            print(f"{r['status']:4} {tag}__{s}: lines {len(lines)}/{n}, bad {bad}, meta {r['meta_matches']}, log {r['log_matches']}, "
                  f"own {own:.5f}, nearest other {nearest_other[0]} {nearest_other[1] if nearest_other[1] is None else round(nearest_other[1], 5)}", flush=True)
        del pred

    out = {"all_ok": bool(ok_all), "tolerance_mean_abs_score_diff": TOL, "scores_per_frame": TOP,
           "freeze_timestamp_utc": fz["meta"]["timestamp_utc"], "checkpoints": ckpts, "caches": rows,
           "note": "recheck_*: frames inferred again with the expected configuration; diff = mean |difference| of the top "
                   f"{TOP} scores per frame, against the cache itself (own) and against the closest other cache of the same frame set."}
    (OUT / "cache_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("ALL OK" if ok_all else "FAILED - do not aggregate", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
