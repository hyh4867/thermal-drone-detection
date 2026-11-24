"""Seed study evaluation (docs/PREREGISTRATION_SEEDS.md, sections 4-5). Detector only; same data, metrics and
bootstrap as stage 3 (tdd.stage3.final_eval for CST val and Anti-UAV test, tdd.stage3.svan_eval for Svanström).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.seeds.evaluate --seed 0|1|2   -> results/seeds/eval_seed<k>.json
       PYTHONPATH=src .venv/bin/python -m tdd.seeds.evaluate --summary      -> results/seeds/summary.json
Refuses to run while the working tree is dirty. Seeds 1 and 2: the frozen file must be committed and the cache
check of the seed must list every cache used as ok. Seed 0 is re-computed from the stage 3 caches with the frozen
stage 3 threshold and must reproduce the stage 3 files exactly (results/stage3 is only read).
A part whose cache does not exist is written as "not run".
"""
from __future__ import annotations

import argparse
import math

from ..common import load_config, read_json, run_metadata, write_json
from ..stage2 import data as d2
from ..stage2.evalset import SvanEval
from ..stage3 import data as d3
from ..stage3.evalx import Ev3, bootstrap3
from ..stage3.final_eval import J3_LINE, J5
from ..stage3.svan_eval import row as svan_row
from . import data as D


def _cache_ok(seed, set_name):
    if seed == 0:
        return True            # stage 3 caches: checked in results/stage3/cache_check__final.json and cache_check__svanstrom.json
    p = D.OUT / f"cache_check__seed{seed}_test.json"
    if not p.exists():
        return False
    chk = read_json(p)
    return any(c["tag"] == D.tag(seed) and c["set"] == set_name and c["status"] == "ok" for c in chk["caches"])


def _preds(seed, set_name):
    """Predictions of one seed, or a reason why the part is not run."""
    if not D.cache_path(D.tag(seed), set_name).exists():
        return None, "no cache"
    try:
        p = D.load_preds(D.tag(seed), set_name)
    except RuntimeError as e:
        return None, str(e)[:200]
    if not _cache_ok(seed, set_name):
        raise SystemExit(f"refused: cache {D.tag(seed)}__{set_name} has no passed cache check")
    return p, None


def cst_val(seed, fz):
    preds, why = _preds(seed, "cst_val")
    if preds is None:
        return {"status": "not run", "detail": why}
    frames = D.frame_set("cst_val")
    s1 = D.s1()
    ref = Ev3(frames, D.load_preds(s1["tag"], "cst_val"), s1["threshold"])
    ev = Ev3(frames, preds, fz["threshold"])
    r = {"status": "ok", "set": "cst_val", "frames": len(frames), "sequences": len({f["seq"] for f in frames}),
         "reference": "S1_m1 (stage 3 cache s1_m1__cst_val, stage 1 threshold); paired_diff is against it",
         "S1_m1": {"threshold": s1["threshold"], "ap50": ref.metrics()["ap50"]},
         "tag": D.tag(seed), **ev.metrics3(), **bootstrap3(ev, ref)}
    d = r["paired_diff"]["ap50"]
    r["J4"] = {"diff_ap50": d, "helped": d["ci95"][0] > 0, "ap50": r["ap50"], "line": J3_LINE, "still_low": r["ap50"] < J3_LINE}
    if seed != 0:               # descriptive: this seed against seed 0 on the same frames, each at its own threshold
        f0 = D.frozen(0)
        r["paired_diff_vs_seed0"] = bootstrap3(ev, Ev3(frames, D.load_preds(f0["tag"], "cst_val"), f0["threshold"]))["paired_diff"]
    return r


def antiuav(seed, fz):
    preds, why = _preds(seed, "auv_test")
    if preds is None:
        return {"status": "not run", "detail": why}
    frames = D.frame_set("auv_test")
    s1 = D.s1()["threshold"]
    ref = Ev3(frames, d2.load_preds("s1", "auv_test"), s1)          # stage 1 model and settings (stage 1 caches), as in stage 3
    ev = Ev3(frames, preds, fz["threshold"])
    r = {"status": "ok", "set": "auv_test", "frames": len(frames), "reference": "S1 stage 1 caches at the stage 1 threshold",
         "reference_S1_stage1": {k: v for k, v in ref.metrics().items() if not isinstance(v, dict)},
         "tag": D.tag(seed), **ev.metrics3(), **bootstrap3(ev, ref)}
    pd = r["paired_diff"]
    bad = {k: bool(pd[k]["diff"] < lim) if lim < 0 else bool(pd[k]["diff"] > lim) for k, lim in J5.items()}
    r["J5"] = {"diffs": {k: pd[k] for k in J5}, "limits": J5, "over_limit": bad, "regression": any(bad.values())}
    p2, why2 = _preds(seed, "auv_test_small")
    if p2 is None:
        r["recall_by_size_with_supplement"] = {"status": "not run", "detail": why2}
    else:
        f2 = D.frame_set("auv_test_small")
        seen = {f["image"] for f in frames}
        extra = [(f, p) for f, p in zip(f2, p2) if f["image"] not in seen]
        r["recall_by_size_with_supplement"] = Ev3(frames + [x[0] for x in extra], preds + [x[1] for x in extra],
                                                  fz["threshold"]).metrics()["recall_by_size"]
    return r


def svanstrom(seed, fz):
    preds, why = _preds(seed, "svan_test_h")
    if preds is None:
        return {"status": "not run", "detail": why}
    frames = D.frame_set("svan_test_h")
    s1 = D.s1()
    e1 = SvanEval(frames, D.load_preds(s1["tag"], "svan_test_h"), 1.0)
    e2 = SvanEval(frames, preds, 1.0)
    return {"status": "ok", "set": "svan_test_h", "frames": len(frames), "videos": len({f["video"] for f in frames}),
            "reference": "S1_m1 (stage 3 cache s1_m1__svan_test_h, stage 1 threshold); Svanström input enlarged 2x; k = 1",
            "S1_m1": {k: v for k, v in e1.metrics(s1["threshold"]).items() if not isinstance(v, dict)},
            "tag": D.tag(seed), **svan_row(e2, fz["threshold"], e1, s1["threshold"])}


def _check_seed0(res):
    """Seed 0 must reproduce the stage 3 files (same caches, same threshold, same code path)."""
    c = read_json(d3.OUT / "final__cst_val.json")
    a = read_json(d3.OUT / "final__antiuav.json")
    s = read_json(d3.OUT / "svanstrom_test.json")
    pairs = []
    for part, old, keys in (("cst_val", c["configs"]["T2_20"], ("ap50", "recall_iou50", "recall_center", "precision", "far_drone_free")),
                            ("antiuav", a["configs"]["T2_20"], ("ap50", "recall_iou50", "recall_center", "precision", "far_drone_free")),
                            ("svanstrom", s["configs"]["T2_20"], ("recall_iou50", "recall_center", "far"))):
        new = res[part]
        assert new["status"] == "ok", part
        pairs += [(f"{part}.{k}", new[k], old[k]) for k in keys]
        pairs += [(f"{part}.paired_diff.{k}.{j}", new["paired_diff"][k]["ci95"][j], old["paired_diff"][k]["ci95"][j])
                  for k in old["paired_diff"] for j in (0, 1)]
    pairs.append(("J4.diff", res["cst_val"]["J4"]["diff_ap50"]["diff"], c["verdicts"]["J4_T2_20"]["diff_ap50"]["diff"]))
    pairs.append(("J4.helped", res["cst_val"]["J4"]["helped"], c["verdicts"]["J4_T2_20"]["helped"]))
    pairs.append(("J5.regression", res["antiuav"]["J5"]["regression"], a["verdicts"]["J5_T2_20"]["regression"]))
    worst = max(abs(float(x) - float(y)) for _, x, y in pairs)
    bad = [n for n, x, y in pairs if abs(float(x) - float(y)) > 1e-12]
    if bad:
        raise SystemExit(f"seed 0 does not reproduce stage 3: {bad}")
    return {"compared_values": len(pairs), "max_abs_diff": worst, "matches": True,
            "files": ["results/stage3/final__cst_val.json", "results/stage3/final__antiuav.json", "results/stage3/svanstrom_test.json"]}


def one_seed(seed, meta):
    if seed != 0 and not D.frozen_committed(seed):
        raise SystemExit(f"refused: results/seeds/frozen_seed{seed}.json is not committed")
    fz = D.frozen(seed)
    res = {"seed": seed, "model": fz["model"], "m": fz["m"], "tag": D.tag(seed), "threshold": fz["threshold"],
           "checkpoint_sha256": fz["checkpoint_sha256"], "hold": fz["hold"],
           "frozen": fz.get("source", f"results/seeds/frozen_seed{seed}.json"),
           "note": "test data already used in stage 3 (CST val, Anti-UAV test, Svanström test); nothing is selected on them",
           "cst_val": cst_val(seed, fz), "antiuav": antiuav(seed, fz), "svanstrom": svanstrom(seed, fz)}
    if seed == 0:
        res["reproduces_stage3"] = _check_seed0(res)
    res["meta"] = meta
    write_json(res, D.OUT / f"eval_seed{seed}.json")
    c, a, s = res["cst_val"], res["antiuav"], res["svanstrom"]
    print(f"== seed {seed}, threshold {fz['threshold']:.4f}")
    if c["status"] == "ok":
        print(f"  CST val: AP {c['ap50']:.4f} {c['ci95']['ap50']} diff {c['J4']['diff_ap50']} helped {c['J4']['helped']} "
              f"tiny centre {c['recall_by_size']['tiny']['center']:.4f} FAR {c['far_drone_free']:.4f}")
    if a["status"] == "ok":
        print(f"  Anti-UAV: AP {a['ap50']:.4f} FAR {a['far_drone_free']:.4f} J5 {a['J5']['over_limit']} regression {a['J5']['regression']}")
    if s["status"] == "ok":
        print(f"  Svanström: FAR {s['far']:.4f} Rc {s['recall_center']:.4f} diff Rc {s['paired_diff']['recall_center']}")


# (name, part, getter) of the numbers summarised over seeds
ROWS = [
    ("hold.threshold", None, lambda r: r["threshold"]),
    ("hold.ap50", None, lambda r: r["hold"]["ap50"]),
    ("cst_val.ap50", "cst_val", lambda p: p["ap50"]),
    ("cst_val.ap50_minus_S1", "cst_val", lambda p: p["paired_diff"]["ap50"]["diff"]),
    ("cst_val.recall_iou50", "cst_val", lambda p: p["recall_iou50"]),
    ("cst_val.recall_center", "cst_val", lambda p: p["recall_center"]),
    ("cst_val.precision", "cst_val", lambda p: p["precision"]),
    ("cst_val.far_drone_free", "cst_val", lambda p: p["far_drone_free"]),
    ("cst_val.recall_tiny_iou50", "cst_val", lambda p: p["recall_by_size"]["tiny"]["iou50"]),
    ("cst_val.recall_tiny_center", "cst_val", lambda p: p["recall_by_size"]["tiny"]["center"]),
    ("cst_val.recall_small_iou50", "cst_val", lambda p: p["recall_by_size"]["small"]["iou50"]),
    ("cst_val.recall_small_center", "cst_val", lambda p: p["recall_by_size"]["small"]["center"]),
    ("antiuav.ap50", "antiuav", lambda p: p["ap50"]),
    ("antiuav.ap50_minus_S1", "antiuav", lambda p: p["paired_diff"]["ap50"]["diff"]),
    ("antiuav.recall_iou50", "antiuav", lambda p: p["recall_iou50"]),
    ("antiuav.recall_iou50_minus_S1", "antiuav", lambda p: p["paired_diff"]["recall_iou50"]["diff"]),
    ("antiuav.far_drone_free", "antiuav", lambda p: p["far_drone_free"]),
    ("antiuav.far_drone_free_minus_S1", "antiuav", lambda p: p["paired_diff"]["far_drone_free"]["diff"]),
    ("svanstrom.far", "svanstrom", lambda p: p["far"]),
    ("svanstrom.far_minus_S1", "svanstrom", lambda p: p["paired_diff"]["far"]["diff"]),
    ("svanstrom.recall_center", "svanstrom", lambda p: p["recall_center"]),
    ("svanstrom.recall_center_minus_S1", "svanstrom", lambda p: p["paired_diff"]["recall_center"]["diff"]),
    ("svanstrom.recall_iou50", "svanstrom", lambda p: p["recall_iou50"]),
]


def spread(by_seed: dict) -> dict:
    v = [by_seed[k] for k in sorted(by_seed)]
    n = len(v)
    mean = sum(v) / n
    out = {"by_seed": {str(k): by_seed[k] for k in sorted(by_seed)}, "n": n, "mean": mean, "min": min(v), "max": max(v), "range": max(v) - min(v)}
    out["sd_n_minus_1"] = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else None
    return out


EPS = 1e-9      # limits are inclusive; guards against binary rounding of a value that sits on a limit


def predictions(tab, verdicts):
    """Pre-registered predictions (section 6) checked mechanically. None = cannot be judged (a seed is missing)."""
    def vals(name, seeds):
        b = tab.get(name, {}).get("by_seed", {})
        return [b[str(s)] for s in seeds] if all(str(s) in b for s in seeds) else None

    def all_in(name, seeds, lo, hi):
        v = vals(name, seeds)
        return None if v is None else all(lo - EPS <= x <= hi + EPS for x in v)

    def rng_le(name, lim):
        v = vals(name, D.SEEDS)
        return None if v is None else (max(v) - min(v)) <= lim + EPS

    def verdict_all(key, seeds, want=True):
        v = [verdicts.get(str(s), {}).get(key) for s in seeds]
        return None if any(x is None for x in v) else all(x is want for x in v)

    new = D.NEW_SEEDS
    return {
        "reviewer": {
            "R1 CST val AP@0.5 of seeds 1 and 2 each in 0.33-0.50": all_in("cst_val.ap50", new, 0.33, 0.50),
            "R2 range of CST val AP@0.5 over 3 seeds <= 0.08": rng_le("cst_val.ap50", 0.08),
            "R3 all 3 seeds regress on Anti-UAV by the alarm-rate criterion (diff > +0.02)":
                (lambda v: None if v is None else all(x > J5["far_drone_free"] for x in v))(vals("antiuav.far_drone_free_minus_S1", D.SEEDS)),
            "R4 all 3 seeds: Svanström bird/airplane/helicopter alarm rate >= 0.60": all_in("svanstrom.far", D.SEEDS, 0.60, 1.0),
        },
        "agent": {
            "A1 CST val AP@0.5 of seeds 1 and 2 each in 0.37-0.46": all_in("cst_val.ap50", new, 0.37, 0.46),
            "A2 range of CST val AP@0.5 over 3 seeds <= 0.06": rng_le("cst_val.ap50", 0.06),
            "A3 J4 helped and AP >= 0.30 for seeds 1 and 2": (lambda h, l: None if h is None or l is None else h and l)(
                verdict_all("J4_helped", new), verdict_all("J4_still_low", new, False)),
            "A4 CST val tiny centre recall of seeds 1 and 2 each in 0.40-0.54": all_in("cst_val.recall_tiny_center", new, 0.40, 0.54),
            "A5 hold-out threshold of seeds 1 and 2 each in 0.88-0.96": all_in("hold.threshold", new, 0.88, 0.96),
            "A6 Anti-UAV AP@0.5 difference to S1 of seeds 1 and 2 each in -0.04..-0.005": all_in("antiuav.ap50_minus_S1", new, -0.04, -0.005),
            "A7 J5 regression for seeds 1 and 2": verdict_all("J5_regression", new),
            "A8 Anti-UAV alarm rate of seeds 1 and 2 each in 0.05-0.10": all_in("antiuav.far_drone_free", new, 0.05, 0.10),
            "A9 Svanström alarm rate of seeds 1 and 2 each in 0.60-0.85": all_in("svanstrom.far", new, 0.60, 0.85),
            "A10 range of Svanström alarm rate over 3 seeds <= 0.15": rng_le("svanstrom.far", 0.15),
            "A11 Svanström centre recall difference to S1 positive for seeds 1 and 2": all_in("svanstrom.recall_center_minus_S1", new, 1e-12, 1.0),
        },
    }


def summary(meta):
    evs = {s: read_json(D.OUT / f"eval_seed{s}.json") for s in D.SEEDS if (D.OUT / f"eval_seed{s}.json").exists()}
    tab = {}
    for name, part, get in ROWS:
        by = {s: get(r if part is None else r[part]) for s, r in evs.items() if part is None or r[part]["status"] == "ok"}
        if by:
            tab[name] = spread(by)
    verdicts = {}
    for s, r in evs.items():
        v = {}
        if r["cst_val"]["status"] == "ok":
            v.update(J4_helped=r["cst_val"]["J4"]["helped"], J4_still_low=r["cst_val"]["J4"]["still_low"],
                     J4_diff_ap50=r["cst_val"]["J4"]["diff_ap50"])
        if r["antiuav"]["status"] == "ok":
            v.update(J5_regression=r["antiuav"]["J5"]["regression"], J5_over_limit=r["antiuav"]["J5"]["over_limit"])
        verdicts[str(s)] = v
    res = {"seeds_evaluated": sorted(evs), "seeds_planned": list(D.SEEDS),
           "parts": {str(s): {p: r[p]["status"] for p in ("cst_val", "antiuav", "svanstrom")} for s, r in evs.items()},
           "note": "spread over training seeds only (n <= 3): mean, min, max, range, sample sd; no interval is claimed for the seed spread. "
                   "Seed 0 = the stage 3 run. Test data were already used in stage 3; nothing was selected on them.",
           "metrics": tab, "verdicts": verdicts, "predictions": predictions(tab, verdicts), "meta": meta}
    write_json(res, D.OUT / "summary.json")
    for name, t in tab.items():
        print(f"{name:36} " + " ".join(f"s{k}={v:.4f}" for k, v in t["by_seed"].items()) + f" | mean {t['mean']:.4f} range {t['range']:.4f}")
    print("verdicts:", {s: {k: x for k, x in v.items() if not isinstance(x, dict)} for s, v in verdicts.items()})
    for who, p in res["predictions"].items():
        for k, x in p.items():
            print(f"  [{who}] {k}: {x}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int, choices=D.SEEDS)
    g.add_argument("--summary", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke tests only; the output is then marked git_dirty")
    a = ap.parse_args()
    meta = run_metadata(load_config("configs/full.yaml"), {"stage": "seeds"})
    if meta["git_dirty"] and not a.allow_dirty:
        raise SystemExit("refused: the working tree is dirty (commit first; results/ does not count)")
    D.OUT.mkdir(parents=True, exist_ok=True)
    summary(meta) if a.summary else one_seed(a.seed, meta)


if __name__ == "__main__":
    main()
