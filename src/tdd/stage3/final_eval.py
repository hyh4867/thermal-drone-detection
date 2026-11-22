"""Stage 3 final evaluation with the frozen configuration (pre-registration 3, section 7). Detector only.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.final_eval --part cst_val|antiuav|cst_test|svanstrom
  -> results/stage3/final__<part>.json   (refuses to run while the working tree is dirty)

cst_val    the new final test: primary metrics and verdicts J1-J4
antiuav    regression check J5 against the stage 1 model and settings
cst_test   second look (half of the stage 2 frames); descriptive, not used for any verdict
svanstrom  already seen (half of the stage 2 test frames); descriptive
"""
from __future__ import annotations

import argparse

from ..common import load_config, read_json, run_metadata, write_json
from ..stage2 import data as d2
from ..stage2.evalset import SvanEval, bootstrap as sv_bootstrap
from .data import OUT, frame_set, load_preds, stage2_subset
from .evalx import Ev3, bootstrap3, paired

J3_LINE = 0.30
J5 = {"recall_iou50": -0.02, "ap50": -0.02, "far_drone_free": 0.02}


def _configs(fz, set_name):
    return [(n, fz["configs"][n]) for n in fz["final_sets"][set_name] if n in fz["configs"]]


def _native(fz, set_name, frames, ref_ev, extra_pairs=()):
    out, evs = {}, {}
    for name, c in _configs(fz, set_name):
        src = name
        if c.get("duplicate_of"):
            src = c["duplicate_of"]
            c = fz["configs"][src]
        try:
            preds = load_preds(c["tag"], set_name)
        except (FileNotFoundError, RuntimeError) as e:
            out[name] = {"status": "not run", "detail": f"{type(e).__name__}: {e}"[:200]}
            continue
        ev = Ev3(frames, preds, c["threshold"])
        evs[name] = ev
        out[name] = {"status": "ok", "tag": c["tag"], "model": c["model"], "m": c["m"], "same_as": src if src != name else None,
                     **ev.metrics3(), **(bootstrap3(ev, ref_ev) if ref_ev is not None else bootstrap3(ev))}
    pairs = {}
    for a, b in extra_pairs:
        if a in evs and b in evs:
            pairs[f"{a}_minus_{b}"] = paired(evs[a], evs[b])
    return out, pairs, evs


def cst_val(fz):
    frames = frame_set("cst_val")
    c = fz["configs"]["S1_m1"]
    ref = Ev3(frames, load_preds(c["tag"], "cst_val"), c["threshold"])
    rows, pairs, evs = _native(fz, "cst_val", frames, ref, (("T2_5", "T1"), ("T2_20", "T1"), ("T2_20", "T2_5"), ("T1", "T1_sibling"), ("T1", "B_m1")))
    v = {}
    if rows.get("T1", {}).get("status") == "ok":
        pd = rows["T1"]["paired_diff"]
        lo = lambda k: pd[k]["ci95"][0]
        v["J1"] = {"diff_ap50": pd["ap50"], "improved": lo("ap50") > 0}
        ti, tc = lo("recall_tiny_iou50") > 0, lo("recall_tiny_center") > 0
        v["J2"] = {"tiny_iou50": pd["recall_tiny_iou50"], "tiny_center": pd["recall_tiny_center"],
                   "verdict": "both" if ti and tc else "iou50 only" if ti else "center only" if tc else "neither",
                   "far_diff": pd["far_drone_free"], "far_rose_more_than_5_points": pd["far_drone_free"]["diff"] > 0.05}
        v["J3"] = {"ap50": rows["T1"]["ap50"], "line": J3_LINE, "still_low": rows["T1"]["ap50"] < J3_LINE}
    for n in ("T2_5", "T2_20"):
        k = f"{n}_minus_T1"
        if k in pairs:
            v[f"J4_{n}"] = {"diff_ap50": pairs[k]["ap50"], "helped": pairs[k]["ap50"]["ci95"][0] > 0,
                            "ap50": rows[n]["ap50"], "still_low": rows[n]["ap50"] < J3_LINE}
    return {"set": "cst_val", "frames": len(frames), "sequences": len({f["seq"] for f in frames}),
            "reference": "S1_m1 (paired_diff of every row is against it)", "configs": rows, "pairs": pairs, "verdicts": v}


def antiuav(fz):
    frames = frame_set("auv_test")
    s1 = fz["configs"]["S1_m1"]["threshold"]
    ref = Ev3(frames, d2.load_preds("s1", "auv_test"), s1)          # stage 1 model and settings (stage 1 caches)
    rows, pairs, evs = _native(fz, "auv_test", frames, ref, (("T2_5", "T1"), ("T2_20", "T1")))
    v, supp = {}, {}
    f2 = frame_set("auv_test_small")
    seen = {f["image"] for f in frames}
    for name, r in rows.items():
        if r.get("status") != "ok":
            continue
        pd = r["paired_diff"]
        bad = {k: bool(pd[k]["diff"] < lim) if lim < 0 else bool(pd[k]["diff"] > lim) for k, lim in J5.items()}
        v[f"J5_{name}"] = {"diffs": {k: pd[k] for k in J5}, "over_limit": bad, "regression": any(bad.values())}
        c = fz["configs"][name]
        try:
            p2 = load_preds(c["tag"], "auv_test_small")
        except (FileNotFoundError, RuntimeError):
            continue
        extra = [(f, p) for f, p in zip(f2, p2) if f["image"] not in seen]
        ev = Ev3(frames + [x[0] for x in extra], load_preds(c["tag"], "auv_test") + [x[1] for x in extra], c["threshold"])
        supp[name] = ev.metrics()["recall_by_size"]
    extra = [(f, p) for f, p in zip(f2, d2.load_preds("s1", "auv_test_small")) if f["image"] not in seen]
    supp["S1_stage1"] = Ev3(frames + [x[0] for x in extra], d2.load_preds("s1", "auv_test") + [x[1] for x in extra], s1).metrics()["recall_by_size"]
    return {"set": "auv_test", "frames": len(frames), "reference_S1_stage1": {**ref.metrics3(), **bootstrap3(ref)},
            "configs": rows, "pairs": pairs, "verdicts": v, "limits_J5": J5, "recall_by_size_with_supplement": supp}


def cst_test(fz):
    frames = frame_set("cst_test_h")
    fz2 = read_json(d2.ROOT / "results" / "stage2" / "frozen_config.json")
    old = {"S1_stage2": ("s1_n", fz2["stage1_threshold"]), "B_stage2_tta": ("r1e1_ntta", fz2["models"]["B"]["antiuav_val"]["threshold"]),
           "C37_stage2_tta": ("r2n37_ntta", fz2["models"]["C37"]["antiuav_val"]["threshold"])}
    ref = Ev3(frames, stage2_subset("s1_n", "cst_test", "cst_test_h"), fz2["stage1_threshold"])
    rows, pairs, evs = _native(fz, "cst_test_h", frames, ref, (("T2_20", "T1"),))
    refs = {}
    for name, (tag, thr) in old.items():
        ev = Ev3(frames, stage2_subset(tag, "cst_test", "cst_test_h"), thr)
        refs[name] = {"tag": tag, "source": "stage 2 cache, same frames", **ev.metrics3(), **bootstrap3(ev, ref)}
    return {"set": "cst_test_h", "note": "SECOND LOOK: CST test was evaluated in stage 2 and informed the stage 3 design; not used for any verdict",
            "frames": len(frames), "reference": "S1_stage2 (stage 2 cache s1_n)", "stage2_rows": refs, "configs": rows, "pairs": pairs}


def svanstrom(fz):
    frames = frame_set("svan_test_h")
    out = {"set": "svan_test_h", "note": "already seen in stage 2; descriptive only", "frames": len(frames),
           "videos": len({f["video"] for f in frames}), "configs": {}}
    fz2 = read_json(d2.ROOT / "results" / "stage2" / "frozen_config.json")
    s1p = stage2_subset("s1", "svan_test", "svan_test_h")
    thr0 = fz2["stage1_threshold"]
    ref = SvanEval(frames, s1p, 1.0)

    def row(ev_k1, ev_k, thr, k):
        t10 = ev_k1.threshold_for_far(0.10)
        return {"threshold": thr, "k": k, "at_threshold_k1": ev_k1.metrics(thr), "at_threshold_k": ev_k.metrics(thr),
                "at_far10_same_frames": {"threshold": t10, "k1": ev_k1.metrics(t10), "k": ev_k.metrics(t10)},
                **sv_bootstrap(ev_k1, thr, ref, thr0)}

    k0 = fz2["stage1_x2"]["k"]
    out["S1_stage1"] = row(ref, SvanEval(frames, s1p, k0), thr0, k0)
    for name, c in _configs(fz, "svan_test_h"):
        try:
            preds = load_preds(c["tag"], "svan_test_h")
        except (FileNotFoundError, RuntimeError) as e:
            out["configs"][name] = {"status": "not run", "detail": f"{type(e).__name__}: {e}"[:200]}
            continue
        k = fz["svan_k"][name]["k"]
        out["configs"][name] = {"status": "ok", "tag": c["tag"], "k_source": fz["svan_k"][name]["source"],
                                **row(SvanEval(frames, preds, 1.0), SvanEval(frames, preds, k), c["threshold"], k)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cst_val", "antiuav"])   # addendum 2: cst_test and svanstrom are not run
    ap.add_argument("--allow-dirty", action="store_true", help="smoke tests only; the output is then marked")
    a = ap.parse_args()
    cfg = load_config("configs/full.yaml")
    meta = run_metadata(cfg, {"stage": 3})
    if meta["git_dirty"] and not a.allow_dirty:
        raise SystemExit("refused: the working tree is dirty (commit first; results/ does not count)")
    fz = read_json(OUT / "frozen_config.json")
    res = {"cst_val": cst_val, "antiuav": antiuav, "cst_test": cst_test, "svanstrom": svanstrom}[a.part](fz)
    res["frozen_config"] = "results/stage3/frozen_config.json"
    res["meta"] = meta
    write_json(res, OUT / f"final__{a.part}.json")
    print("==", a.part)
    for name, r in res["configs"].items():
        if r.get("status") != "ok":
            print(" ", name, r.get("status"), r.get("detail", ""))
        elif "ap50" in r:
            print(f"  {name}: AP {r['ap50']:.3f} R50 {r['recall_iou50']:.3f} Rc {r['recall_center']:.3f} FAR {r['far_drone_free']:.3f} "
                  f"tiny {r['recall_by_size']['tiny']['iou50']:.3f}/{r['recall_by_size']['tiny']['center']:.3f}")
    if "verdicts" in res:
        for k, v in res["verdicts"].items():
            print(" ", k, {x: y for x, y in v.items() if not isinstance(y, dict)})


if __name__ == "__main__":
    main()
