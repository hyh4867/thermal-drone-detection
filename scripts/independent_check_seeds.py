#!/usr/bin/env python3
"""Independent recomputation of the seed-study numbers (standard library only; does NOT import the tdd package).

For every results/seeds/eval_seed<k>.json it reads the label manifests, the prediction cache lines and the
threshold written in that file, and recomputes
  CST val        AP@0.5, IoU 0.5 and centre recall, the same for the tiny (< 8 px) and small (8-16 px) bins
  Anti-UAV test  AP@0.5, IoU 0.5 recall, alarm rate on drone-free frames, their differences to S1 and the J5 flags
                 (S1 here = the stage 3 cache s1_m1__auv_test at the stage 1 threshold; the pipeline uses the stage 1
                 caches, which stage 3 found identical)
  Svanström test centre recall, IoU 0.5 recall, alarm rate on bird / airplane / helicopter frames
then the mean, minimum, maximum and range over seeds, and compares everything with the pipeline files (tolerance 1e-6).
The matching and AP functions are those of scripts/independent_check_stage3.py (also standard library only).

Usage: python3 scripts/independent_check_seeds.py   -> results/seeds/independent_check.json, exit 1 on mismatch
"""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "seeds"
TOL = 1e-6
J5 = {"recall_iou50": -0.02, "ap50": -0.02, "far_drone_free": 0.02}

_spec = importlib.util.spec_from_file_location("indep3", ROOT / "scripts" / "independent_check_stage3.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def cache(tag, set_name):
    base = "seeds" if tag.startswith("t2_20s") else "stage3"
    with open(ROOT / "runs" / base / "preds" / f"{tag}__{set_name}.jsonl") as f:
        return [json.loads(line) for line in f]


def far_drone_free(frames, preds, thr):
    empty = alarm = 0
    for f, p in zip(frames, preds):
        if f.get("invalid") or f["gt"]:
            continue
        empty += 1
        alarm += any(d[4] >= thr for d in p)
    return alarm / empty


def svan(frames, preds, thr):
    n_gt = hit50 = hitc = neg = alarm = 0
    for fr, p in zip(frames, preds):
        kept = [d for d in p if d[4] >= thr]
        if fr["cls"] != "DRONE":
            neg += 1
            alarm += bool(kept)
            continue
        _, m50 = s3.greedy(fr["gt"], kept)
        for g, m in zip(fr["gt"], m50):
            n_gt += 1
            hit50 += m
            hitc += any(g[0] <= (d[0] + d[2]) / 2 <= g[2] and g[1] <= (d[1] + d[3]) / 2 <= g[3] for d in kept)
    return {"recall_center": hitc / n_gt, "recall_iou50": hit50 / n_gt, "far": alarm / neg}


def compare(mine, theirs):
    diff = {k: abs(float(mine[k]) - float(theirs[k])) for k in mine}
    return {"independent": mine, "pipeline": theirs, "max_abs_diff": max(diff.values()), "match": all(v <= TOL for v in diff.values())}


def main():
    s1_thr = load(ROOT / "results" / "stage3" / "frozen_config.json")["configs"]["S1_m1"]["threshold"]
    cst = load(ROOT / "data" / "manifests" / "cst_val.json")
    auv = load(ROOT / "data" / "manifests" / "antiuav_test.json")
    split = {v["video"]: v["split"] for v in load(ROOT / "docs" / "splits" / "svanstrom_stage2_split.json")["videos"]}
    sv = [f for f in load(ROOT / "data" / "manifests" / "svanstrom_ir.json") if split.get(f["video"]) == "test" and f["frame"] % 10 == 0]
    s1_cst = s3.evaluate(cst, cache("s1_m1", "cst_val"), s1_thr)
    s1_auv = s3.evaluate(auv, cache("s1_m1", "auv_test"), s1_thr)
    s1_auv["far_drone_free"] = far_drone_free(auv, cache("s1_m1", "auv_test"), s1_thr)
    s1_sv = svan(sv, cache("s1_m1", "svan_test_h"), s1_thr)
    rows, ok, mine_all = {}, True, {}
    for k in (0, 1, 2):
        p = OUT / f"eval_seed{k}.json"
        if not p.exists():
            continue
        r = load(p)
        thr, tag, row, m = r["threshold"], r["tag"], {}, {}
        c = r["cst_val"]
        if c["status"] == "ok":
            e = s3.evaluate(cst, cache(tag, "cst_val"), thr)
            mine = {"ap50": e["ap50"], "ap50_minus_S1": e["ap50"] - s1_cst["ap50"], "recall_iou50": e["recall_iou50"], "recall_center": e["recall_center"],
                    "recall_tiny_iou50": e["recall_tiny_iou50"], "recall_tiny_center": e["recall_tiny_center"],
                    "recall_small_iou50": e["recall_small_iou50"], "recall_small_center": e["recall_small_center"],
                    "far_drone_free": far_drone_free(cst, cache(tag, "cst_val"), thr)}
            theirs = {"ap50": c["ap50"], "ap50_minus_S1": c["paired_diff"]["ap50"]["diff"], "recall_iou50": c["recall_iou50"], "recall_center": c["recall_center"],
                      "recall_tiny_iou50": c["recall_by_size"]["tiny"]["iou50"], "recall_tiny_center": c["recall_by_size"]["tiny"]["center"],
                      "recall_small_iou50": c["recall_by_size"]["small"]["iou50"], "recall_small_center": c["recall_by_size"]["small"]["center"],
                      "far_drone_free": c["far_drone_free"]}
            row["cst_val"] = compare(mine, theirs)
            m.update({f"cst_val.{x}": v for x, v in mine.items()})
        a = r["antiuav"]
        if a["status"] == "ok":
            pr = cache(tag, "auv_test")
            e = s3.evaluate(auv, pr, thr)
            far = far_drone_free(auv, pr, thr)
            mine = {"ap50": e["ap50"], "recall_iou50": e["recall_iou50"], "far_drone_free": far, "ap50_minus_S1": e["ap50"] - s1_auv["ap50"],
                    "recall_iou50_minus_S1": e["recall_iou50"] - s1_auv["recall_iou50"], "far_drone_free_minus_S1": far - s1_auv["far_drone_free"]}
            d = a["paired_diff"]
            theirs = {"ap50": a["ap50"], "recall_iou50": a["recall_iou50"], "far_drone_free": a["far_drone_free"], "ap50_minus_S1": d["ap50"]["diff"],
                      "recall_iou50_minus_S1": d["recall_iou50"]["diff"], "far_drone_free_minus_S1": d["far_drone_free"]["diff"]}
            row["antiuav"] = compare(mine, theirs)
            flags = {x: (mine[f"{x}_minus_S1"] < lim if lim < 0 else mine[f"{x}_minus_S1"] > lim) for x, lim in J5.items()}
            row["antiuav"]["J5"] = {"independent": {"over_limit": flags, "regression": any(flags.values())},
                                    "pipeline": {"over_limit": a["J5"]["over_limit"], "regression": a["J5"]["regression"]}}
            row["antiuav"]["match"] = row["antiuav"]["match"] and flags == a["J5"]["over_limit"] and any(flags.values()) == a["J5"]["regression"]
            m.update({f"antiuav.{x}": v for x, v in mine.items()})
        s = r["svanstrom"]
        if s["status"] == "ok":
            mine = svan(sv, cache(tag, "svan_test_h"), thr)
            mine["far_minus_S1"] = mine["far"] - s1_sv["far"]
            mine["recall_center_minus_S1"] = mine["recall_center"] - s1_sv["recall_center"]
            theirs = {"recall_center": s["recall_center"], "recall_iou50": s["recall_iou50"], "far": s["far"],
                      "far_minus_S1": s["paired_diff"]["far"]["diff"], "recall_center_minus_S1": s["paired_diff"]["recall_center"]["diff"]}
            row["svanstrom"] = compare(mine, theirs)
            m.update({f"svanstrom.{x}": v for x, v in mine.items()})
        good = all(v["match"] for v in row.values())
        ok &= good
        rows[str(k)] = {"tag": tag, "threshold": thr, **row, "match": good}
        mine_all[k] = m
        print(f"{'ok  ' if good else 'DIFF'} seed {k}: " + ", ".join(f"{part} max |diff| {v['max_abs_diff']:.2e}" for part, v in row.items()))
    spread = {}
    sp = OUT / "summary.json"
    if sp.exists():
        tab = load(sp)["metrics"]
        for name, t in tab.items():
            v = [mine_all[int(k)].get(name) for k in t["by_seed"]]
            if any(x is None for x in v):
                continue
            mine = {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "range": max(v) - min(v)}
            spread[name] = compare(mine, {x: t[x] for x in mine})
            ok &= spread[name]["match"]
        print(f"summary: {len(spread)} rows recomputed, {'all match' if all(v['match'] for v in spread.values()) else 'DIFF'}")
    (OUT / "independent_check.json").write_text(json.dumps({"tolerance": TOL, "all_match": bool(ok), "seeds": rows, "summary_spread": spread}, indent=1) + "\n")
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
