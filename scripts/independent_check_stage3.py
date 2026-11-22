#!/usr/bin/env python3
"""Independent recomputation of the stage 3 headline numbers on CST val.

Does NOT import the tdd package (standard library only). Reads the label manifest, the prediction
cache lines and the frozen thresholds, recomputes for every configuration evaluated on CST val
  AP@0.5 (all-point interpolation, detections ranked by score over the whole set, greedy matching per frame),
  recall at the operating threshold under IoU >= 0.5 and under the centre criterion,
  the same two recalls for the tiny (< 8 px) and small (8-16 px) bins of sqrt(w*h),
and compares them with results/stage3/final__cst_val.json (tolerance 1e-6).

Usage: python3 scripts/independent_check_stage3.py [--reference]   -> results/stage3/independent_check[_reference].json, exit 1 on mismatch
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage3"
TOL = 1e-6


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy(gts, dets):
    """dets sorted by descending score (stable). Returns tp flag per detection, matched flag per GT."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    matched = [False] * len(gts)
    flags = []
    for i in order:
        best, bj = -1.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(dets[i], g)
            if v > best:
                best, bj = v, j
        hit = bj >= 0 and best >= 0.5
        if hit:
            matched[bj] = True
        flags.append((dets[i][4], hit))
    return flags, matched


def evaluate(frames, preds, thr):
    ranked, n_gt = [], 0
    hit50 = hitc = 0
    bins = {"tiny": [0, 0, 0], "small": [0, 0, 0]}        # iou hits, centre hits, total
    for f, p in zip(frames, preds):
        if f.get("invalid"):
            continue
        gts = f["gt"]
        n_gt += len(gts)
        flags, _ = greedy(gts, p)
        ranked.extend(flags)
        kept = [d for d in p if d[4] >= thr]
        _, m50 = greedy(gts, kept)
        for g, m in zip(gts, m50):
            c = any(g[0] <= (d[0] + d[2]) / 2 <= g[2] and g[1] <= (d[1] + d[3]) / 2 <= g[3] for d in kept)
            hit50 += m
            hitc += c
            s = math.sqrt(max(g[2] - g[0], 0.0) * max(g[3] - g[1], 0.0)) * 640.0 / f["width"]
            b = "tiny" if s < 8 else "small" if s < 16 else None
            if b:
                bins[b][0] += m
                bins[b][1] += c
                bins[b][2] += 1
    # AP: stable sort by descending score in frame order (the pipeline concatenates frames, then stable-sorts)
    ranked.sort(key=lambda t: -t[0])
    tp = fp = 0
    rec, prec = [0.0], [1.0]
    for _, h in ranked:
        tp += h
        fp += not h
        rec.append(tp / n_gt)
        prec.append(tp / (tp + fp))
    rec.append(1.0)
    prec.append(0.0)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = sum((rec[i + 1] - rec[i]) * prec[i + 1] for i in range(len(rec) - 1) if rec[i + 1] != rec[i])
    out = {"ap50": ap, "recall_iou50": hit50 / n_gt, "recall_center": hitc / n_gt, "gt_boxes": n_gt}
    for b, (a, c, t) in bins.items():
        out[f"recall_{b}_iou50"] = a / t if t else float("nan")
        out[f"recall_{b}_center"] = c / t if t else float("nan")
        out[f"n_{b}"] = t
    return out


def svanstrom():
    """--svanstrom: addendum 6. Centre recall, IoU 0.5 recall (k = 1) and the alarm rate on bird / airplane /
    helicopter frames of both configurations, from the Svanström manifest, the frozen split and the cache lines."""
    man = json.loads((ROOT / "data" / "manifests" / "svanstrom_ir.json").read_text())
    split = {v["video"]: v["split"] for v in json.loads((ROOT / "docs" / "splits" / "svanstrom_stage2_split.json").read_text())["videos"]}
    frames = [f for f in man if split.get(f["video"]) == "test" and f["frame"] % 10 == 0]
    res = json.loads((OUT / "svanstrom_test.json").read_text())
    rows, ok = {}, True
    for name, r in res["configs"].items():
        with open(ROOT / "runs" / "stage3" / "preds" / f"{r['tag']}__svan_test_h.jsonl") as f:
            preds = [json.loads(line) for line in f]
        assert len(preds) == len(frames) == res["frames"], (name, len(preds), len(frames))
        thr = r["threshold"]
        n_gt = hit50 = hitc = neg = alarm = 0
        for fr, p in zip(frames, preds):
            kept = [d for d in p if d[4] >= thr]
            if fr["cls"] != "DRONE":
                neg += 1
                alarm += bool(kept)
                continue
            _, m50 = greedy(fr["gt"], kept)
            for g, m in zip(fr["gt"], m50):
                n_gt += 1
                hit50 += m
                hitc += any(g[0] <= (d[0] + d[2]) / 2 <= g[2] and g[1] <= (d[1] + d[3]) / 2 <= g[3] for d in kept)
        mine = {"recall_center": hitc / n_gt, "recall_iou50": hit50 / n_gt, "far": alarm / neg, "drone_gt": n_gt, "non_drone_frames": neg}
        diff = {k: abs(mine[k] - r[k]) for k in mine}
        good = all(v <= TOL for v in diff.values())
        ok &= good
        rows[name] = {"tag": r["tag"], "threshold": thr, "independent": mine, "pipeline": {k: r[k] for k in mine},
                      "max_abs_diff": max(diff.values()), "match": good}
        print(f"{'ok  ' if good else 'DIFF'} {name}: centre {mine['recall_center']:.6f} vs {r['recall_center']:.6f}, IoU0.5 {mine['recall_iou50']:.6f} vs "
              f"{r['recall_iou50']:.6f}, alarm rate {mine['far']:.6f} vs {r['far']:.6f}, max |diff| {max(diff.values()):.2e}")
    (OUT / "independent_check_svanstrom.json").write_text(json.dumps({"tolerance": TOL, "all_match": ok, "configs": rows,
                                                                       "pipeline_git_commit": res["meta"]["git_commit"]}, indent=1) + "\n")
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


def main():
    if "--svanstrom" in sys.argv[1:]:
        return svanstrom()
    # --reference: the reference row of addendum 5 (results/stage3/reference__cst_val.json -> independent_check_reference.json)
    reference = "--reference" in sys.argv[1:]
    frames = json.loads((ROOT / "data" / "manifests" / "cst_val.json").read_text())
    final = json.loads((OUT / ("reference__cst_val.json" if reference else "final__cst_val.json")).read_text())
    fz = json.loads((OUT / "frozen_config.json").read_text())
    rows, ok = {}, True
    for name, r in final["configs"].items():
        if r.get("status") != "ok":
            continue
        c = {"tag": r["tag"], "threshold": r["threshold"]} if reference else fz["configs"][r.get("same_as") or name]
        with open(ROOT / "runs" / "stage3" / "preds" / f"{c['tag']}__cst_val.jsonl") as f:
            preds = [json.loads(line) for line in f]
        assert len(preds) == len(frames), (name, len(preds), len(frames))
        mine = evaluate(frames, preds, c["threshold"])
        theirs = {"ap50": r["ap50"], "recall_iou50": r["recall_iou50"], "recall_center": r["recall_center"], "gt_boxes": r["gt_boxes"],
                  "recall_tiny_iou50": r["recall_by_size"]["tiny"]["iou50"], "recall_tiny_center": r["recall_by_size"]["tiny"]["center"],
                  "recall_small_iou50": r["recall_by_size"]["small"]["iou50"], "recall_small_center": r["recall_by_size"]["small"]["center"],
                  "n_tiny": r["recall_by_size"]["tiny"]["total"], "n_small": r["recall_by_size"]["small"]["total"]}
        diff = {k: abs(mine[k] - theirs[k]) for k in theirs if not (mine[k] != mine[k] and theirs[k] != theirs[k])}
        good = all(v <= TOL for v in diff.values())
        ok &= good
        rows[name] = {"tag": c["tag"], "threshold": c["threshold"], "independent": mine, "pipeline": theirs,
                      "max_abs_diff": max(diff.values()), "match": good}
        print(f"{'ok  ' if good else 'DIFF'} {name}: AP {mine['ap50']:.6f} vs {theirs['ap50']:.6f}, recall {mine['recall_iou50']:.6f} vs "
              f"{theirs['recall_iou50']:.6f}, max |diff| {max(diff.values()):.2e}")
    (OUT / ("independent_check_reference.json" if reference else "independent_check.json")).write_text(json.dumps({"tolerance": TOL, "all_match": ok, "configs": rows,
                                                             "pipeline_git_commit": final["meta"]["git_commit"]}, indent=1) + "\n")
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
