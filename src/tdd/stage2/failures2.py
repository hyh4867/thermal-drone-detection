"""Stage 2 failure-case gallery: stage 1 model (full run) on the Svanström DEV videos.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.failures2 [--tag s1] [--set svan_dev]
  -> results/stage2/failures/*.jpg + index.json

Categories at the stage 1 threshold (deterministic, at most one frame per video):
  drone_found_box_too_tight   detection centre inside the label box but IoU < 0.5 (lowest IoU first)
  drone_missed_distant        DISTANT drone with no detection >= threshold inside its label box
                              (lowest best score first)
  bird_flagged / airplane_flagged / helicopter_flagged
                              detection centre inside the labelled object's box (highest score first)
  alarm_away_from_label       non-drone frame, detection not on the labelled object - background OR an unlabelled object (highest score first)
Green = label box, red = detections at or above the threshold.
"""
from __future__ import annotations

import argparse

import numpy as np

from ..common import ROOT, write_json
from ..failures import render
from ..metrics import iou_matrix
from .data import frame_set, load_preds
from .diagnose import s1_threshold

PER_CAT = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--set", default="svan_dev")
    ap.add_argument("--out", default="results/stage2/failures")
    a = ap.parse_args()
    thr = s1_threshold()
    frames, preds = frame_set(a.set), load_preds(a.tag, a.set)
    cands = {k: [] for k in ("drone_found_box_too_tight", "drone_missed_distant", "bird_flagged", "airplane_flagged",
                             "helicopter_flagged", "alarm_away_from_label")}
    for f, p in zip(frames, preds):
        d_all = np.asarray(p, dtype=np.float64).reshape(-1, 5)
        d = d_all[d_all[:, 4] >= thr]
        g = np.asarray(f["gt"][0]) if f["gt"] else None

        def inside(dd):
            cx, cy = (dd[:, 0] + dd[:, 2]) / 2, (dd[:, 1] + dd[:, 3]) / 2
            return (cx >= g[0]) & (cx <= g[2]) & (cy >= g[1]) & (cy <= g[3])
        if f["cls"] == "DRONE" and g is not None:
            ins = inside(d) if len(d) else np.zeros(0, bool)
            if ins.any():
                iou = iou_matrix(d[ins][:, :4], g[None])[:, 0].max()
                if iou < 0.5:
                    cands["drone_found_box_too_tight"].append((iou, f, d, f"IoU {iou:.2f}, centre inside the label"))
            elif f["bin"] == "DISTANT":
                ins_all = inside(d_all) if len(d_all) else np.zeros(0, bool)
                best = float(d_all[ins_all, 4].max()) if ins_all.any() else 0.0
                cands["drone_missed_distant"].append((best, f, d, f"best score inside the label {best:.2f} < {thr:.2f}"))
        elif f["cls"] != "DRONE" and len(d):
            key = f"{f['cls'].lower()}_flagged"
            if g is not None and inside(d).any():
                s = float(d[inside(d), 4].max())
                cands[key].append((-s, f, d, f"{f['cls'].lower()} boxed as drone, score {s:.2f}"))
            else:
                s = float(d[:, 4].max())
                cands["alarm_away_from_label"].append((-s, f, d, f"alarm away from the labelled object, score {s:.2f}"))
    out_dir = ROOT / a.out
    index = []
    for cat, items in cands.items():
        items.sort(key=lambda x: (x[0], x[1]["video"], x[1]["frame"]))
        seen, n = set(), 0
        for _, f, d, note in items:
            if f["video"] in seen:
                continue
            seen.add(f["video"])
            n += 1
            name = f"{cat}_{n:02d}.jpg"
            size = f"{f['obj_size'] * 2:.0f} px" if f.get("obj_size") else "no label"
            render(f, d, 2.0, f"{cat} | {f['video']} frame {f['frame']} | label {size} | {note}", out_dir / name)
            index.append({"file": name, "category": cat, "video": f["video"], "frame": f["frame"], "class": f["cls"],
                          "distance_bin": f["bin"], "label_size_ref_px": f["obj_size"] * 2 if f.get("obj_size") else None,
                          "note": note, "dataset": "Svanström et al. 2021 (Zenodo 10.5281/zenodo.5500576, CC0), dev split"})
            if n == PER_CAT:
                break
    write_json({"model": "stage 1 full run (runs/full_frcnn/best.pt)", "threshold": thr, "set": a.set,
                "candidates_per_category": {k: len(v) for k, v in cands.items()},
                "videos_per_category": {k: len({x[1]["video"] for x in v}) for k, v in cands.items()},
                "cases": index}, out_dir / "index.json")
    print({k: len(v) for k, v in cands.items()}, len(index), "images")


if __name__ == "__main__":
    main()
