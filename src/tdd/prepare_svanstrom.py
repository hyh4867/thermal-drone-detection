"""Svanström et al. 2021 IR videos: labels, distance bins, frame extraction.

The dataset is NEVER used for training or threshold selection; it is a held-out
cross-dataset (generalisation) test.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.prepare_svanstrom

Outputs
  data/frames/svanstrom/<video>/<frame:04d>.jpg   (native 320x256 grayscale)
  data/manifests/svanstrom_ir.json
  results/data/svanstrom_stats.json
"""
from __future__ import annotations

import argparse
import collections

import cv2
import pandas as pd
from mcos_decoder import load_groundtruth

from .common import ROOT, load_config, path, run_metadata, set_seed, write_json
from .metrics import size_of


def read_bins(xlsx) -> dict:
    df = pd.read_excel(xlsx)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["SENSOR"].astype(str).str.strip() == "IR"]
    out = {}
    for _, r in df.iterrows():
        name = f"IR_{str(r['CLASS']).strip()}_{int(r['NUMBER']):03d}"
        src = r.get("DRONE TYPE / INTERNET SOURCE")
        out[name] = {
            "bin": str(r["DISTANCE  BIN"]).strip().upper(),
            "source_note": None if pd.isna(src) else str(src).strip(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    root = path(cfg["data"]["svanstrom_root"])
    vid_dir = root / "Video_IR"
    bins = read_bins(root / "Video_dataset_description.xlsx")
    stride = cfg["eval"]["svanstrom_stride"]
    frames_root = path(cfg["data"]["frames_dir"]) / "svanstrom"

    manifest, excluded = [], []
    st = collections.defaultdict(collections.Counter)
    for mp4 in sorted(vid_dir.glob("IR_*.mp4")):
        name = mp4.stem
        cls = name.split("_")[1]
        info = bins[name]
        # MATLAB groundTruth boxes are [x y w h] in 1-based spatial coordinates
        # (pixel (1,1) spans 0.5..1.5); shift by -0.5 to 0-based pixel-edge coords.
        labels = load_groundtruth(str(vid_dir / f"{name}_LABELS.mat"))
        cap = cv2.VideoCapture(str(mp4))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = min(n_video, len(labels))
        key = f"{cls}/{info['bin']}"
        if abs(n_video - len(labels)) > cfg["eval"]["svanstrom_max_label_mismatch"]:
            # label/frame alignment cannot be trusted -> excluded (listed in stats)
            excluded.append({"video": name, "frames": n_video, "labels": len(labels)})
            st[key]["excluded_videos"] += 1
            cap.release()
            continue
        st[key]["videos"] += 1
        st[key]["label_video_len_mismatch"] += int(n_video != len(labels))
        if info["source_note"] and "internet" in info["source_note"].lower():
            st[key]["internet_videos"] += 1
        i = 0
        while i < n and cap.grab():
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    st[key]["decode_errors"] += 1
                    i += 1
                    continue
                out = frames_root / name / f"{i:04d}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), [cv2.IMWRITE_JPEG_QUALITY, 95])
                b = labels[i]
                gt = []
                if b is not None and b[2] > 0 and b[3] > 0:
                    x, y, bw, bh = (float(v) for v in b)
                    gt = [[x - 0.5, y - 0.5, x - 0.5 + bw, y - 0.5 + bh]]
                st[key]["frames"] += 1
                st[key]["frames_with_object"] += int(bool(gt))
                manifest.append({
                    "video": name, "cls": cls, "bin": info["bin"], "source_note": info["source_note"],
                    "frame": i, "image": f"svanstrom/{name}/{i:04d}.jpg", "width": w, "height": h,
                    # gt = the labelled object; it is a DRONE only when cls == "DRONE"
                    "gt": gt,
                    "obj_size": round(size_of(gt[0]), 2) if gt else None,
                })
            i += 1
        cap.release()
        st[key][f"res_{w}x{h}"] += 1

    write_json(manifest, path(cfg["data"]["manifests_dir"]) / "svanstrom_ir.json")
    write_json({"stride": stride, "excluded_videos": excluded, "per_class_bin": {k: dict(v) for k, v in sorted(st.items())},
                "meta": run_metadata(cfg)}, ROOT / "results" / "data" / "svanstrom_stats.json")
    for k, v in sorted(st.items()):
        print(k, dict(v))


if __name__ == "__main__":
    main()
