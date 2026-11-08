"""Anti-UAV300 IR: source-video-level split, integrity checks, frame extraction.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.prepare_antiuav [--config configs/default.yaml]

Outputs
  data/frames/antiuav/<seq>/<frame:04d>.jpg   (grayscale, JPEG q95; not committed)
  data/manifests/antiuav_<split>.json         (frame list + GT boxes; not committed)
  results/data/antiuav_split.json             (which sequence/source video went where)
  results/data/antiuav_stats.json             (counts, empty-frame share, size distribution)
"""
from __future__ import annotations

import argparse
import collections
import random

import cv2
import numpy as np

from .common import ROOT, load_config, path, read_json, run_metadata, set_seed, write_json
from .metrics import size_of


def source_of(seq: str) -> str:
    return seq.rsplit("_", 1)[0]


def make_split(root, val_frac: float, seed: int) -> dict:
    official = {s: sorted(p.name for p in (root / s).iterdir() if p.is_dir()) for s in ("train", "val", "test")}
    pool = official["train"] + official["val"]
    sources = sorted({source_of(s) for s in pool})
    rng = random.Random(seed)
    shuffled = sources[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(sources) * val_frac))
    val_sources = set(shuffled[:n_val])
    split = {
        "train": sorted(s for s in pool if source_of(s) not in val_sources),
        "val": sorted(s for s in pool if source_of(s) in val_sources),
        "test": official["test"],
    }
    seq_dir = {s: f"{k}/{s}" for k, v in official.items() for s in v}
    src = {k: sorted({source_of(s) for s in v}) for k, v in split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(src[a]) & set(src[b]), f"source leak {a}/{b}"
    return {"sequences": split, "sources": src, "seq_dir": seq_dir}


def load_seq(root, rel: str):
    ann = read_json(root / rel / "infrared.json")
    cap = cv2.VideoCapture(str(root / rel / "infrared.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return ann, cap, n, (w, h)


def gt_for(ann, i: int):
    """Returns (status, box). status: 'present' | 'absent' | 'invalid'."""
    exist = int(ann["exist"][i])
    rect = ann["gt_rect"][i] if i < len(ann["gt_rect"]) else []
    if exist == 0:
        return "absent", None
    if not rect or len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
        return "invalid", None
    x, y, w, h = map(float, rect)
    return "present", [x, y, x + w, y + h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    root = path(cfg["data"]["antiuav_root"])
    split = make_split(root, cfg["split"]["val_fraction_sources"], cfg["seed"])
    frames_root = path(cfg["data"]["frames_dir"]) / "antiuav"
    bins = cfg["eval"]["size_bins"]

    stats = {}
    for sp, seqs in split["sequences"].items():
        stride = cfg["split"]["stride"][sp]
        manifest, supplement = [], []
        st = collections.Counter()
        sizes_all, sizes_sampled = [], []
        resolutions = collections.Counter()
        for seq in seqs:
            rel = split["seq_dir"][seq]
            ann, cap, n_video, wh = load_seq(root, rel)
            resolutions[f"{wh[0]}x{wh[1]}"] += 1
            n_ann = len(ann["exist"])
            if n_video != n_ann:
                st["seqs_frame_count_mismatch"] += 1
            n = min(n_video, n_ann)
            st["seqs"] += 1
            st["frames_total"] += n
            for i in range(n):
                status, box = gt_for(ann, i)
                st[f"frames_{status}"] += 1
                if box is not None:
                    sizes_all.append(size_of(box))
            keep = set(range(0, n, stride))
            if sp in cfg["split"].get("include_all_absent", []):
                # every drone-free frame, so the false-alarm rate rests on all of them
                keep |= {j for j in range(n) if gt_for(ann, j)[0] == "absent"}
            small_extra = set()
            if sp in cfg["split"].get("small_target_supplement", {}).get("splits", []):
                lim = cfg["split"]["small_target_supplement"]["max_size"]
                for j in range(n):
                    b = gt_for(ann, j)[1]
                    if b is not None and size_of(b) < lim and j not in keep:
                        small_extra.add(j)
            i = 0
            while True:
                ok = cap.grab()
                if not ok or i >= n:
                    break
                if i in keep or i in small_extra:
                    status, box = gt_for(ann, i)
                    img_rel = f"antiuav/{seq}/{i:04d}.jpg"
                    if not args.no_extract:
                        ok, frame = cap.retrieve()
                        if not ok:
                            st["decode_errors"] += 1
                            i += 1
                            continue
                        out = frames_root / seq / f"{i:04d}.jpg"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if not out.exists():
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            cv2.imwrite(str(out), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rec = {
                        "seq": seq, "source": source_of(seq), "frame": i, "image": img_rel,
                        "width": wh[0], "height": wh[1],
                        "gt": [box] if box is not None else [],
                    }
                    if i in small_extra:
                        supplement.append(rec)
                        i += 1
                        continue
                    st[f"sampled_{status}"] += 1
                    if box is not None:
                        sizes_sampled.append(size_of(box))
                    # 'invalid' frames (exist=1 but no usable box) are kept out of
                    # evaluation entirely: neither positive nor drone-free.
                    if status != "invalid":
                        manifest.append(rec)
                i += 1
            cap.release()
        write_json(manifest, path(cfg["data"]["manifests_dir"]) / f"antiuav_{sp}.json")
        if supplement:
            # Frames NOT in the main manifest whose target is < max_size px; used only
            # for the size-bin recall, never for precision / AP / false alarms.
            write_json(supplement, path(cfg["data"]["manifests_dir"]) / f"antiuav_{sp}_small_supplement.json")

        def dist(sz):
            a = np.asarray(sz)
            return {
                "n": int(len(a)),
                "percentiles_sqrt_wh": {str(q): round(float(np.percentile(a, q)), 2) for q in (5, 25, 50, 75, 95)} if len(a) else {},
                "bins": {b[0]: int(((a >= b[1]) & (a < b[2])).sum()) for b in bins},
            }

        stats[sp] = {
            "stride": stride,
            "source_videos": len(split["sources"][sp]),
            **dict(st),
            "resolutions": dict(resolutions),
            "eval_frames": len(manifest),
            "eval_frames_without_drone": sum(1 for m in manifest if not m["gt"]),
            "small_supplement_frames": len(supplement),
            "small_supplement_sequences": len({m["seq"] for m in supplement}),
            "size_all_frames": dist(sizes_all),
            "size_sampled_frames": dist(sizes_sampled),
        }
        print(sp, {k: v for k, v in stats[sp].items() if not isinstance(v, dict)})

    out_dir = ROOT / "results" / "data"
    write_json({"note": "train/val re-split by source video; official test kept", **split,
                "meta": run_metadata(cfg)}, out_dir / "antiuav_split.json")
    write_json({"stats": stats, "meta": run_metadata(cfg)}, out_dir / "antiuav_stats.json")


if __name__ == "__main__":
    main()
