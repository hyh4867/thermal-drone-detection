"""Stage 3: CST Anti-UAV val (new final test) and train (track 2 only).

Roles are frozen in docs/PREREGISTRATION_3.md section 3 BEFORE this download:
  val   40 sequences  final test, never used for a choice, images are read by inference code only
  train 120 sequences track 2 only; split into cst_tr_hold (20), cst_tr_20 (20, cst_tr_5 = first 5), unused (80)

Source/terms: see tdd.stage2.prepare_cst (github.com/PCwenyue/CST-Anti-UAV, DATASET_TERMS.md: academic and
non-commercial research only, cite Xie et al. ICCVW 2025). Nothing of the dataset is committed to git.

The split zip is streamed from the start (the test split, already stored in stage 2, is skipped; its stream
hash is compared with the stage 2 log). Kept: all label files and every 4th frame of train/ and val/.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage3.prepare_cst             (download + manifests + split)
       PYTHONPATH=src .venv/bin/python -m tdd.stage3.prepare_cst --manifest  (manifests + split only)

Outputs
  data/raw/cst_antiuav/{train,val}/<seq>/...      git-ignored
  data/raw/cst_antiuav/download_log_stage3.json   git-ignored
  data/manifests/cst_{val,train}.json             git-ignored
  docs/splits/cst_stage3_split.json               sequence names per role (committed before any model runs on CST)
  results/data/cst_stats_stage3.json              counts and size distribution only
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import struct
import time
import zlib

from ..common import ROOT, write_json
from ..metrics import size_of
from ..stage2.prepare_cst import FPS, PARTS, RAW, STRIDE, PartStream, _read_boxes

SPLIT_SEED = 20261122
N_HOLD, N_ADAPT, N_SMALL = 20, 20, 5
BINS = [("tiny", 0, 8), ("small", 8, 16), ("medium", 16, 32), ("large", 32, 64), ("xlarge", 64, 1e9)]
IMG = re.compile(r"^(\d+)\.(jpg|jpeg|png|bmp)$", re.I)


def download():
    s = PartStream()
    assert s.read(4) == b"PK\x07\x08", "expected the split-archive marker"
    stage2 = json.loads((RAW / "download_log.json").read_text())
    kept = skipped = 0
    seqs = {"train": set(), "val": set()}
    other_images = collections.Counter()
    test_check = None
    stop_reason = "end of archive"
    t0 = time.time()
    while True:
        before = s.total
        head = s.read(30)
        if len(head) < 30 or head[:4] != b"PK\x03\x04":
            stop_reason = f"non-entry signature {head[:4]!r}"
            break
        _, flag, comp, _, _, _, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", head[4:])
        name = s.read(nlen).decode("utf-8", "replace")
        s.skip(elen) if elen else None
        if flag & 0x08 or csize == 0xFFFFFFFF:
            raise RuntimeError(f"unsupported zip entry (data descriptor / zip64): {name}")
        if test_check is None and "/test/" not in name and not name.endswith("/test/") and before > 0 and s.total >= stage2["bytes_consumed"]:
            # same point where the stage 2 download stopped: the stream so far must be byte-identical
            test_check = {"bytes": s.total, "sha256": s.sha.hexdigest(),
                          "matches_stage2_log": s.total == stage2["bytes_consumed"] and s.sha.hexdigest() == stage2["sha256_of_consumed_stream"]}
            print("test part of the stream:", test_check, flush=True)
        m = re.match(r"^(?:[^/]+/)*(train|val)/([^/]+)/(.+)$", name)
        keep = False
        if m and not name.endswith("/"):
            sp, seq, rest = m.groups()
            seqs[sp].add(seq)
            fm = IMG.match(rest)
            if fm:
                keep = (int(fm.group(1)) - 1) % STRIDE == 0
            elif re.search(r"\.(jpg|jpeg|png|bmp)$", rest, re.I):
                other_images[sp] += 1          # image with a non-numeric name: not kept, counted
            else:
                keep = True                    # label files
        if not keep:
            s.skip(csize)
            skipped += 1
            continue
        data = s.read(csize)
        if comp == 8:
            data = zlib.decompress(data, -15)
        elif comp != 0:
            raise RuntimeError(f"unsupported compression {comp}: {name}")
        assert len(data) == usize, name
        out = RAW / m.group(1) / m.group(2) / m.group(3)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        kept += 1
        if kept % 2000 == 0:
            print(f"kept {kept} files, train {len(seqs['train'])} / val {len(seqs['val'])} sequences, "
                  f"{s.total / 1e9:.2f} GB read, {(time.time() - t0) / 60:.1f} min", flush=True)
    log = {"parts": PARTS[: s.i + 1], "bytes_consumed": s.total, "sha256_of_consumed_stream": s.sha.hexdigest(),
           "stopped_at": stop_reason, "kept_files": kept, "skipped_entries": skipped,
           "sequences": {k: len(v) for k, v in seqs.items()}, "images_with_non_numeric_name": dict(other_images),
           "test_part_check": test_check, "frame_stride": STRIDE, "minutes": round((time.time() - t0) / 60, 1),
           "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    write_json(log, RAW / "download_log_stage3.json")
    print(json.dumps(log, indent=1)[:2000])


def _image_size(path):
    """(width, height) from the JPEG header, without decoding the picture."""
    b = path.read_bytes()
    i = 2
    while i < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        mk = b[i + 1]
        if mk in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h
        if mk in (0xD8, 0x01) or 0xD0 <= mk <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    raise ValueError(f"no SOF marker in {path}")


def manifest(sp):
    frames, per_seq = [], {}
    sizes = collections.Counter()
    for seq_dir in sorted((RAW / sp).iterdir()):
        if not seq_dir.is_dir():
            continue
        boxes, src = _read_boxes(seq_dir)
        jpgs = sorted(p for p in seq_dir.iterdir() if IMG.match(p.name))
        w, h = _image_size(jpgs[0])
        n_box = 0
        for p in jpgs:
            idx = int(p.stem) - 1                      # file 000001.jpg = frame index 0
            b = boxes[idx] if idx < len(boxes) else None
            gt = []
            if b is not None and b[2] > 0 and b[3] > 0:
                gt = [[b[0], b[1], b[0] + b[2], b[1] + b[3]]]
                sizes[next(n for n, lo, hi in BINS if lo <= size_of(gt[0]) < hi)] += 1
                n_box += 1
            frames.append({"seq": seq_dir.name, "scene": re.sub(r"_\d+$", "", seq_dir.name), "frame": idx,
                           "image": f"../raw/cst_antiuav/{sp}/{seq_dir.name}/{p.name}", "width": w, "height": h, "gt": gt})
        per_seq[seq_dir.name] = {"frames_kept": len(jpgs), "labels_in_file": len(boxes), "boxes_kept": n_box,
                                 "label_file": src, "resolution": f"{w}x{h}"}
    write_json(frames, ROOT / "data" / "manifests" / f"cst_{sp}.json")
    return {"sequences": len(per_seq), "frames": len(frames), "frames_with_box": sum(1 for f in frames if f["gt"]),
            "frames_without_box": sum(1 for f in frames if not f["gt"]), "boxes_by_size_sqrt_wh": dict(sizes),
            "sequences_by_scene": dict(collections.Counter(re.sub(r"_\d+$", "", s) for s in per_seq)),
            "resolutions": dict(collections.Counter(v["resolution"] for v in per_seq.values())), "per_sequence": per_seq}


def split_train(names):
    """Mechanical split by name only (pre-registration section 3): shuffle inside each scene with the fixed
    seed, then deal the sequences round-robin over the scenes; first 20 = hold, next 20 = adapt (first 5 = small)."""
    scene = lambda s: re.sub(r"_\d+$", "", s)
    by = collections.defaultdict(list)
    for n in sorted(names):
        by[scene(n)].append(n)
    rng = random.Random(SPLIT_SEED)
    for k in sorted(by):
        rng.shuffle(by[k])
    # proportional interleave: repeatedly take from the scene with the largest remaining share
    total = {k: len(v) for k, v in by.items()}
    taken = collections.Counter()
    order = []
    while len(order) < len(names):
        k = max((k for k in sorted(by) if taken[k] < total[k]), key=lambda k: (total[k] * (len(order) + 1) / len(names) - taken[k], k))
        order.append(by[k][taken[k]])
        taken[k] += 1
    hold, adapt = order[:N_HOLD], order[N_HOLD:N_HOLD + N_ADAPT]
    out = {"seed": SPLIT_SEED, "rule": split_train.__doc__.strip(), "n_train_sequences": len(names),
           "listing_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
           "cst_tr_hold": sorted(hold), "cst_tr_20": adapt, "cst_tr_5": adapt[:N_SMALL],
           "unused": sorted(order[N_HOLD + N_ADAPT:]),
           "scenes": {r: dict(collections.Counter(scene(s) for s in v)) for r, v in (("cst_tr_hold", hold), ("cst_tr_20", adapt), ("cst_tr_5", adapt[:N_SMALL]))}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="only rebuild the manifests and the split from files on disk")
    a = ap.parse_args()
    if not a.manifest:
        download()
    stats = {"dataset": "CST Anti-UAV (val and train splits)", "source": "github.com/PCwenyue/CST-Anti-UAV",
             "frame_stride": STRIDE, "fps_source": FPS}
    for sp in ("val", "train"):
        stats[sp] = manifest(sp)
    sp_path = ROOT / "docs" / "splits" / "cst_stage3_split.json"
    split = split_train(list(stats["train"]["per_sequence"]))
    if sp_path.exists():
        old = json.loads(sp_path.read_text())
        assert {k: old[k] for k in ("cst_tr_hold", "cst_tr_20", "cst_tr_5")} == {k: split[k] for k in ("cst_tr_hold", "cst_tr_20", "cst_tr_5")}, "frozen split differs"
    else:
        write_json(split, sp_path)
    log = RAW / "download_log_stage3.json"
    stats["download"] = json.loads(log.read_text()) if log.exists() else None
    write_json(stats, ROOT / "results" / "data" / "cst_stats_stage3.json")
    print(json.dumps({sp: {k: v for k, v in stats[sp].items() if k != "per_sequence"} for sp in ("val", "train")}, indent=1))
    print(json.dumps(split["scenes"]))


if __name__ == "__main__":
    main()
