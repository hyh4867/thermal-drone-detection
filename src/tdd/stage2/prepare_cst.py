"""Stage 2: third dataset (final, never used for any choice) - CST Anti-UAV, TEST split only.

Source   github.com/PCwenyue/CST-Anti-UAV (Xie et al., ICCVW 2025, arXiv:2507.23473).
Terms    DATASET_TERMS.md in that repository: academic and non-commercial research
         only; results and statistics may be published with attribution. No image
         or label of this dataset is committed to git.
Download public Google Drive folders (no login): an 11-part split zip, 21.7 GB.
         The test split is stored first in the archive, so this script STREAMS the
         parts in order, keeps the label files and every ``STRIDE``-th test frame,
         and stops at the first entry outside ``test/``. Nothing else is stored.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.prepare_cst            (download + manifest)
       PYTHONPATH=src .venv/bin/python -m tdd.stage2.prepare_cst --manifest (manifest only)

Outputs (all git-ignored except the stats file)
  data/raw/cst_antiuav/test/<seq>/...        kept frames (original JPEG bytes) + label files
  data/raw/cst_antiuav/download_log.json     byte count and SHA-256 of the consumed stream
  data/manifests/cst_test.json               frames with boxes (x1,y1,x2,y2)
  results/data/cst_stats.json                counts, size distribution (no images)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import struct
import time
import urllib.request
import zlib

from ..common import ROOT, write_json
from ..metrics import size_of

# Google Drive file ids, in archive order (.z01 ... .z10, then .zip), listed 2026-11-19.
PARTS = [
    ("CST-AntiUAV.z01", "12il79A75af_ahfQEakKtzrA3LEqWTn2G"),
    ("CST-AntiUAV.z02", "1bWscp-86dgeZ8mLUNQkoLCzLZGA7wvkt"),
    ("CST-AntiUAV.z03", "1V6jql4Kiq6ReTv61sEh3Rx8Yxi2ZYcJw"),
    ("CST-AntiUAV.z04", "1pq7XRB34DI3oMANx1HqU8M3Ah11VJESa"),
    ("CST-AntiUAV.z05", "1XRkXugzW7Aije4vy6dEZHs1fIy6-b5i9"),
    ("CST-AntiUAV.z06", "1PJgfgMbQpQ5kmYaGw51qwdOAUxc03Wt9"),
    ("CST-AntiUAV.z07", "1HMfUlGTOhinMPKRwmt9gYfPZq8OV28dl"),
    ("CST-AntiUAV.z08", "1YlL527rZ6SYgIj-8IN-Mb1vhqJe_kvG5"),
    ("CST-AntiUAV.z09", "1QvqlSLm_XuPBnpYDcFDBJxY7UG0MKoZ_"),
    ("CST-AntiUAV.z10", "1zCALY_qU2KI8NZF_pQ-kSNVP0nCNpR4l"),
    ("CST-AntiUAV.zip", "14zO4bcudHkY2DxmbLyiRKlHpUg6vIv0d"),
]
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
STRIDE = 4            # 25 fps -> 6.25 Hz, close to the 6 Hz used on Svanström (30 fps, every 5th)
RAW = ROOT / "data" / "raw" / "cst_antiuav"
FPS = 25.0


class PartStream:
    """Sequential reader over the concatenated parts, with Range-based retry."""

    def __init__(self):
        self.i, self.off, self.resp = 0, 0, None
        self.total, self.sha = 0, hashlib.sha256()

    def _open(self):
        name, fid = PARTS[self.i]
        req = urllib.request.Request(URL.format(id=fid), headers={"Range": f"bytes={self.off}-", "User-Agent": "curl/8"})
        self.resp = urllib.request.urlopen(req, timeout=120)
        ctype = self.resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError(f"{name}: got an HTML page instead of data (quota or sign-in wall)")

    def read(self, n: int) -> bytes:
        out = bytearray()
        fails = 0
        while len(out) < n:
            try:
                if self.resp is None:
                    self._open()
                chunk = self.resp.read(min(n - len(out), 1 << 20))
            except RuntimeError:
                raise
            except Exception as e:  # network error: reopen at the same offset
                fails += 1
                if fails > 20:
                    raise RuntimeError(f"too many network errors: {e!r}")
                self.resp = None
                time.sleep(min(60, 5 * fails))
                continue
            if not chunk:  # end of this part
                self.resp = None
                self.i += 1
                self.off = 0
                if self.i >= len(PARTS):
                    break
                continue
            fails = 0
            out += chunk
            self.off += len(chunk)
        self.total += len(out)
        self.sha.update(out)
        return bytes(out)

    def skip(self, n: int):
        while n > 0:
            k = min(n, 4 << 20)
            if len(self.read(k)) < k:
                raise EOFError
            n -= k


def download():
    s = PartStream()
    assert s.read(4) == b"PK\x07\x08", "expected the split-archive marker"
    kept = skipped = 0
    seqs = set()
    stop_reason = "end of archive"
    t0 = time.time()
    while True:
        head = s.read(30)
        if len(head) < 30 or head[:4] != b"PK\x03\x04":
            stop_reason = f"non-entry signature {head[:4]!r}"
            break
        _, flag, comp, _, _, _, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", head[4:])
        name = s.read(nlen).decode("utf-8", "replace")
        s.skip(elen) if elen else None
        if flag & 0x08 or csize == 0xFFFFFFFF:
            raise RuntimeError(f"unsupported zip entry (data descriptor / zip64): {name}")
        m = re.match(r"^(?:[^/]+/)*test/([^/]+)/(.+)$", name)
        if "/test/" not in name and not name.endswith("/test/") and seqs:
            stop_reason = f"first entry outside test/: {name}"
            break
        keep = False
        if m and not name.endswith("/"):
            seq, rest = m.group(1), m.group(2)
            seqs.add(seq)
            fm = re.match(r"^(\d+)\.jpg$", rest)
            keep = (int(fm.group(1)) - 1) % STRIDE == 0 if fm else True
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
        out = RAW / "test" / m.group(1) / m.group(2)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        kept += 1
        if kept % 500 == 0:
            print(f"kept {kept} files, {len(seqs)} sequences, {s.total / 1e9:.2f} GB read, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    log = {"parts": PARTS[: s.i + 1], "bytes_consumed": s.total, "sha256_of_consumed_stream": s.sha.hexdigest(),
           "stopped_at": stop_reason, "kept_files": kept, "skipped_entries": skipped, "sequences": len(seqs),
           "frame_stride": STRIDE, "minutes": round((time.time() - t0) / 60, 1),
           "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    write_json(log, RAW / "download_log.json")
    print(json.dumps(log, indent=1)[:1500])


def _read_boxes(seq_dir):
    """Returns (list of [x,y,w,h] or None per frame, source label file)."""
    j = seq_dir / "IR_label.json"
    if j.exists():
        d = json.loads(j.read_text())
        rects = d["gt"] if "gt" in d else d["gt_rect"]
        exist = d.get("exist", [1] * len(rects))
        return [r if e and r and len(r) == 4 else None for r, e in zip(rects, exist)], j.name
    g = seq_dir / "gt.txt"
    if g.exists():
        rows = [[float(v) for v in re.split(r"[,\s]+", ln.strip())] for ln in g.read_text().splitlines() if ln.strip()]
        e = seq_dir / "exist.txt"
        exist = [int(float(x)) for x in e.read_text().split()] if e.exists() else [1] * len(rows)
        return [r if ex else None for r, ex in zip(rows, exist)], g.name
    raise FileNotFoundError(f"no label file in {seq_dir}")


def manifest():
    import cv2

    frames, per_seq = [], {}
    sizes = collections.Counter()
    bins = [("tiny", 0, 8), ("small", 8, 16), ("medium", 16, 32), ("large", 32, 64), ("xlarge", 64, 1e9)]
    for seq_dir in sorted((RAW / "test").iterdir()):
        if not seq_dir.is_dir():
            continue
        boxes, src = _read_boxes(seq_dir)
        jpgs = sorted(seq_dir.glob("*.jpg"))
        img = cv2.imread(str(jpgs[0]), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        n_box = 0
        for p in jpgs:
            idx = int(p.stem) - 1                      # file 000001.jpg = frame index 0
            b = boxes[idx] if idx < len(boxes) else None
            gt = []
            if b is not None and b[2] > 0 and b[3] > 0:
                gt = [[b[0], b[1], b[0] + b[2], b[1] + b[3]]]
                sz = size_of(gt[0])
                sizes[next(n for n, lo, hi in bins if lo <= sz < hi)] += 1
                n_box += 1
            frames.append({"seq": seq_dir.name, "scene": re.sub(r"_\d+$", "", seq_dir.name), "frame": idx,
                           "image": f"../raw/cst_antiuav/test/{seq_dir.name}/{p.name}", "width": w, "height": h, "gt": gt})
        per_seq[seq_dir.name] = {"frames_kept": len(jpgs), "labels_in_file": len(boxes), "boxes_kept": n_box,
                                 "label_file": src, "resolution": f"{w}x{h}"}
    write_json(frames, ROOT / "data" / "manifests" / "cst_test.json")
    log = json.loads((RAW / "download_log.json").read_text()) if (RAW / "download_log.json").exists() else None
    stats = {"dataset": "CST Anti-UAV (test split only)", "source": "github.com/PCwenyue/CST-Anti-UAV",
             "frame_stride": STRIDE, "fps_source": FPS, "sequences": len(per_seq), "frames": len(frames),
             "frames_with_box": sum(1 for f in frames if f["gt"]),
             "frames_without_box": sum(1 for f in frames if not f["gt"]),
             "boxes_by_size_sqrt_wh": dict(sizes),
             "sequences_by_scene": dict(collections.Counter(re.sub(r"_\d+$", "", s) for s in per_seq)),
             "per_sequence": per_seq, "download": log}
    write_json(stats, ROOT / "results" / "data" / "cst_stats.json")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_sequence", "download")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="only rebuild the manifest from files on disk")
    a = ap.parse_args()
    if not a.manifest:
        download()
    manifest()
