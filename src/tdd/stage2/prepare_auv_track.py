"""Stage 2: Anti-UAV test frames for the tracker regression check (pre-registration 7).

First 300 frames of each of the 91 official test sequences, every 3rd frame
(20 fps -> 6.7 Hz). Frames with exist=1 but no usable box are kept (the tracker needs
consecutive frames) and flagged ``invalid``: they count neither as positive nor as
drone-free in any metric.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.prepare_auv_track
  -> data/frames/antiuav/<seq>/<frame>.jpg (shared with stage 1), data/manifests/antiuav_test_track.json
"""
from __future__ import annotations

import cv2

from ..common import ROOT, load_config, path, read_json, write_json
from ..prepare_antiuav import gt_for, load_seq, source_of

STRIDE, FIRST = 3, 300


def main():
    cfg = load_config("configs/full.yaml")
    root = path(cfg["data"]["antiuav_root"])
    frames_root = path(cfg["data"]["frames_dir"]) / "antiuav"
    seqs = sorted({f["seq"] for f in read_json(ROOT / "data" / "manifests" / "antiuav_test.json")})
    manifest, counts = [], {"present": 0, "absent": 0, "invalid": 0}
    for seq in seqs:
        ann, cap, n, wh = load_seq(root, f"test/{seq}")
        n = min(n, len(ann["exist"]), FIRST)
        i = 0
        while i < n and cap.grab():
            if i % STRIDE == 0:
                status, box = gt_for(ann, i)
                out = frames_root / seq / f"{i:04d}.jpg"
                if not out.exists():
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), [cv2.IMWRITE_JPEG_QUALITY, 95])
                counts[status] += 1
                manifest.append({"seq": seq, "source": source_of(seq), "frame": i, "image": f"antiuav/{seq}/{i:04d}.jpg",
                                 "width": wh[0], "height": wh[1], "gt": [box] if box is not None else [],
                                 "invalid": status == "invalid"})
            i += 1
        cap.release()
    write_json(manifest, ROOT / "data" / "manifests" / "antiuav_test_track.json")
    print(len(seqs), "sequences", len(manifest), "frames", counts)


if __name__ == "__main__":
    main()
