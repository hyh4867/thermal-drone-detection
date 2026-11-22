"""Stage 2 demo: one public clip with detections and tracks (pre-registration section 8).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.demo --model C37
  -> results/stage2/demo/demo.mp4 (NOT in git) and results/stage2/demo/demo.json (clip choice, settings)

The clip is not chosen by looking at results of the shown configuration: it is the
dev_b MEDIUM drone video whose stage 1 per-video recall (threshold 0.954, IoU 0.5) is
closest to the median over all dev_b drone videos. Detections come from the cached
6 Hz predictions of the frozen configuration (every 5th frame); a box is held on
screen until the next processed frame. The 10 s clip is played at half speed.
One clip is an illustration, not a performance summary - the video says so on screen.
"""
from __future__ import annotations

import argparse
import collections

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..common import ROOT, read_json, write_json
from . import tables as T
from .data import frame_set, load_preds
from .diagnose import s1_threshold
from .track_eval import run_tracker
from .tracker import TrackerParams

OUT = ROOT / "results" / "stage2" / "demo"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
ZOOM, PANEL, STEP = 3, 132, 5
MAX_SECONDS = 20.0      # output length cap (the request was 15-20 s); the source clip is cut at the end
C_GT, C_DET, C_TRK, INK, MUTED, BG = (200, 200, 196), (235, 104, 52), (42, 120, 214), (245, 245, 243), (170, 169, 164), (17, 17, 17)


def choose_clip():
    frames = frame_set("svan_dev_b")
    preds = load_preds("s1", "svan_dev_b")
    thr0 = s1_threshold()
    gt, _ = T.build(frames, preds, lambda f: f["video"], positive=lambda f: f["cls"] == "DRONE")
    per = {}
    for v in np.unique(gt["unit"]):
        m = gt["unit"] == v
        per[str(v)] = float((gt["iou50"][m] >= thr0).mean())
    bins = {f["video"]: f["bin"] for f in frames}
    med = float(np.median(list(per.values())))
    cands = sorted((abs(r - med), v) for v, r in per.items() if bins[v] == "MEDIUM")
    video = cands[0][1]
    return video, {"rule": "dev_b MEDIUM drone video whose stage 1 per-video recall (thr 0.954, IoU 0.5) is closest to the dev_b drone-video median",
                   "dev_b_drone_videos": len(per), "median_recall": med, "video": video, "video_recall_stage1": per[video],
                   "medium_candidates": sum(1 for v in per if bins[v] == "MEDIUM")}


def _text(draw, xy, s, size, fill):
    draw.text(xy, s, font=ImageFont.truetype(FONT, size), fill=fill)


def _box(draw, b, color, width, dash=0):
    x1, y1, x2, y2 = [float(v) * ZOOM for v in b[:4]]
    y1, y2 = y1 + PANEL, y2 + PANEL
    if not dash:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(1, int(np.hypot(bx - ax, by - ay) // (2 * dash)))
        for i in range(n):
            t0, t1 = i / n, (i + 0.5) / n
            draw.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name in results/stage2/frozen_config.json")
    ap.add_argument("--config", default="frozen_config")
    ap.add_argument("--label", default=None, help="on-screen description of the configuration")
    ap.add_argument("--why", default=None, help="how the shown configuration was chosen (recorded in demo.json)")
    a = ap.parse_args()
    fz = read_json(ROOT / "results" / "stage2" / f"{a.config}.json")
    m = fz["models"][a.model]
    video, why = choose_clip()
    frames_all = frame_set("svan_dev_b")
    preds_all = load_preds(m["svanstrom_tag"], "svan_dev_b")
    sel = [i for i, f in enumerate(frames_all) if f["video"] == video]
    frames = [frames_all[i] for i in sel]
    preds = [preds_all[i] for i in sel]
    tp = TrackerParams(**m["tracker"]["params"])
    outs, ids = run_tracker(frames, preds, tp, m["k"])
    dets = T.scale_boxes(preds, m["k"])
    thr_d, thr_t = m["thresholds"]["T_base"], m["tracker"]["track_score_thresholds"]["T_base"]
    by_frame = {f["frame"]: j for j, f in enumerate(frames)}

    src = sorted((ROOT / "data" / "raw" / "svanstrom").rglob(f"{video}.mp4"))
    assert src, f"raw video {video}.mp4 not found"
    cap = cv2.VideoCapture(str(src[0]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT.mkdir(parents=True, exist_ok=True)
    size = (W * ZOOM, H * ZOOM + PANEL)
    path = OUT / "demo.mp4"
    wr = None
    for codec in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps / 2, size)
        if wr.isOpened():
            break
    assert wr is not None and wr.isOpened(), "no usable mp4 codec"
    label = a.label or f"구성 {a.model}"
    trails = collections.defaultdict(list)
    n, j, stats = 0, None, {"frames_with_label": 0, "label_frames_with_track_on_target": 0, "label_frames_with_detection_on_target": 0, "track_ids": set()}
    while True:
        ok, bgr = cap.read()
        if not ok or n >= int(MAX_SECONDS * fps / 2):
            break
        if n in by_frame:
            j = by_frame[n]
            for b, tid in zip(outs[j], ids[j]):
                if b[4] >= thr_t:
                    trails[tid].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                    stats["track_ids"].add(tid)
            if frames[j]["gt"]:
                g = frames[j]["gt"][0]
                inside = lambda b: g[0] <= (b[0] + b[2]) / 2 <= g[2] and g[1] <= (b[1] + b[3]) / 2 <= g[3]
                stats["frames_with_label"] += 1
                stats["label_frames_with_track_on_target"] += any(b[4] >= thr_t and inside(b) for b in outs[j])
                stats["label_frames_with_detection_on_target"] += any(b[4] >= thr_d and inside(b) for b in dets[j])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, (W * ZOOM, H * ZOOM), interpolation=cv2.INTER_NEAREST)
        canvas = Image.new("RGB", size, BG)
        canvas.paste(Image.fromarray(big).convert("RGB"), (0, PANEL))
        d = ImageDraw.Draw(canvas)
        _text(d, (16, 10), "클립 하나이며 성능 요약이 아님", 26, INK)
        _text(d, (16, 46), f"0.5배속 · 탐지·추적은 5프레임마다(6 Hz) 갱신 · {label}", 17, MUTED)
        _text(d, (16, 70), f"출처: Svanström et al., Drone-detection-dataset v1.0.0 (Zenodo 5500576), {video}.mp4 (IR 320×256)", 15, MUTED)
        x = 16
        for color, name, dash in ((C_GT, "라벨", 4), (C_DET, "검출(임계값 이상)", 0), (C_TRK, "확정 항적", 0)):
            if dash:
                for k in range(3):
                    d.line([x + k * 10, 110, x + k * 10 + 5, 110], fill=color, width=2)
            else:
                d.line([x, 110, x + 26, 110], fill=color, width=3)
            _text(d, (x + 34, 100), name, 15, INK)
            x += 34 + int(d.textlength(name, font=ImageFont.truetype(FONT, 15))) + 28
        _text(d, (size[0] - 150, 100), f"t = {n / fps:4.1f} s", 15, MUTED)
        if j is not None:
            for g in frames[j]["gt"]:
                _box(d, g, C_GT, 1, dash=4)
            for b in dets[j]:
                if b[4] >= thr_d:
                    _box(d, b, C_DET, 2)
            for b, tid in zip(outs[j], ids[j]):
                if b[4] < thr_t:
                    continue
                pts = [(px * ZOOM, py * ZOOM + PANEL) for px, py in trails[tid][-12:]]
                if len(pts) > 1:
                    d.line(pts, fill=C_TRK, width=2)
                grown = [b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2]
                _box(d, grown, C_TRK, 2)
                tx, ty = grown[0] * ZOOM, grown[1] * ZOOM + PANEL - 24
                d.rectangle([tx, ty, tx + 58, ty + 21], fill=C_TRK)
                _text(d, (tx + 5, ty + 1), f"항적 {tid}", 15, INK)
        wr.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        n += 1
    cap.release()
    wr.release()
    stats["track_ids"] = len(stats["track_ids"])
    info = {"clip": why, "model": a.model, "model_choice": a.why, "svanstrom_tag": m["svanstrom_tag"], "k": m["k"], "detector_threshold": thr_d,
            "track_score_threshold": thr_t, "tracker": m["tracker"]["params"], "source_fps": fps, "source_frames": n,
            "output": {"file": "results/stage2/demo/demo.mp4 (not in git)", "fps": fps / 2, "seconds": n / (fps / 2), "size": list(size)},
            "this_clip_only": stats,
            "note": "One clip, half speed. Illustration only, not a performance summary."}
    write_json(info, OUT / "demo.json")
    print(info["clip"], info["output"], stats)


if __name__ == "__main__":
    main()
