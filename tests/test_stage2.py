"""Stage 2 tests: frozen split (leakage), per-GT tables vs greedy matching, tracker, track metrics."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tdd.metrics import operating_point
from tdd.stage2 import tables as T
from tdd.stage2.split import SEED, make_split

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "docs" / "splits" / "svanstrom_stage2_split.json").read_text())


# ---------------------------------------------------------------- split / leakage
def test_split_is_frozen_and_reproducible():
    rows = SPLIT["videos"]
    again = make_split({r["video"]: (r["cls"], r["bin"]) for r in rows}, SEED)
    assert all(again[r["video"]] == r["split"] for r in rows)
    listing = "\n".join(f"{r['video']} {r['split']}" for r in sorted(rows, key=lambda r: r["video"]))
    assert hashlib.sha256(listing.encode()).hexdigest() == SPLIT["listing_sha256"]
    assert SPLIT["listing_sha256"].startswith("49d0386086df6dec")      # value quoted in the pre-registration


def test_split_has_no_video_in_two_splits_and_is_stratified():
    rows = SPLIT["videos"]
    names = [r["video"] for r in rows]
    assert len(names) == len(set(names)) == 360
    assert {r["split"] for r in rows} == {"dev_a", "dev_b", "test"}
    per = collections.Counter((r["cls"], r["bin"], r["split"]) for r in rows)
    for cls, b in {(r["cls"], r["bin"]) for r in rows}:
        n = sum(per[(cls, b, s)] for s in ("dev_a", "dev_b", "test"))
        assert abs(per[(cls, b, "test")] - n / 2) <= 1          # half of every stratum is test
        assert per[(cls, b, "dev_a")] >= 1 and per[(cls, b, "dev_b")] >= 1


def test_adaptation_videos_come_only_from_dev_a():
    pytest.importorskip("torch")
    from tdd.stage2.train import pick_adapt_videos
    split = {r["video"]: r["split"] for r in SPLIT["videos"]}
    cls = {r["video"]: r["cls"] for r in SPLIT["videos"]}
    prev = set()
    for n in (4, 12, 37):
        drone, other = pick_adapt_videos(n)
        assert len(drone) == n and all(split[v] == "dev_a" for v in drone + other)
        assert all(cls[v] == "DRONE" for v in drone) and all(cls[v] != "DRONE" for v in other)
        assert prev <= set(drone)                               # nested: more videos, never different ones
        prev = set(drone)


def test_frame_sets_do_not_share_videos():
    man = ROOT / "data" / "manifests" / "svanstrom_ir.json"
    if not man.exists():
        pytest.skip("dataset manifests are not in the repository")
    from tdd.stage2.data import frame_set
    dev = {f["video"] for f in frame_set("svan_dev")}
    test = {f["video"] for f in frame_set("svan_test")}
    a, b = {f["video"] for f in frame_set("svan_dev_a")}, {f["video"] for f in frame_set("svan_dev_b")}
    assert not dev & test and not a & b and a | b == dev and len(dev | test) == 360


# ---------------------------------------------------------------- tables
def _random_frames(rng, n=300):
    frames, preds = [], []
    for i in range(n):
        has = rng.random() < 0.7
        x, y, w, h = rng.uniform(20, 200), rng.uniform(20, 200), rng.uniform(4, 40), rng.uniform(4, 40)
        gt = [[x, y, x + w, y + h]] if has else []
        dets = []
        for _ in range(rng.integers(0, 5)):
            if has and rng.random() < 0.6:
                j = rng.normal(0, 0.25 * w, 4)
                dets.append([x + j[0], y + j[1], x + w + j[2], y + h + j[3], float(rng.random())])
            else:
                a, b = rng.uniform(0, 280, 2)
                dets.append([a, b, a + rng.uniform(4, 40), b + rng.uniform(4, 40), float(rng.random())])
        frames.append({"video": f"v{i % 7}", "width": 320, "height": 256, "gt": gt, "cls": "DRONE"})
        preds.append(dets)
    return frames, preds


def test_per_gt_tables_equal_greedy_matching():
    rng = np.random.default_rng(0)
    frames, preds = _random_frames(rng)
    gt, ft = T.build(frames, preds, lambda f: f["video"])
    for thr in (0.0, 0.3, 0.7, 0.95):
        op = operating_point([{"gt": f["gt"], "det": p} for f, p in zip(frames, preds)], thr, 0.5)
        assert T.recall(gt, thr) == pytest.approx(op["recall"])
        empty = ~ft["has_gt"]
        assert T.far(ft, thr, empty) == pytest.approx(op["false_alarm_rate_empty_frames"])


def test_threshold_for_far_never_exceeds_target():
    rng = np.random.default_rng(1)
    ft = {"max_score": np.round(rng.random(500), 2)}              # many ties
    mask = np.ones(500, bool)
    for target in (0.0, 0.035, 0.1, 0.378, 1.0):
        thr = T.threshold_for_far(ft, mask, target)
        assert T.far(ft, thr, mask) <= target + 1e-12
        lower = np.nextafter(thr, -np.inf) - 0.0101                # one tie group lower would exceed it
        if thr > 0.02 and target < 1.0:
            assert T.far(ft, max(lower, 0.0), mask) > target


def test_scale_boxes_keeps_centre_and_score():
    out = T.scale_boxes([[[10, 20, 30, 40, 0.9]]], 1.5)[0][0]
    assert out == pytest.approx([5, 15, 35, 45, 0.9])
    assert T.scale_boxes([[[1, 2, 3, 4, 0.5]]], 1.0) == [[[1, 2, 3, 4, 0.5]]]


def test_tiles_cover_image_and_tta_merge_rule():
    pytest.importorskip("torch")
    from tdd.stage2.infer import merge_tta, tile_windows
    for w, h in ((1280, 1024), (960, 768)):
        cov = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in tile_windows(w, h, 2):
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            cov[y1:y2, x1:x2] = True
        assert cov.all()
    a = np.array([[10, 10, 30, 30, 0.9], [100, 100, 120, 120, 0.8]])
    b = np.array([[11, 10, 31, 30, 0.7]])
    m = merge_tta(a, b)
    assert len(m) == 2
    assert m[0, 4] == pytest.approx(0.8)          # seen by both passes: mean of 0.9 and 0.7
    assert m[1, 4] == pytest.approx(0.4)          # seen by one pass only: halved


# ---------------------------------------------------------------- tracker
def _dets(x, y, s=0.9, w=10, h=8):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2, s]


def test_mofn_confirmation_and_deletion():
    from tdd.stage2.tracker import CONFIRMED, TENTATIVE, MofN
    m = MofN(2, 3, 2)                       # created with one hit in one frame
    assert m.status == TENTATIVE
    assert m.step(False) and m.status == TENTATIVE
    assert m.step(True) and m.status == CONFIRMED          # 2 hits within 3 frames
    assert m.step(False)
    assert not m.step(False)                               # 2 consecutive misses -> deleted
    m = MofN(3, 4, 5)
    assert m.step(False)
    assert not m.step(False)                               # 3 of 4 no longer possible
    with pytest.raises(ValueError):
        MofN(4, 3, 2)


def test_tracker_follows_a_moving_target_with_one_id():
    from tdd.stage2.tracker import TrackerParams, track_video
    rng = np.random.default_rng(0)
    dt = 1 / 6
    times = [i * dt for i in range(40)]
    dets = []
    for i, t in enumerate(times):
        x, y = 100 + 30 * t + rng.normal(0, 1), 200 - 12 * t + rng.normal(0, 1)
        frame = [] if i in (15, 16) else [_dets(x, y)]                       # two missed frames
        frame += [_dets(rng.uniform(0, 640), rng.uniform(0, 512), 0.3) for _ in range(2)]   # clutter
        dets.append(frame)
    p = TrackerParams(q=220.0, confirm_m=3, confirm_n=4, delete_misses=3, det_threshold=0.1, coast_outputs=2)
    out = track_video(times, dets, p)
    target_ids = set()
    for i, t in enumerate(times[3:], start=3):
        x, y = 100 + 30 * t, 200 - 12 * t
        near = [r for r in out[i] if abs((r[0] + r[2]) / 2 - x) < 8 and abs((r[1] + r[3]) / 2 - y) < 8]
        assert near, f"no output near the target at frame {i}"
        target_ids.add(near[0][5])
    assert len(target_ids) == 1                       # the gap is bridged without a new identity
    assert not out[0]                                 # nothing is output before confirmation
    assert any(not r[6] for r in out[15])             # coasted (predicted) box during the gap
    clutter_tracks = {r[5] for o in out for r in o} - target_ids
    assert len(clutter_tracks) <= 2                   # random clutter rarely gets 3 hits in 4 frames


def test_tracker_gate_and_det_threshold():
    from tdd.stage2.tracker import TrackerParams, track_video
    p = TrackerParams(q=57.0, confirm_m=2, confirm_n=3, delete_misses=2, det_threshold=0.5)
    times = [0, 1 / 6, 2 / 6]
    # second detection is 200 px away: outside the gate, so no track is confirmed
    out = track_video(times, [[_dets(100, 100)], [_dets(300, 100)], []], p)
    assert all(not o for o in out)
    # detections below the tracker's score threshold are ignored
    out = track_video(times, [[_dets(100, 100, 0.4)], [_dets(101, 100, 0.4)], [_dets(102, 100, 0.4)]], p)
    assert all(not o for o in out)
    out = track_video(times, [[_dets(100, 100, 0.9)], [_dets(101, 100, 0.7)], [_dets(102, 100, 0.8)]], p)
    assert out[1] and out[1][0][4] == pytest.approx(0.8)      # track score = mean detection score
    assert out[2][0][5] == out[1][0][5]


def test_sequence_metrics_latency_maintenance_and_false_tracks():
    from tdd.stage2.track_eval import sequence_metrics
    box = [10.0, 10.0, 30.0, 30.0]
    frames, outs, ids = [], [], []
    for i in range(10):                                  # drone video: found from the 3rd labelled frame, one miss later
        frames.append({"video": "IR_DRONE_001", "cls": "DRONE", "frame": 5 * i, "image": "svanstrom/IR_DRONE_001/x.jpg",
                       "width": 320, "height": 256, "gt": [box]})
        hit = i >= 2 and i != 6
        outs.append([[*box, 0.9]] if hit else [])
        ids.append([1 if i < 6 else 2] if hit else [])
    for i in range(12):                                  # bird video: two false tracks
        frames.append({"video": "IR_BIRD_001", "cls": "BIRD", "frame": 5 * i, "image": "svanstrom/IR_BIRD_001/x.jpg",
                       "width": 320, "height": 256, "gt": []})
        outs.append([[50.0, 50.0, 60.0, 60.0, 0.8]] if i in (1, 2, 8) else [])
        ids.append([7] if i in (1, 2) else [9] if i == 8 else [])
    m = sequence_metrics(frames, outs, ids, 0.5)
    assert m["center"]["first_found_latency_s_median"] == pytest.approx(2 * 5 / 30)
    assert m["center"]["maintenance_after_first"] == pytest.approx(6 / 7)
    assert m["center"]["never_found_videos"] == 0
    assert m["track_ids_per_found_drone_video_mean"] == 2
    assert m["false_confirmed_tracks"] == 2
    assert m["false_confirmed_tracks_per_minute"] == pytest.approx(2 / (12 * (5 / 30) / 60))
    assert sequence_metrics(frames, outs, ids, 0.95)["center"]["never_found_videos"] == 1


def test_paired_bootstrap_of_identical_configurations_is_zero():
    from tdd.stage2.evalset import SvanEval, bootstrap
    rng = np.random.default_rng(3)
    frames, preds = _random_frames(rng, 200)
    for i, f in enumerate(frames):
        f["bin"] = "CLOSE"
        if i % 3 == 0:
            f["cls"] = "BIRD"
    ev = SvanEval(frames, preds)
    b = bootstrap(ev, 0.5, ev, 0.5, B=50)
    assert all(v["diff"] == 0 and v["ci95"] == [0.0, 0.0] for v in b["paired_diff"].values())
    lo, hi = b["ci95"]["recall_iou50"]
    assert lo <= ev.metrics(0.5)["recall_iou50"] <= hi


def test_dev_halves_partition_dev_and_cache_fallback(tmp_path, monkeypatch):
    if not (ROOT / "data" / "manifests" / "svanstrom_ir.json").exists():
        pytest.skip("dataset manifests are not in the repository")
    """dev = dev10 + dev5 (disjoint); a frame set is served from complete caches of other sets."""
    import json

    from tdd.stage2 import data

    dev, a, b = data.frame_set("svan_dev"), data.frame_set("svan_dev10"), data.frame_set("svan_dev5")
    ia, ib = {f["image"] for f in a}, {f["image"] for f in b}
    assert not ia & ib and ia | ib == {f["image"] for f in dev}
    monkeypatch.setattr(data, "PREDS", tmp_path)
    for name, fr in (("svan_dev10", a), ("svan_dev5", b)):
        with open(data.cache_path("t", name), "w") as fh:
            for f in fr:
                fh.write(json.dumps([[0, 0, 1, 1, (f["frame"] % 97) / 100]]) + "\n")
    got = data.load_preds("t", "svan_dev")
    assert len(got) == len(dev) and all(p[0][4] == (f["frame"] % 97) / 100 for f, p in zip(dev, got))
    (tmp_path / "t__svan_dev5.jsonl").unlink()
    try:
        data.load_preds("t", "svan_dev")
        raise AssertionError("incomplete coverage must fail")
    except FileNotFoundError:
        pass
