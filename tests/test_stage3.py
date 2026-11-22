"""Stage 3: evaluation add-ons, the independent re-computation, the CST train split rule, inference guards."""
import importlib.util
import json
import math
import pathlib
import random

import numpy as np
import pytest

from tdd.stage3 import data as d3
from tdd.stage3.evalx import Ev3, bootstrap3
from tdd.stage3.prepare_cst import split_train

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("indep3", ROOT / "scripts" / "independent_check_stage3.py")
indep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indep)


def synthetic(n_seq=6, n=40, seed=3):
    rng = random.Random(seed)
    frames, preds = [], []
    for s in range(n_seq):
        for i in range(n):
            size = rng.choice([4, 6, 10, 14, 24, 40])
            x, y = rng.uniform(20, 500), rng.uniform(20, 400)
            gt = [] if rng.random() < 0.2 else [[x, y, x + size, y + size * rng.uniform(0.7, 1.3)]]
            det = []
            if gt and rng.random() < 0.7:
                j = rng.uniform(-0.4, 0.4) * size
                g = gt[0]
                det.append([g[0] + j, g[1] + j, g[2] + j * rng.uniform(0.5, 1.5), g[3] + j, rng.random()])
            for _ in range(rng.choice([0, 0, 1, 2])):
                a, b = rng.uniform(0, 600), rng.uniform(0, 480)
                det.append([a, b, a + rng.uniform(3, 30), b + rng.uniform(3, 30), rng.random()])
            frames.append({"seq": f"s{s}", "frame": i, "image": f"s{s}/{i}.jpg", "width": 640, "height": 512, "gt": gt})
            preds.append(det)
    return frames, preds


def test_independent_script_matches_pipeline():
    frames, preds = synthetic()
    thr = 0.35
    m = Ev3(frames, preds, thr).metrics3()
    mine = indep.evaluate(frames, preds, thr)
    assert mine["ap50"] == pytest.approx(m["ap50"], abs=1e-9)
    assert mine["recall_iou50"] == pytest.approx(m["recall_iou50"], abs=1e-12)
    assert mine["recall_center"] == pytest.approx(m["recall_center"], abs=1e-12)
    for b in ("tiny", "small"):
        assert mine[f"n_{b}"] == m["recall_by_size"][b]["total"] > 0
        assert mine[f"recall_{b}_iou50"] == pytest.approx(m["recall_by_size"][b]["iou50"], abs=1e-12)
        assert mine[f"recall_{b}_center"] == pytest.approx(m["recall_by_size"][b]["center"], abs=1e-12)


def test_bin_bootstrap_point_estimates_and_pairing():
    frames, preds = synthetic()
    ev = Ev3(frames, preds, 0.35)
    m = ev.metrics3()
    s = ev._stat3(np.asarray(ev.seqs))
    assert s["recall_tiny_iou50"] == pytest.approx(m["recall_by_size"]["tiny"]["iou50"])
    assert s["recall_small_center"] == pytest.approx(m["recall_by_size"]["small"]["center"])
    b = bootstrap3(ev, ev, B=50)
    assert all(v["diff"] == 0 and v["ci95"] == [0.0, 0.0] for v in b["paired_diff"].values())


def test_far10_readout_keeps_alarm_rate_at_or_below_ten_percent():
    frames, preds = synthetic(n_seq=8, n=60)
    r = Ev3(frames, preds, 0.35).metrics3()["at_far10_same_frames"]
    assert r["far_drone_free"] <= 0.10 + 1e-12


def test_cst_train_split_is_deterministic_disjoint_and_nested():
    names = [f"{sc}_{i}" for sc, n in (("building", 30), ("jungle", 40), ("water", 10), ("urban-areas", 40)) for i in range(n)]
    a, b = split_train(names), split_train(list(reversed(names)))
    assert a["cst_tr_hold"] == b["cst_tr_hold"] and a["cst_tr_20"] == b["cst_tr_20"]
    assert len(a["cst_tr_hold"]) == 20 and len(a["cst_tr_20"]) == 20 and a["cst_tr_5"] == a["cst_tr_20"][:5]
    assert not set(a["cst_tr_hold"]) & set(a["cst_tr_20"])
    assert len(set(a["cst_tr_hold"]) | set(a["cst_tr_20"]) | set(a["unused"])) == len(names)
    assert set(a["scenes"]["cst_tr_hold"]) == {"building", "jungle", "water", "urban-areas"}     # every scene is represented


def test_reduced_val_boxes_are_quartered():
    if not (d3.d2.MAN / "antiuav_val.json").exists():
        pytest.skip("data not present")
    full, q = d3.d2.frame_set("auv_val"), d3.frame_set("auv_val_q")
    assert len(full) == len(q) and q[0]["pre_scale"] == 0.25 and q[0]["width"] == full[0]["width"] // 4
    assert q[0]["gt"][0] == [v / 4 for v in full[0]["gt"][0]]
    assert "pre_scale" not in full[0]                                 # the stage 2 set is not modified


def test_cst_val_is_refused_before_the_freeze(tmp_path, monkeypatch):
    from tdd.stage3 import infer
    monkeypatch.setattr(infer, "OUT", tmp_path)
    with pytest.raises(SystemExit):
        infer.guard("S1", "cst_val")
    with pytest.raises(SystemExit):
        infer.guard("B", "cst_tr_hold")          # a track 1 model on CST train before track 1 is frozen
    (tmp_path / "frozen_track1.json").write_text("{}")
    infer.guard("B", "cst_tr_hold")
    (tmp_path / "frozen_config.json").write_text("{}")
    infer.guard("S1", "cst_val")
