"""Seed study: guards of the test inference, the frozen-file rule, the spread summary, the independent script."""
import importlib.util
import pathlib
import subprocess

import pytest

from tdd.seeds import data as D
from tdd.seeds import evaluate as E
from tdd.seeds import infer as I

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_names_and_paths_keep_stage3_apart():
    assert D.tag(0) == "t2_20_m1" and D.model(0) == "T2_20" and D.ckpt(0) == "runs/stage3/t2_20/last.pt"
    assert D.tag(1) == "t2_20s1_m1" and D.model(2) == "T2_20s2" and D.ckpt(2) == "runs/seeds/t2_20_s2/last.pt"
    assert D.cache_path(D.tag(1), "cst_val").parent == D.PREDS
    assert D.cache_path(D.tag(0), "cst_val").parent == D.d3.PREDS and D.cache_path("s1_m1", "cst_val").parent == D.d3.PREDS
    assert "stage3" not in str(D.OUT) and "stage3" not in str(D.PREDS)


def test_test_sets_are_refused_until_the_frozen_file_is_committed(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path)                      # no frozen file at all
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "frozen_path", lambda s: tmp_path / f"frozen_seed{s}.json")
    I.guard("T2_20s1", "cst_tr_hold")                            # the hold-out set is always allowed
    for s in D.TEST_SETS:
        with pytest.raises(SystemExit):
            I.guard("T2_20s1", s)
    (tmp_path / "frozen_seed1.json").write_text("{}")            # exists but is not committed
    assert not D.frozen_committed(1)
    for s in D.TEST_SETS:
        with pytest.raises(SystemExit):
            I.guard("T2_20s1", s)
    with pytest.raises(SystemExit):
        I.guard("T2_20", "cst_val")                              # seed 0 is never inferred again here
    with pytest.raises(SystemExit):
        I.guard("T2_20s1", "cst_test_h")                         # a set outside the study


def test_frozen_committed_needs_tracked_and_unmodified(tmp_path, monkeypatch):
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    p = tmp_path / "results" / "seeds" / "frozen_seed1.json"
    p.parent.mkdir(parents=True)
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "frozen_path", lambda s: p)
    assert not D.frozen_committed(1)
    p.write_text('{"threshold": 0.9}')
    assert not D.frozen_committed(1)                             # untracked
    run("add", "-A")
    assert not D.frozen_committed(1)                             # staged only
    run("commit", "-q", "-m", "freeze")
    assert D.frozen_committed(1)
    p.write_text('{"threshold": 0.5}')
    assert not D.frozen_committed(1)                             # changed after the commit


def test_checkpoint_must_equal_the_frozen_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "frozen_committed", lambda s: True)
    monkeypatch.setattr(D, "frozen", lambda s: {"checkpoint_sha256": "a" * 64})
    monkeypatch.setattr(I.base, "sha256_file", lambda p: "b" * 64)
    with pytest.raises(SystemExit):
        I.guard("T2_20s2", "cst_val")
    monkeypatch.setattr(I.base, "sha256_file", lambda p: "a" * 64)
    I.guard("T2_20s2", "cst_val")


def test_spread_and_prediction_checks():
    s = E.spread({0: 0.40, 1: 0.44, 2: 0.36})
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.40) and s["range"] == pytest.approx(0.08) and s["sd_n_minus_1"] == pytest.approx(0.04)
    tab = {"cst_val.ap50": s, "svanstrom.far": E.spread({0: 0.73, 1: 0.61, 2: 0.59}),
           "antiuav.far_drone_free_minus_S1": E.spread({0: 0.034, 1: 0.03})}
    p = E.predictions(tab, {"0": {"J5_regression": True}, "1": {"J5_regression": True}})
    r = p["reviewer"]
    assert r["R1 CST val AP@0.5 of seeds 1 and 2 each in 0.33-0.50"] is True
    assert r["R2 range of CST val AP@0.5 over 3 seeds <= 0.08"] is True
    assert r["R3 all 3 seeds regress on Anti-UAV by the alarm-rate criterion (diff > +0.02)"] is None      # seed 2 missing
    assert r["R4 all 3 seeds: Svanström bird/airplane/helicopter alarm rate >= 0.60"] is False
    assert p["agent"]["A2 range of CST val AP@0.5 over 3 seeds <= 0.06"] is False
    assert p["agent"]["A7 J5 regression for seeds 1 and 2"] is None


def test_seed0_threshold_follows_the_stage3_rule():
    if not D.cache_path(D.tag(0), D.HOLD).exists():
        pytest.skip("caches not present")
    from tdd.seeds.freeze import hold_point
    h, f0 = hold_point(D.tag(0)), D.frozen(0)
    assert h["f1_threshold"] == f0["threshold"] and h["ap50"] == pytest.approx(f0["hold"]["ap50"], abs=1e-12)


def test_independent_script_uses_no_project_code():
    src = (ROOT / "scripts" / "independent_check_seeds.py").read_text()
    assert "import tdd" not in src and "from tdd" not in src and "numpy" not in src
    spec = importlib.util.spec_from_file_location("indep_seeds", ROOT / "scripts" / "independent_check_seeds.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    frames = [{"gt": [], "width": 640}, {"gt": [[0, 0, 4, 4]], "width": 640}, {"gt": [], "width": 640}, {"gt": [], "invalid": True}]
    preds = [[[0, 0, 2, 2, 0.9]], [[0, 0, 4, 4, 0.99]], [[0, 0, 2, 2, 0.3]], [[0, 0, 2, 2, 0.99]]]
    assert m.far_drone_free(frames, preds, 0.5) == 0.5
    sv = [{"cls": "BIRD", "gt": [[0, 0, 4, 4]]}, {"cls": "DRONE", "gt": [[10, 10, 20, 20]]}, {"cls": "AIRPLANE", "gt": []}]
    r = m.svan(sv, [[[0, 0, 4, 4, 0.9]], [[13, 13, 17, 17, 0.9]], []], 0.5)
    assert r == {"recall_center": 1.0, "recall_iou50": 0.0, "far": 0.5}
