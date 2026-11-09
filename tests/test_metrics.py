import math
import numpy as np
from tdd.metrics import average_precision, best_f1_threshold, iou_matrix, operating_point, size_of


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]]); b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(iou_matrix(a, b), [[50 / 150, 0.0]])


def test_ap_perfect_and_ordering():
    frames = [{"gt": [[0, 0, 10, 10]], "det": [[0, 0, 10, 10, 0.9]]},
              {"gt": [[0, 0, 10, 10]], "det": [[50, 50, 60, 60, 0.95], [0, 0, 10, 10, 0.5]]}]
    r = average_precision(frames)
    # ranked: FP(0.95), TP(0.9), TP(0.5): P at R=0.5 is 1/2, at R=1 is 2/3 -> AP = 0.5*2/3 + 0.5*2/3
    assert math.isclose(r["ap"], 2 / 3, rel_tol=1e-9)


def test_duplicate_detection_is_fp():
    frames = [{"gt": [[0, 0, 10, 10]], "det": [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]]}]
    op = operating_point(frames, 0.0)
    assert op["tp"] == 1 and op["fp"] == 1


def test_operating_point_empty_frames_and_bins():
    frames = [{"gt": [], "det": [[1, 1, 3, 3, 0.7]]}, {"gt": [], "det": [[1, 1, 3, 3, 0.2]]},
              {"gt": [[0, 0, 4, 4]], "det": [[0, 0, 4, 4, 0.9]]}, {"gt": [[0, 0, 40, 40]], "det": []}]
    op = operating_point(frames, 0.5, size_bins=[("tiny", 0, 8), ("large", 32, 1e9)])
    assert op["empty_frames"] == 2 and op["empty_frames_with_alarm"] == 1
    assert op["recall_by_size"]["tiny"]["recall"] == 1.0
    assert op["recall_by_size"]["large"]["recall"] == 0.0
    assert size_of([0, 0, 4, 9]) == 6.0


def test_best_f1_threshold():
    frames = [{"gt": [[0, 0, 10, 10]], "det": [[0, 0, 10, 10, 0.9], [30, 30, 40, 40, 0.1]]}]
    t = best_f1_threshold(average_precision(frames))
    assert t["threshold"] == 0.9 and t["f1"] == 1.0


def test_bootstrap_ap_matches_metric_and_brackets_point():
    import numpy as np
    from tdd.bootstrap import _ap, _units, bootstrap_antiuav
    rng = np.random.default_rng(1)
    frames, preds = [], []
    for s in range(12):
        for i in range(10):
            gt = [[10, 10, 30, 30]] if i % 4 else []
            d = [[10 + rng.normal(0, 3), 10, 30, 30, float(rng.random())], [50, 50, 60, 60, float(rng.random() * 0.5)]]
            frames.append({"seq": f"s{s}", "gt": gt}); preds.append(d)
    units = _units(frames, preds, lambda f: f["seq"], 0.3, 0.5)
    direct = average_precision([{"gt": f["gt"], "det": p} for f, p in zip(frames, preds)])["ap"]
    assert math.isclose(_ap(units), direct, rel_tol=1e-12)
    b = bootstrap_antiuav(frames, preds, 0.3, 0.5, 200, 0)
    lo, hi = b["ci95"]["ap50"]
    assert lo <= direct <= hi and b["n_units"] == 12
