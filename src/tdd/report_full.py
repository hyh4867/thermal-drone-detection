"""Reduced run (results/{baseline,model}) vs full run (results/full/{baseline,model}).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.report_full  -> results/full/COMPARISON.md
Numbers are copied from the JSON files named in each row.
"""
from __future__ import annotations

from .common import ROOT, read_json

RUNS = {
    "reduced": {"baseline": "results/baseline", "model": "results/model"},
    "full": {"baseline": "results/full/baseline", "model": "results/full/model"},
}


def _load(rel, name):
    p = ROOT / rel / name
    return read_json(p) if p.exists() else None


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}%"


def ci(c, scale=100):
    if not c:
        return ""
    return f" [{c[0] * scale:.1f}–{c[1] * scale:.1f}]" if scale == 100 else f" [{c[0]:.3f}–{c[1]:.3f}]"


def main():
    L = ["# 축소 실행 vs 전체 실행 (자동 생성: `python -m tdd.report_full`)", "",
         "축소 실행 = 포트폴리오에 인용된 결과(1 epoch, 축소 표본). 전체 실행 = 4 epoch, 전체 표본.",
         "대괄호: 전체 실행은 **영상 단위 bootstrap 95% 구간**(B=1000), 축소 실행은 Wilson 구간(프레임 독립 가정, 참고용).", ""]
    rows = []

    def add(label, method, getter, fmt):
        cells = []
        for run in ("reduced", "full"):
            a = _load(RUNS[run][method], "antiuav_metrics.json")
            s = _load(RUNS[run][method], "svanstrom_metrics.json")
            try:
                cells.append(fmt(getter(a, s, run)))
            except Exception:
                cells.append("–")
        rows.append(f"| {label} | {method} | {cells[0]} | {cells[1]} | `{RUNS['reduced'][method]}` → `{RUNS['full'][method]}` |")

    def bsa(a, run, key):
        return a["test"].get("bootstrap", {}).get("ci95", {}).get(key) if run == "full" else None

    for m in ("baseline", "model"):
        add("val AP@0.5 (임계값 선택용)", m, lambda a, s, r: (a["val"]["ap50"], a["val"]["frames"]),
            lambda v: f"{v[0]:.4f} ({v[1]} frames)")
        add("임계값 (val F1 최대)", m, lambda a, s, r: a["threshold"], lambda v: f"{v:.4f}")
        add("Anti-UAV test 프레임 / GT 박스 / 드론 없는 프레임", m,
            lambda a, s, r: (a["test"]["frames"], a["test"]["gt_boxes"], a["test"]["drone_free_frames"]),
            lambda v: f"{v[0]:,} / {v[1]:,} / {v[2]:,}")
        add("test AP@0.5", m, lambda a, s, r: (a["test"]["ap50"], bsa(a, r, "ap50")),
            lambda v: f"{v[0]:.4f}{ci(v[1], 1)}")
        add("Precision", m, lambda a, s, r: (a["test"]["precision"], bsa(a, r, "precision") or a["test"]["precision_ci95"]),
            lambda v: f"{pct(v[0])}{ci(v[1])}")
        add("Recall", m, lambda a, s, r: (a["test"]["recall"], bsa(a, r, "recall") or a["test"]["recall_ci95"]),
            lambda v: f"{pct(v[0])}{ci(v[1])}")
        add("드론 없는 프레임 오경보율", m,
            lambda a, s, r: (a["test"]["false_alarm_rate_drone_free_frames"], a["test"]["drone_free_frames_with_alarm"],
                             a["test"]["drone_free_frames"],
                             bsa(a, r, "false_alarm_rate_drone_free_frames") or a["test"]["false_alarm_rate_ci95"]),
            lambda v: f"{pct(v[0])} ({v[1]}/{v[2]}){ci(v[3])}")
        for b in ("tiny", "small", "medium", "large", "xlarge"):
            add(f"recall, 크기 {b}", m, lambda a, s, r, b=b: a["test"]["recall_by_size"][b],
                lambda v: f"{pct(v['recall'])} ({v['matched']}/{v['total']})")
        for b in ("CLOSE", "MEDIUM", "DISTANT", "ALL"):
            def g(a, s, r, b=b):
                d = s["drone_recall_by_bin"][b]
                c = s.get("bootstrap", {}).get("drone_recall_by_bin", {}).get(b, {}).get("ci95") if r == "full" else None
                return d, c or d["recall_ci95"]
            add(f"Svanström 드론 recall {b}", m, g,
                lambda v: f"{pct(v[0]['recall'])} ({v[0]['matched']}/{v[0]['drone_boxes']}, {v[0]['videos']} videos){ci(v[1])}")
        for c in ("BIRD", "AIRPLANE", "HELICOPTER"):
            def ga(a, s, r, c=c):
                d = s["non_drone_flagged_as_drone"][c]["ALL"]
                bb = s.get("bootstrap", {}).get("non_drone", {}).get(c, {}).get("ci95", {}) if r == "full" else {}
                return d, bb.get("frame_alarm_rate") or d["frame_alarm_rate_ci95"], bb.get("object_misidentified_rate") or d["object_misidentified_ci95"]
            add(f"{c} 프레임 오경보율", m, ga,
                lambda v: f"{pct(v[0]['frame_alarm_rate'])} ({v[0]['frames_with_drone_alarm']}/{v[0]['frames']}){ci(v[1])}")
            add(f"{c} 물체 오인율 (라벨 있는 프레임)", m, ga,
                lambda v: f"{pct(v[0]['object_misidentified_rate'])} ({v[0]['object_boxed_as_drone']}/{v[0]['frames_with_object']}){ci(v[2])}")

    L += ["| 지표 | 방법 | 축소 실행 | 전체 실행 | 출처 (축소 → 전체) |", "|---|---|---|---|---|"] + rows + [""]
    g = _load("results/full/baseline", "val_grid_search.json")
    if g:
        sel = g["selected"]
        ks = sorted({x["kernel"] for x in g["grid"]})
        kk = sorted({x["k"] for x in g["grid"]})
        edge = [n for n, v, lo, hi in (("kernel", sel["kernel"], ks[0], ks[-1]), ("k", sel["k"], kk[0], kk[-1]))
                if v in (lo, hi)]
        L += ["## 기준선 하이퍼파라미터 (전체 실행)", "",
              f"선택: kernel {sel['kernel']}, k {sel['k']}, OSD mask {sel['osd_mask']} — val AP@0.5 {sel['val_ap50']:.4f} "
              f"(`results/full/baseline/val_grid_search.json`). 그리드 kernel {ks}, k {kk}.",
              f"그리드 경계 여부: {'경계에 걸림 → ' + ', '.join(edge) if edge else '경계 아님'}.", ""]
    t = _load("results/full/model", "train_log.json")
    if t:
        L += ["## 학습 (전체 실행)", "", "| epoch | train loss | val AP@0.5 (전체 val) | 분 |", "|---|---|---|---|"]
        for h in t["history"]:
            L.append(f"| {h['epoch'] + 1} | {h['train_loss']:.4f} | {h['val_ap50_subset']:.4f} ({h['val_frames']} frames) | {h['epoch_minutes']:.0f} |")
        L += ["", f"선택 epoch: {t.get('best_epoch', t.get('best_epoch_so_far')) + 1}, 완료 여부: {t.get('complete')} (`results/full/model/train_log.json`)", ""]
    out = ROOT / "results" / "full" / "COMPARISON.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
