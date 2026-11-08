"""Render results/*.json into results/RESULTS.md (tables only; numbers copied verbatim).

Usage: PYTHONPATH=src .venv/bin/python -m tdd.report
"""
from __future__ import annotations

from .common import ROOT, read_json

METHODS = [("baseline", "Top-hat 기준선 (학습 없음)"), ("model", "Faster R-CNN R50-FPN 미세조정")]


def pct(x, nd=1):
    return "–" if x is None or x != x else f"{100 * x:.{nd}f}%"


def ci(c):
    return "" if not c or c[0] != c[0] else f" [{100 * c[0]:.1f}–{100 * c[1]:.1f}]"


def main():
    lines = ["# 결과 표 (자동 생성: `python -m tdd.report`)", "",
             "모든 수치는 아래 JSON에서 그대로 옮긴 것이다. 괄호는 95% Wilson 구간"
             "(같은 영상의 프레임끼리 상관되어 있으므로 실제 불확실성은 더 넓다).", ""]
    have = [(k, n) for k, n in METHODS if (ROOT / "results" / k / "antiuav_metrics.json").exists()]
    au = {k: read_json(ROOT / "results" / k / "antiuav_metrics.json") for k, _ in have}
    sv = {k: read_json(ROOT / "results" / k / "svanstrom_metrics.json") for k, _ in have
          if (ROOT / "results" / k / "svanstrom_metrics.json").exists()}

    lines += ["## Anti-UAV300 IR test (공식 test 91 시퀀스, 원본 영상 10개)", "",
              "| 방법 | 출처 | val AP@0.5 | 임계값(val F1 최대) | test AP@0.5 | Precision | Recall | 드론 없는 프레임 오경보율 | 오경보 FP/프레임 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, name in have:
        a, t = au[k], au[k]["test"]
        lines.append(
            f"| {name} | `results/{k}/antiuav_metrics.json` | {a['val']['ap50']:.4f} | {a['threshold']:.3f} | "
            f"{t['ap50']:.4f} | {pct(t['precision'])}{ci(t['precision_ci95'])} | {pct(t['recall'])}{ci(t['recall_ci95'])} | "
            f"{pct(t['false_alarm_rate_drone_free_frames'])}{ci(t['false_alarm_rate_ci95'])} "
            f"({t['drone_free_frames_with_alarm']}/{t['drone_free_frames']}) | {t['fp_per_drone_free_frame']:.3f} |")
    if have:
        t0 = au[have[0][0]]["test"]
        lines += ["", f"평가 프레임 {t0['frames']}장 (stride 10 샘플 + 드론 없는 프레임 전부), GT 박스 {t0['gt_boxes']}개.", ""]
        bins = list(t0["recall_by_size"].keys())
        lines += ["### 표적 크기(sqrt(w·h), px)별 recall — 주 샘플 + 16 px 미만 보충 프레임", "",
                  "| 방법 | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
        for k, name in have:
            rb = au[k]["test"]["recall_by_size"]
            lines.append(f"| {name} | " + " | ".join(
                f"{pct(rb[b]['recall'])} ({rb[b]['matched']}/{rb[b]['total']}){ci(rb[b]['ci95'])}" for b in bins) + " |")
        lines.append("")
        lines += ["| 방법 | 시퀀스별 recall 중앙값 | p10 | p90 |", "|---|---|---|---|"]
        for k, name in have:
            s = au[k]["test"]["per_sequence_recall"]
            lines.append(f"| {name} | {pct(s['median'])} | {pct(s['p10'])} | {pct(s['p90'])} |")
        lines.append("")

    if sv:
        lines += ["## Svanström 2021 IR — 교차 데이터셋 (학습·임계값 선택에 쓰지 않음, 일반화 성능)", "",
                  "임계값은 Anti-UAV val에서 고른 값을 그대로 사용. 320×256 프레임을 2배 확대해 640×512로 입력.", "",
                  "### 거리 구간별 드론 recall", "",
                  "| 방법 | 출처 | Close (<20 m) | Medium (20–60 m) | Distant (>60 m) | 전체 |", "|---|---|---|---|---|---|"]
        for k, name in have:
            if k not in sv:
                continue
            r = sv[k]["drone_recall_by_bin"]
            cells = [f"{pct(r[b]['recall'])} ({r[b]['matched']}/{r[b]['drone_boxes']}, {r[b]['videos']}개 영상){ci(r[b]['recall_ci95'])}"
                     for b in ["CLOSE", "MEDIUM", "DISTANT", "ALL"]]
            lines.append(f"| {name} | `results/{k}/svanstrom_metrics.json` | " + " | ".join(cells) + " |")
        lines += ["", "### 드론이 아닌 물체를 드론으로 본 비율", "",
                  "프레임 오경보율 = 드론 검출이 하나라도 나온 프레임 비율. 물체 오인율 = 라벨된 새/비행기/헬기 박스와 IoU≥0.5로 겹치는 드론 검출이 나온 프레임 비율.", "",
                  "| 방법 | 클래스 | 영상 | 프레임 | 프레임 오경보율 | 물체 오인율 | 오경보가 한 번이라도 난 영상 |", "|---|---|---|---|---|---|---|"]
        for k, name in have:
            if k not in sv:
                continue
            for c, r in sv[k]["non_drone_flagged_as_drone"].items():
                a = r["ALL"]
                lines.append(
                    f"| {name} | {c} | {a['videos']} | {a['frames']} | {pct(a['frame_alarm_rate'])} ({a['frames_with_drone_alarm']}){ci(a['frame_alarm_rate_ci95'])} | "
                    f"{pct(a['object_misidentified_rate'])} ({a['object_boxed_as_drone']}/{a['frames_with_object']}) | {a['videos_with_any_alarm']}/{a['videos']} |")
        lines.append("")
    out = ROOT / "results" / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
