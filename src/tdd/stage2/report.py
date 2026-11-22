"""Stage 2 tables, generated from results/stage2/*.json.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.report
  -> results/stage2/RESULTS.md (all tables)
  -> README.md: the block between <!-- stage2:begin --> and <!-- stage2:end --> is replaced
     by the main tables (the surrounding text is written by hand).
Every number is read from the JSON file named under its table; nothing is typed in.
A missing file gives "실행 못 함 / 아직 없음" instead of a number.
"""
from __future__ import annotations

import pathlib

import os

from ..common import ROOT, read_json

# TDD_STAGE2_RESULTS: other directory, only for the pipeline smoke test (README is then left alone)
R = pathlib.Path(os.environ["TDD_STAGE2_RESULTS"]) if os.environ.get("TDD_STAGE2_RESULTS") else ROOT / "results" / "stage2"
BEGIN, END = "<!-- stage2:begin -->", "<!-- stage2:end -->"
DIST = (("CLOSE", "Close (<20 m)"), ("MEDIUM", "Medium (20–60 m)"), ("DISTANT", "Distant (>60 m)"))
CLS = (("BIRD", "새"), ("AIRPLANE", "비행기"), ("HELICOPTER", "헬기"))
INPUT_LABEL = {"s1_x1": "×1 (320×256 그대로)", "s1": "×2 (1단계 설정)", "s1_x3": "×3", "s1_x4": "×4", "s1_t3": "×3, 2×2 타일", "s1_t4": "×4, 2×2 타일"}
MODEL_LABEL = {"A": "A: 학습 없음", "B": "B: R1 (작은 표적 증강)", "C4": "C4: R2, 드론 영상 4개", "C12": "C12: R2, 드론 영상 12개",
               "C37": "C37: R2, 드론 영상 37개"}


def load(name):
    p = R / name
    return read_json(p) if p.exists() else None


def pct(x, d=1):
    return "–" if x is None or x != x else f"{100 * x:.{d}f}%"


def pp(x, d=1):
    return "–" if x is None or x != x else f"{100 * x:+.{d}f} %p"


def ci(c, scale=100.0, d=1):
    return "" if not c else f" [{c[0] * scale:.{d}f}, {c[1] * scale:.{d}f}]"


def num(x, d=3):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def diff(dd, key, scale=100.0, d=1):
    """Paired difference cell: +x.x %p [lo, hi]."""
    if not dd or key not in dd:
        return "–"
    v = dd[key]
    unit = " %p" if scale == 100.0 else ""
    return f"{v['diff'] * scale:+.{d}f}{unit}{ci(v['ci95'], scale, d)}"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


# ------------------------------------------------------------------ Q1

def sec_diagnosis(name="diagnosis.json", title="Svanström dev"):
    d = load(name)
    if d is None:
        return [f"(`results/stage2/{name}` 아직 없음)", ""]
    thr = f"thr_{d['threshold_stage1']:.6f}"
    low = f"thr_{d['threshold_low']}"
    a, s = d["recall_by_criterion"]["auv_test"], d["recall_by_criterion"]["svanstrom"]
    opa, ops = d["operating_point_stage1"]["auv"], d["operating_point_stage1"]["svanstrom"]
    fm = d["far_matched"]
    L = [f"출처: `results/stage2/{name}` (S1 = 1단계 모델·설정, {title} 영상 {d['videos']['svanstrom']}개 중 드론 영상 {d['videos']['svanstrom_drone']}개, "
         f"프레임 {d['frames']['svanstrom']:,}장. Anti-UAV test {d['frames']['auv_test']:,}장).", ""]
    rows = [["AP@0.5 (드론 영상만 / 전체 프레임)", num(d["ap50"]["auv_test"]),
             f"{num(d['ap50']['svanstrom_drone_videos'])} / {num(d['ap50']['svanstrom_all_frames'])}"]]
    for key, lab in (("iou50", "IoU ≥ 0.5 (등록한 주 기준)"), ("iou30", "IoU ≥ 0.3"), ("iou10", "IoU ≥ 0.1"), ("center", "검출 중심이 라벨 박스 안")):
        rows.append([f"recall, 임계값 0.954, {lab}", pct(a[thr][key]), pct(s[thr][key])])
    for key, lab in (("iou50", "IoU ≥ 0.5"), ("center", "중심 기준")):
        rows.append([f"recall, 임계값 0.05, {lab}", pct(a[low][key]), pct(s[low][key])])
    rows.append(["오경보율, 임계값 0.954 (Anti-UAV: 드론 없는 프레임, Svanström: 새·비행기·헬기 영상 프레임)",
                 pct(opa["far_drone_free"]), pct(ops["far_non_drone_frames (T_base)"])])
    rows.append([f"오경보율을 Anti-UAV 수준({pct(fm['auv_far_at_stage1_thr'])})에 맞춘 임계값에서의 recall (IoU ≥ 0.5)",
                 pct(opa["recall"]), f"{pct(fm['svanstrom_recall_there'])} (임계값 {fm['svanstrom_thr_for_same_far']:.4f})"])
    rows.append(["오경보율 10%에 맞춘 임계값에서의 recall (IoU ≥ 0.5)", "–",
                 f"{pct(fm['svanstrom_recall_at_far10'])} (임계값 {fm['svanstrom_thr_for_far10']:.4f})"])
    q = d["tp_score_quantiles"]
    rows.append(["IoU ≥ 0.5로 맞은 검출의 점수: 10% / 25% / 50% 분위", " / ".join(num(q["auv_test"][k]) for k in ("0.1", "0.25", "0.5")),
                 " / ".join(num(q["svanstrom"][k]) for k in ("0.1", "0.25", "0.5"))])
    bc = d["box_convention"]
    rows.append(["느슨하게(IoU ≥ 0.1) 맞은 쌍에서 예측 박스 / 라벨 박스 크기 (가로, 세로 중앙값)",
                 f"{num(bc['auv_test']['loose_rw']['median'], 2)}, {num(bc['auv_test']['loose_rh']['median'], 2)}",
                 f"{num(bc['svanstrom']['loose_rw']['median'], 2)}, {num(bc['svanstrom']['loose_rh']['median'], 2)}"])
    L += table(["지표", "Anti-UAV test", title], rows)
    return L


def sec_decomposition(name="diagnosis.json"):
    d = load(name)
    if d is None:
        return [f"(`results/stage2/{name}` 아직 없음)", ""]
    p, c = d["decomposition"]["point"], d["decomposition"]["ci95"]
    rows = [["Anti-UAV test recall (임계값 0.954, IoU ≥ 0.5)", pct(p["recall_auv"]) + ci(c["recall_auv"]), ""],
            ["Svanström recall (같은 기준)", pct(p["recall_svan"]) + ci(c["recall_svan"]), ""],
            ["**차이 G**", f"**{100 * p['gap']:.1f} %p**" + ci(c["gap"]), ""],
            ["위치 기준 몫 L (IoU 0.5 → 0.1)", pp(p["share_localisation"]).lstrip("+") + ci(c["share_localisation"]),
             "박스가 라벨과 덜 겹쳐서 틀린 것으로 센 몫"],
            ["임계값 몫 T (0.954 → 0.05)", pp(p["share_threshold"]).lstrip("+") + ci(c["share_threshold"]), "찾았지만 점수가 낮았던 몫"],
            ["크기 몫 S (Anti-UAV recall을 Svanström 크기 분포로 표준화)", pp(p["share_size"]).lstrip("+") + ci(c["share_size"]), "표적이 더 작아서 생긴 몫"],
            ["겹치는 몫", pp(p["overlap"]) + ci(c["overlap"]), "T + L + S의 합에서 셋을 함께 적용했을 때 줄어든 양을 뺀 값"],
            ["설명되지 않는 몫 (셋을 모두 적용한 뒤 남는 차이)", pp(p["unexplained"]) + ci(c["unexplained"]),
             "음수 = 세 기준을 풀면 Svanström 쪽이 더 높다"]]
    L = table(["항목", "값 [95% 구간]", "뜻"], rows)
    fm = d["far_matched"]
    L += [f"- 대괄호는 영상 단위 bootstrap 95% 구간(B=1000, Anti-UAV는 시퀀스, Svanström은 영상을 재표본).",
          f"- 임계값의 다른 정의(두 데이터셋의 오경보율을 {pct(fm['auv_far_at_stage1_thr'])}로 맞춤)로는 임계값 몫이 {pp(fm['threshold_share_alt'])}다. "
          "오경보율을 같게 맞추면 임계값을 더 올려야 해서 recall이 더 떨어진다는 뜻이다.",
          f"- 출처: `results/stage2/{name}`의 `decomposition`, `far_matched`.", ""]
    return L


def sec_size(name="diagnosis.json"):
    d = load(name)
    if d is None:
        return []
    a, s = d["recall_by_size"]["auv_test_plus_supplement"], d["recall_by_size"]["svanstrom"]
    ca, cs = d["recall_by_size_ci95_iou50"]["auv_test_plus_supplement"], d["recall_by_size_ci95_iou50"]["svanstrom"]
    corr = d["recall_by_size_svanstrom_corrected_size"]
    rows = []
    for b in d["size_bins_reference_px"]:
        def cell(t, cc, key="recall_iou50"):
            return "–" if not t["n"] else f"{pct(t[key])} ({t['n']:,}){ci(cc) if cc else ''}"
        rows.append([b, cell(a[b], ca[b]), cell(s[b], cs[b]), "–" if not s[b]["n"] else pct(s[b]["recall_center"]),
                     cell(corr["table"][b], corr["ci95_iou50"][b]), "–" if not corr["table"][b]["n"] else pct(corr["table"][b]["recall_center"])])
    L = table(["sqrt(w·h), 640×512 기준 px", "Anti-UAV test + 보충: IoU 0.5 recall (박스 수)", "Svanström: IoU 0.5 recall (박스 수)", "Svanström: 중심 기준",
               f"Svanström, 라벨 크기 ÷ {corr['k']}: IoU 0.5", "같음: 중심 기준"], rows)
    L += [f"- 임계값 0.954. Svanström 라벨은 표적보다 크게 그려져 있어(박스 보정 k = {corr['k']}), 오른쪽 두 열은 라벨 크기를 k로 나눠 다시 묶은 것이다.",
          f"- 출처: `results/stage2/{name}`의 `recall_by_size`, `recall_by_size_svanstrom_corrected_size`.", ""]
    return L


def sec_distance(name="diagnosis.json"):
    d = load(name)
    if d is None:
        return []
    t = d["svanstrom_recall_by_distance"]
    rows = [[lab, f"{t[k]['gt']:,}", num(t[k]["median_label_size_ref_px"], 1), pct(t[k]["iou50"]), pct(t[k]["center"]), pct(t[k]["iou50_low_thr"]), pct(t[k]["center_low_thr"])]
            for k, lab in (*DIST, ("ALL", "전체"))]
    L = table(["거리 구간", "GT 박스", "라벨 크기 중앙값 (px)", "IoU 0.5, 임계값 0.954", "중심 기준, 0.954", "IoU 0.5, 임계값 0.05", "중심 기준, 0.05"], rows)
    L += [f"- 출처: `results/stage2/{name}`의 `svanstrom_recall_by_distance`.", ""]
    return L


def sec_false_alarms(name="diagnosis.json"):
    d = load(name)
    if d is None:
        return []
    f, o = d["false_alarms_non_drone"], d["object_flagged_as_drone"]
    rel = f["relation_to_labelled_object"]
    n = f["boxes_at_stage1_thr"]
    rows = [["라벨된 물체 위, IoU ≥ 0.5", f"{rel.get('on_object_iou>=0.5', 0):,}", pct(rel.get("on_object_iou>=0.5", 0) / n)],
            ["라벨된 물체 위(박스 1.5배 영역 안), IoU < 0.5", f"{rel.get('on_object_iou<0.5', 0):,}", pct(rel.get("on_object_iou<0.5", 0) / n)],
            ["물체 라벨이 있는 프레임의 다른 곳", f"{rel.get('elsewhere_object_labelled', 0):,}", pct(rel.get("elsewhere_object_labelled", 0) / n)],
            ["물체 라벨이 없는 프레임", f"{rel.get('frame_without_object_label', 0):,}", pct(rel.get("frame_without_object_label", 0) / n)],
            ["**합계**", f"**{n:,}**", ""]]
    L = table(["오경보 박스의 위치 (새·비행기·헬기 영상, 임계값 0.954)", "박스 수", "비율"], rows)
    rows = [[lab, f"{o[k]['frames_with_object_label']:,}", pct(o[k]["iou50"]), pct(o[k]["center"])] for k, lab in CLS]
    L += table(["클래스", "물체 라벨이 있는 프레임", "물체 오인율, IoU ≥ 0.5", "물체 오인율, 중심 기준"], rows)
    L += [f"- 오경보 박스 크기 중앙값 {num(f['size_median_reference_px'], 1)} px, 점수 중앙값 {num(f['score_median'])}, 화면 가장자리에 붙은 박스 {pct(f['near_image_edge_share'])}. "
          f"오경보가 한 번이라도 난 영상 {f['videos_with_alarm']}/{f['non_drone_videos']}개, 상위 10개 영상이 박스의 {pct(f['top10_videos_share_of_boxes'])}.",
          f"- 출처: `results/stage2/{name}`의 `false_alarms_non_drone`, `object_flagged_as_drone`.", ""]
    return L


def sec_inputs():
    d = load("inputs.json")
    if d is None:
        return ["(`results/stage2/inputs.json` 아직 없음)", ""]
    rows = []
    for tag, r in d["configs"].items():
        if r["status"] != "ok":
            rows.append([INPUT_LABEL.get(tag, r["label"]), r["status"], "", "", "", "", "", ""])
            continue
        a, b, c = r["at_stage1_threshold"], r["iso_far_T_base"], r["iso_far_10pct"]
        rows.append([INPUT_LABEL.get(tag, r["label"]) + (" **← 선택**" if tag == d["selection"]["selected"] else ""), num(r["k_selected"], 2),
                     "–" if r["s_per_frame"] is None else num(r["s_per_frame"], 2), pct(a["far"]),
                     f"{pct(b['recall_iou50'])} / {pct(b['recall_center'])}", pct(b["recall_by_distance"]["DISTANT"]["center"]),
                     f"{pct(c['recall_iou50'])} / {pct(c['recall_center'])}", diff(r.get("paired_vs_x2_iso_far"), "recall_iou50")])
    L = table(["입력", "박스 보정 k", "초/장", "임계값 0.954의 오경보율", f"오경보율 {pct(d['T_base'])}에 맞춤: IoU 0.5 / 중심", "같음: Distant 중심 기준",
               "오경보율 10%에 맞춤: IoU 0.5 / 중심", "×2 대비 차이 (IoU 0.5, 짝 비교)"], rows)
    L += [f"- dev 영상의 10프레임마다 1장({d['frames']:,}장). 선택 규칙(사전 등록): 오경보율을 1단계 수준에 맞춘 IoU 0.5 recall이 가장 높은 설정, 1 %p 안이면 계산이 싼 쪽.",
          "- 출처: `results/stage2/inputs.json`.", ""]
    return L


def sec_dev_compare(name, title, labels=None):
    d = load(f"dev_compare__{name}.json")
    if d is None:
        return [f"(`results/stage2/dev_compare__{name}.json` 아직 없음)", ""]
    rows = []
    for tag, r in d["configs"].items():
        b, c = r["T_base"], r["T10"]
        rows.append([(labels or {}).get(tag, f"`{tag}`"), num(r["k"], 2), pct(b["recall_iou50"]) + ci(b["ci95"]["recall_iou50"]), pct(b["recall_center"]),
                     f"{pct(b['recall_by_distance']['DISTANT']['iou50'])} / {pct(b['recall_by_distance']['DISTANT']['center'])}",
                     f"{pct(c['recall_iou50'])} / {pct(c['recall_center'])}", diff(b.get("paired_diff"), "recall_iou50")])
    L = table([title, "박스 보정 k", f"오경보율 {pct(d['T_base'])}에 맞춤: IoU 0.5 recall", "중심 기준", "Distant: IoU 0.5 / 중심",
               "오경보율 10%: IoU 0.5 / 중심", "첫 행 대비 차이 (짝 비교)"], rows)
    L += [f"- {d['set']} ({d['videos']}개 영상, {d['frames']:,}장). 출처: `results/stage2/dev_compare__{name}.json`.", ""]
    return L


def sec_motion():
    d = load("motion.json")
    if d is None:
        return []
    L = ["등속(constant-velocity) 가정이 카메라가 움직일 때 얼마나 깨지는지 먼저 쟀다(GT 중심 궤적, 추적 주기에서 한 스텝 예측 잔차).", ""]
    sv, au = d["svanstrom_dev"], d["antiuav_val"]
    rows = [["Svanström dev 드론 (6 Hz)", f"{sv['drone']['static']['n']:,}", num(sv["drone"]["static"]["median"], 2), num(sv["drone"]["cv"]["median"], 2),
             num(sv["drone"]["cv"]["p75"], 2), num(sv["drone"]["cv"]["p90"], 2), num(sv["drone"]["cv_compensated"]["median"], 2)],
            ["Svanström dev 새·비행기·헬기 (6 Hz)", f"{sv['non_drone_object']['static']['n']:,}", num(sv["non_drone_object"]["static"]["median"], 2),
             num(sv["non_drone_object"]["cv"]["median"], 2), num(sv["non_drone_object"]["cv"]["p75"], 2), num(sv["non_drone_object"]["cv"]["p90"], 2),
             num(sv["non_drone_object"]["cv_compensated"]["median"], 2)],
            [f"Anti-UAV val ({1 / au['dt_s']:.1f} Hz)", f"{au['static']['n']:,}", num(au["static"]["median"], 2), num(au["cv"]["median"], 2),
             num(au["cv"]["p75"], 2), num(au["cv"]["p90"], 2), "재지 않음"]]
    L += table(["자료", "스텝 수", "정지 가정 잔차 중앙값 (px)", "등속 잔차 중앙값", "등속 p75", "등속 p90", "전역 이동 보정 뒤 등속 중앙값"], rows)
    g = sv["global_shift_px_per_step"]
    L += [f"- 640×512 기준 px. Svanström dev의 프레임 간 전역 이동(phase correlation)은 중앙값 {num(g['median'], 2)} px, p90 {num(g['p90'], 2)} px, p99 {num(g['p99'], 1)} px다."]
    dec = d["decision"]
    q = dec["q_levels_px2_per_s3"]
    L += [f"- 결정: 모델 `{dec['model']}`(등속 잔차 중앙값이 정지 가정보다 작음). 전역 이동 보정은 등속 잔차를 {pct(dec['cv_residual_reduction_by_compensation'])}만 줄여 "
          f"{'채택' if dec.get('global_shift_compensation') else '채택 기준(20%)에 못 미쳐 넣지 않았다'}. 과정 잡음 후보 q = {' / '.join(f'{v:.0f}' for v in q.values())} px²/s³.",
          "- 출처: `results/stage2/motion.json`.", ""]
    return L


def sec_tracker_dev(fz):
    if fz is None:
        return ["(`results/stage2/frozen_config.json` 아직 없음)", ""]
    rows = []
    for name, m in fz["models"].items():
        p = m["tracker"]["params"]
        t = read_json(ROOT / m["tracker"]["tuning_file"])
        for point, lab in (("T_base", "1단계 수준"), ("T10", "10%")):
            dd, tt = t["detector_only"][point], m["tracker"]["dev"][point]
            rows.append([MODEL_LABEL.get(name, name) if point == "T_base" else "", lab,
                         f"{pct(dd['recall_iou50'])} / {pct(dd['recall_center'])}", f"{pct(tt['recall_iou50'])} / {pct(tt['recall_center'])}",
                         num(tt["false_tracks_per_min"], 1), pct(tt["never_found_share_center"]), num(tt["latency_median_center"], 2), pct(tt["maintenance_center"]),
                         f"검출 임계값 {p['det_threshold']}, {p['confirm_m']}/{p['confirm_n']} 확정, {p['delete_misses']}회 미검출 삭제, q {p['q']:.0f}, 예측 출력 {p['coast_outputs']}회" if point == "T_base" else ""])
    L = table(["구성", "맞춘 오경보율", "검출만: IoU 0.5 / 중심", "추적: IoU 0.5 / 중심", "분당 가짜 확정 항적", "끝까지 못 찾은 드론 영상", "첫 정탐까지 (초, 중앙값)", "확정 뒤 유지율",
               "dev에서 고른 추적기 값"], rows)
    L += ["- dev에서 고른 값이다(test 아님). 출처: `results/stage2/frozen_config.json`, `results/stage2/tracker_tuning__*.json`.", ""]
    return L


# ------------------------------------------------------------------ Q3

def sec_training():
    rows = []
    for run, lab in (("r1", "R1: 작은 표적 증강, Anti-UAV train만"), ("r2_n4", "R2: + Svanström dev_a 드론 영상 4개"),
                     ("r2_n12", "R2: + 드론 영상 12개"), ("r2_n37", "R2: + 드론 영상 37개")):
        d = load(f"train_{run}.json")
        if d is None:
            rows.append([lab, "실행 못 함 / 아직 없음", "", "", "", "", ""])
            continue
        ad = d.get("adaptation_data")
        data = "Anti-UAV train {:,}장".format(d["train_frames_antiuav"]) + (
            f" + Svanström 드론 영상 {len(ad['drone_videos'])}개·비드론 {len(ad['non_drone_videos'])}개({ad['frames']:,}장, 드론 박스 {ad['drone_frames_with_box']:,}개), batch 8장 중 {ad['per_batch']}장" if ad else "")
        rows.append([lab, d["status"], f"{d['iterations']:,} / {d['planned_iterations']:,}", f"{d['minutes_used']:.0f}분 (한도 {d['time_limit_min']:.0f}분)",
                     " → ".join(num(h["val_ap50"]) for h in d["history"]), data, d["args"]["init"]])
    L = table(["실행", "상태", "iteration", "벽시계 시간", "Anti-UAV val AP@0.5 (epoch별)", "학습 자료", "시작 가중치"], rows)
    L += ["- 공통: `configs/full.yaml`의 저메모리 설정(worker 0, 2장 × 4 gradient accumulation, 평가 batch 1), lr 0.005 cosine, warmup 100, seed 0. "
          "1단계 모델의 val AP@0.5는 0.937이다. 출처: `results/stage2/train_*.json`.", ""]
    return L


# ------------------------------------------------------------------ final

def _svan_row(label, det, ref_diff=True, seq=None, extra="", pd=None):
    pd = pd or (det.get("paired_diff") if ref_diff else None)
    seq = seq or det.get("sequence") or {}
    c = seq.get("center", {})
    return [label, pct(det["recall_iou50"]) + ci(det["ci95"]["recall_iou50"]), diff(pd, "recall_iou50"), pct(det["recall_center"]) + ci(det["ci95"]["recall_center"]),
            diff(pd, "recall_center"), pct(det["far"]) + ci(det["ci95"]["far"]), diff(pd, "far"),
            "–" if "false_confirmed_tracks_per_minute" not in seq else num(seq["false_confirmed_tracks_per_minute"], 1),
            pct(c.get("never_found_share")), num(c.get("first_found_latency_s_median"), 2), pct(c.get("maintenance_after_first")), extra]


SVAN_HEAD = ["구성", "IoU 0.5 recall", "S1 대비 (짝 비교)", "중심 기준 recall", "S1 대비", "프레임 오경보율", "S1 대비", "분당 가짜 확정 항적",
             "끝까지 못 찾은 드론 영상", "첫 정탐까지 (초)", "유지율", "비고"]


def sec_final_svanstrom(point="T_base"):
    d = load("final__svanstrom.json")
    if d is None:
        return ["(`results/stage2/final__svanstrom.json` 아직 없음)", ""]
    ref = d["reference_S1"]
    cal = d["S1_box_calibrated"]
    if point == "T_base":
        rows = [_svan_row("S1: 1단계 모델·설정 (×2, 임계값 0.954)", ref["detector"], False, ref["sequence"], "기준")]
        rows.append(_svan_row(f"S1 + 박스 보정만 (k = {cal['k']})", cal["detector"], True, cal["sequence"], "라벨 관례 맞춤. 탐지 능력은 그대로"))
    else:       # 10% table: the reference is the box-calibrated stage 1 detections at their own dev-matched 10% threshold
        c10 = d["S1_box_calibrated_T10"]
        rows = [_svan_row(f"S1 + 박스 보정 (×2, k = {c10['k']}, 임계값 {c10['threshold']:.3f})", c10["detector"], False, c10["sequence"], "기준")]
    for name, m in d["models"].items():
        lab = MODEL_LABEL.get(name, name)
        if m["status"] != "ok":
            rows.append([lab, "실행 못 함", "", "", "", "", "", "", "", "", "", m.get("detail", "")[:80]])
            continue
        if point == "T_base":
            rows.append(_svan_row(f"{lab}, 검출만", m["detector"][point], True, None, f"k = {m['k']}"))
            rows.append(_svan_row(f"{lab}, + 추적", m["tracker"][point], True, None, ""))
        else:
            vs = m["paired_vs_box_calibrated_S1_T10"]
            rows.append(_svan_row(f"{lab}, 검출만", m["detector"][point], False, None, f"k = {m['k']}", pd=vs["detector"]))
            rows.append(_svan_row(f"{lab}, + 추적", m["tracker"][point], False, None, "", pd=vs["tracker"]))
    head = SVAN_HEAD if point == "T_base" else [h.replace("S1 대비", "기준 대비") for h in SVAN_HEAD]
    L = table(head, rows)
    if point != "T_base":
        L += ["- 이 표의 기준은 1단계 검출(×2)에 박스 보정을 하고 dev에서 오경보율 10%에 맞춘 임계값을 쓴 것이다. 1단계 설정 그대로(임계값 0.954)는 오경보율이 달라 여기서는 기준으로 쓰지 않았다.", ""]
    return L


def sec_final_increments():
    """What tracking adds to the same detector, and what training adds to configuration A."""
    d = load("final__svanstrom.json")
    if d is None:
        return []
    rows = []
    for name, m in d["models"].items():
        if m["status"] != "ok":
            continue
        for point, lab in (("T_base", "1단계 수준"), ("T10", "10%")):
            tr = m["tracker"][point]["paired_vs_same_model_detector"]
            va = (m.get("paired_vs_A") or {}).get(point)
            rows.append([MODEL_LABEL.get(name, name) if point == "T_base" else "", lab, diff(tr, "recall_iou50"), diff(tr, "recall_center"), diff(tr, "far"),
                         diff(va["detector"], "recall_iou50") if va else "–", diff(va["detector"], "recall_center") if va else "–",
                         diff(va["detector"], "far") if va else "–", diff(va["tracker"], "recall_iou50") if va else "–"])
    L = table(["구성", "맞춘 오경보율", "추적 − 같은 모델의 검출: IoU 0.5 recall", "중심 기준", "오경보율", "검출: 이 모델 − A, IoU 0.5 recall", "중심 기준", "오경보율",
               "추적 포함: 이 모델 − A, IoU 0.5 recall"], rows)
    L += ["- 모두 같은 test 영상에서의 짝 차이와 영상 단위 bootstrap 95% 구간이다. 임계값은 구성마다 dev에서 맞춘 값이어서 test 오경보율이 조금씩 다르다(오경보율 열).",
          "- 출처: `results/stage2/final__svanstrom.json` (`paired_vs_same_model_detector`, `paired_vs_A`).", ""]
    return L


def sec_final_svanstrom_notes():
    d = load("final__svanstrom.json")
    fz = load("frozen_config.json")
    if d is None or fz is None:
        return []
    n_d = d["reference_S1"]["detector"]
    L = [f"- Svanström test: 영상 {d['videos']}개, {d['frames']:,}장(5프레임마다 1장, 6 Hz), 드론 GT 박스 {n_d['drone_gt']:,}개, 새·비행기·헬기 영상 프레임 {n_d['non_drone_frames']:,}장.",
         f"- 임계값·박스 보정 k·입력 배율(×{fz['input']['scale']})·추적기 값은 모두 dev에서 고르고 동결했다(`results/stage2/frozen_config.json`). "
         "임계값은 dev에서 S1과 같은 오경보율이 되도록 맞춘 값이라, test의 오경보율은 S1과 정확히 같지 않다. 그래서 오경보율과 그 차이를 함께 싣는다.",
         "- 대괄호는 영상 단위 bootstrap 95% 구간(B=1000, seed 0). \"S1 대비\"는 같은 재표본 영상에서의 짝 차이다.",
         "- 가짜 확정 항적은 새·비행기·헬기 영상에서 출력된 확정 항적 ID 수다. 첫 정탐·유지율은 중심 기준이다.",
         "- 출처: `results/stage2/final__svanstrom.json`.", ""]
    return L


def sec_final_svanstrom_detail():
    d = load("final__svanstrom.json")
    if d is None:
        return []
    cols = [("S1", d["reference_S1"]["detector"]), ("S1 + 박스 보정", d["S1_box_calibrated"]["detector"])]
    for name, m in d["models"].items():
        if m["status"] == "ok":
            cols.append((f"{name} 검출", m["detector"]["T_base"]))
            cols.append((f"{name} + 추적", m["tracker"]["T_base"]))
    rows = []
    for k, lab in DIST:
        rows.append([f"{lab} IoU 0.5 / 중심 (GT {cols[0][1]['recall_by_distance'][k]['gt']:,})"] +
                    [f"{pct(c['recall_by_distance'][k]['iou50'])} / {pct(c['recall_by_distance'][k]['center'])}" for _, c in cols])
    for k, lab in CLS:
        rows.append([f"{lab} 영상 프레임 오경보율"] + [pct(c["far_by_class"][k]) for _, c in cols])
    for k, lab in CLS:
        rows.append([f"{lab} 물체 오인율, 중심 기준"] + [pct(c["object_flagged"][k]["center"]) for _, c in cols])
    L = table(["Svanström test"] + [n for n, _ in cols], rows)
    L += ["- 출처: `results/stage2/final__svanstrom.json` (`recall_by_distance`, `far_by_class`, `object_flagged`).", ""]
    return L


def sec_gap_closure():
    d = load("final__svanstrom.json")
    dg = load("diagnosis.json")
    if d is None or dg is None:
        return []
    r_au = dg["operating_point_stage1"]["auv"]["recall"]
    s1 = d["reference_S1"]["detector"]["recall_iou50"]
    gap = r_au - s1
    rows = [["S1 + 박스 보정만", diff(d["S1_box_calibrated"]["detector"]["paired_diff"], "recall_iou50"),
             pct(d["S1_box_calibrated"]["detector"]["paired_diff"]["recall_iou50"]["diff"] / gap, 0), "–", "–"]]
    for name, m in d["models"].items():
        if m["status"] != "ok":
            continue
        for kind, lab in (("detector", "검출만"), ("tracker", "+ 추적")):
            pd = m[kind]["T_base"]["paired_diff"]
            vs = m["paired_vs_box_calibrated_S1"][kind]
            lo = pd["recall_iou50"]["ci95"][0] if pd["recall_iou50"]["ci95"] else float("nan")
            hi = pd["recall_iou50"]["ci95"][1] if pd["recall_iou50"]["ci95"] else float("nan")
            verdict = "되찾았다" if lo > 0 else "나빠졌다" if hi < 0 else "차이를 확인하지 못했다"
            rows.append([f"{MODEL_LABEL.get(name, name)}, {lab}", diff(pd, "recall_iou50"), pct(pd["recall_iou50"]["diff"] / gap, 0),
                         diff(vs, "recall_iou50"), verdict])
    L = [f"차이 = Anti-UAV test recall {pct(r_au)} − 같은 test 영상에서의 S1 recall {pct(s1)} = {100 * gap:.1f} %p.", ""]
    L += table(["구성", "S1 대비 IoU 0.5 recall 차이 [95% 구간]", "차이의 몇 %", "박스 보정한 S1 대비 차이 [95% 구간]", "등록한 판정 (S1 대비 구간 기준)"], rows)
    L += ["- 등록한 판정 규칙: 짝 차이의 95% 구간이 0보다 위면 \"되찾았다\", 0을 포함하면 \"차이를 확인하지 못했다\", 0보다 아래면 \"나빠졌다\".",
          "- 박스 보정은 라벨 관례(Svanström 라벨이 표적보다 크게 그려짐)를 맞춘 것이므로, 탐지 능력의 변화는 \"박스 보정한 S1 대비\" 열과 중심 기준 recall로 읽어야 한다.",
          "- 출처: `results/stage2/final__svanstrom.json`, `results/stage2/diagnosis.json`.", ""]
    return L


# Configurations the pre-registration (addendum 2) planned for each native-resolution set; others were never queued.
NATIVE_PLANNED = {"cst": ("A", "B", "C37"), "antiuav_track": ("A", "B", "C37")}


def sec_final_native(part, title, with_tracker):
    d = load(f"final__{part}.json")
    if d is None:
        return [f"(`results/stage2/final__{part}.json` 아직 없음)", ""]
    head = ["구성", "임계값", "AP@0.5", "recall (IoU 0.5)", "recall (중심)", "precision", "드론 없는 프레임 오경보율", "S1 대비: AP / recall / 오경보율"]
    rows = []
    ref = d.get("reference_S1", {})
    if "detector" in ref:
        r = ref["detector"]
        rows.append(["S1: 1단계 모델", num(r["threshold"]), num(r["ap50"]) + ci(r["ci95"]["ap50"], 1, 3), pct(r["recall_iou50"]) + ci(r["ci95"]["recall_iou50"]),
                     pct(r["recall_center"]), pct(r["precision"]), pct(r["far_drone_free"]) + ci(r["ci95"]["far_drone_free"]), "기준"])
    else:
        rows.append(["S1: 1단계 모델", "실행 못 함", "", "", "", "", "", ref.get("detail", "")[:80]])
    for name, m in d["models"].items():
        lab = MODEL_LABEL.get(name, name)
        if m["status"] != "ok":
            if name not in NATIVE_PLANNED.get(part, (name,)):
                rows.append([lab, "계획에 없음", "", "", "", "", "", "이 자료에서는 S1, A, B, C37만 평가하기로 했다(사전 등록 추가 2)"])
            else:
                rows.append([lab, "실행 못 함", "", "", "", "", "", m.get("detail", "")[:80]])
            continue
        r = m["detector"]
        pd = r.get("paired_diff")
        rows.append([lab + (", 검출만" if with_tracker else ""), num(r["threshold"]), num(r["ap50"]) + ci(r["ci95"]["ap50"], 1, 3),
                     pct(r["recall_iou50"]) + ci(r["ci95"]["recall_iou50"]), pct(r["recall_center"]), pct(r["precision"]),
                     pct(r["far_drone_free"]) + ci(r["ci95"]["far_drone_free"]),
                     "–" if not pd else f"{diff(pd, 'ap50', 1.0, 3)} / {diff(pd, 'recall_iou50')} / {diff(pd, 'far_drone_free')}"])
        if with_tracker and "tracker" in m:
            t = m["tracker"]
            pdt = t.get("paired_diff")
            rows.append([f"{lab}, + 추적", num(t["threshold"]), "–", pct(t["recall_iou50"]) + ci(t["ci95"]["recall_iou50"]),
                         pct(t["recall_center"]), pct(t["precision"]), pct(t["far_drone_free"]) + ci(t["ci95"]["far_drone_free"]),
                         "같은 모델 검출 대비: – / " + (f"{diff(pdt, 'recall_iou50')} / {diff(pdt, 'far_drone_free')}" if pdt else "–")])
    L = table(head, rows)
    r0 = (ref.get("detector") or next((m["detector"] for m in d["models"].values() if m["status"] == "ok"), None))
    if r0:
        L += [f"- {title}: {r0['sequences']}개 시퀀스, {r0['frames']:,}장, GT 박스 {r0['gt_boxes']:,}개, 드론 없는 프레임 {r0['drone_free_frames']:,}장. "
              "임계값은 각 구성의 Anti-UAV val F1 최대값, 박스 보정 없음(k = 1), 입력 ×1. S1은 1단계 설정 그대로이고, 나머지 구성은 동결한 추론 설정(TTA 포함 여부는 `frozen_config.json`)을 쓴다. 대괄호는 시퀀스 단위 bootstrap 95% 구간."]
    if with_tracker:
        L += ["- 추적기 값과 항적 점수 임계값은 Svanström dev에서 동결한 것을 그대로 썼다. 추적 행의 임계값은 항적 점수 임계값이고, AP는 계산하지 않았다."]
    L += [f"- 출처: `results/stage2/final__{part}.json`.", ""]
    return L


def sec_native_sequence(part):
    d = load(f"final__{part}.json")
    if d is None:
        return []
    rows = []

    def row(lab, s):
        c = s.get("center", {})
        return [lab, pct(c.get("never_found_share")), num(c.get("first_found_latency_s_median"), 2), num(c.get("first_found_latency_s_p90"), 2),
                pct(c.get("maintenance_after_first")), num(s.get("track_ids_per_found_drone_video_mean"), 2),
                num(s.get("tracks_never_on_target_in_drone_videos_per_minute"), 2)]
    if "sequence" in d.get("reference_S1", {}):
        rows.append(row("S1 검출만", d["reference_S1"]["sequence"]))
    for name, m in d["models"].items():
        if m["status"] != "ok":
            continue
        lab = MODEL_LABEL.get(name, name)
        rows.append(row(f"{lab}, 검출만", m["detector"]["sequence"]))
        if "tracker" in m:
            rows.append(row(f"{lab}, + 추적", m["tracker"]["sequence"]))
    L = table(["구성", "끝까지 못 찾은 시퀀스", "첫 정탐까지 중앙값 (초)", "p90 (초)", "유지율", "표적에 붙은 항적 ID 수 (평균)", "표적에 한 번도 안 붙은 항적 (분당)"], rows)
    L += [f"- 중심 기준. 출처: `results/stage2/final__{part}.json`의 `sequence`.", ""]
    return L


def sec_size_final():
    d = load("final__antiuav.json")
    if d is None or not d.get("recall_by_size_with_supplement"):
        return []
    s = d["recall_by_size_with_supplement"]
    names = list(s)
    bins = list(s[names[0]])
    rows = [[b] + [f"{pct(s[n][b]['iou50'])} ({s[n][b]['total']:,})" for n in names] for b in bins]
    L = table(["크기 구간 (Anti-UAV test + 작은 표적 보충)"] + names, rows)
    L += ["- IoU 0.5 recall(박스 수), 각 모델의 Anti-UAV val 임계값. 출처: `results/stage2/final__antiuav.json`의 `recall_by_size_with_supplement`.", ""]
    return L


def build():
    fz = load("frozen_config.json")
    full = ["# 2단계 결과 표 (자동 생성: `python -m tdd.stage2.report`)", "",
            "교차 데이터셋 성능 저하의 원인 분해와 회복. 모든 수치는 `results/stage2/*.json`에서 읽은 값이다. 사전 등록: `docs/PREREGISTRATION_2.md`.", "",
            "## 1. 진단 (1단계 모델, 새 학습 없음, Svanström dev)", "", "### 1.1 임계값·판정 기준별 수치", ""]
    full += sec_diagnosis()
    full += ["### 1.2 요인 분해", ""] + sec_decomposition()
    full += ["### 1.3 크기 구간", ""] + sec_size()
    full += ["### 1.4 거리 구간 (Svanström dev)", ""] + sec_distance()
    full += ["### 1.5 입력 배율과 타일", ""] + sec_inputs()
    full += ["### 1.6 오경보가 나오는 곳", ""] + sec_false_alarms()
    full += ["## 2. 학습 없는 회복: dev에서 고른 것", "", "### 2.1 TTA (좌우 반전)", ""] + sec_dev_compare("tta", "입력", {"s1_x1": "×1", "s1_x1tta": "×1 + 좌우 반전 TTA"})
    full += ["### 2.2 등속 가정", ""] + sec_motion()
    full += ["### 2.3 추적기 (dev)", ""] + sec_tracker_dev(fz)
    full += ["## 3. 학습 실행", ""] + sec_training()
    full += ["### 3.1 dev_b에서의 모델 비교 (체크포인트·시작 가중치 선택용)", ""] + sec_dev_compare("models_dev_b", "모델")
    full += ["## 4. 최종 평가 (구성 동결 뒤, 한 번씩)", "", "### 4.1 Svanström test, 오경보율을 1단계 수준에 맞춘 임계값", ""]
    full += sec_final_svanstrom("T_base") + ["### 4.2 Svanström test, 오경보율 10%에 맞춘 임계값", ""] + sec_final_svanstrom("T10") + sec_final_svanstrom_notes()
    full += ["### 4.3 되찾은 양", ""] + sec_gap_closure() + ["#### 추적이 더한 것, 학습이 더한 것", ""] + sec_final_increments()
    full += ["### 4.4 Svanström test: 거리 구간·클래스별", ""] + sec_final_svanstrom_detail()
    full += ["### 4.5 Anti-UAV test (퇴행 확인)", ""] + sec_final_native("antiuav", "Anti-UAV test", False) + sec_size_final()
    full += ["### 4.6 Anti-UAV test 추적 (시퀀스 앞 300프레임, 6.7 Hz)", ""] + sec_final_native("antiuav_track", "Anti-UAV 추적 부분집합", True) + sec_native_sequence("antiuav_track")
    full += ["### 4.7 세 번째 데이터셋: CST Anti-UAV test (선택에 쓰지 않음)", ""] + sec_final_native("cst", "CST Anti-UAV test", True) + sec_native_sequence("cst")
    full += ["## 5. test에서 다시 잰 진단 (S1)", "", "dev에서 한 진단을 동결 뒤 test 영상에서 같은 코드로 다시 계산한 것이다.", ""]
    full += sec_diagnosis("diagnosis_test.json", "Svanström test") + sec_decomposition("diagnosis_test.json") + sec_distance("diagnosis_test.json")

    readme = ["#### 요인 분해 (1단계 모델, Svanström dev)", ""] + sec_decomposition()
    readme += ["#### Svanström test (오경보율을 1단계 수준에 맞춘 임계값)", ""] + sec_final_svanstrom("T_base") + sec_final_svanstrom_notes()
    readme += ["#### 되찾은 양", ""] + sec_gap_closure() + ["#### 추적이 더한 것, 학습이 더한 것", ""] + sec_final_increments()
    readme += ["#### Anti-UAV test (퇴행 확인)", ""] + sec_final_native("antiuav", "Anti-UAV test", False)
    readme += ["#### 세 번째 데이터셋: CST Anti-UAV test", ""] + sec_final_native("cst", "CST Anti-UAV test", True)
    return full, readme


def main():
    full, readme = build()
    (R / "RESULTS.md").write_text("\n".join(full).rstrip() + "\n")
    p = ROOT / "README.md"
    s = p.read_text()
    if os.environ.get("TDD_STAGE2_RESULTS"):
        print("smoke directory: README left alone")
    elif BEGIN in s and END in s:
        a, b = s.index(BEGIN) + len(BEGIN), s.index(END)
        p.write_text(s[:a] + "\n" + "\n".join(readme).rstrip() + "\n" + s[b:])
        print("README block updated")
    else:
        print("README has no stage 2 markers; only results/stage2/RESULTS.md written")
    print("results/stage2/RESULTS.md", len(full), "lines")


if __name__ == "__main__":
    main()
