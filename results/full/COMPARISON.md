# 축소 실행 vs 전체 실행 (자동 생성: `python -m tdd.report_full`)

축소 실행 = 포트폴리오에 인용된 결과(1 epoch, 축소 표본). 전체 실행 = 4 epoch, 전체 표본.
대괄호: 전체 실행은 **영상 단위 bootstrap 95% 구간**(B=1000), 축소 실행은 Wilson 구간(프레임 독립 가정, 참고용).

| 지표 | 방법 | 축소 실행 | 전체 실행 | 출처 (축소 → 전체) |
|---|---|---|---|---|
| val AP@0.5 (임계값 선택용) | baseline | 0.1296 (800 frames) | 0.1305 (1596 frames) | `results/baseline` → `results/full/baseline` |
| 임계값 (val F1 최대) | baseline | 5.1659 | 4.9407 | `results/baseline` → `results/full/baseline` |
| Anti-UAV test 프레임 / GT 박스 / 드론 없는 프레임 | baseline | 544 / 430 / 114 | 9,582 / 8,429 / 1,153 | `results/baseline` → `results/full/baseline` |
| test AP@0.5 | baseline | 0.1228 | 0.1388 [0.108–0.177] | `results/baseline` → `results/full/baseline` |
| Precision | baseline | 13.6% [12.1–15.3] | 14.7% [12.7–17.0] | `results/baseline` → `results/full/baseline` |
| Recall | baseline | 58.6% [53.9–63.2] | 56.5% [48.7–64.2] | `results/baseline` → `results/full/baseline` |
| 드론 없는 프레임 오경보율 | baseline | 99.1% (113/114) [95.2–99.8] | 99.5% (1147/1153) [98.2–100.0] | `results/baseline` → `results/full/baseline` |
| recall, 크기 tiny | baseline | 0.0% (0/6) | 0.0% (0/17) | `results/baseline` → `results/full/baseline` |
| recall, 크기 small | baseline | 1.8% (2/110) | 1.6% (6/365) | `results/baseline` → `results/full/baseline` |
| recall, 크기 medium | baseline | 31.6% (31/98) | 38.5% (757/1965) | `results/baseline` → `results/full/baseline` |
| recall, 크기 large | baseline | 66.4% (213/321) | 61.9% (3880/6272) | `results/baseline` → `results/full/baseline` |
| recall, 크기 xlarge | baseline | 88.9% (8/9) | 81.3% (122/150) | `results/baseline` → `results/full/baseline` |
| Svanström 드론 recall CLOSE | baseline | 8.7% (2/23, 23 videos) [2.4–26.8] | 25.1% (367/1462, 23 videos) [15.0–35.8] | `results/baseline` → `results/full/baseline` |
| Svanström 드론 recall MEDIUM | baseline | 3.4% (3/88, 93 videos) [1.2–9.6] | 9.6% (529/5528, 93 videos) [6.0–13.0] | `results/baseline` → `results/full/baseline` |
| Svanström 드론 recall DISTANT | baseline | 0.0% (0/38, 38 videos) [0.0–9.2] | 0.1% (3/2392, 38 videos) [0.0–0.4] | `results/baseline` → `results/full/baseline` |
| Svanström 드론 recall ALL | baseline | 3.4% (5/149, 154 videos) [1.4–7.6] | 9.6% (899/9382, 154 videos) [6.4–12.5] | `results/baseline` → `results/full/baseline` |
| BIRD 프레임 오경보율 | baseline | 74.0% (57/77) [63.3–82.5] | 77.5% (3777/4872) [69.1–85.0] | `results/baseline` → `results/full/baseline` |
| BIRD 물체 오인율 (라벨 있는 프레임) | baseline | – (0/0) [nan–nan] | 0.1% (3/4330) [0.0–0.2] | `results/baseline` → `results/full/baseline` |
| AIRPLANE 프레임 오경보율 | baseline | 87.8% (65/74) [78.5–93.5] | 89.1% (4166/4678) [82.4–94.8] | `results/baseline` → `results/full/baseline` |
| AIRPLANE 물체 오인율 (라벨 있는 프레임) | baseline | 2.7% (2/74) [0.7–9.3] | 4.7% (218/4678) [1.8–8.4] | `results/baseline` → `results/full/baseline` |
| HELICOPTER 프레임 오경보율 | baseline | 76.4% (42/55) [63.6–85.6] | 78.7% (2689/3415) [68.6–88.0] | `results/baseline` → `results/full/baseline` |
| HELICOPTER 물체 오인율 (라벨 있는 프레임) | baseline | 7.3% (4/55) [2.9–17.3] | 6.3% (215/3415) [2.3–11.2] | `results/baseline` → `results/full/baseline` |
| val AP@0.5 (임계값 선택용) | model | 0.8839 (800 frames) | 0.9374 (1596 frames) | `results/model` → `results/full/model` |
| 임계값 (val F1 최대) | model | 0.8979 | 0.9542 | `results/model` → `results/full/model` |
| Anti-UAV test 프레임 / GT 박스 / 드론 없는 프레임 | model | 544 / 430 / 114 | 9,582 / 8,429 / 1,153 | `results/model` → `results/full/model` |
| test AP@0.5 | model | 0.8457 | 0.8847 [0.839–0.926] | `results/model` → `results/full/model` |
| Precision | model | 95.5% [92.8–97.2] | 96.7% [95.2–97.8] | `results/model` → `results/full/model` |
| Recall | model | 79.1% [75.0–82.7] | 84.9% [80.0–89.6] | `results/model` → `results/full/model` |
| 드론 없는 프레임 오경보율 | model | 2.6% (3/114) [0.9–7.4] | 3.5% (40/1153) [0.8–8.9] | `results/model` → `results/full/model` |
| recall, 크기 tiny | model | 0.0% (0/6) | 0.0% (0/17) | `results/model` → `results/full/model` |
| recall, 크기 small | model | 69.1% (76/110) | 71.8% (262/365) | `results/model` → `results/full/model` |
| recall, 크기 medium | model | 73.5% (72/98) | 78.2% (1537/1965) | `results/model` → `results/full/model` |
| recall, 크기 large | model | 80.4% (258/321) | 86.7% (5440/6272) | `results/model` → `results/full/model` |
| recall, 크기 xlarge | model | 100.0% (9/9) | 100.0% (150/150) | `results/model` → `results/full/model` |
| Svanström 드론 recall CLOSE | model | 52.2% (12/23, 23 videos) [33.0–70.8] | 66.0% (965/1462, 23 videos) [54.0–77.2] | `results/model` → `results/full/model` |
| Svanström 드론 recall MEDIUM | model | 47.7% (42/88, 93 videos) [37.6–58.0] | 57.4% (3175/5528, 93 videos) [50.1–64.4] | `results/model` → `results/full/model` |
| Svanström 드론 recall DISTANT | model | 18.4% (7/38, 38 videos) [9.2–33.4] | 17.6% (421/2392, 38 videos) [9.8–26.7] | `results/model` → `results/full/model` |
| Svanström 드론 recall ALL | model | 40.9% (61/149, 154 videos) [33.4–49.0] | 48.6% (4561/9382, 154 videos) [42.1–54.8] | `results/model` → `results/full/model` |
| BIRD 프레임 오경보율 | model | 33.8% (26/77) [24.2–44.9] | 23.0% (1119/4872) [16.1–30.3] | `results/model` → `results/full/model` |
| BIRD 물체 오인율 (라벨 있는 프레임) | model | – (0/0) [nan–nan] | 3.1% (133/4330) [1.4–5.3] | `results/model` → `results/full/model` |
| AIRPLANE 프레임 오경보율 | model | 48.6% (36/74) [37.6–59.8] | 45.3% (2118/4678) [35.6–54.6] | `results/model` → `results/full/model` |
| AIRPLANE 물체 오인율 (라벨 있는 프레임) | model | 1.4% (1/74) [0.2–7.3] | 11.8% (552/4678) [7.2–16.6] | `results/model` → `results/full/model` |
| HELICOPTER 프레임 오경보율 | model | 52.7% (29/55) [39.8–65.3] | 51.7% (1764/3415) [40.3–63.5] | `results/model` → `results/full/model` |
| HELICOPTER 물체 오인율 (라벨 있는 프레임) | model | 14.5% (8/55) [7.6–26.2] | 19.1% (651/3415) [11.3–29.0] | `results/model` → `results/full/model` |

## 기준선 하이퍼파라미터 (전체 실행)

선택: kernel 61, k 1.5, OSD mask True — val AP@0.5 0.1305 (`results/full/baseline/val_grid_search.json`). 그리드 kernel [9, 15, 21, 31, 45, 61, 81, 101], k [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0].
그리드 경계 여부: 경계에 걸림 → k.

## 학습 (전체 실행)

| epoch | train loss | val AP@0.5 (전체 val) | 분 |
|---|---|---|---|
| 1 | 0.0405 | 0.8916 (1596 frames) | 43 |
| 2 | 0.0406 | 0.9303 (1596 frames) | 44 |
| 3 | 0.0414 | 0.9373 (1596 frames) | 45 |
| 4 | 0.0415 | 0.9374 (1596 frames) | 45 |

선택 epoch: 4, 완료 여부: True (`results/full/model/train_log.json`)

