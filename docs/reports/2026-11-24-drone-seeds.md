# T2-20 미세조정의 seed 반복 (seed 1, 2)

- 작성: 2026-11-24 새벽(KST). 브랜치 `hyh4867/drone-seeds`. 사전 등록: `docs/PREREGISTRATION_SEEDS.md`(커밋 8765bb1, 학습 시작 전에 push).
- 앞 보고서: `docs/reports/2026-11-22-thermal-drone-stage3.md`. 3단계 결과와 동결 설정은 바꾸지 않았다. 사이트, 공개 저장소, main은 건드리지 않았다.

**세 줄 요약**

1. **seed 간 차이.** 학습 seed만 바꾼 세 번의 학습(seed 0, 1, 2)에서 임계값과 무관한 지표는 거의 같았다. CST val AP@0.5의 범위는 0.007(0.409–0.415)로, 시험 자료를 다시 뽑을 때의 불확실성(95% 구간 반폭 약 0.10)보다 훨씬 작다. J4("도움이 됐다")와 J5("퇴행") 판정은 세 seed에서 같았다.
2. **핵심 수치.** 이득: CST val AP@0.5 평균 0.413, S1과의 차이 +0.328 ~ +0.335. 대가: Anti-UAV test AP@0.5 0.860–0.863(S1 0.885), 오경보율 5.5–7.8%(S1 3.5%), Svanström 새·비행기·헬기 오경보율 63.0–73.4%(S1 39.3%). 임계값에 달린 지표(오경보율, recall)는 seed마다 더 벌어졌고, 그 대부분은 seed마다 고른 임계값(0.909–0.938)의 차이였다(사후 확인, 5절).
3. **남은 문제.** seed가 3개뿐이라 seed 분포의 구간은 낼 수 없다. 세 시험 자료는 모두 3단계에서 이미 쓴 것이다. seed 1은 J5의 오경보율 항목을 프레임 1장 차이로 넘지 않았고(+1.99 %p, 한도 +2 %p), AP 항목으로 퇴행 판정을 받았다. 검수자 예측 R3은 그래서 틀렸다.

## 1. 한 일

1. `hyh4867/drone` 최신(9f5f5ad)에서 `hyh4867/drone-seeds`를 만들었다.
2. 실행 전에 사전 등록 문서와 도구를 커밋·push했다(8765bb1). 도구: `src/tdd/seeds/`(보호 장치가 있는 추론, 동결, 캐시 확인, 집계), `scripts/seeds_run.sh`(무인 실행), `scripts/independent_check_seeds.py`(표준 라이브러리 독립 재계산), `tests/test_seeds.py`. 학습 스크립트에는 출력 폴더를 바꾸는 `--group` 인자만 더했다.
3. seed 1, seed 2를 차례로 돌렸다. seed마다 학습 → 선택용 시퀀스 추론 → 캐시 확인 → 임계값 동결 → **동결 파일 커밋** → CST val → Anti-UAV test(+ 보충 프레임) → Svanström test → 캐시 확인 → 집계.
4. seed 0(3단계 실행)을 같은 집계 코드로 다시 집계해 3단계 파일과 일치하는지 확인하고, 3 seed 요약과 독립 재계산을 돌렸다.
5. 결과를 본 뒤 설명용 집계 하나를 덧붙였다(세 seed를 같은 임계값에서 읽기, 5절). 사전 등록에 없던 것이다.

### 실행 기록

출처: `results/seeds/logs/stages.log`(git에 넣지 않는 로그), `results/seeds/train_t2_20_s1.json`, `train_t2_20_s2.json`.

| 작업 | seed 1 | seed 2 |
|---|---|---|
| 학습 (1,075 iteration) | 11-23 20:34–21:20, 45.4분, 완료 | 11-23 23:37–11-24 00:19, 42.5분, 완료 |
| 선택용 추론 (5,596장) | 25분 | 21분 |
| 동결 커밋 | e041a59 (21:50) | 3cf98ef (00:41) |
| CST val (11,335장) | 43분 | 43분 |
| Anti-UAV test (9,582장) + 보충 (340장) | 36분 + 1분 | 36분 + 1분 |
| Svanström test (5,819장) | 22분 | 22분 |
| 집계 커밋 | aecc0e4 (23:36) | 32c81fd (02:27) |

- 전체는 2026-11-23 20:34에 시작해 11-24 02:27에 끝났다. seed 2 학습은 23:37에 시작했다(한도 02:30). 04:30 이후에 시작한 긴 실행은 없다.
- 학습 인자는 seed 0과 같고 다른 것은 `seed`(없음 → 1, 2), 실행 이름, `group`뿐이다. CST 학습 시퀀스 20개도 같다(파일로 비교).
- 시작 전 확인: ASTRA 작업 폴더는 마지막 커밋이 11-23 18:12였고 그 뒤 바뀐 파일이 없었으며, 20:34에 돌고 있던 python 프로세스는 레이더 세션의 CPU 시뮬레이션 2개뿐이었다. 이것을 "ASTRA 세션이 끝났다"로 읽고 시작했다.
- 메모리: 작업마다 시작 전에 스왑과 여유 메모리를 기록했다. 스왑 사용이 3 GB를 넘어(3,596 MB) 5분 기다린 일이 한 번 있었다(21:45, seed 1 선택용 캐시 확인 전). 여유 메모리는 가장 낮을 때 41%였다. 재시도와 실패는 없었다. 다른 세션의 프로세스는 건드리지 않았다.

## 2. 결과

모든 수치는 seed마다 자기 임계값(선택용 시퀀스에서 F1 최대)에서 읽은 값이다. 구간은 95% bootstrap(시퀀스 또는 영상 단위, B = 1000)이다. "범위"는 3 seed의 최대 − 최소다.

### 학습과 동결 (선택용 시퀀스 20개, 5,596장)

출처: `results/stage3/frozen_config.json`(seed 0), `results/seeds/frozen_seed1.json`, `frozen_seed2.json`, 학습 기록 파일.

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 학습 손실(1 epoch 평균) | 0.0400 | 0.0398 | 0.0388 |
| Anti-UAV val AP@0.5 (학습 끝) | 0.895 | 0.893 | 0.895 |
| 선택용 AP@0.5 | 0.639 | 0.643 | 0.625 |
| **동작 임계값** | **0.921** | **0.938** | **0.909** |
| 그 점의 F1 / precision / recall | 0.644 / 0.748 / 0.565 | 0.644 / 0.785 / 0.545 | 0.628 / 0.702 / 0.568 |

### 이득: CST val (40 시퀀스, 11,335장)

출처: `results/seeds/eval_seed0.json`, `eval_seed1.json`, `eval_seed2.json`, `summary.json`. 기준 S1은 AP@0.5 0.081이다.

| | seed 0 | seed 1 | seed 2 | 평균 | 범위 |
|---|---|---|---|---|---|
| **AP@0.5** | 0.415 [0.314, 0.521] | 0.414 [0.313, 0.520] | 0.409 [0.308, 0.512] | **0.413** | **0.007** |
| S1과의 차이 | +0.335 [0.241, 0.433] | +0.334 [0.240, 0.433] | +0.328 [0.235, 0.426] | +0.332 | 0.007 |
| recall (IoU 0.5) | 41.3% | 39.2% | 42.3% | 40.9% | 3.2 %p |
| recall (중심 기준) | 44.0% | 41.7% | 45.7% | 43.8% | 4.0 %p |
| tiny 구간 중심 기준 recall | 47.0% | 43.5% | 48.0% | 46.2% | 4.5 %p |
| tiny 구간 IoU 0.5 recall | 40.6% | 37.6% | 40.4% | 39.5% | 3.0 %p |
| precision | 57.6% | 60.7% | 55.5% | 57.9% | 5.3 %p |
| 박스 없는 프레임의 오경보율 (1,571장) | 13.4% (211장) | 8.1% (127장) | 15.5% (243장) | 12.3% | 7.4 %p |

- AP@0.5의 표본 표준편차는 0.004다(n = 3).
- seed 0과의 짝 차이(같은 시퀀스 bootstrap, 설명용): AP@0.5는 seed 1이 −0.001 [−0.007, +0.005], seed 2가 −0.007 [−0.019, +0.003]으로 둘 다 구간이 0을 포함한다. 중심 기준 recall은 seed 1이 −2.3 %p [−3.5, −1.4], seed 2가 +1.7 %p [+1.0, +2.4]로 0을 벗어난다. 임계값이 높은 seed 1은 덜 잡고 덜 울리고, 낮은 seed 2는 반대다.

### 대가 1: Anti-UAV test (91 시퀀스, 9,582장)

출처: 같은 파일. 기준은 1단계 모델과 설정: AP@0.5 0.885, recall 84.9%, 오경보율 3.5%(1,153장 중 40장).

| | seed 0 | seed 1 | seed 2 | 평균 | 범위 |
|---|---|---|---|---|---|
| AP@0.5 | 0.863 | 0.860 | 0.863 | 0.862 | 0.003 |
| 차이 (한도 −0.02) | −0.022 [−0.031, −0.014] | −0.025 [−0.036, −0.016] | −0.022 [−0.031, −0.014] | −0.023 | 0.003 |
| recall (IoU 0.5) | 84.1% | 83.4% | 84.2% | 83.9% | 0.8 %p |
| 차이 (한도 −2 %p) | −0.8 %p [−1.5, −0.1] | −1.5 %p [−2.3, −0.7] | −0.7 %p [−1.5, −0.0] | −1.0 %p | 0.8 %p |
| 오경보율 (박스 없는 1,153장) | 6.9% (79장) | 5.5% (63장) | 7.8% (90장) | 6.7% | 2.3 %p |
| 차이 (한도 +2 %p) | +3.4 %p [0.3, 7.2] | **+1.99 %p** [0.1, 5.3] | +4.3 %p [1.3, 9.5] | +3.2 %p | 2.3 %p |
| 한도를 넘은 항목 | AP, 오경보율 | AP | AP, 오경보율 | | |

- 보충 프레임을 더한 small 구간(8–16 px, 365개) IoU 0.5 recall은 79.7% / 78.9% / 80.3%다(S1 71.8%). tiny 구간(17개)은 세 seed 모두 0%다.
- seed 1의 오경보율 차이 +1.99 %p는 한도 바로 아래다. 63장이 울렸고, 64장이었으면 한도를 넘는다.

### 대가 2: Svanström test (183개 영상, 5,819장)

출처: 같은 파일. 기준 S1(같은 프레임, 3단계 캐시): 새·비행기·헬기 프레임(3,335장) 오경보율 39.3%, 중심 기준 recall 85.9%.

| | seed 0 | seed 1 | seed 2 | 평균 | 범위 |
|---|---|---|---|---|---|
| **새·비행기·헬기 오경보율** | 73.3% [67.1, 79.0] | 63.0% [56.4, 69.7] | 73.4% [67.1, 79.4] | **69.9%** | **10.4 %p** |
| S1과의 차이 | +33.9 %p [28.3, 39.7] | +23.7 %p [19.1, 28.5] | +34.0 %p [28.3, 40.0] | +30.5 %p | 10.4 %p |
| 새 / 비행기 / 헬기 | 59.9 / 79.7 / 83.0% | 46.4 / 70.8 / 75.7% | 59.7 / 79.7 / 83.8% | | |
| 드론 중심 기준 recall (박스 2,389개) | 95.2% | 94.2% | 95.3% | 94.9% | 1.1 %p |
| S1과의 차이 | +9.4 %p [5.5, 13.6] | +8.3 %p [4.7, 12.3] | +9.5 %p [5.6, 13.7] | +9.1 %p | 1.1 %p |
| 드론 IoU 0.5 recall (k = 1) | 29.7% | 30.1% | 32.5% | 30.8% | 2.8 %p |

- 3단계 추가 6의 해석 규칙으로는 세 seed 모두 "Svanström에서는 이득이 있었다"(중심 기준 recall 차이의 구간 하한 > 0)이다. 동시에 세 seed 모두 새·비행기·헬기 프레임의 60% 이상에서 경보가 울린다.

## 3. 판정과 읽는 규칙

출처: `results/seeds/summary.json`의 `verdicts`.

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| J4 (CST val, AP 차이의 구간 하한 > 0) | 도움이 됐다 (+0.335, 하한 0.241) | 도움이 됐다 (+0.334, 하한 0.240) | 도움이 됐다 (+0.328, 하한 0.235) |
| AP@0.5 0.30 선 | 넘음 (0.415) | 넘음 (0.414) | 넘음 (0.409) |
| J5 (Anti-UAV) | **퇴행** (AP, 오경보율) | **퇴행** (AP) | **퇴행** (AP, 오경보율) |

사전에 고정한 읽는 규칙을 적용하면 다음과 같다.

- **판정: 판정은 seed에 따라 바뀌지 않았다.** 세 seed의 J4가 같고 J5가 같다. 다만 J5에서 한도를 넘은 항목은 다르다(seed 1은 AP만).
- **크기: seed 차이는 시험 자료를 다시 뽑을 때의 불확실성보다 작다.** CST val AP@0.5의 3 seed 범위 0.007은 기준선 0.10(seed 0의 구간 반폭)보다 작다.

## 4. 예측 대 실제

판정은 `results/seeds/summary.json`의 `predictions`에서 코드로 했다.

| 번호 | 예측 | 실제 | 결과 |
|---|---|---|---|
| R1 (검수자) | seed 1, 2의 CST val AP@0.5 각각 0.33–0.50 | 0.414, 0.409 | 맞음 |
| R2 (검수자) | 3 seed 범위 0.08 이하 | 0.007 | 맞음. 예측한 상한의 1/10 수준 |
| R3 (검수자) | 세 seed 모두 Anti-UAV에서 퇴행(오경보율 기준) | 오경보율 차이 +3.4, +1.99, +4.3 %p | **틀림.** seed 1이 한도 +2 %p를 프레임 1장 차이로 넘지 않았다. J5 판정 자체는 세 seed 모두 퇴행이다(AP 항목) |
| R4 (검수자) | 세 seed 모두 Svanström 오경보율 60% 이상 | 73.3%, 63.0%, 73.4% | 맞음 |
| A1 (에이전트) | seed 1, 2의 CST val AP@0.5 각각 0.37–0.46 | 0.414, 0.409 | 맞음 |
| A2 | 3 seed 범위 0.06 이하 | 0.007 | 맞음. 범위를 훨씬 넓게 잡았다 |
| A3 | seed 1, 2의 J4 "도움이 됐다", AP 0.30 이상 | 둘 다 그렇다 | 맞음 |
| A4 | tiny 구간 중심 기준 recall 각각 40–54% | 43.5%, 48.0% | 맞음 |
| A5 | 임계값 각각 0.88–0.96 | 0.938, 0.909 | 맞음 |
| A6 | Anti-UAV AP 차이 각각 −0.04 ~ −0.005 | −0.025, −0.022 | 맞음 |
| A7 | seed 1, 2의 J5 퇴행 | 둘 다 퇴행 | 맞음. seed 1은 AP 항목만으로 |
| A8 | Anti-UAV 오경보율 각각 5–10% | 5.5%, 7.8% | 맞음 |
| A9 | Svanström 오경보율 각각 60–85% | 63.0%, 73.4% | 맞음 |
| A10 | Svanström 오경보율 범위 15 %p 이하 | 10.4 %p | 맞음 |
| A11 | Svanström 중심 기준 recall 차이 둘 다 양수 | +8.3, +9.5 %p | 맞음 |

두 예측 모두 seed 차이를 실제보다 크게 봤다. 에이전트 예측은 범위를 넓게 잡은 것이 많아서 "모두 맞음"이 정밀한 예측이었다는 뜻은 아니다.

## 5. 사후 확인: 같은 임계값에서 읽기 (사전 등록에 없음, 설명용)

결과를 본 뒤 덧붙였다. 판정에 쓰지 않았고 새 추론도 없다. 세 seed의 캐시를 seed 0의 동결 임계값(0.921) 하나로 읽었다. 출처: `results/seeds/posthoc_common_threshold.json`(코드 `src/tdd/seeds/posthoc_threshold.py`, 커밋 29f742f).

| 지표 | seed 0 / 1 / 2 (같은 임계값 0.921) | 범위 | 자기 임계값일 때의 범위 |
|---|---|---|---|
| CST val 중심 기준 recall | 44.0 / 44.6 / 44.2% | 0.6 %p | 4.0 %p |
| CST val tiny 구간 중심 기준 recall | 47.0 / 47.3 / 46.1% | 1.2 %p | 4.5 %p |
| CST val 오경보율 | 13.4 / 11.6 / 13.2% | 1.9 %p | 7.4 %p |
| Anti-UAV recall (IoU 0.5) | 84.1 / 83.8 / 83.9% | 0.3 %p | 0.8 %p |
| Anti-UAV 오경보율 | 6.9 / 7.4 / 6.6% | 0.8 %p | 2.3 %p |
| Svanström 새·비행기·헬기 오경보율 | 73.3 / 71.6 / 68.5% | 4.7 %p | 10.4 %p |
| Svanström 중심 기준 recall | 95.2 / 95.1 / 95.1% | 0.2 %p | 1.1 %p |

- 임계값을 맞추면 범위가 절반 이하로 줄어든다. seed마다 달라진 것은 주로 가중치가 아니라 **선택용 시퀀스에서 고른 임계값**이다. 선택용 시퀀스의 F1 최댓값 근처가 평평해서(F1 0.628–0.644) 임계값이 0.909–0.938 사이에서 움직였다.
- 같은 임계값에서는 seed 1의 Anti-UAV 오경보율도 7.4%(차이 +3.9 %p)로 한도를 넘는다. seed 1이 오경보율 항목을 넘지 않은 것은 임계값이 높게 잡혔기 때문이다.
- Svanström 오경보율은 같은 임계값에서도 4.7 %p 벌어진다. 세 자료 중 seed에 가장 민감하다.

## 6. 사전 등록과 달라진 점

1. **사후 집계를 덧붙였다(5절).** 등록하지 않은 설명용 집계다. 결과를 본 뒤에 정했고, 등록한 수치와 판정은 그대로다.
2. **독립 재계산의 범위.** 요약 23행 중 20행을 다시 계산했다. 선택용 시퀀스의 임계값과 AP, CST val precision 3행은 독립 스크립트가 다시 계산하지 않는다.
3. **실행 전 seed 0 집계.** 사전 등록 2절에 적은 실행 전 seed 0 집계는 작업 트리가 깨끗하지 않은 상태(`--allow-dirty`)로 돌린 확인용이었고, 그 파일은 지웠다. 보고한 seed 0 수치는 실행이 끝난 뒤 깨끗한 트리(커밋 32c81fd)에서 다시 만든 것이다.
4. 그 밖의 순서, 인자, 자료, 지표, 판정 규칙, 시간 한도는 등록한 대로다. 학습과 추론의 재시도는 없었다.

## 7. 검증

- **동결 뒤에만 시험 추론.** seed 1의 동결 커밋 e041a59(21:50:51) 뒤에 CST val 추론이 시작됐고(21:50:53), seed 2는 3cf98ef(00:41:16) 뒤에 시작됐다(00:41:18). 로그의 START 줄에 그때의 커밋이 적혀 있다. 동결 파일이 커밋되지 않으면 시험 추론을 거부하는 것은 테스트로 확인했다.
- **깨끗한 트리에서 집계.** `eval_seed*.json`, `summary.json`, `posthoc_common_threshold.json`의 `git_dirty`는 모두 false다.
- **캐시 확인: 10개 모두 통과.** 출처: `results/seeds/cache_check__seed{1,2}_{hold,test}.json`. meta, 체크포인트 sha256(동결값과 일치), 로그의 DONE 줄, 줄 수가 맞고, 재추론 5장의 점수 차이는 0.00000이었다. 가장 가까운 다른 캐시(seed 0)와의 차이는 0.0095–0.043이다. seed 1의 Anti-UAV 캐시는 seed 0 캐시와 0.0095밖에 차이 나지 않는다. 두 모델이 이 자료에서 거의 같은 점수를 낸다는 뜻이고, 자기 캐시와의 차이가 0이라 구별은 된다.
- **seed 0 재현.** 같은 집계 코드에 넣은 seed 0이 3단계 파일의 값 58개와 최대 차이 0으로 일치했다(`eval_seed0.json`의 `reproduces_stage3`).
- **독립 재계산: 모두 일치.** `python3 scripts/independent_check_seeds.py` → `results/seeds/independent_check.json`. seed 3개의 CST val, Anti-UAV(J5 항목 포함), Svanström 수치와 3 seed 평균·범위 20행이 최대 차이 1.1e-16으로 일치했다.

### pytest 전체 출력 (33개 통과)

```
============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-9.2.0, pluggy-1.7.0 -- <저장소>/projects/thermal-drone-detection/.venv/bin/python
cachedir: .pytest_cache
rootdir: <저장소>/projects/thermal-drone-detection
collecting ... collected 33 items

tests/test_metrics.py::test_iou_basic PASSED                             [  3%]
tests/test_metrics.py::test_ap_perfect_and_ordering PASSED               [  6%]
tests/test_metrics.py::test_duplicate_detection_is_fp PASSED             [  9%]
tests/test_metrics.py::test_operating_point_empty_frames_and_bins PASSED [ 12%]
tests/test_metrics.py::test_best_f1_threshold PASSED                     [ 15%]
tests/test_metrics.py::test_bootstrap_ap_matches_metric_and_brackets_point PASSED [ 18%]
tests/test_seeds.py::test_names_and_paths_keep_stage3_apart PASSED       [ 21%]
tests/test_seeds.py::test_test_sets_are_refused_until_the_frozen_file_is_committed PASSED [ 24%]
tests/test_seeds.py::test_frozen_committed_needs_tracked_and_unmodified PASSED [ 27%]
tests/test_seeds.py::test_checkpoint_must_equal_the_frozen_sha PASSED    [ 30%]
tests/test_seeds.py::test_spread_and_prediction_checks PASSED            [ 33%]
tests/test_seeds.py::test_seed0_threshold_follows_the_stage3_rule PASSED [ 36%]
tests/test_seeds.py::test_independent_script_uses_no_project_code PASSED [ 39%]
tests/test_stage2.py::test_split_is_frozen_and_reproducible PASSED       [ 42%]
tests/test_stage2.py::test_split_has_no_video_in_two_splits_and_is_stratified PASSED [ 45%]
tests/test_stage2.py::test_adaptation_videos_come_only_from_dev_a PASSED [ 48%]
tests/test_stage2.py::test_frame_sets_do_not_share_videos PASSED         [ 51%]
tests/test_stage2.py::test_per_gt_tables_equal_greedy_matching PASSED    [ 54%]
tests/test_stage2.py::test_threshold_for_far_never_exceeds_target PASSED [ 57%]
tests/test_stage2.py::test_scale_boxes_keeps_centre_and_score PASSED     [ 60%]
tests/test_stage2.py::test_tiles_cover_image_and_tta_merge_rule PASSED   [ 63%]
tests/test_stage2.py::test_mofn_confirmation_and_deletion PASSED         [ 66%]
tests/test_stage2.py::test_tracker_follows_a_moving_target_with_one_id PASSED [ 69%]
tests/test_stage2.py::test_tracker_gate_and_det_threshold PASSED         [ 72%]
tests/test_stage2.py::test_sequence_metrics_latency_maintenance_and_false_tracks PASSED [ 75%]
tests/test_stage2.py::test_paired_bootstrap_of_identical_configurations_is_zero PASSED [ 78%]
tests/test_stage2.py::test_dev_halves_partition_dev_and_cache_fallback PASSED [ 81%]
tests/test_stage3.py::test_independent_script_matches_pipeline PASSED    [ 84%]
tests/test_stage3.py::test_bin_bootstrap_point_estimates_and_pairing PASSED [ 87%]
tests/test_stage3.py::test_far10_readout_keeps_alarm_rate_at_or_below_ten_percent PASSED [ 90%]
tests/test_stage3.py::test_cst_train_split_is_deterministic_disjoint_and_nested PASSED [ 93%]
tests/test_stage3.py::test_reduced_val_boxes_are_quartered PASSED        [ 96%]
tests/test_stage3.py::test_cst_val_is_refused_before_the_freeze PASSED   [100%]

============================= 33 passed in 11.51s ==============================
```

실행: `PYTHONPATH=src .venv/bin/python -m pytest -v`, 커밋 29f742f에서. 출력의 저장소 절대경로만 `<저장소>`로 바꿨다. 건너뛴 테스트는 없다(자료와 캐시가 이 맥에 있다).

## 8. 남은 문제

1. **seed 3개.** 평균과 범위만 냈다. seed 분포의 구간이나 "seed 차이가 없다"는 결론은 낼 수 없다. 범위 0.007은 세 번에서 본 값이다.
2. **이미 쓴 시험 자료.** CST val, Anti-UAV test, Svanström test 모두 3단계에서 쓴 자료다. 여기서 고른 것은 없지만 처음 보는 시험은 아니다.
3. **임계값 규칙이 대가를 좌우한다.** 오경보율의 seed 차이는 대부분 임계값 차이였다(5절, 사후 확인). 선택용 시퀀스의 F1 곡선이 평평해서 임계값이 쉽게 움직인다. 임계값 규칙을 바꾸는 일은 하지 않았다.
4. **J5의 오경보율 항목은 경계에 있다.** seed 1은 프레임 1장 차이로 한도 아래였다. "오경보율 기준으로 퇴행"이라는 문장은 seed에 따라 달라진다. "AP 기준으로 퇴행"은 세 seed에서 같다.
5. **같은 seed를 두 번 돌리지 않았다.** seed 사이의 차이에는 MPS 연산의 비결정성이 있다면 그것도 섞여 있다.
6. **T2-20만 반복했다.** S1, B, T1a, T1b 같은 다른 설정은 여전히 한 번씩만 학습했다. 배율 m도 seed마다 다시 고르지 않았다.
7. **Svanström 오경보율은 세 seed 모두 60%를 넘는다.** seed를 바꿔도 이 대가는 사라지지 않았다.
8. 사이트와 공개 저장소에는 반영하지 않았다. 반영하려면 승인이 필요하다.

## 9. 사이트 문구 제안 (반영하지 않음)

지금 문구: "학습은 설정당 한 번이었습니다."

제안:

> 학습은 설정당 한 번이었습니다. 목표 자료로 미세조정한 구성(T2-20)만 학습 seed를 바꿔 세 번 학습했습니다. CST val AP@0.5는 0.409–0.415로 거의 같았고, 도움이 됐다는 판정과 Anti-UAV 퇴행 판정도 세 번 모두 같았습니다. 오경보율은 seed마다 고른 임계값에 따라 달라졌습니다(Anti-UAV 5.5–7.8%, Svanström 새·비행기·헬기 프레임 63–73%). seed가 세 개뿐이라 범위만 적습니다.

T2-20의 대표 수치는 3단계의 seed 0(0.415 [0.314, 0.521])을 그대로 두는 것을 권한다. 세 seed 중 가운데도 최고도 아닌, 먼저 등록하고 돌린 값이기 때문이다.

## 10. 커밋

브랜치 `hyh4867/drone-seeds`(시작점 9f5f5ad).

| 커밋 | 내용 |
|---|---|
| 8765bb1 | 사전 등록 문서와 seed 도구(실행 전) |
| e041a59 | seed 1 동결(학습 기록, 선택용 캐시 확인, 임계값). 시험 추론 전 |
| aecc0e4 | seed 1 집계(CST val, Anti-UAV test, Svanström test)와 시험 캐시 확인 |
| 3cf98ef | seed 2 동결. 시험 추론 전 |
| 32c81fd | seed 2 집계와 시험 캐시 확인 |
| 29f742f | 사후 집계 코드(같은 임계값에서 읽기) |
| 31e9ec0 | seed 0 재집계, 3 seed 요약, 독립 재계산, 사후 집계 결과 |
| (이 보고서) | 이 커밋 바로 다음. 브랜치의 맨 끝 |

가중치(`runs/seeds/`), 예측 캐시, 로그는 git에 넣지 않았다. 다시 만들려면 `screen -dmS drone-seeds bash scripts/seeds_run.sh`를 실행한다(자료 내려받기는 README 참고).
