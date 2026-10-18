# Detector · DETR 결과

**담당:** 광빈

데이터셋 v5 (2026-10-17 확정) 기준 결과를 모읍니다. 실행 방법은 [README](README.md)에 있습니다.

## 채점 기준

| 항목 | 내용 |
|------|------|
| 데이터 | `mendi_damage_v5_20261017`: v5 train / val / test, 2025 평가셋 val25 / test25 (val25 · test25는 640 · 1024 복사본만 있음) |
| 모델 · conf 선택 | v5 val + val25. 두 set의 합집합에서 균형 정확도가 가장 높은 conf 쌍을 0.05 간격 격자 탐색으로 고름 |
| 최종 채점 | v5 test + test25. 고른 conf를 고정하고(`--fixed-conf`) 마지막에 한 번만. 결과는 이 문서 끝에 따로 추가 |
| 사진 단위 판정 | 사진의 클래스별 최고 score가 기준 이상이면 heel / outsole, 둘 다면 both, 아니면 normal |
| 사진 단위 지표 | 균형 정확도 = 사진이 있는 클래스의 재현율 평균 (both는 0장이라 제외), 전체 정확도, 클래스별 재현율 N / H / O |
| 박스 지표 | v5 val. RT-DETR은 Ultralytics `val`로 best.pt를 다시 채점, RF-DETR은 rfdetr 내장 COCO 평가(`metrics.csv`의 best epoch 행). AP 열은 AP50-95 |
| 공통 설정 | seed 0, 30 epoch, 입력 크기 = 복사본 크기, 그 외 라이브러리 기본값 |

표에 붙일 한 줄은 스크립트가 출력합니다: Box 표는 `train_rtdetr.py` / `train_rfdetr.py`, Photo-level 표의 conf부터 재현율까지는 `eval_image_level.py`.

## Box metrics (v5 val)

| 날짜 | 모델 | 해상도 | epoch | seed | best epoch | mAP50 | mAP50-95 | outsole AP | heel_lining AP | 메모 |
|------|------|--:|--:|--:|--:|--:|--:|--:|--:|------|
| 2026-10-17 | RT-DETR-l | 640 | 30 | 0 | 28 | 0.429 | 0.182 | 0.063 | 0.301 | batch 8, bf16, AdamW lr0 1e-4 (기본값은 NaN, README 참고), 157분 |
| 2026-10-18 | RF-DETR-S | 640 | 30 | 0 | 9 | 0.502 | 0.218 | 0.103 | 0.318 | batch 4x4, 155분 |
| 2026-10-18 | RF-DETR-S + LoRA | 640 | 30 | 0 | 18 | 0.485 | 0.203 | 0.092 | 0.306 | batch 4x4, 174분 |

best epoch는 1부터 셉니다 (rfdetr `metrics.csv`의 epoch 값은 0부터라 1을 더했습니다).

## Photo-level

| 날짜 | 모델 | 해상도 | seed | conf (outsole / heel) | v5 val 균형 / 전체 | v5 val 재현율 N / H / O | val25 균형 / 전체 | val25 재현율 N / H / O | 메모 |
|------|------|--:|--:|--:|--:|--:|--:|--:|------|
| 2026-10-17 | RT-DETR-l | 640 | 0 | 0.45 / 0.55 | 0.813 / 0.859 | 0.87 / 0.88 / 0.69 | 0.762 / 0.747 | 0.74 / 0.79 / 0.75 | conf는 v5 val + val25 합집합에서 균형 정확도 최대 |
| 2026-10-18 | RF-DETR-S | 640 | 0 | 0.20 / 0.45 | 0.829 / 0.865 | 0.87 / 0.89 / 0.73 | 0.768 / 0.724 | 0.70 / 0.76 / 0.84 | 같음 |
| 2026-10-18 | RF-DETR-S + LoRA | 640 | 0 | 0.20 / 0.35 | 0.844 / 0.850 | 0.82 / 0.91 / 0.80 | 0.768 / 0.690 | 0.66 / 0.81 / 0.84 | 같음 |

640 관찰
- 박스 지표는 RF-DETR-S > RF-DETR-S + LoRA > RT-DETR-l. 사진 단위 균형 정확도는 v5 val에서 LoRA(0.844) > RF-DETR-S(0.829) > RT-DETR-l(0.813)이고, val25에서는 셋이 0.76~0.77로 같다.
- val25는 세 모델 모두 normal 재현율이 0.66~0.74로 낮다. 정상 사진을 outsole로 판정한 경우가 많다 (549장 중 79~115장). 2025년 사진은 학습 분포(2024년)와 달라서 생기는 차이로 보인다.
- outsole AP50-95가 0.06~0.10으로 낮다. 박스가 작고(640에서 짧은 변 약 33px) 수가 적은 클래스라 박스 위치 정밀도가 떨어진다. 사진 단위 outsole 재현율은 0.69~0.84로 박스 지표보다 훨씬 높다.
- RF-DETR-S는 best epoch가 9로 이르고 그 뒤 val mAP가 내려간다 (과적합 경향). LoRA는 18, RT-DETR은 28.

## 혼동 행렬

행 = 정답, 열 = 판정, 순서 normal / heel / outsole / both. 각 run의 `image_eval\image_eval.md`에서 복사했습니다.

### RT-DETR-l 640 (conf 0.45 / 0.55)

v5 val (n=639)

| 정답 \ 판정 | normal | heel | outsole | both |
|---|--:|--:|--:|--:|
| normal | 316 | 10 | 37 | 1 |
| heel | 21 | 198 | 2 | 3 |
| outsole | 16 | 0 | 35 | 0 |
| both | 0 | 0 | 0 | 0 |

val25 (n=680)

| 정답 \ 판정 | normal | heel | outsole | both |
|---|--:|--:|--:|--:|
| normal | 407 | 60 | 79 | 3 |
| heel | 11 | 49 | 2 | 0 |
| outsole | 13 | 1 | 52 | 3 |
| both | 0 | 0 | 0 | 0 |

### RF-DETR-S 640 (conf 0.20 / 0.45)

v5 val (n=639)

| 정답 \ 판정 | normal | heel | outsole | both |
|---|--:|--:|--:|--:|
| normal | 316 | 4 | 44 | 0 |
| heel | 16 | 200 | 2 | 6 |
| outsole | 14 | 0 | 37 | 0 |
| both | 0 | 0 | 0 | 0 |

val25 (n=680)

| 정답 \ 판정 | normal | heel | outsole | both |
|---|--:|--:|--:|--:|
| normal | 387 | 55 | 104 | 3 |
| heel | 12 | 47 | 2 | 1 |
| outsole | 6 | 2 | 58 | 3 |
| both | 0 | 0 | 0 | 0 |

### RF-DETR-S + LoRA 640 (conf 0.20 / 0.35)

v5 val (n=639)

| 정답 \ 판정 | normal | heel | outsole | both |
|---|--:|--:|--:|--:|
| normal | 299 | 17 | 47 | 1 |
| heel | 12 | 203 | 2 | 7 |
| outsole | 7 | 1 | 41 | 2 |
| both | 0 | 0 | 0 | 0 |

val25 (n=680)

| 정답 \ 판정 | normal | heel | outsole | both |
|---|--:|--:|--:|--:|
| normal | 361 | 72 | 115 | 1 |
| heel | 10 | 50 | 2 | 0 |
| outsole | 6 | 2 | 58 | 3 |
| both | 0 | 0 | 0 | 0 |
