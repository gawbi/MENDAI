# Detector · YOLO 결과

**담당:** 보인

데이터셋 v5 (2026-10-17 확정) 기준 결과입니다. 실행 방법은 [README](README.md)에 있습니다.

## 요약

- 기본 학습 2회: **B11 = YOLO11m**, **B26 = YOLO26m**. 둘 다 사전학습 가중치에서 시작, 입력 1024, 50 epoch, seed 0, 같은 설정.
- 사진 단위 균형 정확도 (conf 0.25 고정): v5 val에서 B11 64.4% / B26 66.0%, val25에서 B11 62.1% / B26 61.6%. 1회 학습 결과라 두 모델의 차이는 작은 편이고 우열을 정하기 어렵습니다.
- 두 모델 모두 **outsole 재현율이 13.7~24.6%로 매우 낮습니다.** outsole 사진 대부분을 normal로 판정합니다. 박스 지표에서도 outsole AP50-95가 0.03~0.04로 heel_lining(0.32~0.34)보다 훨씬 낮습니다.
- test(v5 test, test25)는 아직 채점하지 않았습니다.

## 채점 기준

| 항목 | 내용 |
|------|------|
| 데이터 | `mendi_damage_v5_20261017` 1024 복사본. split · 라벨 · 클래스 수정 없음 |
| 평가셋 | v5 val (639장), val25 (680장). 합치지 않고 따로 출력 |
| 가중치 | 각 run의 `best.pt` (Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95, v5 val 기준) |
| 사진 단위 판정 | conf 0.25 이상인 박스의 클래스 존재 여부. 없음 → normal, heel_lining만 → heel, outsole만 → outsole, 둘 다 → both. 같은 클래스 박스 여러 개는 both가 아님. IoU 조건 없음 |
| 사진 정답 | `photo_labels/*.csv`의 `label` 열 |
| 지표 | 균형 정확도 = (Recall normal + Recall heel + Recall outsole) / 3, 전체 정확도 = 4개 대각선 합 / N, 클래스별 재현율. both 정답 0장 → Recall both N/A |
| 예측 설정 | imgsz 1024, NMS IoU 0.7, max_det 300, TTA 없음. `nms` 미지정 → Ultralytics 8.4.182 기본 NMS 경로 (YOLO26도 one-to-many head + NMS, NMS-free 모드는 아직 평가하지 않음) |
| conf | 0.25 (Ultralytics 기본값, 미리 고정, 두 클래스 공통). 튜닝하지 않음 |

## 학습 설정 (두 run 공통)

| 항목 | 값 |
|------|------|
| 시작 가중치 | `yolo11m.pt` / `yolo26m.pt` (공식 사전학습) |
| epoch · 입력 | 50 · 1024 |
| batch | 4 (8은 8 GB에서 메모리 부족). Ultralytics가 nbs=64장까지 gradient를 누적 |
| optimizer | AdamW, lr0 0.001, lrf 0.01, weight_decay 0.0005, warmup 3 epoch (로그에서 실제 적용 확인) |
| 조기 종료 | patience 100 (> 50, 50 epoch 모두 수행) |
| 증강 | Ultralytics 기본값: mosaic 1.0 (마지막 10 epoch 끔), scale 0.5, fliplr 0.5, hsv 0.015/0.7/0.4, translate 0.1 |
| 기타 | seed 0, deterministic, AMP(fp16), workers 8 |

두 run 모두 학습 중 batch 자동 축소(`Reducing to batch`)는 없었습니다.

## Box metrics (v5 val, best.pt)

| 날짜 | 실험 | 모델 | best epoch | P | R | mAP50 | mAP50-95 | outsole AP50 / AP50-95 | heel_lining AP50 / AP50-95 | 학습 시간 | 메모 |
|------|------|------|--:|--:|--:|--:|--:|--:|--:|--:|------|
| 2026-10-20 | B11 | YOLO11m | 45 | 0.364 | 0.439 | 0.399 | 0.177 | 0.072 / 0.033 | 0.727 / 0.322 | 약 2.9시간 | epoch 10 도중 배터리 방전으로 꺼져 `last.pt`(epoch 9)에서 이어 학습 |
| 2026-10-20 | B26 | YOLO26m | 47 | 0.413 | 0.415 | 0.414 | 0.187 | 0.079 / 0.039 | 0.749 / 0.336 | 2.8시간 | 중단 없음 |

best epoch는 1부터 셉니다. 박스 지표는 학습 종료 시 Ultralytics가 `best.pt`로 다시 잰 값입니다.

## Photo-level

| 평가셋 | 실험 | 모델 | 균형 정확도 | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |
|------|------|------|--:|--:|--:|--:|--:|--:|
| v5 val | B11 | YOLO11m · 50 · 1024 | **64.4%** | 84.7% | 94.5% | 84.8% | 13.7% | N/A |
| v5 val | B26 | YOLO26m · 50 · 1024 | **66.0%** | 85.1% | 93.7% | 86.6% | 17.6% | N/A |
| val25 | B11 | YOLO11m · 50 · 1024 | **62.1%** | 78.7% | 85.8% | 75.8% | 24.6% | N/A |
| val25 | B26 | YOLO26m · 50 · 1024 | **61.6%** | 77.1% | 83.8% | 82.3% | 18.8% | N/A |

관찰
- 가장 큰 오류는 **outsole → normal**입니다: v5 val 51장 중 B11 43장, B26 41장, val25 69장 중 B11 47장, B26 51장을 normal로 판정했습니다. 박스 지표에서도 outsole AP50이 0.07~0.08에 그칩니다.
- val25(2025년 사진)에서 normal 재현율이 v5 val보다 8.7~9.9%p 낮습니다. 주로 normal → heel 오탐입니다 (B11 56장, B26 61장).
- heel 재현율은 B26이 두 셋 모두 더 높고 (+1.8%p, +6.5%p), normal 재현율은 B11이 조금 더 높습니다.
- conf 0.25 하나로 두 클래스를 판정했으므로, outsole 재현율은 conf 조정(재학습 없음)으로 달라질 수 있습니다. `per_image.csv`의 클래스별 최고 confidence로 다시 계산할 수 있습니다.

## 사진 단위 결과 (run별)

행이 실제 정답, 열이 모델 예측입니다. 순서는 normal → heel → outsole → both.

### B11 · YOLO11m 1024 (`yolo11m_1024_s0`, conf 0.25)

#### v5 val (N=639)

| 균형 정확도 | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |
|--:|--:|--:|--:|--:|--:|
| 64.4% | 84.7% | 94.5% | 84.8% | 13.7% | N/A |

| 실제 정답 / 모델 예측 | normal | heel | outsole | both | 합계 |
|---|--:|--:|--:|--:|--:|
| normal | 344 | 12 | 8 | 0 | 364 |
| heel | 31 | 190 | 2 | 1 | 224 |
| outsole | 43 | 1 | 7 | 0 | 51 |
| both | 0 | 0 | 0 | 0 | 0 |

#### val25 (N=680)

| 균형 정확도 | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |
|--:|--:|--:|--:|--:|--:|
| 62.1% | 78.7% | 85.8% | 75.8% | 24.6% | N/A |

| 실제 정답 / 모델 예측 | normal | heel | outsole | both | 합계 |
|---|--:|--:|--:|--:|--:|
| normal | 471 | 56 | 21 | 1 | 549 |
| heel | 13 | 47 | 1 | 1 | 62 |
| outsole | 47 | 5 | 17 | 0 | 69 |
| both | 0 | 0 | 0 | 0 | 0 |

### B26 · YOLO26m 1024 (`yolo26m_1024_s0`, conf 0.25)

#### v5 val (N=639)

| 균형 정확도 | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |
|--:|--:|--:|--:|--:|--:|
| 66.0% | 85.1% | 93.7% | 86.6% | 17.6% | N/A |

| 실제 정답 / 모델 예측 | normal | heel | outsole | both | 합계 |
|---|--:|--:|--:|--:|--:|
| normal | 341 | 12 | 10 | 1 | 364 |
| heel | 25 | 194 | 3 | 2 | 224 |
| outsole | 41 | 1 | 9 | 0 | 51 |
| both | 0 | 0 | 0 | 0 | 0 |

#### val25 (N=680)

| 균형 정확도 | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |
|--:|--:|--:|--:|--:|--:|
| 61.6% | 77.1% | 83.8% | 82.3% | 18.8% | N/A |

| 실제 정답 / 모델 예측 | normal | heel | outsole | both | 합계 |
|---|--:|--:|--:|--:|--:|
| normal | 460 | 61 | 27 | 1 | 549 |
| heel | 11 | 51 | 0 | 0 | 62 |
| outsole | 51 | 3 | 13 | 2 | 69 |
| both | 0 | 0 | 0 | 0 | 0 |

