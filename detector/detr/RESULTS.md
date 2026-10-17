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
|  |  |  |  |  |  |  |  |  |  |  |

## Photo-level

| 날짜 | 모델 | 해상도 | seed | conf (outsole / heel) | v5 val 균형 / 전체 | v5 val 재현율 N / H / O | val25 균형 / 전체 | val25 재현율 N / H / O | 메모 |
|------|------|--:|--:|--:|--:|--:|--:|--:|------|
|  |  |  |  |  |  |  |  |  |  |

## 혼동 행렬

눈여겨볼 run의 4×4 혼동 행렬을 여기에 붙입니다 (행 = 정답, 열 = 판정, 순서 normal / heel / outsole / both). 각 run의 `image_eval\image_eval.md`에서 복사합니다.

아직 없음.
