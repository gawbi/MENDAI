# Detector · YOLO

**담당:** 보인

신발 사진에서 손상된 자리를 찾는 YOLO 탐지기 실험입니다. 모델은 Ultralytics 구현을 그대로 씁니다. 탐지 결과를 사진 한 장의 판정(normal / heel / outsole / both)으로 바꿔 균형 정확도로 채점합니다. 전체 결과와 혼동 행렬은 [RESULTS.md](RESULTS.md)에 있습니다.

## 결과 요약 — 기본 학습 2회

데이터셋 v5 (2026-10-17 확정), 입력 1024, 50 epoch, seed 0, `best.pt`, 사진 판정 conf 0.25 (두 클래스 공통, 미리 고정). v5 val과 val25는 합치지 않고 따로 잽니다.

| 평가셋 | 실험 | 모델 | 균형 정확도 | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |
|------|------|------|--:|--:|--:|--:|--:|--:|
| v5 val (639장) | B11 | YOLO11m · 50 · 1024 | **64.4%** | 84.7% | 94.5% | 84.8% | 13.7% | N/A |
| v5 val (639장) | B26 | YOLO26m · 50 · 1024 | **66.0%** | 85.1% | 93.7% | 86.6% | 17.6% | N/A |
| val25 (680장) | B11 | YOLO11m · 50 · 1024 | **62.1%** | 78.7% | 85.8% | 75.8% | 24.6% | N/A |
| val25 (680장) | B26 | YOLO26m · 50 · 1024 | **61.6%** | 77.1% | 83.8% | 82.3% | 18.8% | N/A |

> [!IMPORTANT]
> - 균형 정확도 = (Recall normal + Recall heel + Recall outsole) / 3. both 정답은 0장이라 Recall both는 N/A이고 평균에 넣지 않습니다. both 예측 열은 유지하며, 정상·heel·outsole 사진을 both로 판정하면 오답입니다.
> - conf 0.25는 Ultralytics 기본값으로 미리 정한 값이고 튜닝하지 않았습니다. DETR 결과는 클래스별 conf를 v5 val + val25 합집합에서 골라 같은 셋으로 잰 값이라 **이 표와 바로 비교하지 않습니다.**
> - test(v5 test, test25)는 아직 채점하지 않았습니다. 모델 1회 학습 결과이므로 두 모델의 차이(1~2%p)는 우열의 근거로 보기 어렵습니다.

## 추가 실험 요약 (선정 모델 B26)

두 검증셋 균형 정확도의 평균으로 B26(YOLO26m)을 골라 추가 실험을 했습니다. 자세한 표 · 혼동 행렬은 [RESULTS.md](RESULTS.md#추가-실험-선정-모델-b26), 진행 상황 · 남은 계획 · 재개 방법은 [PROGRESS.md](PROGRESS.md)에 있습니다.

| ID | 실험 | v5 val 균형 정확도 | val25 균형 정확도 | 상태 |
|------|------|--:|--:|------|
| B26 | 기본선 (conf 0.25) | 66.0% | 61.6% | 완료 |
| A1 | mosaic 끔 | 65.9% (−0.1%p) | 62.7% (+1.0%p) | 완료 |
| A2 | scale 0.5 → 0.25 | 70.5% (+4.5%p) | 66.5% (+4.8%p) | 완료 |
| A3 | lr0 0.001 → 0.0005 | 75.9% (+10.0%p) | **76.8% (+15.2%p)** | 완료 · val25 최고 |
| A4 | lr0 0.001 → 0.00025 | **82.0% (+16.1%p)** | 72.7% (+11.0%p) | 완료 · v5 val 최고 (전체 정확도 88.7%) |
| T1 | conf 튜닝 (outsole 0.05 / heel 0.25, 재학습 없음) | 75.6% (+9.6%p) | 76.9% (+15.3%p) | 완료 · 낙관적 수치 |
| T2 | SAHI 타일 추론 (재학습 없음) | 64.5% (−1.5%p) | 59.7% (−1.9%p) | 완료 · 채택 안 함 (처리 시간 약 10배) |
| T3 | NMS-free (재학습 없음) | 66.0% (0.0%p) | 58.8% (−2.9%p) | 완료 · 채택 안 함 |
| T4 | TTA (재학습 없음) | 70.7% (+4.7%p) | 66.7% (+5.1%p) | 완료 |
| C1 | A3 + conf 튜닝 (outsole 0.15 / heel 0.30) | 77.4% (+11.5%p) | 80.5% (+18.9%p) | 완료 · 낙관적 수치 |
| A3+T3 | A3 + NMS-free | 71.3% (+5.3%p) | 66.1% (+4.5%p) | 완료 · 채택 안 함 |
| A3+T4 | A3 + TTA | 76.7% (+10.7%p) | 78.9% (+17.3%p) | 완료 |

## 폴더 구성

| 경로 | 역할 |
|------|------|
| `configs/*.json` | 설정 6개 (기본선 2 + A1~A4). 모델, 해상도, epoch, seed, batch, optimizer, lr0, 바꾼 항목(`overrides`), 사진 판정 conf, 평가셋 |
| `train_yolo.py` | config 하나로 학습. `--resume`이면 `last.pt`에서 이어서 학습 |
| `eval_photo_level.py` | `best.pt`로 평가셋을 예측해 사진 단위 판정으로 바꾸고 4×4 혼동 행렬 · 균형 정확도 · 전체 정확도 · 클래스별 재현율 계산. `--nms-free`(T3), `--augment`(TTA, T4), `--out-name`(재학습 없는 변형의 결과 폴더) |
| `tune_conf.py` | T1. 저장된 사진별 점수로 클래스별 conf 탐색 (재학습 · 재예측 없음) |
| `sahi_eval.py` | T2. 같은 가중치로 일반 추론과 SAHI 타일 추론 비교, 사진당 처리 시간 기록 |
| `make_tables.py` | 저장된 `metrics.json`으로 결과 표 생성. 결과가 없는 실험은 미완료로 표시 |
| `results/predictions/<config>/` | 완료된 run의 v5 val · val25 사진별 클래스 최고 점수 (`image_id, max_outsole, max_heel_lining`). 사진 · 정답 라벨 없음 |
| `tests/` | 평가 · conf 탐색 검사 (`python -m unittest discover -s detector/yolo/tests`) |
| `requirements.txt` | 가상환경 전체 패키지 버전 |
| `RESULTS.md` | 결과 표, 혼동 행렬, 박스 지표, 관찰 |
| `PROGRESS.md` | 완료 · 미완료 실험, 남은 계획, 재개 방법 |

가중치, 학습 로그, 예측 원본은 레포 밖(`--runs-root`)에 생기고 커밋하지 않습니다. 커밋한 사진별 점수 파일에는 사진이나 정답 라벨이 없습니다.

## 체크포인트

가중치는 깃허브에 올리지 않고 **팀 구글 드라이브(데이터셋을 공유한 드라이브)의 `yolo_weights_v5` 폴더**에 둡니다. DETR과 같은 이유입니다: 이 레포는 공개라서 접수증이 섞인 데이터로 학습한 모델을 공개하지 않습니다. 받은 파일은 아래 SHA-256으로 확인합니다 (`sha256sum <파일>`).

| 실험 | config | 파일 (`yolo_weights_v5/`) | 크기 | SHA-256 |
|------|------|------|--:|------|
| B11 | `yolo11m_1024_s0` | `yolo11m_1024_s0/best.pt` | 39 MB | `9152432eda7f79057fcafb7fcba3035d64aee2ddc3b37df4c7b00942e09dcb76` |
| B26 | `yolo26m_1024_s0` | `yolo26m_1024_s0/best.pt` | 43 MB | `50471a2cd7202334637478ecf1e5350af42cca1149d84758af1e418b79ec4096` |
| A1 | `yolo26m_1024_s0_mosaic0` | `yolo26m_1024_s0_mosaic0/best.pt` | 43 MB | `0763b423e3aac9ebe91520112405feafde80fdb034b2b185702fd1e6bd8accd9` |
| A2 | `yolo26m_1024_s0_scale025` | `yolo26m_1024_s0_scale025/best.pt` | 43 MB | `9bdd3041478065bebd2a43424cce61255d7f298b19455be9e052401dd966cfa3` |
| A3 | `yolo26m_1024_s0_lr5e-4` | `yolo26m_1024_s0_lr5e-4/best.pt` | 43 MB | `ca49a9de1926f5daef0c9ab7bb77d80f13a195eed5fa0068048c120602259918` |
| A4 | `yolo26m_1024_s0_lr2.5e-4` | `yolo26m_1024_s0_lr2.5e-4/best.pt` | 43 MB | `098dda3756485dba31e2f2511f70b26ec52942b8b6018e77623cf1a682220a2d` |

> [!NOTE]
> 드라이브 업로드는 보인이 직접 합니다. 업로드 전에는 보인 노트북 `~/aioncampus/runs/`에만 있습니다.

학습이 끝나면 Ultralytics가 `best.pt` · `last.pt`에서 optimizer 상태를 지워 40 MB로 줄입니다. 추론 · 평가에는 `best.pt`를 씁니다. 학습 도중에는 `last.pt`(약 200 MB)에 optimizer 상태가 있어 `--resume`으로 이어 학습할 수 있습니다.

## 환경

| 항목 | 버전 |
|------|------|
| GPU | RTX 5070 Laptop 8 GB, NVIDIA 드라이버 595.99.02 |
| OS · Python | Ubuntu 24.04.5, Python 3.12.3 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| ultralytics | 8.4.182 |

RTX 50 시리즈(Blackwell)는 cu128 이상의 torch가 필요합니다. torch를 CUDA 인덱스에서 **먼저** 설치합니다.

```bash
python3 -m venv ~/envs/yolo
~/envs/yolo/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
~/envs/yolo/bin/pip install ultralytics==8.4.182
```

스크립트는 Ultralytics 사용 통계 전송을 끕니다 (`sync=False`).

## 데이터와 평가셋

팀 드라이브의 `mendi_v5_1024.zip`을 풀면 생기는 `mendi_damage_v5_20261017/`을 `--data-root`로 넘깁니다. 데이터는 그대로 씁니다 (split · 라벨 · 클래스 수정 없음).

| 평가셋 | 이미지 | 정답 | 사진 | normal / heel / outsole |
|------|------|------|--:|--:|
| v5 val | `dataset_damage_v5_1024/images/val` | `damage_image_labels_v5.csv`에서 `split == val` | 639 | 364 / 224 / 51 |
| val25 | `dataset_damage_eval25_1024/images/val` | `damage_image_labels_eval25.csv`에서 `split == val` | 680 | 549 / 62 / 69 |

사진 정답은 CSV의 `label` 열입니다. val25의 라벨 txt는 손상 사진도 비어 있으므로 정답으로 쓰지 않습니다. test 셋(`v5_test`, `test25`)은 `--final`을 줄 때만 채점합니다.

## 재현

```bash
# 레포 루트에서
DATA=~/data/mendi_damage_v5_20261017
RUNS=~/runs
PY=~/envs/yolo/bin/python

# 학습 (B11, B26)
$PY detector/yolo/train_yolo.py --config detector/yolo/configs/yolo11m_1024_s0.json --data-root $DATA --runs-root $RUNS
$PY detector/yolo/train_yolo.py --config detector/yolo/configs/yolo26m_1024_s0.json --data-root $DATA --runs-root $RUNS

# 중간에 멈추면 같은 명령에 --resume
$PY detector/yolo/train_yolo.py --config detector/yolo/configs/yolo11m_1024_s0.json --data-root $DATA --runs-root $RUNS --resume

# 사진 단위 평가 (v5 val, val25)
$PY detector/yolo/eval_photo_level.py --config detector/yolo/configs/yolo11m_1024_s0.json --data-root $DATA --runs-root $RUNS
$PY detector/yolo/eval_photo_level.py --config detector/yolo/configs/yolo26m_1024_s0.json --data-root $DATA --runs-root $RUNS
```

| config | 학습 시간 (RTX 5070 Laptop, 전원 연결) | 최고 VRAM |
|------|--:|--:|
| `yolo11m_1024_s0` | 약 2.9시간 | 5.3 GB |
| `yolo26m_1024_s0` | 2.8시간 | 6.2 GB |

- 결과는 `<runs-root>/yolo/<name>/`에 생깁니다: `weights/best.pt · last.pt · epoch{0,10,20,30,40}.pt`, `results.csv`, `args.yaml`, `env_record.json`, `photo_eval/<평가셋>/`(`metrics.json`, `confusion_matrix.md`, `per_image.csv`), 그리고 `<runs-root>/yolo/photo_eval_summary.md`.
- `per_image.csv`에는 사진마다 클래스별 최고 confidence가 들어 있어 다른 conf 기준을 모델 재실행 없이 다시 적용할 수 있습니다.
- 다른 가중치로 평가하려면 `--weights <경로>`를 줍니다. 드라이브의 `best.pt`를 `<runs-root>/yolo/<name>/weights/`에 두면 학습 없이 평가만 다시 할 수 있습니다.
- 추가 실험 재현 (A1~A3는 같은 학습 명령에 config만 바꿈):

```bash
$PY detector/yolo/train_yolo.py --config detector/yolo/configs/yolo26m_1024_s0_scale025.json --data-root $DATA --runs-root $RUNS
$PY detector/yolo/eval_photo_level.py --config detector/yolo/configs/yolo26m_1024_s0_scale025.json --data-root $DATA --runs-root $RUNS
# T1: 기본선 평가 결과(per_image.csv)가 있어야 함
$PY detector/yolo/tune_conf.py --config detector/yolo/configs/yolo26m_1024_s0.json --runs-root $RUNS
# T3 · T4 (재학습 없음, 같은 best.pt)
$PY detector/yolo/eval_photo_level.py --config detector/yolo/configs/yolo26m_1024_s0.json --data-root $DATA --runs-root $RUNS --nms-free --out-name yolo26m_1024_s0_e2e --exp-id T3
$PY detector/yolo/eval_photo_level.py --config detector/yolo/configs/yolo26m_1024_s0.json --data-root $DATA --runs-root $RUNS --augment --out-name yolo26m_1024_s0_tta --exp-id T4
# C1: A3 평가 결과가 있어야 함
$PY detector/yolo/tune_conf.py --config detector/yolo/configs/yolo26m_1024_s0_lr5e-4.json --runs-root $RUNS --exp-id C1
# T2: pip install sahi==0.12.8 필요
$PY detector/yolo/sahi_eval.py --config detector/yolo/configs/yolo26m_1024_s0.json --data-root $DATA --runs-root $RUNS
$PY detector/yolo/make_tables.py --runs-root $RUNS --out yolo_experiments.md
```

- 노트북은 반드시 전원을 연결합니다. 배터리로 돌리면 GPU가 35 W로 제한되어 약 2배 느려지고, 배터리가 떨어지면 학습이 멈춥니다.
