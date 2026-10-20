# Detector · DETR

**담당:** 광빈

신발 사진에서 손상된 자리를 찾는 탐지기 실험입니다. 모델은 직접 짜지 않고 라이브러리 구현을 그대로 씁니다 (Ultralytics RT-DETR, Roboflow `rfdetr`). 탐지 결과를 사진 한 장의 판정(normal / heel / outsole / both)으로 바꿔 팀 공통 지표인 균형 정확도로 채점합니다. 전체 결과와 혼동 행렬은 [RESULTS.md](RESULTS.md)에 있습니다.

## 결과 요약

데이터셋 v5 (2026-10-17 확정), seed 0, 30 epoch. 사진 단위 수치는 v5 val + val25 합집합(1,319장)입니다.

| 설정 (config) | 모델 | 해상도 | conf (outsole / heel) | 균형 정확도 | 전체 정확도 | v5 val 균형 | val25 균형 | box mAP50-95 (v5 val) |
|------|------|--:|--:|--:|--:|--:|--:|--:|
| `rtdetr-l_640_s0` | RT-DETR-l | 640 | 0.45 / 0.55 | 79.35% | 80.14% | 81.3% | 76.2% | 0.182 |
| `rfdetr-s_640_s0` | RF-DETR-S | 640 | 0.20 / 0.45 | 80.84% | 79.23% | 82.9% | 76.8% | 0.218 |
| `rfdetr-s-lora_640_s0` | RF-DETR-S + LoRA | 640 | 0.20 / 0.35 | 81.08% | 76.72% | 84.4% | 76.8% | 0.203 |
| `rtdetr-l_1024_s0` | RT-DETR-l | 1024 | 0.35 / 0.55 | 80.56% | 76.95% | 85.6% | 75.7% | 0.199 |
| **`rfdetr-s_1024_s0`** (대표) | RF-DETR-S | 1024 | 0.15 / 0.25 | **82.71%** | 81.27% | 84.7% | 79.1% | **0.248** |
| **`rfdetr-s-lora_1024_s0`** (대표) | RF-DETR-S + LoRA | 1024 | 0.20 / 0.30 | 82.50% | **84.84%** | 83.7% | 80.1% | 0.218 |

> [!IMPORTANT]
> - conf는 v5 val + val25 합집합에서 균형 정확도가 가장 높은 값으로 골랐고, 위 사진 단위 수치도 같은 셋에서 잰 값입니다. 기준을 고른 셋으로 다시 채점했으므로 **수치가 낙관적일 수 있습니다.** test(v5 test, test25)는 아직 채점하지 않았습니다.
> - 모델 간 차이는 대부분 노이즈 범위입니다. 균형 정확도의 95% 신뢰구간은 run마다 ±2.7~3.0%p입니다 (사진 단위 bootstrap). 두 run 차이의 신뢰구간이 0을 벗어나는 경우는 RF-DETR-S 1024와 RT-DETR-l 640 사이뿐입니다.
> - 1536은 run 하나에 15~21시간이 걸려 돌리지 않았습니다.

## 폴더 구성

| 경로 | 역할 |
|------|------|
| `configs/*.json` | 설정 6개. 모델, 해상도, epoch, seed, batch, 평가셋 목록. 데이터 · 출력 경로는 넣지 않습니다 |
| `tools/run_experiment.py` | config 하나로 학습 → 평가셋 예측 → 사진 단위 평가를 차례로 실행. 경로는 인자로 받습니다 |
| `train_rtdetr.py` / `train_rfdetr.py` | 라이브러리 학습 호출. 끝나면 best 체크포인트의 v5 val 박스 지표를 `box_metrics_val.json`에 기록 |
| `predict_rtdetr.py` / `predict_rfdetr.py` | 이미지 폴더 하나를 예측해 공통 예측 CSV(`file, class_id, class_name, score, x1, y1, x2, y2`)로 저장 |
| `eval_image_level.py` | 예측 CSV를 사진 단위 판정으로 바꿔 conf를 고르고, 균형 정확도 · 혼동 행렬을 계산 |
| `common.py` | 클래스 이름, 예측 CSV 형식, 실행 중 절전 방지 |
| `tests/` | 평가 스크립트 검사 (`python -m unittest discover -s detector/detr/tests`) |
| `requirements-rtdetr.txt` / `requirements-rfdetr.txt` | 두 가상환경의 전체 패키지 버전 |
| `RESULTS.md` | 결과 표, run별 혼동 행렬, 관찰 |
| `results/predictions/<config>/` | 6개 설정의 v5 val · val25 예측 (박스 CSV gzip + 사진별 요약표). [아래](#예측-파일) 참고 |
| `tools/summarize_predictions.py` | 예측 CSV를 사진별 요약표(클래스별 최고 점수, 박스 수)로 바꿈 |

데이터, 가중치, 로그, 시각화 이미지는 레포 밖(`--runs-root`)에 생기고 커밋하지 않습니다. 커밋한 예측 파일에는 사진이나 정답 라벨이 없습니다.

## 환경

| 항목 | 버전 |
|------|------|
| GPU | RTX 4060 Laptop 8 GB, NVIDIA 드라이버 546.32 (CUDA 12.3까지 지원) |
| OS · Python | Windows 11, Python 3.12.15 |
| torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 (두 환경 공통) |
| RT-DETR 환경 | ultralytics 8.4.178 |
| RF-DETR 환경 | rfdetr 1.11.2 (`[train]`), peft 0.21.2, transformers 5.20.1, supervision 0.31.0 |

두 라이브러리의 의존성이 달라 가상환경을 둘로 나눕니다. torch는 반드시 CUDA 인덱스에서 **먼저** 설치합니다. PyPI의 Windows torch는 CPU 전용이라 라이브러리를 먼저 깔면 GPU를 못 씁니다. 드라이버가 580 이상이면 cu126 대신 더 새 CUDA 휠을 써도 됩니다.

```powershell
$envs = "D:\envs"   # 레포 밖 아무 폴더
uv python install 3.12
uv venv $envs\rtdetr --python 3.12
uv pip install --python $envs\rtdetr\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision==0.29.1
uv pip install --python $envs\rtdetr\Scripts\python.exe ultralytics==8.4.178

uv venv $envs\rfdetr --python 3.12
uv pip install --python $envs\rfdetr\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision==0.29.1
uv pip install --python $envs\rfdetr\Scripts\python.exe "rfdetr[train]==1.11.2" "torch-hungarian==0.1.0rc0"
```

> [!WARNING]
> `import torch`가 `WinError 1114 ... c10.dll`로 실패하면 Visual C++ 재배포 패키지를 업데이트합니다: `winget install --id Microsoft.VCRedist.2015+.x64 -e`

사진이 밖으로 나가지 않도록 스크립트가 Ultralytics 사용 통계 전송을 끕니다 (`sync=False`). wandb · mlflow 같은 클라우드 로거는 설치하지 않습니다.

## 데이터와 평가셋

팀 드라이브의 `mendi_v5_640.zip`, `mendi_v5_1024.zip`을 같은 폴더에 풀면 `mendi_damage_v5_20261017/` 하나로 합쳐집니다. 이 폴더를 `--data-root`로 넘깁니다.

```
<DATA_ROOT>/
  dataset_damage_v5_640/  dataset_damage_v5_1024/       학습 · v5 val (images/, labels/, data.yaml)
  dataset_damage_eval25_640/  dataset_damage_eval25_1024/  2025 평가셋 (박스 없음)
  photo_labels/damage_image_labels_v5.csv               사진 단위 정답 (file, split, label, ...)
  photo_labels/damage_image_labels_eval25.csv
```

| 평가셋 | 이미지 | 정답 | 사진 | normal / heel / outsole |
|------|------|------|--:|--:|
| v5 val | `dataset_damage_v5_<해상도>/images/val` | `damage_image_labels_v5.csv`에서 `split == val` | 639 | 364 / 224 / 51 |
| val25 | `dataset_damage_eval25_<해상도>/images/val` | `damage_image_labels_eval25.csv`에서 `split == val` | 680 | 549 / 62 / 69 |

각 config의 `eval_sets`에 같은 경로가 적혀 있습니다. 두 셋 모두 both 사진은 0장입니다.

> [!IMPORTANT]
> - split은 다시 나누지 않습니다. 같은 신발이 split을 넘지 않게 나눠 두었습니다.
> - 빈 라벨 파일은 눈으로 확인한 정상 사진입니다. 음성(배경)으로 그대로 학습에 넣습니다.
> - 입력 크기는 복사본 크기와 같게 둡니다 (640 / 1024).
> - 일부 사진에 고객 이름 · 전화번호가 적힌 수선 접수증이 찍혀 있습니다. 데이터셋 이미지, 예측 시각화, zip은 커밋하지 않습니다.

## 재현

config 하나를 그 모델의 가상환경 python으로 실행합니다. 경로는 모두 인자로 줍니다.

```powershell
# 레포 루트에서
$data = "D:\data\mendi_damage_v5_20261017"
$runs = "D:\runs"
& $envs\rtdetr\Scripts\python.exe detector\detr\tools\run_experiment.py --config detector\detr\configs\rtdetr-l_640_s0.json --data-root $data --runs-root $runs
& $envs\rfdetr\Scripts\python.exe detector\detr\tools\run_experiment.py --config detector\detr\configs\rfdetr-s_1024_s0.json --data-root $data --runs-root $runs
```

| config | 환경 | 학습 시간 (RTX 4060 Laptop) | 최고 VRAM |
|------|------|--:|--:|
| `rtdetr-l_640_s0` | rtdetr | 2.6시간 | 7.1 GB |
| `rfdetr-s_640_s0` | rfdetr | 2.6시간 | 5.2 GB |
| `rfdetr-s-lora_640_s0` | rfdetr | 2.9시간 | 5.3 GB |
| `rtdetr-l_1024_s0` | rtdetr | 5.5시간 | 7.4 GB |
| `rfdetr-s_1024_s0` | rfdetr | 5.5시간 | 4.9 GB |
| `rfdetr-s-lora_1024_s0` | rfdetr | 6.3시간 | 4.9 GB |

- 결과는 `<runs-root>/<family>/<name>/`에 생깁니다: `box_metrics_val.json`, `pred_v5_val.csv`, `pred_val25.csv`, `image_eval/image_eval.md`(퍼센트 요약표 + 혼동 행렬), `run_<step>.log`.
- 이미 끝난 단계는 건너뜁니다. 같은 명령을 다시 실행하면 멈춘 곳부터 이어서 돌고, 학습은 `last.pt` / `last.ckpt`에서 이어집니다.
- `--dry-run`은 실행할 명령만 보여 줍니다. `--weights-dir`는 사전학습 가중치를 받을 폴더입니다 (기본 `<runs-root>/pretrained`).
- 가중치가 있으면 학습 없이 예측과 평가만 다시 할 수 있습니다. `best.pt`(RT-DETR) 또는 `checkpoint_best_total.pth`(RF-DETR)와 `box_metrics_val.json`을 `<runs-root>/<family>/<name>/`에 두고 같은 명령을 실행합니다.
- `--tta`는 좌우반전 예측을 더해 점수를 평균하는 선택 기능입니다. 위 결과에는 쓰지 않았습니다.

단계별로 직접 돌릴 때는 각 스크립트의 `--help`를 봅니다. 예: `python detector\detr\eval_image_level.py --help`.

마지막 test 채점은 val에서 고른 conf를 고정해 한 번만 합니다 (아직 안 함).

```powershell
python detector\detr\predict_rfdetr.py --weights <run>\checkpoint_best_total.pth --size small --resolution 1024 --images $data\dataset_damage_v5_1024\images\test --out <run>\pred_v5_test.csv
python detector\detr\eval_image_level.py --out <run>\image_eval_test --fixed-conf 0.15 0.25 --set v5_test <run>\pred_v5_test.csv $data\photo_labels\damage_image_labels_v5.csv test
```

## 설정

epoch · 입력 크기 · seed · batch 외에는 라이브러리 기본값입니다.

| 항목 | RT-DETR-l | RF-DETR-S (+ LoRA) |
|------|------|------|
| 사전학습 | `rtdetr-l.pt` (COCO) | RF-DETR Small (COCO) |
| batch | 640: 8, 1024: 4. Ultralytics가 64장까지 자동 누적 | 640: 4 x 4, 1024: 2 x 8 (`batch_size x grad_accum = 16`) |
| 정밀도 · optimizer | `amp=bf16`, `optimizer=AdamW`, `lr0=1e-4`, `deterministic=False` | rfdetr 기본 (bf16 AMP, AdamW lr 1e-4, EMA, multi-scale) |
| LoRA | - | `backbone_lora=True`: DINOv2 인코더에 DoRA rank 16. rank와 적용 레이어는 라이브러리가 고정 |
| best 체크포인트 | v5 val mAP50-95가 가장 높은 epoch (`best.pt`) | v5 val mAP50-95가 가장 높은 epoch, EMA (`checkpoint_best_total.pth`) |

## 가중치

가중치는 깃허브에 올리지 않고 **팀 구글 드라이브(데이터셋을 공유한 드라이브)의 `detr_weights_v5` 폴더**에 둡니다. 이유는 두 가지입니다: RF-DETR 파일이 깃허브 한도(100 MB)를 넘고, 이 레포는 공개라서 접수증이 섞인 데이터로 학습한 모델을 공개하지 않기 위해서입니다. 받은 파일은 아래 SHA-256으로 확인합니다.

| config | 파일 (`detr_weights_v5/`) | 크기 | SHA-256 |
|------|------|--:|------|
| `rtdetr-l_640_s0` | `rtdetr-l_640_s0/best.pt` | 63 MB | `899e1ffa0d5fcdf627f887ab0401a843e51b3a5bc9a3c761982b6c7d9eec6e69` |
| `rfdetr-s_640_s0` | `rfdetr-s_640_s0/checkpoint_best_total.pth` | 122 MB | `104c63bb8f80fac87a8e3a5d25d7efff937b30ac58083f665aa714c51faaffa7` |
| `rfdetr-s-lora_640_s0` | `rfdetr-s-lora_640_s0/checkpoint_best_total.pth` | 124 MB | `7fd0609bd85628d30b0dc61e53a2bdc5c3dabe01d368b9a8ce13e80a59ea09ea` |
| `rtdetr-l_1024_s0` | `rtdetr-l_1024_s0/best.pt` | 63 MB | `ab165d9817f7a32881f3d93c6fe51279689d1c4b610b65cea66994546ad52891` |
| `rfdetr-s_1024_s0` (대표) | `rfdetr-s_1024_s0/checkpoint_best_total.pth` | 126 MB | `e16eabfc0b3007d36d977214d789b1aa1e6f259996057e35d11b6e446ab60d06` |
| `rfdetr-s-lora_1024_s0` (대표) | `rfdetr-s-lora_1024_s0/checkpoint_best_total.pth` | 128 MB | `bfe2d85f99dba8a22c60709df81a35a93421e13e4f4dbaaf2597d16592a077e8` |

Windows에서 확인: `Get-FileHash <파일> -Algorithm SHA256`

## 예측 파일

`results/predictions/<config>/`에 6개 설정의 예측을 그대로 올렸습니다. 위 가중치로 v5 val(639장)과 val25(680장)를 예측한 결과이고, RESULTS.md의 수치가 이 파일로 재현됩니다. test · test25 예측은 아직 만들지 않았습니다.

| 파일 | 내용 |
|------|------|
| `<subset>_boxes.csv.gz` | 박스 단위 원본 (`file, class_id, class_name, score, x1, y1, x2, y2`). 사진마다 score 0.001 이상 박스 최대 300개, 좌표는 입력 이미지(640 또는 1024) 픽셀. gzip 압축 |
| `<subset>_photo_scores.csv` | 사진별 요약 (`image_id, max_outsole, max_heel_lining, n_<class>_ge0.3, n_<class>_ge0.5`). 박스가 없는 사진은 0. `tools/summarize_predictions.py`로 만듦 |

- `<subset>`은 `v5_val` 또는 `val25`입니다. 정답은 데이터셋의 `photo_labels/`에 있습니다 (`damage_image_labels_v5.csv` / `damage_image_labels_eval25.csv`의 `split == val`).
- 사진 단위 판정은 `max_outsole >= t_out`이면 outsole, `max_heel_lining >= t_heel`이면 heel입니다. 설정별 conf는 위 결과 요약 표에 있습니다.
- 다른 모델과 결합할 때도 conf나 결합 방식은 v5 val로만 정하고, val25는 확인용으로 둡니다.

## 평가 지표 (사진 단위)

사진마다 클래스별 최고 score를 두 기준(`t_out`, `t_heel`)과 비교합니다.

| 조건 | 판정 |
|------|------|
| outsole 최고 score >= `t_out`, heel_lining 최고 score >= `t_heel` | both |
| outsole만 기준 이상 | outsole |
| heel_lining만 기준 이상 | heel |
| 둘 다 미만 (예측 박스가 없는 사진 포함) | normal |

- 4x4 혼동 행렬: 행 = 정답, 열 = 예측, 순서 normal / heel / outsole / both
- 균형 정확도 (핵심 지표) = 사진이 있는 클래스의 재현율 평균. both는 val · test에 0장이라 빠집니다
- 전체 정확도 = 대각선 합 / 전체 사진 수, 클래스별 재현율 = 대각선 / 행 합
- conf는 0.05 간격 격자에서 고릅니다 (동점이면 전체 정확도가 높은 쪽, 그다음 더 높은 기준)
- 박스 지표(v5 val mAP50, mAP50-95, 클래스별 AP50-95)도 기록합니다. RT-DETR은 Ultralytics `val`, RF-DETR은 rfdetr 내장 COCO 평가로 계산해서 계열 사이 비교는 참고용입니다

## 재현할 때 주의할 점

- RT-DETR을 Ultralytics 기본값(`optimizer=auto` → AdamW lr 0.00167, fp16)으로 돌리면 640에서 epoch 4~6에 loss가 NaN이 됩니다. 그래서 config에 `bf16`, `AdamW`, `lr0=1e-4`를 넣었습니다.
- `rtdetr-l_1024_s0`은 epoch 22에서 CUDA out of memory로 멈춘 뒤 `last.pt`에서 이어 학습했습니다. 끊김 없이 돌린 것과 완전히 같지는 않습니다. 이 run의 `box_metrics_val.json`은 `model`이 `last.pt`로, `train_minutes`가 이어 학습한 시간만으로 적혀 있습니다. 다시 돌릴 때 OOM이 나면 `train_rtdetr.py --resume <last.pt> --resume-batch 2`로 이어 갑니다.
- seed를 고정해도 GPU 연산이 결정적이지 않아(RT-DETR은 `deterministic=False`) 다시 돌리면 수치가 조금 달라집니다.
- rfdetr `metrics.csv`의 epoch는 0부터 셉니다. RESULTS.md의 best epoch는 1을 더한 값입니다.
- Windows에서 몇 시간짜리 학습 중에 절전으로 들어가면 학습이 멈춥니다. 스크립트가 실행 중에는 절전을 막지만, 노트북 덮개를 닫으면 막지 못합니다.
- 이 노트북에서는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 설정하면 CUDA 초기화가 멈췄습니다. 실행기는 이 변수를 지우고 하위 프로세스를 띄웁니다.
- 2025 평가셋은 1536 복사본이 없습니다. 1536을 돌리려면 val25를 같은 크기로 채점할 방법을 먼저 정해야 합니다.
