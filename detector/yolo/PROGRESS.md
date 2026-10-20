# YOLO 진행 상황

**담당:** 보인 · **마지막 업데이트:** 2026-10-20 17:45 (KST)

결과 수치는 [RESULTS.md](RESULTS.md), 실행 방법은 [README](README.md)에 있습니다. 이 문서는 무엇이 끝났고 무엇이 남았는지, 멈춘 실험을 어떻게 이어 가는지만 적습니다.

## 현재 상태

**모든 실험이 멈춰 있습니다.** 2026-10-20 17:42에 A3 학습을 일시 정지했고, T2(SAHI)는 시작하지 않았습니다. 보인이 재개를 지시할 때까지 새 실험을 시작하지 않습니다.

| ID | 실험 (config) | 상태 | 결과 위치 (`<runs-root>/yolo/<config>/`) |
|------|------|------|------|
| B11 | YOLO11m 기본선 (`yolo11m_1024_s0`) | ✅ 완료 (50 epoch, 평가 완료) | `weights/`, `results.csv`, `photo_eval/` |
| B26 | YOLO26m 기본선 (`yolo26m_1024_s0`) · **선정 모델** | ✅ 완료 | 같음 |
| A1 | mosaic 0 (`yolo26m_1024_s0_mosaic0`) | ✅ 완료 | 같음 |
| A2 | scale 0.25 (`yolo26m_1024_s0_scale025`) | ✅ 완료 | 같음 |
| T1 | conf 튜닝 (`yolo26m_1024_s0_T1conf`) | ✅ 완료 (재학습 없음) | `choice.json`, `grid.csv`, `photo_eval/` |
| A3 | lr0 0.0005 (`yolo26m_1024_s0_lr5e-4`) | ⏸ **일시 정지: 5/50 epoch** | `weights/last.pt` (epoch 5 끝, optimizer 상태 포함) |
| T2 | SAHI (`yolo26m_1024_s0_T2sahi`) | ⏳ **시작 전** | — |

- 완료된 run은 모두 50 epoch을 다 돌았고, 학습 중 batch 자동 축소가 없었습니다.
- B11은 epoch 10 도중 배터리 방전으로 한 번 꺼져 epoch 9의 `last.pt`에서 이어 학습했습니다. A3도 epoch 6 도중 멈췄으므로 같은 방식으로 이어 갑니다 (진행 중이던 epoch 6은 처음부터 다시 돕니다).

## 남은 계획

순서대로 진행합니다. 모델 · 기법 · conf는 val(v5 val, val25)로만 고르고, test는 마지막에 한 번만 씁니다.

1. **A3 이어 학습** — `last.pt`(epoch 5)에서 50 epoch까지 (약 2.5시간), 끝나면 사진 단위 평가 (conf 0.25).
2. **T2 SAHI** — B26 `best.pt`로 일반 추론과 SAHI 타일 추론을 v5 val · val25에서 비교 (약 15분 예상). 설정은 고정: 512×512 타일, 겹침 0.2, 타일 입력 1024, 전체 사진 예측 포함, GREEDYNMM · IOS 0.5. 사진당 처리 시간도 기록.
3. **결과 표 갱신** — `make_tables.py`로 RESULTS.md 표를 다시 만들고 A3 · T2를 채웁니다.
4. **(조건부) C1 결합** — 단독으로 효과가 확인된 기법만 결합합니다. 지금까지는 A2(scale 0.25)와 T1(conf 튜닝)이 후보입니다. 예: A2 `best.pt`의 사진별 점수로 conf를 다시 고름 (재학습 없음). A3 · T2 결과를 본 뒤 정합니다. 결합 결과가 단독보다 나쁜 경우도 기록합니다.
5. **최종 평가** — 모델 · 기법 · conf를 모두 고정한 뒤 v5 test · test25를 `--final`로 한 번 채점하고, B26 기본선과 최종안을 비교합니다.

## 재개 방법

보인 노트북 (`~/aioncampus`, 가상환경 `.venv`, 데이터 `data/mendi_damage_v5_20261017`, 출력 `runs/`):

```bash
cd ~/aioncampus
./resume_remaining.sh   # A3 이어 학습 → A3 평가 → T2 SAHI → 결과 표 갱신 (절전 방지 포함)
```

다른 환경에서 같은 순서를 수동으로:

```bash
PY=<python>; DATA=<mendi_damage_v5_20261017>; RUNS=<runs-root>; Y=detector/yolo
$PY $Y/train_yolo.py --config $Y/configs/yolo26m_1024_s0_lr5e-4.json --data-root $DATA --runs-root $RUNS --resume
$PY $Y/eval_photo_level.py --config $Y/configs/yolo26m_1024_s0_lr5e-4.json --data-root $DATA --runs-root $RUNS
$PY $Y/sahi_eval.py --config $Y/configs/yolo26m_1024_s0.json --data-root $DATA --runs-root $RUNS
$PY $Y/make_tables.py --runs-root $RUNS --out yolo_experiments.md
```

- `--resume`은 `<runs-root>/yolo/yolo26m_1024_s0_lr5e-4/weights/last.pt`가 있어야 합니다. 이 파일은 보인 노트북에만 있습니다 (약 220 MB, optimizer 상태 포함).
- 노트북은 전원을 연결합니다. 배터리로는 GPU가 절반 속도로 떨어지고, 방전되면 학습이 멈춥니다.
- 학습 · SAHI는 GPU를 함께 쓰지 않도록 차례로 돌립니다 (8 GB에서 학습이 6.2 GB를 씀).

## 기록해 둘 결정

- **모델 선정 기준:** 두 검증셋 균형 정확도의 평균 (B26 63.80% vs B11 63.21%). 합집합 기준으로는 B11이 0.13%p 높아 사실상 같습니다. 계획서의 "평가셋을 합치지 않는다" 원칙에 따라 평균을 썼습니다.
- **A1~A3:** 모두 공식 `yolo26m.pt`에서 새로 시작하고 B26과 한 항목만 다릅니다 (실제 `args.yaml` 비교로 확인).
- **T1:** 같은 기준(두 셋 평균)으로 0.05 간격 격자에서 클래스별 conf를 골랐습니다. 같은 셋으로 고르고 쟀으므로 낙관적입니다.
- **YOLO26 후처리:** Ultralytics 8.4.182에서 `nms`를 지정하지 않아 one-to-many head + NMS로 평가했습니다 (B11과 같은 방식). NMS-free 모드는 평가하지 않았습니다.
- **가중치:** 팀 드라이브 `yolo_weights_v5/`에 올립니다 (README의 SHA-256 표). 업로드는 보인이 직접 합니다.
