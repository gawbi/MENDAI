<div align="center">

# 🛠️ MEND AI

**분류기(DINO) · 탐지기(DETR / YOLO) 모델 실험 & 비교**

![Status](https://img.shields.io/badge/status-in%20progress-orange)
![Classifier](https://img.shields.io/badge/classifier-DINO-green)
![Detector](https://img.shields.io/badge/detector-DETR%20%7C%20YOLO-blue)
![Fine-tuning](https://img.shields.io/badge/fine--tuning-LoRA-purple)

</div>

---

## 📌 Overview

새 데이터셋으로 **분류기**와 **탐지기**를 나눠 실험하고, 학습 설정과 결과를 정리하는 레포입니다.

- **분류기:** DINO
- **탐지기:** DETR 계열(RF-DETR, RT-DETR), YOLO

## 🧑‍🤝‍🧑 Team

| 파트 | 모델 | 담당 |
|------|------|:--:|
| 🧠 분류기 | DINO | 재구 |
| 🎯 탐지기 | DETR | 광빈 |
| 🎯 탐지기 | YOLO | 보인 |

## 🤝 협업 규칙

충돌을 줄이고 변경 사항을 한곳에서 확인할 수 있도록 아래처럼 진행합니다.

**폴더 구조**

```
classifier/dino/   🧠 DINO 분류기 (재구)
detector/detr/     🎯 DETR 탐지기 (광빈)
detector/yolo/     🎯 YOLO 탐지기 (보인)
```

**작업 순서**

1. 작업 전에 `git pull origin main`으로 최신 내용 받기
2. 내 브랜치 만들기: `dino/<작업명>`, `detr/<작업명>`, `yolo/<작업명>`
3. 되도록 **내 파트 폴더 안에서만** 수정
4. push 후 `main`으로 **Pull Request** 열기 → 템플릿에 맞춰 작업 내용 보고
5. @gawbi 승인 후 merge

> [!IMPORTANT]
> `main`에는 직접 push할 수 없고, Pull Request + @gawbi 승인으로만 반영됩니다.
> 모델 가중치·데이터셋 같은 큰 파일은 올리지 않습니다.

## 🎯 Detector · DETR

지금까지 실험해 보니 DETR 계열은 **과적합이 잘 생기는 편**이라, 과적합을 줄이는 방향으로 실험을 진행합니다.

### 🔬 Model Comparison

지금까지 결과 기준, **과적합이 적은 순서**입니다.

| 순위 | 모델 | 과적합 | 비고 |
|:--:|------|:--:|------|
| 🥇 | **RF-DETR + LoRA** | 가장 적음 | 메인 후보 |
| 🥈 | RF-DETR | 보통 | 비교 기준 |
| 🥉 | RT-DETR | 가장 많음 | 새 데이터셋에서 한 번 확인한 뒤 쓸지 결정 |

> [!NOTE]
> RT-DETR은 아직 새 데이터셋으로 학습해 보지 않았습니다.

### ⚙️ Training Setup

| 항목 | 설정 | 이유 |
|------|------|------|
| Epochs | **30** | 보통 10~20 에폭 전에 best가 나오고 그 뒤로 성능이 떨어짐 |
| Resolution | **640** 기본 → **1024** 추가 실험 | 640에서 성능이 잘 나온 설정만 1024로 다시 돌려봄 |
| Main track | RF-DETR 계열 | 과적합이 가장 적음 |

## 🗺️ Roadmap

**공통**
- [ ] 새 데이터셋 라벨링이 끝나면 공유
- [ ] 평가지표 정해서 통일

**Detector · DETR**
- [ ] RT-DETR 3-seed 실험 결과 공유 (산학 쪽에서 학습 중)
- [ ] RT-DETR을 새 데이터셋에서 확인하고 쓸지 결정
- [ ] RF-DETR / RF-DETR + LoRA 실험 @ 640
- [ ] 결과가 좋은 설정은 1024로 다시 실험

## 👥 Contributors

<a href="https://github.com/gawbi">
  <img src="https://avatars.githubusercontent.com/u/70633886?v=4" width="72" alt="gawbi" />
</a>

## 🏗️ Built by

**[@gawbi](https://github.com/gawbi)**
