<div align="center">

# 🛠️ MEND AI

**DETR 계열 객체 탐지 모델 실험 & 비교**

![Status](https://img.shields.io/badge/status-in%20progress-orange)
![Models](https://img.shields.io/badge/models-RF--DETR%20%7C%20RT--DETR-blue)
![Fine-tuning](https://img.shields.io/badge/fine--tuning-LoRA-purple)
![Resolution](https://img.shields.io/badge/resolution-640%20%7C%201024-informational)

</div>

---

## 📌 Overview

새 데이터셋에서 DETR 계열 탐지 모델(**RF-DETR**, **RT-DETR**)을 비교하고 학습 설정을 정리하는 레포입니다.
지금까지 실험해 보니 DETR 계열은 **과적합이 잘 생기는 편**이라, 과적합을 줄이는 방향으로 실험을 진행합니다.

## 🔬 Model Comparison

지금까지 결과 기준, **과적합이 적은 순서**입니다.

| 순위 | 모델 | 과적합 | 비고 |
|:--:|------|:--:|------|
| 🥇 | **RF-DETR + LoRA** | 가장 적음 | 메인 후보 |
| 🥈 | RF-DETR | 보통 | 비교 기준 |
| 🥉 | RT-DETR | 가장 많음 | 새 데이터셋에서 한 번 확인한 뒤 쓸지 결정 |

> [!NOTE]
> RT-DETR은 아직 새 데이터셋으로 학습해 보지 않았습니다.

## ⚙️ Training Setup

| 항목 | 설정 | 이유 |
|------|------|------|
| Epochs | **30** | 보통 10~20 에폭 전에 best가 나오고 그 뒤로 성능이 떨어짐 |
| Resolution | **640** 기본 → **1024** 추가 실험 | 640에서 성능이 잘 나온 설정만 1024로 다시 돌려봄 |
| Main track | RF-DETR 계열 | 과적합이 가장 적음 |

## 🗺️ Roadmap

- [ ] RT-DETR 3-seed 실험 결과 공유 (산학 쪽에서 학습 중)
- [ ] 새 데이터셋 라벨링이 끝나면 공유
- [ ] RT-DETR을 새 데이터셋에서 확인하고 쓸지 결정
- [ ] RF-DETR / RF-DETR + LoRA 실험 @ 640
- [ ] 결과가 좋은 설정은 1024로 다시 실험
- [ ] 평가지표 정해서 통일

## 👥 Contributors

<a href="https://github.com/gawbi">
  <img src="https://avatars.githubusercontent.com/u/70633886?v=4" width="72" alt="gawbi" />
</a>

## 🏗️ Built by

**[@gawbi](https://github.com/gawbi)**
