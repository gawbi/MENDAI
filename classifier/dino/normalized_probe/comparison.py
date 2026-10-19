import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import core as n

p = n.p


def read_probs(path, rows):
    with path.open(encoding="utf-8", newline="") as f:
        values = list(csv.DictReader(f))
    if [r["file"] for r in values] != [r["file"] for r in rows]:
        raise ValueError("Comparison prediction order mismatch")
    return np.array([[float(r["p_heel_lining"]), float(r["p_outsole"])] for r in values])


def compare(runs, output):
    output.relative_to(n.ROOT / "runs/dino")
    output.mkdir(parents=True, exist_ok=True)
    protocols = [n.load(run) for run in runs]
    if protocols[0]["identity"]["data_sha256"] != protocols[1]["identity"]["data_sha256"]:
        raise ValueError("Pair datasets differ")
    rows = protocols[0]["parts"]["val"] + protocols[0]["parts"]["val25"]
    y = p.labels(rows)
    groups = np.array([f"{r['family']}::{r['group']}" for r in rows])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups))
    results, group_results = {}, {}
    lines = ["# DINOv2/v3 LoRA 고정 후 분류층 정규화 비교", "", f"완료: {p.now()}. 합산 val 1,319장. test 미사용. GPU 학습·평가 프로세스 종료 후 CPU 집계.", "",
             "| 모델·조건 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 |", "|---|---:|---:|---:|---:|---:|"]
    for run, protocol in zip(runs, protocols):
        if p.read_json(run / "training_status.json")["stage"] != "completed":
            raise ValueError("Both requested experiments must finish before comparison")
        current = p.read_json(run / "metrics.json")
        name = "DINOv2" if protocol["config"]["experiment"] == "EX-04" else "DINOv3"
        results[name], group_results[name] = current, {}
        for variant, result in current.items():
            values = [result["balanced_accuracy"], result["accuracy"], *[result["recall"][label] for label in n.ORDER[:3]]]
            lines.append(f"| {name} {variant} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
            path = run / "original_probs.csv" if variant == "original" else run / variant / "val_combined_probs.csv"
            probabilities = read_probs(path, rows)
            bits = np.zeros((len(y), 2), dtype=float)
            fold_results = []
            for index, (fit, held) in enumerate(folds):
                if set(groups[fit]) & set(groups[held]):
                    raise ValueError("Shoe group leakage")
                thresholds = n.choose_thresholds(y[fit], probabilities[fit])
                bits[held] = probabilities[held] >= np.asarray(thresholds)
                fold_results.append({"fold": index, "fit_images": len(fit), "held_images": len(held), "thresholds": thresholds})
            pooled = n.metrics(y, bits)
            pooled["thresholds"] = None
            pooled["threshold_policy"] = "chosen on other shoe groups separately for each fold"
            group_results[name][variant] = {"metrics": pooled, "folds": fold_results}
        p.status(run, "completed", gpu_released=True, gpu_release_pending=False)
    lines += ["", "## 정규화 효과", ""]
    for name, current in results.items():
        raw, standard, original = current["raw_control"], current["standardized"], current["original"]
        delta = 100*(standard["balanced_accuracy"]-raw["balanced_accuracy"])
        original_delta = 100*(standard["balanced_accuracy"]-original["balanced_accuracy"])
        lines.append(f"- {name}: 동일 조건 대조군 대비 균형 정확도 {delta:+.2f}%p, 원본 LoRA 대비 {original_delta:+.2f}%p.")
    lines += ["", "## 신발 그룹 분리 내부 검증", "", "| 모델 | 원본 LoRA | 정규화 없는 대조군 | 정규화 적용 |", "|---|---:|---:|---:|"]
    for name, current in group_results.items():
        lines.append(f"| {name} | " + " | ".join(f"{current[key]['metrics']['balanced_accuracy']:.2%}" for key in ("original", "raw_control", "standardized")) + " |")
    lines += ["", "신발 그룹 5-fold, seed 0. 각 평가 fold를 제외한 네 fold에서 판정 기준을 고른 뒤 예측을 합쳤습니다. 모델·분류층 개발에는 전체 val을 사용했으므로 독립적인 최종 평가나 전체 모델 교차검증은 아닙니다."]
    for name, current in results.items():
        for variant in ("raw_control", "standardized"):
            lines += ["", f"## {name} {variant}", "", n.render(current[variant])]
    lines += ["", "정규화는 표현력을 추가하지 않으며 최적화와 weight decay의 작용을 바꿉니다. 원본 대비 변화에는 분류층 추가 학습·4개 고정 증강·백본 평가 모드·FP32 분류층이 포함됩니다. 정규화의 직접 비교는 raw_control 대 standardized입니다. 원본 가중치와 판정 기준은 자동 교체하지 않았습니다."]
    report = "\n".join(lines)+"\n"
    p.write_json(output / "comparison.json", results)
    p.write_json(output / "groupcv.json", group_results)
    p.write_json(output / "completion.json", {"time": p.now(), "runs": [str(run) for run in runs], "gpu_released": True, "test_used": False})
    (output / "report.md").write_text(report, encoding="utf-8")
    document = n.ROOT / "classifier/dino/EXPERIMENTS.md"
    text = document.read_text(encoding="utf-8")
    start, end = "<!-- NORMALIZED-PAIR-START -->", "<!-- NORMALIZED-PAIR-END -->"
    block = f"{start}\n\n## EX-04·05 정규화 비교 결과\n\n보고서: `{(output / 'report.md').relative_to(n.ROOT).as_posix()}`\n\n" + report.split("\n", 1)[1].strip().replace("\n## ", "\n### ") + f"\n\n{end}"
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    else:
        text = text.rstrip()+"\n\n"+block+"\n"
    document.write_text(text.rstrip()+"\n", encoding="utf-8")
    p.log(f"Requested pair complete; report: {output / 'report.md'}")
