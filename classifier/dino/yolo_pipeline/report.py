"""저장한 같은 val의 점수로 단독·50:50 결합을 비교한다. GPU를 사용하지 않는다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common as c
from ..evaluate import choose_thresholds, metrics, predict_classes
from ..tools.report_combined import render


def fitted(labels, probabilities):
    return metrics(labels, probabilities, choose_thresholds(labels, probabilities))


def rank(results):
    return max(range(len(results)), key=lambda i: (round(results[i]["balanced_accuracy"], 12),
                                                  round(results[i]["accuracy"], 12), -i))


def dino_candidate(rows):
    report = c.read(c.DINO_CHAIN / "final_comparison.json")
    if report["source_chain_sha256"] != c.read(c.DINO_CHAIN / "protocol.json")["sha256"] or report["test_evaluated"]:
        raise ValueError("DINO 최종 비교 출처 오류")
    names, scores, results, sources = [], [], [], {}
    for name, evidence in report["evidence"].items():
        run = Path(evidence["run"])
        if c.sha(run / "metrics.json") != evidence["metrics_sha256"]:
            raise ValueError("DINO metrics 변경")
        if "TTA" in name:
            complete = c.read(run / "complete.json")
            if not complete.get("gpu_released") or complete["data_sha256"] != c.DATA_SHA:
                raise ValueError("DINO TTA 검증·GPU 종료 미완료")
            for file, expected in complete["output_files_sha256"].items():
                if c.sha(run / file) != expected:
                    raise ValueError("DINO TTA 출력 변경")
        else:
            if c.read(run / "training_status.json")["status"] != "complete":
                raise ValueError("DINO 최종 val 미완료")
        p = c.read_scores(run / "val_combined_probs.csv", rows)
        y = [c.ORDER.index(r["label"]) for r in rows]
        result = metrics(y, p, report["results"][name]["thresholds"])
        if result != report["results"][name]:
            raise ValueError("DINO 점수 CSV·최종 비교 불일치")
        names.append(name)
        scores.append(p)
        results.append(result)
        sources[name] = {"run": str(run), "scores_sha256": c.sha(run / "val_combined_probs.csv")}
    if len(names) != 4:
        raise ValueError("DINO 네 결과가 모두 필요")
    index = rank(results)
    return names[index], scores[index], {"candidates": sources, "chosen": names[index]}


def group_folds(labels, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    labels, groups = np.asarray(labels), np.asarray(groups)
    present = set(labels.tolist())
    for count in (5, 3, 2):
        if any(len(set(groups[labels == cls])) < count for cls in present):
            continue
        folds = list(StratifiedGroupKFold(count, shuffle=True, random_state=0).split(np.zeros(len(labels)), labels, groups))
        if all(set(labels[train]) == present and set(labels[val]) == present and
               not set(groups[train]) & set(groups[val]) for train, val in folds):
            return folds
    raise ValueError("신발 그룹·클래스를 유지하는 보조 검증 분할 불가")


def grouped_check(rows, dino, yolo):
    labels = np.asarray([c.ORDER.index(r["label"]) for r in rows])
    groups = np.asarray([f"{r['family']}:{r['group']}" for r in rows])
    folds = group_folds(labels, groups)
    predictions = {name: np.full(len(rows), -1, dtype=int) for name in ("DINO", "YOLO", "DINO+YOLO")}
    details = []
    for train, val in folds:
        variants = [fitted(labels[train], p[train]) for p in yolo.values()]
        choice = list(yolo)[rank(variants)]
        scores = {"DINO": dino, "YOLO": yolo[choice], "DINO+YOLO": c.mean_scores(dino, yolo[choice])}
        record = {"val_files": [rows[i]["file"] for i in val], "yolo": choice, "thresholds": {}}
        for name, p in scores.items():
            thresholds = choose_thresholds(labels[train], p[train])
            predictions[name][val] = predict_classes(p[val], thresholds)
            record["thresholds"][name] = thresholds
        details.append(record)
    results = {}
    for name, predicted in predictions.items():
        if (predicted < 0).any():
            raise ValueError("보조 검증 예측 누락")
        # fold마다 다른 기준으로 이미 결정된 exact-match 판정을 집계한다.
        result = metrics(labels, np.column_stack((predicted % 2, predicted // 2)), [.5, .5])
        result["thresholds"] = None
        results[name] = result
    unique = sorted(set(groups))
    matrices = {}
    for name, p in predictions.items():
        matrices[name] = np.asarray([np.bincount(4*labels[groups == g]+p[groups == g], minlength=16).reshape(4, 4) for g in unique])
    random = np.random.default_rng(0)
    differences = {"DINO+YOLO - DINO": [], "DINO+YOLO - YOLO": []}
    for _ in range(1000):
        sample = random.integers(0, len(unique), len(unique))
        values = {}
        for name, cm in matrices.items():
            combined = cm[sample].sum(0)
            counts = combined.sum(1)[:3]
            if (counts == 0).any():
                break
            values[name] = float(np.mean(np.diag(combined)[:3] / counts))
        if len(values) == 3:
            for name in ("DINO", "YOLO"):
                differences[f"DINO+YOLO - {name}"].append(100*(values["DINO+YOLO"]-values[name]))
    intervals = {name: np.percentile(delta, [2.5, 97.5]).tolist() for name, delta in differences.items() if delta}
    return {"folds": details, "results": results, "paired_group_bootstrap_95_percentile_pp": intervals,
            "caveat": "이미 전체 val로 DINO·설계를 개발했음. 독립 test 또는 백본 교차검증 아님."}


def table(results):
    lines = ["| 모델 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 | both 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        values = [result["balanced_accuracy"], result["accuracy"], *[result["recall"][key] for key in c.ORDER]]
        lines.append(f"| {name} | " + " | ".join("N/A" if v is None else f"{v:.2%}" for v in values)+" |")
    return "\n".join(lines)


def report(run):
    protocol = c.load(run)
    parts = c.verify_data(run)
    rows = parts["val"] + parts["val25"]
    labels = [c.ORDER.index(r["label"]) for r in rows]
    training = c.read(run / "training_complete.json")
    dino_name, dino, dino_sources = dino_candidate(rows)
    yolo = {}
    for head, ids in (("nms", ("Y1 NMS", "Y2 NMS+TTA")), ("free", ("Y3 NMS-free", "Y4 NMS-free+TTA"))):
        folder = run / head
        evidence = c.read(folder / "complete.json")
        if evidence["checkpoint_sha256"] != training["checkpoint_sha256"] or not evidence.get("gpu_released"):
            raise ValueError("YOLO 같은 가중치·GPU 프로세스 종료 미확인")
        for name, expected in evidence["files"].items():
            if c.sha(folder / name) != expected:
                raise ValueError("YOLO 추론 저장물 변경")
        original = c.read_scores(folder / "original_probs.csv", rows)
        repeated = c.read_scores(folder / "reproduced_probs.csv", rows)
        flipped = c.read_scores(folder / "flip_probs.csv", rows)
        averaged = c.read_scores(folder / "tta_probs.csv", rows)
        np.testing.assert_array_equal(averaged, c.mean_scores(original, flipped))
        np.testing.assert_allclose(original, repeated, atol=1e-4, rtol=0)
        thresholds = choose_thresholds(labels, original)
        if metrics(labels, original, thresholds)["confusion_matrix"] != metrics(labels, repeated, thresholds)["confusion_matrix"]:
            raise ValueError("YOLO 원본 최종 판정 재현 불일치")
        yolo[ids[0]], yolo[ids[1]] = original, averaged
    results = {name: fitted(labels, p) for name, p in yolo.items()}
    chosen = list(yolo)[rank(list(results.values()))]
    results = {f"DINO {dino_name}": fitted(labels, dino), **results}
    averaged = c.mean_scores(dino, yolo[chosen])
    c.save_scores(run / "ensemble_probs.csv", rows, averaged)
    results[f"DINO+YOLO ({chosen}, 50:50)"] = fitted(labels, averaged)
    supplemental = grouped_check(rows, dino, yolo)
    c.write(run / "group_validation.json", supplemental)
    payload = {"completed_at": c.now(), "protocol_sha256": protocol["sha256"], "results": results,
               "dino_sources": dino_sources, "chosen_yolo": chosen, "weights": [.5, .5],
               "test_evaluated": False, "detr_used": False, "group_validation": supplemental,
               "checkpoint_sha256": training["checkpoint_sha256"], "all_gpu_processes_released": True}
    c.write(run / "final_comparison.json", payload)
    text = f"# YOLO-01 · 9번까지 비교 완료\n\n생성: {c.now()}. 합산 val 1,319장의 개발 결과입니다. test·DETR는 사용하지 않았습니다.\n\n"
    text += f"DINO 대표: {dino_name}. YOLO 대표: {chosen}. 결합은 클래스별 두 점수의 50:50 평균입니다.\n\n"+table(results)+"\n"
    for name, result in results.items():
        text += f"\n## {name}\n\n판정 기준: heel {result['thresholds'][0]:.2f}, outsole {result['thresholds'][1]:.2f}.\n\n{render(result)}\n"
    text += "\n## 신발 그룹 분리 보조 검증\n\n"+table(supplemental["results"])+"\n\n"+supplemental["caveat"]+"\n"
    for name, interval in supplemental["paired_group_bootstrap_95_percentile_pp"].items():
        text += f"\n- {name}: 균형 정확도 차이의 그룹 bootstrap 구간 {interval[0]:+.2f}~{interval[1]:+.2f}%p.\n"
    (run / "final_comparison.md").write_text(text, encoding="utf-8")
    record = c.ROOT / "classifier/dino/EXPERIMENTS.md"
    current = record.read_text(encoding="utf-8")
    begin, end = "<!-- YOLO-01-RUNTIME-START -->", "<!-- YOLO-01-RUNTIME-END -->"
    if current.count(begin) != 1 or current.count(end) != 1:
        raise ValueError("YOLO 결과 문서 표식 누락·중복")
    first, tail = current.split(begin)
    _, last = tail.split(end)
    record.write_text(first+begin+"\n\n"+text+"\n"+end+last, encoding="utf-8")
    print(text, flush=True)
