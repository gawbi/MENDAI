"""최종 val 보고 완료 후 저장된 결과만 CPU로 분석한다. 모델·이미지·test는 읽지 않는다."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

# 이 프로세스에만 적용한다. 진행 중인 학습이나 다른 앱의 환경은 바꾸지 않는다.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ORDER = ["normal", "heel", "outsole", "both"]
REPO = Path(__file__).resolve().parents[3]
SOURCES = {
    "calibration": "https://scikit-learn.org/stable/modules/calibration.html",
    "temperature": "https://proceedings.mlr.press/v70/guo17a.html",
    "ensemble": "https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier",
    "frozen": "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m",
    "pos_weight": "https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html",
}


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def completion(run):
    """학습뿐 아니라 별도 최종 평가·합산 보고 프로세스까지 끝나야 준비됨."""
    log = run / "console.log"
    if not log.exists():
        return "waiting"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 8192))
        tail = stream.read().decode("utf-8", errors="replace")
    exits = re.findall(r"^\[exit (-?\d+)\]\s*$", tail, re.MULTILINE)
    if not exits:
        return "waiting"
    if int(exits[-1]) != 0:
        return "failed"
    status = read(run / "training_status.json")
    planned = read(run / "protocol.json")["config"]["epochs"]
    return "ready" if status["status"] == "complete" and status["completed_epoch"] == planned else "failed"


def score(y, prediction):
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    support = matrix.sum(1)
    recall = {name: float(matrix[i, i] / support[i]) if support[i] else None for i, name in enumerate(ORDER)}
    present = [value for value in recall.values() if value is not None]
    return {"images": int(support.sum()), "confusion_matrix": matrix.tolist(), "recall": recall,
            "accuracy": float(matrix.trace() / support.sum()), "balanced_accuracy": float(np.mean(present))}


def prediction(probabilities, thresholds):
    return (probabilities[:, 0] >= thresholds[0]).astype(int) + 2 * (probabilities[:, 1] >= thresholds[1]).astype(int)


def load_run(run):
    protocol, manifest = read(run / "protocol.json"), read(run / "data_manifest.json")
    if manifest["sha256"] != protocol["identity"]["data_sha256"]:
        raise ValueError("검증 명세와 protocol 해시 불일치")
    labels, groups = {}, {}
    for part in ("val", "val25"):
        for row in manifest["images"][part]:
            name = row["file"]
            if name in labels or Path(name).name != name:
                raise ValueError("검증 파일 중복 또는 파일명 오류")
            labels[name] = ORDER.index(row["label"])
            groups[name] = part + ":" + str(row["group"])
    probabilities = {}
    with (run / "val_combined_probs.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["file"]
            values = [float(row["p_heel_lining"]), float(row["p_outsole"])]
            if name in probabilities or any(not np.isfinite(p) or not 0 <= p <= 1 for p in values):
                raise ValueError("확률 중복·범위 오류")
            probabilities[name] = values
    if set(probabilities) != set(labels):
        raise ValueError("검증 정답·예측 목록 불일치")
    files = sorted(labels)
    y = np.array([labels[name] for name in files])
    p = np.array([probabilities[name] for name in files])
    thresholds = read(run / "thresholds.json")["values"]
    if len(thresholds) != 2 or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("판정 기준 오류")
    pred = prediction(p, thresholds)
    measured = score(y, pred)
    stored = read(run / "metrics.json")["val_combined"]
    if (measured["confusion_matrix"] != stored["confusion_matrix"] or measured["images"] != stored["images"]
            or thresholds != stored["thresholds"]):
        raise ValueError("저장된 지표와 사진별 재계산 결과가 다름")
    return {"files": files, "y": y, "p": p, "prediction": pred, "groups": [groups[name] for name in files],
            "thresholds": thresholds, "score": measured, "protocol": protocol}


def paired_comparison(a, b):
    if a["files"] != b["files"] or not np.array_equal(a["y"], b["y"]) or a["groups"] != b["groups"]:
        raise ValueError("두 모델의 검증 사진·정답·그룹이 다름")
    if a["protocol"]["identity"]["data_sha256"] != b["protocol"]["identity"]["data_sha256"]:
        raise ValueError("서로 다른 데이터 확정본")
    first, second = a["prediction"] == a["y"], b["prediction"] == b["y"]
    return {"both_correct": int(np.sum(first & second)), "only_v2_correct": int(np.sum(first & ~second)),
            "only_v3_correct": int(np.sum(~first & second)), "both_wrong": int(np.sum(~first & ~second)),
            "different_predictions": int(np.sum(a["prediction"] != b["prediction"]))}


def paired_interval(a, b, repeats=2000):
    """신발 그룹 단위 짝지은 부트스트랩. 이미 고른 기준에 조건부인 설명용 구간."""
    unique = sorted(set(a["groups"]))
    group_index = {name: i for i, name in enumerate(unique)}
    matrices = np.zeros((len(unique), 2, 4, 4), dtype=np.int64)
    for i, group in enumerate(a["groups"]):
        for j, model in enumerate((a, b)):
            matrices[group_index[group], j, a["y"][i], model["prediction"][i]] += 1
    present = np.bincount(a["y"], minlength=4) > 0
    rng, differences = np.random.default_rng(0), []
    for _ in range(repeats):
        sampled = matrices[rng.integers(0, len(unique), len(unique))].sum(0)
        support = sampled.sum(2)
        if np.any(support[:, present] == 0):
            continue
        recalls = np.diagonal(sampled, axis1=1, axis2=2)[:, present] / support[:, present]
        differences.append(float(recalls[1].mean() - recalls[0].mean()))
    return {"groups": len(unique), "valid_repeats": len(differences),
            "v3_minus_v2_ba_interval": np.quantile(differences, [.025, .975]).tolist() if differences else None}


def diagnostics(model):
    p, y = model["p"], model["y"]
    binary = np.column_stack([y % 2, y // 2])
    calibration = {}
    for column, name in enumerate(("heel", "outsole")):
        bins = []
        for indices in np.array_split(np.argsort(p[:, column], kind="stable"), min(10, len(y))):
            bins.append({"images": len(indices), "mean_probability": float(p[indices, column].mean()),
                         "positive_fraction": float(binary[indices, column].mean())})
        calibration[name] = {"brier": float(np.mean((p[:, column] - binary[:, column])**2)), "bins": bins}
    nearby = []
    for dh in (-.01, 0, .01):
        for do in (-.01, 0, .01):
            thresholds = np.clip(np.array(model["thresholds"]) + [dh, do], .001, .999)
            nearby.append(score(y, prediction(p, thresholds))["balanced_accuracy"])
    return {"calibration": calibration, "nearby_threshold_ba_min": min(nearby), "nearby_threshold_ba_max": max(nearby)}


def curve(run):
    history = read(run / "epochs.json")
    rows = []
    for row in history:
        sources = list(row["validation_fixed_0_5"].values())
        matrix = np.array([s["confusion_matrix"] for s in sources]).sum(0)
        support = matrix.sum(1)
        ba = float(np.mean(matrix.diagonal()[support > 0] / support[support > 0]))
        rows.append({"epoch": row["epoch"], "train_loss": row["train"]["loss"], "fixed_ba": ba,
                     "val_loss": sum(s["loss"] * s["images"] for s in sources) / sum(s["images"] for s in sources)})
    best = max(rows, key=lambda r: r["fixed_ba"])
    tail = rows[max(0, len(rows) - 5)]
    flag = rows[-1]["train_loss"] < tail["train_loss"] and rows[-1]["val_loss"] > tail["val_loss"] * 1.2 and rows[-1]["fixed_ba"] < best["fixed_ba"] - .01
    return {"epochs": rows, "best_fixed_ba_epoch": best["epoch"], "possible_overfit": bool(flag)}


def percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def make_report(models, overlap, interval, detailed, learning):
    lines = ["# DINOv3 완료 후 개선 방향 · CPU 분석", "", f"생성 시각: {now()}", "",
             "저장된 합산 val 결과만 사용했습니다. 모델·가중치·이미지·test를 열거나 새 추론을 실행하지 않았습니다.", "",
             "## 검증 결과 비교", "", "| 모델 | 사진 수 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 | both 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in models.items():
        s = model["score"]
        values = [s["balanced_accuracy"], s["accuracy"], *s["recall"].values()]
        lines.append(f"| {name} | {s['images']} | " + " | ".join(percent(v) for v in values) + " |")
    for name, model in models.items():
        lines += ["", f"**{name} 합산 혼동 행렬** · 행 실제 / 열 예측", "",
                  "| 실제 / 예측 | normal | heel | outsole | both |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {label} | " + " | ".join(map(str, row)) + " |" for label, row in zip(ORDER, model["score"]["confusion_matrix"])]
    a, b = models["DINOv2"], models["DINOv3"]
    delta = b["score"]["balanced_accuracy"] - a["score"]["balanced_accuracy"]
    lines += ["", f"DINOv3 − DINOv2 균형 정확도 차이: **{delta * 100:+.2f}%p**.",
              "학습 스케줄·중단 조건이 다르므로 실제 후보 성능 비교이며 백본만의 효과로 해석하지 않습니다.", "",
              f"둘 다 정답 {overlap['both_correct']}장, DINOv2만 정답 {overlap['only_v2_correct']}장, DINOv3만 정답 {overlap['only_v3_correct']}장, 둘 다 오답 {overlap['both_wrong']}장입니다.",
              "한쪽만 맞힌 사례는 보완 가능성을 보여 주지만 평균 앙상블의 개선을 보장하지 않습니다."]
    ci = interval["v3_minus_v2_ba_interval"]
    if ci:
        lines += [f"신발 그룹 {interval['groups']}개를 재표집한 차이의 95% 구간: {ci[0]*100:+.2f} ~ {ci[1]*100:+.2f}%p.",
                  "val에서 이미 고른 기준을 고정한 설명용 표본 변동입니다. 기준·모델 선택 편향과 seed 변동은 포함하지 않으며 최종 test 성능 구간이 아닙니다."]
    chosen = max(models, key=lambda name: models[name]["score"]["balanced_accuracy"])
    weakest = min((name for name in ORDER if models[chosen]["score"]["recall"][name] is not None), key=lambda name: models[chosen]["score"]["recall"][name])
    lines += ["", "## 개선 우선순위", "",
              f"1. **오분류 원인 확인 · GPU 불필요.** 현재 val 점수가 높은 후보는 {chosen}이고 가장 낮은 재현율은 {weakest}입니다. `paired_errors.csv`에서 해당 정답의 공통 오판부터 확인합니다. 라벨·분할·기준은 자동으로 바꾸지 않습니다.",
              "2. **탐지기와의 앙상블 검증 준비 · 기존 예측이 있으면 CPU만 사용.** YOLO·DETR의 같은 val 파일별 확률을 받아 미리 정한 소수 조합으로 비교하는 것을 우선합니다. 실제 추가 모델 추론은 실행하지 않았습니다. "
              f"[공식 확률 평균 방식]({SOURCES['ensemble']})",
              "3. **확률 보정 검토 · 저장된 점수로 CPU 실험 가능.** reliability bin과 클래스별 Brier를 같이 봅니다. Brier만으로 보정 품질을 단정하지 않습니다. 앙상블 전 두 sigmoid 출력에 별도 sigmoid 보정을 검토하고, 보정 학습과 평가는 신발 그룹을 나눈 val 내부 교차검증으로 분리합니다. 두 출력을 softmax로 정규화하지 않습니다. "
              f"[공식 보정 지침]({SOURCES['calibration']})",
              "단조 보정은 점수 순서를 보존하므로 이미 기준을 선택한 단일 분류기의 정확도를 반드시 높이지 않습니다. 온도 보정 연구도 분류 정확도와 확률 보정을 구분합니다. "
              f"[Guo 외, ICML 2017]({SOURCES['temperature']})",
              "4. **추가 학습은 근거가 있을 때만.** 조건을 맞추기 위한 DINOv2 15 epoch 재학습은 우선하지 않습니다."]
    if learning["possible_overfit"]:
        lines += ["DINOv3의 후반 train loss 감소와 val loss 상승·고정 기준 성능 하락이 함께 관찰됐습니다. 향후 GPU 사용이 다시 승인되면 낮은 학습률 또는 백본 고정 + 분류층 학습을 각각 후보로 검토합니다. 원인이 확정된 것은 아니며 개선도 보장되지 않습니다."]
    else:
        lines += ["설정한 곡선 점검 조건에서는 뚜렷한 후반 과적합 신호가 함께 잡히지 않았습니다. 이것만으로 과적합이 없다고 단정하거나 epoch를 늘리지 않습니다."]
    lines += [f"DINOv3 공식 모델 카드는 고정 특징에 간단한 분류기를 붙이는 사용법도 제시합니다. 특징이 저장돼 있지 않으면 추출에 별도 연산이 필요하므로 현재 GPU로 실행하지 않습니다. [공식 모델 카드]({SOURCES['frozen']})", "",
              "## 판정 기준과 학습 곡선", "",
              f"DINOv3 고정 기준 0.5의 최고 검증 epoch는 {learning['best_fixed_ba_epoch']}입니다. 해당 epoch의 가중치가 보존돼 있다는 뜻은 아니며 best 재선택도 하지 않았습니다."]
    for name, model in models.items():
        d = detailed[name]
        lines += [f"- {name}: 선택 기준 {model['thresholds']}, 두 기준을 각각 ±0.01 움직인 9개 진단 조합의 균형 정확도 범위 {percent(d['nearby_threshold_ba_min'])} ~ {percent(d['nearby_threshold_ba_max'])}. 원래 기준은 유지했습니다."]
    lines += ["", "기준값이 극단적이라는 이유만으로 불량 모델이나 보정 실패로 판단하지 않습니다. 가중 BCE는 정밀도·재현율의 균형에도 영향을 주며, 데이터 비율과 순위 성능도 함께 확인해야 합니다. "
              f"[PyTorch 공식 손실 설명]({SOURCES['pos_weight']})", "",
              "모델·학습 설정·앙상블·판정 기준은 val로 선택하고 test는 설정이 확정된 뒤 사용합니다. 개선안은 제안이며 자동 적용·새 학습은 실행하지 않았습니다."]
    return "\n".join(lines) + "\n"


def analyze(run, baseline, update_document=True):
    if completion(run) != "ready":
        raise RuntimeError("최종 val·합산 보고가 아직 성공적으로 끝나지 않음")
    models = {"DINOv2": load_run(baseline), "DINOv3": load_run(run)}
    overlap = paired_comparison(*models.values())
    interval = paired_interval(*models.values())
    detailed = {name: diagnostics(model) for name, model in models.items()}
    learning = curve(run)
    report = make_report(models, overlap, interval, detailed, learning)
    a, b = models.values()
    with (run / "paired_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "actual", "v2_prediction", "v3_prediction", "v2_heel", "v2_outsole", "v3_heel", "v3_outsole"])
        for i, name in enumerate(a["files"]):
            if a["prediction"][i] != a["y"][i] or b["prediction"][i] != b["y"][i]:
                writer.writerow([name, ORDER[a["y"][i]], ORDER[a["prediction"][i]], ORDER[b["prediction"][i]], *a["p"][i], *b["p"][i]])
    (run / "post_run_analysis.md").write_text(report, encoding="utf-8")
    write(run / "post_run_analysis.json", {"created_at": now(), "gpu_used": False, "model_inference": False,
          "test_evaluated": False, "metrics": {name: model["score"] for name, model in models.items()},
          "paired_errors": overlap, "paired_group_interval": interval, "diagnostics": detailed, "learning_curve": learning,
          "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "references": SOURCES,
          "inputs_sha256": {str(folder.relative_to(run.parent)) + "/" + name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                            for folder in (baseline, run) for name in ("protocol.json", "data_manifest.json", "thresholds.json", "val_combined_probs.csv", "metrics.json", "epochs.json")}})
    if update_document:
        path = REPO / "classifier/dino/EXPERIMENTS.md"
        text = path.read_text(encoding="utf-8")
        start, end = "<!-- EX-02-CPU-ANALYSIS-START -->", "<!-- EX-02-CPU-ANALYSIS-END -->"
        block = f"{start}\n\n### 학습 후 CPU 분석\n\n완료 시각: {now()}. GPU·모델 추론·test 사용 없음.\n\n보고서: `{run.relative_to(REPO).as_posix()}/post_run_analysis.md`. 오분류 목록은 `paired_errors.csv`, 세부 수치는 `post_run_analysis.json`입니다.\n\n{end}"
        if start in text:
            before, tail = text.split(start, 1)
            _, after = tail.split(end, 1)
            text = before + block + after
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        text = text.replace("| 15 / 진행 중 |", "| 15 / 15 |")
        path.write_text(text, encoding="utf-8")
    if "torch" in sys.modules:
        raise RuntimeError("CPU 분석 프로세스에 예상 밖 torch import 발생")
    return {"status": "complete", "finished_at": now(), "gpu_used": False, "report": str(run / "post_run_analysis.md")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=12)
    args = parser.parse_args()
    run, baseline = args.run_dir.resolve(), args.baseline_run_dir.resolve()
    if not all(p.is_relative_to(REPO / "runs/dino") for p in (run, baseline)):
        parser.error("현재 저장소의 runs/dino 안에 있는 실행만 사용")
    status_path = run / "post_run_analysis_status.json"
    write(status_path, {"status": "waiting_for_final_validation", "pid": os.getpid(), "started_at": now(), "gpu_used": False})
    deadline = time.monotonic() + args.timeout_hours * 3600
    try:
        while completion(run) == "waiting" and args.watch and time.monotonic() < deadline:
            time.sleep(30)
        result = analyze(run, baseline)
        write(status_path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        result = {"status": "failed_or_not_ready", "finished_at": now(), "error_type": type(exc).__name__, "error": str(exc), "gpu_used": False}
        write(status_path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
