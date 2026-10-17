"""기존 예측을 합산해 검증 결과 하나를 보고한다. 모델·GPU는 사용하지 않는다."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ORDER = ["normal", "heel", "outsole", "both"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def combine(results):
    if not results:
        raise ValueError("합산할 평가 결과가 없음")
    thresholds = list(next(iter(results.values()))["thresholds"])
    matrix = [[0] * 4 for _ in range(4)]
    for result in results.values():
        current = result["confusion_matrix"]
        if result["order"] != ORDER or result["thresholds"] != thresholds:
            raise ValueError("클래스 순서·판정 기준이 달라 합산할 수 없음")
        if len(current) != 4 or any(len(row) != 4 for row in current):
            raise ValueError("4×4 혼동 행렬이 아님")
        if any(not isinstance(n, int) or n < 0 for row in current for n in row):
            raise ValueError("음수가 아닌 정수 건수만 합산할 수 있음")
        if sum(map(sum, current)) != result["images"]:
            raise ValueError("사진 수와 혼동 행렬 건수가 다름")
        matrix = [[matrix[i][j] + current[i][j] for j in range(4)] for i in range(4)]
    counts = list(map(sum, matrix))
    total = sum(counts)
    if not total:
        raise ValueError("빈 평가 결과")
    recall = {name: matrix[i][i]/counts[i] if counts[i] else None for i, name in enumerate(ORDER)}
    present = [value for value in recall.values() if value is not None]
    return {"images": total, "thresholds": thresholds, "order": ORDER, "confusion_matrix": matrix,
            "accuracy": sum(matrix[i][i] for i in range(4))/total,
            "recall": recall, "balanced_accuracy": sum(present)/len(present)}


def render(result):
    def percent(value):
        return "N/A" if value is None else f"{value*100:.2f}%"
    values = [result["balanced_accuracy"], result["accuracy"]] + [result["recall"][name] for name in ORDER]
    lines = ["### 평가 결과", "", "| 검증셋 | 사진 수 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 | both 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             f"| 합산 val | {result['images']} | " + " | ".join(map(percent, values)) + " |", "",
             "**합산 val 혼동 행렬** — 행: 실제 정답 / 열: 예측", "",
             "| 실제 / 예측 | normal | heel | outsole | both |", "|---|---:|---:|---:|---:|"]
    for name, row in zip(ORDER, result["confusion_matrix"]):
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def verify_predictions(run_dir, protocol, thresholds, expected):
    root = Path(__file__).resolve().parents[3] / protocol["config"]["dataset_root"]
    rows, seen = [], set()
    matrix = [[0] * 4 for _ in range(4)]
    for split, family in [("val", "v5"), ("val25", "eval25")]:
        with (root / "photo_labels" / f"damage_image_labels_{family}.csv").open(encoding="utf-8-sig", newline="") as f:
            labels = {row["file"]: row["label"] for row in csv.DictReader(f) if row["split"] == "val"}
        split_files = set()
        with (run_dir / f"{split}_probs.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name = row["file"]
                if name in seen or name not in labels:
                    raise ValueError("예측 파일명 중복 또는 정답 누락")
                seen.add(name)
                split_files.add(name)
                heel, outsole = float(row["p_heel_lining"]), float(row["p_outsole"])
                if not (0 <= heel <= 1 and 0 <= outsole <= 1):
                    raise ValueError("예측 점수 범위 오류")
                prediction = int(heel >= thresholds[0]) + 2*int(outsole >= thresholds[1])
                matrix[ORDER.index(labels[name])][prediction] += 1
                rows.append([name, heel, outsole])
        if split_files != set(labels):
            raise ValueError("정답과 예측의 파일 목록 불일치")
    if matrix != expected["confusion_matrix"] or len(rows) != expected["images"]:
        raise ValueError("합산 혼동 행렬과 사진별 예측 재계산 결과가 다름")
    with (run_dir / "val_combined_probs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "p_heel_lining", "p_outsole"])
        writer.writerows(rows)


def refresh_report(run_dir):
    run_dir = Path(run_dir).resolve()
    protocol = read_json(run_dir / "protocol.json")
    current = read_json(run_dir / "metrics.json")
    pooled = combine(current)
    fixed = combine(read_json(run_dir / "metrics_fixed_0_5.json")) if (run_dir / "metrics_fixed_0_5.json").exists() else None
    verify_predictions(run_dir, protocol, pooled["thresholds"], pooled)
    archive = run_dir / "raw_by_split"
    archive.mkdir(exist_ok=True)
    for filename in ("metrics.json", "metrics_fixed_0_5.json", "report.md"):
        source, backup = run_dir / filename, archive / filename
        if source.exists() and not backup.exists():
            backup.write_bytes(source.read_bytes())
    write_json(run_dir / "metrics.json", {"val_combined": pooled})
    if fixed:
        write_json(run_dir / "metrics_fixed_0_5.json", {"val_combined": fixed})
    with (run_dir / "val_combined_confusion.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *ORDER])
        writer.writerows([name, *row] for name, row in zip(ORDER, pooled["confusion_matrix"]))
    status = read_json(run_dir / "training_status.json")
    actual = status["completed_epoch"]
    planned = status.get("original_planned_epochs", protocol["config"]["epochs"])
    epochs = read_json(run_dir / "epochs.json")
    combined_history = []
    for epoch in epochs:
        sources = epoch["validation_fixed_0_5"]
        validation = combine(sources)
        validation["loss"] = sum(value["loss"]*value["images"] for value in sources.values())/validation["images"]
        combined_history.append({"epoch": epoch["epoch"], "train_loss": epoch["train"]["loss"], "validation_fixed_0_5": validation})
    write_json(run_dir / "epochs_combined.json", combined_history)
    summary = (
        f"원래 계획: **{planned} epoch**. 실제: **{actual} epoch**.\n\n"
        f"- 평가 대상: **합산 val {pooled['images']:,}장**\n"
        f"- 학습·epoch별 검증 합계: {sum(row['epoch_seconds'] for row in epochs)/3600:.2f}시간\n"
        f"- 공통 판정 기준: 안감 **{pooled['thresholds'][0]:.2f}**, 밑창 **{pooled['thresholds'][1]:.2f}**\n"
        "- 사용자 결정: 합산 결과만 보고하며, 건수를 합친 혼동 행렬에서 지표를 다시 계산합니다.\n"
        "- 아래 점수는 판정 기준 선택에 사용한 검증 결과입니다. test는 미평가이며 다음 실험은 실행하지 않았습니다.\n\n" + render(pooled))
    (run_dir / "report.md").write_text(summary + "\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[3]
    record_path = repo / "classifier/dino/EXPERIMENTS.md"
    text = record_path.read_text(encoding="utf-8")
    start, end = "<!-- EX-01-RUNTIME-START -->", "<!-- EX-01-RUNTIME-END -->"
    if start in text and end in text:
        before, tail = text.split(start, 1)
        _, after = tail.split(end, 1)
        relative = run_dir.relative_to(repo).as_posix()
        block = f"{start}\n\n### 실제 실행 기록\n\n**run:** `{relative}`\n\n{summary}\n\n{end}"
        record_path.write_text(before + block + after, encoding="utf-8")
    write_json(run_dir / "combined_reporting.json", {
        "updated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
        "reporting": "combined_only", "members": ["val", "val25"], "images": pooled["images"],
        "model_rerun": False, "thresholds_changed": False, "predictions_rechecked": True,
        "aggregation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return summary


def main():
    parser = argparse.ArgumentParser(description="기존 val 예측을 합산한 보고서 생성 · CPU 전용")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(refresh_report(args.run_dir), flush=True)


if __name__ == "__main__":
    main()
