"""같은 데이터의 완료된 LoRA 실험을 저장된 합산 val 지표로 비교한다."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .report_combined import ORDER, read_json, update_experiment_record, write_json

REPO = Path(__file__).resolve().parents[3]


def compare(run_dir, baseline_run):
    run_dir, baseline_run = Path(run_dir).resolve(), Path(baseline_run).resolve()
    runs = [baseline_run, run_dir]
    protocols = [read_json(run / "protocol.json") for run in runs]
    if len({p["identity"]["data_sha256"] for p in protocols}) != 1:
        raise ValueError("데이터 해시가 다른 실험은 비교하지 않음")
    results = []
    for run in runs:
        status = read_json(run / "training_status.json")
        if status["status"] not in ("complete", "completed"):
            raise ValueError(f"최종 평가가 완료되지 않음: {run}")
        result = read_json(run / "metrics.json")["val_combined"]
        if result["order"] != ORDER:
            raise ValueError("클래스 순서 불일치")
        results.append(result)
    if results[0]["images"] != results[1]["images"]:
        raise ValueError("검증 사진 수 불일치")
    def percent(value):
        return "N/A" if value is None else f"{100*value:.2f}%"
    lines = ["### 기존 LoRA와 비교", "",
             "| 실험 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 | both 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for protocol, result in zip(protocols, results):
        values = [result["balanced_accuracy"], result["accuracy"], *[result["recall"][c] for c in ORDER]]
        lines.append(f"| {protocol['config']['experiment']} | " + " | ".join(map(percent, values)) + " |")
    changes = {key: 100*(results[1][key] - results[0][key]) for key in ("balanced_accuracy", "accuracy")}
    lines += ["", f"기존 대비 균형 정확도 **{changes['balanced_accuracy']:+.2f}%p**, 전체 정확도 **{changes['accuracy']:+.2f}%p**입니다.", "",
              "각 실험의 마지막 가중치와 합산 val에서 선택한 판정 기준을 사용합니다. 같은 val을 반복 사용한 개발 결과이며 독립적인 test 결과는 아닙니다. 두 변경을 함께 적용했으므로 개별 변경의 효과는 분리할 수 없습니다."]
    comparison = "\n".join(lines)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    marker = "<!-- LORA-COMPARISON-START -->"
    report = report.split(marker, 1)[0].rstrip() + f"\n\n{marker}\n\n{comparison}\n\n<!-- LORA-COMPARISON-END -->\n"
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    write_json(run_dir / "comparison.json", {
        "data_sha256": protocols[0]["identity"]["data_sha256"], "test_evaluated": False,
        "runs": [{"run": str(run), "experiment": protocol["config"]["experiment"], "identity": protocol["identity"],
                  "metrics_sha256": hashlib.sha256((run / "metrics.json").read_bytes()).hexdigest(), "result": result}
                 for run, protocol, result in zip(runs, protocols, results)],
        "candidate_minus_baseline_percentage_points": changes})
    update_experiment_record(REPO / "classifier/dino/EXPERIMENTS.md", protocols[1]["config"]["experiment"],
                             f"{protocols[1]['config']['epochs']} epoch 및 val 비교 완료",
                             f"### 실제 실행 기록\n\n**run:** `{run_dir.relative_to(REPO).as_posix()}`\n\n{report}")
    print(comparison, flush=True)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run_dir, args.baseline_run)


if __name__ == "__main__":
    main()
