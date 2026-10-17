"""사용자가 중단을 승인한 epoch의 보존 체크포인트만 val로 평가한다."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def validate_stop(checkpoint, protocol, decision):
    expected = decision.get("requested_epoch")
    planned = protocol["config"]["epochs"]
    if decision.get("status") != "stopped_at_checkpoint":
        raise ValueError("체크포인트 저장 후 중단이 확인되지 않음")
    if checkpoint["epoch"] != expected or decision.get("completed_epoch") != expected:
        raise ValueError("사용자가 선택한 중단 epoch와 체크포인트가 다름")
    if not 0 < expected < planned or decision.get("original_planned_epochs") != planned:
        raise ValueError("원래 계획과 중단 epoch 불일치")
    if checkpoint["identity"] != protocol["identity"]:
        raise ValueError("체크포인트의 데이터·설정·코드 identity 불일치")
    if not checkpoint["history"] or checkpoint["history"][-1]["epoch"] != expected:
        raise ValueError("체크포인트 평가 기록의 epoch 불일치")
    return expected, planned


def evaluate(run_dir):
    import numpy as np
    import torch
    from classifier.dino.data import ORDER
    from classifier.dino.evaluate import choose_thresholds, metrics, report_markdown, save_predictions
    from classifier.dino.model import DamageDINO
    from classifier.dino.train import cleanup, gpu_gate, log, now, prepare, read_json, record, seed_all, write_json

    protocol = read_json(run_dir / "protocol.json")
    decision = read_json(run_dir / "stop_decision.json")
    saved = torch.load(run_dir / "last_epoch12.pt", map_location="cpu", weights_only=False)
    actual_epochs, planned_epochs = validate_stop(saved, protocol, decision)
    config = protocol["config"]
    parts, _, _ = prepare(config, run_dir, protocol["backbone_path"], existing=True)
    selection = read_json(run_dir / "selection.json")
    if saved["selection"] != selection:
        raise ValueError("학습한 배치·누적 설정과 다름")
    gpu_gate(config, initial=True)
    seed_all(config["seed"])
    model = DamageDINO(config, protocol["backbone_path"], selection["checkpointing"]).to("cuda")
    model.load_trainable(saved["model"])
    from classifier.dino.train import predict
    probabilities, labels, fixed = {}, {}, {}
    record(run_dir, f"사용자 결정으로 {actual_epochs}/{planned_epochs} epoch 중단 · val 평가 중",
           f"`last_epoch12.pt`를 사용합니다. 원래 30 epoch 학습률 스케줄의 12 epoch 시점입니다. test는 미평가입니다.")
    for name in ("val", "val25"):
        log(f"중단 가중치 평가: epoch={actual_epochs}, split={name}, images={len(parts[name])}")
        probabilities[name], _ = predict(model, parts[name], config, selection["batch"])
        labels[name] = [ORDER.index(row["label"]) for row in parts[name]]
        fixed[name] = metrics(labels[name], probabilities[name])
        reference = saved["history"][-1]["validation_fixed_0_5"][name]
        if fixed[name]["confusion_matrix"] != reference["confusion_matrix"]:
            raise RuntimeError("저장 전 검증과 재로딩 후 고정 기준 혼동 행렬이 다름")
        save_predictions(run_dir / f"{name}_probs.csv", parts[name], probabilities[name])
    thresholds = choose_thresholds(np.concatenate(list(labels.values())), np.concatenate(list(probabilities.values())))
    results = {name: metrics(labels[name], probabilities[name], thresholds) for name in probabilities}
    write_json(run_dir / "thresholds.json", {
        "values": thresholds, "output_order": config["output_classes"],
        "selection": "pooled val+val25 balanced accuracy; predefined 0.01..0.99 grid; accuracy/distance/heel/outsole tie-break",
        "checkpoint_epoch": actual_epochs, "original_planned_epochs": planned_epochs})
    write_json(run_dir / "metrics.json", results)
    write_json(run_dir / "metrics_fixed_0_5.json", fixed)
    for name, result in results.items():
        with (run_dir / f"{name}_confusion.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true/pred", *ORDER])
            writer.writerows([label, *row] for label, row in zip(ORDER, result["confusion_matrix"]))
    model.backbone.save_pretrained(run_dir / "lora_adapter", safe_serialization=True)
    torch.save(model.head.state_dict(), run_dir / "head.pt")
    write_json(run_dir / "inference_config.json", {**config, "checkpoint_epoch": actual_epochs, "thresholds": thresholds})
    seconds = sum(row["epoch_seconds"] for row in saved["history"])
    summary = (
        f"원래 계획: **{planned_epochs} epoch**. 실제: **{actual_epochs} epoch 완료 후 사용자 요청으로 중단**.\n\n"
        "검증 성능 정체와 검증 loss 상승을 보고 중단했습니다. 원래 30 epoch cosine 스케줄을 유지한 채 멈춘 것이며, "
        "12 epoch 전용 스케줄로 학습한 실험과는 구분합니다.\n\n"
        f"- 체크포인트: `last_epoch12.pt`\n- 학습·epoch별 검증 합계: {seconds/3600:.2f}시간 (저장·별도 최종 평가 제외)\n"
        f"- 공통 판정 기준: 안감 **{thresholds[0]:.2f}**, 밑창 **{thresholds[1]:.2f}**\n"
        "- 기준 선택: v5 val + val25 합산 균형 정확도. 아래 점수는 기준 선택에 사용한 val 결과입니다.\n"
        "- 재로딩 검증: 두 split의 고정 기준 0.5 혼동 행렬이 저장 전과 일치합니다.\n"
        "- test 미평가, 다음 실험 미실행.\n\n" + report_markdown(results))
    (run_dir / "report.md").write_text(summary + "\n", encoding="utf-8")
    record(run_dir, f"{actual_epochs}/{planned_epochs} epoch에서 중단 · val 평가 완료", summary)
    write_json(run_dir / "evaluation_metadata.json", {"finished_at": now(), "actual_epochs": actual_epochs,
        "planned_epochs": planned_epochs, "stop_reason": decision["reason"], "identity": protocol["identity"],
        "test_evaluated": False, "thresholds_selected_on": ["val", "val25"], "reload_confusion_matches": True})
    write_json(run_dir / "training_status.json", {"status": "stopped_evaluated", "completed_epoch": actual_epochs,
        "original_planned_epochs": planned_epochs, "finished_at": now(), "test_evaluated": False})
    from classifier.dino.tools.report_combined import refresh_report
    log(refresh_report(run_dir))
    del model, saved
    cleanup()


def main():
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(repo / "runs" / "dino"):
        parser.error("현재 저장소의 runs/dino 경로만 사용합니다")
    try:
        evaluate(run_dir)
    except Exception as exc:
        from classifier.dino.train import now, record, write_json
        write_json(run_dir / "stopped_evaluation_failure.json", {"time": now(), "error": f"{type(exc).__name__}: {exc}"})
        record(run_dir, "12 epoch 중단 · val 평가 오류", f"`{type(exc).__name__}: {exc}`\n\n보존한 체크포인트를 확인한 뒤 평가를 재개합니다.")
        raise


if __name__ == "__main__":
    main()
