"""Recover completed EX-03 reporting from verified artifacts, without inference."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# No CUDA context is needed to check stored tensors and predictions.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

from ..linear_probe import pipeline as p


def verify_snapshot(run, protocol, folder):
    if p.digest(protocol["sources"]) != protocol["identity"]["source_sha256"]:
        raise ValueError("Archived source manifest mismatch")
    for name, expected in protocol["sources"].items():
        if p.file_sha(run / folder / name) != expected:
            raise ValueError(f"Archived source bytes mismatch: {name}")


def finalize(run):
    run = run.resolve()
    run.relative_to(p.ROOT / "runs/dino")
    original = p.read_json(run / "protocol.json")
    gpu = p.read_json(run / "gpu_head_protocol.json")
    verify_snapshot(run, original, "source_snapshot")
    verify_snapshot(run, gpu, "gpu_head_source_snapshot")
    if gpu["origin_identity"] != original["identity"] or gpu["identity"]["feature_identity"] != original["identity"]:
        raise ValueError("Feature provenance mismatch")
    for protocol, filename in ((original, "config.json"), (gpu, "gpu_head_config.json")):
        if p.read_json(run / filename) != protocol["config"] or p.digest(protocol["config"]) != protocol["identity"]["config_sha256"]:
            raise ValueError("Archived configuration mismatch")
    manifest = p.read_json(run / "data_manifest.json")
    if p.digest(manifest["images"]) != original["identity"]["data_sha256"]:
        raise ValueError("Dataset manifest mismatch")
    rows = original["parts"]["val"] + original["parts"]["val25"]
    for split, records in original["parts"].items():
        if [{key: r[key] for key in ("file", "label", "session", "group", "sha256")} for r in records] != manifest["images"][split]:
            raise ValueError("Stored row metadata mismatch")
    _, cache_id = p.validate_cache(run, original)
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    exported = torch.load(run / "head.pt", map_location="cpu", weights_only=True)
    if checkpoint["identity"] != gpu["identity"] or checkpoint["cache_identity"] != cache_id or checkpoint["epoch"] != 15 or checkpoint["device"] != "cuda":
        raise ValueError("Final checkpoint provenance/epoch/device mismatch")
    if exported["identity"] != gpu["identity"] or exported["epoch"] != 15:
        raise ValueError("Exported head provenance mismatch")
    for key in checkpoint["head"]:
        torch.testing.assert_close(exported["head"][key], checkpoint["head"][key], rtol=0, atol=0)
    with (run / "val_combined_probs.csv").open(encoding="utf-8", newline="") as f:
        predictions = list(csv.DictReader(f))
    if [r["file"] for r in predictions] != [r["file"] for r in rows]:
        raise ValueError("Saved prediction order mismatch")
    probabilities = np.array([[float(r["p_heel_lining"]), float(r["p_outsole"])] for r in predictions])
    with np.load(run / "last_validation_reference.npz", allow_pickle=False) as reference:
        np.testing.assert_array_equal(probabilities, reference["probabilities"])
    thresholds = p.read_json(run / "thresholds.json")
    result = p.metrics(p.labels(rows), probabilities, [thresholds["heel_lining"], thresholds["outsole"]])
    fixed = p.metrics(p.labels(rows), probabilities)
    if result != p.read_json(run / "metrics.json")["val_combined"] or fixed != p.read_json(run / "metrics_fixed_0_5.json")["val_combined"]:
        raise ValueError("Recounted metric mismatch")
    if fixed != checkpoint["history"][-1]["val_fixed_0_5"]:
        raise ValueError("Checkpoint validation mismatch")
    if torch.cuda.is_initialized():
        raise RuntimeError("Reporting recovery must not initialize CUDA")
    previous = p.read_json(run / "training_status.json")
    if previous.get("completed_epoch") != 15 or not (run / "report.md").exists():
        raise ValueError("Evaluation artifacts are incomplete")
    recovery = {"time": p.now(), "action": "report-only recovery; no new training or inference",
                "reason": "Epoch/status updater also matched the short metric comparison table",
                "previous_training_status": previous,
                "previous_gpu_head_status": p.read_json(run / "gpu_head_status.json"),
                "checkpoint_sha256": p.file_sha(run / "last.pt"),
                "head_sha256": p.file_sha(run / "head.pt"),
                "metrics_sha256": p.file_sha(run / "metrics.json"),
                "verified": ["sealed source snapshots", "config/data/cache fingerprints",
                             "15-epoch CUDA checkpoint and exported weights", "saved probabilities and both metric tables"],
                "reporting_source_sha256": p.file_sha(__file__)}
    p.write_json(run / "report_finalization.json", recovery)
    snapshot = run / "report_finalization_source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(p.__file__), p.ROOT / "classifier/dino/tools/report_combined.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    with (run / "val_combined_confusion.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *p.ORDER])
        writer.writerows([name, *row] for name, row in zip(p.ORDER, result["confusion_matrix"]))
    complete = {k: v for k, v in previous.items() if k not in ("error", "failed_stage")}
    complete.update(stage="completed", updated_at=p.now(), gpu_released=True, gpu_release_pending=False,
                    report_finalized_at=p.now())
    p.write_json(run / "training_status.json", complete)
    p.write_json(run / "gpu_head_status.json", {"stage": "completed", "time": p.now(), "device": "cuda",
                                               "gpu_released": True, "report_recovery": "report_finalization.json"})
    cache = p.read_json(run / "cache_complete.json")
    report = (run / "report.md").read_text(encoding="utf-8")
    body = report.split("\n", 1)[1].strip()
    body += (f"\n\n**소요 시간:** 특징 추출 {cache['extraction_seconds']/60:.2f}분, GPU 분류층 15 epoch {previous['head_train_seconds']:.2f}초. "
             "학습·GPU 재로딩 검증·최종 val 계산은 완료됐으며, 비교 표를 상태 표로 오인한 문서 갱신 오류만 CPU에서 복구했습니다. 가중치·예측·지표는 변경하지 않았습니다. GPU는 비웠습니다.")
    p.record(run, "GPU 15 epoch 학습·합산 val 평가 완료", body)
    document = p.ROOT / "classifier/dino/EXPERIMENTS.md"
    document.write_text(document.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    with (run / "gpu_head_console.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{p.now()}] Report recovery complete on CPU; artifacts verified; no new training/inference; GPU released.\n")
    p.log(f"Reporting finalized. Feature extraction {cache['extraction_seconds']:.1f}s; head training {previous['head_train_seconds']:.2f}s; BA={result['balanced_accuracy']:.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    finalize(parser.parse_args().run_dir)
