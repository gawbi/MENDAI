"""마지막 가중치의 원본 예측 재현, 좌우 반전 TTA, 두 번의 비교 보고."""
from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import ORDER, seed_worker
from ..evaluate import choose_thresholds, metrics, save_predictions
from ..model import DamageDINO
from ..tools.report_combined import render
from . import core as c


def read_probabilities(path, rows):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    names = [row["file"] for row in records]
    if len(names) != len(set(names)) or set(names) != {row["file"] for row in rows}:
        raise ValueError("예측 파일명 중복·누락")
    mapping = {row["file"]: [float(row["p_heel_lining"]), float(row["p_outsole"])] for row in records}
    result = np.asarray([mapping[row["file"]] for row in rows])
    if result.shape != (len(rows), 2) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("저장 확률 형식 오류")
    return result


def predict_flipped(model, rows, config, batch):
    loader = DataLoader(c.FlippedPhotos(rows, config["input_size"]), batch_size=batch, shuffle=False,
                        num_workers=config["workers"], pin_memory=True, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(config["seed"]))
    values = []
    model.eval()
    with torch.inference_mode():
        for pixels, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixels.to("cuda", non_blocking=True))
            values.append(logits.float().sigmoid().cpu().numpy())
            if torch.cuda.memory_reserved()/1024**2 > config["max_reserved_mib"]:
                raise RuntimeError("TTA GPU 메모리 한도 초과")
    torch.cuda.synchronize()
    return np.concatenate(values)


def tta(chain, which):
    protocol = c.load(chain)
    origin = Path(protocol["origin_run"] if which == "lora" else protocol["plus_run"])
    output = Path(chain) / f"tta_{which}"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        raise ValueError("TTA가 이미 완료되어 덮어쓰지 않음")
    state = c.t.read_json(origin / "training_status.json")
    if state["status"] != "complete":
        raise ValueError("원본 최종 val 평가 미완료")
    run_protocol = c.t.read_json(origin / "protocol.json")
    config = run_protocol["config"]
    parts, checked, _ = c.t.prepare(config, origin, run_protocol["backbone_path"], existing=True)
    if checked["identity"] != (protocol["origin_identity"] if which == "lora" else protocol["plus_identity"]):
        raise ValueError("TTA 원본 identity 불일치")
    backbone_sha = c.sha(Path(run_protocol["backbone_path"]) / "model.safetensors")
    if backbone_sha != protocol["backbone_sha256"]:
        raise ValueError("백본 변경")
    preserved = {name: c.sha(origin / name) for name in ("last.pt", "metrics.json", "thresholds.json", "val_combined_probs.csv", "config.json", "protocol.json")}
    saved = torch.load(origin / "last.pt", map_location="cpu", weights_only=False)
    if saved["identity"] != checked["identity"] or saved["epoch"] != config["epochs"]:
        raise ValueError("TTA 마지막 체크포인트 identity·epoch 불일치")
    selection = c.t.read_json(origin / "selection.json")
    if saved["selection"] != selection:
        raise ValueError("TTA 원본 배치 불일치")
    c.t.gpu_gate(config, initial=True)
    c.t.seed_all(config["seed"])
    model = DamageDINO(config, run_protocol["backbone_path"], selection["checkpointing"]).to("cuda")
    model.load_trainable(saved["model"])
    del saved
    original, flipped, rows = [], [], []
    started = time.perf_counter()
    for split in ("val", "val25"):
        plain, _ = c.t.predict(model, parts[split], config, selection["batch"])
        transformed = predict_flipped(model, parts[split], config, selection["batch"])
        original.append(plain)
        flipped.append(transformed)
        rows.extend(parts[split])
        c.t.log(f"{which} TTA: {split} {len(plain)}장 원본·반전 완료")
    original, flipped = np.concatenate(original), np.concatenate(flipped)
    previous = read_probabilities(origin / "val_combined_probs.csv", rows)
    np.testing.assert_allclose(original, previous, atol=1e-4, rtol=1e-4)
    labels = [ORDER.index(row["label"]) for row in rows]
    old_metrics = c.t.read_json(origin / "metrics.json")["val_combined"]
    reproduced = metrics(labels, original, old_metrics["thresholds"])
    if reproduced["confusion_matrix"] != old_metrics["confusion_matrix"]:
        raise ValueError("원본의 최종 판정 재현 실패")
    averaged = c.mean_probabilities(original, flipped)
    thresholds = choose_thresholds(labels, averaged)
    result = metrics(labels, averaged, thresholds)
    fixed = metrics(labels, averaged, old_metrics["thresholds"])
    save_predictions(output / "original_reproduced_probs.csv", rows, original)
    save_predictions(output / "flipped_probs.csv", rows, flipped)
    save_predictions(output / "val_combined_probs.csv", rows, averaged)
    # 저장한 CSV에서 다시 읽어 지표를 검증한다.
    reread = read_probabilities(output / "val_combined_probs.csv", rows)
    if metrics(labels, reread, thresholds) != result:
        raise ValueError("TTA 저장 확률 재계산 결과 불일치")
    for name, fingerprint in preserved.items():
        if c.sha(origin / name) != fingerprint:
            raise ValueError("TTA 중 원본 저장물 변경")
    c.t.write_json(output / "metrics.json", {"val_combined": result})
    c.t.write_json(output / "metrics_origin_thresholds.json", {"val_combined": fixed})
    c.t.write_json(output / "thresholds.json", {"values": thresholds, "selection": protocol["tta"]["threshold_selection"]})
    c.t.write_json(output / "inference_config.json", {"origin_run": str(origin), "origin_checkpoint_sha256": preserved["last.pt"],
                                                         "model_config": config, "tta": protocol["tta"], "thresholds": thresholds})
    evidence = {"completed_at": c.t.now(), "seconds": time.perf_counter()-started, "source_chain_sha256": protocol["sha256"],
                "origin_identity": checked["identity"], "data_sha256": checked["identity"]["data_sha256"],
                "origin_files_sha256": preserved, "origin_reproduction_max_abs_error": float(np.max(np.abs(original-previous))),
                "original_confusion_reproduced": True, "saved_probabilities_rechecked": True,
                "output_files_sha256": {name: c.sha(output / name) for name in ("metrics.json", "val_combined_probs.csv", "flipped_probs.csv", "inference_config.json")},
                "gpu_release_pending": True, "test_evaluated": False, "gpu_peaks": c.t.peaks()}
    c.t.write_json(output / "complete.json", evidence)
    report = f"# {which} 원본·좌우 반전 TTA\n\n합산 val의 두 sigmoid 점수를 각각 50:50 평균합니다.\n\n{render(result)}\n\n"
    report += f"TTA 판정 기준: {thresholds}. 원본 기준을 유지했을 때 균형 정확도 {fixed['balanced_accuracy']:.2%}, 전체 정확도 {fixed['accuracy']:.2%}. 원본 예측 재현 최대 오차 {evidence['origin_reproduction_max_abs_error']}. test는 미평가입니다.\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    c.t.log(f"{which} TTA 완료: BA={result['balanced_accuracy']:.2%}, accuracy={result['accuracy']:.2%}; GPU 프로세스 종료 예정")


def publish(chain, final=False):
    protocol = c.load(chain)
    specs = [("LoRA", Path(protocol["origin_run"])), ("LoRA + TTA", Path(chain) / "tta_lora")]
    if final:
        specs += [("LoRA+", Path(protocol["plus_run"])), ("LoRA+ + TTA", Path(chain) / "tta_plus")]
    results, evidence = {}, {}
    for name, run in specs:
        result = c.t.read_json(run / "metrics.json")["val_combined"]
        if result["order"] != ORDER or result["images"] != 1319:
            raise ValueError("보고할 합산 val 크기·순서 불일치")
        if "TTA" in name:
            complete = c.t.read_json(run / "complete.json")
            if complete["source_chain_sha256"] != protocol["sha256"] or complete["data_sha256"] != protocol["origin_identity"]["data_sha256"]:
                raise ValueError("TTA 보고 출처 불일치")
            for filename, expected in complete["output_files_sha256"].items():
                if c.sha(run / filename) != expected:
                    raise ValueError("TTA 보고 저장물 해시 불일치")
            for filename, expected in complete["origin_files_sha256"].items():
                source = Path(protocol["origin_run"] if name == "LoRA + TTA" else protocol["plus_run"])
                if c.sha(source / filename) != expected:
                    raise ValueError("TTA 원본 저장물 변경")
        else:
            if c.t.read_json(run / "training_status.json")["status"] != "complete":
                raise ValueError("최종 val 미완료")
        results[name] = result
        evidence[name] = {"run": str(run), "metrics_sha256": c.sha(run / "metrics.json")}
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"
    title = "LoRA·LoRA+와 TTA 네 결과 비교" if final else "LoRA TTA 적용 전후 비교"
    lines = [f"# {title}", "", f"생성 시각: {c.t.now()}. 합산 val **1,319장**, 각 학습의 마지막 15 epoch 가중치입니다.", "",
             "| 모델 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 | both 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        values = [result["balanced_accuracy"], result["accuracy"], *[result["recall"][key] for key in ORDER]]
        lines.append(f"| {name} | " + " | ".join(map(percent, values)) + " |")
    lines += ["", "각 방식의 판정 기준은 같은 합산 val에서 기존과 동일한 격자로 각각 선택했습니다. 같은 val을 개발에 반복 사용한 결과이며 test는 미평가입니다. both 정답이 없어 해당 재현율은 N/A입니다."]
    comparisons = [("LoRA", "LoRA + TTA")]
    if final:
        comparisons += [("LoRA", "LoRA+"), ("LoRA+", "LoRA+ + TTA")]
    differences = {}
    for before, after in comparisons:
        change = {key: 100*(results[after][key]-results[before][key]) for key in ("balanced_accuracy", "accuracy")}
        differences[f"{before} -> {after}"] = change
        lines += ["", f"- **{before} → {after}:** 균형 정확도 {change['balanced_accuracy']:+.2f}%p, 전체 정확도 {change['accuracy']:+.2f}%p."]
    for name, result in results.items():
        lines += ["", f"## {name}", "", f"판정 기준: heel_lining {result['thresholds'][0]:.2f}, outsole {result['thresholds'][1]:.2f}.", "", render(result)]
    report = "\n".join(lines).rstrip() + "\n"
    stem = "final_comparison" if final else "lora_tta_comparison"
    (Path(chain) / f"{stem}.md").write_text(report, encoding="utf-8")
    c.t.write_json(Path(chain) / f"{stem}.json", {"created_at": c.t.now(), "source_chain_sha256": protocol["sha256"],
                                                    "results": results, "evidence": evidence, "differences_percentage_points": differences, "test_evaluated": False})
    text = c.RECORD.read_text(encoding="utf-8")
    begin, end = "<!-- LORA-TTA-CHAIN-START -->", "<!-- LORA-TTA-CHAIN-END -->"
    block = f"{begin}\n\n{report}\n{end}"
    if begin not in text or end not in text:
        raise ValueError("문서의 후속 비교 표식 누락")
    first, rest = text.split(begin, 1)
    _, last = rest.split(end, 1)
    text = first + block + last
    if final:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            cells = line.split("|")
            if line.startswith("| EX-07 |") and len(cells) == 8:
                cells[5] = " 15 / 15 "
                lines[index] = "|".join(cells)
        text = "\n".join(lines)
    c.RECORD.write_text(text.rstrip()+"\n", encoding="utf-8")
    # 이 출력이 끝난 뒤에만 다음 GPU 단계를 시작한다.
    print(report, flush=True)
    return results
