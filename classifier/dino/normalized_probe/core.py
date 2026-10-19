from __future__ import annotations

import csv
import importlib.metadata
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import digest, inventory, ORDER
from ..evaluate import choose_thresholds, metrics, save_predictions
from ..linear_probe import pipeline as p
from ..model import DamageDINO, pooled_features
from ..tools.report_combined import render, update_experiment_record

ROOT = p.ROOT


def source_files():
    base = Path(__file__).parent
    paths = [*base.glob("*.py"), base / "run_pair.cmd", base.parent / "model.py",
             base.parent / "data.py", base.parent / "evaluate.py",
             base.parent / "linear_probe/pipeline.py", base.parent / "tools/report_combined.py"]
    return {path.relative_to(ROOT).as_posix(): p.file_sha(path) for path in sorted(paths)}


def record(run, state, body):
    protocol = p.read_json(run / "protocol.json")
    experiment = protocol["config"]["experiment"]
    path = ROOT / "classifier/dino/EXPERIMENTS.md"
    update_experiment_record(path, experiment, state,
                             f"**실행 경로:** `{run.relative_to(ROOT).as_posix()}`\n\n{body}")
    completed = p.read_json(run / "training_status.json").get("completed_epoch", 0)
    def update(match):
        cells = match.group(0).split("|")
        if len(cells) == 8:
            cells[5] = f" 15 / {completed} "
        return "|".join(cells)
    text = re.sub(rf"^\| {experiment} \|.*$", update, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def prepare(run, origin, experiment):
    run, origin = run.resolve(), origin.resolve()
    run.relative_to(ROOT / "runs/dino")
    origin.relative_to(ROOT / "runs/dino")
    if (run / "protocol.json").exists():
        return load(run)
    if experiment not in ("EX-04", "EX-05"):
        raise ValueError("Only the authorized EX-04/05 pair is supported")
    old = p.read_json(origin / "protocol.json")
    if p.read_json(origin / "config.json") != old["config"] or digest(old["config"]) != old["identity"]["config_sha256"]:
        raise ValueError("Original configuration mismatch")
    checkpoint_path = origin / ("last_epoch12.pt" if old["config"]["experiment"] == "EX-01" else "last.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["identity"] != old["identity"] or checkpoint["epoch"] != (12 if experiment == "EX-04" else 15):
        raise ValueError("Original checkpoint identity/epoch mismatch")
    if old["config"]["experiment"] != ("EX-01" if experiment == "EX-04" else "EX-02"):
        raise ValueError("Original experiment mismatch")
    parts, manifest = inventory(ROOT / old["config"]["dataset_root"])
    if manifest["sha256"] != old["identity"]["data_sha256"] or manifest["sha256"] != p.DATA_SHA:
        raise ValueError("Frozen train/val dataset mismatch")
    backbone = Path(old["backbone_path"])
    if not backbone.is_absolute():
        backbone = ROOT / backbone
    config = {"experiment": experiment, "origin_run": origin.relative_to(ROOT).as_posix(),
              "feature_size": 2304, "train_views": 4, "epochs": 15, "seed": 0,
              "batch_size": 8, "extract_batch": 4, "workers": 1, "cpu_threads": 1,
              "head_lr": .001, "weight_decay": .01, "warmup_epochs": 1, "clip_grad_norm": 1.,
              "max_reserved_mib": 6656, "min_free_mib": 6656,
              "normalization": "per-feature train-only mean/std across all 4 fixed views; std floor 1e-6",
              "head_initialization": "warm start from original head; standardized head reparameterized to preserve FP32 logits",
              "variants": ["raw_control", "standardized"], "allow_tf32": False,
              "trainable": "linear head only; pretrained backbone and trained LoRA frozen",
              "augmentation": "4 fixed weak views; same per-image epoch rotation; no val augmentation"}
    source = source_files()
    identity = {"config_sha256": digest(config), "source_sha256": digest(source),
                "data_sha256": manifest["sha256"], "origin_identity": old["identity"],
                "checkpoint_sha256": p.file_sha(checkpoint_path),
                "backbone_sha256": p.file_sha(backbone / "model.safetensors")}
    protocol = {"created_at": p.now(), "config": config, "identity": identity, "sources": source,
                "origin_protocol": old, "origin_checkpoint": str(checkpoint_path),
                "origin_metrics": p.read_json(origin / "metrics.json")["val_combined"],
                "backbone_path": str(backbone.resolve()), "parts": parts,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "pillow")}}
    run.mkdir(parents=True, exist_ok=True)
    for name in source:
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    p.write_json(run / "config.json", config)
    p.write_json(run / "data_manifest.json", manifest)
    p.write_json(run / "protocol.json", protocol)
    p.status(run, "prepared", completed_epoch=0, planned_epochs=15, gpu_released=True)
    record(run, "준비 완료", "기존 학습된 LoRA·백본은 고정하고 분류층만 15 epoch 학습합니다. 같은 캐시로 정규화 없는 대조군과 train 평균·표준편차 정규화를 비교합니다. test는 미사용입니다.")
    p.log(f"{experiment} prepared: original epoch {checkpoint['epoch']}; checkpoint {identity['checkpoint_sha256']}")
    return protocol


def load(run):
    protocol = p.read_json(run / "protocol.json")
    if source_files() != protocol["sources"] or digest(protocol["sources"]) != protocol["identity"]["source_sha256"]:
        raise ValueError("Sealed experiment source changed")
    if p.read_json(run / "config.json") != protocol["config"] or digest(protocol["config"]) != protocol["identity"]["config_sha256"]:
        raise ValueError("Sealed configuration changed")
    manifest = p.read_json(run / "data_manifest.json")
    index = {name: [{key: r[key] for key in ("file", "label", "session", "group", "sha256")} for r in rows]
             for name, rows in protocol["parts"].items()}
    if manifest["images"] != index or digest(index) != protocol["identity"]["data_sha256"]:
        raise ValueError("Sealed dataset/row order changed")
    return protocol


def load_original(protocol, device="cuda"):
    path = Path(protocol["origin_checkpoint"])
    if p.file_sha(path) != protocol["identity"]["checkpoint_sha256"]:
        raise ValueError("Original checkpoint changed")
    backbone = Path(protocol["backbone_path"])
    if p.file_sha(backbone / "model.safetensors") != protocol["identity"]["backbone_sha256"]:
        raise ValueError("Original backbone changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["identity"] != protocol["origin_protocol"]["identity"]:
        raise ValueError("Original checkpoint provenance mismatch")
    model = DamageDINO(protocol["origin_protocol"]["config"], backbone)
    model.load_trainable(checkpoint["model"])
    model.requires_grad_(False).eval().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Backbone, trained LoRA and old head must all be frozen")
    return model, checkpoint["model"]["head"]


def infer(model, pixels):
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("Feature extraction requires eval and inference_mode")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.backbone(pixel_values=pixels.to("cuda", non_blocking=True)).last_hidden_state
        features = pooled_features(hidden, model.registers)
        probabilities = model.head(features).float().sigmoid()
    if not torch.isfinite(features).all() or features.shape[1] != 2304:
        raise ValueError("Invalid LoRA features")
    return features.float().cpu().numpy(), probabilities.cpu().numpy()


def extract(run, protocol):
    c, parts = protocol["config"], protocol["parts"]
    if (run / "cache_complete.json").exists():
        validate_cache(run, protocol)
        p.log("Verified existing trained-LoRA cache")
        return
    p.gpu_guard(c, initial=True)
    _, current = inventory(ROOT / protocol["origin_protocol"]["config"]["dataset_root"])
    if current["sha256"] != protocol["identity"]["data_sha256"]:
        raise ValueError("Dataset changed before extraction")
    model, head = load_original(protocol)
    original_state = model.trainable_state()
    size = protocol["origin_protocol"]["config"]["input_size"]
    ds = p.FixedViews(parts["train"][:2], size, 4, c["seed"], True)
    with torch.inference_mode():
        sample = torch.stack([ds[i][0] for i in range(4)])
        start = time.monotonic()
        feature, _ = infer(model, sample)
        single, _ = infer(model, sample[:1])
        batch_difference = float(np.max(np.abs(feature[0] - single[0])))
        # Different SDPA/GEMM batch sizes may cause small numeric differences.
        np.testing.assert_allclose(feature[0], single[0], rtol=1e-3, atol=1e-3)
        smoke_seconds = time.monotonic() - start
    p.write_json(run / "smoke.json", {"time": p.now(), "passed": True, "frozen_parameters": True,
                                      "train_extract_batch": c["extract_batch"], "val_extract_batch": 1,
                                      "batch_vs_single_feature_max_abs": batch_difference,
                                      "seconds": smoke_seconds, "gpu": p.gpu_guard(c)})
    (run / "cache").mkdir(exist_ok=True)
    files, original_predictions = {}, []
    done, total = 0, len(parts["train"]) * 4 + len(parts["val"]) + len(parts["val25"])
    started = time.monotonic()
    p.status(run, "extracting_features", started_at=p.now(), feature_rows=0, feature_rows_total=total, gpu_released=False)
    record(run, "LoRA 특징 추출 중", "기존 학습된 LoRA·백본을 고정해 특징을 추출합니다. train은 고정 증강 4개, val은 무증강입니다. 이후 정규화 없음/있음 분류층을 같은 조건으로 비교합니다.")
    with torch.inference_mode():
        for split, rows in parts.items():
            views = 4 if split == "train" else 1
            ds = p.FixedViews(rows, size, views, c["seed"], split == "train")
            loader = DataLoader(ds, batch_size=c["extract_batch"] if split == "train" else 1,
                                num_workers=c["workers"], pin_memory=True, shuffle=False)
            shape = (len(rows), 4, 2304) if split == "train" else (len(rows), 2304)
            temporary = run / "cache" / f"{split}.partial.npy"
            cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            for pixels, row_ids, view_ids in loader:
                features, probabilities = infer(model, pixels)
                if split == "train":
                    cache[row_ids.numpy(), view_ids.numpy()] = features
                else:
                    cache[row_ids.numpy()] = features
                    original_predictions.append(probabilities)
                done += len(features)
                if done <= 4 or done % 512 == 0 or done == total:
                    elapsed = time.monotonic() - started
                    memory = p.gpu_guard(c)
                    p.status(run, "extracting_features", feature_rows=done, feature_rows_total=total,
                             elapsed_seconds=elapsed, eta_seconds=(total-done)*elapsed/done, gpu=memory)
                    p.log(f"{c['experiment']} features {done:,}/{total:,} ({done/total:.1%}) | elapsed {elapsed/60:.1f}min | ETA {(total-done)*elapsed/done/60:.1f}min | reserved {memory['reserved_mib']}MiB")
            cache.flush()
            del cache
            final = run / "cache" / f"{split}.npy"
            temporary.replace(final)
            files[split] = {"sha256": p.file_sha(final), "shape": list(shape)}
    for key, value in model.trainable_state()["adapter"].items():
        torch.testing.assert_close(value, original_state["adapter"][key], rtol=0, atol=0)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen model acquired gradients")
    val_rows = parts["val"] + parts["val25"]
    probabilities = np.concatenate(original_predictions)
    origin = ROOT / c["origin_run"]
    with (origin / "val_combined_probs.csv").open(encoding="utf-8", newline="") as f:
        expected = {r["file"]: [float(r["p_heel_lining"]), float(r["p_outsole"])] for r in csv.DictReader(f)}
    reference = np.asarray([expected[r["file"]] for r in val_rows])
    np.testing.assert_allclose(probabilities, reference, rtol=0, atol=1e-6)
    reproduced = metrics(p.labels(val_rows), probabilities, protocol["origin_metrics"]["thresholds"])
    if reproduced["confusion_matrix"] != protocol["origin_metrics"]["confusion_matrix"]:
        raise ValueError("Original checkpoint predictions do not reproduce the published confusion matrix")
    p.write_json(run / "original_reproduction.json", {"max_abs_probability_difference": float(np.abs(probabilities-reference).max()),
                                                     "confusion_matches": True, "metrics": reproduced})
    save_predictions(run / "original_probs.csv", val_rows, probabilities)
    p.atomic_torch_save(head, run / "original_head.pt")
    p.write_json(run / "cache_complete.json", {"identity": protocol["identity"], "files": files,
                                               "completed_at": p.now(), "extraction_seconds": time.monotonic()-started,
                                               "original_head_sha256": p.file_sha(run / "original_head.pt"),
                                               "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2})
    p.status(run, "features_complete", feature_rows=total, feature_rows_total=total)
    p.log("Frozen trained-LoRA cache complete; original validation predictions reproduced")


def validate_cache(run, protocol):
    complete = p.read_json(run / "cache_complete.json")
    if complete["identity"] != protocol["identity"] or p.file_sha(run / "original_head.pt") != complete["original_head_sha256"]:
        raise ValueError("Cache/head provenance mismatch")
    arrays = {}
    for split, rows in protocol["parts"].items():
        path = run / "cache" / f"{split}.npy"
        if p.file_sha(path) != complete["files"][split]["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {split}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = (len(rows), 4, 2304) if split == "train" else (len(rows), 2304)
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Cache values/shape/dtype mismatch: {split}")
        arrays[split] = array
    return arrays, digest(complete)


def fit_scaler(train_features):
    if train_features.ndim != 3 or not np.isfinite(train_features).all():
        raise ValueError("Scaler requires finite [train image, view, feature] input")
    return train_features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32), np.maximum(
        train_features.std(axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)


def initialize_head(state, mean, scale, normalized, device):
    head = torch.nn.Linear(len(mean), 2)
    head.load_state_dict(state)
    if normalized:
        with torch.no_grad():
            head.weight.copy_(state["weight"] * scale)
            head.bias.copy_(state["bias"] + state["weight"] @ mean)
    return head.to(device)
