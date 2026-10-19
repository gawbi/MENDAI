"""EX-03: frozen GPU features, then a separate CPU-only linear probe."""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import ORDER, digest, inventory, preprocess
from ..evaluate import choose_thresholds, metrics, save_predictions
from ..tools.report_combined import render, update_experiment_record, write_json

ROOT = Path(__file__).resolve().parents[3]
DATA_SHA = "fd197304188b961d1d2097966789205b7c854d298f90dbc0ea3d09eb1843d856"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sources():
    base = Path(__file__).resolve().parent
    paths = sorted(base.glob("*.py")) + [base.parent / "data.py", base.parent / "evaluate.py",
                                            base.parent / "tools/report_combined.py",
                                            base.parent / "run_linear_probe.cmd"]
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}


def stable_seed(seed, filename, view):
    text = f"sha256_seed_filename_view_v1\0{seed}\0{filename}\0{view}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def view_indices(rows, seed, views, epoch):
    return np.array([(stable_seed(seed, r["file"], "start") + epoch - 1) % views for r in rows])


class FixedViews(Dataset):
    def __init__(self, rows, size, views, seed, augment):
        self.rows, self.size, self.views, self.seed, self.augment = rows, size, views, seed, augment

    def __len__(self):
        return len(self.rows) * self.views

    def __getitem__(self, index):
        row_index, view = divmod(index, self.views)
        row = self.rows[row_index]
        content = Path(row["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed: {row['file']}")
        state = random.getstate()
        try:
            random.seed(stable_seed(self.seed, row["file"], view))
            with Image.open(io.BytesIO(content)) as image:
                pixels = preprocess(image, self.size, self.augment)
        finally:
            random.setstate(state)
        return pixels, row_index, view


def pool(hidden, registers=4):
    patches = hidden[:, 1 + registers:]
    if not patches.shape[1]:
        raise ValueError("Missing patch tokens")
    return torch.cat([hidden[:, 0], patches.mean(1), patches.amax(1)], dim=1)


def cpu_only(config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized():
        raise RuntimeError("CPU stage must start in a new CUDA-disabled process")
    torch.set_num_threads(config["cpu_threads"])


def status(run, stage, **extra):
    current = read_json(run / "training_status.json") if (run / "training_status.json").exists() else {}
    write_json(run / "training_status.json", {**current, "stage": stage, "updated_at": now(), **extra})


def record(run, state, body):
    path = ROOT / "classifier/dino/EXPERIMENTS.md"
    update_experiment_record(path, "EX-03", state,
                             f"**실행 경로:** `{run.relative_to(ROOT).as_posix()}`\n\n{body}")
    completed = read_json(run / "training_status.json").get("completed_epoch", 0)
    def epochs(match):
        cells = match.group(0).split("|")
        if len(cells) != 8:
            return match.group(0)
        cells[5] = f" 15 / {completed} "
        return "|".join(cells)
    path.write_text(re.sub(r"^\| EX-03 \|.*$", epochs, path.read_text(encoding="utf-8"), flags=re.MULTILINE), encoding="utf-8")


def prepare(args):
    run = args.run_dir.resolve()
    run.relative_to(ROOT / "runs/dino")
    if (run / "protocol.json").exists():
        return load_protocol(run)
    if not args.config or not args.backbone_path:
        raise ValueError("First check requires --config and --backbone-path")
    config = read_json(args.config)
    required = {"experiment": "EX-03", "training_mode": "frozen_linear_probe", "input_size": 1024,
                "feature_size": 2304, "train_views": 4, "epochs": 15, "seed": 0,
                "batch_size": 8, "extract_batch": 1, "workers": 1,
                "pooling": "cls_patch_mean_patch_max_excluding_4_registers",
                "view_policy": "sha256_seed_filename_view_v1",
                "output_classes": ["heel_lining", "outsole"], "confusion_order": ORDER}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Configuration differs from approved EX-03 design")
    backbone = args.backbone_path.resolve()
    if backbone.name != config["revision"] or file_sha(backbone / "model.safetensors") != config["backbone_sha256"]:
        raise ValueError("Pretrained backbone revision/hash mismatch")
    log("Checking train/val inventory and image hashes (test excluded)")
    parts, manifest = inventory(ROOT / config["dataset_root"])
    if manifest["sha256"] != DATA_SHA:
        raise ValueError("Frozen dataset fingerprint mismatch")
    run.mkdir(parents=True, exist_ok=True)
    source = sources()
    identity = {"config_sha256": digest(config), "source_sha256": digest(source),
                "data_sha256": manifest["sha256"], "backbone_sha256": config["backbone_sha256"]}
    protocol = {"created_at": now(), "config": config, "backbone_path": str(backbone),
                "identity": identity, "sources": source, "parts": parts,
                "torch": torch.__version__, "numpy": np.__version__,
                "packages": {name: importlib.metadata.version(name) for name in ("transformers", "pillow", "safetensors")},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8")}
    for name in source:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    write_json(run / "config.json", config)
    write_json(run / "data_manifest.json", manifest)
    write_json(run / "protocol.json", protocol)
    status(run, "checked", completed_epoch=0, planned_epochs=config["epochs"], gpu_released=False)
    log(f"Check passed: {identity}; train=4584, pooled val=1319, head=4610 parameters")
    return protocol


def load_protocol(run):
    p = read_json(run / "protocol.json")
    if digest(p["config"]) != p["identity"]["config_sha256"] or read_json(run / "config.json") != p["config"]:
        raise ValueError("Configuration changed after sealing")
    if sources() != p["sources"] or digest(p["sources"]) != p["identity"]["source_sha256"]:
        raise ValueError("Source changed after sealing; use a new run or restore sealed source")
    manifest = read_json(run / "data_manifest.json")
    if digest(manifest["images"]) != p["identity"]["data_sha256"]:
        raise ValueError("Manifest changed after sealing")
    actual = {name: [{k: r[k] for k in ("file", "label", "session", "group", "sha256")} for r in rows]
              for name, rows in p["parts"].items()}
    if actual != manifest["images"]:
        raise ValueError("Feature row index changed after sealing")
    return p


def gpu_memory():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    total, used, util = map(int, result.strip().splitlines()[0].split(","))
    return total, used, util


def gpu_guard(config, initial=False):
    total, used, util = gpu_memory()
    reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_initialized() else 0
    if initial and (total - used < config["min_free_mib"] or util > 20):
        raise RuntimeError(f"GPU busy: free={total-used} MiB, utilization={util}%")
    if not initial and (reserved > config["max_reserved_mib"] or used - reserved > 2300):
        raise RuntimeError(f"GPU memory budget exceeded: used={used}, reserved={reserved:.0f} MiB")
    return {"total_mib": total, "used_mib": used, "reserved_mib": round(reserved), "utilization": util}


def frozen_backbone(protocol):
    c = protocol["config"]
    gpu_guard(c, initial=True)
    backbone = Path(protocol["backbone_path"])
    if file_sha(backbone / "model.safetensors") != c["backbone_sha256"]:
        raise ValueError("Pretrained weights changed")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(backbone), local_files_only=True, attn_implementation="sdpa")
    model.requires_grad_(False).eval().to("cuda")
    config = model.config
    if (config.model_type, config.patch_size, config.hidden_size, config.num_register_tokens) != ("dinov3_vit", 16, 768, 4):
        raise ValueError("Unexpected backbone architecture")
    if any(p.requires_grad for p in model.parameters()) or any("lora" in n.lower() for n, _ in model.named_parameters()):
        raise RuntimeError("Backbone must be fully frozen and contain no LoRA adapters")
    return model


def infer(model, pixels):
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("Frozen feature extraction requires eval + inference_mode")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(pixel_values=pixels.to("cuda", non_blocking=True)).last_hidden_state
        features = pool(hidden)
    if features.shape[1] != 2304 or not torch.isfinite(features).all():
        raise RuntimeError("Invalid features")
    return features.float().cpu().numpy()


def smoke(run, protocol):
    c, parts = protocol["config"], protocol["parts"]
    ds = FixedViews(parts["train"][:2], c["input_size"], c["train_views"], c["seed"], True)
    if not torch.equal(ds[0][0], ds[0][0]):
        raise RuntimeError("Augmentation is not deterministic")
    model = frozen_backbone(protocol)
    sentinel = [p.detach().clone() for p in (next(model.parameters()), list(model.parameters())[-1])]
    features = []
    with torch.inference_mode():
        for index in (0, 1, 4):
            features.append(infer(model, ds[index][0][None]))
        repeated = infer(model, ds[0][0][None])
    np.testing.assert_allclose(features[0], repeated, rtol=0, atol=1e-6)
    if not all(torch.equal(a, b) for a, b in zip(sentinel, (next(model.parameters()), list(model.parameters())[-1]))):
        raise RuntimeError("Frozen parameters changed")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Backbone acquired gradients")
    head = torch.nn.Linear(c["feature_size"], 2)
    head(torch.from_numpy(np.concatenate(features))).square().mean().backward()
    if sum(p.numel() for p in head.parameters()) != 4610 or any(p.grad is None for p in head.parameters()):
        raise RuntimeError("Head gradient check failed")
    summary = {"time": now(), "passed": True, "identity": protocol["identity"],
               "backbone_trainable_parameters": 0, "head_trainable_parameters": 4610,
               "features": [len(features), 2304], "deterministic_features": True,
               "gpu": gpu_guard(c), "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    write_json(run / "smoke.json", summary)
    log(f"GPU smoke passed: {summary}")


def validate_cache(run, protocol):
    complete = read_json(run / "cache_complete.json")
    if complete["identity"] != protocol["identity"]:
        raise ValueError("Feature cache identity mismatch")
    if file_sha(run / "cache_index.json") != complete["index_sha256"]:
        raise ValueError("Cache row metadata checksum mismatch")
    c = protocol["config"]
    arrays = {}
    for split, rows in protocol["parts"].items():
        path = run / "cache" / f"{split}.npy"
        info = complete["files"][split]
        if file_sha(path) != info["sha256"]:
            raise ValueError(f"Feature cache checksum mismatch: {split}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (len(rows), c["train_views"], c["feature_size"]) if split == "train" else (len(rows), c["feature_size"])
        if arr.shape != expected or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError(f"Feature cache shape/dtype/value mismatch: {split}")
        arrays[split] = arr
    return arrays, digest(complete)


def extract(run, protocol):
    if not (run / "smoke.json").exists() or read_json(run / "smoke.json")["identity"] != protocol["identity"]:
        raise RuntimeError("Matching GPU smoke check required")
    if (run / "cache_complete.json").exists():
        validate_cache(run, protocol)
        log("Verified completed feature cache; extraction skipped")
        return
    if (run / "last.pt").exists():
        raise RuntimeError("Refusing to replace a cache used by an existing head")
    c, parts = protocol["config"], protocol["parts"]
    # Recheck actual bytes before starting; no test images are accessed.
    _, current = inventory(ROOT / c["dataset_root"])
    if current["sha256"] != protocol["identity"]["data_sha256"]:
        raise ValueError("Dataset changed before extraction")
    model = frozen_backbone(protocol)
    sentinels = [p.detach().clone() for p in (next(model.parameters()), list(model.parameters())[-1])]
    (run / "cache").mkdir(exist_ok=True)
    started, done = time.monotonic(), 0
    total = len(parts["train"]) * c["train_views"] + len(parts["val"]) + len(parts["val25"])
    status(run, "extracting_features", started_at=now(), feature_rows=0, feature_rows_total=total)
    record(run, "GPU 특징 추출 중", f"LoRA 없는 원본 DINOv3를 고정했습니다. 총 {total:,}개 특징 추출 후 GPU 프로세스를 종료하고 CPU에서 15 epoch 학습·합산 val 평가를 진행합니다. test는 미사용입니다.")
    files, index = {}, {}
    with torch.inference_mode():
        for split, rows in parts.items():
            views = c["train_views"] if split == "train" else 1
            ds = FixedViews(rows, c["input_size"], views, c["seed"], split == "train")
            loader = DataLoader(ds, batch_size=c["extract_batch"], num_workers=c["workers"],
                                pin_memory=True, shuffle=False, persistent_workers=False)
            shape = (len(rows), views, c["feature_size"]) if split == "train" else (len(rows), c["feature_size"])
            temporary = run / "cache" / f"{split}.partial.npy"
            cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            for pixels, row_ids, view_ids in loader:
                values = infer(model, pixels)
                if split == "train":
                    cache[row_ids.numpy(), view_ids.numpy()] = values
                else:
                    cache[row_ids.numpy()] = values
                done += len(values)
                if done == 1 or done % 256 == 0 or done == total:
                    memory = gpu_guard(c)
                    elapsed = time.monotonic() - started
                    remaining = (total - done) * elapsed / done
                    status(run, "extracting_features", feature_rows=done, feature_rows_total=total,
                           elapsed_seconds=round(elapsed, 1), eta_seconds=round(remaining), gpu=memory)
                    log(f"Features {done:,}/{total:,} ({100*done/total:.1f}%) | {elapsed/60:.1f} min elapsed | ETA {remaining/60:.1f} min | reserved {memory['reserved_mib']} MiB")
            cache.flush()
            # Repeated inference validates row placement and float32 persistence.
            check = infer(model, ds[0][0][None])[0]
            np.testing.assert_allclose(cache[0, 0] if split == "train" else cache[0], check, rtol=0, atol=1e-6)
            del cache
            destination = run / "cache" / f"{split}.npy"
            temporary.replace(destination)
            files[split] = {"sha256": file_sha(destination), "shape": list(shape), "dtype": "float32"}
            index[split] = [{k: row[k] for k in ("file", "label", "session", "group", "target", "sha256")}
                            | {"view_seeds": [stable_seed(c["seed"], row["file"], v) for v in range(views)]} for row in rows]
    if any(p.grad is not None for p in model.parameters()) or not all(torch.equal(a, b) for a, b in zip(sentinels, (next(model.parameters()), list(model.parameters())[-1]))):
        raise RuntimeError("Frozen backbone invariant failed")
    write_json(run / "cache_index.json", index)
    complete = {"identity": protocol["identity"], "completed_at": now(), "files": files,
                "index_sha256": file_sha(run / "cache_index.json"), "backbone_trainable_parameters": 0,
                "extraction_seconds": time.monotonic() - started,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    write_json(run / "cache_complete.json", complete)
    status(run, "features_complete", feature_rows=total, feature_rows_total=total)
    log("Feature cache complete. Exiting GPU process before CPU training.")


def labels(rows):
    return np.array([ORDER.index(r["label"]) for r in rows], dtype=int)


def validation(head, arrays, parts):
    rows = parts["val"] + parts["val25"]
    x = torch.from_numpy(np.concatenate([arrays["val"], arrays["val25"]]))
    head.eval()
    with torch.inference_mode():
        logits = head(x).numpy()
        probabilities = torch.from_numpy(logits).sigmoid().numpy()
    return rows, logits, probabilities, metrics(labels(rows), probabilities)


def atomic_torch_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def scheduler_for(optimizer, steps, config):
    warmup, total = steps * config["warmup_epochs"], steps * config["epochs"]
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / (total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train(run, protocol, resume=False):
    c, parts = protocol["config"], protocol["parts"]
    cpu_only(c)
    arrays, cache_id = validate_cache(run, protocol)
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    head = torch.nn.Linear(c["feature_size"], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=c["head_lr"], weight_decay=c["weight_decay"])
    steps = math.ceil(len(parts["train"]) / c["batch_size"])
    scheduler = scheduler_for(optimizer, steps, c)
    targets = torch.tensor([r["target"] for r in parts["train"]], dtype=torch.float32)
    positive = targets.sum(0)
    if (positive == 0).any():
        raise ValueError("Missing positive train targets")
    pos_weight = (len(targets) - positive) / positive
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    start_epoch, history = 1, []
    if (run / "last.pt").exists():
        if not resume:
            raise RuntimeError("Existing checkpoint requires --resume")
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != protocol["identity"] or checkpoint["cache_identity"] != cache_id:
            raise ValueError("Checkpoint/cache identity mismatch")
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start_epoch, history = checkpoint["epoch"] + 1, checkpoint["history"]
    status(run, "training_head_cpu", gpu_released=True, cpu_only=True, pos_weight=pos_weight.tolist())
    record(run, "CPU 분류층 학습 중", "GPU 특징 추출 프로세스가 종료되었습니다. CUDA를 비활성화한 새 프로세스에서 분류층 4,610개 파라미터만 15 epoch 학습합니다. test는 미사용입니다.")
    log(f"CPU head training: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, batch={c['batch_size']}, steps/epoch={steps}, pos_weight={pos_weight.tolist()}")
    training_started = time.monotonic()
    for epoch in range(start_epoch, c["epochs"] + 1):
        started = time.monotonic()
        choices = view_indices(parts["train"], c["seed"], c["train_views"], epoch)
        x = torch.from_numpy(np.array(arrays["train"][np.arange(len(targets)), choices], copy=True))
        permutation = torch.randperm(len(targets), generator=torch.Generator().manual_seed(c["seed"] + epoch))
        total_loss = 0.
        head.train()
        for begin in range(0, len(targets), c["batch_size"]):
            batch = permutation[begin:begin + c["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x[batch]), targets[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), c["clip_grad_norm"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)
        rows, logits, probabilities, result = validation(head, arrays, parts)
        with torch.inference_mode():
            val_loss = criterion(torch.from_numpy(logits), torch.tensor([r["target"] for r in rows], dtype=torch.float32)).item()
        item = {"epoch": epoch, "train_loss": total_loss / len(targets),
                "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                "val_fixed_0_5": result, "seconds": time.monotonic() - started}
        history.append(item)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU stage unexpectedly initialized CUDA")
        atomic_torch_save({"identity": protocol["identity"], "cache_identity": cache_id, "epoch": epoch,
                           "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                           "history": history, "pos_weight": pos_weight.tolist()}, run / "last.pt")
        write_json(run / "history.json", history)
        if epoch == c["epochs"]:
            temporary = run / "last_validation_reference.tmp.npz"
            np.savez(temporary, logits=logits, probabilities=probabilities)
            temporary.replace(run / "last_validation_reference.npz")
        status(run, "training_head_cpu", completed_epoch=epoch, loss=item["train_loss"],
               val_balanced_accuracy_fixed=result["balanced_accuracy"], gpu_released=True)
        log(f"Epoch {epoch}/{c['epochs']} | train loss={item['train_loss']:.5f}, val loss={val_loss:.5f} | pooled val BA@0.5={result['balanced_accuracy']:.4%}, accuracy={result['accuracy']:.4%} | {item['seconds']:.1f}s")
    if not (run / "last_validation_reference.npz").exists() and history:
        _, logits, probabilities, _ = validation(head, arrays, parts)
        np.savez(run / "last_validation_reference.npz", logits=logits, probabilities=probabilities)
    status(run, "head_trained", completed_epoch=c["epochs"], gpu_released=True,
           cpu_train_seconds=time.monotonic() - training_started)


def evaluate(run, protocol):
    c, parts = protocol["config"], protocol["parts"]
    cpu_only(c)
    arrays, cache_id = validate_cache(run, protocol)
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    if checkpoint["identity"] != protocol["identity"] or checkpoint["cache_identity"] != cache_id or checkpoint["epoch"] != c["epochs"]:
        raise ValueError("Final checkpoint identity/epoch mismatch")
    head = torch.nn.Linear(c["feature_size"], 2)
    head.load_state_dict(checkpoint["head"])
    rows, logits, probabilities, fixed = validation(head, arrays, parts)
    with np.load(run / "last_validation_reference.npz", allow_pickle=False) as reference:
        np.testing.assert_allclose(logits, reference["logits"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(probabilities, reference["probabilities"], rtol=0, atol=1e-6)
    if fixed["confusion_matrix"] != checkpoint["history"][-1]["val_fixed_0_5"]["confusion_matrix"]:
        raise ValueError("Reloaded confusion matrix differs")
    thresholds = choose_thresholds(labels(rows), probabilities)
    result = metrics(labels(rows), probabilities, thresholds)
    write_json(run / "metrics.json", {"val_combined": result})
    write_json(run / "metrics_fixed_0_5.json", {"val_combined": fixed})
    write_json(run / "thresholds.json", {"heel_lining": thresholds[0], "outsole": thresholds[1],
                                        "selected_on": "pooled val, 99x99 grid", "test_used": False})
    save_predictions(run / "val_combined_probs.csv", rows, probabilities)
    atomic_torch_save({"head": head.state_dict(), "identity": protocol["identity"], "epoch": c["epochs"]}, run / "head.pt")
    write_json(run / "inference_config.json", {"config": c, "identity": protocol["identity"],
                                               "thresholds": thresholds, "lora": False, "backbone_frozen": True,
                                               "backbone_mode": "eval; inference_mode; SDPA; CUDA bf16 autocast",
                                               "head_dtype": "float32", "reload_verified_atol": 1e-6})
    comparison = ["| 실험 | 균형 정확도 | 전체 정확도 |", "|---|---:|---:|"]
    for reference in c["reference_runs"]:
        path = ROOT / reference / "metrics.json"
        if path.exists():
            old = read_json(path)["val_combined"]
            comparison.append(f"| {Path(reference).name.split('_')[0]} | {old['balanced_accuracy']:.2%} | {old['accuracy']:.2%} |")
    comparison.append(f"| EX-03 | {result['balanced_accuracy']:.2%} | {result['accuracy']:.2%} |")
    report = (f"# EX-03 DINOv3 고정 특징 + 선형 분류층\n\n완료: {now()}. 15 epoch의 마지막 분류층을 사용했습니다. GPU 특징 추출 후 CPU에서 학습·평가했습니다. test는 미사용입니다.\n\n"
              + render(result) + f"\n\n판정 기준: heel_lining {thresholds[0]:.2f}, outsole {thresholds[1]:.2f}. 합산 val에서 선택한 기준으로 같은 val을 평가한 개발 결과입니다. both 정답은 0장이므로 재현율은 N/A이며 균형 정확도는 나머지 3개 클래스의 평균입니다.\n\n"
              + "\n".join(comparison) + "\n\n고정 4-view 증강·백본 eval 모드·CPU float32 분류층은 EX-02와 다릅니다. 이 비교만으로 LoRA의 효과를 단정하지 않습니다. 분류층 재로딩 후 logits/확률 오차 1e-6 이내 및 혼동행렬 일치를 확인했습니다.\n")
    (run / "report.md").write_text(report, encoding="utf-8")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU evaluation initialized CUDA")
    status(run, "completed", completed_epoch=c["epochs"], completed_at=now(), gpu_released=True,
           test_used=False, final_balanced_accuracy=result["balanced_accuracy"], final_accuracy=result["accuracy"])
    record(run, "15 epoch 학습·합산 val 평가 완료", report.split("\n", 1)[1].strip())
    log(f"EX-03 complete | BA={result['balanced_accuracy']:.4%}, accuracy={result['accuracy']:.4%}, thresholds={thresholds}; GPU released, test unused")


def execute(args):
    run = args.run_dir.resolve()
    run.relative_to(ROOT / "runs/dino")
    protocol = prepare(args) if args.stage == "check" else load_protocol(run)
    try:
        if args.stage == "check":
            cpu_only(protocol["config"])
        elif args.stage == "smoke":
            smoke(run, protocol)
        elif args.stage == "extract":
            extract(run, protocol)
        elif args.stage == "train":
            train(run, protocol, args.resume)
        else:
            evaluate(run, protocol)
    except Exception as exc:
        if args.stage != "check":
            status(run, "failed", failed_stage=args.stage, error=f"{type(exc).__name__}: {exc}")
            record(run, f"{args.stage} 단계 중단", f"`{type(exc).__name__}: {exc}`. test는 미사용입니다. 저장된 산출물을 보존합니다.")
        raise
