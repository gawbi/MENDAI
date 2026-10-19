"""GPU head training from a separately sealed, unchanged frozen-feature cache."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .. import pipeline as p


def source_files():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.append(Path(__file__).parent / "run_gpu_head.cmd")
    return {path.relative_to(p.ROOT).as_posix(): p.file_sha(path) for path in paths}


def prepare(run):
    original = p.load_protocol(run)
    if (run / "last.pt").exists():
        raise RuntimeError("Head already trained; use a separate run for a device change")
    path = run / "gpu_head_protocol.json"
    if path.exists():
        return load(run)
    config = {**original["config"], "head_device": "cuda", "head_dtype": "float32",
              "allow_tf32": False, "evaluation_device": "cuda"}
    sources = source_files()
    protocol = {"created_at": p.now(), "reason": "User requested GPU head training and validation",
                "origin_identity": original["identity"], "config": config, "sources": sources,
                "identity": {"feature_identity": original["identity"],
                             "config_sha256": p.digest(config), "source_sha256": p.digest(sources)}}
    for name in sources:
        target = run / "gpu_head_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((p.ROOT / name).read_bytes())
    p.write_json(run / "gpu_head_config.json", config)
    p.write_json(path, protocol)
    p.write_json(run / "gpu_head_status.json", {"stage": "prepared", "time": p.now(), "device": "cuda"})
    p.log("GPU continuation sealed. Original extraction source/config/cache identity preserved.")
    return original, protocol


def load(run):
    original = p.load_protocol(run)
    protocol = p.read_json(run / "gpu_head_protocol.json")
    if protocol["origin_identity"] != original["identity"] or protocol["identity"]["feature_identity"] != original["identity"]:
        raise ValueError("Original feature provenance changed")
    if protocol["sources"] != source_files() or p.digest(protocol["sources"]) != protocol["identity"]["source_sha256"]:
        raise ValueError("GPU continuation source changed")
    if p.read_json(run / "gpu_head_config.json") != protocol["config"] or p.digest(protocol["config"]) != protocol["identity"]["config_sha256"]:
        raise ValueError("GPU continuation configuration changed")
    return original, protocol


def wait_for_extraction(run, pid):
    if os.name != "nt":
        raise RuntimeError("Detached extraction wait is implemented for Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87 and (run / "cache_complete.json").exists():
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        p.log(f"Waiting for existing GPU extraction PID {pid}; CPU continuation disabled")
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                code = ctypes.c_uint32()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if code.value != 0:
                    raise RuntimeError(f"Extraction exited with code {code.value}")
                break
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    if not (run / "cache_complete.json").exists():
        raise RuntimeError("Extraction exited without a complete cache")


def setup_device(config, device):
    torch.set_num_threads(config["cpu_threads"])
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU head requested but CUDA unavailable")
        # Allow utilization from the just-exited extractor to settle.
        for attempt in range(30):
            try:
                p.gpu_guard(config, initial=True)
                break
            except RuntimeError:
                if attempt == 29:
                    raise
                time.sleep(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return torch.device(device)


def gpu_smoke(train_features, dimension):
    torch.manual_seed(0)
    head = torch.nn.Linear(dimension, 2).to("cuda")
    before = head.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    loss = head(train_features[:4, 0]).square().mean()
    loss.backward()
    if sum(v.numel() for v in head.parameters()) != 4610 or any(v.grad is None or v.grad.device.type != "cuda" for v in head.parameters()):
        raise RuntimeError("GPU head gradient check failed")
    optimizer.step()
    if torch.equal(before, head.weight):
        raise RuntimeError("GPU optimizer did not update the head")
    reference = head(train_features[:4, 0]).detach()
    restored = torch.nn.Linear(dimension, 2).to("cuda")
    restored.load_state_dict(head.state_dict())
    torch.testing.assert_close(restored(train_features[:4, 0]), reference, rtol=0, atol=1e-6)
    return {"passed": True, "time": p.now(), "device": "cuda", "trainable_parameters": 4610,
            "gradient_device": "cuda", "optimizer_update": True, "reload_atol": 1e-6}


def validation(head, val_x, rows):
    head.eval()
    with torch.inference_mode():
        logits = head(val_x)
        probabilities = logits.sigmoid()
    return logits.cpu().numpy(), probabilities.cpu().numpy()


def train(run, original, protocol, resume=False, device="cuda"):
    c, parts = protocol["config"], original["parts"]
    arrays, cache_id = p.validate_cache(run, original)
    target_device = setup_device(c, device)
    all_features = torch.from_numpy(np.array(arrays["train"], copy=True)).to(target_device)
    rows = parts["val"] + parts["val25"]
    val_x = torch.from_numpy(np.concatenate([arrays["val"], arrays["val25"]])).to(target_device)
    val_targets = torch.tensor([r["target"] for r in rows], dtype=torch.float32, device=target_device)
    if device == "cuda":
        p.write_json(run / "gpu_head_smoke.json", gpu_smoke(all_features, c["feature_size"]))
    # Initialize on CPU with the same seed, then place the head on the chosen device.
    torch.manual_seed(c["seed"])
    head = torch.nn.Linear(c["feature_size"], 2).to(target_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=c["head_lr"], weight_decay=c["weight_decay"])
    steps = math.ceil(len(parts["train"]) / c["batch_size"])
    scheduler = p.scheduler_for(optimizer, steps, c)
    targets = torch.tensor([r["target"] for r in parts["train"]], dtype=torch.float32, device=target_device)
    positive = targets.sum(0)
    if (positive == 0).any():
        raise ValueError("Missing positive train targets")
    pos_weight = (len(targets) - positive) / positive
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epoch_start, history = 1, []
    if (run / "last.pt").exists():
        if not resume:
            raise RuntimeError("Existing head checkpoint requires --resume")
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != protocol["identity"] or checkpoint["cache_identity"] != cache_id or checkpoint["device"] != device:
            raise ValueError("GPU checkpoint/cache/device mismatch")
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        epoch_start, history = checkpoint["epoch"] + 1, checkpoint["history"]
    started_training = time.monotonic()
    p.status(run, f"training_head_{device}", gpu_released=device != "cuda", cpu_only=device == "cpu",
             head_device=device, pos_weight=pos_weight.tolist(), gpu_head_identity=protocol["identity"])
    p.record(run, "GPU 분류층 학습 중", "사용자 요청으로 분류층 학습과 최종 val 추론도 GPU로 변경했습니다. 추출 캐시·원본 해시는 보존하고 변경 설정·소스를 `gpu_head_protocol.json`에 별도로 기록했습니다. 백본 고정·LoRA 없음·15 epoch·test 보류는 유지합니다.")
    p.log(f"{device.upper()} head training: {sum(v.numel() for v in head.parameters())} parameters; {steps} steps/epoch; float32; TF32 disabled")
    image_ids = torch.arange(len(targets), device=target_device)
    for epoch in range(epoch_start, c["epochs"] + 1):
        started = time.monotonic()
        choices = torch.from_numpy(p.view_indices(parts["train"], c["seed"], c["train_views"], epoch)).to(target_device)
        x = all_features[image_ids, choices]
        permutation = torch.randperm(len(targets), generator=torch.Generator().manual_seed(c["seed"] + epoch)).to(target_device)
        head.train()
        total_loss = 0.
        for begin in range(0, len(targets), c["batch_size"]):
            batch = permutation[begin:begin + c["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x[batch]), targets[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite GPU head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), c["clip_grad_norm"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)
        logits, probabilities = validation(head, val_x, rows)
        result = p.metrics(p.labels(rows), probabilities)
        with torch.inference_mode():
            val_loss = criterion(torch.from_numpy(logits).to(target_device), val_targets).item()
        item = {"epoch": epoch, "train_loss": total_loss / len(targets), "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"], "val_fixed_0_5": result,
                "seconds": time.monotonic() - started, "device": device}
        history.append(item)
        if device == "cuda":
            p.gpu_guard(c)
        p.atomic_torch_save({"identity": protocol["identity"], "cache_identity": cache_id,
                             "device": device, "epoch": epoch,
                             "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "torch_rng": torch.get_rng_state(), "history": history,
                             "pos_weight": pos_weight.tolist()}, run / "last.pt")
        p.write_json(run / "history.json", history)
        if epoch == c["epochs"]:
            temporary = run / "last_validation_reference.tmp.npz"
            np.savez(temporary, logits=logits, probabilities=probabilities)
            temporary.replace(run / "last_validation_reference.npz")
        p.status(run, f"training_head_{device}", completed_epoch=epoch, loss=item["train_loss"],
                 val_balanced_accuracy_fixed=result["balanced_accuracy"], gpu_released=device != "cuda")
        p.log(f"Epoch {epoch}/{c['epochs']} [{device}] | train loss={item['train_loss']:.5f}, val loss={val_loss:.5f} | pooled val BA@0.5={result['balanced_accuracy']:.4%}, accuracy={result['accuracy']:.4%} | {item['seconds']:.1f}s")
    if not (run / "last_validation_reference.npz").exists():
        logits, probabilities = validation(head, val_x, rows)
        np.savez(run / "last_validation_reference.npz", logits=logits, probabilities=probabilities)
    p.status(run, "head_trained", completed_epoch=c["epochs"], head_device=device,
             head_train_seconds=time.monotonic() - started_training)


def evaluate(run, original, protocol, device="cuda"):
    c, parts = protocol["config"], original["parts"]
    arrays, cache_id = p.validate_cache(run, original)
    target_device = setup_device(c, device)
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    if checkpoint["identity"] != protocol["identity"] or checkpoint["cache_identity"] != cache_id or checkpoint["epoch"] != c["epochs"] or checkpoint["device"] != device:
        raise ValueError("Final GPU checkpoint/cache/epoch/device mismatch")
    head = torch.nn.Linear(c["feature_size"], 2).to(target_device)
    head.load_state_dict(checkpoint["head"])
    rows = parts["val"] + parts["val25"]
    val_x = torch.from_numpy(np.concatenate([arrays["val"], arrays["val25"]])).to(target_device)
    logits, probabilities = validation(head, val_x, rows)
    with np.load(run / "last_validation_reference.npz", allow_pickle=False) as reference:
        np.testing.assert_allclose(logits, reference["logits"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(probabilities, reference["probabilities"], rtol=0, atol=1e-6)
    fixed = p.metrics(p.labels(rows), probabilities)
    if fixed["confusion_matrix"] != checkpoint["history"][-1]["val_fixed_0_5"]["confusion_matrix"]:
        raise ValueError("GPU reloaded confusion matrix differs")
    thresholds = p.choose_thresholds(p.labels(rows), probabilities)
    result = p.metrics(p.labels(rows), probabilities, thresholds)
    p.write_json(run / "metrics.json", {"val_combined": result})
    p.write_json(run / "metrics_fixed_0_5.json", {"val_combined": fixed})
    p.write_json(run / "thresholds.json", {"heel_lining": thresholds[0], "outsole": thresholds[1],
                                          "selected_on": "pooled val, 99x99 grid", "test_used": False})
    p.save_predictions(run / "val_combined_probs.csv", rows, probabilities)
    p.atomic_torch_save({"head": {k: v.cpu() for k, v in head.state_dict().items()}, "identity": protocol["identity"],
                         "epoch": c["epochs"]}, run / "head.pt")
    p.write_json(run / "inference_config.json", {"config": c, "identity": protocol["identity"],
                                                 "feature_identity": original["identity"], "thresholds": thresholds,
                                                 "lora": False, "backbone_frozen": True,
                                                 "backbone_mode": "eval; inference_mode; SDPA; CUDA bf16 autocast",
                                                 "head_device": device, "head_dtype": "float32", "allow_tf32": False,
                                                 "reload_verified_atol": 1e-6})
    comparison = ["| 실험 | 균형 정확도 | 전체 정확도 |", "|---|---:|---:|"]
    for reference in c["reference_runs"]:
        old = p.read_json(p.ROOT / reference / "metrics.json")["val_combined"]
        comparison.append(f"| {Path(reference).name.split('_')[0]} | {old['balanced_accuracy']:.2%} | {old['accuracy']:.2%} |")
    comparison.append(f"| EX-03 | {result['balanced_accuracy']:.2%} | {result['accuracy']:.2%} |")
    report = (f"# EX-03 DINOv3 고정 특징 + 선형 분류층\n\n완료: {p.now()}. 15 epoch 마지막 분류층 사용. 특징 추출·분류층 학습·최종 val 추론 장치는 GPU입니다. test는 미사용입니다.\n\n"
              + p.render(result) + f"\n\n판정 기준: heel_lining {thresholds[0]:.2f}, outsole {thresholds[1]:.2f}. 합산 val에서 선택한 기준으로 같은 val을 평가한 개발 결과입니다. both 정답 0장은 N/A이며 균형 정확도는 나머지 3개 클래스의 평균입니다.\n\n"
              + "\n".join(comparison) + "\n\n고정 4-view 증강·백본 eval 모드·float32 분류층은 EX-02와 다릅니다. LoRA의 효과를 단정하지 않습니다. 별도 GPU 프로세스에서 분류층을 재로딩해 logits/확률 오차 1e-6 이내와 혼동행렬 일치를 확인했습니다. CPU 계획에서 GPU로 바꾼 사용자 결정·설정·소스는 gpu_head_protocol.json에 보존했습니다.\n")
    (run / "report.md").write_text(report, encoding="utf-8")
    p.status(run, "completed", completed_epoch=c["epochs"], completed_at=p.now(), head_device=device,
             gpu_release_pending=True, gpu_released=False, test_used=False,
             final_balanced_accuracy=result["balanced_accuracy"], final_accuracy=result["accuracy"])
    p.record(run, "GPU 15 epoch 학습·합산 val 평가 완료", report.split("\n", 1)[1].strip())
    p.log(f"EX-03 complete [{device}] | BA={result['balanced_accuracy']:.4%}, accuracy={result['accuracy']:.4%}, thresholds={thresholds}; exiting GPU process, test unused")


def execute(args):
    run = args.run_dir.resolve()
    run.relative_to(p.ROOT / "runs/dino")
    original, protocol = prepare(run) if args.stage == "prepare" else load(run)
    try:
        if args.stage == "prepare":
            return
        if args.stage == "continue":
            p.write_json(run / "gpu_head_status.json", {"stage": "waiting_for_extraction", "time": p.now(),
                                                         "wait_for_pid": args.wait_for_pid, "device": "cuda"})
            if args.wait_for_pid:
                wait_for_extraction(run, args.wait_for_pid)
            p.validate_cache(run, original)
            for stage in ("train", "evaluate"):
                command = [sys.executable, "-u", "-m", "classifier.dino.linear_probe.gpu_head",
                           "--stage", stage, "--run-dir", str(run)]
                if stage == "train" and args.resume:
                    command.append("--resume")
                subprocess.run(command, check=True, cwd=p.ROOT)
            # Both child processes exited, so their CUDA contexts have been released.
            p.status(run, "completed", gpu_released=True, gpu_release_pending=False)
            p.write_json(run / "gpu_head_status.json", {"stage": "completed", "time": p.now(), "device": "cuda",
                                                         "gpu_released": True})
        elif args.stage == "train":
            train(run, original, protocol, args.resume)
        else:
            evaluate(run, original, protocol)
    except Exception as exc:
        p.write_json(run / "gpu_head_status.json", {"stage": "failed", "time": p.now(), "error": f"{type(exc).__name__}: {exc}"})
        # Avoid racing the extractor's status while it is still running.
        if (run / "cache_complete.json").exists():
            p.status(run, "failed", failed_stage=f"gpu_head_{args.stage}", error=f"{type(exc).__name__}: {exc}")
            p.record(run, "GPU 분류층 단계 중단", f"`{type(exc).__name__}: {exc}`. 특징 캐시는 보존하며 test는 미사용입니다.")
        raise
