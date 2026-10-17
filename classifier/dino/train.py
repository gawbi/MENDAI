"""DINOv2 LoRA: 점검, 고정 30 epoch 학습, 마지막 가중치의 val 평가."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ORDER, Photos, digest, inventory, seed_worker
from .evaluate import choose_thresholds, metrics, report_markdown, save_predictions
from .model import DamageDINO, optimizer_for

REPO = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).with_name("EXPERIMENTS.md")


def now():
    return datetime.now(timezone(timedelta(hours=9), name="Asia/Seoul")).isoformat(timespec="seconds")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def gpu_status():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    used, total, utilization = map(int, raw.splitlines()[0].split(","))
    return {"used_mib": used, "total_mib": total, "utilization_percent": utilization}


def gpu_gate(config, initial=False):
    status = gpu_status()
    reserved = torch.cuda.memory_reserved() / 1024**2
    if initial and status["total_mib"] - status["used_mib"] < config["min_free_mib"]:
        raise RuntimeError(f"GPU 여유 부족: {status}")
    if not initial and status["used_mib"] - reserved > 2300:
        raise RuntimeError(f"다른 GPU 작업 또는 예상 밖 메모리 사용: {status}, reserved={reserved:.0f}MiB")
    return status


def peaks():
    return {"allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1),
            "reserved_mib": round(torch.cuda.max_memory_reserved()/1024**2, 1)}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def loader(rows, config, batch, augment=False, epoch=0):
    return DataLoader(Photos(rows, config["input_size"], augment), batch_size=batch, shuffle=augment,
                      num_workers=config["workers"], pin_memory=True, persistent_workers=False,
                      generator=torch.Generator().manual_seed(config["seed"] + epoch),
                      worker_init_fn=seed_worker)


def scheduler_for(optimizer, updates_per_epoch, config):
    warmup = updates_per_epoch * config["warmup_epochs"]
    total = updates_per_epoch * config["epochs"]
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1, (step-warmup)/max(1, total-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_epoch(model, dl, opt, criterion, accumulation, scheduler=None, device="cuda", config=None, epoch=0):
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss, samples, updates = 0., 0, 0
    t = time.perf_counter()
    gpu = device == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    for index, (x, y) in enumerate(dl):
        window_start = (index // accumulation) * accumulation
        window_samples = min(accumulation * dl.batch_size, len(dl.dataset) - window_start * dl.batch_size)
        x, y = x.to(device, non_blocking=gpu), y.to(device, non_blocking=gpu)
        with torch.autocast("cuda", dtype=torch.bfloat16) if gpu else nullcontext():
            loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("유한하지 않은 학습 손실")
        (loss * len(x) / window_samples).backward()
        total_loss += loss.detach().float().item() * len(x)
        samples += len(x)
        if (index + 1) % accumulation == 0 or index + 1 == len(dl):
            if gpu and updates == 0:
                lora_grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad]
                if not lora_grads or not any(g is not None and torch.count_nonzero(g).item() for g in lora_grads):
                    raise RuntimeError("LoRA gradient 없음")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("고정 파라미터에 gradient 발생")
                if not all(p.grad is not None for p in model.head.parameters()):
                    raise RuntimeError("분류층 gradient 없음")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"] if config else 1.0, error_if_nonfinite=True)
            opt.step()
            if scheduler:
                scheduler.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
        if gpu and torch.cuda.memory_reserved()/1024**2 > config["max_reserved_mib"]:
            raise RuntimeError("GPU reserved 메모리 한도 초과")
        if gpu and (index + 1) % max(1, 64 // dl.batch_size) == 0:
            elapsed = time.perf_counter()-t
            log(f"epoch={epoch} samples={samples}/{len(dl.dataset)} loss={total_loss/samples:.4f} elapsed={elapsed:.1f}s")
    if gpu:
        torch.cuda.synchronize()
    return {"loss": total_loss/samples, "samples": samples, "optimizer_updates": updates,
            "seconds": time.perf_counter()-t, **(peaks() if gpu else {})}


def predict(model, rows, config, batch, criterion=None):
    model.eval()
    values, loss_sum = [], 0.
    t = time.perf_counter()
    with torch.inference_mode():
        for x, y in loader(rows, config, batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to("cuda", non_blocking=True))
                if criterion is not None:
                    loss_sum += criterion(logits, y.to("cuda", non_blocking=True)).float().item() * len(x)
            values.append(logits.float().sigmoid().cpu().numpy())
            if torch.cuda.memory_reserved()/1024**2 > config["max_reserved_mib"]:
                raise RuntimeError("검증 중 GPU 메모리 한도 초과")
    torch.cuda.synchronize()
    p = np.concatenate(values)
    return p, {"seconds": time.perf_counter()-t, "loss": loss_sum/len(rows) if criterion is not None else None}


def source_hash():
    return digest({p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")})


def prepare(config, run_dir, backbone_path, existing=False):
    parts, data = inventory(REPO / config["dataset_root"])
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": digest(config), "data_sha256": data["sha256"], "source_sha256": source_hash()}
    if (run_dir / "protocol.json").exists():
        protocol = read_json(run_dir / "protocol.json")
        if protocol["identity"] != identity:
            raise RuntimeError("데이터·설정·코드가 바뀌어 기존 실행과 혼합할 수 없음")
    else:
        if existing:
            raise RuntimeError("기존 protocol이 없는 경로")
        protocol = {"created_at": now(), "identity": identity, "config": config, "backbone_path": str(backbone_path) if backbone_path else None,
                    "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "numpy", "pillow")},
                    "python": sys.version, "data_summary": data["summary"],
                    "command": subprocess.list2cmdline([sys.executable] + sys.argv),
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).strip()}
        write_json(run_dir / "protocol.json", protocol)
        write_json(run_dir / "data_manifest.json", data)
        write_json(run_dir / "config.json", config)
    positives = np.asarray([r["target"] for r in parts["train"]]).sum(0)
    pos_weight = (len(parts["train"]) - positives) / positives
    return parts, protocol, torch.tensor(pos_weight, dtype=torch.float32)


def checkpoint(path, model, opt, scheduler, epoch, history, identity, selection):
    value = {"model": model.trainable_state(), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "history": history, "rng": rng_state(), "identity": identity, "selection": selection}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def record(run_dir, status, extra=""):
    """실험 문서의 지정 영역만 갱신한다. 다른 실험 본문은 보존한다."""
    header, footer = "<!-- EX-01-RUNTIME-START -->", "<!-- EX-01-RUNTIME-END -->"
    relative = run_dir.relative_to(REPO).as_posix()
    text = f"{header}\n\n### 실제 실행 기록\n\n**상태:** {status}  \n**기록 시각:** {now()}  \n**run:** `{relative}`\n\n{extra}\n\n{footer}"
    current = RECORD.read_text(encoding="utf-8")
    if header in current:
        start, tail = current.split(header, 1)
        _, end = tail.split(footer, 1)
        current = start + text + end
    else:
        current = current.rstrip() + "\n\n" + text + "\n"
    current = re.sub(r"^(\| EX-01 \|.*\|) [^|]* \|$", lambda m: f"{m.group(1)} {status} |", current, flags=re.MULTILINE)
    current = re.sub(r"^\*\*상태:\*\* .*$", f"**상태:** {status}", current, count=1, flags=re.MULTILINE)
    RECORD.write_text(current, encoding="utf-8")


def smoke(config, run_dir, parts, protocol, pos_weight):
    gpu_gate(config, initial=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 지원 안 됨")
    # 학습 사진만 사용한다. 균일 간격으로 선택해 파일 순서 편중을 줄인다.
    indices = np.linspace(0, len(parts["train"])-1, config["smoke_samples"], dtype=int)
    rows = [parts["train"][i] for i in indices]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to("cuda"))
    attempts, successful = [], []
    first_cp = False
    candidates = [(1, False), (1, True), (2, False), (4, False)]
    for batch, cp in candidates:
        if batch == 1 and cp and successful:
            continue
        if batch > 1:
            if not successful:
                break
            cp = first_cp
        model = opt = scheduler = None
        cleanup()
        try:
            seed_all(config["seed"])
            model = DamageDINO(config, protocol["backbone_path"], cp).to("cuda")
            sentinel = next(p for p in model.backbone.parameters() if not p.requires_grad)
            original = sentinel.detach().clone()
            opt = optimizer_for(model, config)
            accumulation = config["effective_batch"] // batch
            scheduler = scheduler_for(opt, math.ceil(len(rows)/config["effective_batch"]), config)
            log(f"128장 점검 시작: batch={batch}, accumulation={accumulation}, checkpointing={cp}")
            stat = train_epoch(model, loader(rows, config, batch, True), opt, criterion, accumulation, scheduler, config=config)
            if not torch.equal(original, sentinel):
                raise RuntimeError("고정 백본 가중치가 변경됨")
            del sentinel, original
            stat.update(batch=batch, accumulation=accumulation, checkpointing=cp, completed_at=now())
            # 별도로 초기화한 모델로 저장물을 읽어 추론을 확인한다.
            expected, _ = predict(model, rows[:1], config, 1)
            checkpoint(run_dir / "smoke_last.pt", model, opt, scheduler, 1, [], protocol["identity"], stat)
            del model, opt, scheduler
            model = opt = scheduler = None
            cleanup()
            saved = torch.load(run_dir / "smoke_last.pt", map_location="cpu", weights_only=False)
            model = DamageDINO(config, protocol["backbone_path"], cp).to("cuda")
            model.load_trainable(saved["model"])
            actual, _ = predict(model, rows[:1], config, 1)
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            del saved
            stat["reload_max_abs_error"] = float(np.abs(actual-expected).max())
            successful.append(stat)
            attempts.append({"status": "passed", **stat})
            first_cp = successful[0]["checkpointing"]
            log(f"점검 성공: {json.dumps(stat)}")
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            # 구현·gradient·재로딩 오류는 batch 축소로 숨기지 않는다.
            message = str(exc)
            if not isinstance(exc, torch.OutOfMemoryError) and "메모리 한도" not in message:
                raise
            attempts.append({"status": "memory_limit", "batch": batch, "checkpointing": cp, "error": message.splitlines()[0]})
            log(f"메모리 한도: batch={batch}, checkpointing={cp}")
            if batch > 1:
                break
        finally:
            del model, opt, scheduler
            cleanup()
            write_json(run_dir / "smoke_attempts.json", attempts)
    if not successful:
        raise RuntimeError("1022 입력이 메모리 한도 안에 들어오지 않음")
    best = min(successful, key=lambda r: r["seconds"]/r["samples"])
    selection = {k: best[k] for k in ("batch", "accumulation", "checkpointing")}
    selection["seconds_per_train_image"] = best["seconds"]/best["samples"]
    selection["estimated_training_hours"] = selection["seconds_per_train_image"]*len(parts["train"])*config["epochs"]/3600
    selection["identity"] = protocol["identity"]
    write_json(run_dir / "selection.json", selection)
    record(run_dir, "128장 점검 완료 · 본 학습 대기", f"```json\n{json.dumps(selection, ensure_ascii=False, indent=2)}\n```\n\n시간 추정은 점검 기반이며 검증·저장 시간은 별도입니다. test는 미평가입니다.")
    log(f"실행 조합 확정: {selection}")


def run_training(config, run_dir, parts, protocol, pos_weight, resume=False):
    selection = read_json(run_dir / "selection.json")
    if selection["identity"] != protocol["identity"]:
        raise RuntimeError("점검한 코드·데이터·설정과 불일치")
    gpu_gate(config, initial=True)
    seed_all(config["seed"])
    model = DamageDINO(config, protocol["backbone_path"], selection["checkpointing"]).to("cuda")
    opt = optimizer_for(model, config)
    updates_per_epoch = math.ceil(len(parts["train"])/config["effective_batch"])
    scheduler = scheduler_for(opt, updates_per_epoch, config)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to("cuda"))
    history, start_epoch = [], 1
    cp_path = run_dir / "last.pt"
    if cp_path.exists():
        if not resume:
            raise RuntimeError("last.pt가 이미 있음. 재개에는 --resume 필요")
        saved = torch.load(cp_path, map_location="cpu", weights_only=False)
        if saved["identity"] != protocol["identity"] or saved["selection"] != selection:
            raise RuntimeError("체크포인트 실행 조건 불일치")
        model.load_trainable(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        start_epoch, history = saved["epoch"] + 1, saved["history"]
        del saved
    elif resume:
        raise RuntimeError("재개할 last.pt 없음")
    write_json(run_dir / "training_status.json", {"status": "running", "started_at": now(), "pid": os.getpid(), "start_epoch": start_epoch})
    record(run_dir, "학습 중", f"- PID: `{os.getpid()}`\n- 30 epoch, batch {selection['batch']} × 누적 {selection['accumulation']}, checkpointing={selection['checkpointing']}\n- 최종 점수는 학습 완료 후 기록합니다. test는 미평가입니다.")
    log(f"본 학습 시작: epoch {start_epoch}~{config['epochs']}, {selection}, pos_weight={pos_weight.tolist()}")
    for epoch in range(start_epoch, config["epochs"]+1):
        gpu_gate(config)
        t = time.perf_counter()
        train = train_epoch(model, loader(parts["train"], config, selection["batch"], True, epoch), opt, criterion,
                            selection["accumulation"], scheduler, config=config, epoch=epoch)
        if train["optimizer_updates"] != updates_per_epoch:
            raise RuntimeError("optimizer 갱신 횟수 오류")
        validations, validation_probabilities = {}, {}
        for name in ("val", "val25"):
            p, cost = predict(model, parts[name], config, selection["batch"], criterion)
            y = [ORDER.index(r["label"]) for r in parts[name]]
            validations[name] = {**metrics(y, p), **cost}
            validation_probabilities[name] = p
        stat = {"epoch": epoch, "train": train, "validation_fixed_0_5": validations,
                "learning_rates": scheduler.get_last_lr(), "epoch_seconds": time.perf_counter()-t, "finished_at": now(), "gpu_peaks": peaks()}
        history.append(stat)
        checkpoint(cp_path, model, opt, scheduler, epoch, history, protocol["identity"], selection)
        if epoch == config["epochs"]:
            np.savez(run_dir / "last_validation_reference.npz", **validation_probabilities)
        write_json(run_dir / "epochs.json", history)
        write_json(run_dir / "training_status.json", {"status": "running", "pid": os.getpid(), "completed_epoch": epoch, "updated_at": now()})
        ba = {k: round(v["balanced_accuracy"], 4) for k, v in validations.items()}
        log(f"epoch {epoch}/{config['epochs']} 완료: train_loss={train['loss']:.5f}, val_BA_fixed0.5={ba}, seconds={stat['epoch_seconds']:.1f}")
    model.backbone.save_pretrained(run_dir / "lora_adapter", safe_serialization=True)
    torch.save(model.head.state_dict(), run_dir / "head.pt")
    del model, opt, scheduler, criterion
    cleanup()
    write_json(run_dir / "training_status.json", {"status": "trained", "completed_epoch": config["epochs"], "finished_at": now()})
    log("학습 완료. 별도 프로세스에서 last 재로딩 및 최종 val 평가 예정")


def finalize(run_dir):
    protocol = read_json(run_dir / "protocol.json")
    config = protocol["config"]
    parts, checked_protocol, _ = prepare(config, run_dir, protocol["backbone_path"], existing=True)
    selection = read_json(run_dir / "selection.json")
    saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    if saved["epoch"] != config["epochs"] or saved["identity"] != checked_protocol["identity"]:
        raise RuntimeError("완료된 30 epoch 체크포인트가 아님")
    gpu_gate(config, initial=True)
    model = DamageDINO(config, protocol["backbone_path"], selection["checkpointing"]).to("cuda")
    model.load_trainable(saved["model"])
    probabilities, labels = {}, {}
    for name in ("val", "val25"):
        probabilities[name], _ = predict(model, parts[name], config, selection["batch"])
        labels[name] = [ORDER.index(r["label"]) for r in parts[name]]
        save_predictions(run_dir / f"{name}_probs.csv", parts[name], probabilities[name])
        # 학습 종료 시 저장한 추론과 별도 프로세스 결과를 대조한다.
        with np.load(run_dir / "last_validation_reference.npz") as reference:
            np.testing.assert_allclose(probabilities[name], reference[name], atol=1e-4, rtol=1e-4)
        fixed = metrics(labels[name], probabilities[name])
        if fixed["confusion_matrix"] != saved["history"][-1]["validation_fixed_0_5"][name]["confusion_matrix"]:
            raise RuntimeError("별도 프로세스에서 최종 고정 기준 판정을 재현하지 못함")
    thresholds = choose_thresholds(np.concatenate([labels[k] for k in probabilities]), np.concatenate(list(probabilities.values())))
    results = {k: metrics(labels[k], probabilities[k], thresholds) for k in probabilities}
    write_json(run_dir / "thresholds.json", {"values": thresholds, "output_order": config["output_classes"], "selection": "pooled val+val25 balanced accuracy; 0.01..0.99 grid; accuracy/distance/heel/outsole tie-break"})
    write_json(run_dir / "metrics.json", results)
    for name, result in results.items():
        import csv
        with (run_dir / f"{name}_confusion.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred", *ORDER])
            writer.writerows([label, *row] for label, row in zip(ORDER, result["confusion_matrix"]))
    summary = f"공통 기준: 안감 {thresholds[0]:.2f}, 밑창 {thresholds[1]:.2f}. last 가중치, test 미평가.\n\n" + report_markdown(results)
    (run_dir / "report.md").write_text(summary + "\n", encoding="utf-8")
    seconds = sum(r["epoch_seconds"] for r in saved["history"])
    record(run_dir, "30 epoch 및 val 평가 완료", f"- 학습·epoch별 검증 합계: {seconds/3600:.2f}시간 (저장·별도 최종 평가 시간 제외)\n- batch {selection['batch']} × 누적 {selection['accumulation']}, checkpointing={selection['checkpointing']}\n\n" + summary)
    write_json(run_dir / "training_status.json", {"status": "complete", "completed_epoch": config["epochs"], "finished_at": now(), "test_evaluated": False})
    log(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/dinov2b_lora_1022_e30.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--stage", choices=["check", "smoke", "train"], default="check")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((REPO / "runs" / "dino").resolve()):
        raise ValueError("run은 이 저장소의 runs/dino 안에 저장")
    try:
        config = read_json(args.config)
        parts, protocol, pos_weight = prepare(config, run_dir, args.backbone_path)
        log(f"입력 점검 완료: {protocol['data_summary']}, sha256={protocol['identity']['data_sha256']}")
        if args.stage == "smoke":
            smoke(config, run_dir, parts, protocol, pos_weight)
        elif args.stage == "train":
            run_training(config, run_dir, parts, protocol, pos_weight, args.resume)
    except Exception as exc:
        log(f"실패: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if run_dir.exists():
            write_json(run_dir / "failure.json", {"time": now(), "stage": args.stage, "error": f"{type(exc).__name__}: {exc}"})
            record(run_dir, f"{args.stage} 단계 중단", f"`{type(exc).__name__}: {exc}`\n\n완료된 체크포인트와 로그를 확인한 뒤 재개합니다. test는 미평가입니다.")
        raise


if __name__ == "__main__":
    main()
