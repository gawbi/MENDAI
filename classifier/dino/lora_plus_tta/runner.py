"""선행 실행 성공을 기다리고 GPU 단계를 겹치지 않게 연결한다."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

from . import core as c
from .evaluation import publish, tta


def status(chain, stage, **extra):
    c.t.write_json(Path(chain) / "status.json", {"stage": stage, "updated_at": c.t.now(), "pid": os.getpid(),
                                                "test_evaluated": False, **extra})


def pid_alive(pid):
    if os.name == "nt":
        output = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
        return any(len(row) > 1 and row[1] == str(pid) for row in csv.reader(output.decode(errors="replace").splitlines()))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_origin(chain, protocol):
    origin = Path(protocol["origin_run"])
    pid = protocol["wait_for_pid"]
    status(chain, "waiting_for_origin", wait_for_pid=pid, gpu_work_started=False)
    c.t.log(f"선행 EX-06 학습·최종 val·비교 성공 종료 대기: wrapper PID {pid}")
    while pid_alive(pid):
        time.sleep(15)
    # PID가 사라졌다는 이유만으로 성공이라고 가정하지 않는다.
    console = (origin / "console.log").read_text(encoding="utf-8")
    if not re.search(r"\[exit 0\]\s*$", console):
        raise RuntimeError("EX-06 wrapper가 성공 종료하지 않음; 후속 GPU 실행 중단")
    state = c.t.read_json(origin / "training_status.json")
    report = c.t.read_json(origin / "combined_reporting.json")
    if state["status"] != "complete" or state["completed_epoch"] != 15 or not report["predictions_rechecked"]:
        raise RuntimeError("EX-06 최종 val·저장 예측 검증 미완료")
    c.load(chain)
    c.t.log("EX-06 학습·최종 val·기존 모델 비교 성공 확인")


def wait_gpu(chain, protocol, stage):
    config = protocol["config"]
    waiting = False
    while True:
        gpu = c.t.gpu_status()
        if gpu["total_mib"] - gpu["used_mib"] >= config["min_free_mib"] and gpu["utilization_percent"] <= 20:
            return gpu
        if not waiting:
            c.t.log(f"{stage}: GPU 여유 대기 {gpu}")
            waiting = True
        status(chain, "waiting_for_gpu", next_stage=stage, gpu=gpu)
        time.sleep(15)


def child(chain, protocol, stage, logfile, gpu=True, resume=False):
    c.load(chain)
    if gpu:
        wait_gpu(chain, protocol, stage)
    command = [sys.executable, "-u", "-m", "classifier.dino.lora_plus_tta", "--stage", stage, "--chain-dir", str(chain)]
    if resume:
        command.append("--resume")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if gpu else "-1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a" if resume else "w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        status(chain, stage, active_child_pid=process.pid, active_log=str(logfile), gpu_work_started=gpu)
        c.t.log(f"단계 시작: {stage}, PID={process.pid}, log={logfile}")
        code = process.wait()
    if code:
        raise RuntimeError(f"{stage} 프로세스 실패 (exit {code}): {logfile}")
    if stage in ("tta-lora", "tta-plus"):
        folder = "tta_lora" if stage == "tta-lora" else "tta_plus"
        complete = c.t.read_json(Path(chain) / folder / "complete.json")
        complete.update(gpu_release_pending=False, gpu_released=True)
        c.t.write_json(Path(chain) / folder / "complete.json", complete)
    c.t.log(f"단계 완료: {stage}")


def execute_gpu_stage(chain, stage, resume=False):
    protocol = c.load(chain)
    plus = Path(protocol["plus_run"])
    if stage in ("tta-lora", "tta-plus"):
        tta(chain, "lora" if stage == "tta-lora" else "plus")
        return
    config = protocol["config"]
    origin = c.t.read_json(Path(protocol["origin_run"]) / "protocol.json")
    parts, run_protocol, weights = c.t.prepare(config, plus, origin["backbone_path"], existing=True)
    if run_protocol["identity"] != protocol["plus_identity"]:
        raise ValueError("LoRA+ identity 불일치")
    if c.sha(Path(origin["backbone_path"]) / "model.safetensors") != protocol["backbone_sha256"]:
        raise ValueError("LoRA+ 사전학습 백본 변경")
    if stage == "smoke-plus":
        if (plus / "selection.json").exists():
            raise ValueError("LoRA+ 점검을 중복 실행하지 않음")
        c.t.seed_all(config["seed"])
        model = c.t.DamageDINO(config, origin["backbone_path"])
        c.t.write_json(plus / "optimizer_manifest.json", c.optimizer_manifest(model, config))
        del model
        original_optimizer = c.t.optimizer_for
        try:
            c.t.optimizer_for = c.optimizer
            c.t.smoke(config, plus, parts, run_protocol, weights)
        finally:
            c.t.optimizer_for = original_optimizer
        selection = c.t.read_json(plus / "selection.json")
        if any(selection[key] != protocol["selection"][key] for key in ("batch", "accumulation", "checkpointing")):
            raise ValueError("EX-06과 LoRA+의 배치 조건이 다름")
    elif stage == "train-plus":
        original_optimizer = c.t.optimizer_for
        try:
            c.t.optimizer_for = c.optimizer
            c.t.run_training(config, plus, parts, run_protocol, weights, resume)
        finally:
            c.t.optimizer_for = original_optimizer
    elif stage == "evaluate-plus":
        c.t.finalize(plus)
    else:
        raise ValueError(f"알 수 없는 GPU 단계: {stage}")


def queue(chain, resume=False):
    chain = c.safe_run(chain)
    lock = chain / "controller.lock"
    if lock.exists():
        previous_pid = int(lock.read_text(encoding="ascii"))
        if pid_alive(previous_pid):
            raise RuntimeError(f"후속 controller가 이미 실행 중: PID {previous_pid}")
        if not resume:
            raise RuntimeError("이전 controller 기록이 있음; --resume으로 명시 재개 필요")
        previous = c.t.read_json(chain / "status.json")
        child_pid = previous.get("active_child_pid")
        while child_pid and pid_alive(child_pid):
            time.sleep(15)
        lock.unlink()
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        _queue(chain, resume)
    finally:
        lock.unlink(missing_ok=True)


def _queue(chain, resume=False):
    chain = c.safe_run(chain)
    protocol = c.load(chain)
    plus = Path(protocol["plus_run"])
    try:
        wait_origin(chain, protocol)
        if not (chain / "tta_lora/complete.json").exists():
            child(chain, protocol, "tta-lora", chain / "tta_lora/console.log")
        status(chain, "publishing_lora_tta_comparison", gpu_released=True)
        publish(chain, final=False)
        c.t.log("일반 LoRA TTA 전후 비교 출력 완료. 승인된 LoRA+ 실험을 이어 시작합니다.")
        if not (plus / "selection.json").exists():
            child(chain, protocol, "smoke-plus", plus / "smoke.log")
        plus_state = c.t.read_json(plus / "training_status.json") if (plus / "training_status.json").exists() else {}
        if plus_state.get("status") not in ("trained", "complete"):
            has_checkpoint = (plus / "last.pt").exists()
            if has_checkpoint and not resume:
                raise RuntimeError("LoRA+ 체크포인트가 이미 있음; --resume으로 명시 재개 필요")
            child(chain, protocol, "train-plus", plus / "console.log", resume=has_checkpoint)
        plus_state = c.t.read_json(plus / "training_status.json")
        if plus_state.get("status") != "complete":
            child(chain, protocol, "evaluate-plus", plus / "evaluation_raw.log")
        if not (plus / "combined_reporting.json").exists():
            status(chain, "combining_plus_val", gpu_released=True)
            from ..tools.report_combined import refresh_report
            refresh_report(plus)
        if not (chain / "tta_plus/complete.json").exists():
            child(chain, protocol, "tta-plus", chain / "tta_plus/console.log")
        status(chain, "publishing_final_comparison", gpu_released=True)
        publish(chain, final=True)
        status(chain, "completed", completed_at=c.t.now(), gpu_released=True, final_report=str(chain / "final_comparison.md"))
        c.t.log("승인한 네 결과 비교 완료. 모든 GPU 자식 프로세스 종료. test 미평가.")
    except Exception as exc:
        status(chain, "failed", error=f"{type(exc).__name__}: {exc}", gpu_released=True)
        c.t.log(f"후속 실행 중단: {type(exc).__name__}: {exc}")
        raise
