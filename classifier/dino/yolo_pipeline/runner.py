"""기존 DINO wrapper의 성공 종료 후 7~9번을 순차 실행하고 종료한다."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import time

from . import common as c


def alive(pid):
    output = subprocess.check_output(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"])
    return any(len(row) > 1 and row[1] == str(pid) for row in csv.reader(output.decode(errors="replace").splitlines()))


def predecessor_complete(protocol, wrapper_log):
    state = c.read(c.DINO_CHAIN / "status.json")
    if state["stage"] != "completed" or not state.get("gpu_released"):
        raise RuntimeError("DINO 네 결과 보고·GPU 종료 미완료")
    if not re.search(r"\[exit 0\]\s*$", wrapper_log):
        raise RuntimeError("DINO wrapper 성공 종료 미확인")
    if not (c.DINO_CHAIN / "final_comparison.json").exists() or not (c.DINO_CHAIN / "final_comparison.md").exists():
        raise RuntimeError("DINO 최종 비교 보고서 누락")
    if state.get("active_child_pid") and alive(state["active_child_pid"]):
        raise RuntimeError("DINO GPU 자식 프로세스 생존")
    for path in (protocol["dino_chain"],):
        if Path(path) != c.DINO_CHAIN:
            raise RuntimeError("다른 선행 실행")


def gpu_status():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits", "--id=0"], text=True)
    total, used, utilization = map(int, output.strip().split(","))
    return {"total_mib": total, "used_mib": used, "utilization_percent": utilization}


def child(run, stage, batch=2):
    protocol = c.load(run)
    gpu = stage not in ("preflight", "report")
    if gpu:
        while True:
            state = gpu_status()
            if state["total_mib"] - state["used_mib"] >= 6656 and state["utilization_percent"] <= 20:
                break
            c.status(run, "waiting_for_gpu", next_stage=stage, gpu=state, gpu_work_started=False)
            time.sleep(15)
    python = protocol["dino_python"] if stage == "report" else protocol["yolo_python"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if gpu else "-1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", YOLO_CONFIG_DIR=str(run / "ultralytics_config"), YOLO_OFFLINE="true")
    logfile = run / f"{stage}_b{batch}.log"
    with logfile.open("x", encoding="utf-8") as output:
        process = subprocess.Popen([python, "-u", "-m", "classifier.dino.yolo_pipeline", "--stage", stage,
                                    "--run-dir", str(run), "--batch", str(batch)], cwd=c.ROOT, env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        c.status(run, stage, active_child_pid=process.pid, active_log=str(logfile), gpu_work_started=gpu)
        c.log(f"YOLO 단계 시작: {stage}, PID={process.pid}")
        code = process.wait()
        output.write(f"\n[exit {code}]\n")
    if code and not (stage == "smoke" and code == 3):
        raise RuntimeError(f"{stage} 실패 exit={code}: {logfile}")
    if not code and stage in ("head-nms", "head-free"):
        file = run / ("nms" if stage == "head-nms" else "free") / "complete.json"
        evidence = c.read(file)
        evidence["gpu_released"] = True
        c.write(file, evidence)
    c.log(f"YOLO 단계 종료: {stage}, exit={code}")
    return code


def queue(run):
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("controller는 GPU를 사용하지 않음")
    run = c.safe_run(run)
    protocol = c.load(run)
    lock = run / "controller.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        if not c.read(run / "augmentation_review.json")["passed"]:
            raise RuntimeError("증강 검토 미완료")
        c.status(run, "waiting_for_dino", wait_for_pid=protocol["wait_for_wrapper_pid"], gpu_work_started=False)
        c.log("현재 DINO 학습·최종 val·TTA·네 결과 보고 성공 종료 대기. 10·11번 제외.")
        while alive(protocol["wait_for_wrapper_pid"]):
            stage = c.read(c.DINO_CHAIN / "status.json")["stage"]
            if stage == "failed":
                raise RuntimeError("선행 DINO 실패; YOLO 시작하지 않음")
            time.sleep(15)
        predecessor_complete(protocol, (c.DINO_CHAIN / "console.log").read_text(encoding="utf-8"))
        c.verify_data(run)
        batch = 2
        if child(run, "smoke", batch):
            failure = c.read(run / "smoke_b2_failure.json")
            if not failure["memory"]:
                raise RuntimeError("메모리 이외 점검 오류")
            batch = 1
            if child(run, "smoke", batch):
                raise RuntimeError("batch1도 메모리 점검 실패")
        smoke = c.read(run / f"smoke_b{batch}_complete.json")
        if smoke["epochs"] != 1 or smoke["ema_updates"] < 2 or smoke["reserved_mib"] > 6656:
            raise RuntimeError("GPU 점검 근거 부족")
        c.write(run / "selection.json", {"batch": batch, "nbs": 64, "accumulation_after_warmup": 64//batch,
                                         "smoke_sha256": c.sha(run / f"smoke_b{batch}_complete.json")})
        child(run, "train", batch)
        trained = c.read(run / "training_complete.json")
        if trained["epochs"] != 50 or trained["batch"] != batch:
            raise RuntimeError("예정 학습 미완료")
        child(run, "head-nms", batch)
        child(run, "head-free", batch)
        child(run, "report", batch)
        c.status(run, "completed", through_step=9, all_gpu_processes_released=True,
                 final_report=str(run / "final_comparison.md"))
        c.log("9번 DINO+YOLO 비교까지 완료. GPU 해제. DETR·test·추가 학습 없음.")
    except Exception as exc:
        c.status(run, "failed", error=f"{type(exc).__name__}: {exc}", gpu_work_started=False)
        c.log(f"YOLO 후속 실행 중단: {type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink(missing_ok=True)
