import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="EX-06 완료 대기 → TTA 비교 → EX-07 LoRA+ → TTA 네 결과 비교")
    parser.add_argument("--stage", required=True, choices=["prepare", "queue", "tta-lora", "smoke-plus", "train-plus", "evaluate-plus", "tta-plus", "publish"])
    parser.add_argument("--chain-dir", type=Path, required=True)
    parser.add_argument("--origin-run", type=Path)
    parser.add_argument("--plus-run", type=Path)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name, value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}.items():
        os.environ[name] = value
    if args.stage in ("prepare", "queue", "publish"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from . import core
    from .runner import execute_gpu_stage, queue
    chain = args.chain_dir.resolve()
    if args.stage == "prepare":
        if not args.origin_run or not args.plus_run or not args.wait_for_pid:
            parser.error("prepare에는 --origin-run, --plus-run, --wait-for-pid 필요")
        core.prepare(chain, args.origin_run.resolve(), args.plus_run.resolve(), args.wait_for_pid)
    elif args.stage == "queue":
        queue(chain, args.resume)
    elif args.stage == "publish":
        from .evaluation import publish
        publish(chain, final=True)
    else:
        execute_gpu_stage(chain, args.stage, args.resume)


if __name__ == "__main__":
    main()
