import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "extract", "train", "evaluate", "compare"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--origin-run", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--runs", type=Path, nargs=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage in ("prepare", "compare"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name, value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        os.environ[name] = value
    from . import core as n
    from . import training
    n.torch.set_num_threads(1)
    if args.stage == "compare":
        from .comparison import compare
        compare([path.resolve() for path in args.runs], args.output.resolve())
        return
    run = args.run_dir.resolve()
    run.relative_to(n.ROOT / "runs/dino")
    protocol = n.prepare(run, args.origin_run, args.experiment) if args.stage == "prepare" else n.load(run)
    try:
        if args.stage == "extract":
            n.extract(run, protocol)
        elif args.stage == "train":
            training.train(run, protocol)
        elif args.stage == "evaluate":
            training.evaluate(run, protocol)
    except Exception as exc:
        n.p.status(run, "failed", failed_stage=args.stage, error=f"{type(exc).__name__}: {exc}")
        n.record(run, f"{args.stage} 단계 중단", f"`{type(exc).__name__}: {exc}`. 원본 LoRA·가중치·결과를 보존했습니다. test는 미사용입니다.")
        raise


if __name__ == "__main__":
    main()
