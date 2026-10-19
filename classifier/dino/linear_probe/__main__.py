import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["check", "smoke", "extract", "train", "evaluate"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("check", "train", "evaluate"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key, value in {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
                       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
        os.environ[key] = value
    from .pipeline import execute
    execute(args)


if __name__ == "__main__":
    main()
