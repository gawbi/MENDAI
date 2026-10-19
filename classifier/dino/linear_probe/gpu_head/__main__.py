import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "continue", "train", "evaluate"], required=True)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name, value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        os.environ[name] = value
    from .runner import execute
    execute(args)


if __name__ == "__main__":
    main()
