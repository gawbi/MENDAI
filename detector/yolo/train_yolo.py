"""YOLO baseline training from a config (configs/*.json).

    python detector/yolo/train_yolo.py --config detector/yolo/configs/yolo11m_1024_s0.json \
        --data-root <mendi_damage_v5_20261017> --runs-root <runs>
    # resume an interrupted run from <runs-root>/yolo/<name>/weights/last.pt
    python detector/yolo/train_yolo.py --config ... --data-root ... --runs-root ... --resume

Outputs (weights, results.csv, args.yaml, plots) go to <runs-root>/yolo/<name>/ and are not committed.
The dataset is used as provided: no re-split, no file changes.
"""

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO, settings

settings.update({"sync": False})  # no Ultralytics usage analytics (dataset photos contain customer info)


def train_kwargs(cfg: dict, data_root: Path) -> dict:
    return dict(
        data=str(data_root / cfg["train_dataset"] / "data.yaml"),
        epochs=cfg["epochs"],
        imgsz=cfg["resolution"],
        batch=cfg["batch"],
        patience=cfg["patience"],      # > epochs, so early stopping never triggers
        optimizer=cfg["optimizer"],    # explicit, so lr0 is actually used (optimizer=auto would override it)
        lr0=cfg["lr0"],
        seed=cfg["seed"],
        deterministic=cfg["deterministic"],
        workers=cfg["workers"],
        save=True,                     # last.pt every epoch + best.pt (by val fitness)
        save_period=cfg["save_period"],
        plots=True,
        device=0,
    )


def write_env_record(run_dir: Path, cfg: dict, weights: str, resumed: bool) -> None:
    record = {
        "name": cfg["name"],
        "resumed": resumed,
        "weights": weights,
        "started": datetime.now().isoformat(timespec="seconds"),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "config": cfg,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "env_record.json", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True, help="unzipped mendi_damage_v5_20261017 folder")
    ap.add_argument("--runs-root", required=True, help="output folder outside the repo")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    project = Path(args.runs_root).resolve() / "yolo"
    run_dir = project / cfg["name"]
    last = run_dir / "weights/last.pt"

    if args.resume:
        if not last.exists():
            raise SystemExit(f"No checkpoint to resume: {last}")
        write_env_record(run_dir, cfg, str(last), resumed=True)
        # resume=True restores all original args (epochs, lr, batch, optimizer state, epoch counter).
        YOLO(str(last)).train(resume=True)
        return

    if any((run_dir / "weights").glob("*.pt")):
        raise SystemExit(f"{run_dir} already has checkpoints; use --resume or move them away.")
    write_env_record(run_dir, cfg, cfg["model"], resumed=False)
    YOLO(cfg["model"]).train(**train_kwargs(cfg, Path(args.data_root).resolve()),
                             project=str(project), name=cfg["name"], exist_ok=True)


if __name__ == "__main__":
    main()
