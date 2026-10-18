"""Train Ultralytics RT-DETR from COCO weights on the v5 damage dataset.

Only team-level settings are passed (epochs, image size, seed, batch); everything
else is the Ultralytics default unless overridden with --amp / --optimizer / --lr0.
With the defaults (optimizer=auto -> AdamW lr 0.00167, fp16 AMP) the v5 640 run
went NaN at epoch 4-6 on 2026-10-17, so the queue passes
--amp bf16 --optimizer AdamW --lr0 0.0001 (the RT-DETR paper learning rate).
After training, best.pt is re-scored on val and the numbers for RESULTS.md are
written to <run>/box_metrics_val.json.

Example
  python train_rtdetr.py --data C:\\mendai\\data\\mendi_damage_v5_20261017\\dataset_damage_v5_640\\data.yaml ^
    --imgsz 640 --batch 8 --project C:\\mendai\\runs\\rtdetr --name rtdetr-l_640_s0 --weights-dir C:\\mendai\\weights
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

from common import CLASS_NAMES, chdir_weights, keep_awake


def best_epoch(results_csv):
    """Ultralytics keeps best.pt by fitness = val mAP50-95; return (epoch, value) of its maximum."""
    best = None
    with open(results_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            value = float(row["metrics/mAP50-95(B)"])
            if best is None or value > best[1]:
                best = (int(row["epoch"]), value)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="absolute path to data.yaml (not needed with --resume)")
    ap.add_argument("--imgsz", type=int, help="input size = size of the dataset copy (640 / 1024 / 1536)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="rtdetr-l.pt", help="COCO pretrained weights: rtdetr-l.pt or rtdetr-x.pt")
    ap.add_argument("--amp", default="True", help="Ultralytics amp: True (fp16), False or bf16")
    ap.add_argument("--optimizer", default="auto", help="Ultralytics optimizer: auto (library default), AdamW, SGD, ...")
    ap.add_argument("--lr0", type=float, help="initial learning rate; only used with an explicit --optimizer (auto picks its own)")
    ap.add_argument("--project", help="runs folder outside the repo, e.g. C:\\mendai\\runs\\rtdetr")
    ap.add_argument("--name", help="run name, e.g. rtdetr-l_640_s0")
    ap.add_argument("--exist-ok", action="store_true", help="reuse an existing run folder instead of numbering it")
    ap.add_argument("--weights-dir", help="where pretrained weights are downloaded (becomes the working directory)")
    ap.add_argument("--resume", help="last.pt of an interrupted run; the other settings are read from it")
    ap.add_argument("--resume-batch", type=int, help="with --resume: continue with a smaller batch (after an out-of-memory); "
                                                     "Ultralytics still accumulates to nbs=64, so the effective batch is unchanged")
    args = ap.parse_args()
    if not args.resume and not (args.data and args.imgsz and args.project and args.name):
        ap.error("--data, --imgsz, --project and --name are required unless --resume is given")

    chdir_weights(args.weights_dir)
    from ultralytics import RTDETR, settings

    settings.update(sync=False)  # no usage analytics or uploads
    amp = {"true": True, "false": False}.get(args.amp.lower(), args.amp)

    t0 = time.time()
    with keep_awake():
        if args.resume:
            model = RTDETR(args.resume)
            model.train(resume=True, **({"batch": args.resume_batch} if args.resume_batch else {}))
        else:
            model = RTDETR(args.model)
            extra = {"optimizer": args.optimizer}
            if args.lr0 is not None:
                extra["lr0"] = args.lr0
            # deterministic=False: Ultralytics recommends it for RT-DETR (grid_sample has no deterministic kernel).
            model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, seed=args.seed,
                        workers=args.workers, amp=amp, deterministic=False,
                        project=args.project, name=args.name, exist_ok=args.exist_ok, **extra)
        train_min = (time.time() - t0) / 60
        targs = model.trainer.args
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights" / "best.pt"

        # Score the kept checkpoint on val so RESULTS.md numbers come from best.pt itself.
        val = RTDETR(str(best)).val(data=targs.data, imgsz=targs.imgsz, batch=targs.batch, split="val",
                                    plots=False, project=str(run_dir), name="val_best", exist_ok=True)

    per_class = {}
    for i, c in enumerate(val.box.ap_class_index):
        per_class[CLASS_NAMES.get(int(c), str(c))] = {"AP50": float(val.box.ap50[i]), "AP50-95": float(val.box.ap[i])}
    epoch, fitness = best_epoch(run_dir / "results.csv")
    summary = {
        "date": date.today().isoformat(),
        "model": Path(targs.model).name if targs.model else args.model,
        "imgsz": targs.imgsz, "epochs": targs.epochs, "batch": targs.batch, "seed": targs.seed, "amp": str(targs.amp),
        "optimizer": str(targs.optimizer), "lr0": targs.lr0,
        "best_epoch": epoch, "best_fitness_mAP50-95": fitness, "train_minutes": round(train_min, 1),
        "val": {"mAP50": float(val.box.map50), "mAP50-95": float(val.box.map), "per_class": per_class},
        "run_dir": str(run_dir), "best_weights": str(best), "data": str(targs.data),
    }
    (run_dir / "box_metrics_val.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ap_out = per_class.get("outsole", {}).get("AP50-95")
    ap_heel = per_class.get("heel_lining", {}).get("AP50-95")
    memo = f"batch {targs.batch}, amp {targs.amp}, {targs.optimizer} lr0 {targs.lr0}, {train_min:.0f} min"
    row = (f"| {summary['date']} | RT-DETR ({summary['model']}) | {targs.imgsz} | {targs.epochs} | {targs.seed} | {epoch} "
           f"| {val.box.map50:.3f} | {val.box.map:.3f} | {ap_out:.3f} | {ap_heel:.3f} | {memo} |")
    print("RESULTS.md box row:")
    print(row)
    print(f"written {run_dir / 'box_metrics_val.json'}")


if __name__ == "__main__":
    main()
