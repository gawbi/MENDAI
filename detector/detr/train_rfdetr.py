"""Train RF-DETR (Roboflow `rfdetr`, pretrained weights) on the v5 damage dataset, optionally with LoRA.

Only team-level settings are passed (epochs, resolution, seed) plus the batch /
accumulation split needed to fit 8 GB; everything else is the library default.
--lora turns on the library's built-in LoRA (DoRA on the DINOv2 backbone; rank and
targets are fixed by rfdetr). The test split is never evaluated here (run_test=False).
After training, the best val epoch is read from <output-dir>/metrics.csv and written
to <output-dir>/box_metrics_val.json.

Example
  python train_rfdetr.py --dataset-dir C:\\mendai\\data\\mendi_damage_v5_20261017\\dataset_damage_v5_640 ^
    --resolution 640 --size small --batch-size 4 --grad-accum 4 ^
    --output-dir C:\\mendai\\runs\\rfdetr\\rfdetr-s_640_s0 --weights-dir C:\\mendai\\weights
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

from common import chdir_weights, keep_awake

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def best_val_row(metrics_csv):
    """Row of metrics.csv with the highest val mAP50-95 (the metric rfdetr uses for its best checkpoint)."""
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    key = next((c for c in rows[0] if c.strip().lower() in ("val/map_50_95", "val/map50-95", "val/map")), None)
    if key is None:
        key = next((c for c in rows[0] if "map_50_95" in c.lower() or "map50-95" in c.lower()), None)
    if key is None:
        return None, sorted(rows[0])
    rows = [r for r in rows if r.get(key, "").strip() not in ("", "nan")]
    best = max(rows, key=lambda r: float(r[key]))
    return best, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="folder that holds data.yaml, images/ and labels/")
    ap.add_argument("--resolution", type=int, required=True, help="input size = size of the dataset copy; multiple of 32")
    ap.add_argument("--size", choices=sorted(SIZES), default="small", help="team baseline is small (same size as the earlier overfitting comparison)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="batch-size x grad-accum = effective batch (16 by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--output-dir", required=True, help="run folder outside the repo")
    ap.add_argument("--weights-dir", help="where pretrained weights are downloaded (becomes the working directory)")
    ap.add_argument("--lora", action="store_true", help="rfdetr built-in LoRA on the backbone (backbone_lora=True)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing, if the resolution does not fit otherwise")
    ap.add_argument("--multi-scale", choices=["per-batch", "off"], help="rfdetr multi_scale (default per-batch)")
    ap.add_argument("--no-expanded-scales", action="store_true", help="expanded_scales=False: narrower multi-scale range, lower peak VRAM")
    ap.add_argument("--resume", help="checkpoint to resume from (rfdetr `resume`)")
    args = ap.parse_args()

    chdir_weights(args.weights_dir)
    import rfdetr

    model_kwargs = {"resolution": args.resolution}
    if args.lora:
        model_kwargs["backbone_lora"] = True
    if args.grad_ckpt:
        model_kwargs["gradient_checkpointing"] = True
    model = getattr(rfdetr, SIZES[args.size])(**model_kwargs)

    train_kwargs = {
        "dataset_dir": args.dataset_dir, "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum, "seed": args.seed, "num_workers": args.num_workers,
        "output_dir": args.output_dir, "log_per_class_metrics": True, "run_test": False, "tensorboard": False,
    }
    if args.multi_scale:
        train_kwargs["multi_scale"] = args.multi_scale
    if args.no_expanded_scales:
        train_kwargs["expanded_scales"] = False
    if args.resume:
        train_kwargs["resume"] = args.resume

    t0 = time.time()
    with keep_awake():
        model.train(**train_kwargs)
    train_min = (time.time() - t0) / 60

    out_dir = Path(args.output_dir)
    metrics_csv = out_dir / "metrics.csv"
    row, key = best_val_row(metrics_csv) if metrics_csv.exists() else (None, None)
    val = {k.strip(): r for k, r in (row or {}).items() if k.strip().startswith("val")}
    summary = {
        "date": date.today().isoformat(),
        "model": f"RF-DETR {args.size}" + (" + LoRA" if args.lora else ""),
        "resolution": args.resolution, "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum, "seed": args.seed, "gradient_checkpointing": args.grad_ckpt,
        "multi_scale": args.multi_scale or "per-batch (default)", "expanded_scales": not args.no_expanded_scales,
        "best_epoch": row.get("epoch") if row else None, "best_metric_column": key, "train_minutes": round(train_min, 1),
        "val": val, "run_dir": str(out_dir), "best_weights": str(out_dir / "checkpoint_best_total.pth"),
        "dataset_dir": args.dataset_dir,
    }
    (out_dir / "box_metrics_val.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if row is None:
        print(f"could not find a val mAP50-95 column in {metrics_csv}; columns: {key}")
    else:
        def g(*names):
            for n in names:
                for k, v in val.items():
                    if k.lower() == n.lower():
                        try:
                            return f"{float(v):.3f}"
                        except ValueError:
                            return v
            return "-"
        memo = f"batch {args.batch_size}x{args.grad_accum}, {train_min:.0f} min" + (", grad ckpt" if args.grad_ckpt else "")
        print("RESULTS.md box row:")
        print(f"| {summary['date']} | {summary['model']} | {args.resolution} | {args.epochs} | {args.seed} | {row.get('epoch')} "
              f"| {g('val/mAP_50', 'val/map50')} | {g('val/mAP_50_95', 'val/map50-95', 'val/map')} "
              f"| {g('val/AP/outsole')} | {g('val/AP/heel_lining')} | {memo} |")
    print(f"written {out_dir / 'box_metrics_val.json'}")


if __name__ == "__main__":
    main()
