"""Per-photo summary of a prediction CSV: best score and box counts per class, one row per image of a folder.

Photos without any prediction row get score 0 and count 0, so the output always has one row per image.
The input may be the plain CSV written by predict_*.py or its .csv.gz copy.

  python tools/summarize_predictions.py --pred RUN/pred_v5_val.csv --images DATA_ROOT/dataset_damage_v5_1024/images/val --out v5_val_photo_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CLASS_NAMES, list_images  # noqa: E402


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="prediction CSV (file, class_id, class_name, score, x1, y1, x2, y2), .csv or .csv.gz")
    ap.add_argument("--images", required=True, help="image folder that was predicted (gives the list of photos)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.3, 0.5], help="score cutoffs for the box counts")
    args = ap.parse_args()

    classes = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    photos = [p.name for p in list_images(args.images)]
    best = {f: {c: 0.0 for c in classes} for f in photos}
    counts = {f: {(c, t): 0 for c in classes for t in args.cutoffs} for f in photos}
    unknown = 0
    with open_text(args.pred) as f:
        for row in csv.DictReader(f):
            name, cls, score = Path(row["file"]).name, row["class_name"].strip(), float(row["score"])
            if name not in best or cls not in classes:
                unknown += 1
                continue
            best[name][cls] = max(best[name][cls], score)
            for t in args.cutoffs:
                if score >= t:
                    counts[name][(cls, t)] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [f"max_{c}" for c in classes] + [f"n_{c}_ge{t:g}" for t in args.cutoffs for c in classes])
        for name in photos:
            w.writerow([name] + [f"{best[name][c]:.5f}" for c in classes] + [counts[name][(c, t)] for t in args.cutoffs for c in classes])
    print(f"{len(photos)} photos -> {out} ({unknown} rows for other photos or classes ignored)")


if __name__ == "__main__":
    main()
