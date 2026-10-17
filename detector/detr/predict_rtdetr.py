"""Predict with a trained RT-DETR checkpoint on one image folder and write the shared prediction CSV.

The default confidence floor (0.001) keeps almost every query, so any photo-level
rule or box metric can be recomputed later from the CSV without re-running the model.

Example
  python predict_rtdetr.py --weights C:\\mendai\\runs\\rtdetr\\rtdetr-l_640_s0\\weights\\best.pt --imgsz 640 ^
    --images C:\\mendai\\data\\mendi_damage_v5_20261017\\dataset_damage_v5_640\\images\\val ^
    --out C:\\mendai\\runs\\rtdetr\\rtdetr-l_640_s0\\pred_v5_val.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import list_images, prediction_row, write_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="best.pt of a run")
    ap.add_argument("--images", required=True, help="folder of images (one split)")
    ap.add_argument("--imgsz", type=int, required=True, help="same size the model was trained with")
    ap.add_argument("--out", required=True, help="prediction CSV to write")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import RTDETR, settings

    settings.update(sync=False)
    images = list_images(args.images)
    model = RTDETR(args.weights)

    def rows():
        for r in model.predict(source=str(Path(args.images)), imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                               batch=args.batch, stream=True, verbose=False, save=False):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            name = Path(r.path).name
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(xyxy, scores, classes):
                yield prediction_row(name, cls, score, box)

    n = write_predictions(args.out, rows())
    print(f"{n} boxes for {len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()
