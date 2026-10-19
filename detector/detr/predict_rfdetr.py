"""Predict with a trained RF-DETR checkpoint on one image folder and write the shared prediction CSV.

The model is rebuilt the way it was trained (--size, --resolution, --lora) and loaded
from checkpoint_best_total.pth. Class ids are taken from the class names rfdetr
returns, so they match data.yaml (0 outsole, 1 heel_lining) whatever rfdetr uses internally.

Example
  python predict_rfdetr.py --weights RUNS_ROOT\\rfdetr\\rfdetr-s_640_s0\\checkpoint_best_total.pth ^
    --size small --resolution 640 ^
    --images DATA_ROOT\\dataset_damage_v5_640\\images\\val ^
    --out RUNS_ROOT\\rfdetr\\rfdetr-s_640_s0\\pred_v5_val.csv
"""

from __future__ import annotations

import argparse

from common import CLASS_IDS, list_images, prediction_row, write_predictions
from train_rfdetr import SIZES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="checkpoint_best_total.pth of a run")
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--resolution", type=int, required=True, help="resolution the model was trained with")
    ap.add_argument("--lora", action="store_true", help="the run was trained with --lora")
    ap.add_argument("--images", required=True, help="folder of images (one split)")
    ap.add_argument("--out", required=True, help="prediction CSV to write")
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hflip", action="store_true",
                    help="TTA view: predict on horizontally flipped images; boxes are mapped back to the original image")
    args = ap.parse_args()

    import rfdetr

    kwargs = {"pretrain_weights": args.weights, "resolution": args.resolution, "num_classes": len(CLASS_IDS)}
    if args.lora:
        kwargs["backbone_lora"] = True
    model = getattr(rfdetr, SIZES[args.size])(**kwargs)

    images = list_images(args.images)
    dropped = {}

    def rows():
        for i in range(0, len(images), args.batch):
            chunk = images[i:i + args.batch]
            if args.hflip:
                from PIL import Image, ImageOps

                inputs = [ImageOps.mirror(Image.open(p).convert("RGB")) for p in chunk]
                widths = [img.width for img in inputs]
            else:
                inputs, widths = [str(p) for p in chunk], [None] * len(chunk)
            dets = model.predict(inputs, threshold=args.threshold)
            if not isinstance(dets, list):
                dets = [dets]
            for path, w, d in zip(chunk, widths, dets):
                if w is not None and len(d):
                    xyxy = d.xyxy.copy()
                    d.xyxy[:, 0], d.xyxy[:, 2] = w - xyxy[:, 2], w - xyxy[:, 0]  # back to the unflipped image
                names = d.data.get("class_name") if getattr(d, "data", None) else None
                if names is None:
                    raise SystemExit("rfdetr returned no class names; cannot map class ids safely")
                for j in range(len(d)):
                    name = str(names[j])
                    if name not in CLASS_IDS:  # rfdetr also scores a `__background__` class; those are not boxes we want
                        dropped[name] = dropped.get(name, 0) + 1
                        continue
                    yield prediction_row(path.name, CLASS_IDS[name], d.confidence[j], d.xyxy[j])

    n = write_predictions(args.out, rows())
    if dropped:
        print(f"dropped detections of classes outside data.yaml: {dropped}")
    print(f"{n} boxes for {len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()
