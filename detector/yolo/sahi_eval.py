"""Standard vs SAHI tiled inference without retraining (T2).

Same weights, same photo-level rule and threshold. SAHI slices each photo in memory (files and labels
untouched), predicts every tile at imgsz, adds a full-image prediction, and merges duplicate boxes with
class-wise NMS (IOU 0.5). SAHI itself switches to NMS/IOU when boxes are kept below conf 0.1, and both NMS and
GREEDYNMM keep each class's best score, so the photo verdict does not depend on the merge type.
Standard inference is re-timed in the same process so the per-photo times are comparable.

    python detector/yolo/sahi_eval.py --config detector/yolo/configs/yolo26m_1024_s0.json \
        --data-root <mendi_damage_v5_20261017> --runs-root <runs>

Writes <runs-root>/yolo/<name>_T2sahi/photo_eval/<set>/{per_image.csv, metrics.json, confusion_matrix.md}.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import eval_photo_level as ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--sets", nargs="+", default=["v5_val", "val25"], choices=["v5_val", "val25"])
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1024, help="model input size for each tile")
    ap.add_argument("--min-conf", type=float, default=0.01, help="cache boxes down to this conf")
    ap.add_argument("--no-full", action="store_true", help="do not add the full-image prediction")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import sahi
    import ultralytics
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    cfg = json.loads(Path(args.config).read_text())
    runs = Path(args.runs_root).resolve() / "yolo"
    weights = str(runs / cfg["name"] / "weights/best.pt")
    conf = cfg["photo_conf"]
    data_root = Path(args.data_root).resolve()

    sahi_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights,
                                                    confidence_threshold=args.min_conf, device=args.device,
                                                    image_size=args.imgsz)
    std_model = YOLO(weights)
    sahi_cfg = {"slice": args.slice, "overlap": args.overlap, "tile_imgsz": args.imgsz,
                "full_image_pred": not args.no_full, "postprocess": "NMS", "match_metric": "IOU",
                "match_threshold": 0.5, "class_agnostic": False, "min_conf_cached": args.min_conf,
                "sahi": sahi.__version__, "ultralytics": ultralytics.__version__}

    for s in args.sets:
        img_rel, csv_rel, split = ev.EVAL_SETS[s]
        img_dir = data_root / img_rel
        gt = ev.load_gt(data_root / csv_rel, split)
        files = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if args.limit:
            files = files[:args.limit]

        # warm-up so the first photo's CUDA init is not timed
        std_model.predict(str(img_dir / files[0]), imgsz=args.imgsz, conf=args.min_conf, device=args.device,
                          verbose=False)
        get_sliced_prediction(str(img_dir / files[0]), sahi_model, slice_height=args.slice, slice_width=args.slice,
                              overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                              perform_standard_pred=not args.no_full, verbose=0)

        t_std = t_sahi = 0.0
        rows, out_rows = [], []
        for fn in files:
            t0 = time.perf_counter()
            std_model.predict(str(img_dir / fn), imgsz=args.imgsz, conf=args.min_conf, device=args.device,
                              verbose=False)
            t1 = time.perf_counter()
            res = get_sliced_prediction(str(img_dir / fn), sahi_model, slice_height=args.slice,
                                        slice_width=args.slice, overlap_height_ratio=args.overlap,
                                        overlap_width_ratio=args.overlap, perform_standard_pred=not args.no_full,
                                        postprocess_type="NMS", postprocess_match_metric="IOU",
                                        postprocess_match_threshold=0.5, postprocess_class_agnostic=False,
                                        verbose=0)
            t2 = time.perf_counter()
            t_std += t1 - t0
            t_sahi += t2 - t1
            mo = max([p.score.value for p in res.object_prediction_list if p.category.id == ev.OUTSOLE_ID],
                     default=0.0)
            mh = max([p.score.value for p in res.object_prediction_list if p.category.id == ev.HEEL_ID],
                     default=0.0)
            pred = ev.verdict(mo, mh, conf)
            rows.append((gt[fn], pred))
            out_rows.append([fn, gt[fn], f"{mo:.4f}", f"{mh:.4f}", pred])

        out_dir = runs / f"{cfg['name']}_T2sahi" / "photo_eval" / s
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "per_image.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "gt", "max_conf_outsole", "max_conf_heel", f"pred@{conf}"])
            w.writerows(out_rows)
        m = ev.metrics(rows)
        rec = {"name": f"{cfg['name']}_T2sahi", "exp_id": "T2", "set": s, "weights": weights, "conf": conf,
               "model": cfg["model"], "resolution": cfg["resolution"], "epochs": cfg["epochs"],
               "predict": sahi_cfg, "limit": args.limit,
               "ms_per_photo": {"standard": 1000 * t_std / len(files), "sahi": 1000 * t_sahi / len(files)},
               "time": datetime.now().isoformat(timespec="seconds"), **m}
        (out_dir / "metrics.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        (out_dir / "confusion_matrix.md").write_text(f"# T2 SAHI · {s} · conf={conf} (N={m['n']})\n\n"
                                                     f"{ev.cm_markdown(m)}\n")
        print(f"[T2 · {s} · N={m['n']}] balanced {ev.pct(m['balanced_acc'])} overall {ev.pct(m['overall_acc'])} "
              f"| ms/photo standard {rec['ms_per_photo']['standard']:.0f} sahi {rec['ms_per_photo']['sahi']:.0f}")


if __name__ == "__main__":
    main()
