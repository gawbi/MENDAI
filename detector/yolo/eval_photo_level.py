"""Photo-level evaluation: 4x4 confusion matrix + balanced / overall accuracy + per-class recall.

Per photo: boxes with confidence >= conf decide the verdict by class presence
    none -> normal, heel_lining(1) only -> heel, outsole(0) only -> outsole, both classes -> both
Several boxes of one class are still that class (not both). No IoU condition.
Ground truth: `label` column of photo_labels/*.csv (never the box txt files; eval25 txt are all empty).
Balanced accuracy = mean(Recall_normal, Recall_heel, Recall_outsole). Recall_both = N/A when no both photos.

Inference runs once at a low conf and per-class max confidences are cached in per_image.csv,
so other thresholds can be applied later without rerunning the model.

    python detector/yolo/eval_photo_level.py --config detector/yolo/configs/yolo11m_1024_s0.json \
        --data-root <mendi_damage_v5_20261017> --runs-root <runs>
    python detector/yolo/eval_photo_level.py --summary --runs-root <runs>   # rebuild summary.md
    # test splits only at the final stage:
    python detector/yolo/eval_photo_level.py --config ... --sets v5_test test25 --final

Weights default to <runs-root>/yolo/<name>/weights/best.pt; outputs go to <runs-root>/yolo/<name>/photo_eval/.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

CLASSES = ["normal", "heel", "outsole", "both"]  # confusion-matrix order (rows = truth, columns = verdict)
OUTSOLE_ID, HEEL_ID = 0, 1                        # box class ids from data.yaml

# name -> (image folder, photo label csv, split), relative to --data-root
EVAL_SETS = {
    "v5_val": ("dataset_damage_v5_1024/images/val", "photo_labels/damage_image_labels_v5.csv", "val"),
    "val25": ("dataset_damage_eval25_1024/images/val", "photo_labels/damage_image_labels_eval25.csv", "val"),
    "v5_test": ("dataset_damage_v5_1024/images/test", "photo_labels/damage_image_labels_v5.csv", "test"),
    "test25": ("dataset_damage_eval25_1024/images/test", "photo_labels/damage_image_labels_eval25.csv", "test"),
}
TEST_SETS = {"v5_test", "test25"}


def verdict(conf_outsole: float, conf_heel: float, thr: float) -> str:
    has_o, has_h = conf_outsole >= thr, conf_heel >= thr
    if has_o and has_h:
        return "both"
    if has_h:
        return "heel"
    if has_o:
        return "outsole"
    return "normal"


def load_gt(csv_path: Path, split: str) -> dict:
    with open(csv_path, encoding="utf-8-sig") as f:
        return {r["file"]: r["label"] for r in csv.DictReader(f) if r["split"] == split}


def run_inference(weights: str, img_dir: Path, files: list, args) -> tuple:
    """Return ({file: (max_conf_outsole, max_conf_heel)} over boxes with conf >= args.min_conf, meta)."""
    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no Ultralytics usage analytics
    model = YOLO(weights)
    # nms left unset: Ultralytics 8.4 then uses NMS (one-to-many head for YOLO26), same for every model.
    meta = {"end2end": bool(getattr(model.model, "end2end", False)), "nms": "default (NMS)"}
    out = {}
    paths = [str(img_dir / f) for f in files]
    for i in range(0, len(paths), args.batch):
        for r in model.predict(paths[i:i + args.batch], imgsz=args.imgsz, conf=args.min_conf, iou=args.iou,
                               max_det=args.max_det, device=args.device, verbose=False):
            cls = r.boxes.cls.int().tolist()
            cf = r.boxes.conf.tolist()
            mo = max([c for k, c in zip(cls, cf) if k == OUTSOLE_ID], default=0.0)
            mh = max([c for k, c in zip(cls, cf) if k == HEEL_ID], default=0.0)
            out[Path(r.path).name] = (mo, mh)
    return out, meta


def metrics(rows: list) -> dict:
    """rows: [(truth, verdict), ...]"""
    cm = [[0] * 4 for _ in range(4)]
    for gt, pred in rows:
        cm[CLASSES.index(gt)][CLASSES.index(pred)] += 1
    n = sum(map(sum, cm))
    recall = {}
    for k, name in enumerate(CLASSES):
        support = sum(cm[k])
        recall[name] = cm[k][k] / support if support else None   # None -> N/A
    core = [recall[c] for c in ("normal", "heel", "outsole")]
    bal = sum(core) / 3 if all(r is not None for r in core) else None
    acc = sum(cm[k][k] for k in range(4)) / n if n else None
    return {"cm": cm, "n": n, "support": {c: sum(cm[k]) for k, c in enumerate(CLASSES)},
            "recall": recall, "balanced_acc": bal, "overall_acc": acc}


def pct(x) -> str:
    return "N/A" if x is None else f"{100 * x:.1f}%"


def cm_markdown(m: dict) -> str:
    lines = ["| 실제 정답 / 모델 예측 | normal | heel | outsole | both | 합계 |", "|---|--:|--:|--:|--:|--:|"]
    for k, name in enumerate(CLASSES):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in m["cm"][k]) + f" | {sum(m['cm'][k])} |")
    return "\n".join(lines)


def evaluate(args, cfg: dict) -> None:
    for s in args.sets:
        if s in TEST_SETS and not args.final:
            raise SystemExit(f"{s} is a test split: only at the final stage (pass --final).")

    data_root = Path(args.data_root).resolve()
    run_dir = Path(args.runs_root).resolve() / "yolo" / cfg["name"]
    weights = args.weights or str(run_dir / "weights/best.pt")
    conf = args.conf if args.conf is not None else cfg["photo_conf"]

    for s in args.sets:
        img_rel, csv_rel, split = EVAL_SETS[s]
        img_dir = data_root / img_rel
        gt = load_gt(data_root / csv_rel, split)
        files = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        missing = [f for f in files if f not in gt]
        if missing or len(files) != len(gt):
            raise SystemExit(f"{s}: {len(files)} images vs {len(gt)} CSV rows, {len(missing)} without label")
        if args.limit:
            files = files[:args.limit]

        out_dir = run_dir / "photo_eval" / s
        out_dir.mkdir(parents=True, exist_ok=True)
        raw, meta = run_inference(weights, img_dir, files, args)

        rows = []
        with open(out_dir / "per_image.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "gt", "max_conf_outsole", "max_conf_heel", f"pred@{conf}"])
            for fn in files:
                mo, mh = raw.get(fn, (0.0, 0.0))
                pred = verdict(mo, mh, conf)
                rows.append((gt[fn], pred))
                w.writerow([fn, gt[fn], f"{mo:.4f}", f"{mh:.4f}", pred])

        m = metrics(rows)
        record = {"name": cfg["name"], "exp_id": cfg.get("exp_id", cfg["name"]), "set": s, "weights": weights,
                  "conf": conf, "model": cfg["model"], "resolution": cfg["resolution"], "epochs": cfg["epochs"],
                  "predict": {"imgsz": args.imgsz, "iou": args.iou, "max_det": args.max_det,
                              "min_conf_cached": args.min_conf, **meta},
                  "limit": args.limit, "time": datetime.now().isoformat(timespec="seconds"), **m}
        (out_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        (out_dir / "confusion_matrix.md").write_text(
            f"# {cfg['name']} · {s} · conf={conf} (N={m['n']})\n\n{cm_markdown(m)}\n")
        print(f"\n[{cfg['name']} · {s} · conf={conf} · N={m['n']}]")
        print(cm_markdown(m))
        print(f"balanced {pct(m['balanced_acc'])} | overall {pct(m['overall_acc'])} | "
              + " | ".join(f"R_{c} {pct(m['recall'][c])}" for c in CLASSES))
    write_summary(Path(args.runs_root).resolve())


def write_summary(runs_root: Path) -> None:
    recs = [json.loads(p.read_text()) for p in sorted((runs_root / "yolo").glob("*/photo_eval/*/metrics.json"))]
    order = list(EVAL_SETS)
    recs.sort(key=lambda r: (order.index(r["set"]), r["exp_id"]))
    lines = ["# 사진 단위 평가 결과", "",
             "균형 정확도 = (Recall normal + Recall heel + Recall outsole) / 3. "
             "행=실제 정답, 열=모델 예측. both 정답이 0장인 평가셋은 Recall both = N/A.", "",
             "| 평가셋 | 실험 | 모델 | conf | N | 균형 정확도(핵심) | 전체 정확도 | Recall normal | Recall heel | Recall outsole | Recall both |",
             "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in recs:
        rc = r["recall"]
        flag = " (부분)" if r.get("limit") else ""
        model = f"{Path(r['model']).stem.upper()} · {r['epochs']} · {r['resolution']}"
        lines.append(f"| {r['set']}{flag} | {r['exp_id']} | {model} | {r['conf']} | {r['n']} | "
                     f"**{pct(r['balanced_acc'])}** | {pct(r['overall_acc'])} | {pct(rc['normal'])} | "
                     f"{pct(rc['heel'])} | {pct(rc['outsole'])} | {pct(rc['both'])} |")
    lines += [""]
    for r in recs:
        lines += [f"### {r['set']} · {r['exp_id']} · conf={r['conf']} (N={r['n']})", "", cm_markdown(r), ""]
    out = runs_root / "yolo" / "photo_eval_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data-root")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--weights", help="default: <runs-root>/yolo/<name>/weights/best.pt")
    ap.add_argument("--sets", nargs="+", default=["v5_val", "val25"], choices=EVAL_SETS)
    ap.add_argument("--conf", type=float, help="photo verdict threshold (default: photo_conf in the config)")
    ap.add_argument("--min-conf", type=float, default=0.001, help="cache boxes down to this conf")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images only")
    ap.add_argument("--final", action="store_true", help="allow test splits")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        write_summary(Path(args.runs_root).resolve())
        return
    if not (args.config and args.data_root):
        ap.error("--config and --data-root are required")
    evaluate(args, json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
