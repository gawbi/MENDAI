"""Confidence-threshold tuning without retraining (T1).

Reads per_image.csv (per-class max confidence per photo) written by eval_photo_level.py for the
validation sets and grid-searches the photo verdict thresholds. Selection criterion: mean of the
per-set balanced accuracies over the validation sets (sets are not merged). Test sets are never read.

    python detector/yolo/tune_conf.py --config detector/yolo/configs/yolo26m_1024_s0.json --runs-root <runs>

Writes <runs-root>/yolo/<name>_T1conf/: grid.csv, choice.json, photo_eval/<set>/{metrics.json, confusion_matrix.md}.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import eval_photo_level as ev

VAL_SETS = ["v5_val", "val25"]


def load_cached(run_dir: Path, s: str) -> list:
    with open(run_dir / "photo_eval" / s / "per_image.csv") as f:
        return [(r["gt"], float(r["max_conf_outsole"]), float(r["max_conf_heel"])) for r in csv.DictReader(f)]


def verdicts(rows: list, thr_o: float, thr_h: float) -> list:
    out = []
    for gt, co, ch in rows:
        o, h = co >= thr_o, ch >= thr_h
        out.append((gt, "both" if o and h else "heel" if h else "outsole" if o else "normal"))
    return out


def score(data: dict, thr_o: float, thr_h: float) -> dict:
    return {s: ev.metrics(verdicts(rows, thr_o, thr_h)) for s, rows in data.items()}


def mean_bal(ms: dict) -> float:
    return sum(m["balanced_acc"] for m in ms.values()) / len(ms)


def search(data: dict, grid: list, shared: bool) -> tuple:
    best = None
    for to in grid:
        for th in ([to] if shared else grid):
            ms = score(data, to, th)
            key = (round(mean_bal(ms), 10), round(sum(m["overall_acc"] for m in ms.values()), 10))
            if best is None or key > best[0]:
                best = (key, to, th, ms)
    return best[1], best[2], best[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--exp-id", default="T1")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    runs = Path(args.runs_root).resolve() / "yolo"
    src = runs / cfg["name"]
    data = {s: load_cached(src, s) for s in VAL_SETS}
    n = int(round(1 / args.step))
    grid = [round(i * args.step, 4) for i in range(1, n)]

    out = runs / f"{cfg['name']}_T1conf"  # e.g. yolo26m_1024_s0_lr5e-4_T1conf for A3 + conf tuning
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conf_outsole", "conf_heel"] + [f"bal_{s}" for s in VAL_SETS] + ["mean_bal"])
        for to in grid:
            for th in grid:
                ms = score(data, to, th)
                w.writerow([to, th] + [f"{ms[s]['balanced_acc']:.4f}" for s in VAL_SETS] + [f"{mean_bal(ms):.4f}"])

    to, th, ms = search(data, grid, shared=False)
    so, _, sms = search(data, grid, shared=True)
    # how well a threshold picked on one validation set carries over to the other
    cross = {}
    for a, b in [("v5_val", "val25"), ("val25", "v5_val")]:
        co, ch, _ = search({a: data[a]}, grid, shared=False)
        cross[f"picked_on_{a}"] = {"conf_outsole": co, "conf_heel": ch,
                                   f"balanced_on_{b}": ev.metrics(verdicts(data[b], co, ch))["balanced_acc"]}

    base = score(data, cfg["photo_conf"], cfg["photo_conf"])
    choice = {"criterion": "mean of per-set balanced accuracy over " + ", ".join(VAL_SETS), "grid_step": args.step,
              "baseline_conf": cfg["photo_conf"], "baseline_mean_bal": mean_bal(base),
              "per_class": {"conf_outsole": to, "conf_heel": th, "mean_bal": mean_bal(ms)},
              "shared": {"conf": so, "mean_bal": mean_bal(sms)}, "cross_check": cross,
              "note": "thresholds chosen and scored on the same validation sets, so the gain is optimistic",
              "time": datetime.now().isoformat(timespec="seconds")}
    (out / "choice.json").write_text(json.dumps(choice, indent=2))

    for s, m in ms.items():
        d = out / "photo_eval" / s
        d.mkdir(parents=True, exist_ok=True)
        rec = {"name": out.name, "exp_id": args.exp_id, "set": s, "weights": str(src / "weights/best.pt"),
               "conf": f"o{to}/h{th}", "model": cfg["model"], "resolution": cfg["resolution"],
               "epochs": cfg["epochs"], "predict": "cached from " + cfg["name"], "limit": 0, **m}
        (d / "metrics.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        (d / "confusion_matrix.md").write_text(f"# {args.exp_id} · {s} · conf outsole {to} / heel {th} (N={m['n']})\n\n"
                                               f"{ev.cm_markdown(m)}\n")
    print(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
