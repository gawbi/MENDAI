"""Image-level (photo) evaluation for the MENDAI detectors.

Turns box predictions into one verdict per photo (normal / heel / outsole / both)
and reports the team metrics: a 4x4 confusion matrix, balanced accuracy (mean of
per-class recall, the main metric), overall accuracy and per-class recall.

Verdict rule
  heel    if the photo's best heel_lining score >= t_heel
  outsole if the photo's best outsole score     >= t_out
  both    if both, normal if neither. Photos with no prediction rows are normal.

Thresholds (t_out, t_heel) are chosen on the union of the --select-on sets by
sweeping a grid and taking the highest balanced accuracy (ties: higher overall
accuracy, then higher thresholds). Use --fixed-conf for the one-time test run.

The report (image_eval.md) uses the team format: a summary row in percent
(balanced accuracy, overall accuracy, per-class recall) and the confusion matrix
with rows = truth and columns = verdict, for the union of all sets and for each set.

Example
  python eval_image_level.py --out RUN_DIR\\image_eval ^
    --set v5_val RUN_DIR\\pred_v5_val.csv DATA\\photo_labels\\damage_image_labels_v5.csv val ^
    --set val25  RUN_DIR\\pred_val25.csv  DATA\\photo_labels\\damage_image_labels_eval25.csv val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import CLASS_NAMES

LABELS = ["normal", "heel", "outsole", "both"]
LABEL_IDX = {name: i for i, name in enumerate(LABELS)}


def read_labels(csv_path, split):
    """file -> photo label for one split of a photo_labels CSV."""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"].strip() != split:
                continue
            label = row["label"].strip()
            if label not in LABEL_IDX:
                raise SystemExit(f"unknown label {label!r} for {row['file']} in {csv_path}")
            labels[row["file"].strip()] = label
    if not labels:
        raise SystemExit(f"no rows with split={split!r} in {csv_path}")
    return labels


def read_scores(pred_csv, files):
    """file -> [best outsole score, best heel_lining score]. Files without rows get zeros."""
    best = {name: [0.0, 0.0] for name in files}
    seen, unknown = set(), 0
    with open(pred_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = Path(row["file"]).name
            if name not in best:
                unknown += 1
                continue
            class_name = (row.get("class_name") or CLASS_NAMES[int(row["class_id"])]).strip()
            if class_name == "outsole":
                idx = 0
            elif class_name == "heel_lining":
                idx = 1
            else:
                continue
            seen.add(name)
            score = float(row["score"])
            if score > best[name][idx]:
                best[name][idx] = score
    return best, seen, unknown


def verdict(outsole_score, heel_score, t_out, t_heel):
    has_out = outsole_score >= t_out
    has_heel = heel_score >= t_heel
    if has_out and has_heel:
        return "both"
    if has_out:
        return "outsole"
    if has_heel:
        return "heel"
    return "normal"


def confusion(items, t_out, t_heel):
    """items: (label, outsole_score, heel_score). Rows = truth, columns = verdict."""
    cm = [[0] * 4 for _ in LABELS]
    for label, o, h in items:
        cm[LABEL_IDX[label]][LABEL_IDX[verdict(o, h, t_out, t_heel)]] += 1
    return cm


def summarize(cm):
    total = sum(sum(row) for row in cm)
    diag = sum(cm[i][i] for i in range(4))
    support = {label: sum(cm[i]) for i, label in enumerate(LABELS)}
    recall = {label: (cm[i][i] / support[label] if support[label] else None) for i, label in enumerate(LABELS)}
    present = [r for r in recall.values() if r is not None]
    return {
        "n": total,
        "balanced_acc": sum(present) / len(present) if present else None,
        "overall_acc": diag / total if total else None,
        "recall": recall,
        "support": support,
    }


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|---|" + "--:|" * (len(header) - 1)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def team_block(title, cm, m):
    """Team report format: percent summary row, then the confusion matrix (rows truth, columns verdict)."""
    summary = md_table(["균형 정확도", "전체 정확도", "normal 재현율", "heel 재현율", "outsole 재현율"],
                       [[pct(m["balanced_acc"]), pct(m["overall_acc"])]
                        + [pct(m["recall"][label]) for label in ("normal", "heel", "outsole")]])
    matrix = md_table(["정답 / 예측"] + LABELS, [[label] + cm[i] for i, label in enumerate(LABELS)])
    return "\n".join([f"### {title} (n={m['n']})", "", summary, "", "혼동 행렬은 행이 정답, 열이 예측입니다.", "", matrix])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs=4, action="append", required=True, metavar=("NAME", "PRED_CSV", "LABELS_CSV", "SPLIT"),
                    help="an evaluation set; repeatable. SPLIT is the split column value to score (val / test)")
    ap.add_argument("--select-on", help="comma-separated set names whose union picks the thresholds (default: all sets)")
    ap.add_argument("--fixed-conf", nargs=2, type=float, metavar=("OUTSOLE", "HEEL"), help="skip the sweep and use these thresholds")
    ap.add_argument("--step", type=float, default=0.05, help="sweep grid step; grid runs from step to 1-step")
    ap.add_argument("--out", required=True, help="output folder for image_eval.md / .json / _sweep.csv / _decisions.csv")
    args = ap.parse_args()

    sets = {}
    for name, pred_csv, labels_csv, split in args.set:
        labels = read_labels(labels_csv, split)
        scores, seen, unknown = read_scores(pred_csv, labels)
        files = sorted(labels)
        sets[name] = {
            "files": files,
            "items": [(labels[f], scores[f][0], scores[f][1]) for f in files],
            "pred_csv": pred_csv, "labels_csv": labels_csv, "split": split,
            "photos_without_predictions": len(files) - len(seen),
        }
        print(f"[{name}] {len(files)} photos, {len(files) - len(seen)} without any prediction row, {unknown} rows for other files ignored")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    select_names = [n.strip() for n in args.select_on.split(",")] if args.select_on else list(sets)
    for n in select_names:
        if n not in sets:
            raise SystemExit(f"--select-on names unknown set {n!r}")

    if args.fixed_conf:
        t_out, t_heel = args.fixed_conf
        how = "fixed"
        stale = out_dir / "image_eval_sweep.csv"  # from an earlier sweep into the same folder
        if stale.exists():
            stale.unlink()
    else:
        grid = [round(args.step * i, 6) for i in range(1, int(round(1 / args.step)))]
        selection = [item for n in select_names for item in sets[n]["items"]]
        best_key, t_out, t_heel = None, None, None
        with open(out_dir / "image_eval_sweep.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_outsole", "t_heel", "selection_balanced_acc", "selection_overall_acc"]
                            + [f"{n}_{k}" for n in sets for k in ("balanced_acc", "overall_acc")])
            for to in grid:
                for th in grid:
                    m = summarize(confusion(selection, to, th))
                    per_set = [summarize(confusion(sets[n]["items"], to, th)) for n in sets]
                    writer.writerow([to, th, fmt(m["balanced_acc"], 5), fmt(m["overall_acc"], 5)]
                                    + [fmt(v, 5) for ps in per_set for v in (ps["balanced_acc"], ps["overall_acc"])])
                    key = (m["balanced_acc"] or -1.0, m["overall_acc"] or -1.0, to + th)
                    if best_key is None or key > best_key:
                        best_key, t_out, t_heel = key, to, th
        how = f"chosen on {' + '.join(select_names)} ({sum(len(sets[n]['items']) for n in select_names)} photos, balanced acc {best_key[0]:.4f})"

    results = {}
    with open(out_dir / "image_eval_decisions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "file", "truth", "verdict", "outsole_score", "heel_score"])
        for name, s in sets.items():
            for file_name, (label, o, h) in zip(s["files"], s["items"]):
                writer.writerow([name, file_name, label, verdict(o, h, t_out, t_heel), round(o, 5), round(h, 5)])
            cm = confusion(s["items"], t_out, t_heel)
            results[name] = {"confusion": cm, "labels": LABELS, **summarize(cm),
                             "photos_without_predictions": s["photos_without_predictions"],
                             "pred_csv": s["pred_csv"], "labels_csv": s["labels_csv"], "split": s["split"]}

    # Union of every set, in the team's report format.
    all_items = [item for s in sets.values() for item in s["items"]]
    cm_all = confusion(all_items, t_out, t_heel)
    combined = {"confusion": cm_all, "labels": LABELS, "sets": list(sets), **summarize(cm_all)}

    # One line per set in the RESULTS.md column order: balanced / overall | recall N / H / O
    row_cells = [f"{t_out:.2f} / {t_heel:.2f}"]
    for name, r in results.items():
        row_cells.append(f"{fmt(r['balanced_acc'])} / {fmt(r['overall_acc'])}")
        row_cells.append(" / ".join(fmt(r["recall"][label], 2) for label in ("normal", "heel", "outsole")))
    results_row = " | ".join(row_cells)

    header = [
        "# Image-level evaluation",
        "",
        f"- thresholds: outsole >= {t_out:.2f}, heel_lining >= {t_heel:.2f} ({how})",
        "- verdict: heel / outsole if the photo's best score of that class reaches its threshold; both if both; otherwise normal",
        "- balanced accuracy = mean recall over classes that have photos (both has none in val/test)",
        f"- RESULTS.md cells ({' | '.join(['conf (outsole / heel)'] + [c for n in results for c in (f'{n} balanced / overall', f'{n} recall N / H / O')])}):",
        f"  `{results_row}`",
        "",
    ]
    blocks = [team_block("전체 (" + " + ".join(sets) + ")", cm_all, combined)]
    blocks += [team_block(name, r["confusion"], r) for name, r in results.items()]
    md = "\n".join(header) + "\n\n".join(blocks) + "\n"
    (out_dir / "image_eval.md").write_text(md, encoding="utf-8")
    (out_dir / "image_eval.json").write_text(json.dumps({
        "thresholds": {"outsole": t_out, "heel_lining": t_heel, "how": how, "select_on": select_names, "step": args.step},
        "results_row": results_row,
        "combined": combined,
        "sets": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
