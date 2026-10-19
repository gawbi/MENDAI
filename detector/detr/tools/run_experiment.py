"""Run one DETR experiment from a config: train -> predict each eval set -> photo-level evaluation.

The config (configs/*.json) holds the settings only; every location on disk comes from the
command line. Run it with the Python of the environment that matches the config's family
(rtdetr -> the Ultralytics env, rfdetr -> the rfdetr env); each step runs with that same interpreter.

  python tools/run_experiment.py --config configs/rfdetr-s_1024_s0.json --data-root DATA_ROOT --runs-root RUNS_ROOT

Steps whose output already exists are skipped, so running the same command again continues
where it stopped (an interrupted training resumes from weights/last.pt or last.ckpt).

  <runs-root>/<family>/<name>/
    box_metrics_val.json        train: best checkpoint and its v5 val box metrics
    pred_<set>.csv              predict: one CSV per eval set (pred_<set>_hflip.csv as well with --tta)
    image_eval/image_eval.md    eval: photo-level metrics, thresholds chosen on the eval sets
    image_eval_tta/             eval with --tta (original + horizontal flip, scores averaged)
    run_<step>.log              console output of each step
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # detector/detr
sys.path.insert(0, str(HERE))
from common import keep_awake  # noqa: E402

LIBRARY = {"rtdetr": "ultralytics", "rfdetr": "rfdetr"}
REQUIRED = ["name", "family", "model", "resolution", "epochs", "seed", "train_dataset", "eval_sets"]


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    if cfg["family"] not in LIBRARY:
        raise SystemExit(f"{path}: family must be one of {sorted(LIBRARY)}")
    family_keys = ["batch"] if cfg["family"] == "rtdetr" else ["batch_size", "grad_accum"]
    missing = [k for k in family_keys if k not in cfg]
    if missing:
        raise SystemExit(f"{path}: {cfg['family']} configs need {missing}")
    return cfg


def best_weights(cfg, run_dir):
    return run_dir / "weights" / "best.pt" if cfg["family"] == "rtdetr" else run_dir / "checkpoint_best_total.pth"


def train_step(cfg, run_dir, data_root, weights_dir):
    py, data = sys.executable, data_root / cfg["train_dataset"]
    if cfg["family"] == "rtdetr":
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            return [py, HERE / "train_rtdetr.py", "--resume", last, "--weights-dir", weights_dir], f"resume from {last}"
        cmd = [py, HERE / "train_rtdetr.py", "--data", data / "data.yaml", "--imgsz", cfg["resolution"],
               "--epochs", cfg["epochs"], "--batch", cfg["batch"], "--seed", cfg["seed"], "--model", cfg["model"],
               "--workers", cfg.get("workers", 4), "--amp", cfg.get("amp", "True"),
               "--optimizer", cfg.get("optimizer", "auto"),
               # run_<step>.log already lives in the run folder, so let Ultralytics reuse it instead of numbering a new one
               "--project", run_dir.parent, "--name", run_dir.name, "--exist-ok", "--weights-dir", weights_dir]
        if cfg.get("lr0") is not None:
            cmd += ["--lr0", cfg["lr0"]]
        return cmd, "fresh training"
    cmd = [py, HERE / "train_rfdetr.py", "--dataset-dir", data, "--resolution", cfg["resolution"],
           "--size", cfg["model"], "--epochs", cfg["epochs"], "--batch-size", cfg["batch_size"],
           "--grad-accum", cfg["grad_accum"], "--seed", cfg["seed"], "--num-workers", cfg.get("num_workers", 2),
           "--output-dir", run_dir, "--weights-dir", weights_dir]
    if cfg.get("lora"):
        cmd.append("--lora")
    if cfg.get("grad_ckpt"):
        cmd.append("--grad-ckpt")
    last = run_dir / "last.ckpt"
    if last.exists():
        return cmd + ["--resume", last], f"resume from {last}"
    return cmd, "fresh training"


def predict_step(cfg, run_dir, data_root, eval_set, hflip):
    out = run_dir / f"pred_{eval_set['name']}{'_hflip' if hflip else ''}.csv"
    images = data_root / eval_set["images"]
    if cfg["family"] == "rtdetr":
        cmd = [sys.executable, HERE / "predict_rtdetr.py", "--weights", best_weights(cfg, run_dir),
               "--imgsz", cfg["resolution"], "--images", images, "--out", out]
    else:
        cmd = [sys.executable, HERE / "predict_rfdetr.py", "--weights", best_weights(cfg, run_dir),
               "--size", cfg["model"], "--resolution", cfg["resolution"], "--images", images, "--out", out]
        if cfg.get("lora"):
            cmd.append("--lora")
    if hflip:
        cmd.append("--hflip")
    return cmd, out


def eval_step(cfg, run_dir, data_root, tta):
    out = run_dir / ("image_eval_tta" if tta else "image_eval")
    cmd = [sys.executable, HERE / "eval_image_level.py", "--out", out]
    for es in cfg["eval_sets"]:
        cmd += ["--set", es["name"], run_dir / f"pred_{es['name']}.csv", data_root / es["labels"], es["split"]]
    if tta:
        for es in cfg["eval_sets"]:
            cmd += ["--add-view", es["name"], run_dir / f"pred_{es['name']}_hflip.csv"]
        cmd += ["--merge", "mean"]
    return cmd, out / "image_eval.json"


def child_env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True made CUDA initialization hang on the Windows laptop
    # these runs came from, so it is removed for the children.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def run(cmd, log_path, cwd, note):
    cmd = [str(c) for c in cmd]
    t0 = time.time()
    with open(log_path, "ab") as log:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n==== {stamp} | {note}\n{subprocess.list2cmdline(cmd)}\n".encode("utf-8"))
        log.flush()
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd), env=child_env()).returncode
        log.write(f"==== exit {code}\n".encode("utf-8"))
    return code, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<name>.json")
    ap.add_argument("--data-root", required=True, help="folder that holds dataset_damage_v5_*, dataset_damage_eval25_* and photo_labels")
    ap.add_argument("--runs-root", required=True, help="output folder outside the repo; the run goes to <runs-root>/<family>/<name>")
    ap.add_argument("--weights-dir", help="where the libraries download pretrained weights (default <runs-root>/pretrained)")
    ap.add_argument("--tta", action="store_true", help="also predict horizontally flipped images and evaluate the averaged scores")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root, runs_root = Path(args.data_root).resolve(), Path(args.runs_root).resolve()
    weights_dir = Path(args.weights_dir).resolve() if args.weights_dir else runs_root / "pretrained"
    run_dir = runs_root / cfg["family"] / cfg["name"]

    needed = [data_root / cfg["train_dataset"] / "data.yaml"]
    needed += [p for es in cfg["eval_sets"] for p in (data_root / es["images"], data_root / es["labels"])]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit("not found under --data-root:\n  " + "\n  ".join(missing))
    if not args.dry_run and importlib.util.find_spec(LIBRARY[cfg["family"]]) is None:
        raise SystemExit(f"{LIBRARY[cfg['family']]} is not installed in {sys.executable}; "
                         f"run this config with the {cfg['family']} environment's python")

    steps = []  # (name, output that marks the step done, (cmd, note) builder)
    steps.append(("train", run_dir / "box_metrics_val.json", lambda: train_step(cfg, run_dir, data_root, weights_dir)))
    views = [False, True] if args.tta else [False]
    for es in cfg["eval_sets"]:
        for hflip in views:
            cmd, out = predict_step(cfg, run_dir, data_root, es, hflip)
            steps.append((f"predict_{es['name']}{'_hflip' if hflip else ''}", out,
                          lambda cmd=cmd, es=es: (cmd, f"predict {es['name']}")))
    for tta in views:
        cmd, out = eval_step(cfg, run_dir, data_root, tta)
        steps.append(("eval_tta" if tta else "eval", out, lambda cmd=cmd: (cmd, "photo-level evaluation")))

    print(f"### {cfg['name']} ({cfg['family']}, {cfg['resolution']}px) -> {run_dir}")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
    with keep_awake():
        for name, out, build in steps:
            if out.exists():
                print(f"[{name}] skip, output exists: {out}")
                continue
            cmd, note = build()
            print(f"[{name}] {note}\n    {subprocess.list2cmdline([str(c) for c in cmd])}")
            if args.dry_run:
                continue
            code, minutes = run(cmd, run_dir / f"run_{name}.log", weights_dir, note)
            print(f"[{name}] exit {code} after {minutes:.1f} min")
            if code != 0 or not out.exists():
                raise SystemExit(f"step {name} failed (exit {code}); see {run_dir / f'run_{name}.log'}")
    if not args.dry_run:
        report = run_dir / ("image_eval_tta" if args.tta else "image_eval") / "image_eval.md"
        print(f"done. photo-level report: {report}")


if __name__ == "__main__":
    main()
