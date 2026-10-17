"""Shared bits for the DETR scripts: class names, the prediction CSV format, keep-awake."""

from __future__ import annotations

import contextlib
import csv
import ctypes
import os
import sys
from pathlib import Path

# Box classes of the v5 dataset (same order as data.yaml `names`).
CLASS_NAMES = {0: "outsole", 1: "heel_lining"}
CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items()}

# Columns of the prediction CSV written by predict_rtdetr.py and predict_rfdetr.py.
# `file` is the image file name only; coordinates are pixels in the input image.
PRED_COLUMNS = ["file", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    """All images directly inside `folder`, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def prediction_row(file_name, class_id, score, xyxy):
    class_id = int(class_id)
    x1, y1, x2, y2 = (round(float(v), 1) for v in xyxy)
    return [file_name, class_id, CLASS_NAMES.get(class_id, str(class_id)), round(float(score), 5), x1, y1, x2, y2]


def write_predictions(path, rows):
    """Write prediction rows (see PRED_COLUMNS) to a CSV and return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PRED_COLUMNS)
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


@contextlib.contextmanager
def keep_awake():
    """Keep Windows from sleeping while the block runs. The display may still turn off.

    Uses SetThreadExecutionState, so the request ends with the process; no power
    plan setting is changed.
    """
    if sys.platform != "win32":
        yield
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


def chdir_weights(weights_dir):
    """Ultralytics and rfdetr download pretrained weights into the current directory.

    Changing into a folder outside the repo keeps those files (and any helper
    downloads) from landing next to the code.
    """
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        os.chdir(weights_dir)
