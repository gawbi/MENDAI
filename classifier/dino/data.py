"""확정본의 train/val만 읽고, 사진 전체를 보존한다."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

ORDER = ["normal", "heel", "outsole", "both"]
TARGETS = {"normal": [0, 0], "heel": [1, 0], "outsole": [0, 1], "both": [1, 1]}
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def inventory(root: Path, hash_images=True):
    """test 이미지는 열지 않고 CSV·manifest의 train/val 행만 유지한다."""
    root = root.resolve()
    parts = {}
    for family, selections in [("v5", ("train", "val")), ("eval25", ("val",))]:
        dataset = root / f"dataset_damage_{family}_1024"
        with (root / "photo_labels" / f"damage_image_labels_{family}.csv").open(encoding="utf-8-sig", newline="") as f:
            selected = [dict(file=r["file"], split=r["split"], label=r["label"]) for r in csv.DictReader(f) if r["split"] in selections]
        metadata = json.loads((dataset / "manifest.json").read_text(encoding="utf-8-sig"))
        entries = {}
        for r in metadata["images"]:
            if r["split"] not in selections:
                continue
            key = (r["split"], r["file"])
            if key in entries:
                raise ValueError("manifest 중복")
            entries[key] = {k: str(r[k]) for k in ("session", "group")}
        for split in selections:
            name = "val25" if family == "eval25" else split
            rows, seen = [], set()
            for r in selected:
                if r["split"] != split:
                    continue
                filename = r["file"]
                if Path(filename).name != filename or filename in seen or r["label"] not in TARGETS:
                    raise ValueError("파일명·정답·중복 오류")
                seen.add(filename)
                path = dataset / "images" / split / filename
                meta = entries[(split, filename)]
                row = {**r, **meta, "family": family, "part": name, "path": str(path), "target": TARGETS[r["label"]]}
                # 이미지 내용까지 고정한다. test 파일은 열지 않는다.
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if hash_images else str(path.stat().st_size)
                rows.append(row)
            disk = {p.name for p in (dataset / "images" / split).iterdir() if p.is_file()}
            if disk != seen or {key[1] for key in entries if key[0] == split} != seen:
                raise ValueError(f"{name}: 이미지·CSV·manifest 불일치")
            parts[name] = sorted(rows, key=lambda r: r["file"])
    for a, b in [("train", "val"), ("train", "val25"), ("val", "val25")]:
        if {r["file"] for r in parts[a]} & {r["file"] for r in parts[b]}:
            raise ValueError(f"{a}/{b}: 파일 중복")
        if {r["file"][:8] for r in parts[a]} & {r["file"][:8] for r in parts[b]}:
            raise ValueError(f"{a}/{b}: 촬영일 중복")
        if parts[a][0]["family"] == parts[b][0]["family"]:
            for key in ("session", "group"):
                if {r[key] for r in parts[a]} & {r[key] for r in parts[b]}:
                    raise ValueError(f"{a}/{b}: {key} 중복")
    expected = {"train": 4584, "val": 639, "val25": 680}
    if {k: len(v) for k, v in parts.items()} != expected:
        raise ValueError("2026-10-17 확정본의 사진 수와 다름")
    manifest = {k: [{x: r[x] for x in ("file", "label", "session", "group", "sha256")} for r in v] for k, v in parts.items()}
    summary = {k: {"images": len(v), "labels": dict(Counter(r["label"] for r in v))} for k, v in parts.items()}
    return parts, {"sha256": digest(manifest), "summary": summary, "images": manifest}


def preprocess(image, size=1022, augment=False):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = size / max(image.size)
    dimensions = tuple(max(1, round(v * scale)) for v in image.size)
    image = image.resize(dimensions, Image.Resampling.LANCZOS)
    if augment:
        if random.random() < .5:
            image = ImageOps.mirror(image)
        for enhancer in random.sample([ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color], 3):
            image = enhancer(image).enhance(random.uniform(.9, 1.1))
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    pixels = torch.from_numpy(np.array(canvas, dtype=np.float32)).permute(2, 0, 1) / 255
    return (pixels - MEAN) / STD


class Photos(Dataset):
    def __init__(self, rows, size=1022, augment=False):
        self.rows, self.size, self.augment = rows, size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            pixels = preprocess(image, self.size, self.augment)
        return pixels, torch.tensor(row["target"], dtype=torch.float32)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
