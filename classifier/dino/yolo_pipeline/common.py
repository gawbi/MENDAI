"""실행 범위, 출처, 사진별 두 점수의 무결성 검사."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset/mendi_damage_v5"
MENDI = ROOT.parent / "Mendi"
DINO_PYTHON = MENDI / ".venv-dino/Scripts/python.exe"
YOLO_PYTHON = MENDI / ".venv/Scripts/python.exe"
YOLO_SITE = MENDI / ".venv/Lib/site-packages/ultralytics"
DINO_CHAIN = ROOT / "runs/dino/LORA-TTA_20261018"
ORDER = ["normal", "heel", "outsole", "both"]
DATA_SHA = "fd197304188b961d1d2097966789205b7c854d298f90dbc0ea3d09eb1843d856"
NAMES = {0: "outsole", 1: "heel_lining"}
TRAIN_ARGS = dict(task="detect", imgsz=1024, epochs=50, patience=0, seed=0,
                  deterministic=True, batch=2, nbs=64, optimizer="AdamW", lr0=.001,
                  momentum=.9, weight_decay=.0005, warmup_epochs=3., warmup_bias_lr=0.,
                  cos_lr=True, lrf=.01, freeze=None, box=7.5, cls=.5, dfl=1.5,
                  cls_pw=0., amp=True, workers=1, cache=False, rect=False,
                  multi_scale=0., mosaic=.2, close_mosaic=10, scale=.2, translate=.05,
                  fliplr=.5, hsv_h=.01, hsv_s=.3, hsv_v=.2, degrees=0., flipud=0.,
                  shear=0., perspective=0., mixup=0., cutmix=0., copy_paste=0., bgr=0.,
                  save=True, save_period=-1, val=True, split="val", plots=False,
                  exist_ok=False, pretrained=True, device="0", verbose=True)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def safe_run(path):
    path = Path(path).resolve()
    path.relative_to((ROOT / "runs/yolo").resolve())
    if path == (ROOT / "runs/yolo").resolve():
        raise ValueError("실행별 하위 폴더 필요")
    return path


def source_paths():
    paths = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.cmd"))
    # 현재 DINO의 봉인된 공유 코드에는 새 파일을 추가하지 않는다.
    paths += [ROOT / name for name in read(DINO_CHAIN / "protocol.json")["sources"]]
    return sorted(set(paths))


def load(run):
    run = safe_run(run)
    protocol = read(run / "protocol.json")
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    if digest(payload) != protocol["sha256"] or protocol["run"] != str(run):
        raise ValueError("YOLO protocol 변경")
    if protocol["scope"] != {"through_step": 9, "detr": False, "test": False}:
        raise ValueError("승인 범위 변경")
    for name, expected in protocol["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"실행 소스 변경: {name}")
    for name, expected in protocol["ultralytics_sources"].items():
        if sha(YOLO_SITE / name) != expected:
            raise ValueError(f"Ultralytics 변경: {name}")
    for name, expected in protocol["inputs"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"고정 입력 변경: {name}")
    if sha(run / "parts.json") != protocol["parts_sha256"]:
        raise ValueError("사진 manifest 변경")
    return protocol


def parts(run):
    result = read(Path(run) / "parts.json")
    if set(result) != {"train", "val", "val25"}:
        raise ValueError("train·val 외 split 포함")
    expected = {"train": 4584, "val": 639, "val25": 680}
    seen = set()
    for split, rows in result.items():
        if len(rows) != expected[split]:
            raise ValueError("사진 수 불일치")
        for row in rows:
            path = Path(row["path"]).resolve()
            path.relative_to(DATA.resolve())
            if path.parent.name != ("train" if split == "train" else "val"):
                raise ValueError("test 또는 예상 밖 경로")
            if row["file"] != path.name or row["file"] in seen:
                raise ValueError("중복·파일명 오류")
            seen.add(row["file"])
    return result


def verify_data(run, images=True):
    protocol = load(run)
    result = parts(run)
    if images:
        for rows in result.values():
            for row in rows:
                if sha(row["path"]) != row["sha256"]:
                    raise ValueError(f"사진 변경: {row['file']}")
    for name, expected in protocol["box_labels"].items():
        if sha(DATA / name) != expected:
            raise ValueError(f"박스 라벨 변경: {name}")
    return result


def validated_scores(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("사진 점수 크기·유한성 오류")
    if (value < 0).any() or (value > 1).any():
        raise ValueError("사진 점수 범위 오류")
    return value


def mean_scores(first, second):
    first, second = validated_scores(first), validated_scores(second)
    if first.shape != second.shape:
        raise ValueError("모델별 사진 수 불일치")
    return (first + second) / 2


def photo_score(classes, confidences, names):
    names = {int(k): v for k, v in names.items()}
    if names != NAMES or len(classes) != len(confidences):
        raise ValueError("탐지 클래스 이름·박스 수 오류")
    output = [0., 0.]
    for cls, confidence in zip(classes, confidences):
        if float(cls) != int(cls) or int(cls) not in names or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("탐지 클래스·점수 오류")
        index = ["heel_lining", "outsole"].index(names[int(cls)])
        output[index] = max(output[index], float(confidence))
    return output


def save_scores(path, rows, scores):
    scores = validated_scores(scores)
    if len(rows) != len(scores):
        raise ValueError("사진·점수 수 불일치")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "p_heel_lining", "p_outsole"])
        writer.writerows((r["file"], *map(float, p)) for r, p in zip(rows, scores))


def read_scores(path, rows):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"file", "p_heel_lining", "p_outsole"}:
            raise ValueError("사진 점수 CSV 열 오류")
        loaded = list(reader)
    names = [r["file"] for r in loaded]
    expected = [r["file"] for r in rows]
    if len(set(names)) != len(names) or set(names) != set(expected) or len(names) != len(expected):
        raise ValueError("사진 점수 누락·중복·예상 밖 파일")
    lookup = {r["file"]: [r["p_heel_lining"], r["p_outsole"]] for r in loaded}
    return validated_scores([lookup[name] for name in expected])


def status(run, stage, **extra):
    write(Path(run) / "status.json", {"stage": stage, "updated_at": now(), "pid": os.getpid(),
                                       "test_evaluated": False, "detr_used": False, **extra})


def training_arguments(run, batch, smoke=False):
    run = safe_run(run)
    args = {**TRAIN_ARGS, "batch": batch, "data": str(run / ("smoke.yaml" if smoke else "data.yaml")),
            "project": str(run), "name": f"smoke_b{batch}" if smoke else "training"}
    if smoke:
        args.update(epochs=1, close_mosaic=0, warmup_epochs=0., plots=False)
    return args
