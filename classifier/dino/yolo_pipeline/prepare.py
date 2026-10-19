"""CPU에서 train/val 출처와 실행 범위를 고정한다."""
from __future__ import annotations

from collections import Counter
import shutil

from . import common as c


def smoke_rows(rows):
    chosen = {}
    # 양쪽 손상·정상·both를 포함하고 클래스별 파일명 구간을 넓게 잡는다.
    import numpy as np
    for label in c.ORDER:
        subset = [r for r in rows if r["label"] == label]
        for i in np.linspace(0, len(subset)-1, min(24, len(subset)), dtype=int):
            chosen[subset[i]["file"]] = subset[i]
    small = []
    for row in rows:
        label = c.DATA / "dataset_damage_v5_1024/labels/train" / (row["file"].rsplit(".", 1)[0]+".txt")
        boxes = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if boxes:
            small.append((min(min(float(x[3]), float(x[4])) for x in boxes), row["file"], row))
    for _, name, row in sorted(small)[:16]:
        chosen[name] = row
    for i in np.linspace(0, len(rows)-1, len(rows), dtype=int):
        if len(chosen) >= 128:
            break
        chosen.setdefault(rows[i]["file"], rows[i])
    if len(chosen) != 128:
        raise ValueError("점검 128장 선택 실패")
    return sorted(chosen.values(), key=lambda r: r["file"])


def prepare(run):
    import torch
    from ..data import inventory
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("준비는 CUDA_VISIBLE_DEVICES=-1에서 실행")
    run = c.safe_run(run)
    run.mkdir(parents=True, exist_ok=False)
    c.status(run, "preparing_cpu", gpu_work_started=False)
    original = c.read(c.DINO_CHAIN / "protocol.json")
    for name, expected in original["sources"].items():
        if c.sha(c.ROOT / name) != expected:
            raise ValueError("진행 중 DINO 소스 변경")
    parts, data = inventory(c.DATA)
    if data["sha256"] != c.DATA_SHA:
        raise ValueError("DINO와 YOLO 데이터 불일치")
    c.write(run / "parts.json", parts)
    boxes = {}
    counts = Counter()
    for split in ("train", "val"):
        for row in parts[split]:
            path = c.DATA / f"dataset_damage_v5_1024/labels/{split}" / (row["file"].rsplit(".", 1)[0]+".txt")
            boxes[path.relative_to(c.DATA).as_posix()] = c.sha(path)
            present = set()
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                fields = list(map(float, line.split()))
                if len(fields) != 5 or fields[0] not in (0, 1) or any(not 0 <= v <= 1 for v in fields[1:]) or min(fields[3:]) <= 0:
                    raise ValueError(f"박스 라벨 범위·형식 오류: {path.name}")
                present.add(int(fields[0]))
                if split == "train":
                    counts[int(fields[0])] += 1
            expected = {"normal": set(), "heel": {1}, "outsole": {0}, "both": {0, 1}}[row["label"]]
            if present != expected:
                raise ValueError(f"박스와 사진 정답 불일치: {path.name}")
    if counts != {0: 1054, 1: 4150}:
        raise ValueError("train 박스 수 변경")
    base = c.DATA / "dataset_damage_v5_1024"
    dataset = {"path": str(base), "train": "images/train", "val": "images/val", "names": c.NAMES}
    c.write(run / "data.yaml", dataset)  # JSON은 YAML의 유효한 부분집합이다.
    smoke = smoke_rows(parts["train"])
    (run / "smoke_images.txt").write_text("\n".join(r["path"] for r in smoke)+"\n", encoding="utf-8")
    c.write(run / "smoke.yaml", {**dataset, "train": str(run / "smoke_images.txt")})
    c.write(run / "smoke_rows.json", smoke)
    inputs = {str(c.MENDI / "weights/yolo26m.pt"): c.sha(c.MENDI / "weights/yolo26m.pt"),
              str(c.DINO_CHAIN / "protocol.json"): c.sha(c.DINO_CHAIN / "protocol.json")}
    for name in ("data.yaml", "smoke.yaml", "smoke_images.txt", "smoke_rows.json"):
        inputs[str(run / name)] = c.sha(run / name)
    package = {p.relative_to(c.YOLO_SITE).as_posix(): c.sha(p) for p in sorted(c.YOLO_SITE.rglob("*.py"))}
    package["cfg/default.yaml"] = c.sha(c.YOLO_SITE / "cfg/default.yaml")
    protocol = {"created_at": c.now(), "run": str(run), "scope": {"through_step": 9, "detr": False, "test": False},
                "wait_for_wrapper_pid": 2184, "dino_chain": str(c.DINO_CHAIN), "data_sha256": data["sha256"],
                "parts_sha256": c.sha(run / "parts.json"), "box_labels": boxes,
                "sources": {p.relative_to(c.ROOT).as_posix(): c.sha(p) for p in c.source_paths()},
                "ultralytics_sources": package, "ultralytics_version": "8.4.171", "inputs": inputs,
                "train_args": c.TRAIN_ARGS, "max_reserved_mib": 6656, "min_free_mib": 6656,
                "yolo_python": str(c.YOLO_PYTHON), "dino_python": str(c.DINO_PYTHON)}
    protocol["sha256"] = c.digest(protocol)
    c.write(run / "protocol.json", protocol)
    for source in c.source_paths():
        target = run / "source_snapshot" / source.relative_to(c.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # AMP 사전 점검의 작은 COCO 모델을 workspace cache에 준비한다.
    cache = run / "weights_cache"
    cache.mkdir()
    shutil.copyfile(c.MENDI / "weights/yolo26n.pt", cache / "yolo26n.pt")
    c.status(run, "prepared", gpu_work_started=False, train_images=4584, val_images=1319)
    c.log(f"CPU 준비 완료: {run}; GPU·test 사용 없음")
