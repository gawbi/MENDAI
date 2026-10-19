"""CPU 데이터 점검과 단일 GPU 학습·헤드별 추론. 각 호출은 독립 프로세스다."""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import random
import time

import numpy as np

from . import common as c


class MemoryBudgetExceeded(RuntimeError):
    pass


def configure(run, gpu):
    import torch
    import ultralytics
    from ultralytics import settings
    from ultralytics import utils
    if ultralytics.__version__ != "8.4.171":
        raise ValueError("조사한 Ultralytics 버전과 다름")
    updates = dict(sync=False, weights_dir=str(run / "weights_cache"), runs_dir=str(c.ROOT / "runs/yolo"),
                   datasets_dir=str(c.ROOT / "dataset"), wandb=False, mlflow=False, comet=False,
                   clearml=False, dvc=False, neptune=False, raytune=False, tensorboard=False)
    settings.update({k: v for k, v in updates.items() if k in settings})
    # AMP 검사에서 함수 안으로 import하는 경로 상수도 workspace cache로 고정한다.
    utils.WEIGHTS_DIR = run / "weights_cache"
    torch.set_num_threads(1)
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("승인된 GPU 단계에 CUDA 없음")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(6656 * 1024**2 / total, .90), 0)
        torch.cuda.reset_peak_memory_stats()
    elif torch.cuda.is_available():
        raise RuntimeError("preflight는 CUDA_VISIBLE_DEVICES=-1 필요")
    return torch


def preflight(run):
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from PIL import Image
    parts = c.verify_data(run)
    model = YOLO(str(c.MENDI / "weights/yolo26m.pt"))
    head = model.model.model[-1]
    if not hasattr(head, "one2one_cv2") or not hasattr(head, "cv2"):
        raise ValueError("두 헤드 사전학습 가중치가 아님")
    try:
        metadata.version("albumentations")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("예상 밖 Albumentations 설치: 기본 열화 증강을 다시 검토해야 함")
    for rows in parts.values():
        for row in rows:
            with Image.open(row["path"]) as picture:
                if picture.getexif().get(274, 1) != 1:
                    raise ValueError("EXIF 회전이 남아 있음")
    cfg = get_cfg(overrides=c.training_arguments(run, 2))
    data = {"names": c.NAMES, "nc": 2, "channels": 3}
    dataset = build_yolo_dataset(cfg, str(c.DATA / "dataset_damage_v5_1024/images/train"), 2, data, "train")
    if len(dataset) != 4584:
        raise ValueError("YOLO train 파일 수 불일치")
    expected = {r["file"] for r in parts["train"]}
    if {Path(x).name for x in dataset.im_files} != expected:
        raise ValueError("YOLO loader에 다른 split 포함")
    mosaic = dataset.transforms.transforms[0].transforms[0]
    if type(mosaic).__name__ != "Mosaic" or mosaic.p != .2:
        raise ValueError("Mosaic 실제 파이프라인 불일치")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    mosaic.p = 1.0  # 미리보기 표본만 강제로 합성. 본 학습의 .2는 변경하지 않는다.
    selected = c.read(run / "smoke_rows.json")
    lookup = {Path(path).name: i for i, path in enumerate(dataset.im_files)}
    crops, evidence = [], []
    for row in selected[:8]:
        sample = dataset[lookup[row["file"]]]
        image = sample["img"].permute(1, 2, 0).numpy().copy()  # Format: RGB CHW
        boxes = sample["bboxes"].numpy()
        if image.shape != (1024, 1024, 3) or not np.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
            raise ValueError("증강 이미지·박스 범위 오류")
        evidence.append({"file": row["file"], "boxes_after": len(boxes)})
        for box in boxes[:2]:
            x, y, w, h = box * 1024
            x1, y1, x2, y2 = int(max(0, x-w/2-8)), int(max(0, y-h/2-8)), int(min(1024, x+w/2+8)), int(min(1024, y+h/2+8))
            if x2 > x1 and y2 > y1:
                crops.append(cv2.resize(image[y1:y2, x1:x2], (192, 192)))
    if not crops:
        raise ValueError("검토할 증강 손상 crop 없음")
    crops = crops[:16]
    while len(crops) % 4:
        crops.append(np.full((192, 192, 3), 114, dtype=np.uint8))
    grid = np.concatenate([np.concatenate(crops[i:i+4], axis=1) for i in range(0, len(crops), 4)], axis=0)
    Image.fromarray(grid).save(run / "augmentation_damage_crops.png")
    mosaic.p = .2
    c.verify_data(run, images=False)
    c.write(run / "preflight.json", {"completed_at": c.now(), "gpu_used": False, "test_read": False,
                                    "ultralytics": metadata.version("ultralytics"), "torch": torch.__version__,
                                    "train_images": len(dataset), "transform": repr(dataset.transforms),
                                    "preview_only_forced_mosaic": True, "examples": evidence,
                                    "dual_heads_present": True, "config_validated": True,
                                    "preview_sha256": c.sha(run / "augmentation_damage_crops.png")})
    c.log("CPU loader·두 헤드·증강·학습 인자 점검 완료. 검토용 손상 crop 저장")


def train(run, batch, smoke):
    import torch
    from ultralytics import YOLO
    c.verify_data(run)
    started = time.perf_counter()
    epochs = 1 if smoke else 50
    output = run / (f"smoke_b{batch}" if smoke else "training")
    if output.exists():
        raise ValueError("기존 학습 run을 덮어쓰지 않음")
    steps = {"updates": 0}

    def budget(trainer):
        if trainer.batch_size != batch:
            raise RuntimeError("본 실험 중 물리 배치 자동 변경 금지")
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        if reserved > 6656:
            raise MemoryBudgetExceeded(f"GPU memory budget exceeded: {reserved:.1f}MiB")
        if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
            raise RuntimeError("학습 loss가 유한하지 않음")

    def start_epoch(trainer):
        # 설치본의 첫 epoch 자동 batch 감소는 optimizer 상태를 유지하므로 비활성화.
        # 점검 실패 시 별도 프로세스에서 batch1·새 COCO 초기화로 재점검한다.
        trainer._oom_retries = 3
        if trainer.batch_size != batch:
            raise RuntimeError("물리 배치 변경")

    def progress(trainer):
        budget(trainer)
        steps["updates"] = trainer.ema.updates
        c.write(run / (f"smoke_b{batch}_progress.json" if smoke else "training_progress.json"),
                {"at": c.now(), "completed_epoch": min(trainer.epoch+1, epochs), "epochs": epochs,
                 "seconds": time.perf_counter()-started, "ema_updates": steps["updates"],
                 "reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
                 "allocated_mib": torch.cuda.max_memory_allocated()/1024**2})

    model = YOLO(str(c.MENDI / "weights/yolo26m.pt"))
    model.add_callback("on_train_epoch_start", start_epoch)
    model.add_callback("on_train_batch_end", budget)
    model.add_callback("on_fit_epoch_end", progress)
    model.train(**c.training_arguments(run, batch, smoke))
    checkpoint = output / "weights/last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    last = saved.get("ema") if saved.get("ema") is not None else saved.get("model")
    if last is None or {int(k): v for k, v in last.names.items()} != c.NAMES:
        raise ValueError("마지막 EMA 클래스 이름 불일치")
    if not hasattr(last.model[-1], "one2one_cv2") or not hasattr(last.model[-1], "cv2"):
        raise ValueError("마지막 EMA에 두 헤드 누락")
    rows = list(__import__("csv").DictReader((output / "results.csv").open(encoding="utf-8")))
    if len(rows) != epochs or int(float(rows[-1]["epoch"])) != epochs:
        raise ValueError("예정 epoch 미완료")
    if smoke and steps["updates"] < 2:
        raise ValueError("점검에서 optimizer 갱신 두 번 미완료")
    # 실제 저장 모델을 재로딩해 출처와 구조도 확인한다.
    reloaded = YOLO(str(checkpoint))
    if {int(k): v for k, v in reloaded.names.items()} != c.NAMES:
        raise ValueError("저장 모델 재로딩 실패")
    c.verify_data(run, images=False)
    artifact = {"completed_at": c.now(), "epochs": epochs, "batch": batch, "ema_updates": steps["updates"],
                "checkpoint": str(checkpoint), "checkpoint_sha256": c.sha(checkpoint),
                "seconds": time.perf_counter()-started, "reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
                "allocated_mib": torch.cuda.max_memory_allocated()/1024**2, "saved_model_reloaded": True,
                "test_evaluated": False, "protocol_sha256": c.load(run)["sha256"]}
    c.write(run / (f"smoke_b{batch}_complete.json" if smoke else "training_complete.json"), artifact)
    c.log(f"{'점검' if smoke else '본 학습'} 완료: batch{batch}, {epochs} epoch")


def predict_head(run, nms):
    import cv2
    import torch
    from ultralytics import YOLO
    parts = c.verify_data(run)
    rows = parts["val"] + parts["val25"]
    trained = c.read(run / "training_complete.json")
    checkpoint = Path(trained["checkpoint"])
    if c.sha(checkpoint) != trained["checkpoint_sha256"]:
        raise ValueError("최종 가중치 변경")
    head_name = "nms" if nms else "free"
    output = run / head_name
    output.mkdir(exist_ok=False)
    model = YOLO(str(checkpoint))
    arguments = dict(imgsz=1024, rect=False, device="0", quantize=32, conf=.001, iou=.7,
                     max_det=300, nms=None if nms else False, augment=False, agnostic_nms=False,
                     verbose=False, save=False, save_txt=False, plots=False, batch=1)
    started = time.perf_counter()
    scores = {}
    saturated = {"original": 0, "flip": 0}
    for view in ("original", "flip", "reproduced"):
        collected = []
        with (output / f"{view}_boxes.jsonl").open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                if c.sha(row["path"]) != row["sha256"]:
                    raise ValueError("추론 이미지 변경")
                image = cv2.imdecode(np.fromfile(row["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"이미지 디코딩 실패: {row['file']}")
                if view == "flip":
                    image = np.ascontiguousarray(image[:, ::-1])
                results = model.predict(image, **arguments)
                if len(results) != 1 or results[0].boxes is None:
                    raise ValueError("사진 추론 결과 누락")
                backend = model.predictor.model
                if bool(backend.end2end) != (not nms) or backend.fp16:
                    raise ValueError("실제 추론 head·FP32 설정 불일치")
                result = results[0]
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                score = c.photo_score(cls, conf, result.names)
                collected.append(score)
                xyxy = result.boxes.xyxy.cpu().numpy().tolist()
                stream.write(json.dumps({"file": row["file"], "status": "ok", "view": view,
                                         "shape": list(image.shape), "classes": cls.tolist(),
                                         "confidences": conf.tolist(), "xyxy": xyxy})+"\n")
                if view in saturated and len(cls) >= 300:
                    saturated[view] += 1
                if (index+1) % 128 == 0:
                    c.log(f"{head_name} {view}: {index+1}/{len(rows)}")
        scores[view] = c.validated_scores(collected)
        c.save_scores(output / f"{view}_probs.csv", rows, scores[view])
    np.testing.assert_allclose(scores["original"], scores["reproduced"], atol=1e-4, rtol=0)
    averaged = c.mean_scores(scores["original"], scores["flip"])
    c.save_scores(output / "tta_probs.csv", rows, averaged)
    np.testing.assert_array_equal(c.read_scores(output / "tta_probs.csv", rows), averaged)
    c.verify_data(run, images=False)
    evidence = {"completed_at": c.now(), "checkpoint_sha256": trained["checkpoint_sha256"], "head": head_name,
                "views": ["original", "horizontal_flip_before_letterbox"], "arguments": arguments,
                "seconds": time.perf_counter()-started, "images": len(rows), "max_det_reached": saturated,
                "reproduction_max_abs_error": float(np.max(np.abs(scores["original"]-scores["reproduced"]))),
                "reserved_mib": torch.cuda.max_memory_reserved()/1024**2, "test_evaluated": False,
                "files": {p.name: c.sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    c.write(output / "complete.json", evidence)


def execute(run, stage, batch):
    c.load(run)
    configure(run, gpu=stage != "preflight")
    if stage == "preflight":
        preflight(run)
    elif stage in ("smoke", "train"):
        try:
            train(run, batch, smoke=stage == "smoke")
        except Exception as exc:
            text = str(exc).lower()
            memory = any(word in text for word in ("out of memory", "memory budget exceeded", "cublas_status_alloc_failed"))
            c.write(run / f"{stage}_b{batch}_failure.json", {"at": c.now(), "error": f"{type(exc).__name__}: {exc}", "memory": memory})
            if stage == "smoke" and memory:
                raise SystemExit(3)
            raise
    else:
        predict_head(run, nms=stage == "head-nms")
