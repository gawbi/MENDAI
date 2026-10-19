"""소스 봉인, 공식 LoRA+ optimizer, 같은 전처리의 좌우 반전."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from peft.optimizers import create_loraplus_optimizer

from .. import train as t
from ..data import digest, preprocess

ROOT = t.REPO
RECORD = ROOT / "classifier/dino/EXPERIMENTS.md"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sources():
    own = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.cmd"))
    shared = list(Path(t.__file__).parent.glob("*.py")) + list(Path(t.__file__).parent.joinpath("tools").glob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(own + shared)}


def safe_run(path):
    path = Path(path).resolve()
    path.relative_to((ROOT / "runs/dino").resolve())
    return path


def plus_config(origin, ratio=16):
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("LoRA+ 비율은 1보다 큰 유한수여야 함")
    return {**origin, "experiment": "EX-07", "optimizer": "loraplus", "loraplus_lr_ratio": ratio}


def optimizer(model, config):
    if config.get("optimizer") != "loraplus":
        raise ValueError("LoRA+ 설정 누락")
    ratio = config["loraplus_lr_ratio"]
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("잘못된 LoRA+ 비율")
    # 공식 helper는 백본에만 적용한다. 별도 head LR와 기존 weight decay를 보존한다.
    result = create_loraplus_optimizer(
        model.backbone, torch.optim.AdamW, lr=config["lora_lr"], loraplus_lr_ratio=ratio,
        weight_decay=config["weight_decay"], loraplus_weight_decay=config["weight_decay"])
    for group, name in zip(result.param_groups, ("lora_A", "embedding", "lora_B", "lora_B_no_decay")):
        group["group_name"] = name
    result.add_param_group({"params": list(model.head.parameters()), "lr": config["head_lr"],
                            "weight_decay": config["weight_decay"], "group_name": "head"})
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in result.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer에 학습 파라미터 누락·중복")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = next(g for g in result.param_groups if any(p is parameter for p in g["params"]))
        expected_lr = config["head_lr"] if name.startswith("head.") else config["lora_lr"] * (ratio if "lora_B" in name else 1)
        if group["lr"] != expected_lr or group["weight_decay"] != config["weight_decay"]:
            raise ValueError(f"학습률 또는 weight decay 불일치: {name}")
    return result


def optimizer_manifest(model, config):
    result = optimizer(model, config)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [{"name": group["group_name"], "lr": group["lr"], "weight_decay": group["weight_decay"],
             "parameters": [names[id(parameter)] for parameter in group["params"]],
             "parameter_count": sum(parameter.numel() for parameter in group["params"])} for group in result.param_groups]


class FlippedPhotos(Dataset):
    """EXIF 회전 후 사진을 반전한다. resize·패딩·정규화는 원래 함수와 같다."""
    def __init__(self, rows, size):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = Path(row["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("TTA 사진 해시 불일치")
        with Image.open(io.BytesIO(payload)) as image:
            image = ImageOps.mirror(ImageOps.exif_transpose(image).convert("RGB"))
            pixels = preprocess(image, self.size, augment=False)
        return pixels, torch.tensor(row["target"], dtype=torch.float32)


def mean_probabilities(original, flipped):
    original, flipped = np.asarray(original), np.asarray(flipped)
    if (original.shape != flipped.shape or original.ndim != 2 or original.shape[1] != 2
            or not np.isfinite(original).all() or not np.isfinite(flipped).all()
            or np.any((original < 0) | (original > 1)) or np.any((flipped < 0) | (flipped > 1))):
        raise ValueError("TTA 점수 형식 오류")
    return (original + flipped) / 2


def prepare(chain, origin, plus, wait_pid):
    chain, origin, plus = map(safe_run, (chain, origin, plus))
    if (chain / "protocol.json").exists():
        raise ValueError("후속 실행이 이미 준비되어 있음")
    protocol = t.read_json(origin / "protocol.json")
    if t.source_hash() != protocol["identity"]["source_sha256"]:
        raise ValueError("진행 중인 EX-06 소스와 현재 소스가 다름")
    selection = t.read_json(origin / "selection.json")
    config = plus_config(protocol["config"])
    parts, plus_protocol, _ = t.prepare(config, plus, protocol["backbone_path"])
    if plus_protocol["identity"]["data_sha256"] != protocol["identity"]["data_sha256"]:
        raise ValueError("기존 실험과 데이터 해시 불일치")
    assert {key for key in config if config[key] != protocol["config"].get(key)} == {"experiment", "optimizer", "loraplus_lr_ratio"}
    import peft.optimizers.loraplus as official
    source_files = sources()
    value = {"created_at": t.now(), "origin_run": str(origin), "plus_run": str(plus), "wait_for_pid": wait_pid,
             "origin_identity": protocol["identity"], "origin_protocol_sha256": sha(origin / "protocol.json"),
             "origin_selection_sha256": sha(origin / "selection.json"), "plus_identity": plus_protocol["identity"],
             "sources": source_files, "source_sha256": digest(source_files),
             "official_optimizer_path": official.__file__, "official_optimizer_sha256": sha(official.__file__),
             "backbone_sha256": sha(Path(protocol["backbone_path"]) / "model.safetensors"),
             "tta": {"views": ["original", "horizontal_flip_before_resize_and_pad"], "aggregation": "mean of two sigmoid probabilities per output",
                     "threshold_selection": "same combined val grid as origin; separate thresholds for each variant"},
             "config": config, "selection": {key: selection[key] for key in ("batch", "accumulation", "checkpointing")},
             "test_evaluated": False}
    chain.mkdir(parents=True, exist_ok=True)
    value["sha256"] = digest(value)
    t.write_json(chain / "protocol.json", value)
    t.write_json(plus / "extension_protocol.json", {"chain_run": str(chain), "chain_sha256": value["sha256"]})
    for name in source_files:
        target = chain / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (chain / "source_snapshot/official_loraplus.py").write_bytes(Path(official.__file__).read_bytes())
    t.write_json(chain / "status.json", {"stage": "prepared", "time": t.now(), "gpu_work_started": False, "test_evaluated": False})
    t.log(f"후속 실행 준비 완료: {chain}; LoRA+ A={config['lora_lr']}, B={config['lora_lr']*config['loraplus_lr_ratio']}, head={config['head_lr']}")
    return value


def load(chain):
    chain = safe_run(chain)
    value = t.read_json(chain / "protocol.json")
    if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("후속 protocol 해시 불일치")
    if value["sources"] != sources() or sha(value["official_optimizer_path"]) != value["official_optimizer_sha256"]:
        raise ValueError("봉인 후 후속 코드·공식 optimizer 변경")
    for name, expected in value["sources"].items():
        if sha(chain / "source_snapshot" / name) != expected:
            raise ValueError("소스 snapshot 불일치")
    origin, plus = Path(value["origin_run"]), Path(value["plus_run"])
    if sha(origin / "protocol.json") != value["origin_protocol_sha256"] or sha(origin / "selection.json") != value["origin_selection_sha256"]:
        raise ValueError("기존 실행 protocol·배치 변경")
    if t.read_json(plus / "protocol.json")["identity"] != value["plus_identity"]:
        raise ValueError("LoRA+ 실행 identity 변경")
    return value
