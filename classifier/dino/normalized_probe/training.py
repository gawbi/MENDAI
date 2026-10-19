from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch

from . import core as n

p = n.p


def device_setup(config, device):
    torch.set_num_threads(config["cpu_threads"])
    if device == "cuda":
        for attempt in range(30):
            try:
                p.gpu_guard(config, initial=True)
                break
            except RuntimeError:
                if attempt == 29:
                    raise
                time.sleep(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return torch.device(device)


def predict(head, features):
    head.eval()
    with torch.inference_mode():
        logits = head(features)
        probabilities = logits.sigmoid()
    return logits.cpu().numpy(), probabilities.cpu().numpy()


def train(run, protocol, device="cuda"):
    c, parts = protocol["config"], protocol["parts"]
    arrays, cache_id = n.validate_cache(run, protocol)
    target_device = device_setup(c, device)
    mean, scale = n.fit_scaler(arrays["train"])
    np.savez(run / "normalizer.npz", mean=mean, scale=scale)
    scaler_sha = p.file_sha(run / "normalizer.npz")
    p.write_json(run / "normalizer_metadata.json", {"fit_split": "train", "images": len(parts["train"]),
                                                   "views_per_image": 4, "dtype": "float32",
                                                   "std_floor": 1e-6, "sha256": scaler_sha,
                                                   "cache_identity": cache_id, "val_used_for_fit": False})
    original_head = torch.load(run / "original_head.pt", map_location="cpu", weights_only=True)
    mean_cpu, scale_cpu = torch.from_numpy(mean), torch.from_numpy(scale)
    mu, sigma = mean_cpu.to(target_device), scale_cpu.to(target_device)
    train_features = torch.from_numpy(np.array(arrays["train"], copy=True)).to(target_device)
    rows = parts["val"] + parts["val25"]
    val_features = torch.from_numpy(np.concatenate([arrays["val"], arrays["val25"]])).to(target_device)
    targets = torch.tensor([r["target"] for r in parts["train"]], dtype=torch.float32, device=target_device)
    val_targets = torch.tensor([r["target"] for r in rows], dtype=torch.float32, device=target_device)
    positives = targets.sum(0)
    if (positives == 0).any():
        raise ValueError("Missing positive training class")
    pos_weight = (len(targets)-positives) / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raw_initial = n.initialize_head(original_head, mean_cpu, scale_cpu, False, target_device)
    normalized_initial = n.initialize_head(original_head, mean_cpu, scale_cpu, True, target_device)
    with torch.inference_mode():
        raw_logits = raw_initial(val_features)
        normalized_logits = normalized_initial((val_features-mu)/sigma)
        initial_error = (raw_logits-normalized_logits).abs().max().item()
        torch.testing.assert_close(raw_logits, normalized_logits, rtol=2e-6, atol=5e-5)
    p.write_json(run / "initial_function_check.json", {"passed": True, "max_abs_logit_difference": initial_error,
                                                        "atol": 5e-5, "rtol": 2e-6,
                                                        "same_initial_decision_function": True})
    del raw_initial, normalized_initial
    started_all = time.monotonic()
    for variant in c["variants"]:
        destination = run / variant
        destination.mkdir(exist_ok=True)
        if (destination / "last.pt").exists():
            existing = torch.load(destination / "last.pt", map_location="cpu", weights_only=True)
            if existing["identity"] == protocol["identity"] and existing["cache_identity"] == cache_id and existing["epoch"] == c["epochs"] and existing["normalizer_sha256"] == scaler_sha:
                p.log(f"Verified completed {variant} head; skipping training")
                continue
            raise ValueError("Partial or incompatible head checkpoint; preserve it and use a fresh run")
        normalized = variant == "standardized"
        torch.manual_seed(c["seed"])
        head = n.initialize_head(original_head, mean_cpu, scale_cpu, normalized, target_device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=c["head_lr"], weight_decay=c["weight_decay"])
        scheduler = p.scheduler_for(optimizer, math.ceil(len(targets)/c["batch_size"]), c)
        val_x = (val_features-mu)/sigma if normalized else val_features
        history = []
        p.status(run, "training_head", variant=variant, completed_epoch=0, gpu_released=device != "cuda", head_device=device)
        n.record(run, f"{variant} 분류층 학습 중", "학습된 LoRA와 백본은 고정했습니다. 정규화 없음/있음은 같은 초기 판정 함수·증강 view·셔플·학습률·15 epoch를 사용합니다. 평균·표준편차는 train만 사용합니다.")
        ids = torch.arange(len(targets), device=target_device)
        for epoch in range(1, c["epochs"]+1):
            started = time.monotonic()
            choices = torch.from_numpy(p.view_indices(parts["train"], c["seed"], 4, epoch)).to(target_device)
            x = train_features[ids, choices]
            if normalized:
                x = (x-mu)/sigma
            permutation = torch.randperm(len(targets), generator=torch.Generator().manual_seed(c["seed"]+epoch)).to(target_device)
            total_loss = 0.
            head.train()
            for begin in range(0, len(targets), c["batch_size"]):
                batch = permutation[begin:begin+c["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x[batch]), targets[batch])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite head training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), c["clip_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(batch)
            logits, probabilities = predict(head, val_x)
            result = n.metrics(p.labels(rows), probabilities)
            with torch.inference_mode():
                val_loss = criterion(torch.from_numpy(logits).to(target_device), val_targets).item()
            item = {"epoch": epoch, "train_loss": total_loss/len(targets), "val_loss": val_loss,
                    "val_fixed_0_5": result, "seconds": time.monotonic()-started,
                    "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(item)
            p.atomic_torch_save({"identity": protocol["identity"], "cache_identity": cache_id,
                                 "normalizer_sha256": scaler_sha, "variant": variant,
                                 "epoch": epoch, "device": device,
                                 "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "history": history}, destination / "last.pt")
            p.write_json(destination / "history.json", history)
            if epoch == c["epochs"]:
                np.savez(destination / "last_validation_reference.npz", logits=logits, probabilities=probabilities)
            if device == "cuda":
                p.gpu_guard(c)
            p.status(run, "training_head", variant=variant, completed_epoch=epoch, loss=item["train_loss"])
            p.log(f"{c['experiment']} {variant} {epoch}/{c['epochs']} | train={item['train_loss']:.5f} val={val_loss:.5f} | BA@0.5={result['balanced_accuracy']:.4%} | {item['seconds']:.2f}s")
    p.status(run, "heads_trained", completed_epoch=c["epochs"], head_train_seconds=time.monotonic()-started_all)


def evaluate(run, protocol, device="cuda"):
    c, parts = protocol["config"], protocol["parts"]
    arrays, cache_id = n.validate_cache(run, protocol)
    target_device = device_setup(c, device)
    scaler_sha = p.file_sha(run / "normalizer.npz")
    with np.load(run / "normalizer.npz", allow_pickle=False) as archive:
        mean, scale = torch.from_numpy(archive["mean"]).to(target_device), torch.from_numpy(archive["scale"]).to(target_device)
    rows = parts["val"] + parts["val25"]
    features = torch.from_numpy(np.concatenate([arrays["val"], arrays["val25"]])).to(target_device)
    results = {"original": protocol["origin_metrics"]}
    for variant in c["variants"]:
        destination = run / variant
        checkpoint = torch.load(destination / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != protocol["identity"] or checkpoint["cache_identity"] != cache_id or checkpoint["epoch"] != c["epochs"] or checkpoint["normalizer_sha256"] != scaler_sha:
            raise ValueError("Final head identity/cache/scaler/epoch mismatch")
        head = torch.nn.Linear(c["feature_size"], 2).to(target_device)
        head.load_state_dict(checkpoint["head"])
        logits, probabilities = predict(head, (features-mean)/scale if variant == "standardized" else features)
        with np.load(destination / "last_validation_reference.npz", allow_pickle=False) as reference:
            np.testing.assert_allclose(logits, reference["logits"], rtol=0, atol=1e-6)
            np.testing.assert_allclose(probabilities, reference["probabilities"], rtol=0, atol=1e-6)
        fixed = n.metrics(p.labels(rows), probabilities)
        if fixed != checkpoint["history"][-1]["val_fixed_0_5"]:
            raise ValueError("Reloaded validation metrics differ")
        thresholds = n.choose_thresholds(p.labels(rows), probabilities)
        result = n.metrics(p.labels(rows), probabilities, thresholds)
        results[variant] = result
        p.write_json(destination / "metrics.json", {"val_combined": result})
        p.write_json(destination / "metrics_fixed_0_5.json", {"val_combined": fixed})
        p.write_json(destination / "thresholds.json", thresholds)
        n.save_predictions(destination / "val_combined_probs.csv", rows, probabilities)
        p.atomic_torch_save({"identity": protocol["identity"], "head": checkpoint["head"],
                             "normalizer_sha256": scaler_sha if variant == "standardized" else None}, destination / "head.pt")
        p.write_json(destination / "inference_config.json", {"identity": protocol["identity"], "origin_run": c["origin_run"],
                                                             "origin_checkpoint_sha256": protocol["identity"]["checkpoint_sha256"],
                                                             "variant": variant, "thresholds": thresholds, "head_dtype": "float32",
                                                             "lora_frozen": True, "normalizer": "../normalizer.npz" if variant == "standardized" else None})
        (destination / "report.md").write_text(n.render(result)+"\n", encoding="utf-8")
    p.write_json(run / "metrics.json", results)
    lines = [f"# {c['experiment']} 학습된 LoRA 고정 후 분류층 정규화", "", f"완료: {p.now()}. 원본: {c['origin_run']}. test 미사용.", "",
             "| 조건 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 |", "|---|---:|---:|---:|---:|---:|"]
    for variant, result in results.items():
        values = [result["balanced_accuracy"], result["accuracy"], *[result["recall"][name] for name in n.ORDER[:3]]]
        lines.append(f"| {variant} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    for variant, result in results.items():
        lines += ["", f"## {variant}", "", n.render(result)]
    lines += ["", "원본 대비 변화에는 분류층 추가 학습·고정 4-view·백본 eval 모드·FP32 분류층 차이가 함께 있습니다. 정규화 효과는 같은 조건인 raw_control 대 standardized로 비교합니다. 정규화는 선형 분류기의 표현력을 늘리지 않고 최적화·정규화 벌점의 작용을 바꿉니다. 평균·표준편차는 train만 사용했습니다. 기준 선택과 평가는 합산 val에서 수행한 개발 결과이며 독립적인 최종 test 성능이 아닙니다."]
    report = "\n".join(lines)+"\n"
    (run / "report.md").write_text(report, encoding="utf-8")
    p.status(run, "completed", completed_epoch=c["epochs"], completed_at=p.now(), gpu_release_pending=True,
             test_used=False, normalization_delta_ba=results["standardized"]["balanced_accuracy"]-results["raw_control"]["balanced_accuracy"])
    n.record(run, "정규화 비교·합산 val 평가 완료", report.split("\n", 1)[1].strip().replace("\n## ", "\n### "))
    p.log(f"{c['experiment']} complete | original={results['original']['balanced_accuracy']:.4%} raw={results['raw_control']['balanced_accuracy']:.4%} standardized={results['standardized']['balanced_accuracy']:.4%}")
