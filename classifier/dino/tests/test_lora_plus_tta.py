"""추가 GPU 없이 LoRA+ 학습률·재개, TTA 전처리·순서·실패 중단을 검증한다."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model
from transformers import DINOv3ViTConfig

from classifier.dino.data import preprocess
from classifier.dino.lora_plus_tta import core as c
from classifier.dino.lora_plus_tta import runner, evaluation
from classifier.dino.evaluate import metrics, save_predictions


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DINOv3ViTConfig()
        self.q_proj = torch.nn.Linear(2, 2)
        self.v_proj = torch.nn.Linear(2, 2)

    def forward(self, pixels):
        return self.q_proj(pixels) + self.v_proj(pixels)


class TinyDamage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = get_peft_model(TinyBackbone(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
        self.head = torch.nn.Linear(2, 2)

    def forward(self, pixels):
        return self.head(self.backbone(pixels))


CFG = {"optimizer": "loraplus", "lora_lr": 3e-5, "head_lr": 1e-3, "loraplus_lr_ratio": 16,
       "weight_decay": .01, "warmup_epochs": 1, "epochs": 3}


class LoRAPlusTTATests(unittest.TestCase):
    def test_official_optimizer_preserves_head_decay_coverage_and_ratio_on_resume(self):
        torch.manual_seed(0)
        model = TinyDamage()
        initial = copy.deepcopy(model.state_dict())
        optimizer = c.optimizer(model, CFG)
        scheduler = c.t.scheduler_for(optimizer, 2, CFG)
        groups = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(groups["head"]["initial_lr"], .001)
        self.assertEqual(groups["lora_A"]["initial_lr"], 3e-5)
        self.assertEqual(groups["lora_B"]["initial_lr"], 4.8e-4)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for g in groups.values() for p in g["params"]]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        for group in groups.values():
            if group["params"]:
                self.assertEqual(group["weight_decay"], .01)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
        pixels, targets = torch.tensor([[1., -1.], [.5, 2.]]), torch.tensor([[1., 0.], [0., 1.]])
        def step():
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy_with_logits(model(pixels), targets).backward()
            optimizer.step()
            scheduler.step()
        step()
        saved = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
        step()
        expected_state, expected_lr = copy.deepcopy(model.state_dict()), scheduler.get_last_lr()
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step()
        for name in expected_state:
            torch.testing.assert_close(model.state_dict()[name], expected_state[name], atol=0, rtol=0)
        self.assertEqual(scheduler.get_last_lr(), expected_lr)
        restored = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertAlmostEqual(restored["lora_B"]["lr"]/restored["lora_A"]["lr"], 16)

    def test_tta_flips_photo_before_padding_and_rejects_image_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.png"
            image = Image.fromarray(np.arange(3*5*3, dtype=np.uint8).reshape(5,3,3))
            image.save(path)
            row = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "target": [1, 0]}
            dataset = c.FlippedPhotos([row], 8)
            actual, target = dataset[0]
            expected = preprocess(ImageOps.mirror(image), 8)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(target, torch.tensor([1., 0.]))
            # 홀수 너비 패딩에서는 최종 패딩 텐서 반전과 다르다.
            self.assertFalse(torch.equal(actual, preprocess(image, 8).flip(-1)))
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                dataset[0]

    def test_tta_keeps_two_independent_sigmoid_outputs(self):
        actual = c.mean_probabilities([[.9, .8], [.1, .2]], [[.7, .6], [.3, .4]])
        np.testing.assert_allclose(actual, [[.8, .7], [.2, .3]])
        self.assertGreater(actual[0].sum(), 1)
        for invalid in ([[float("nan"), .2]], [[1.1, .2]]):
            with self.assertRaises(ValueError):
                c.mean_probabilities(invalid, [[.1, .2]])

    def test_only_optimizer_settings_differ_from_origin(self):
        origin = {"experiment": "EX-06", "lora_lr": 3e-5, "head_lr": .001, "input_size": 1024, "pooling": "cls_mean", "epochs": 15}
        changed = c.plus_config(origin)
        self.assertEqual({k for k in changed if changed[k] != origin.get(k)}, {"experiment", "optimizer", "loraplus_lr_ratio"})
        self.assertEqual(changed["lora_lr"], 3e-5)

    def test_tta_artifacts_preserve_origin_and_reproduce_csv_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin, chain, backbone = root / "origin", root / "chain", root / "backbone"
            for folder in (origin, chain, backbone):
                folder.mkdir()
            config = {"epochs": 15, "seed": 0}
            identity = {"data_sha256": "same"}
            parts = {"val": [{"file":"a", "label":"normal"}, {"file":"b", "label":"heel"}],
                     "val25": [{"file":"c", "label":"outsole"}]}
            rows = parts["val"] + parts["val25"]
            plain = np.asarray([[.1,.1], [.9,.1], [.1,.9]], dtype=np.float32)
            flipped = np.asarray([[.2,.1], [.8,.2], [.2,.8]], dtype=np.float32)
            old_result = metrics([0,1,2], plain)
            c.t.write_json(origin / "training_status.json", {"status":"complete"})
            run_protocol = {"config":config, "identity":identity, "backbone_path":str(backbone)}
            for name, value in [("protocol.json",run_protocol), ("config.json",config), ("selection.json",{"batch":1,"checkpointing":False}),
                                ("metrics.json",{"val_combined":old_result}), ("thresholds.json",{"values":[.5,.5]})]:
                c.t.write_json(origin / name, value)
            (origin / "last.pt").write_bytes(b"local checkpoint")
            (backbone / "model.safetensors").write_bytes(b"local backbone")
            save_predictions(origin / "val_combined_probs.csv", rows, plain)
            preserved = {p.name:p.read_bytes() for p in origin.iterdir()}
            protocol = {"origin_run":str(origin), "origin_identity":identity, "sha256":"chain",
                        "backbone_sha256":c.sha(backbone/"model.safetensors"),
                        "tta":{"threshold_selection":"grid"}}
            saved = {"identity":identity, "epoch":15, "model":{}, "selection":{"batch":1,"checkpointing":False}}
            model = SimpleNamespace(to=lambda device: model, load_trainable=lambda state: None)
            with patch.object(c, "load", return_value=protocol), \
                 patch.object(c.t, "prepare", return_value=(parts,run_protocol,None)), \
                 patch.object(c.t, "gpu_gate"), patch.object(c.t, "seed_all"), patch.object(c.t, "peaks", return_value={}), \
                 patch.object(evaluation, "DamageDINO", return_value=model), patch.object(torch, "load", return_value=saved), \
                 patch.object(c.t, "predict", side_effect=[(plain[:2],{}),(plain[2:],{})]), \
                 patch.object(evaluation, "predict_flipped", side_effect=[flipped[:2],flipped[2:]]):
                evaluation.tta(chain,"lora")
            self.assertEqual(preserved,{p.name:p.read_bytes() for p in origin.iterdir()})
            output = chain / "tta_lora"
            probabilities = evaluation.read_probabilities(output/"val_combined_probs.csv",rows)
            np.testing.assert_allclose(probabilities,(plain+flipped)/2)
            result = c.t.read_json(output/"metrics.json")["val_combined"]
            self.assertEqual(metrics([0,1,2],probabilities,result["thresholds"]),result)
            self.assertTrue(c.t.read_json(output/"complete.json")["original_confusion_reproduced"])

    def test_queue_publishes_before_training_and_stops_on_tta_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            chain, plus = Path(directory) / "chain", Path(directory) / "plus"
            chain.mkdir()
            plus.mkdir()
            protocol = {"plus_run": str(plus)}
            events = []
            def child(*args, **kwargs):
                stage = args[2]
                events.append(stage)
                if stage == "train-plus":
                    (plus / "training_status.json").write_text('{"status":"trained"}', encoding="utf-8")
                if stage == "evaluate-plus":
                    (plus / "training_status.json").write_text('{"status":"complete"}', encoding="utf-8")
                    (plus / "combined_reporting.json").write_text('{}', encoding="utf-8")
            with patch.object(c, "safe_run", side_effect=lambda path: Path(path)), patch.object(c, "load", return_value=protocol), \
                 patch.object(runner, "wait_origin"), patch.object(runner, "status"), patch.object(runner, "child", side_effect=child), \
                 patch.object(runner, "publish", side_effect=lambda *a, **k: events.append("publish-final" if k["final"] else "publish-before-after")):
                runner.queue(chain)
                self.assertEqual(events, ["tta-lora", "publish-before-after", "smoke-plus", "train-plus", "evaluate-plus", "tta-plus", "publish-final"])
                events.clear()
                with patch.object(runner, "child", side_effect=RuntimeError("TTA failed")):
                    with self.assertRaises(RuntimeError):
                        runner.queue(chain)
                self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
