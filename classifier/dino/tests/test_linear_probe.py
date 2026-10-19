"""CPU checks for deterministic views, cache integrity, resume and final evaluation."""
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from classifier.dino.linear_probe import pipeline as p
from classifier.dino.linear_probe.gpu_head import runner as gpu


class LinearProbeTests(unittest.TestCase):
    def test_fixed_views_preserve_random_state_and_image_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(np.arange(32*40*3, dtype=np.uint8).reshape(32, 40, 3)).save(path)
            row = {"file": path.name, "path": str(path), "sha256": p.file_sha(path)}
            ds = p.FixedViews([row], 32, 4, 0, True)
            state = random.getstate()
            self.assertTrue(torch.equal(ds[0][0], ds[0][0]))
            self.assertEqual(random.getstate(), state)
            self.assertFalse(torch.equal(ds[0][0], ds[1][0]))
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image changed"):
                ds[0]

    def test_view_rotation_preserves_epoch_size(self):
        rows = [{"file": str(n)} for n in range(4584)]
        selections = np.stack([p.view_indices(rows, 0, 4, epoch) for epoch in range(1, 16)])
        self.assertEqual(selections.shape, (15, 4584))
        for image in range(len(rows)):
            self.assertEqual(sorted(np.bincount(selections[:, image], minlength=4)), [3, 4, 4, 4])

    def test_pool_excludes_register_tokens(self):
        hidden = torch.tensor([[[1., 2.], [999., 999.], [999., 999.], [999., 999.],
                                [999., 999.], [3., 4.], [5., 8.]]])
        torch.testing.assert_close(p.pool(hidden), torch.tensor([[1., 2., 4., 6., 5., 8.]]))

    def fixture(self, run):
        c = {"feature_size": 2304, "train_views": 4, "seed": 0, "batch_size": 8, "epochs": 2,
             "cpu_threads": 1, "head_lr": .001, "weight_decay": .01, "warmup_epochs": 1,
             "clip_grad_norm": 1., "reference_runs": []}
        parts = {split: [{"file": f"{split}_{n}.png", "label": p.ORDER[n % 4],
                          "target": [[0, 0], [1, 0], [0, 1], [1, 1]][n % 4]}
                         for n in range(size)] for split, size in [("train", 24), ("val", 8), ("val25", 8)]}
        protocol = {"config": c, "parts": parts, "identity": {"fixture": "synthetic"}}
        (run / "cache").mkdir()
        rng, files = np.random.default_rng(42), {}
        for split, rows in parts.items():
            shape = (len(rows), 4, 2304) if split == "train" else (len(rows), 2304)
            path = run / "cache" / f"{split}.npy"
            np.save(path, rng.normal(size=shape).astype(np.float32))
            files[split] = {"sha256": p.file_sha(path)}
        p.write_json(run / "cache_index.json", {"synthetic": True})
        p.write_json(run / "cache_complete.json", {"identity": protocol["identity"], "files": files,
                     "index_sha256": p.file_sha(run / "cache_index.json")})
        return protocol

    def test_tampered_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol = self.fixture(run)
            p.validate_cache(run, protocol)
            with (run / "cache/train.npy").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                p.validate_cache(run, protocol)

    def test_cpu_training_resume_and_reloaded_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "record"):
            run = Path(directory)
            protocol = self.fixture(run)
            captured = {}
            real_save = p.atomic_torch_save

            def capture(value, path):
                if value.get("epoch") == 1:
                    buffer = io.BytesIO()
                    torch.save(value, buffer)
                    captured["epoch1"] = buffer.getvalue()
                real_save(value, path)

            with patch.object(p, "atomic_torch_save", side_effect=capture):
                p.train(run, protocol)
            uninterrupted = torch.load(run / "last.pt", weights_only=True)
            (run / "last.pt").write_bytes(captured["epoch1"])
            p.train(run, protocol, resume=True)
            resumed = torch.load(run / "last.pt", weights_only=True)
            for key in uninterrupted["head"]:
                torch.testing.assert_close(uninterrupted["head"][key], resumed["head"][key], rtol=0, atol=0)
            self.assertEqual(resumed["history"][-1]["val_fixed_0_5"], uninterrupted["history"][-1]["val_fixed_0_5"])
            p.evaluate(run, protocol)
            self.assertFalse(torch.cuda.is_initialized())
            result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["val_combined"]
            self.assertEqual(result["images"], 16)
            self.assertEqual(sum(map(sum, result["confusion_matrix"])), 16)
            self.assertEqual(p.read_json(run / "training_status.json")["stage"], "completed")

    def test_gpu_continuation_preserves_feature_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            original = self.fixture(run)
            with patch.object(p, "load_protocol", return_value=original):
                _, protocol = gpu.prepare(run)
                self.assertEqual(protocol["origin_identity"], original["identity"])
                self.assertEqual(protocol["config"]["head_device"], "cuda")
                self.assertEqual(gpu.load(run)[1], protocol)
                changed = p.read_json(run / "gpu_head_protocol.json")
                changed["origin_identity"]["fixture"] = "changed"
                p.write_json(run / "gpu_head_protocol.json", changed)
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    gpu.load(run)

    def test_new_device_path_resume_and_evaluation_on_cpu(self):
        # Exercise the actual device-aware optimizer/checkpoint/evaluation path;
        # actual CUDA gradients and reload are checked by gpu_smoke before the run.
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "record"):
            run = Path(directory)
            original = self.fixture(run)
            protocol = {"config": original["config"], "identity": {"head": "new-device-path"}}
            captured, real_save = {}, p.atomic_torch_save

            def capture(value, path):
                if value.get("epoch") == 1:
                    buffer = io.BytesIO()
                    torch.save(value, buffer)
                    captured["epoch1"] = buffer.getvalue()
                real_save(value, path)

            with patch.object(p, "atomic_torch_save", side_effect=capture):
                gpu.train(run, original, protocol, device="cpu")
            uninterrupted = torch.load(run / "last.pt", weights_only=True)
            (run / "last.pt").write_bytes(captured["epoch1"])
            gpu.train(run, original, protocol, resume=True, device="cpu")
            resumed = torch.load(run / "last.pt", weights_only=True)
            for key in uninterrupted["head"]:
                torch.testing.assert_close(uninterrupted["head"][key], resumed["head"][key], rtol=0, atol=0)
            gpu.evaluate(run, original, protocol, device="cpu")
            result = p.read_json(run / "metrics.json")["val_combined"]
            self.assertEqual(result["images"], 16)
            self.assertEqual(p.read_json(run / "inference_config.json")["head_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
