from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from classifier.dino.normalized_probe import core as n
from classifier.dino.normalized_probe import training


class NormalizedProbeTests(unittest.TestCase):
    def test_standardized_warm_start_preserves_logits(self):
        torch.manual_seed(13)
        old = torch.nn.Linear(2304, 2)
        mean = torch.randn(2304) * 3
        scale = torch.rand(2304) + .1
        x = torch.randn(16, 2304)
        transformed = n.initialize_head(old.state_dict(), mean, scale, True, "cpu")
        torch.testing.assert_close(old(x), transformed((x-mean)/scale), rtol=0, atol=5e-5)

    def test_train_only_population_scaler_with_zero_variance(self):
        values = np.array([[[1., 3.], [3., 3.]], [[5., 3.], [7., 3.]]], dtype=np.float32)
        mean, scale = n.fit_scaler(values)
        np.testing.assert_allclose(mean, [4., 3.])
        np.testing.assert_allclose(scale, [np.sqrt(5), 1e-6])
        self.assertEqual(scale.dtype, np.float32)
        with self.assertRaises(ValueError):
            n.fit_scaler(np.full((2, 4, 2), np.nan))

    def test_matched_heads_cache_integrity_and_reloaded_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(n, "record"):
            run = Path(directory)
            parts = {split: [{"file": f"{split}_{i}", "label": n.ORDER[i%4],
                              "target": [[0,0],[1,0],[0,1],[1,1]][i%4]}
                             for i in range(size)] for split, size in [("train", 24), ("val", 8), ("val25", 8)]}
            c = {"experiment": "EX-04", "epochs": 2, "variants": ["raw_control", "standardized"], "seed": 0,
                 "feature_size": 2304, "cpu_threads": 1, "batch_size": 8, "head_lr": .001,
                 "weight_decay": .01, "warmup_epochs": 1, "clip_grad_norm": 1., "origin_run": "synthetic"}
            protocol = {"parts": parts, "config": c, "identity": {"synthetic": True},
                        "origin_metrics": {"balanced_accuracy": .5, "accuracy": .5,
                            "images": 16, "order": n.ORDER, "thresholds": [.5, .5],
                            "confusion_matrix": [[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,4]],
                            "recall": dict.fromkeys(n.ORDER, 1.)}}
            protocol["identity"]["checkpoint_sha256"] = "synthetic"
            (run / "cache").mkdir()
            rng, files = np.random.default_rng(0), {}
            train_mean = None
            for split, rows in parts.items():
                shape = (len(rows), 4, 2304) if split == "train" else (len(rows), 2304)
                values = (rng.normal(size=shape) + (100 if split != "train" else 2)).astype(np.float32)
                if split == "train":
                    train_mean = values.mean((0,1), dtype=np.float64).astype(np.float32)
                path = run / "cache" / f"{split}.npy"
                np.save(path, values)
                files[split] = {"sha256": n.p.file_sha(path)}
            torch.manual_seed(0)
            torch.save(torch.nn.Linear(2304,2).state_dict(), run / "original_head.pt")
            n.p.write_json(run / "cache_complete.json", {"identity": protocol["identity"], "files": files,
                              "original_head_sha256": n.p.file_sha(run / "original_head.pt")})
            training.train(run, protocol, device="cpu")
            with np.load(run / "normalizer.npz") as stats:
                np.testing.assert_array_equal(stats["mean"], train_mean)
            training.evaluate(run, protocol, device="cpu")
            results = n.p.read_json(run / "metrics.json")
            self.assertEqual(set(results), {"original", "raw_control", "standardized"})
            self.assertEqual(results["standardized"]["images"], 16)
            self.assertTrue(n.p.read_json(run / "initial_function_check.json")["passed"])
            with (run / "cache/train.npy").open("ab") as f:
                f.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                n.validate_cache(run, protocol)


if __name__ == "__main__":
    unittest.main()
