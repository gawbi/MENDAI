"""지표·누적·재개를 작은 CPU 예제로 검증한다. 실제 데이터나 GPU는 쓰지 않는다."""
import copy
import random
import unittest

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from classifier.dino.data import TARGETS, preprocess
from classifier.dino.evaluate import choose_thresholds, metrics, predict_classes, report_markdown
from classifier.dino.train import restore_rng, rng_state, scheduler_for, seed_all, train_epoch


class MetricsTests(unittest.TestCase):
    def test_confusion_and_missing_both(self):
        # normal 2개 중 1개 정답, heel 1개 정답, outsole은 both로 오판.
        result = metrics([0, 0, 1, 2], [[.1, .1], [.8, .1], [.8, .1], [.8, .8]])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["recall"], {"normal": .5, "heel": 1., "outsole": 0., "both": None})
        self.assertIn("N/A", report_markdown({"val": result}))

    def test_both_present_and_boundary(self):
        result = metrics([0, 1, 2, 3], [[.1, .1], [.5, .1], [.1, .5], [.5, .5]])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertEqual(predict_classes([[.5, .5]]).tolist(), [3])
        self.assertEqual(TARGETS["heel"], [1, 0])

    def test_threshold_search_finds_non_default_optimum(self):
        y = np.array([0, 0, 0, 1, 2])
        p = np.array([[.3, .2], [.2, .1], [.3, .3], [.4, .3], [.3, .9]])
        best = choose_thresholds(y, p)
        self.assertEqual(metrics(y, p, best)["balanced_accuracy"], 1)
        # 다수 정답 클래스 비중에 관계없이 perfect 후보 중 .5에 가장 가까운 값.
        self.assertEqual(best, [.4, .5])
        inverted = p[:, ::-1]
        self.assertLess(metrics(y, inverted)["balanced_accuracy"], 1)

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            metrics([0], [[float("nan"), .5]])


class TrainingTests(unittest.TestCase):
    def test_accumulation_matches_full_batches_including_partial_window(self):
        seed_all(7)
        x, y = torch.randn(11, 3), torch.rand(11, 2)
        a = torch.nn.Linear(3, 2)
        b = copy.deepcopy(a)
        opt_a, opt_b = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
        # 큰 clipping 한도로 누적의 샘플 가중치만 확인한다.
        cfg = {"clip_grad_norm": 1000}
        loss = torch.nn.BCEWithLogitsLoss()
        first = train_epoch(a, DataLoader(TensorDataset(x, y), batch_size=2), opt_a, loss, 4, device="cpu", config=cfg)
        second = train_epoch(b, DataLoader(TensorDataset(x, y), batch_size=8), opt_b, loss, 1, device="cpu", config=cfg)
        self.assertEqual(first["optimizer_updates"], 2)
        self.assertEqual(second["optimizer_updates"], 2)
        for v, w in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(v, w, atol=1e-7, rtol=1e-6)

    def test_rng_optimizer_schedule_resume(self):
        seed_all(0)
        cfg = {"warmup_epochs": 1, "epochs": 3, "clip_grad_norm": 1000}
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        sched = scheduler_for(opt, 2, cfg)
        x, y = torch.randn(8, 3), torch.rand(8, 2)
        dl = DataLoader(TensorDataset(x, y), batch_size=2)
        criterion = torch.nn.BCEWithLogitsLoss()
        train_epoch(model, dl, opt, criterion, 2, sched, device="cpu", config=cfg)
        saved = copy.deepcopy({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng_state()})
        train_epoch(model, dl, opt, criterion, 2, sched, device="cpu", config=cfg)
        expected = copy.deepcopy(model.state_dict())
        expected_lr = sched.get_last_lr()
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        random.random(); np.random.rand(); torch.rand(1)
        restore_rng(saved["rng"])
        train_epoch(model, dl, opt, criterion, 2, sched, device="cpu", config=cfg)
        self.assertEqual(sched.get_last_lr(), expected_lr)
        for key in expected:
            torch.testing.assert_close(model.state_dict()[key], expected[key], atol=0, rtol=0)


class PreprocessTests(unittest.TestCase):
    def test_portrait_is_padded_without_crop_or_stretch(self):
        image = Image.new("RGB", (50, 100), (255, 0, 0))
        actual = preprocess(image, 140)
        self.assertEqual(tuple(actual.shape), (3, 140, 140))
        expected_red = torch.tensor([(1-.485)/.229, - .456/.224, - .406/.225])
        torch.testing.assert_close(actual[:, 70, 70], expected_red)
        self.assertLess(abs(actual[:, 70, 0]).max().item(), .01)


if __name__ == "__main__":
    unittest.main()
