"""사진 평가에서 실제로 위험한 순서·매핑·누락·검증 누출을 CPU로 확인한다."""
import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from classifier.dino.yolo_pipeline import common as c
from classifier.dino.yolo_pipeline import report as r
from classifier.dino.yolo_pipeline import runner
from classifier.dino.evaluate import metrics, predict_classes


class PhotoScores(unittest.TestCase):
    def test_detection_order_is_mapped_by_name(self):
        self.assertEqual(c.photo_score([0, 1, 1], [.8, .3, .7], c.NAMES), [.7, .8])

    def test_successful_empty_detection_is_zero(self):
        self.assertEqual(c.photo_score([], [], c.NAMES), [0., 0.])

    def test_wrong_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            c.photo_score([0], [.9], {0: "heel_lining", 1: "outsole"})

    def test_invalid_class_or_score_is_rejected(self):
        for cls, score in [(2, .8), (.5, .8), (0, np.nan), (0, 1.1)]:
            with self.subTest(cls=cls, score=score), self.assertRaises(ValueError):
                c.photo_score([cls], [score], c.NAMES)

    def test_ensemble_keeps_both_and_normal(self):
        mean = c.mean_scores([[.8, .9], [.1, .2]], [[.6, .7], [.3, .2]])
        np.testing.assert_allclose(mean, [[.7, .8], [.2, .2]])
        self.assertEqual(predict_classes(mean).tolist(), [3, 0])

    def test_mismatched_or_nan_scores_are_not_averaged(self):
        for other in ([[.1, .2], [.1, .2]], [[np.nan, .2]], [[-1., .2]]):
            with self.assertRaises(ValueError):
                c.mean_scores([[.2, .3]], other)

    def test_both_false_positive_counts_as_wrong(self):
        result = metrics([0, 1, 2], [[0, 0], [.9, .9], [0, .9]])
        self.assertEqual(result["confusion_matrix"][1][3], 1)
        self.assertIsNone(result["recall"]["both"])
        self.assertAlmostEqual(result["balanced_accuracy"], 2/3)


class SavedScores(unittest.TestCase):
    def setUp(self):
        parent = c.ROOT / "runs/yolo"
        parent.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix="cpu-test-", dir=parent)).resolve()
        self.folder.relative_to(parent.resolve())
        self.path = self.folder / "scores.csv"
        self.rows = [{"file": "a.jpg"}, {"file": "b.jpg"}]

    def tearDown(self):
        self.folder.resolve().relative_to((c.ROOT / "runs/yolo").resolve())
        shutil.rmtree(self.folder)

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file", "p_outsole", "p_heel_lining"])
            writer.writerows(rows)

    def test_reordered_files_and_columns_align_by_name(self):
        self.write_rows([["b.jpg", .2, .8], ["a.jpg", .3, .7]])
        np.testing.assert_array_equal(c.read_scores(self.path, self.rows), [[.7, .3], [.8, .2]])

    def test_missing_duplicate_and_extra_file_fail(self):
        for rows in ([["a.jpg", .3, .7]], [["a.jpg", .3, .7], ["a.jpg", .2, .8]],
                     [["a.jpg", .3, .7], ["c.jpg", .2, .8]]):
            self.write_rows(rows)
            with self.assertRaises(ValueError):
                c.read_scores(self.path, self.rows)

    def test_save_read_keeps_exact_scores(self):
        values = np.asarray([[.123456789, .987654321], [0., 1.]])
        c.save_scores(self.path, self.rows, values)
        np.testing.assert_array_equal(c.read_scores(self.path, self.rows), values)


class ExecutionRules(unittest.TestCase):
    def test_predecessor_failure_cannot_start_yolo(self):
        protocol = {"dino_chain": str(c.DINO_CHAIN)}
        with mock.patch.object(c, "read", return_value={"stage": "failed", "gpu_released": True}):
            with self.assertRaises(RuntimeError):
                runner.predecessor_complete(protocol, "[exit 0]\n")

    def test_disappeared_pid_is_not_success(self):
        with mock.patch.object(c, "read", return_value={"stage": "completed", "gpu_released": True}):
            with self.assertRaises(RuntimeError):
                runner.predecessor_complete({"dino_chain": str(c.DINO_CHAIN)}, "[exit 1]\n")

    def test_paths_and_recipe_do_not_include_test(self):
        with self.assertRaises(ValueError):
            c.safe_run(c.ROOT.parent / "Mendi/runs/other")
        run = c.ROOT / "runs/yolo/recipe-test"
        args = c.training_arguments(run, 2)
        self.assertEqual(args["epochs"], 50)
        self.assertEqual(args["split"], "val")
        self.assertEqual(args["mosaic"], .2)
        self.assertEqual(args["scale"], .2)
        self.assertEqual(args["optimizer"], "AdamW")
        self.assertFalse(args["exist_ok"])
        self.assertNotIn("resume", args)
        self.assertNotIn("test", args)

    def test_smoke_recipe_does_not_change_full_recipe(self):
        run = c.ROOT / "runs/yolo/recipe-test"
        self.assertEqual(c.training_arguments(run, 1, True)["epochs"], 1)
        self.assertEqual(c.training_arguments(run, 1)["epochs"], 50)
        self.assertEqual(c.TRAIN_ARGS["batch"], 2)

    def test_rank_uses_ba_then_accuracy_then_fixed_order(self):
        values = [{"balanced_accuracy": .8, "accuracy": .99},
                  {"balanced_accuracy": .9, "accuracy": .8},
                  {"balanced_accuracy": .9, "accuracy": .81},
                  {"balanced_accuracy": .9, "accuracy": .81}]
        self.assertEqual(r.rank(values), 2)


class GroupValidation(unittest.TestCase):
    def test_shoe_groups_never_cross_folds(self):
        labels = np.repeat(np.arange(3), 20)
        groups = np.asarray([f"g{i//2}" for i in range(60)])
        folds = r.group_folds(labels, groups)
        self.assertEqual(len(folds), 5)
        counts = np.zeros(60, dtype=int)
        for train, val in folds:
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertEqual(set(labels[train]), {0, 1, 2})
            counts[val] += 1
        np.testing.assert_array_equal(counts, np.ones(60, dtype=int))

    def test_score_selection_does_not_use_held_out_labels(self):
        labels = np.repeat(np.arange(3), 20)
        groups = np.asarray([f"g{i//2}" for i in range(60)])
        probabilities = np.asarray([[.1, .1] if y == 0 else ([.8, .1] if y == 1 else [.1, .8]) for y in labels])
        rows = [{"file": f"{i}.jpg", "label": c.ORDER[y], "family": "v5", "group": groups[i]} for i, y in enumerate(labels)]
        result = r.grouped_check(rows, probabilities, {"Y1": probabilities, "Y2": probabilities})
        for model in result["results"].values():
            self.assertEqual(model["images"], 60)
            self.assertEqual(model["balanced_accuracy"], 1.)
        self.assertEqual(len(result["folds"]), 5)


if __name__ == "__main__":
    unittest.main()
