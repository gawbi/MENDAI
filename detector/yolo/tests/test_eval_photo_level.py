"""Checks for the photo-level evaluation (python -m unittest discover -s detector/yolo/tests)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_photo_level as ev  # noqa: E402


class VerdictTests(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(ev.verdict(0.6, 0.1, 0.25), "outsole")
        self.assertEqual(ev.verdict(0.1, 0.6, 0.25), "heel")
        self.assertEqual(ev.verdict(0.6, 0.6, 0.25), "both")
        self.assertEqual(ev.verdict(0.2, 0.2, 0.25), "normal")
        self.assertEqual(ev.verdict(0.0, 0.0, 0.25), "normal")    # no boxes
        self.assertEqual(ev.verdict(0.25, 0.0, 0.25), "outsole")  # threshold is inclusive


class MetricTests(unittest.TestCase):
    def test_balanced_accuracy_ignores_both_without_photos(self):
        rows = ([("normal", "normal")] * 8 + [("normal", "heel"), ("normal", "both")]
                + [("heel", "heel")] * 3 + [("heel", "normal")]
                + [("outsole", "outsole"), ("outsole", "normal")])
        m = ev.metrics(rows)
        self.assertEqual(m["cm"], [[8, 1, 0, 1], [1, 3, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(m["balanced_acc"], (8 / 10 + 3 / 4 + 1 / 2) / 3)
        self.assertAlmostEqual(m["overall_acc"], 12 / 16)  # "both" verdict on a normal photo is an error
        self.assertIsNone(m["recall"]["both"])
        self.assertEqual(ev.pct(m["recall"]["both"]), "N/A")

    def test_missing_core_class_gives_na(self):
        m = ev.metrics([("normal", "normal"), ("heel", "heel")])
        self.assertIsNone(m["recall"]["outsole"])
        self.assertIsNone(m["balanced_acc"])


class TestSplitGuard(unittest.TestCase):
    def test_test_sets_need_final(self):
        class A:
            sets, final = ["v5_test"], False
        with self.assertRaises(SystemExit):
            ev.evaluate(A, {"name": "x"})


if __name__ == "__main__":
    unittest.main()
