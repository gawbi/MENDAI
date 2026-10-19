"""Checks for the photo-level evaluation (python -m unittest discover -s detector/detr/tests)."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import eval_image_level as ev  # noqa: E402


class VerdictTests(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(ev.verdict(0.6, 0.1, 0.5, 0.5), "outsole")
        self.assertEqual(ev.verdict(0.1, 0.6, 0.5, 0.5), "heel")
        self.assertEqual(ev.verdict(0.6, 0.6, 0.5, 0.5), "both")
        self.assertEqual(ev.verdict(0.4, 0.4, 0.5, 0.5), "normal")
        self.assertEqual(ev.verdict(0.5, 0.0, 0.5, 0.5), "outsole")  # threshold is inclusive

    def test_balanced_accuracy_skips_classes_without_photos(self):
        cm = [[8, 1, 1, 0], [1, 3, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]  # rows truth, columns verdict
        m = ev.summarize(cm)
        self.assertAlmostEqual(m["balanced_acc"], (8 / 10 + 3 / 4 + 1 / 2) / 3)
        self.assertAlmostEqual(m["overall_acc"], 12 / 16)
        self.assertIsNone(m["recall"]["both"])


class CommandLineTests(unittest.TestCase):
    """End to end through the CLI with tiny CSVs, including the TTA view merge."""

    LABELS = [("a.jpg", "normal"), ("b.jpg", "heel"), ("c.jpg", "outsole"), ("d.jpg", "normal")]

    @staticmethod
    def write_csv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def run_eval(self, tmp, extra):
        out = Path(tmp) / "out"
        cmd = [sys.executable, str(HERE / "eval_image_level.py"), "--out", str(out), "--fixed-conf", "0.5", "0.5",
               "--set", "val", str(Path(tmp) / "pred.csv"), str(Path(tmp) / "labels.csv"), "val", *extra]
        subprocess.run(cmd, check=True, capture_output=True)
        return json.loads((out / "image_eval.json").read_text(encoding="utf-8"))

    def test_false_positive_removed_by_averaged_flip_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_csv(Path(tmp) / "labels.csv", ["file", "split", "label"], [(f, "val", l) for f, l in self.LABELS])
            header = ["file", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"]
            # d.jpg is a normal photo with a confident outsole box in the original view only.
            self.write_csv(Path(tmp) / "pred.csv", header, [
                ("b.jpg", 1, "heel_lining", 0.9, 0, 0, 10, 10),
                ("c.jpg", 0, "outsole", 0.9, 0, 0, 10, 10),
                ("d.jpg", 0, "outsole", 0.8, 0, 0, 10, 10),
            ])
            self.write_csv(Path(tmp) / "flip.csv", header, [
                ("b.jpg", 1, "heel_lining", 0.9, 0, 0, 10, 10),
                ("c.jpg", 0, "outsole", 0.9, 0, 0, 10, 10),
            ])

            single = self.run_eval(tmp, [])
            self.assertAlmostEqual(single["combined"]["recall"]["normal"], 0.5)
            self.assertEqual(single["sets"]["val"]["confusion"][0], [1, 0, 1, 0])

            merged = self.run_eval(tmp, ["--add-view", "val", str(Path(tmp) / "flip.csv")])  # d.jpg: (0.8 + 0) / 2 < 0.5
            self.assertAlmostEqual(merged["combined"]["balanced_acc"], 1.0)
            self.assertEqual(merged["merge"], "mean")


if __name__ == "__main__":
    unittest.main()
