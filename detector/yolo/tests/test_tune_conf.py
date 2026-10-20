"""Checks for the threshold search (python -m unittest discover -s detector/yolo/tests)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_photo_level as ev  # noqa: E402
import tune_conf as tc  # noqa: E402


class TuneConfTests(unittest.TestCase):
    ROWS = [("normal", 0.02, 0.10), ("outsole", 0.08, 0.00), ("heel", 0.00, 0.60), ("outsole", 0.30, 0.05)]

    def test_verdicts_match_shared_rule(self):
        for thr in (0.05, 0.25, 0.5):
            got = tc.verdicts(self.ROWS, thr, thr)
            want = [(gt, ev.verdict(o, h, thr)) for gt, o, h in self.ROWS]
            self.assertEqual(got, want)

    def test_search_finds_lower_outsole_threshold(self):
        data = {"a": self.ROWS}
        grid = [0.05, 0.1, 0.25, 0.5]
        to, th, ms = tc.search(data, grid, shared=False)
        # outsole at 0.05 recovers the 0.08 photo without flagging the normal one (0.02)
        self.assertEqual(to, 0.05)
        self.assertAlmostEqual(ms["a"]["balanced_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
