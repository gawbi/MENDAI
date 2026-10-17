import unittest

from classifier.dino.tools.report_combined import ORDER, combine, render


def result(matrix, thresholds=None):
    return {"order": ORDER, "thresholds": thresholds or [.96, .02], "confusion_matrix": matrix, "images": sum(map(sum, matrix))}


class CombinedValidationTests(unittest.TestCase):
    def test_metrics_are_recomputed_from_counts(self):
        a = result([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        b = result([[1, 0, 0, 0], [2, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        pooled = combine({"a": a, "b": b})
        self.assertEqual(pooled["images"], 9)
        self.assertAlmostEqual(pooled["accuracy"], 5/9)
        self.assertAlmostEqual(pooled["balanced_accuracy"], (2/3 + 2/4 + 1/2)/3)
        self.assertIsNone(pooled["recall"]["both"])
        self.assertEqual(render(pooled).count("**합산 val 혼동 행렬**"), 1)
        self.assertEqual(combine({"combined": pooled}), pooled)

    def test_different_thresholds_cannot_be_merged(self):
        cm = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ValueError):
            combine({"a": result(cm), "b": result(cm, [.5, .5])})


if __name__ == "__main__":
    unittest.main()
