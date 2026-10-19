"""후속 실험의 상태·보고 갱신이 EX-01을 덮어쓰지 않는지 검증한다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from classifier.dino.tools.report_combined import update_experiment_record
from classifier.dino.linear_probe import pipeline


FIRST = "## EX-01 first\n\n**상태:** complete\n\n<!-- EX-01-RUNTIME-START -->\noriginal result\n<!-- EX-01-RUNTIME-END -->\n\n"
SECOND = "## EX-02 second\n\n**상태:** planned\n\nKeep this design.\n"
TABLE = "| EX-01 | model1 | complete |\n| EX-02 | model2 | planned |\n\n"


class ExperimentRecordTests(unittest.TestCase):
    def test_second_experiment_appends_then_replaces_only_its_own_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EXPERIMENTS.md"
            path.write_text(TABLE + FIRST + SECOND, encoding="utf-8")
            update_experiment_record(path, "EX-02", "smoke ready", "first smoke result")
            update_experiment_record(path, "EX-02", "evaluated", "final result")
            actual = path.read_text(encoding="utf-8")
            self.assertIn(FIRST, actual)
            self.assertIn("| EX-01 | model1 | complete |", actual)
            self.assertIn("| EX-02 | model2 | evaluated |", actual)
            self.assertIn("**상태:** evaluated", actual)
            self.assertIn("Keep this design.", actual)
            self.assertEqual(actual.count("<!-- EX-02-RUNTIME-START -->"), 1)
            self.assertNotIn("first smoke result", actual)
            self.assertIn("final result", actual)

    def test_unknown_experiment_does_not_change_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EXPERIMENTS.md"
            original = TABLE + FIRST + SECOND
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(ValueError):
                update_experiment_record(path, "EX-03", "running", "must not appear")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_linear_final_report_preserves_comparison_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "classifier/dino/EXPERIMENTS.md"
            path.parent.mkdir(parents=True)
            path.write_text("| EX-03 | model | data | 1024 | 15 / 0 | running |\n\n"
                            "## EX-03 linear\n\n**상태:** running\n", encoding="utf-8")
            run = root / "runs/dino/EX-03"
            run.mkdir(parents=True)
            (run / "training_status.json").write_text('{"completed_epoch": 15}', encoding="utf-8")
            comparison = "| 실험 | 균형 정확도 | 전체 정확도 |\n|---|---:|---:|\n| EX-03 | 74.23% | 74.98% |"
            with patch.object(pipeline, "ROOT", root):
                pipeline.record(run, "completed", comparison)
            actual = path.read_text(encoding="utf-8")
            self.assertIn("| EX-03 | model | data | 1024 | 15 / 15 | completed |", actual)
            self.assertIn(comparison, actual)


if __name__ == "__main__":
    unittest.main()
