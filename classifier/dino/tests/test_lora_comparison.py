"""LoRA 비교 보고의 지표·출처 보존과 다른 데이터 거부를 확인한다."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from classifier.dino.evaluate import metrics
from classifier.dino.tools.compare_lora import compare


class LoRAComparisonTests(unittest.TestCase):
    def test_completed_reports_are_compared_without_changing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "base", root / "new"
            for run, experiment, probability in [(baseline, "EX-02", [[.1, .1], [.9, .1], [.1, .9]]),
                                                  (candidate, "EX-06", [[.9, .1], [.9, .1], [.1, .9]])]:
                run.mkdir()
                (run / "protocol.json").write_text(json.dumps({"config": {"experiment": experiment, "epochs": 15},
                                                               "identity": {"data_sha256": "same"}}), encoding="utf-8")
                (run / "training_status.json").write_text('{"status":"complete"}', encoding="utf-8")
                (run / "metrics.json").write_text(json.dumps({"val_combined": metrics([0, 1, 2], probability)}), encoding="utf-8")
                (run / "report.md").write_text("original report\n", encoding="utf-8")
            before = [p.read_bytes() for p in (baseline / "metrics.json", candidate / "metrics.json")]
            with patch("classifier.dino.tools.compare_lora.update_experiment_record"), patch("classifier.dino.tools.compare_lora.REPO", root):
                changes = compare(candidate, baseline)
                compare(candidate, baseline)
            self.assertAlmostEqual(changes["balanced_accuracy"], -100/3)
            self.assertEqual(before, [p.read_bytes() for p in (baseline / "metrics.json", candidate / "metrics.json")])
            self.assertEqual((candidate / "report.md").read_text(encoding="utf-8").count("<!-- LORA-COMPARISON-START -->"), 1)
            protocol = json.loads((candidate / "protocol.json").read_text(encoding="utf-8"))
            protocol["identity"]["data_sha256"] = "different"
            (candidate / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                compare(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
