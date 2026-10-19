"""CPU 전용 후처리의 대기 조건·파일 정렬·최종 보고를 합성 저장물로 검증한다."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from classifier.dino.analysis.after_run import analyze, completion, load_run, paired_comparison, prediction, score, write


def fixture(run, values):
    run.mkdir()
    labels = [{"file": "n.jpg", "label": "normal", "group": "g1"}, {"file": "h.jpg", "label": "heel", "group": "g2"}]
    extra = [{"file": "o.jpg", "label": "outsole", "group": "g3"}]
    write(run / "protocol.json", {"identity": {"data_sha256": "fixed"}, "config": {"epochs": 2}})
    write(run / "data_manifest.json", {"sha256": "fixed", "images": {"val": labels, "val25": extra}})
    write(run / "thresholds.json", {"values": [.5, .5]})
    with (run / "val_combined_probs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "p_heel_lining", "p_outsole"])
        for name, p in zip(("n.jpg", "h.jpg", "o.jpg"), values):
            writer.writerow([name, *p])
    p, y = np.array(values), np.array([0, 1, 2])
    measured = score(y, prediction(p, [.5, .5]))
    write(run / "metrics.json", {"val_combined": {**measured, "thresholds": [.5, .5]}})
    write(run / "training_status.json", {"status": "complete", "completed_epoch": 2})
    history = []
    for epoch in (1, 2):
        validation = {"val": {**score(y[:2], prediction(p[:2], [.5, .5])), "loss": .4},
                      "val25": {**score(y[2:], prediction(p[2:], [.5, .5])), "loss": .5}}
        history.append({"epoch": epoch, "train": {"loss": .5 / epoch}, "validation_fixed_0_5": validation})
    write(run / "epochs.json", history)
    (run / "console.log").write_text("[exit 0]\n", encoding="utf-8")


class CpuAnalysisTests(unittest.TestCase):
    def test_completion_waits_for_final_process_exit_and_full_epoch_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            fixture(run, [[.1, .1], [.8, .1], [.1, .8]])
            self.assertEqual(completion(run), "ready")
            (run / "console.log").write_text("epoch done\n", encoding="utf-8")
            self.assertEqual(completion(run), "waiting")
            (run / "console.log").write_text("[exit 1]\n", encoding="utf-8")
            self.assertEqual(completion(run), "failed")
            (run / "console.log").write_text("[exit 0]\n", encoding="utf-8")
            write(run / "training_status.json", {"status": "complete", "completed_epoch": 1})
            self.assertEqual(completion(run), "failed")

    def test_complete_analysis_uses_only_saved_files_and_rejects_wrong_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, run = root / "v2", root / "v3"
            fixture(baseline, [[.1, .1], [.8, .1], [.1, .8]])
            fixture(run, [[.8, .1], [.8, .1], [.1, .8]])
            a, b = load_run(baseline), load_run(run)
            overlap = paired_comparison(a, b)
            self.assertEqual(overlap["only_v2_correct"], 1)
            self.assertEqual(overlap["both_correct"], 2)
            self.assertAlmostEqual(b["score"]["balanced_accuracy"], 2 / 3)
            self.assertEqual(analyze(run, baseline, update_document=False)["status"], "complete")
            self.assertTrue((run / "post_run_analysis.md").is_file())
            self.assertTrue((run / "paired_errors.csv").is_file())
            self.assertNotIn("torch", sys.modules)
            b["files"][0] = "different.jpg"
            with self.assertRaises(ValueError):
                paired_comparison(a, b)
            with (run / "val_combined_probs.csv").open("a", encoding="utf-8") as stream:
                stream.write("missing.jpg,0.1,0.1\n")
            with self.assertRaises(ValueError):
                load_run(run)


if __name__ == "__main__":
    unittest.main()
