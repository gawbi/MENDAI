import copy
import unittest

from classifier.dino.tools.evaluate_stopped import validate_stop


class StoppedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"identity": {"source": "original"}, "config": {"epochs": 30}}
        self.checkpoint = {"epoch": 12, "identity": self.protocol["identity"], "history": [{"epoch": 12}]}
        self.decision = {"status": "stopped_at_checkpoint", "requested_epoch": 12, "completed_epoch": 12, "original_planned_epochs": 30}

    def test_authorized_stop_preserves_original_schedule(self):
        original = copy.deepcopy(self.protocol)
        self.assertEqual(validate_stop(self.checkpoint, self.protocol, self.decision), (12, 30))
        self.assertEqual(self.protocol, original)

    def test_stale_checkpoint_or_unconfirmed_stop_is_rejected(self):
        self.checkpoint["epoch"] = 11
        with self.assertRaises(ValueError):
            validate_stop(self.checkpoint, self.protocol, self.decision)
        self.checkpoint["epoch"] = 12
        self.decision["status"] = "waiting_for_checkpoint"
        with self.assertRaises(ValueError):
            validate_stop(self.checkpoint, self.protocol, self.decision)


if __name__ == "__main__":
    unittest.main()
