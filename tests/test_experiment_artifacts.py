"""Regression tests for lossless, secret-safe experiment replicate records."""
import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dev"))
from run_experiments import create_experiment_dir, persist_replicate


class ExperimentArtifactTests(unittest.TestCase):
    def test_each_replicate_gets_exclusive_private_files(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = os.path.join(temp, "runs")
            os.mkdir(runs)
            stamp = datetime(2026, 10, 18, tzinfo=timezone.utc)
            first = create_experiment_dir(runs, "visible", now=stamp)
            second = create_experiment_dir(runs, "visible", now=stamp)
            self.assertNotEqual(first, second)
            self.assertEqual(stat.S_IMODE(os.stat(first).st_mode), 0o700)

            items = [{"id": "SYN-000"}]
            answers = {"SYN-000": {"case": "unique", "assignment": {"A": {"block": "07:00"}}}}
            metas = [{"id": "SYN-000", "calls": 1, "responses_received": 1,
                      "parses": [], "final": {"1": "AT|A|07:00"},
                      "raw": [{"text": "PRIVATE RESPONSE MUST NOT BE WRITTEN"}],
                      "outcome": {"kind": "unique", "n": 1, "core": [], "plausible": True}}]
            score = {"macro_exact_match": 1.0, "per_case_rate": {"unique": 1.0}}
            config = {"dataset": "visible", "variant": "full", "budget_per_item": 1,
                      "replicate": 1, "complete": True}
            paths = persist_replicate(items, answers, metas, score, config, first)
            self.assertEqual(set(paths), {"answers", "score", "diagnostics", "config"})
            for path in paths.values():
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            saved_answers = json.loads(Path(paths["answers"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_answers, answers)
            saved_score = json.loads(Path(paths["score"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_score, score)
            diagnostic_text = Path(paths["diagnostics"]).read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE RESPONSE", diagnostic_text)
            self.assertNotIn("AT|A|07:00", diagnostic_text)
            self.assertTrue(json.loads(diagnostic_text)["complete"])
            with self.assertRaises(FileExistsError):
                persist_replicate(items, answers, metas, score, config, first)

    def test_incomplete_replicate_is_named_partial_and_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = os.path.join(temp, "experiment")
            os.mkdir(out_dir, 0o700)
            items = [{"id": "SYN-000"}, {"id": "SYN-001"}]
            answers = {"SYN-000": {"case": "unique", "assignment": {}}}
            metas = [
                {"id": "SYN-000", "calls": 1, "responses_received": 1,
                 "parses": [], "final": {}, "outcome": {"kind": "unique", "n": 1}},
                {"id": "SYN-001", "calls": 1, "responses_received": 1,
                 "failure": "model_uncertain_extraction", "failure_stage": "semantic_extraction",
                 "parses": [], "final": {}},
            ]
            config = {"dataset": "visible", "variant": "full", "budget_per_item": 1,
                      "replicate": 2, "complete": False}
            paths = persist_replicate(items, answers, metas, {"macro_exact_match": 0.0}, config, out_dir)
            self.assertTrue(paths["answers"].endswith(".answers.partial.json"))
            self.assertTrue(os.path.exists(paths["answers"]))
            self.assertFalse(os.path.exists(paths["answers"].replace(".partial", "")))
            diagnostic = json.loads(Path(paths["diagnostics"]).read_text(encoding="utf-8"))
            self.assertFalse(diagnostic["complete"])
            self.assertEqual(diagnostic["failed_item_count"], 1)
            self.assertEqual(diagnostic["attempted_model_calls"], 2)


if __name__ == "__main__":
    unittest.main()
