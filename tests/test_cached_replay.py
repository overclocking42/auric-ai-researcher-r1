"""Offline tests for fixed-completion replay; no model endpoint is contacted."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dev"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from replay_cached import CachedCompletionClient, ReplayInputError, replay
from auric.extract import fmt
from auric.client import ModelCallFailure
from synthetic import gold_for, items as synthetic_items

ITEMS = synthetic_items(1)
GOLD = gold_for(ITEMS)


class FakeScorer:
    @staticmethod
    def score(key, answers, items):
        return {"macro_exact_match": len(answers) / len(items),
                "exact_match": len(answers) / len(items)}


class CachedReplayTests(unittest.TestCase):
    def test_client_returns_saved_whole_and_targeted_records_by_kind(self):
        client = CachedCompletionClient([
            {"kind": "whole", "text": "whole-1"},
            {"kind": "targeted", "text": "target-1"},
            {"kind": "whole", "text": "whole-2"},
        ])
        self.assertEqual(client.complete("x", [{"content": "NOTES"}], 1), "whole-1")
        self.assertEqual(client.complete("x", [{"content": "MODE: TARGETED"}], 1), "target-1")
        self.assertEqual(client.complete("x", [{"content": "NOTES"}], 1), "whole-2")
        with self.assertRaises(ModelCallFailure) as caught:
            client.complete("x", [{"content": "NOTES"}], 1)
        self.assertEqual(caught.exception.diagnostic["category"], "cached_replay_missing")

    def test_fixed_synthetic_completions_run_all_downstream_variants_without_raw_output(self):
        item = ITEMS[0]
        text = "\n".join("%d|%s" % (i + 1, fmt(row["reading"]))
                          for i, row in enumerate(GOLD[item["id"]]))
        records = [{"kind": "whole", "variant": i, "text": text} for i in range(3)]
        trace = [{"id": item["id"], "raw": records}]
        with tempfile.TemporaryDirectory() as temp:
            out = os.path.join(temp, "replay")
            result = replay([item], trace, {}, FakeScorer, out)
            self.assertEqual(len(result), 5)
            self.assertTrue(all(row["complete"] and row["answer_count"] == 1 for row in result))
            for name in os.listdir(out):
                with open(os.path.join(out, name), encoding="utf-8") as stream:
                    content = stream.read()
                self.assertNotIn(text, content)

    def test_trace_must_match_item_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ReplayInputError):
                replay(ITEMS[:1], [{"id": "OTHER", "raw": []}], {}, FakeScorer,
                       os.path.join(temp, "new"))


if __name__ == "__main__":
    unittest.main()
