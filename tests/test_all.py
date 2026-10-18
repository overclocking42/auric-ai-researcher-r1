"""Offline tests (no network, no key). Run:  python3 -m unittest discover -s tests -v"""
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from collections import Counter
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "dev"))

from preflight import find_key_material                     # noqa: E402
from mock_model import FailingModel, MockModel              # noqa: E402
from synthetic import (BETWEEN_BODY, PEOPLE, STATIONS, UNIQUE_BODY,
                       gold_for, item, items as synthetic_items)  # noqa: E402
from auric.cli import main, run                             # noqa: E402
from auric.config import Config                             # noqa: E402
from auric.extract import NONE, fmt, parse_output            # noqa: E402
from auric.header import parse_header, split_item           # noqa: E402
from auric.resolve import consensus, constraints            # noqa: E402
from auric.solver import Rota, solve                        # noqa: E402
from auric.prompts import COMPLETENESS_RULES, RULES, system_prompt, user_whole  # noqa: E402
from completeness import find_possible_omissions             # noqa: E402

ITEMS = synthetic_items()
GOLD = gold_for(ITEMS)


class SolverSynthetic(unittest.TestCase):
    def test_constructed_constraints_produce_the_known_assignment(self):
        from auric.pipeline import _answer
        fixture = item()
        h = parse_header(fixture["text"].split("\n")[0])
        rota = Rota(h)
        readings = {i: g["reading"] for i, g in enumerate(GOLD[fixture["id"]])}
        answer = _answer(rota, UNIQUE_BODY, solve(rota, constraints(readings)), Config(), readings)
        self.assertEqual(answer["case"], "unique")
        self.assertEqual(answer["assignment"], {
            PEOPLE[0]: {"block": "07:00", "station": STATIONS[0]},
            PEOPLE[1]: {"block": "09:00", "station": STATIONS[1]},
            PEOPLE[2]: {"block": "11:00", "station": STATIONS[2]},
            PEOPLE[3]: {"block": "13:00"}, PEOPLE[4]: {"block": "15:00"},
        })

    def test_world_space_is_720(self):
        it = ITEMS[0]
        self.assertEqual(len(Rota(parse_header(it["text"].split("\n")[0])).worlds), 720)


class Budgets(unittest.TestCase):
    def _run(self, budget, model, **kw):
        return run(ITEMS[:20], budget, Config(**kw) if kw else Config(), model, workers=1, progress=False)

    def test_1x_is_exactly_one_call(self):
        mm = MockModel(GOLD)
        _, metas = self._run(1, mm)
        self.assertTrue(all(m["calls"] == 1 for m in metas))
        self.assertTrue(all(mm.calls[it["id"]] == 1 for it in ITEMS[:20]))

    def test_3x_and_10x_within_cap_and_at_least_one(self):
        for b in (3, 10):
            mm = MockModel(GOLD, 0.08, 0.08)
            _, metas = self._run(b, mm)
            for m in metas:
                self.assertTrue(1 <= m["calls"] <= b, (b, m["id"], m["calls"]))
            self.assertTrue(all(mm.calls[it["id"]] <= b for it in ITEMS[:20]))

    def test_errors_still_count_and_never_retry(self):
        fm = FailingModel()
        ans, metas = self._run(1, fm)
        self.assertTrue(all(n == 1 for n in fm.n.values()))      # one attempt, no retry
        self.assertTrue(all(m["calls"] == 1 for m in metas))
        self.assertEqual(ans, {})                                 # failed calls cannot fabricate answers
        self.assertTrue(all(m["failure"] == "transport_error" for m in metas))
        self.assertTrue(all(m["failure_stage"] == "transport" for m in metas))

    def test_cli_atomically_replaces_explicit_existing_output_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            item_path, out_path, trace_path, diag_path, keep_path = [
                os.path.join(d, n) for n in
                ("items.json", "answers.json", "trace.jsonl",
                 "answers.json.diagnostics.json", "keep.txt")]
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(ITEMS[:1], f)
            for path in (out_path, trace_path, diag_path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("stale non-JSON content")
            with open(keep_path, "w", encoding="utf-8") as f:
                f.write("unrelated content")
            client = MockModel(GOLD)
            rc = main([item_path, "--budget", "1x", "--out", out_path,
                       "--trace", trace_path], client=client)
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls[ITEMS[0]["id"]], 1)
            with open(out_path, encoding="utf-8") as f:
                answers = json.load(f)
            self.assertEqual(set(answers), {ITEMS[0]["id"]})
            self.assertIn(answers[ITEMS[0]["id"]]["case"],
                          ("unique", "ambiguous", "inconsistent"))
            with open(diag_path, encoding="utf-8") as f:
                diagnostics = json.load(f)
            self.assertTrue(diagnostics["complete"])
            self.assertEqual(diagnostics["attempted_model_calls"], 1)
            with open(trace_path, encoding="utf-8") as f:
                trace_rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(trace_rows), 1)
            self.assertEqual(trace_rows[0]["id"], ITEMS[0]["id"])
            with open(keep_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "unrelated content")

    def test_malformed_input_does_not_call_model_or_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            item_path = os.path.join(d, "items.json")
            out_path = os.path.join(d, "answers.json")
            diag_path = out_path + ".diagnostics.json"
            with open(item_path, "w", encoding="utf-8") as f:
                f.write("{ definitely not JSON")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("preserve prior output")
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write("preserve prior diagnostics")
            client = FailingModel()
            with self.assertRaises(SystemExit) as error:
                main([item_path, "--budget", "1x", "--out", out_path], client=client)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(client.n, {})
            with open(out_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "preserve prior output")
            with open(diag_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "preserve prior diagnostics")

    def test_output_path_cannot_alias_input_items_file(self):
        with tempfile.TemporaryDirectory() as d:
            item_path = os.path.join(d, "items.json")
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(ITEMS[:1], f)
            client = FailingModel()
            with self.assertRaises(SystemExit) as error:
                main([item_path, "--budget", "1x", "--out", item_path], client=client)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(client.n, {})
            with open(item_path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), ITEMS[:1])

    def test_all_none_extraction_cannot_be_serialized_as_ambiguous(self):
        item = ITEMS[0]
        _, body = split_item(item)

        class AllNone:
            def complete(self, item_id, messages, max_tokens):
                return "\n".join("%d|NONE" % i for i in range(1, len(body) + 1))

        ans, metas = run([item], 1, Config(), AllNone(), workers=1, progress=False)
        self.assertEqual(ans, {})
        self.assertEqual(metas[0]["failure"], "implausible_solver_outcome")
        self.assertEqual(metas[0]["outcome"]["n"], 720)

    def test_unsettled_vote_tie_is_not_broken_by_first_parse(self):
        item = ITEMS[0]
        rota = Rota(parse_header(item["text"].split("\n")[0]))
        left = ("AT", PEOPLE[1], "07:00")
        right = ("AT", PEOPLE[1], "09:00")
        readings, info = consensus(
            rota, 1, {0: Counter({left: 1, right: 1})},
            {0: left})
        self.assertEqual(readings, {})
        self.assertEqual(info["ties"], [0])
        self.assertEqual(info["unresolved"], [0])

    def test_multiple_plausible_vote_resolutions_remain_uncertain(self):
        item = ITEMS[0]
        rota = Rota(parse_header(item["text"].split("\n")[0]))
        left = ("AT", PEOPLE[1], "07:00")
        right = ("AT", PEOPLE[1], "09:00")

        class Plausible:
            def plausible(self):
                return True

        with patch("auric.resolve.solve", return_value=Plausible()):
            readings, info = consensus(
                rota, 1, {0: Counter({left: 1, right: 1})}, {0: left})
        self.assertEqual(readings, {})
        self.assertEqual(info["ties"], [0])
        self.assertEqual(info["unresolved"], [0])

    def test_missing_vote_stays_absent_instead_of_becoming_none(self):
        item = ITEMS[0]
        rota = Rota(parse_header(item["text"].split("\n")[0]))
        readings, info = consensus(rota, 2, {}, {})
        self.assertEqual(readings, {})
        self.assertEqual(info["unresolved"], [0, 1])

    def test_every_item_gets_a_call(self):
        mm = MockModel(GOLD)
        _, metas = run(ITEMS, 10, Config(), mm, workers=2, progress=False)
        self.assertTrue(all(m["calls"] >= 1 for m in metas))


class ModelDependence(unittest.TestCase):
    def test_answers_change_with_model_output(self):
        base, _ = run(ITEMS, 1, Config(), MockModel(GOLD), workers=1, progress=False)
        blind, _ = run(ITEMS, 1, Config(), MockModel(GOLD, mode="blind"), workers=1, progress=False)
        flip, _ = run(ITEMS, 1, Config(), MockModel(GOLD, mode="flipped"), workers=1, progress=False)
        diff_blind = sum(base[i] != blind.get(i) for i in base)
        diff_flip = sum(base[i] != flip.get(i) for i in base)
        self.assertGreater(diff_blind, 50)
        self.assertGreater(diff_flip, 50)


class Parser(unittest.TestCase):
    def setUp(self):
        it = ITEMS[0]
        self.h = parse_header(it["text"].split("\n")[0])
        _, self.body = split_item(it)
        self.cfg = Config()

    def test_tolerates_noise_in_format(self):
        txt = "Sure!\n```\n1|NONE\n- 2 | not_at | Rhea Vale | 11:00\n[3]: AT|Kian|7:00\n```"
        body = ["x", "Rhea Vale is off 11:00", "Kian 07:00"]
        p = parse_output(txt, self.h, body, self.cfg)
        self.assertEqual(p.readings[0], NONE)
        self.assertEqual(p.readings[1], ("NOT_AT", "Rhea Vale", "11:00"))
        self.assertEqual(p.readings[2], ("AT", "Kian", "07:00"))

    def test_rejects_invented_entities_and_ungrounded_args(self):
        body = ["Rhea Vale is on cold-store."]
        p = parse_output("1|ON|Zed|cold-store", self.h, body, self.cfg)
        self.assertIn(0, p.invalid)
        p = parse_output("1|ON|Noel|cold-store", self.h, body, self.cfg)   # Noel not in the line
        self.assertIn(0, p.invalid)
        p = parse_output("1|ON|Noel|cold-store", self.h, body, Config.from_names(["no_grounding"]))
        self.assertIn(0, p.readings)

    def test_direction_aliases(self):
        body = ["Rhea Vale follows Kian."]
        p = parse_output("1|IMMEDIATELY_AFTER|Rhea Vale|Kian", self.h, body, self.cfg)
        self.assertEqual(p.readings[0], ("RIGHT_BEFORE", ("p", "Kian"), ("p", "Rhea Vale")))

    def test_station_holder_reference(self):
        body = ["Whoever has dispatch is on before Noel."]
        p = parse_output("1|BEFORE|@dispatch|Noel", self.h, body, self.cfg)
        self.assertEqual(p.readings[0], ("BEFORE", ("h", "dispatch"), ("p", "Noel")))

    def test_changed_header_names_station_labels_blocks_and_order_are_supported(self):
        header = (
            "Shift notes. 4 staff on rota: Aria Vale, Bo, Cy, Dee. "
            "Blocks run dawn shift, midday, dusk, night, one person per block, "
            "and each person works exactly one block. There are 2 stations, one person on each: "
            "cold-store, dispatch. The people on a station are Aria Vale, Bo; the rest are on no station."
        )
        h = parse_header(header)
        self.assertEqual(len(Rota(h).worlds), 48)
        body = ["I'm fairly sure Aria Vale is on dawn shift.",
                "Worth repeating: Aria Vale is on dawn shift.",
                "The delivery arrives at noon."]
        parsed = parse_output("1|AT|Aria Vale|dawn shift\n"
                              "2|AT|Aria Vale|dawn shift\n3|NONE", h, body, Config())
        self.assertEqual(parsed.invalid, {})
        self.assertEqual(parsed.readings[0], ("AT", "Aria Vale", "dawn shift"))
        self.assertEqual(parsed.readings[1], ("AT", "Aria Vale", "dawn shift"))
        self.assertEqual(parsed.readings[2], NONE)
        ordered = user_whole(h, body, 1, "UNRELATED-ID", Config())
        self.assertLess(ordered.index("[3] The delivery"), ordered.index("[1] I'm fairly sure"))
        self.assertIn("[1] I'm fairly sure", ordered)

    def test_garbage_never_raises(self):
        for txt in (None, "", "no structure at all", "99|AT|x|y", "1|BETWEEN|a"):
            parse_output(txt, self.h, self.body, self.cfg)

    def test_missing_operand_is_rejected_and_unsure_is_explicit_failure(self):
        body = ["Rhea Vale is on the 11:00 block."]
        missing = parse_output("1|AT|Rhea Vale", self.h, body, self.cfg)
        self.assertIn(0, missing.invalid)
        self.assertNotIn(0, missing.readings)
        unsure = parse_output("1|UNSURE", self.h, body, self.cfg)
        self.assertEqual(unsure.invalid, {0: "model_uncertain"})
        self.assertNotIn(0, unsure.readings)
        mixed = parse_output("1|UNSURE\n1|AT|Rhea Vale|11:00", self.h, body, self.cfg)
        self.assertEqual(mixed.invalid, {0: "model_uncertain"})
        self.assertNotIn(0, mixed.readings)

    def test_duplicate_identical_record_is_not_silently_deduplicated(self):
        body = ["The schedule has no current fact here."]
        parsed = parse_output("1|NONE" + chr(10) + "1|NONE", self.h, body, self.cfg)
        self.assertEqual(parsed.invalid, {0: "duplicate statement records"})
        self.assertNotIn(0, parsed.readings)

    def test_reversed_between_endpoints_are_equivalent(self):
        line = "Put Rhea Vale between Kian and Noel, though not necessarily next to either."
        a = parse_output("1|BETWEEN|Rhea Vale|Kian|Noel", self.h, [line], self.cfg)
        b = parse_output("1|BETWEEN|Rhea Vale|Noel|Kian", self.h, [line], self.cfg)
        self.assertEqual(a.readings[0], b.readings[0])
        rota = Rota(self.h)
        self.assertEqual(rota.mask(a.readings[0]), rota.mask(b.readings[0]))

    def test_conflicting_notes_produce_minimal_inconsistent_solver_core(self):
        body = ["Kian is on 07:00.", "Kian is on 09:00."]
        parsed = parse_output("1|AT|Kian|07:00\n2|AT|Kian|09:00", self.h, body, self.cfg)
        self.assertEqual(parsed.invalid, {})
        result = solve(Rota(self.h), constraints(parsed.readings))
        self.assertEqual((result.kind, result.n), ("inconsistent", 0))
        self.assertEqual(result.core, (0, 1))
        for removed in range(len(result.core)):
            kept = [c for i, c in enumerate(constraints(parsed.readings)) if i != removed]
            self.assertGreater(solve(Rota(self.h), kept).n, 0)

    def test_prompt_requires_operand_grounding_in_the_cited_note(self):
        self.assertIn("supported by its numbered note alone", RULES)
        self.assertIn("AT/NOT_AT needs its person and time there", RULES)
        self.assertIn("Never borrow a missing operand from another note", RULES)
        self.assertIn("UNSURE is an explicit extraction failure", RULES)
        self.assertIn("the final two endpoints are interchangeable", RULES)
        self.assertIn("Do not infer adjacency from a generic handover", RULES)
        self.assertIn("BEFORE means strictly earlier and permits any gap", RULES)
        self.assertIn('"X is later than Y" means BEFORE|Y|X', RULES)
        self.assertIn('"Y takes over from X later in the day" means BEFORE|X|Y', RULES)
        self.assertIn('"by the time X starts, Y has already worked" means BEFORE|Y|X', RULES)
        self.assertIn("For BETWEEN, the first operand is the middle", RULES)


class CompletenessAudit(unittest.TestCase):
    def test_prompt_has_relation_completeness_self_check(self):
        self.assertIn("Completeness self-check before answering", COMPLETENESS_RULES)
        self.assertIn("If such a cue states a current rota fact, encode that relation", COMPLETENESS_RULES)
        self.assertIn("This is a check for omissions, not permission to infer unstated relations", COMPLETENESS_RULES)
        self.assertIn(COMPLETENESS_RULES, system_prompt(0, Config()))
        self.assertNotIn(COMPLETENESS_RULES, system_prompt(0, Config.from_names(["no_completeness"])))

    def test_audit_flags_relation_cues_without_creating_constraints(self):
        examples = [
            ("Rhea Vale then Kian, back to back.", "RIGHT_BEFORE", "RIGHT_BEFORE"),
            ("Noel will not work the 15:00 block.", "NOT_AT", "NOT_AT"),
            ("Mara-June has not been put on dispatch.", "NOT_ON", "NOT_ON"),
            ("Kian runs label room this week.", "ON", "ON"),
            ("Label room is covered earlier in the day than Noel's block.", "ORDERING", "BEFORE"),
            ("Mara-June is between Rhea Vale and Kian.", "BETWEEN", "BETWEEN"),
        ]
        for text, cue, relation in examples:
            item_fixture = item("fixture", [text])
            alerts = find_possible_omissions(item_fixture, {"1": "NONE"})
            self.assertEqual(alerts[0]["cue_families"], [cue])
            self.assertEqual(alerts[0]["observed_relation"], "NONE")
            accepted = find_possible_omissions(item_fixture, {"1": relation + "|..."})
            self.assertEqual(accepted, [])

    def test_historical_between_is_not_reported_as_a_current_omission(self):
        fixture = item("history", ["Last month Mara-June was between Rhea Vale and Kian."])
        self.assertEqual(find_possible_omissions(fixture, {"1": "NONE"}), [])

    def test_other_than_station_exclusion_matches_not_on(self):
        fixture = item("station-exclusion", ["Label room is covered by someone other than Noel."])
        self.assertEqual(find_possible_omissions(fixture, {"1": "NOT_ON|Noel|label room"}), [])

    def test_synthetic_between_completeness_changes_world_count(self):
        """An omitted source-grounded BETWEEN is flagged and changes 2 worlds to 1."""
        fixture = item("synthetic-between", BETWEEN_BODY)
        header, body = split_item(fixture)
        rota = Rota(parse_header(header))
        constraints_by_line = {
            1: "AT|Rhea Vale|07:00",
            2: "AT|Kian|09:00",
            3: "AT|Tao|15:00",
            5: "ON|Rhea Vale|cold-store",
            6: "ON|Kian|label room",
            7: "ON|Mara-June|dispatch",
        }

        def extract(include_between):
            rows = []
            for i in range(1, len(body) + 1):
                reading = ("BETWEEN|Mara-June|Kian|Noel" if include_between and i == 4
                           else constraints_by_line.get(i, "NONE"))
                rows.append(f"{i}|{reading}")
            parsed = parse_output("\n".join(rows), parse_header(header), body, Config())
            self.assertEqual(parsed.invalid, {})
            self.assertEqual(len(parsed.readings), len(body))
            return parsed.readings

        omitted = extract(False)
        corrected = extract(True)
        omitted_alerts = find_possible_omissions(
            fixture, {str(i + 1): fmt(reading) for i, reading in omitted.items()})
        corrected_alerts = find_possible_omissions(
            fixture, {str(i + 1): fmt(reading) for i, reading in corrected.items()})
        self.assertIn(4, [alert["statement_index"] for alert in omitted_alerts])
        self.assertNotIn(4, [alert["statement_index"] for alert in corrected_alerts])
        without = solve(rota, constraints(omitted))
        with_relation = solve(rota, constraints(corrected))
        self.assertEqual((without.kind, without.n), ("ambiguous", 2))
        self.assertTrue(without.plausible())
        self.assertEqual((with_relation.kind, with_relation.n), ("unique", 1))
        self.assertTrue(with_relation.plausible())
        self.assertEqual(corrected[3][0], "BETWEEN")


class Rules(unittest.TestCase):
    def test_preflight_ignores_local_env_but_scans_submission_files(self):
        with tempfile.TemporaryDirectory() as d:
            fake_key = "sk-or-v1-" + ("a1" * 20)
            with open(os.path.join(d, ".env"), "w", encoding="utf-8") as f:
                f.write("OPENROUTER_API_KEY=" + fake_key)
            self.assertEqual(find_key_material(d), [])
            with open(os.path.join(d, "module.py"), "w", encoding="utf-8") as f:
                f.write("credential = " + repr(fake_key))
            self.assertEqual(find_key_material(d), ["module.py"])

    def test_runtime_never_touches_dev_or_key(self):
        bad = re.compile(r"oracle|visible_key|score\.py|import score|dev\.|from dev|import gold", re.I)
        for d, _, files in os.walk(os.path.join(ROOT, "src")):
            for f in files:
                if f.endswith(".py"):
                    src = Path(d, f).read_text(encoding="utf-8")
                    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"""')))
                    self.assertIsNone(bad.search(code), "%s references dev/key code" % f)
        self.assertNotIn("oracle", Path(ROOT, "run").read_text(encoding="utf-8"))

    def test_client_headers_and_settings(self):
        src = Path(ROOT, "src", "auric", "client.py").read_text(encoding="utf-8")
        for needle in ('"X-Item-Id": item_id', "temperature=1.0", "top_p=0.95",
                       '"reasoning": {"enabled": False}', "max_retries=0"):
            self.assertIn(needle, src)

    def test_no_secret_in_repo(self):
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in {".git", ".venv", "runs", "__pycache__"}]
            for f in files:
                if f.endswith((".py", ".md", ".txt", ".json", ".example")) or f == "run":
                    t = Path(d, f).read_text(encoding="utf-8", errors="ignore")
                    self.assertNotRegex(t, r"sk-or-v1-[0-9a-f]{16,}", f)

    def test_submission_zip_excludes_answer_keys_and_local_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "review.zip")
            subprocess.run([sys.executable, os.path.join(ROOT, "dev", "make_zip.py"),
                            "--out", out], check=True, cwd=ROOT, capture_output=True, text=True)
            with zipfile.ZipFile(out) as z:
                names = set(z.namelist())
            self.assertIn("auric-r1/tests/synthetic.py", names)
            self.assertIn("auric-r1/src/auric/pipeline.py", names)
            self.assertIn("auric-r1/ABLATION_TABLE.md", names)
            self.assertIn("auric-r1/SUBMISSION_CHECKLIST.md", names)
            self.assertIn("auric-r1/dev/replay_cached.py", names)
            self.assertIn("auric-r1/EVALUATION_REPORT.md", names)
            self.assertFalse(any("/data/" in n or n.endswith("/.env") or n.endswith("/visible_key.json")
                                 or n.endswith("/heldout_synth/key.json")
                                 or n.endswith("/heldout_synth/gold.json")
                                 or n.endswith("/.DS_Store") or n.endswith(".zip")
                                 or "/runs/" in n or "__pycache__" in n for n in names))


if __name__ == "__main__":
    unittest.main()
