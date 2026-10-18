"""End-to-end through the REAL client code path against a local fake OpenAI-compatible server.
Verifies headers/sampling parameters on the wire and per-item call counts. No key, no internet."""
import json
import io
import os
import socket
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "dev"))
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from mock_model import MockModel              # noqa: E402
from synthetic import gold_for, items as synthetic_items  # noqa: E402
from auric.cli import _item_diagnostic, _write_diagnostics, main, run  # noqa: E402
from auric.client import Budgeted, ModelCallFailure, exception_diagnostic  # noqa: E402
from auric.pipeline import ItemRunFailure, answer_item  # noqa: E402
from auric.config import Config                # noqa: E402
from auric.header import parse_header          # noqa: E402
from smoke_test import failure_summary, write_protected_capture  # noqa: E402

ITEMS = synthetic_items()
GOLD = gold_for(ITEMS)
SEEN = []


class H(BaseHTTPRequestHandler):
    mock = MockModel(GOLD, 0.05, 0.05)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        iid = self.headers.get("X-Item-Id")
        SEEN.append({"item": iid, "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                     "reasoning": body.get("reasoning"), "model": body.get("model")})
        text = H.mock.complete(iid, body["messages"], body.get("max_tokens"))
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model"),
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text}}]}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class Wire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ["OPENROUTER_BASE_URL"] = "http://127.0.0.1:%d/v1" % cls.srv.server_address[1]
        os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
        os.environ["MODEL"] = "ibm-granite/granite-4.2-8b"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _go(self, budget, n=12, eps=0.05, trap=0.05):
        SEEN.clear()
        H.mock = MockModel(GOLD, eps, trap)
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "items.json")
            out = os.path.join(d, "answers.json")
            with open(sub, "w", encoding="utf-8") as stream:
                json.dump(ITEMS[:n], stream)
            rc = main([sub, "--budget", "%dx" % budget, "--out", out, "--workers", "3"])
            with open(out, encoding="utf-8") as stream:
                answers = json.load(stream)
            return rc, answers

    def test_1x_wire(self):
        # A perfect synthetic parse isolates the HTTP contract from the
        # separately tested behavior for malformed or implausible extractions.
        rc, ans = self._go(1, eps=0, trap=0)
        self.assertEqual(rc, 0)
        self.assertEqual(len(ans), 12)
        per = {}
        for s in SEEN:
            per[s["item"]] = per.get(s["item"], 0) + 1
            self.assertEqual(s["temperature"], 1.0)
            self.assertEqual(s["top_p"], 0.95)
            self.assertEqual(s["reasoning"], {"enabled": False})
            self.assertEqual(s["model"], "ibm-granite/granite-4.2-8b")
        self.assertEqual(set(per), {it["id"] for it in ITEMS[:12]})
        self.assertTrue(all(v == 1 for v in per.values()), per)

    def test_10x_wire_within_cap(self):
        # Keep this wire/budget test deterministic; noisy tie handling is
        # exercised separately by the consensus tests.
        rc, _ = self._go(10, eps=0, trap=0)
        self.assertEqual(rc, 0)
        per = {}
        for s in SEEN:
            per[s["item"]] = per.get(s["item"], 0) + 1
        self.assertTrue(all(1 <= v <= 10 for v in per.values()), per)
        self.assertTrue(all(s["item"] for s in SEEN))              # header always present


class FailureDiagnostics(unittest.TestCase):
    class FakeFailure(Exception):
        pass

    def _failure(self, status, body, headers=None):
        exc = self.FakeFailure("provider error")
        exc.status_code = status
        exc.body = body
        exc.response = SimpleNamespace(headers=headers or {})
        return exc

    def test_429_keeps_allowlisted_provider_details_and_retry_after(self):
        exc = self._failure(429, {"error": {"code": "rate_limit", "message": "shared pool busy",
            "metadata": {"limit_source": "upstream_provider_shared_pool", "reason": "capacity",
                         "secret": "must not appear"}}}, {"Retry-After": "13", "X-Request-Id": "req-1"})
        d = exception_diagnostic(exc, "B2-000")
        self.assertEqual((d["http_status"], d["category"], d["stage"]), (429, "http_429", "http_response"))
        self.assertEqual(d["limit_source"], "upstream_provider_shared_pool")
        self.assertEqual(d["retry_after"], "13")
        self.assertEqual(d["x_request_id"], "req-1")
        self.assertNotIn("must not appear", repr(d))

    def test_other_non_2xx_response_is_distinguished(self):
        d = exception_diagnostic(self._failure(403, {"error": {"message": "forbidden"}}), "B2-000")
        self.assertEqual(d["http_status"], 403)
        self.assertEqual(d["category"], "http_error")
        self.assertEqual(d["stage"], "http_response")
        self.assertEqual(d["provider_message"], "forbidden")

    def test_transport_errno_is_preserved(self):
        exc = socket.gaierror(8, "nodename not known")
        d = exception_diagnostic(exc, "B2-000")
        self.assertEqual(d["category"], "transport_error")
        self.assertEqual(d["stage"], "transport")
        self.assertEqual(d["errno"], 8)
        self.assertIn("nodename", d["message"])

    def test_nested_transport_cause_is_preserved_safely(self):
        outer = ConnectionError("connection failed")
        inner = socket.gaierror(8, "temporary DNS failure")
        outer.__cause__ = inner
        d = exception_diagnostic(outer, "B2-000")
        self.assertEqual(d["exception_class"], "ConnectionError")
        self.assertEqual(d["cause_class"], "gaierror")
        self.assertEqual(d["errno"], 8)
        self.assertIn("temporary DNS", d["message"])

    def test_json_decoding_exception_has_response_stage(self):
        exc = json.JSONDecodeError("invalid JSON", "{", 1)
        d = exception_diagnostic(exc, "B2-000")
        self.assertEqual(d["item_id"], "B2-000")
        self.assertEqual(d["stage"], "response_decoding")
        self.assertEqual(d["category"], "response_decoding_failure")

    def test_error_text_redacts_credentials_and_authorization(self):
        bearer = "sk-or-v1-" + ("0123456789abcdef" * 2)
        env_value = "sk-or-v1-" + ("abcdef0123456789" * 2)
        exc = ConnectionError("Authorization: Bearer %s OPENROUTER_API_KEY=%s" % (bearer, env_value))
        d = exception_diagnostic(exc, "B2-000")
        self.assertNotIn("sk-or-v1", repr(d))
        self.assertNotIn("0123456789abcdef", repr(d))
        self.assertIn("[REDACTED", d["message"])

    def test_provider_key_management_link_identifier_is_redacted(self):
        opaque_id = "opaque-management-id-123456"
        message = ("Key limit exceeded. Manage it using "
                   "https://openrouter.ai/workspaces/default/keys/" + opaque_id)
        d = exception_diagnostic(ConnectionError(message), "B2-000")
        self.assertIn("Key limit exceeded", d["message"])
        self.assertIn("/keys/[REDACTED]", d["message"])
        self.assertNotIn(opaque_id, repr(d))

    def test_budgeted_http_error_is_counted_and_not_retried(self):
        class Failing:
            calls = 0
            def complete(self, item_id, messages, max_tokens):
                self.calls += 1
                raise FailureDiagnostics()._failure(429, {"error": {"code": "rate_limit"}})
        client = Failing()
        proxy = Budgeted(client, "B2-000", 1, 100)
        with self.assertRaises(ModelCallFailure) as caught:
            proxy.ask("system", "user")
        self.assertEqual((proxy.used, client.calls), (1, 1))
        self.assertEqual(caught.exception.diagnostic["http_status"], 429)

    def test_http_failure_survives_pipeline_into_runner_metadata(self):
        class Failing:
            calls = 0
            def complete(inner, item_id, messages, max_tokens):
                inner.calls += 1
                raise self._failure(429, {"error": {"code": "rate_limit",
                    "metadata": {"limit_source": "shared_pool"}}})
        ans, metas = run([ITEMS[0]], 1, Config(), Failing(), workers=1, progress=False)
        self.assertEqual(ans, {})
        self.assertEqual(metas[0]["calls"], 1)
        self.assertEqual(metas[0]["failure"], "http_429")
        self.assertEqual(metas[0]["diagnostic"]["limit_source"], "shared_pool")

    def test_stop_on_provider_failure_does_not_start_later_items(self):
        class Failing:
            calls = 0
            def complete(inner, item_id, messages, max_tokens):
                inner.calls += 1
                raise self._failure(429, {"error": {"code": "rate_limit"}})

        client = Failing()
        ans, metas = run(ITEMS[:3], 1, Config(), client, workers=4,
                         progress=False, stop_on_provider_failure=True)
        self.assertEqual(ans, {})
        self.assertEqual(client.calls, 1)
        self.assertEqual(metas[0]["failure"], "http_429")
        self.assertEqual([m["calls"] for m in metas], [1, 0, 0])
        self.assertEqual([m["failure_stage"] for m in metas[1:]],
                         ["not_attempted", "not_attempted"])

    def test_incomplete_cli_run_writes_complete_operand_free_diagnostics(self):
        class AllNone:
            def complete(inner, item_id, messages, max_tokens):
                item = next(i for i in ITEMS if i["id"] == item_id)
                body = item["text"].split("\n")[1:]
                return "\n".join("%d|NONE" % n for n in range(1, len(body) + 1))

        with tempfile.TemporaryDirectory() as d:
            item_path, answer_path, diag_path = [os.path.join(d, n) for n in
                ("items.json", "answers.json", "diagnostics.json")]
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(ITEMS[:1], f)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = main([item_path, "--budget", "1x", "--out", answer_path,
                           "--diagnostics", diag_path], client=AllNone())
            self.assertEqual(rc, 2)
            with open(answer_path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {})
            with open(diag_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertFalse(report["complete"])
            self.assertTrue(report["answer_file_written"])
            self.assertEqual(report["answer_file_status"], "partial")
            self.assertEqual(report["missing_item_ids"], [ITEMS[0]["id"]])
            self.assertEqual(report["item_count"], 1)
            self.assertEqual(report["attempted_model_calls"], 1)
            row = report["items"][0]
            self.assertEqual(row["model_responses_received"], 1)
            self.assertEqual(row["failure"], "implausible_solver_outcome")
            self.assertTrue(row["solver_ran"])
            self.assertEqual(row["solver_world_count"], 720)
            self.assertEqual(row["solver_classification"], "ambiguous")
            self.assertFalse(row["answer_serialized"])
            self.assertNotIn("Meera", json.dumps(report))
            self.assertEqual(stat.S_IMODE(os.stat(diag_path).st_mode), 0o600)

    def test_partial_answers_are_exclusive_private_and_never_marked_complete(self):
        class OneSuccessOneFailure:
            def __init__(self):
                self.good = MockModel(GOLD)

            def complete(self, item_id, messages, max_tokens):
                if item_id == ITEMS[1]["id"]:
                    raise OSError("synthetic transport failure")
                return self.good.complete(item_id, messages, max_tokens)

        with tempfile.TemporaryDirectory() as d:
            item_path = os.path.join(d, "items.json")
            answer_path = os.path.join(d, "answers.json")
            partial_path = os.path.join(d, "partial.json")
            diag_path = os.path.join(d, "diagnostics.json")
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(ITEMS[:2], f)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = main([item_path, "--budget", "1x", "--out", answer_path,
                           "--diagnostics", diag_path,
                           "--workers", "1"], client=OneSuccessOneFailure())
            self.assertEqual(rc, 2)
            with open(answer_path, encoding="utf-8") as f:
                partial = json.load(f)
            self.assertEqual(set(partial), {ITEMS[0]["id"]})
            self.assertEqual(stat.S_IMODE(os.stat(answer_path).st_mode), 0o600)
            with open(diag_path, encoding="utf-8") as f:
                diagnostic = json.load(f)
            self.assertFalse(diagnostic["complete"])
            self.assertTrue(diagnostic["answer_file_written"])
            self.assertEqual(diagnostic["answer_file_status"], "partial")
            self.assertTrue(diagnostic["partial_answer_file_written"])
            self.assertTrue(diagnostic["items"][0]["answer_serialized"])
            self.assertFalse(diagnostic["items"][1]["answer_serialized"])
            self.assertIn("PARTIAL answer map", stderr.getvalue())

    def test_partial_failure_replaces_outputs_and_preserves_safe_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            item_path = os.path.join(d, "items.json")
            answer_path = os.path.join(d, "answers.json")
            partial_path = os.path.join(d, "partial.json")
            diag_path = os.path.join(d, "diagnostics.json")
            trace_path = os.path.join(d, "trace.jsonl")
            unrelated = os.path.join(d, "prior-run.txt")
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(ITEMS[:1], f)
            for path in (answer_path, partial_path, diag_path, trace_path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("stale non-JSON content")
            with open(unrelated, "w", encoding="utf-8") as f:
                f.write("preserve this unrelated artifact")

            class FailsOnce:
                def __init__(self):
                    self.calls = 0

                def complete(inner, item_id, messages, max_tokens):
                    inner.calls += 1
                    raise OSError("synthetic transport failure")

            client = FailsOnce()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = main([item_path, "--budget", "1x", "--out", answer_path,
                           "--partial-out", partial_path, "--diagnostics", diag_path,
                           "--trace", trace_path], client=client)
            self.assertEqual(rc, 2)
            self.assertEqual(client.calls, 1)
            for path in (answer_path, partial_path):
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(json.load(f), {})
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            with open(diag_path, encoding="utf-8") as f:
                diagnostic = json.load(f)
            self.assertFalse(diagnostic["complete"])
            self.assertEqual(diagnostic["answer_file_status"], "partial")
            self.assertEqual(diagnostic["attempted_model_calls"], 1)
            self.assertEqual(diagnostic["items"][0]["failure_stage"], "transport")
            with open(trace_path, encoding="utf-8") as f:
                trace_row = json.loads(f.readline())
            self.assertEqual(trace_row["calls"], 1)
            with open(unrelated, encoding="utf-8") as f:
                self.assertEqual(f.read(), "preserve this unrelated artifact")
            self.assertIn("run incomplete", stderr.getvalue())

    def test_safe_diagnostic_keeps_solver_result_but_not_operands_or_raw_text(self):
        meta = {
            "id": "SYNTH-1", "calls": 1, "responses_received": 1,
            "failure": "implausible_solver_outcome", "failure_stage": None,
            "n_lines": 1, "final": {"1": "AT|Secret Person|07:00"},
            "raw": [{"text": "Secret Person is at 07:00"}],
            "outcome": {"kind": "ambiguous", "n": 720, "core": [], "plausible": False},
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "diag.json")
            _write_diagnostics(path, [{"id": "SYNTH-1"}], [meta], {}, False, 1)
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            row = report["items"][0]
            self.assertEqual(row["relation_counts"], {"AT": 1})
            self.assertEqual(row["solver_world_count"], 720)
            self.assertNotIn("Secret Person", json.dumps(report))
            self.assertNotIn("07:00", json.dumps(report))
            self.assertNotIn("Secret Person is at", json.dumps(report))

    def test_synthetic_unique_extraction_preserves_solver_result_and_serializes(self):
        item = {
            "id": "SYNTH-UNIQUE",
            "text": (
                "Shift notes, Test Bay. 5 staff on the rota: Alex, Blake, Casey, Drew, Evan. "
                "Blocks run 07:00, 09:00, 11:00, 13:00, 15:00, one person per block, "
                "and each person works exactly one block. There are 3 stations, one person "
                "on each: intake, calibration, packing. The people on a station are Alex, "
                "Blake, Casey; the rest are on no station.\n\n"
                "Alex is on 07:00.\nBlake is on 09:00.\nCasey is on 11:00.\n"
                "Drew is on 13:00.\nEvan is on 15:00.\n"
                "Alex is on intake.\nBlake is on calibration.\nCasey is on packing."
            ),
        }

        class SyntheticModel:
            def complete(self, item_id, messages, max_tokens):
                return "\n".join([
                    "1|AT|Alex|07:00", "2|AT|Blake|09:00", "3|AT|Casey|11:00",
                    "4|AT|Drew|13:00", "5|AT|Evan|15:00", "6|ON|Alex|intake",
                    "7|ON|Blake|calibration", "8|ON|Casey|packing",
                ])

        answers, metas = run([item], 1, Config(), SyntheticModel(), workers=1,
                             progress=False)
        self.assertEqual(answers["SYNTH-UNIQUE"]["case"], "unique")
        self.assertEqual(metas[0]["outcome"]["kind"], "unique")
        self.assertEqual(metas[0]["outcome"]["n"], 1)
        self.assertEqual(metas[0]["outcome"]["plausibility_reasons"], [])
        row = _item_diagnostic("SYNTH-UNIQUE", metas[0], True)
        self.assertEqual((row["solver_world_count"], row["solver_classification"],
                          row["solver_implausibility_reasons"], row["answer_serialized"]),
                         (1, "unique", [], True))
        self.assertIsNone(row["conflict_core_size"])

    def test_implausibility_diagnostic_reports_reason_and_retains_solver_outcome(self):
        item = {
            "id": "SYNTH-ALL-NONE",
            "text": (
                "Shift notes, Test Bay. 5 staff on the rota: Alex, Blake, Casey, Drew, Evan. "
                "Blocks run 07:00, 09:00, 11:00, 13:00, 15:00, one person per block, "
                "and each person works exactly one block. There are 3 stations, one person "
                "on each: intake, calibration, packing. The people on a station are Alex, "
                "Blake, Casey; the rest are on no station.\n\nAlex is on 07:00."
            ),
        }

        class OmittedRelation:
            def complete(self, item_id, messages, max_tokens):
                return "1|NONE"

        answers, metas = run([item], 1, Config(), OmittedRelation(), workers=1,
                             progress=False)
        self.assertEqual(answers, {})
        meta = metas[0]
        self.assertEqual(meta["failure"], "implausible_solver_outcome")
        self.assertEqual(meta["failure_stage"], "solver_plausibility")
        self.assertEqual(meta["outcome"]["kind"], "ambiguous")
        self.assertEqual(meta["outcome"]["n"], 720)
        self.assertFalse(meta["outcome"]["plausible"])
        self.assertEqual(meta["outcome"]["plausibility_reasons"],
                         ["ambiguous_world_count_out_of_range"])
        row = _item_diagnostic("SYNTH-ALL-NONE", meta, False)
        self.assertEqual(row["solver_world_count"], 720)
        self.assertEqual(row["solver_classification"], "ambiguous")
        self.assertFalse(row["solver_plausible"])
        self.assertEqual(row["solver_implausibility_reasons"],
                         ["ambiguous_world_count_out_of_range"])
        self.assertFalse(row["answer_serialized"])

    def test_explicit_model_uncertainty_is_a_counted_extraction_failure(self):
        item = ITEMS[0]

        class Unsure:
            calls = 0
            def complete(inner, item_id, messages, max_tokens):
                inner.calls += 1
                body = item["text"].split("\n")[1:]
                return "\n".join("%d|UNSURE" % n for n in range(1, len(body) + 1))

        model = Unsure()
        answers, metas = run([item], 1, Config(), model, workers=1, progress=False)
        self.assertEqual(answers, {})
        self.assertEqual(model.calls, 1)
        self.assertEqual(metas[0]["calls"], 1)
        self.assertEqual(metas[0]["failure"], "model_uncertain_extraction")
        self.assertEqual(metas[0]["failure_stage"], "semantic_extraction")

    def test_omitted_statement_record_fails_coverage_without_defaulting_to_none(self):
        item = ITEMS[0]

        class OmittedRecord:
            calls = 0
            def complete(inner, item_id, messages, max_tokens):
                inner.calls += 1
                return "1|NONE"  # only one record; all unmentioned notes stay unresolved

        model = OmittedRecord()
        answers, metas = run([item], 1, Config(), model, workers=1, progress=False)
        self.assertEqual(answers, {})
        self.assertEqual(model.calls, 1)
        self.assertEqual(metas[0]["failure"], "incomplete_extraction")
        self.assertGreater(len(metas[0]["unresolved_line_indices"]), 0)
        self.assertEqual(metas[0]["calls"], 1)

    def test_inconsistent_diagnostic_records_conflict_core_size_and_guard_reason(self):
        meta = {
            "id": "SYNTH-CORE", "calls": 1, "responses_received": 1,
            "failure": "implausible_solver_outcome", "n_lines": 3,
            "final": {"1": "AT|hidden", "2": "AT|hidden", "3": "AT|hidden"},
            "outcome": {"kind": "inconsistent", "n": 0, "core": [1, 3],
                        "plausible": False,
                        "plausibility_reasons": ["conflict_core_too_small"]},
        }
        row = _item_diagnostic("SYNTH-CORE", meta, False)
        self.assertEqual(row["solver_world_count"], 0)
        self.assertEqual(row["solver_classification"], "inconsistent")
        self.assertEqual(row["conflict_statement_indices"], [1, 3])
        self.assertEqual(row["conflict_core_size"], 2)
        self.assertEqual(row["solver_implausibility_reasons"], ["conflict_core_too_small"])

    def test_cli_failure_summary_is_not_truncated_and_accounts_for_unattempted_items(self):
        class RateLimited:
            calls = 0
            def complete(inner, item_id, messages, max_tokens):
                inner.calls += 1
                raise self._failure(429, {"error": {"code": "rate_limit"}})

        with tempfile.TemporaryDirectory() as d:
            item_path, answer_path, diag_path = [os.path.join(d, n) for n in
                ("items.json", "answers.json", "diagnostics.json")]
            subset = ITEMS[:12]
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(subset, f)
            client = RateLimited()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = main([item_path, "--budget", "1x", "--out", answer_path,
                           "--diagnostics", diag_path, "--stop-on-provider-failure"],
                          client=client)
            self.assertEqual(rc, 2)
            self.assertEqual(client.calls, 1)
            with open(answer_path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {})
            self.assertTrue(all(item["id"] in stderr.getvalue() for item in subset))
            with open(diag_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["item_count"], 12)
            self.assertEqual(report["attempted_model_calls"], 1)
            self.assertEqual(report["items"][0]["failure"], "http_429")
            self.assertTrue(all(row["calls"] == 0 for row in report["items"][1:]))

    def test_cli_prints_safe_http_diagnostic_without_response_text(self):
        class Failing:
            def complete(inner, item_id, messages, max_tokens):
                raise self._failure(429, {"error": {"message": "limited"}})
        with tempfile.TemporaryDirectory() as d:
            item_path = os.path.join(d, "items.json")
            output_path = os.path.join(d, "answers.json")
            diag_path = os.path.join(d, "diagnostics.json")
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump([ITEMS[0]], f)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = main([item_path, "--budget", "1x", "--out", output_path,
                           "--diagnostics", diag_path], client=Failing())
            self.assertEqual(rc, 2)
            with open(diag_path, encoding="utf-8") as f:
                diagnostic = f.read()
            self.assertIn('"http_status": 429', diagnostic)
            self.assertIn('"failure": "http_429"', diagnostic)
            self.assertNotIn("raw", stderr.getvalue())

    def test_empty_and_malformed_responses_do_not_become_valid_extractions(self):
        item = ITEMS[0]
        class Empty:
            def complete(self, item_id, messages, max_tokens):
                return ""
        ans, metas = run([item], 1, Config(), Empty(), workers=1, progress=False)
        self.assertEqual(ans, {})
        self.assertEqual(metas[0]["failure"], "empty_response_content")
        self.assertEqual(metas[0]["failure_stage"], "response_decoding")
        self.assertEqual(metas[0]["calls"], 1)

        class Malformed:
            def complete(self, item_id, messages, max_tokens):
                return "not a structured extraction"
        ans, metas = run([item], 1, Config(), Malformed(), workers=1, progress=False)
        self.assertEqual(ans, {})
        self.assertEqual(metas[0]["failure"], "incomplete_extraction")
        self.assertEqual(metas[0]["failure_stage"], "semantic_extraction")

    def test_invalid_shape_and_grounding_fail_at_distinct_stages(self):
        item = dict(ITEMS[0])
        header, _, _ = item["text"].partition("\n")
        h = parse_header(header)
        person, station = h.staff[0], h.stations[0]
        item["text"] = header + "\n[1] %s is on %s." % (person, station)

        class Output:
            def __init__(self, text):
                self.text = text
            def complete(self, item_id, messages, max_tokens):
                return self.text

        _, bad_shape = run([item], 1, Config(), Output("1|UNKNOWN|x"), workers=1, progress=False)
        self.assertEqual(bad_shape[0]["failure"], "extraction_validation_failure")
        self.assertEqual(bad_shape[0]["failure_stage"], "extraction_parsing")

        absent = next(p for p in h.staff if p != person)
        _, bad_grounding = run([item], 1, Config(),
                               Output("1|ON|%s|%s" % (absent, station)), workers=1, progress=False)
        self.assertEqual(bad_grounding[0]["failure"], "source_grounding_failure")
        self.assertEqual(bad_grounding[0]["failure_stage"], "source_grounding")

    def test_smoke_summary_never_includes_raw_response(self):
        summary = failure_summary({"id": "B2-000", "calls": 1, "failure": "incomplete_extraction",
            "raw": [{"text": "private response"}], "parses": []})
        self.assertNotIn("raw", summary)
        self.assertNotIn("private response", repr(summary))

    def test_opt_in_capture_is_owner_only_and_contains_only_completion_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "response.txt")
            meta = {"raw": [{"kind": "whole", "variant": 0, "text": "completion only"}]}
            self.assertTrue(write_protected_capture(path, meta))
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            with open(path, encoding="utf-8") as f:
                contents = f.read()
            self.assertEqual(contents, "completion only")
            self.assertNotIn("messages", contents)


if __name__ == "__main__":
    unittest.main()
