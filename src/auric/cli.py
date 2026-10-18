"""./run <items.json> --budget <1x|3x|10x> --out <answers.json>"""
import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .client import client_from_env, load_env
from .config import ABLATIONS, Config
from .pipeline import ItemRunFailure, answer_item


def _atomic_write_text(path, text, mode=0o600):
    """Atomically replace one explicitly requested output file.

    The temporary file is created beside the destination so os.replace stays
    on one filesystem. Existing output artifacts are never truncated before
    the complete replacement content has been written.
    """
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(absolute) + ".",
                                     suffix=".tmp", dir=directory)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, absolute)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path, value, mode=0o600):
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n", mode)


def _load_items(path, parser):
    try:
        with open(path, encoding="utf-8") as stream:
            items = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error("cannot read valid JSON input: %s" % type(exc).__name__)
    if not isinstance(items, list) or not items:
        parser.error("items input must be a non-empty JSON array")
    seen = set()
    for index, item in enumerate(items):
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not item["id"].strip() or not isinstance(item.get("text"), str)
                or not item["text"].strip()):
            parser.error("item %d must contain non-empty string id and text fields" % index)
        if item["id"] in seen:
            parser.error("duplicate item id at index %d" % index)
        seen.add(item["id"])
    return items


def _failure_summary(meta):
    """Safe runner diagnostic: no prompt, source text, operands, or raw response."""
    parses = []
    for p in meta.get("parses", []):
        parses.append({"source": p.get("source"),
                       "invalid_indices": sorted(p.get("invalid", {}).keys(), key=str),
                       "invalid_count": len(p.get("invalid", {}))})
    return {"item_id": meta.get("id"), "calls": meta.get("calls"),
            "failure": meta.get("failure"), "failure_stage": meta.get("failure_stage"),
            "diagnostic": meta.get("diagnostic"),
            "unresolved_line_indices": meta.get("unresolved_line_indices", []),
            "parses": parses}


def _item_diagnostic(item_id, meta, answer_serialized):
    """Operand-free per-item record suitable for persistent run diagnostics."""
    outcome = meta.get("outcome") or {}
    invalid_indices = sorted({str(i) for p in meta.get("parses", [])
                              for i in p.get("invalid", {})}, key=lambda x: int(x))
    relation_counts = {}
    for value in (meta.get("final") or {}).values():
        if isinstance(value, str):
            relation = value.split("|", 1)[0]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
    return {
        "item_id": item_id,
        "calls": meta.get("calls"),
        "model_responses_received": meta.get("responses_received"),
        "failure": meta.get("failure"),
        "failure_stage": meta.get("failure_stage"),
        "provider_diagnostic": meta.get("diagnostic"),
        "source_statement_count": meta.get("n_lines"),
        "parsed_statement_indices": sorted((meta.get("final") or {}).keys(), key=lambda x: int(x)),
        "invalid_statement_indices": invalid_indices,
        "unresolved_statement_indices": meta.get("unresolved_line_indices", []),
        "relation_counts": dict(sorted(relation_counts.items())),
        "solver_ran": bool(outcome),
        "solver_world_count": outcome.get("n"),
        "solver_classification": outcome.get("kind"),
        "solver_plausible": outcome.get("plausible"),
        "conflict_statement_indices": outcome.get("core", []),
        "conflict_core_size": len(outcome.get("core", [])) if outcome.get("kind") == "inconsistent" else None,
        "solver_implausibility_reasons": outcome.get("plausibility_reasons", []),
        "answer_serialized": bool(answer_serialized),
    }


def _write_diagnostics(path, items, metas, answers, complete, budget,
                       partial_answer_file_written=False):
    rows = []
    by_id = {m.get("id"): m for m in metas}
    for item in items:
        iid = item["id"]
        rows.append(_item_diagnostic(
            iid, by_id.get(iid, {}), iid in answers and
            (complete or partial_answer_file_written)))
    failures = [r for r in rows if r["failure"]]
    failure_counts = {}
    stage_counts = {}
    for row in failures:
        failure_counts[row["failure"]] = failure_counts.get(row["failure"], 0) + 1
        stage = row["failure_stage"] or "unspecified"
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    payload = {
        "complete": bool(complete),
        "budget_per_item": budget,
        "item_count": len(items),
        "attempted_model_calls": sum(m.get("calls") or 0 for m in metas),
        "answer_file_written": bool(complete or partial_answer_file_written),
        "answer_file_status": "complete" if complete else (
            "partial" if partial_answer_file_written else "not_written"),
        "partial_answer_file_written": bool(partial_answer_file_written and not complete),
        "missing_item_ids": [row["item_id"] for row in rows if not row["answer_serialized"]],
        "failed_item_count": len(failures),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "items": rows,
    }
    _atomic_write_json(path, payload)


def parse_budget(s):
    s = s.strip().lower()
    if s not in ("1x", "3x", "10x"):
        raise argparse.ArgumentTypeError("budget must be 1x, 3x or 10x")
    return int(s[:-1])


_PROVIDER_FAILURES = {"http_429", "http_error", "transport_error", "request_error"}


def run(items, budget, cfg, client, workers=4, progress=True,
        stop_on_provider_failure=False):
    """Answer every item. Returns (answers, metas) in item order."""
    def one(item):
        try:
            return answer_item(item, client, budget, cfg)
        except ItemRunFailure as e:
            return None, e.meta
        except Exception as e:                   # never lose an item to a bug
            return None, {"id": item.get("id"), "calls": None,
                          "failure": "internal_error", "error_type": type(e).__name__}
    results = []
    if stop_on_provider_failure:
        # Sequential execution is required for strict fail-stop semantics: with
        # queued/concurrent futures, additional provider calls can start after a 429.
        for k, item in enumerate(items, 1):
            r = one(item)
            results.append(r)
            if progress and (k % 10 == 0 or k == len(items)):
                print("  %d/%d items" % (k, len(items)), file=sys.stderr)
            if r[1].get("failure") in _PROVIDER_FAILURES:
                for pending in items[k:]:
                    results.append((None, {"id": pending.get("id"), "calls": 0,
                                           "failure": "not_attempted_after_provider_failure",
                                           "failure_stage": "not_attempted"}))
                break
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for k, r in enumerate(ex.map(one, items), 1):
                results.append(r)
                if progress and (k % 10 == 0 or k == len(items)):
                    print("  %d/%d items" % (k, len(items)), file=sys.stderr)
    answers = {it["id"]: r[0] for it, r in zip(items, results) if r[0] is not None}
    return answers, [r[1] for r in results]


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(prog="run", description="Auric R1 pipeline (Granite 4.2 8B + exact solver)")
    ap.add_argument("items")
    ap.add_argument("--budget", required=True, type=parse_budget)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial-out", default="",
                    help="deprecated additional copy of the partial answer map; --out is always written")
    ap.add_argument("--diagnostics", default="",
                    help="safe per-item diagnostics JSON (default: <out>.diagnostics.json)")
    ap.add_argument("--workers", type=int, default=4, help="items answered in parallel (default 4)")
    ap.add_argument("--stop-on-provider-failure", action="store_true",
                    help="run sequentially and do not start later items after an HTTP/transport failure")
    ap.add_argument("--ablate", default="", help="comma list of: " + ", ".join(ABLATIONS))
    ap.add_argument("--trace", default="", help="optional owner-only JSONL trace including raw model output")
    ap.add_argument("--env", default="", help="path to .env (default: ./.env or repo root)")
    a = ap.parse_args(argv)

    diagnostics_path = a.diagnostics or (a.out + ".diagnostics.json")
    paths = [a.out, diagnostics_path] + ([a.trace] if a.trace else [])
    if a.partial_out:
        paths.append(a.partial_out)
    absolute_paths = [os.path.abspath(path) for path in paths]
    if len(set(absolute_paths)) != len(absolute_paths):
        ap.error("--out, diagnostics, trace, and partial-output paths must be distinct")
    input_path = os.path.realpath(a.items)
    if any(os.path.realpath(path) == input_path for path in paths):
        ap.error("an output path must not replace the input items file")
    directories = [os.path.dirname(os.path.abspath(path)) or "." for path in paths]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            ap.error("cannot prepare output directory: %s" % type(exc).__name__)
    unsafe_destinations = [path for path in paths
                           if os.path.lexists(path)
                           and (os.path.islink(path) or not os.path.isfile(path))]
    if unsafe_destinations:
        ap.error("existing output destinations must be regular, non-symlink files")

    cfg = Config.from_names([x for x in a.ablate.split(",") if x])
    items = _load_items(a.items, ap)
    if client is None:
        load_env(a.env or None)
        client = client_from_env(cfg.timeout_s)
    answers, metas = run(items, a.budget, cfg, client, a.workers,
                         stop_on_provider_failure=a.stop_on_provider_failure)

    if a.trace:
        trace_text = "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas)
        _atomic_write_text(a.trace, trace_text)

    calls = [m.get("calls") for m in metas]
    item_ids = {it["id"] for it in items}
    bad = [m.get("id") for m in metas
           if m.get("failure") or m.get("calls") is None or not (1 <= m["calls"] <= a.budget)]
    missing = sorted(item_ids - set(answers))
    complete = not bad and not missing
    print("budget %dx: %d items, %d model calls (min %s / max %s per item)"
          % (a.budget, len(items), sum(c or 0 for c in calls), min(c or 0 for c in calls),
             max(c or 0 for c in calls)), file=sys.stderr)
    if bad or missing:
        _atomic_write_json(a.out, answers)
        partial_written = True
        if a.partial_out:
            _atomic_write_json(a.partial_out, answers)
            partial_written = True
        _write_diagnostics(diagnostics_path, items, metas, answers, False, a.budget,
                           partial_answer_file_written=partial_written)
        print("diagnostics: " + diagnostics_path, file=sys.stderr)
        print("PARTIAL answer map (missing item IDs are omitted): " + a.out,
              file=sys.stderr)
        if partial_written:
            if a.partial_out:
                print("additional partial answer copy: " + a.partial_out, file=sys.stderr)
        failure_counts = {}
        for m in metas:
            if m.get("failure"):
                key = m["failure"]
                failure_counts[key] = failure_counts.get(key, 0) + 1
        print("ERROR: run incomplete; failed items=%s missing answers=%s" % (bad, missing),
              file=sys.stderr)
        print("failure counts: " + json.dumps(failure_counts, sort_keys=True), file=sys.stderr)
        return 2
    _atomic_write_json(a.out, answers)
    _write_diagnostics(diagnostics_path, items, metas, answers, True, a.budget)
    print("diagnostics: " + diagnostics_path, file=sys.stderr)
    return 0
