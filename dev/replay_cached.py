"""Offline fixed-completion replay for selected downstream ablations.

This tool reuses saved real endpoint completions; it makes no network calls.
It can measure downstream aggregation/validation/minimization changes only.
It cannot measure prompt variants or direct-answer/no-solver behavior because
those require completions generated from different prompts. Raw completions
are read in memory and are never copied to replay artifacts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from auric.config import Config  # noqa: E402
from auric.pipeline import ItemRunFailure, answer_item  # noqa: E402
from auric.cli import _write_diagnostics  # noqa: E402
from auric.client import ModelCallFailure  # noqa: E402


VARIANTS = {
    "full": (),
    "no_vote": ("no_vote",),
    "no_targeted": ("no_targeted",),
    "no_grounding": ("no_grounding",),
    "no_minimize": ("no_minimize",),
}


class ReplayInputError(RuntimeError):
    pass


class CachedCompletionClient:
    """Serve one item's original completions by call kind, preserving order."""

    def __init__(self, records):
        self.records = {kind: [r["text"] for r in records if r.get("kind") == kind]
                        for kind in ("whole", "targeted")}
        self.used = {"whole": 0, "targeted": 0}

    def complete(self, item_id, messages, max_tokens):
        user = messages[-1]["content"]
        kind = "targeted" if "MODE: TARGETED" in user else "whole"
        offset = self.used[kind]
        values = self.records[kind]
        if offset >= len(values):
            raise ModelCallFailure({"item_id": str(item_id), "stage": "offline_replay",
                                    "category": "cached_replay_missing",
                                    "exception_class": "ReplayInputError",
                                    "http_status": None})
        self.used[kind] += 1
        return values[offset]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_sha():
    digest = hashlib.sha256()
    source_dir = os.path.join(ROOT, "src", "auric")
    for filename in sorted(os.listdir(source_dir)):
        if filename.endswith(".py"):
            digest.update(filename.encode("utf-8"))
            with open(os.path.join(source_dir, filename), "rb") as stream:
                digest.update(stream.read())
    return digest.hexdigest()


def _load_scorer(path):
    spec = importlib.util.spec_from_file_location("auric_official_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def replay(items, trace_rows, key, scorer, out_dir, budget=10,
           trace_sha256="unspecified", input_sha256="unspecified"):
    if os.path.exists(out_dir):
        raise FileExistsError("replay output directory already exists")
    os.mkdir(out_dir, 0o700)
    os.chmod(out_dir, 0o700)
    trace_by_id = {row.get("id"): row for row in trace_rows}
    item_ids = [item.get("id") for item in items]
    if len(set(item_ids)) != len(item_ids) or set(trace_by_id) != set(item_ids):
        raise ReplayInputError("trace and input item IDs must match exactly and uniquely")
    summaries = []
    for variant, switches in VARIANTS.items():
        cfg = Config.from_names(switches)
        answers, metas = {}, []
        for item in items:
            records = trace_by_id[item["id"]].get("raw")
            if not isinstance(records, list) or any(
                    not isinstance(r, dict) or r.get("kind") not in ("whole", "targeted")
                    or not isinstance(r.get("text"), str) for r in records):
                raise ReplayInputError("trace contains missing or malformed completion records")
            try:
                answer, meta = answer_item(item, CachedCompletionClient(records), budget, cfg)
                answers[item["id"]] = answer
            except ItemRunFailure as exc:
                meta = exc.meta
            except ReplayInputError as exc:
                meta = {"id": item["id"], "calls": None,
                        "failure": "cached_replay_input_exhausted",
                        "failure_stage": "offline_replay", "error_type": type(exc).__name__}
            metas.append(meta)
        complete = len(answers) == len(items) and all(not m.get("failure") for m in metas)
        measurement_valid = not any(m.get("failure") == "cached_replay_missing" for m in metas)
        score_result = scorer.score(key, answers, items)
        stem = "10x_%s" % variant
        answer_path = os.path.join(out_dir, stem + (".answers.json" if complete else ".answers.partial.json"))
        diag_path = os.path.join(out_dir, stem + ".diagnostics.json")
        score_path = os.path.join(out_dir, stem + ".score.json")
        config_path = os.path.join(out_dir, stem + ".config.json")
        _atomic_json(answer_path, answers)
        _write_diagnostics(diag_path, items, metas, answers, complete, budget,
                           partial_answer_file_written=not complete)
        _atomic_json(score_path, score_result)
        config = {
            "experiment_type": "offline_fixed_completion_replay",
            "variant": variant,
            "ablations": list(switches),
            "budget_label": "%dx replay cap" % budget,
            "network_requests": 0,
            "source_sha256": _source_sha(),
            "trace_sha256": trace_sha256,
            "input_sha256": input_sha256,
            "item_count": len(items), "answer_count": len(answers),
            "complete": complete,
            "measurement_valid": measurement_valid,
            "score_is_granite_endpoint_evaluation": False,
            "score": score_result,
            "artifact_files": [os.path.basename(x) for x in
                               (answer_path, diag_path, score_path, config_path)],
        }
        _atomic_json(config_path, config)
        summaries.append({"variant": variant, "complete": complete,
                          "answer_count": len(answers),
                          "measurement_valid": measurement_valid,
                          "score": score_result if measurement_valid else None})
    _atomic_json(os.path.join(out_dir, "summary.json"), {
        "experiment_type": "offline_fixed_completion_replay",
        "network_requests": 0,
        "trace_item_count": len(items),
        "variants": summaries,
        "limitations": [
            "Replays saved real completions; does not measure endpoint behavior or prompt effects.",
            "Only downstream vote, targeted-call selection, grounding, and minimization toggles are compared.",
            "No-solver/direct-answer and prompt-only variants require newly generated completions and are not measured.",
        ],
    })
    return summaries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--trace", required=True, help="private saved JSONL trace with raw completions")
    ap.add_argument("--key", required=True, help="offline scoring key; never used by runtime answering")
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out-dir", required=True, help="new, non-existing directory outside deliverable")
    a = ap.parse_args(argv)
    items = json.load(open(a.items, encoding="utf-8"))
    with open(a.trace, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    key = json.load(open(a.key, encoding="utf-8"))
    scorer = _load_scorer(a.scorer)
    result = replay(items, rows, key, scorer, os.path.abspath(a.out_dir),
                    trace_sha256=_sha(a.trace), input_sha256=_sha(a.items))
    print(json.dumps([{"variant": r["variant"], "answers": r["answer_count"],
                       "measurement_valid": r["measurement_valid"],
                       "macro_exact_match": (r["score"].get("macro_exact_match")
                                              if r["score"] is not None else None),
                       "exact_match": (r["score"].get("exact_match")
                                       if r["score"] is not None else None)} for r in result], indent=2))


if __name__ == "__main__":
    main()
