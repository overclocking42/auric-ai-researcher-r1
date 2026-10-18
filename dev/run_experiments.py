"""Run real-endpoint experiments with one private artifact set per replicate.

Example (supply the assignment-provided files locally; they are not bundled):
  python3 dev/run_experiments.py --items /path/items.json --key /path/visible_key.json \
    --scorer /path/score.py --dataset visible --runs 1 --budgets 1

Each invocation creates a fresh mode-0700 directory under runs/ unless
--out-dir is given. Per-replicate answer maps, official scorer output,
operand-free diagnostics, and configuration metadata are mode 0600. Raw
model completions are deliberately not persisted by this harness.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from auric.cli import _write_diagnostics, run  # noqa: E402
from auric.client import client_from_env, load_env  # noqa: E402
from auric.config import ABLATIONS, Config  # noqa: E402

MODEL = "ibm-granite/granite-4.2-8b"
PROVIDER_FAILURES = {"http_429", "http_error", "transport_error", "request_error"}


def load_score(path):
    spec = importlib.util.spec_from_file_location("score", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_exclusive(path, payload):
    """Write JSON without overwriting an existing evidence file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def create_experiment_dir(runs_dir, dataset, out_dir=None, now=None):
    """Create a unique private output directory; never reuse prior evidence."""
    if out_dir:
        path = os.path.abspath(out_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.mkdir(path, 0o700)
    else:
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        root = os.path.join(runs_dir, "experiment_%s_%s" % (dataset, stamp))
        path = root
        suffix = 1
        while True:
            try:
                os.mkdir(path, 0o700)
                break
            except FileExistsError:
                path = "%s_%02d" % (root, suffix)
                suffix += 1
    os.chmod(path, 0o700)
    return path


def _run_name(dataset, config, budget, replicate):
    return "%s_%s_%dx_run-%02d" % (dataset, config, budget, replicate)


def persist_replicate(items, answers, metas, score, config_record, out_dir):
    """Persist one replicate's answer, score, config, and safe diagnostics."""
    complete = bool(config_record["complete"])
    base = _run_name(config_record["dataset"], config_record["variant"],
                     config_record["budget_per_item"], config_record["replicate"])
    paths = {
        "answers": os.path.join(out_dir, base + (".answers.json" if complete else ".answers.partial.json")),
        "score": os.path.join(out_dir, base + ".score.json"),
        "diagnostics": os.path.join(out_dir, base + ".diagnostics.json"),
        "config": os.path.join(out_dir, base + ".config.json"),
    }
    _json_exclusive(paths["answers"], answers)
    _json_exclusive(paths["score"], score)
    _write_diagnostics(paths["diagnostics"], items, metas, answers, complete,
                       config_record["budget_per_item"],
                       partial_answer_file_written=not complete)
    config_record["score"] = score
    config_record["artifact_files"] = {key: os.path.basename(value)
                                       for key, value in paths.items()}
    _json_exclusive(paths["config"], config_record)
    return paths


def _run_one(items, key, scorer, client, config_name, ablations, budget,
             replicate, out_dir, dataset, input_sha256, source_sha256, workers,
             stop_on_provider_failure):
    started = datetime.now(timezone.utc)
    start = time.monotonic()
    cfg = Config.from_names(ablations)
    answers, metas = run(items, budget, cfg, client, workers, progress=False,
                         stop_on_provider_failure=stop_on_provider_failure)
    elapsed = time.monotonic() - start
    result = scorer.score(key, answers, items)
    calls_by_item = [m.get("calls") for m in metas]
    calls = sum(n or 0 for n in calls_by_item)
    complete = len(answers) == len(items) and all(
        m.get("failure") is None and isinstance(m.get("calls"), int)
        and 1 <= m["calls"] <= budget for m in metas
    )
    ended = datetime.now(timezone.utc)
    failure_counts = {}
    for meta in metas:
        failure = meta.get("failure")
        if failure:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    config_record = {
        "dataset": dataset,
        "variant": config_name,
        "ablations": list(ablations),
        "budget_per_item": budget,
        "replicate": replicate,
        "model": MODEL,
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_enabled": False,
        "automatic_retries": 0,
        "item_count": len(items),
        "input_sha256": input_sha256,
        "source_sha256": source_sha256,
        "answer_count": len(answers),
        "complete": complete,
        "calls": calls,
        "calls_per_item_min": min((n or 0 for n in calls_by_item), default=0),
        "calls_per_item_max": max((n or 0 for n in calls_by_item), default=0),
        "failure_counts": dict(sorted(failure_counts.items())),
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
    }
    persist_replicate(items, answers, metas, result, config_record, out_dir)
    provider_failed = any(m.get("failure") in PROVIDER_FAILURES for m in metas)
    return config_record, provider_failed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="assignment input JSON (path supplied by evaluator)")
    ap.add_argument("--key", required=True, help="offline scoring key; read only for scoring")
    ap.add_argument("--scorer", required=True, help="official scorer Python file")
    ap.add_argument("--dataset", choices=["visible", "heldout"], default="visible")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--budgets", default="1,3,10")
    ap.add_argument("--ablations", default="", help="comma list; each is run alone (plus the full system)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", default="", help="new, non-existing private output directory")
    ap.add_argument("--continue-after-provider-failure", action="store_true",
                    help="do not stop the experiment after a provider/transport failure")
    a = ap.parse_args(argv)
    if a.runs < 1:
        ap.error("--runs must be at least 1")
    try:
        budgets = [int(value) for value in a.budgets.split(",")]
    except ValueError:
        ap.error("--budgets must be a comma-separated subset of 1,3,10")
    if not budgets or any(value not in (1, 3, 10) for value in budgets) or len(set(budgets)) != len(budgets):
        ap.error("--budgets must be a unique, non-empty subset of 1,3,10")
    ablations = [value for value in a.ablations.split(",") if value]
    unknown = sorted(set(ablations) - set(ABLATIONS))
    if unknown:
        ap.error("unknown ablation(s): " + ", ".join(unknown))

    input_path = os.path.abspath(a.items)
    key_path = os.path.abspath(a.key)
    scorer_path = os.path.abspath(a.scorer)
    for path, label in ((input_path, "items"), (key_path, "key"), (scorer_path, "scorer")):
        if not os.path.isfile(path):
            ap.error("%s file does not exist: %s" % (label, path))
    with open(input_path, encoding="utf-8") as stream:
        items = json.load(stream)
    with open(key_path, encoding="utf-8") as stream:
        key = json.load(stream)
    scorer = load_score(scorer_path)
    load_env()
    client = client_from_env()
    runs_dir = os.path.join(ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_dir = create_experiment_dir(runs_dir, a.dataset, a.out_dir)
    with open(input_path, "rb") as stream:
        input_sha256 = hashlib.sha256(stream.read()).hexdigest()
    digest = hashlib.sha256()
    source_dir = os.path.join(ROOT, "src", "auric")
    for filename in sorted(os.listdir(source_dir)):
        if filename.endswith(".py"):
            path = os.path.join(source_dir, filename)
            digest.update(filename.encode("utf-8"))
            with open(path, "rb") as stream:
                digest.update(stream.read())
    source_sha256 = digest.hexdigest()
    configs = [("full", [])] + [(name, [name]) for name in ablations]
    summary = {
        "dataset": a.dataset,
        "item_count": len(items),
        "budgets": budgets,
        "replicates_requested": a.runs,
        "configs": [name for name, _ in configs],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stopped_after_provider_failure": False,
        "runs": [],
    }
    stop = False
    for config_name, config_ablations in configs:
        for budget in budgets:
            for replicate in range(1, a.runs + 1):
                record, provider_failed = _run_one(
                    items, key, scorer, client, config_name, config_ablations,
                    budget, replicate, out_dir, a.dataset, input_sha256,
                    source_sha256, a.workers,
                    stop_on_provider_failure=not a.continue_after_provider_failure,
                )
                summary["runs"].append(record)
                print("%-14s %2dx run %d/%d: macro %.4f, %d/%d answers, %d calls" % (
                    config_name, budget, replicate, a.runs,
                    record["score"]["macro_exact_match"], record["answer_count"],
                    record["item_count"], record["calls"]), file=sys.stderr)
                if provider_failed and not a.continue_after_provider_failure:
                    summary["stopped_after_provider_failure"] = True
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    summary_path = os.path.join(out_dir, "summary.json")
    _json_exclusive(summary_path, summary)
    print("experiment artifacts: " + out_dir, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
