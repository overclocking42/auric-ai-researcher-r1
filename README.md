# Auric R1 — Architecture Under a Fixed, Weak Model

## What this submission does

Auric R1 uses the required `ibm-granite/granite-4.2-8b` model to interpret
shift notes, strictly validates its extracted relations against the cited
note, then uses a deterministic exhaustive solver to classify and answer each
item. Extraction is semantic/model-based; production code does not use a
symbolic or regex note interpreter. The solver enumerates assignments,
returns every plausible ambiguous world, and computes minimal verbatim
conflict sets for inconsistent items.

The best saved visible-set result is a **partial 0.8667 macro exact match
(56/60)** from one independent 10x run. It is not a complete score and was
measured on an earlier prompt/source snapshot. The latest recorded 1x score
is 0.2500 (19/60); later 3x/10x results vary. The most recent attempts at
each budget stopped on the first request with HTTP 403
(`openrouter_key_limit`), so they yielded no scored answers. There is no
complete 60-item score, measured component ablation, or private held-out
result. See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) and
[ABLATION_TABLE.md](ABLATION_TABLE.md) for details.

## Requirements and setup

Python **3.10+** is supported; verification was performed with Python
3.13.7. `requirements.txt` pins the only external runtime dependency.
From the extracted project root—the directory containing `run`:

```sh
cd /path/to/auric-r1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally with the assignment-provided `OPENROUTER_API_KEY`.
`OPENROUTER_BASE_URL` is set in the example to
`https://openrouter.ai/api/v1`; `MODEL` is set to the required Granite model.
The client reads these names (it also accepts the OpenAI-compatible aliases
`OPENAI_API_KEY` and `OPENAI_BASE_URL`). Existing shell environment values
take precedence over `.env`. Never upload `.env` or share the key. The
submission archive contains placeholders/configuration only, no credentials,
visible key, scorer, or dataset.

## Run the required CLI

The one system entrypoint is `./run`. Run it from the extracted project root
and replace the input placeholder with the actual evaluator-supplied file
path; `/path/to/items.json` is explanatory text, not a file supplied by this
archive.

```sh
cd /path/to/auric-r1
mkdir -p runs
./run /absolute/path/to/items.json --budget 1x --out runs/answers-1x.json
./run /absolute/path/to/items.json --budget 3x --out runs/answers-3x.json
./run /absolute/path/to/items.json --budget 10x --out runs/answers-10x.json
```

Every attempted call consumes one per-item slot. The OpenAI client has
automatic retries disabled; 1x makes exactly one request per item, while 3x
and 10x allow at most three and ten. Every request uses temperature 1.0,
top-p 0.95, disabled reasoning, and `X-Item-Id` set to that request's actual
item ID. If a provider/transport error requires stopping, add
`--stop-on-provider-failure`; later items are then recorded as not attempted,
so that run is incomplete and is not a valid full evaluation.

`--out` is atomically replaced after a run completes. Malformed input is
rejected before model calls and before replacing output. On an item-level
failure, `--out` still receives parseable JSON containing only validated
answers; unresolved IDs are omitted, never guessed. Safe per-item details
go to `<out>.diagnostics.json`, and the CLI returns exit code 2. The supplied
scorer counts omissions as unparsed against the full dataset. Do not describe
a partial map as a complete evaluation.

When the evaluator supplies the scorer and key, score with:

```sh
python3 /absolute/path/to/score.py /absolute/path/to/visible_key.json \
  runs/answers-1x.json /absolute/path/to/items.json
```

The key is for offline scoring only and is never read by runtime code.

## Offline checks and ablations

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests dev
./run --help
python3 dev/run_experiments.py --help
python3 dev/preflight.py
```

The synthetic tests use constructed fixtures and a local mock endpoint;
they are not Granite accuracy measurements. The ablation harness runs the
full system and one component-removed variant per requested ablation, each
at the selected budgets, in independent artifacts. Example one-replicate
matrix (supply the evaluator's local files):

```sh
python3 dev/run_experiments.py \
  --items /absolute/path/to/items.json \
  --key /absolute/path/to/visible_key.json \
  --scorer /absolute/path/to/score.py \
  --dataset visible --runs 1 --budgets 1,3,10 \
  --ablations no_solver,no_guidance,no_grounding,no_vote,no_targeted,no_minimize,same_prompt,no_completeness
```

This full matrix is at most 7,560 calls for one replicate. Do not start it
without authorized endpoint capacity. Endpoint-based component ablations
remain unmeasured because of the provider limit. A bounded offline
fixed-completion replay was run from a saved 10x trace for downstream
`no_vote`, `no_targeted`, and `no_minimize` switches. It is not a new Granite
evaluation and does not measure prompt changes. See
`ABLATION_TABLE.md` and `EVALUATION_REPORT.md` for results and limitations.

To repeat the offline replay using private local artifacts (none are bundled):

```sh
python3 dev/replay_cached.py \
  --items /absolute/path/to/items.json \
  --trace /private/path/to/saved-10x.trace.jsonl \
  --key /absolute/path/to/visible_key.json \
  --scorer /absolute/path/to/score.py \
  --out-dir /private/tmp/auric-replay-new-run
```

The replay makes zero HTTP requests and never writes raw completions. It marks
a variant unmeasured if the cached trace lacks a required call. Prompt-only
and direct-answer/no-solver comparisons need new outputs from those prompts
and cannot be inferred from an extraction trace.

## Limitations

Granite frequently omits or malforms constraints, and strict grounding can
reject unsupported operands. Grounding confirms that an operand is present
in its cited note; it does not prove the semantic interpretation is right.
Synthetic tests vary names, labels, note order, and phrasing, but private
held-out generalization is **NOT RUN**. No local Granite runtime was
available. The latest provider diagnostics show the account key's total
limit was exhausted (HTTP 403); no further live calls were made. The real
ablation matrix and a complete full-dataset score remain outstanding.
