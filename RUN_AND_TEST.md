# Run and test Auric R1

Python 3.10 or newer is required. Dependencies are pinned in
`requirements.txt` (verified with Python 3.13.7).

## Setup

```sh
cd /path/to/auric-r1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `MODEL` in the local
`.env`. Keep that file private and out of version control and archives.

## Offline verification

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests dev
./run --help
python3 dev/run_experiments.py --help
python3 dev/preflight.py
```

The tests use synthetic fixtures and a localhost mock endpoint; they do not
contact the model endpoint. Simulation output, if generated separately, must
be labeled as simulation, not Granite evaluation.

## Production runs

The required single entrypoint is `./run`:

```sh
cd /path/to/auric-r1
./run /absolute/path/to/items.json --budget 1x --out runs/answers-1x.json
./run /absolute/path/to/items.json --budget 3x --out runs/answers-3x.json
./run /absolute/path/to/items.json --budget 10x --out runs/answers-10x.json
```

Replace `/absolute/path/to/items.json` with the real supplied input file.
`./run` is relative to the current directory, so run these commands from the
extracted project root (the directory containing the executable launcher).

Output destinations are atomically replaced after the run. Use fresh names
when preserving earlier answers and diagnostics. The launcher rejects path
collisions and outputs that alias the input file. Malformed JSON or invalid
item records are rejected before model calls and before output replacement.
On a completed run, the requested `--out` is parseable JSON even when some
items fail: the map then contains only valid results, with unresolved IDs
omitted. A partial run returns exit code 2 and writes
`<out>.diagnostics.json`; omitted IDs count as unparsed when scored. Never
call a partial map a complete evaluation. Failed calls consume budget; there
are no automatic retries or fallback models. Every request includes the
actual item ID in `X-Item-Id` and the required Granite sampling settings.

## Score with evaluator-supplied files

The scorer and key are supplied separately and intentionally are not in the
submission archive:

```sh
python3 /path/to/score.py /path/to/visible_key.json \
  /path/to/answers-1x.json /path/to/items.json
```

The key is for offline scoring only, never runtime answer generation.

## One-item smoke test

Only run when the endpoint is available and a request is authorized. This
performs one real model call for the first item in the supplied input:

```sh
python3 dev/smoke_test.py --items /path/to/items.json
```

Optional raw response capture is sensitive and should be used only in a
private owner-only location:

```sh
mkdir -m 700 /private/tmp/auric-capture
python3 dev/smoke_test.py --items /path/to/items.json \
  --capture-response /private/tmp/auric-capture/completion.txt
```

## Ablations

Each variant is run separately from the full system. The visible key and
scorer must be explicitly supplied for offline scoring; no evaluator data is
bundled:

```sh
python3 dev/run_experiments.py \
  --items /path/to/items.json --key /path/to/visible_key.json \
  --scorer /path/to/score.py --dataset visible --runs 1 --budgets 1,3,10 \
  --ablations no_solver,no_guidance,no_grounding,no_vote,no_targeted,no_minimize,same_prompt,no_completeness
```

This matrix’s maximum is 7,560 calls for one replicate. Each run writes
separate answer/partial-answer, score, safe diagnostics, and configuration
artifacts to a new private directory. The harness stops after a
provider/transport failure by default. Do not bypass provider limits.

For a zero-network fixed-completion replay of downstream components, use
private saved traces:

```sh
python3 dev/replay_cached.py --items /absolute/path/to/items.json \
  --trace /private/path/to/saved.trace.jsonl \
  --key /absolute/path/to/visible_key.json \
  --scorer /absolute/path/to/score.py \
  --out-dir /private/tmp/auric-replay-new-run
```

This is not an endpoint experiment. Prompt changes and direct-answer/no-solver
variants cannot be tested from extraction completions.

## Packaging

`python3 dev/make_zip.py --out /path/to/new-unique-name.zip` builds the
allowlisted package. The ZIP excludes `.env`, datasets, scorer, keys, caches,
run artifacts, and previous ZIP files. Extract it to a fresh directory and
run the offline verification commands above.
