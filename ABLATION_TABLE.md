# Ablation score table

The brief requires scores with and without each component at 1x, 3x, and
10x. No component-removed endpoint run has completed. The saved 10x trace
does support a controlled, zero-network fixed-completion replay for a subset
of downstream components. Replay figures are separated below; they are not
new Granite endpoint evaluations and do not measure prompt effects.

| Component removed | Baseline 1x | Without, 1x | Baseline 3x | Without, 3x | Baseline 10x | Without, 10x |
|---|---:|---|---:|---|---:|---|
| Solver (`no_solver`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Prompt guidance (`no_guidance`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Source grounding (`no_grounding`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Multi-sample vote (`no_vote`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Targeted rereads (`no_targeted`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Conflict minimization (`no_minimize`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Prompt/order variation (`same_prompt`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |
| Completeness self-check (`no_completeness`) | 0.2500 partial, 19/60 | NOT MEASURED | 0.4667 partial, 33/60 | NOT MEASURED | 0.7333 partial, 51/60 | NOT MEASURED |

## Fixed-completion offline replay (10x trace)

The trace `runs/a10-independent3-2026-10-18.trace.jsonl` contains 318 saved
completions across all 60 items. The replay applied the current pipeline to
those fixed completions and scored each resulting answer map with the
supplied scorer. Raw completions were kept out of replay outputs. The full
replay produced 56/60 and 0.8667 (unique 15/20, ambiguous 17/20, inconsistent
20/20); this matches the previously reported best independent endpoint score
but is not a second independent evaluation.

| Replay variant | Cached completions used | Valid answers | Macro exact match | Unique / Ambiguous / Inconsistent |
|---|---:|---:|---:|---:|
| Full pipeline | 318 | 56/60 | 0.8667 | 0.75 / 0.85 / 1.00 |
| Without multi-sample vote (`no_vote`) | 60 | 30/60 | 0.3833 | 0.35 / 0.30 / 0.50 |
| Without targeted rereads (`no_targeted`) | 180 | 29/60 | 0.4500 | 0.40 / 0.30 / 0.65 |
| Without conflict minimization (`no_minimize`) | 318 | 56/60 | 0.5333 | 0.75 / 0.85 / 0.00 |
| Without source grounding (`no_grounding`) | cache short by 3 calls | 50/60 provisional | NOT MEASURED | invalid replay; scorer output diagnostic only |

These fixed-response downstream ablations are not causal estimates of how a
different live prompt would affect Granite. The no-grounding variant changed
the subsequent request schedule and exhausted the available per-item cache
for three items, so its score is excluded. Solver removal, guidance,
same-prompt, and completeness-prompt variants require completions generated
under their changed prompts and remain **NOT MEASURED**. No 1x or 3x ablation
was measured.

Baseline columns are reference measurements only, not matched controls. The
1x score is the latest recorded 1x result, the 3x score is a current-source
replicate, and the 10x score is from the completeness-prompt run. The best
independent 10x run scored 0.8667 partial (56/60) on an earlier prompt/source
snapshot; it is reported separately in `EVALUATION_REPORT.md` and is not
substituted into this illustrative baseline column.

Reproduction command (use the evaluator-supplied local input/key/scorer):

```sh
python3 dev/run_experiments.py \
  --items /absolute/path/to/items.json \
  --key /absolute/path/to/visible_key.json \
  --scorer /absolute/path/to/score.py \
  --dataset visible --runs 1 --budgets 1,3,10 \
  --ablations no_solver,no_guidance,no_grounding,no_vote,no_targeted,no_minimize,same_prompt,no_completeness
```

This schedules baseline plus eight one-component variants, separately at
each budget. The maximum is 7,560 calls for one replicate. The latest
provider response was HTTP 403 with `openrouter_key_limit`; the matrix was
not run. Do not retry until the assignment endpoint owner restores authorized
capacity. Ablation-only switches do not weaken the production configuration.
