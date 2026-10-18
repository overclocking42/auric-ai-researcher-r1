# Auric R1 — Technical report and evaluation

## Model characterization and design

The assignment brief characterizes `ibm-granite/granite-4.2-8b` as a weak,
fixed model and reports a 0% no-harness average over three one-shot runs; it
also reports a 67.5% frontier comparison over two runs. Those figures come
from the brief and were not reproduced here. Candidate runs show that the
model can return useful structured extractions, but omissions, malformed
records, explicit uncertainty, and unsupported operands remain frequent.
Results vary substantially between stochastic runs; the available evidence
does not isolate a single prompt component as the cause of that variation.

Extraction is **model-based**, not a symbolic or regex interpretation of
notes. A deterministic header parser reads the rota facts. Granite receives
the full numbered notes and emits one typed relation per note. Strict
validation checks relation shape, operand types, entity membership, complete
and unique statement indices, and grounding in the cited note itself. The
omission audit only flags relation-like cues; it never creates a constraint.

The solver then exhaustively evaluates the 720 assignment worlds in the
standard five-person/three-station case. It implements temporal order,
immediate adjacency, symmetric-endpoint strict `BETWEEN`, station relations,
and exclusions. Unique cases serialize one assignment; ambiguous cases
serialize every consistent world; inconsistent cases cite deletion-minimal
sets of complete original statements. A plausibility guard records and
rejects outcomes outside the assignment's specified ambiguity/conflict
ranges rather than guessing. The solver audit found no confirmed defect.
For repeated budgets, independently validated parses are aggregated and
ties remain unresolved. Every attempted request consumes budget; the client
sets `max_retries=0` and sends the required Granite model, temperature 1.0,
top-p 0.95, disabled reasoning, and the actual `X-Item-Id`.

## Visible-set results

Each scored row below is a distinct saved answer map. Partial scores use the
official scorer's full 60-item denominator; omitted items are unparsed. Runs
are not combined. Older artifacts do not retain a source hash, so their exact
code revision cannot be independently reconstructed. Labels identify the
run's contemporaneous prompt/source family, not a verified commit.

| Run label | Budget / calls | Valid answers | Macro exact | Unique | Ambiguous | Inconsistent | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Latest recorded 1x score | 1x / 60 | 19/60 | 0.2500 | 4/20 | 4/20 | 7/20 | Partial |
| Earlier 3x | 3x / 180 | 52/60 | 0.7333 | 13/20 | 14/20 | 17/20 | Partial; older source |
| Current-source 3x replicate | 3x / 180 | 33/60 | 0.4667 | 6/20 | 10/20 | 12/20 | Partial |
| 3x stopped on provider limit | 3x / 12 | 1/60 | 0.0167 | 0/20 | 1/20 | 0/20 | Partial; HTTP 402 |
| Earlier 10x | 10x / 315 | 56/60 | 0.8167 | 13/20 | 19/20 | 17/20 | Partial |
| 10x continuation series | 10x / 318 | 56/60 | 0.8167 | 14/20 | 17/20 | 18/20 | Partial; continued run |
| Best independent 10x observed | 10x / 319 | 56/60 | **0.8667** | 15/20 | 17/20 | 20/20 | Partial |
| Completeness-prompt 10x | 10x / 305 | 51/60 | 0.7333 | 13/20 | 16/20 | 15/20 | Partial |

After those scored experiments, three separately named command-line runs
(1x, 3x, and 10x) each attempted only `B2-000`, received HTTP 403 with
`limit_source=openrouter_key_limit`, and stopped before the remaining 59
items. They each used one request and produced an empty partial answer map;
they were not scored. The recorded provider diagnostic indicated the key's
total usage limit had been reached and did not reset. No live calls were made
after those failures in this final review. The candidate has no complete
60/60 official score.

The latest blocked attempts ran against source digest
`9f7bbb8fa434f60db15902b875484e6e46d4d1f7b41494f0303d9ccfb52ac35b`
(SHA-256 over the sorted `src/auric/*.py` filenames and contents). That
revision predates the final diagnostic-redaction-only change in this package.
Historical score artifacts do not record comparable source digests.

## Ablations and generalization

The live harness defines one-at-a-time variants for solver, prompt guidance,
grounding, voting, targeted rereads, conflict minimization, prompt/order
variation, and the completeness self-check. The assignment asks for each
component with and without it at 1x, 3x, and 10x. The endpoint matrix command
is in `README.md`; one replicate of full-plus-eight-variant runs has a maximum
of 7,560 calls. The provider limit prevented that matrix. A limited offline
replay was feasible from saved real completions:

| Fixed-completion replay variant | Scored answers | Macro exact | Unique | Ambiguous | Inconsistent | Replay calls |
|---|---:|---:|---:|---:|---:|---:|
| Full pipeline | 56/60 | 0.8667 | 0.75 | 0.85 | 1.00 | 318 |
| no_vote | 30/60 | 0.3833 | 0.35 | 0.30 | 0.50 | 60 |
| no_targeted | 29/60 | 0.4500 | 0.40 | 0.30 | 0.65 | 180 |
| no_minimize | 56/60 | 0.5333 | 0.75 | 0.85 | 0.00 | 318 |
| no_grounding | invalid/incomplete cache | NOT MEASURED | — | — | — | 321 attempts; 3 cache misses |

The trace is `runs/a10-independent3-2026-10-18.trace.jsonl`. It has 318
completion records; some historical run summaries report 319 total attempts,
so this replay uses only the retained 318 completions and is not a new
independent endpoint run. The replay runs the official scorer on each valid
resulting map, but prompt-dependent effects are not measured. no_solver,
no_guidance, same_prompt, no_completeness, 1x/3x ablations, and no_grounding
remain NOT MEASURED. no_grounding exhausted the cache for B2-001, B2-009, and
B2-029; its diagnostic scorer output is not reported as an ablation result.
See `ABLATION_TABLE.md` and `README.md` for details and commands.

Offline synthetic tests vary names, punctuation, station names, block
labels, note order, hedges, irrelevant padding, and header size. They check
constructed solver results and request/output invariants; they are not
Granite measurements. No local Granite runtime was available. The private
held-out phrasing set was not available, so held-out performance is **NOT
RUN**. Different authors may express relations using unfamiliar idioms;
grounding cannot guarantee semantic interpretation, and implicit cross-note
references remain a limitation.

The largest measured weakness is extraction reliability, not a confirmed
solver failure. A month of additional work would start with protected replay
of representative failures, then pre-registered paired prompt/component
experiments at 1x before spending calls at higher budgets. It would reserve
a disjoint phrasing set for generalization and report run-to-run uncertainty.
No result here supports a claim that the candidate beats the brief's
frontier reference.

## Verification and limits

Synthetic unit tests cover the 720-world enumeration, all supported relation
families, strict grounding, ambiguity enumeration, minimal conflicts,
model-dependence, call accounting, partial output, and HTTP request
configuration against a local mock endpoint. Those tests do not establish
real endpoint availability or private-set accuracy. The CLI writes a
parseable answer map and safe per-item diagnostics even for partial runs,
returns exit code 2 when incomplete, and omits unresolved IDs rather than
inventing an answer. The official scorer counts omissions against the full
denominator.

Endpoint ablation results beyond the limited fixed-completion replay and
held-out Granite results remain **NOT MEASURED**.
The provider's latest recorded key-limit response blocks further endpoint
experiments until authorized capacity is restored. See `SUBMISSION_CHECKLIST.md`
for requirement status and the precise incomplete items.

At final review, **67 offline tests passed** with the candidate virtual
environment. Compilation, launcher help, experiment-harness help, and
preflight are verified separately for the packaged copy. The final local
changes add an offline cached-completion replay harness and redact opaque
provider key-management URL identifiers from sanitized diagnostics; neither
changes production extraction or answers.
