# Submission requirement checklist

Statuses reflect inspected code and saved evidence, not intended behavior.

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Exact `./run <items.json> --budget <1x|3x|10x> --out <answers.json>` entrypoint | COMPLETE | Launcher and CLI help verified; run from the directory containing `run`. |
| Required Granite model and sampling/reasoning settings | COMPLETE | Client fixes model, temperature 1.0, top-p 0.95, and reasoning disabled; local mock-wire test checks the request. |
| Correct `X-Item-Id` on every model call | COMPLETE | Client passes the current item ID; mock-wire test checks item attribution. |
| 1x exact call count; 3x/10x cap; failures consume slots; no hidden retry | COMPLETE | Per-item budget gate, OpenAI `max_retries=0`, and offline budget/error tests. |
| Model output genuinely affects answers | COMPLETE | Synthetic regression changes model output and verifies answer changes. |
| Output schema, full ambiguity, minimal verbatim conflicts | COMPLETE (offline) | Synthetic solver/serializer tests; no complete live 60-item run. |
| Input path, partial output, exit status, diagnostics | COMPLETE | CLI validates input, atomically writes parseable partial maps, omits unresolved IDs, emits per-item diagnostics, and returns 2 when incomplete. |
| Pinned dependencies, documented Python version, setup | COMPLETE | `openai==1.109.1`; README documents Python 3.10+ (verified with 3.13.7). |
| `.env` template contains no credential | COMPLETE | Template has an empty key field, documented endpoint, and required model. Local `.env` excluded from archive. |
| Required ablation table | INCOMPLETE | A fixed-completion replay measured selected downstream 10x switches; endpoint matrix, prompt changes, 1x/3x and no-solver remain NOT MEASURED. |
| Required 1x/3x/10x scores | INCOMPLETE | Multiple partial scored runs exist; no 60/60 valid answer map. Latest per-budget attempts each stopped on HTTP 403 after one B2-000 call. |
| Exact source revision for all historical scores | INCOMPLETE | Older run artifacts lack source hashes; labels and uncertainty are disclosed in the report. |
| Held-out Granite generalization | NOT RUN | Private held-out set unavailable. Synthetic changed-name/label/order/phrasing tests are not held-out evidence. |
| Local inference with the same Granite weights | NOT RUN | No local Granite runtime/model was available; no model was downloaded. |
| Clean-room archive tests | COMPLETE | The allowlisted package was extracted with macOS `unzip`, installed into a fresh venv, and passed the final offline tests, compilation, CLI help, experiment help, and preflight. The final ZIP is independently rechecked after creation. |

## Measured run record

The best saved partial score is 0.8667 (56/60) from an independent 10x run;
it is not a complete score and is from an earlier prompt/source snapshot.
All scored runs and category counts are tabulated in
`EVALUATION_REPORT.md`. The latest attempted 1x, 3x, and 10x commands used
one POST each for B2-000 and stopped with HTTP 403, provider limit source
`openrouter_key_limit`; their 59 remaining items were not attempted and
those runs were not scored.

## Endpoint and archive policy

No further endpoint calls are authorized while the latest key-limit error
remains unresolved. The final archive must exclude `.env`, answer keys,
visible input/scorer files, runs, private traces, caches, virtual
environments, and older ZIPs. The archive must preserve the executable bit
on `run` and include source, tests, setup instructions, this checklist, the
ablation table, and the technical report.
