# Take-Home — Architecture Under a Fixed, Weak Model

## The situation

You get a small open-weight model. It is not a good model. It is roughly the class of model we run in production, for reasons that are not negotiable: our systems are self-hosted, air-gapped, and we do not fine-tune.

On the 60 items in this package, the fixed weak model — **`ibm-granite/granite-4.2-8b`** — scored **0.0% macro exact match** across three one-shot runs without a harness. In two earlier one-shot runs on those same items, a frontier model (**Claude Sonnet 5**) scored **67.5%**. Both figures are measured averages, not promises about any single run.

**Your job is to close as much of that gap as you can, using architecture alone, with the weak model.** The frontier model is a point of comparison, not a model you may use to answer items. We care about the held-out result, the 1×/3×/10× curve, and what your ablations show—not a predetermined percentage.

"Completely right" means the whole item: the right kind, and the right content for that kind. There is no partial credit in the score.

## The task

Each item is a page of shift notes from a facility. Somewhere in the hedging, the padding and the restatements are statements constraining who is on which time block, and which station.

An item is one of three kinds, and you are not told which:

- **unique** — exactly one assignment is consistent with the notes.
- **ambiguous** — more than one is consistent. **Return all of them.** There are between two and four, and you are not told how many.
- **inconsistent** — none is consistent; the notes contradict themselves. Cite the statements that conflict.

**The three kinds are equally common.** This set is 20 / 20 / 20, and the held-out set is drawn to the same equal-thirds mix. We tell you because guessing the prior is not the skill being measured.

No domain knowledge is needed. Every item is decidable from its own text.

**Hedging carries full force.** "I'm fairly sure Priya is on calibration" states that Priya is on calibration, exactly as firmly as a sentence without the hedge. Hedges are texture, not doubt, and none is a signal to down-weight anything.

## What you submit

**1. A runnable system.**

```
./run <items.json> --budget <1x|3x|10x> --out <answers.json>
```

Provide a repo-root `run` launcher that accepts an items path, a budget, and an output path, and writes the answers JSON to that output path. The budget flag bounds **model calls per item** in both directions. `1x` is exactly one call per item; `3x` and `10x` are at least one and at most three or ten. Every item needs at least one model call — the task is to make the weak model do the work.

**We verify that the model's output actually affects your answers.**

`answers.json` maps each item id to one of three forms:

```json
{
  "EXAMPLE-000": {"case": "unique",
            "assignment": {"<name>": {"block": "11:00", "station": "inspection"},
                           "<name not on a station>": {"block": "13:00"}}},
  "EXAMPLE-001": {"case": "ambiguous",
            "assignments": [{"<name>": {"block": "...", "station": "..."}},
                            {"<name>": {"block": "...", "station": "..."}}]},
  "EXAMPLE-002": {"case": "inconsistent",
            "conflicts": ["<statement from the notes>", "..."]}
}
```

A bare assignment object with no `case` field is read as a `unique` claim. The ids above are illustrative; use the exact ids in `items.json`. See `example_answers.json` for the shape.

**2. An ablation table.** For each component of your system, the score with it and without it, at each budget. We re-run this.

**3. Two pages.** How you characterised the model before designing around it. Why each component exists. What you built and then removed, and why. Where your system breaks. What you would do with a month. Also: **state whether your constraint extraction is model-based or symbolic, and how your approach would hold up if the notes were written by different people rather than generated to a template.**

## What counts as correct

An item counts only if you get its kind right **and** the content for that kind right:

- **unique** — the assignment.
- **ambiguous** — *every* consistent assignment. Order does not matter; count does. A missing or extra one is wrong.
- **inconsistent** — the statements that actually conflict. Declaring the kind and citing nothing earns nothing, exactly as declaring `unique` and giving no assignment does.

**The citation rule, in one sentence: a citation is correct when the statements you cite cannot all hold at once, and dropping any one of them leaves a set that can.** An item usually has more than one such set and every one of them is correct — you are being asked to localise a contradiction, not to guess which statement we added. Citing every line fails the second half of the rule; citing one statement short fails the first. On this set every such set has at least three statements in it.

Quote each statement **in full and verbatim**, one citation per statement, and do not let a citation run on into the next line of the notes. Dropping a hedge from the front of a line is fine; trimming or paraphrasing is not.

Every assignment must name **every** person on that item's rota. Not everyone is on a station: the header says who is, and you give `"station"` only for those people. A station volunteered for someone the notes put on no station is ignored rather than penalised — but working out *which* station each station-holder is on is part of the item.

**Matching is exact string equality** on names, blocks and stations — case- and whitespace-sensitive. Echo the strings from the notes verbatim rather than tidying them.

Two diagnostics are reported beside your score and never ranked on: `ambiguous_partial` (you declared ambiguous and every assignment you gave is a real one, whether or not you gave them all) and `inconsistent_label_only` (you declared inconsistent but your citations did not meet the minimal-conflict rule). They show you which side of the cliff you are on.

## Rules

- **No frontier models in the answering pipeline.** Use them to help you think and write code; not to answer items. Your submission is re-run against the provided endpoint alone.
- **No fine-tuning.**
- **No per-item hand-written logic.** Solvers, libraries, frameworks, any tooling: all fine.
- **Send an `X-Item-Id` header on every model call**, carrying the id of the item the call is for. One line in your client setup. Our re-run puts a counting proxy in front of the endpoint and enforces the budget per item from that header. **Without it we cannot attribute calls to items, and the run fails the budget check** — so this is not optional.

## Auto-fail

These are checked before a submission is ranked. Any one of these prevents a valid score; we record the reason and independently check an apparent auto-fail before finalising it:

- **It does not run** on the documented entrypoint.
- **The output does not parse** as the format above.
- **The ablation does not reproduce** when we re-run it, beyond ordinary run-to-run variation.
- **The budget is violated in either direction** — over the cap, or an item answered with **no model call at all** — measured at the proxy.
- **Your answers do not change when the model's output changes** — a system that returns the same answers regardless of what the model says has not used it.

None of these are about how good your system is. They are about whether we can grade it at all, and each is cheap to check before sending.

## Running it

- **Pin your dependencies** (`requirements.txt` with versions, or a lockfile) and **state your Python version.**
- **One documented entrypoint**, exactly the command above.
- **Use `ibm-granite/granite-4.2-8b` with `temperature=1.0`, `top_p=0.95`, and reasoning/thinking disabled on every call.** With the OpenAI Python client pointed at the provided endpoint, pass `extra_body={"reasoning": {"enabled": False}}`. Sampling can vary between runs; report how many runs each of your numbers came from.
- The weak model is **open-weight.** You can use the same Granite 4.2 8B weights locally (Ollama, LM Studio, vLLM) while developing, with the same sampling settings and non-thinking mode. **Your submitted system must run through the provided endpoint** for our re-run.

## How it is graded

Your system is re-run as submitted on this set and on a fixed private held-out set generated from a different seed and a phrasing pool disjoint from this one's — the same kinds of item, with the constraint statements worded in forms that do not appear in `items.json`. We consider four dimensions, without a fine-grained composite score:

1. **Accuracy and curve shape** — at 1×, 3× and 10×. We grade the *shape*. **1× is weighted at least as heavily as 10×.** A system that only works when you throw calls at it is not what we are looking for.
2. **Case handling** — as a 3×3 matrix of declared kind against true kind, never a single number. The directions mean different things: over-declaring `ambiguous` is a system failing honestly, usually one repair loop from working, while a system that only ever says `unique` has no formal representation underneath at all.
3. **Ablation fidelity and reproducibility** — we re-run your ablation.
4. **The two pages**, read by a human, weighted heavily.

**The headline number is macro-averaged across the three kinds** — the mean of the three per-kind rates, each worth a third, whatever share of the set each occupies. A system that answers one kind well and has no representation of the other two cannot rise above 33%. Getting all three kinds right is the grade.

Your score is reported beside knowledge-free floors — constant-verdict arms that declare every item the same kind, and citation arms that cite lines at random. All of them score 0.0% on this set: a right kind with wrong content earns nothing.

## Expectations

**This is hard, and the frontier result is not the expected score.** We are looking at how you got the score you got: which architectural choices genuinely help the weak model, and where they fail on new wording.

A modest result with a sharp writeup and a strong 1× number is a better submission than a higher one assembled from extra passes. We mean this and we grade this way.

**Budget 6–8 hours.** Not a soft number — we would rather see six focused hours and an honest account of what you did not get to than a week of grinding.

If the obvious approach plateaus, that plateau is the interesting part. Tell us why it happens.

## Provided

- `items.json` — the visible set, 60 items
- `visible_key.json` — **the answers to the visible set.** Yes, really. Score yourself as often as you like. You are graded on sets you have never seen, so what these 60 items are worth is the understanding you take from them rather than the score you reach on them.
- `score.py` — the scorer. `python3 score.py visible_key.json <your answers>.json`
- `example_answers.json` — output format only
- `.env` — the base URL, the model id and an API key

**Use the key only for `ibm-granite/granite-4.2-8b`.** It has been tested with that model; our re-run enforces the model and call settings. Keep the key private and do not commit `.env` or the key to your submission repository.
