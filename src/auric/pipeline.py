"""Per-item pipeline: header -> Granite extraction (1..N calls) -> vote -> exact solve -> answer."""
import json
import re

from .client import Budgeted, ModelCallFailure
from .config import Config
from .extract import NONE, Parse, fmt, parse_output
from .header import parse_header, split_item
from .prompts import system_prompt, user_direct, user_targeted, user_whole
from .resolve import constraints, consensus, disputed, options_for, tally
from .solver import Rota, solve


class ItemRunFailure(RuntimeError):
    """An item could not be answered from a complete, plausible model extraction."""

    def __init__(self, message, meta):
        self.meta = meta
        super().__init__(message)


# -------------------------------------------------------------------------------------- answers
def _answer(rota, body, outcome, cfg, readings):
    if outcome.kind == "inconsistent":
        core = outcome.core
        if cfg.off("no_minimize"):
            core = tuple(i for i, _ in constraints(readings))
        return {"case": "inconsistent", "conflicts": [body[i] for i in sorted(core)]}
    worlds = outcome.worlds[:4] if outcome.n > 4 else outcome.worlds
    sols = [rota.assignment(w) for w in worlds]
    if outcome.n == 1:
        return {"case": "unique", "assignment": sols[0]}
    return {"case": "ambiguous", "assignments": sols}


# ---------------------------------------------------------------------------- direct (ablation)
def _direct(item, h, llm, cfg, ask=None):
    ask = ask or llm.ask
    raw = ask("You solve shift-rota puzzles and reply with JSON only.", user_direct(item["text"]))
    ans = None
    if raw:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                ans = json.loads(m.group(0))
            except Exception:
                ans = None
    if not isinstance(ans, dict) or ans.get("case") not in ("unique", "ambiguous", "inconsistent"):
        meta = {"id": item.get("id"), "mode": "direct", "calls": llm.used,
                "errors": llm.errors, "failure": "invalid_direct_answer", "raw": [raw]}
        raise ItemRunFailure("model did not return a supported direct answer", meta)
    return ans, {"mode": "direct", "calls": llm.used, "errors": llm.errors, "raw": [raw]}


# ------------------------------------------------------------------------------- main per-item
def answer_item(item, client, budget, cfg=None):
    """Returns (answer, meta). `budget` = max model calls for this item (1 for 1x means exactly 1)."""
    cfg = cfg or Config()
    iid = item["id"]
    llm = Budgeted(client, iid, budget, cfg.max_tokens)
    meta = {"id": iid, "budget": budget, "parses": [], "raw": [],
            "responses_received": 0}

    def ask(system, user):
        try:
            result = llm.ask(system, user)
            meta["responses_received"] = llm.responses_received
            return result
        except ModelCallFailure as e:
            meta.update(calls=llm.used, responses_received=llm.responses_received,
                        errors=llm.errors, failure=e.diagnostic["category"],
                        failure_stage=e.diagnostic["stage"], diagnostic=e.diagnostic)
            raise ItemRunFailure("model request failed", meta) from None

    h = rota = None
    body = []
    try:
        head, body = split_item(item)
        h = parse_header(head)
        rota = Rota(h)
    except Exception as e:                       # unusual header: still make the one required call
        meta["header_error"] = str(e)
        ans, m2 = _direct(item, h, llm, cfg, ask)
        meta.update(m2, calls=llm.used, errors=llm.errors)
        return ans, meta
    if cfg.off("no_solver"):
        ans, m2 = _direct(item, h, llm, cfg, ask)
        meta.update(m2, calls=llm.used, errors=llm.errors)
        return ans, meta

    n = len(body)
    parses = []

    def whole(variant):
        raw = ask(system_prompt(variant, cfg), user_whole(h, body, variant, iid, cfg))
        p = parse_output(raw, h, body, cfg, source="whole:%d" % variant)
        parses.append(p)
        meta["raw"].append({"kind": "whole", "variant": variant, "text": raw})
        return p

    whole(0)
    multi = budget > 1 and not cfg.off("no_vote")
    if multi:
        for v in (1, 2):
            if llm.left > 0:
                whole(v)

    def settle(extra_parses=(), extra_w=()):
        allp = parses + list(extra_parses)
        w = [1.0] * len(parses) + list(extra_w)
        votes, invalid = tally(allp, w)
        first = parses[0].readings if parses else {}
        readings, info = consensus(rota, n, votes, first)
        return votes, invalid, readings, info

    votes, invalid, readings, info = settle()
    targeted, tw = [], []
    if multi and budget >= 10 and not cfg.off("no_targeted"):
        dis = disputed(votes, invalid, n)
        batches = [dis[i:i + cfg.targeted_batch] for i in range(0, len(dis), cfg.targeted_batch)]
        for b in batches:
            if llm.left <= 0:
                break
            opts = {i: options_for(votes, i) for i in b}
            raw = ask(system_prompt(0, cfg), user_targeted(h, body, b, opts, iid, cfg))
            p = parse_output(raw, h, body, cfg, only=set(b), source="targeted")
            targeted.append(p)
            tw.append(cfg.targeted_weight)
            meta["raw"].append({"kind": "targeted", "lines": [i + 1 for i in b], "text": raw})
        votes, invalid, readings, info = settle(targeted, tw)
        # still implausible and budget left: add independent whole-item parses
        v = 3
        while llm.left > 0 and not solve(rota, constraints(readings)).plausible():
            whole(v)
            v += 1
            votes, invalid, readings, info = settle(targeted, tw)

    meta.update(
        mode="solver", calls=llm.used, errors=llm.errors, n_lines=n,
        parses=[{"source": p.source,
                 "readings": {str(i + 1): fmt(r) for i, r in sorted(p.readings.items())},
                 "invalid": {str(i + 1): why for i, why in sorted(p.invalid.items())}}
                for p in parses + targeted],
        final={str(i + 1): fmt(r) for i, r in sorted(readings.items())},
        ties=[i + 1 for i in info["ties"]], unresolved=[i + 1 for i in info["unresolved"]])
    if info["unresolved"]:
        reasons = [why for p in parses + targeted for why in p.invalid.values()]
        if "argument not in source line" in reasons:
            meta["failure"] = "source_grounding_failure"
            meta["failure_stage"] = "source_grounding"
        elif "model_uncertain" in reasons:
            meta["failure"] = "model_uncertain_extraction"
            meta["failure_stage"] = "semantic_extraction"
        elif reasons:
            meta["failure"] = "extraction_validation_failure"
            meta["failure_stage"] = "extraction_parsing"
        else:
            meta["failure"] = "incomplete_extraction"
            meta["failure_stage"] = "semantic_extraction"
        meta["unresolved_line_indices"] = [i + 1 for i in info["unresolved"]]
        raise ItemRunFailure("no valid extraction for one or more source lines", meta)

    outcome = solve(rota, constraints(readings), minimize=not cfg.off("no_minimize"))
    meta["outcome"] = {"kind": outcome.kind, "n": outcome.n,
                       "core": [i + 1 for i in outcome.core], "plausible": outcome.plausible(),
                       "plausibility_reasons": outcome.plausibility_reasons()}
    if not outcome.plausible():
        meta["failure"] = "implausible_solver_outcome"
        meta["failure_stage"] = "solver_plausibility"
        raise ItemRunFailure("solver outcome is outside the assignment's allowed range", meta)
    ans = _answer(rota, body, outcome, cfg, readings)
    return ans, meta
