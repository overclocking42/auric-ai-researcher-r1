"""Parse and validate the model's compact output into constraints."""
import re
from dataclasses import dataclass, field

from .solver import canon

NONE = "NONE"
_ALIASES = {
    "AFTER": ("BEFORE", True), "RIGHT_AFTER": ("RIGHT_BEFORE", True),
    "IMMEDIATELY_AFTER": ("RIGHT_BEFORE", True), "IMMEDIATELY_BEFORE": ("RIGHT_BEFORE", False),
    "DIRECTLY_BEFORE": ("RIGHT_BEFORE", False), "DIRECTLY_AFTER": ("RIGHT_BEFORE", True),
    "NEXT": ("RIGHT_BEFORE", False),
}
_ARITY = {"AT": 2, "NOT_AT": 2, "ON": 2, "NOT_ON": 2, "BEFORE": 2, "RIGHT_BEFORE": 2, "BETWEEN": 3}
_LINE = re.compile(r"^\s*[-*•]?\s*\[?(?:L|line\s*)?(\d+)\]?\s*[|:.)]\s*(.*?)\s*$", re.I)


@dataclass
class Parse:
    readings: dict = field(default_factory=dict)    # line idx0 -> constraint tuple | NONE
    invalid: dict = field(default_factory=dict)     # line idx0 -> reason (rejected output)
    raw: str = ""
    source: str = ""                                # label: "whole:0", "targeted", ...


def fmt(c):
    """Human/compact form of a reading (also used as option text in targeted prompts)."""
    if c == NONE:
        return NONE
    def r(x):
        return x if isinstance(x, str) else ("@" + x[1] if x[0] == "h" else x[1])
    return "|".join([c[0]] + [r(a) for a in c[1:]])


def _clean(tok):
    t = tok.strip().strip("\"'`*").strip()
    t = re.sub(r"['’]s$", "", t)
    return t.rstrip(".,;:").strip()


def _find(options, tok):
    low = tok.casefold()
    for o in options:
        if o.casefold() == low:
            return o
    return None


def _time(blocks, tok):
    t = _find(blocks, tok)
    if t:
        return t
    m = re.fullmatch(r"(\d{1,2}):?(\d\d)?", tok)
    if m:
        cand = "%02d:%s" % (int(m.group(1)), m.group(2) or "00")
        return _find(blocks, cand)
    return None


def _ref(h, tok):
    if tok.startswith("@"):
        s = _find(h.stations, _clean(tok[1:]))
        return ("h", s) if s else None
    p = _find(h.staff, tok)
    if p:
        return ("p", p)
    s = _find(h.stations, tok)
    return ("h", s) if s else None


def _build(h, rel, args):
    """Turn tokens into a canonical constraint or raise ValueError(reason)."""
    swap = False
    if rel in _ALIASES:
        rel, swap = _ALIASES[rel]
    if rel not in _ARITY:
        raise ValueError("unknown relation %r" % rel)
    if len(args) != _ARITY[rel]:
        raise ValueError("%s needs %d arguments" % (rel, _ARITY[rel]))
    if rel in ("AT", "NOT_AT"):
        p, t = _find(h.staff, args[0]), _time(h.blocks, args[1])
        if not p or not t:
            raise ValueError("bad person/time")
        return (rel, p, t)
    if rel in ("ON", "NOT_ON"):
        p, s = _find(h.staff, args[0]), _find(h.stations, args[1].lstrip("@"))
        if not p or not s:
            raise ValueError("bad person/station")
        return (rel, p, s)
    refs = [_ref(h, a) for a in args]
    if any(x is None for x in refs):
        raise ValueError("unknown name in %s" % rel)
    if len(set(refs)) != len(refs):
        raise ValueError("repeated argument")
    if swap and rel in ("BEFORE", "RIGHT_BEFORE"):
        refs = refs[::-1]
    return canon((rel, *refs))


def _mentions(h, text):
    low = text.casefold()
    names = {n for n in h.staff if re.search(r"(?<!\w)" + re.escape(n.casefold()) + r"(?!\w)", low)}
    stations = {s for s in h.stations if re.search(r"(?<!\w)" + re.escape(s.casefold()) + r"(?!\w)", low)}
    times = set()
    for b in h.blocks:
        # The header contract makes block labels strings. Exact labels need
        # grounding even when they are not clock times (e.g. "dawn shift").
        if re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", text, re.I):
            times.add(b)
            continue
        # Preserve the accepted short-hour spelling for clock-like labels, but
        # do not let arbitrary valid labels raise during validation.
        m = re.fullmatch(r"(\d{1,2}):(\d\d)", b)
        if m:
            hh = int(m.group(1))
            if (re.search(r"(?<![\d:])%d(?![\d:])" % hh, text)
                    or re.search(r"(?<!\d)0%d(?![\d])" % hh, text)):
                times.add(b)
    return names, stations, times


def grounded(h, c, line_text):
    """Every person / station / time in the constraint must occur in its source line."""
    names, stations, times = _mentions(h, line_text)
    args = []
    if c[0] in ("AT", "NOT_AT"):
        return c[1] in names and c[2] in times
    if c[0] in ("ON", "NOT_ON"):
        return c[1] in names and c[2] in stations
    for a in c[1:]:
        args.append(a)
    for a in args:
        if a[0] == "p" and a[1] not in names:
            return False
        if a[0] == "h" and a[1] not in stations:
            return False
    return True


def parse_output(text, h, body, cfg, only=None, source=""):
    """Parse model text. `only`: restrict to these idx0 (targeted calls). Never raises."""
    res = Parse(raw=text or "", source=source)
    n = len(body)
    seen = {}
    for row in (text or "").replace("```", "\n").splitlines():
        m = _LINE.match(row)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < n) or (only is not None and idx not in only):
            continue
        toks = [_clean(t) for t in m.group(2).split("|")]
        rel = re.sub(r"[\s\-]+", "_", toks[0].upper()) if toks else ""
        seen.setdefault(idx, []).append((rel, toks[1:]))
    for idx, items in seen.items():
        if any(rel in ("UNSURE", "UNCERTAIN") for rel, _ in items):
            res.invalid[idx] = "model_uncertain"
            continue
        if len(items) > 1:
            # The prompt contract is one record per source note. Deduplicating
            # repeated identical readings would hide a malformed response and
            # could make an otherwise incomplete extraction look complete.
            res.invalid[idx] = "duplicate statement records"
            continue
        outs = set()
        err = None
        for rel, args in items:
            if rel == NONE and not [a for a in args if a]:
                outs.add(NONE)
                continue
            try:
                c = _build(h, rel, args)
                if not cfg.off("no_grounding") and not grounded(h, c, body[idx]):
                    raise ValueError("argument not in source line")
                outs.add(c)
            except ValueError as e:
                err = str(e)
        if len(outs) == 1:
            res.readings[idx] = next(iter(outs))
        elif len(outs) > 1:
            res.invalid[idx] = "conflicting duplicate outputs"
        else:
            res.invalid[idx] = err or "empty"
    return res
