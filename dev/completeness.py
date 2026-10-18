"""DEV-ONLY audit for possible relation omissions; never constructs constraints.

This lexical review aid is deliberately outside ``src/``. It reports note
indices whose surface wording suggests a relation family while the supplied
extraction is missing, NONE, or a different relation family. It is a review
signal, not a parser, answer rule, or substitute for model extraction.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from auric.header import parse_header, split_item  # noqa: E402


_CUES = {
    "between": re.compile(r"\b(?:between|sandwiched)\b", re.I),
    "adjacency": re.compile(
        r"\b(?:back[ -]to[ -]back|straight\s+(?:after|before|over)|"
        r"directly\s+(?:after|before)|immediately\s+(?:after|before)|"
        r"no\s+gap|adjacent|consecutive)\b", re.I),
    "ordering": re.compile(r"\b(?:earlier|later|before|after|precedes)\b", re.I),
    "negation": re.compile(r"\b(?:not|never|cannot|can't|won't|doesn't|do\s+not|ruled\s+out|exclude[sd]?|other\s+than)\b", re.I),
}
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_HISTORICAL_OR_COUNTERFACTUAL = re.compile(
    r"\b(?:last\s+(?:month|quarter|week|year)|back\s+in\s+the\s+old|"
    r"old\s+(?:arrangement|rota|roster)|previous\s+(?:cycle|rota|roster)|"
    r"would\s+have|had\s+.+\s+would|asked\s+to\s+move|request\s+was\s+declined|"
    r"wanted\s+.+\s+did\s+not\s+get|disagreement\s+about\s+whether|"
    r"nobody\s+(?:has\s+)?confirmed\s+whether)\b", re.I)


def _has_entity(text, entity):
    return re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", text, re.I) is not None


def cue_families(line, header):
    """Return possible relation families from wording only; never operands."""
    low = line.casefold()
    if _HISTORICAL_OR_COUNTERFACTUAL.search(line):
        return set()
    families = set()
    if _CUES["between"].search(line):
        families.add("BETWEEN")
    if _CUES["adjacency"].search(line):
        families.add("RIGHT_BEFORE")
    if _CUES["ordering"].search(line):
        families.add("ORDERING")
    if _TIME.search(line) and _CUES["negation"].search(line):
        families.add("NOT_AT")
    mentioned_stations = [s for s in header.stations if _has_entity(low, s.casefold())]
    if mentioned_stations and _CUES["negation"].search(line):
        families.add("NOT_ON")
    elif mentioned_stations and not families.intersection({"BETWEEN", "RIGHT_BEFORE", "ORDERING"}):
        # A station mention with assignment-like wording is only a review cue.
        if re.search(r"\b(?:on|station|covered|runs?|handles?|assigned|job|holder)\b", line, re.I):
            families.add("ON")
    return families


def _matches(family, relation, reading):
    if relation is None or relation == "NONE":
        return False
    if family == "ORDERING":
        # BETWEEN entails ordering on both sides; do not flag a valid compact
        # representation of a note that states a person is between two others.
        return relation in {"BEFORE", "RIGHT_BEFORE", "BETWEEN"}
    return relation == family


def find_possible_omissions(item, extracted):
    """Return safe alerts (id/index/family/relation only), not constraints.

    ``extracted`` maps one-based note index to a relation name or compact
    reading. The output intentionally excludes source strings and operands.
    """
    header_line, body = split_item(item)
    header = parse_header(header_line)
    alerts = []
    for idx, line in enumerate(body, 1):
        cues = cue_families(line, header)
        if not cues:
            continue
        value = extracted.get(str(idx), extracted.get(idx))
        relation = value.split("|", 1)[0] if isinstance(value, str) else None
        missing = sorted(f for f in cues if not _matches(f, relation, value))
        if missing:
            alerts.append({"item_id": item.get("id"), "statement_index": idx,
                           "cue_families": missing,
                           "observed_relation": relation or "MISSING"})
    return alerts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/items.json")
    ap.add_argument("--trace", required=True, help="private local JSONL trace; raw fields are ignored")
    ap.add_argument("--ids", default="", help="optional comma-separated item IDs")
    args = ap.parse_args()
    items = json.load(open(args.items, encoding="utf-8"))
    selected = set(filter(None, (x.strip() for x in args.ids.split(","))))
    by_id = {item["id"]: item for item in items}
    reports = []
    with open(args.trace, encoding="utf-8") as trace:
        for raw in trace:
            row = json.loads(raw)
            iid = row.get("id")
            if iid not in by_id or (selected and iid not in selected):
                continue
            reports.extend(find_possible_omissions(by_id[iid], row.get("final", {})))
    print(json.dumps({"review_only": True, "alerts": reports}, indent=2))


if __name__ == "__main__":
    main()
