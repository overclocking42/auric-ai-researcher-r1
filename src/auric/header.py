"""Deterministic header parsing: who is on the rota, which blocks, which stations."""
import re
from dataclasses import dataclass


@dataclass
class Header:
    staff: list
    blocks: list
    stations: list
    holders: list          # people who hold a station (subset of staff)


def _split(s):
    parts = re.split(r",\s*|\s+and\s+", s.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_header(line):
    m = re.search(r"rota:\s*(.+?)\.\s*Blocks run\s*(.+?),\s*one person per block", line)
    if not m:
        raise ValueError("cannot parse staff/blocks from header")
    staff, blocks = _split(m.group(1)), _split(m.group(2))
    m2 = re.search(r"one person on each:\s*(.+?)\.\s*The people on a station are\s*(.+?)(?:;|\.)", line)
    if not m2:
        raise ValueError("cannot parse stations from header")
    stations, holders = _split(m2.group(1)), _split(m2.group(2))
    if any(h not in staff for h in holders):
        raise ValueError("station-holder not on rota")
    return Header(staff, blocks, stations, holders)


def split_item(item):
    """(header_line, [body lines]) -- same body definition the scorer uses."""
    text = item["text"]
    rows = text.split("\n")
    body = [l.strip() for l in rows[1:] if l.strip()]
    return rows[0], body
