"""DEV ONLY: single-edit mutations of a constraint (used by the fake model to inject errors)."""
from auric.extract import NONE
from auric.solver import canon


def _edits(c):
    t = c[0]
    out = [NONE]
    if t == "AT":
        out.append(("NOT_AT",) + c[1:])
    elif t == "NOT_AT":
        out.append(("AT",) + c[1:])
    elif t == "ON":
        out.append(("NOT_ON",) + c[1:])
    elif t == "NOT_ON":
        out.append(("ON",) + c[1:])
    elif t == "BEFORE":
        out += [("BEFORE", c[2], c[1]), ("RIGHT_BEFORE", c[1], c[2])]
    elif t == "RIGHT_BEFORE":
        out += [("RIGHT_BEFORE", c[2], c[1]), ("BEFORE", c[1], c[2])]
    elif t == "BETWEEN":
        out += [canon(("BETWEEN", c[2], c[1], c[3])), canon(("BETWEEN", c[3], c[1], c[2]))]
    return out
