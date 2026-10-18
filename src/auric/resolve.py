"""Combine several parses into one reading per line (majority vote, solver-settled ties)."""
from collections import Counter
from itertools import product

from .extract import NONE, fmt
from .solver import canon, solve

MAX_COMBOS = 512


def constraints(readings):
    return [(i, c) for i, c in sorted(readings.items()) if c != NONE]


def tally(parses, weights=None):
    """votes[idx0] = Counter(reading -> weight); `invalid[idx0]` = number of rejected outputs."""
    votes, invalid = {}, Counter()
    for k, p in enumerate(parses):
        w = 1.0 if weights is None else weights[k]
        for i, r in p.readings.items():
            votes.setdefault(i, Counter())[r] += w
        for i in p.invalid:
            invalid[i] += 1
    return votes, invalid


def disputed(votes, invalid, n):
    """Lines worth re-asking about, most contested first."""
    out = []
    for i in range(n):
        v = votes.get(i)
        if not v:
            out.append((-1.0, i))                      # never read validly
            continue
        tot = sum(v.values())
        top = max(v.values())
        if len(v) > 1 or invalid.get(i):
            out.append((top / tot, i))
    return [i for _, i in sorted(out)]


def options_for(votes, i):
    opts = [r for r, _ in (votes.get(i) or Counter()).most_common()]
    if NONE not in opts:
        opts.append(NONE)
    return [fmt(o) for o in opts]


def consensus(rota, n, votes, first):
    """Majority per line; only solver-settled ties are resolved.

    Missing votes and ties for which no plausible joint resolution exists stay
    unresolved. In particular, the first parse is only an ordering preference
    when testing combinations; it is never evidence that breaks a tie.
    """
    readings, ties, unresolved = {}, {}, []
    for i in range(n):
        v = votes.get(i)
        if not v:
            unresolved.append(i)
            continue
        top = max(v.values())
        best = [r for r, w in v.items() if w == top]
        if len(best) == 1:
            readings[i] = best[0]
        else:
            pref = first.get(i)
            best.sort(key=lambda r: (r != pref, fmt(r)))
            ties[i] = best
    info = {"ties": sorted(ties), "unresolved": unresolved}
    if ties:
        idxs = sorted(ties)
        combos = 1
        for i in idxs:
            combos *= len(ties[i])
        if combos <= MAX_COMBOS:
            plausible = []
            for combo in product(*[ties[i] for i in idxs]):
                trial = dict(readings)
                trial.update(dict(zip(idxs, combo)))
                if solve(rota, constraints(trial)).plausible():
                    plausible.append(combo)
            if len(plausible) == 1:
                readings.update(dict(zip(idxs, plausible[0])))
            elif len(plausible) > 1:
                # A solver-plausible outcome is a filter, not a tie-breaker
                # when several distinct extractions survive it.
                for col, i in enumerate(idxs):
                    values = {combo[col] for combo in plausible}
                    if len(values) == 1:
                        readings[i] = next(iter(values))
                    else:
                        unresolved.append(i)
            else:
                unresolved.extend(i for i in idxs if i not in unresolved)
        else:
            unresolved.extend(i for i in idxs if i not in unresolved)
    info["unresolved"] = sorted(unresolved)
    return readings, info
