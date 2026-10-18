"""Exact solver over the full world space (blocks x station holders).

World = (perm, hp): perm[i] = block index of staff i; hp[j] = which holder runs station j.
With 5 staff and 3 station-holders that is 5! * 3! = 720 worlds, enumerated exhaustively.

Constraints (canonical tuples; refs are ("p", name) or ("h", station)):
  ("AT", name, time) ("NOT_AT", name, time) ("ON", name, station) ("NOT_ON", name, station)
  ("BEFORE", ref, ref)        first is strictly earlier
  ("RIGHT_BEFORE", ref, ref)  first's block is immediately before second's
  ("BETWEEN", ref, ref, ref)  first is strictly between the other two (either order)
"""
from itertools import combinations, permutations
from dataclasses import dataclass, field


def canon(c):
    """Canonical form used for equality/voting."""
    if c[0] == "BETWEEN":
        a, b = sorted([c[2], c[3]])
        return ("BETWEEN", c[1], a, b)
    return c


class Rota:
    def __init__(self, header):
        self.staff, self.blocks = list(header.staff), list(header.blocks)
        self.stations, self.holders = list(header.stations), list(header.holders)
        if len(self.blocks) != len(self.staff) or len(self.holders) != len(self.stations):
            raise ValueError("unsupported rota shape")
        self.sidx = {n: i for i, n in enumerate(self.staff)}
        self.bidx = {b: i for i, b in enumerate(self.blocks)}
        self.stidx = {s: i for i, s in enumerate(self.stations)}
        self.hidx = [self.sidx[h] for h in self.holders]
        self.worlds = [(p, hp) for p in permutations(range(len(self.staff)))
                       for hp in permutations(range(len(self.stations)))]
        self.full = (1 << len(self.worlds)) - 1
        self._mask = {}

    # -- semantics -------------------------------------------------------------------------
    def _block(self, w, ref):
        perm, hp = w
        if ref[0] == "p":
            return perm[self.sidx[ref[1]]]
        return perm[self.hidx[hp[self.stidx[ref[1]]]]]

    def holds(self, c, w):
        perm, hp = w
        t = c[0]
        if t == "AT":
            return perm[self.sidx[c[1]]] == self.bidx[c[2]]
        if t == "NOT_AT":
            return perm[self.sidx[c[1]]] != self.bidx[c[2]]
        if t == "ON":
            return self.hidx[hp[self.stidx[c[2]]]] == self.sidx[c[1]]
        if t == "NOT_ON":
            return self.hidx[hp[self.stidx[c[2]]]] != self.sidx[c[1]]
        if t == "BEFORE":
            return self._block(w, c[1]) < self._block(w, c[2])
        if t == "RIGHT_BEFORE":
            return self._block(w, c[1]) + 1 == self._block(w, c[2])
        if t == "BETWEEN":
            a, b, d = self._block(w, c[1]), self._block(w, c[2]), self._block(w, c[3])
            return min(b, d) < a < max(b, d)
        raise ValueError(t)

    def mask(self, c):
        c = canon(c)
        m = self._mask.get(c)
        if m is None:
            m = 0
            for i, w in enumerate(self.worlds):
                if self.holds(c, w):
                    m |= 1 << i
            self._mask[c] = m
        return m

    def assignment(self, wi):
        perm, hp = self.worlds[wi]
        who = {self.hidx[hp[j]]: self.stations[j] for j in range(len(self.stations))}
        out = {}
        for i, name in enumerate(self.staff):
            d = {"block": self.blocks[perm[i]]}
            if i in who:
                d["station"] = who[i]
            out[name] = d
        return out


@dataclass
class Outcome:
    kind: str                       # unique | ambiguous | inconsistent
    n: int                          # number of consistent worlds
    worlds: list = field(default_factory=list)   # first (up to 50) consistent world indices
    core: tuple = ()                # line indices of a minimal conflict (inconsistent only)

    def plausibility_reasons(self):
        """Explain assignment-range violations without discarding the solver result."""
        if self.kind == "inconsistent":
            return ["conflict_core_too_small"] if len(self.core) < 3 else []
        if self.kind == "unique":
            return ["unique_world_count_mismatch"] if self.n != 1 else []
        if self.kind == "ambiguous":
            return ["ambiguous_world_count_out_of_range"] if not 2 <= self.n <= 4 else []
        return ["unsupported_solver_classification"]

    def plausible(self):
        """The brief: 1 unique, 2-4 ambiguous, or an inconsistent core of >= 3 statements."""
        return not self.plausibility_reasons()


def _and(masks, full):
    m = full
    for x in masks:
        m &= x
    return m


def deletion_mus(masks, full):
    """One minimal unsatisfiable subset (indices into `masks`), deterministic by position."""
    keep = list(range(len(masks)))
    for i in list(keep):
        trial = [j for j in keep if j != i]
        if _and([masks[j] for j in trial], full) == 0:
            keep = trial
    return keep


def solve(rota, cons, minimize=True):
    """cons: list of (line_index, constraint). Returns Outcome."""
    masks = [rota.mask(c) for _, c in cons]
    m = _and(masks, rota.full)
    if m == 0:
        keep = deletion_mus(masks, rota.full) if minimize else list(range(len(masks)))
        return Outcome("inconsistent", 0, [], tuple(cons[j][0] for j in keep))
    n = bin(m).count("1")
    worlds = []
    for i in range(len(rota.worlds)):
        if (m >> i) & 1:
            worlds.append(i)
            if len(worlds) >= 50:
                break
    return Outcome("unique" if n == 1 else "ambiguous", n, worlds)


def all_mus(masks, full, maxk=8):
    """Every minimal unsatisfiable subset up to size maxk (dev/validation use)."""
    cores, n = [], len(masks)
    for k in range(1, min(maxk, n) + 1):
        for combo in combinations(range(n), k):
            cs = set(combo)
            if any(c <= cs for c in cores):
                continue
            if _and([masks[i] for i in combo], full) == 0:
                cores.append(cs)
    return cores
