"""DEV ONLY. A fake Granite built from gold readings plus injected extraction noise, so the whole
pipeline can be exercised without network access. Results from it are SIMULATION, not Granite."""
import random
import re
import threading
import zlib

from auric.extract import NONE, fmt
from edits import _edits


class MockModel:
    def __init__(self, gold, eps_real=0.0, eps_trap=0.0, p_garbage=0.0, seed=0, mode="gold"):
        """mode: gold | blind (always NONE) | flipped (negate every real constraint)."""
        self.gold, self.eps_real, self.eps_trap = gold, eps_real, eps_trap
        self.p_garbage, self.seed, self.mode = p_garbage, seed, mode
        self._n, self._lock = {}, threading.Lock()
        self.calls = {}

    def _rng(self, item_id):
        with self._lock:
            k = self._n.get(item_id, 0)
            self._n[item_id] = k + 1
            self.calls[item_id] = k + 1
        return random.Random(zlib.crc32(("%s|%d|%d" % (item_id, k, self.seed)).encode()))

    def _read(self, g, rng):
        real, spur = g["reading"], g["spur"]
        if self.mode == "blind":
            return NONE
        if real != NONE:
            if self.mode == "flipped":
                e = [x for x in _edits(real) if x != NONE]
                return e[0] if e else real
            if rng.random() < self.eps_real:
                opts = _edits(real)
                return rng.choice(opts)           # includes NONE (a drop) and misreads
            return real
        if spur and rng.random() < self.eps_trap:
            return spur
        return NONE

    def complete(self, item_id, messages, max_tokens):
        rng = self._rng(item_id)
        gold = self.gold[item_id]
        user = messages[-1]["content"]
        if "MODE: TARGETED" in user:
            tg = re.search(r"Only answer for these notes: ([\d, ]+)", user).group(1)
            targets = [int(x) for x in re.findall(r"\d+", tg)]
            rows = []
            for k in targets:
                opts = re.search(r"^%d: options: (.*)$" % k, user, re.M)
                g = gold[k - 1]
                want = g["reading"]
                if rng.random() < self.eps_real and opts:
                    rows.append("%d|%s" % (k, rng.choice(opts.group(1).split(" || "))))
                else:
                    rows.append("%d|%s" % (k, fmt(want)))
            return "\n".join(rows)
        notes = [int(m.group(1)) for m in re.finditer(r"^\[(\d+)\] ", user.split("NOTES", 1)[1], re.M)]
        rows = []
        for k in notes:
            if rng.random() < self.p_garbage:
                rows.append("%d|SOMETHING|odd" % k)
                continue
            rows.append("%d|%s" % (k, fmt(self._read(gold[k - 1], rng))))
        return "Here is the answer:\n" + "\n".join(sorted(rows, key=lambda s: int(s.split("|")[0])))


class FailingModel:
    """Every call raises (simulates an outage). Used to prove budgets hold even on errors."""
    def __init__(self):
        self.n = {}

    def complete(self, item_id, messages, max_tokens):
        self.n[item_id] = self.n.get(item_id, 0) + 1
        raise ConnectionError("simulated outage")
