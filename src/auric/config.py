"""Run configuration, including the ablation switches (see README)."""
from dataclasses import dataclass, field

ABLATIONS = {
    "no_solver": "Granite answers the item directly (no extraction, no solver, no MUS).",
    "no_guidance": "Remove the filler/trap rules and filler examples from the extraction prompt.",
    "no_grounding": "Skip the deterministic check that every argument occurs in its source line.",
    "no_vote": "3x/10x: use only the first extraction (extra calls unused for voting).",
    "no_targeted": "10x: never spend calls on disputed lines.",
    "no_minimize": "Inconsistent items cite every extracted constraint line (no MUS).",
    "same_prompt": "Repeated calls reuse the identical prompt instead of varied examples/order.",
    "no_completeness": "Remove the statement-by-statement omission self-check from the extraction prompt.",
}


@dataclass
class Config:
    ablate: frozenset = field(default_factory=frozenset)
    targeted_weight: float = 2.0     # a focused re-read counts this much in the line vote
    targeted_batch: int = 3          # disputed lines per targeted call
    max_tokens: int = 1200
    timeout_s: float = 120.0

    def off(self, name):
        return name in self.ablate

    @staticmethod
    def from_names(names):
        bad = [n for n in names if n not in ABLATIONS]
        if bad:
            raise ValueError("unknown ablation(s): %s (known: %s)" % (bad, ", ".join(ABLATIONS)))
        return Config(ablate=frozenset(names))
