"""Stage 4 — utility decision.

    U(m) = p̂(m) − λ · norm(log ĉ(m))

λ is the budget knob (higher = more cost-averse). Cost is log-scaled then min-max
normalized across the candidate set so λ means the same thing regardless of the
absolute price spread. Candidates are restricted to tier >= floor; ties in utility
break to the cheaper (lower-tier) model.
"""
from __future__ import annotations

import math

import config


def decide(estimates: dict[str, tuple[float, float]], lam: float, floor_tier: int
           ) -> tuple[str, dict[str, float]]:
    """Return (chosen_key, {key: utility}). Only models at/above floor_tier are
    considered; if the floor excludes everything, the top tier is used."""
    candidates = [k for k in estimates if config.tier_of(k) >= floor_tier]
    if not candidates:
        candidates = [config.key_at_tier(config.MAX_TIER)]

    log_costs = {k: math.log(max(estimates[k][1], 1e-12)) for k in candidates}
    lo, hi = min(log_costs.values()), max(log_costs.values())
    span = hi - lo

    def norm_cost(k: str) -> float:
        return 0.0 if span < 1e-12 else (log_costs[k] - lo) / span

    utilities = {k: estimates[k][0] - lam * norm_cost(k) for k in candidates}

    # argmax U; tie -> cheaper (lower tier).
    best = max(candidates, key=lambda k: (utilities[k], -config.tier_of(k)))
    return best, utilities
