"""Router v2 — demo-grade, literature-grounded (see plan.md §5).

Pipeline: hard rules → session stickiness → features → estimator (Haiku tagger +
SCOPE-lite fingerprint) → utility decision. Post-generation fallback lives in the
gateway (main.py). The estimator runs only on the first turn of a session; later
turns reuse the sticky tier (≈0 added latency, and it protects the prompt cache).

Entry point: `route(payload, ...) -> RouteDecision`. Synchronous; the gateway
calls it via run_in_threadpool so the Haiku tagger's blocking call doesn't stall
the event loop.
"""
from __future__ import annotations

import functools
import pathlib
from dataclasses import dataclass
from typing import Callable, Optional

import yaml

import config
from . import fingerprint, rules, tagger, utility
from .features import extract_features
from .sticky import StickyTiers

_POLICIES_FILE = pathlib.Path(__file__).resolve().parent / "policies.yaml"

# One sticky cache per process (in-proc; swap for ElastiCache to scale out).
STICKY = StickyTiers()


@functools.lru_cache(maxsize=1)
def policies() -> dict:
    with open(_POLICIES_FILE) as f:
        return yaml.safe_load(f)


@dataclass
class RouteDecision:
    model: str                     # catalog key (haiku45 | sonnet5 | opus5 | fable5)
    reason: str
    lam: float
    family: Optional[str] = None
    difficulty: Optional[int] = None
    p_hat: Optional[float] = None
    c_hat: Optional[float] = None
    source: str = "rules"          # rules | sticky | prior | anchors


def resolve_lambda(alpha: Optional[str], preset: Optional[str]) -> float:
    """λ from header x-routing-alpha (raw float) or x-routing-preset (name)."""
    pol = policies()
    presets = pol.get("lambda_presets", {})
    if alpha:
        try:
            return float(alpha)
        except ValueError:
            pass
    name = (preset or "").strip().lower()
    if name in presets:
        return float(presets[name])
    return float(presets.get(pol.get("default_preset", "balanced"), 0.5))


def route(payload: dict, *, alpha: Optional[str] = None, preset: Optional[str] = None,
          invoke_fn: Optional[Callable[[str], str]] = None,
          embed_fn: Optional[Callable[[str], list[float]]] = None,
          session: Optional[str] = None) -> RouteDecision:
    features = extract_features(payload)
    lam = resolve_lambda(alpha, preset)

    # Stage 0 — hard rules. A forced key skips the estimator entirely.
    outcome = rules.hard_rules(features)
    if outcome.forced_key is not None:
        tier = config.tier_of(outcome.forced_key)
        if session:
            tier = STICKY.escalate(session, tier)
        return RouteDecision(config.key_at_tier(tier), outcome.reason or "rule", lam,
                             source="rules")

    floor = outcome.floor_tier

    # Stage 1 — session stickiness. On a later turn, reuse the tier (escalate-only)
    # and skip the estimator: the estimator is a first-turn cost.
    cached = STICKY.get(session) if session else None
    if cached is not None:
        floor = max(floor, cached)
        chosen = config.key_at_tier(floor)
        STICKY.escalate(session, config.tier_of(chosen))
        reason = "sticky" if outcome.reason is None else f"sticky+{outcome.reason}"
        return RouteDecision(chosen, reason, lam, source="sticky")

    # Stage 3 — estimator (first turn).
    tag = tagger.tag(features, invoke_fn=invoke_fn)
    # Policy floor per family (Arch-Router policies.yaml), respected by utility.
    fam_floor = policies().get("families", {}).get(tag.family, {}).get("min_tier", config.MIN_TIER)
    floor = max(floor, int(fam_floor))

    keys = list(config.MODELS.keys())
    query = f"{features.system_text[:500]}\n{features.user_text}"
    est, source = fingerprint.estimates(tag, features, policies(), keys,
                                        query_text=query, embed_fn=embed_fn)

    # Stage 4 — utility.
    chosen, utilities = utility.decide(est, lam, floor)
    p_hat, c_hat = est[chosen]
    if session:
        STICKY.escalate(session, config.tier_of(chosen))

    reason = f"{tag.family}:d{tag.difficulty}"
    if outcome.reason:
        reason = f"{outcome.reason}+{reason}"
    return RouteDecision(chosen, reason, lam, family=tag.family, difficulty=tag.difficulty,
                         p_hat=round(p_hat, 4), c_hat=round(c_hat, 8), source=source)
