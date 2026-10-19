"""Stage 0 — hard rules and floors (µs, no model calls).

Rules either FORCE a model (skip the estimator entirely) or raise a tier FLOOR
the utility stage must respect. Grounded in plan.md §5.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from .features import Features

# Thresholds.
LONG_CONTEXT_TOKENS = 150_000   # >= this -> needs a long-context model (>= sonnet5)
MAX_IMAGES_HAIKU = 4            # more attachments than this -> >= sonnet5
AUX_MAX_TOKENS = 96            # tiny completions => auxiliary call

# Cowork skill signatures in the system prompt => real document work, floor sonnet5.
COWORK_SKILL_SIGNATURES = (
    "python-pptx", "python-docx", "openpyxl", ".pptx", ".docx", ".xlsx",
    "presentation skill", "document skill", "spreadsheet skill",
)
# Auxiliary-call signatures Desktop/Cowork uses for housekeeping completions.
AUX_SIGNATURES = (
    "generate a title", "concise title", "summarize the conversation",
    "conversation summary", "for the purpose of compaction", "classify the",
)


@dataclass
class RuleOutcome:
    forced_key: str | None      # if set, use this model and skip the estimator
    floor_tier: int             # minimum tier the estimator must respect
    reason: str | None          # set when a rule fired


def hard_rules(features: Features) -> RuleOutcome:
    low_sys = features.system_text.lower()
    low_user = features.user_text.lower()

    # Auxiliary calls (title gen, compaction, short classification) -> cheapest.
    is_aux = any(s in low_sys or s in low_user for s in AUX_SIGNATURES)
    if is_aux or (features.max_tokens and features.max_tokens <= AUX_MAX_TOKENS
                  and features.n_tools == 0 and features.tokens_in < 2000):
        return RuleOutcome(config.key_at_tier(config.MIN_TIER), config.MIN_TIER, "rule:aux")

    floor = config.MIN_TIER
    reason = None

    # Long context or heavy attachments -> at least sonnet5.
    sonnet_tier = config.tier_of("sonnet5")
    if features.tokens_in >= LONG_CONTEXT_TOKENS:
        floor, reason = max(floor, sonnet_tier), "rule:long-context"
    if (features.n_images + features.n_documents) > MAX_IMAGES_HAIKU:
        floor, reason = max(floor, sonnet_tier), "rule:attachments"

    # Cowork document-skill work -> at least sonnet5.
    if any(sig in low_sys for sig in COWORK_SKILL_SIGNATURES):
        floor, reason = max(floor, sonnet_tier), "rule:cowork-skill"

    return RuleOutcome(None, floor, reason)
