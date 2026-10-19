"""Real-time cost ledger: token usage x rate card -> $ estimate, emitted as
CloudWatch EMF plus a JSONL routing-decision line on stdout.

This is the "Ledger (estimate)" cost view in plan.md §2 — the live number shown
during the demo. It's reconciled later against Bedrock invocation logs (Athena)
and Cost Explorer (AIP tags). Because it uses each model's OWN token counts from
the Bedrock `usage` block, it must never reprice one model's usage with another
model's rates.
"""
from __future__ import annotations

import json
import sys
import time

import yaml

import config

# CloudWatch namespace for the EstimatedCostUSD metric.
EMF_NAMESPACE = "AZ/ModelRouting"

with open(config.REPO_ROOT / "pricing.yaml") as _f:
    _PRICING: dict[str, dict[str, float]] = yaml.safe_load(_f)["models"]


def compute_cost(usage: dict, key: str) -> float:
    """USD for one response, from an Anthropic `usage` block and a model key.

    usage keys (any may be absent): input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens. Rates are $/MTok.
    `input_tokens` from Anthropic excludes cache tokens, so the three input-side
    buckets are billed separately and never double-counted.
    """
    rate = _PRICING.get(key)
    if not rate:
        return 0.0
    tok = lambda name: usage.get(name, 0) or 0
    return (
        tok("input_tokens") * rate["input"]
        + tok("output_tokens") * rate["output"]
        + tok("cache_read_input_tokens") * rate.get("cache_read", 0.0)
        + tok("cache_creation_input_tokens") * rate.get("cache_write", 0.0)
    ) / 1_000_000.0


def emit(*, strategy: str, routed_model: str, requested_model: str, reason: str,
         usage: dict, key: str, fallback: bool = False, extra: dict | None = None) -> float:
    """Compute cost and print one EMF line (metric + full routing decision).

    Returns the cost so the caller can also surface it as a response header.
    A single EMF line doubles as the greppable JSONL decision record: CloudWatch
    extracts EstimatedCostUSD by {strategy, routed_model}; the log keeps the rest.
    """
    cost = compute_cost(usage, key)
    record = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": EMF_NAMESPACE,
                "Dimensions": [["strategy", "routed_model"]],
                "Metrics": [
                    {"Name": "EstimatedCostUSD", "Unit": "None"},
                    {"Name": "InputTokens", "Unit": "Count"},
                    {"Name": "OutputTokens", "Unit": "Count"},
                ],
            }],
        },
        "strategy": strategy,
        "routed_model": routed_model,
        "requested_model": requested_model,
        "route_reason": reason,
        "fallback": fallback,
        "EstimatedCostUSD": round(cost, 6),
        "InputTokens": usage.get("input_tokens", 0) or 0,
        "OutputTokens": usage.get("output_tokens", 0) or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
        **(extra or {}),
    }
    # One line to stdout -> CloudWatch Logs; flush so it isn't buffered per-request.
    print(json.dumps(record), flush=True, file=sys.stdout)
    return cost
