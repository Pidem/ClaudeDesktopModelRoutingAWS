"""Gateway configuration, model catalog, and strategy/routing resolution.

Loads settings from environment (.env). Credentials for Bedrock come from the
AWS default credential chain (env vars, ~/.aws profile, or an IAM role). When
GATEWAY_ASSUME_ROLE_ARN is set, the gateway assumes that role via STS instead.

Phase 1/2 additions (see plan.md):
  - MODELS catalog: one source of truth (tier, context, vision, Fable param rules).
  - Application Inference Profiles (AIPs): (strategy x model) -> ARN, loaded from
    infra/inference_profiles.json so Bedrock spend is tagged by routing-strategy.
  - Strategy resolution: baseline | auto | pinned.
  - sanitize_params(): per-model body rules (Fable temperature/top_k/top_p).
  - Fallback ladder: escalate to the next tier on refusal / throttle / 5xx.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = pathlib.Path(__file__).resolve().parent

# --- Server / auth (Claude Desktop -> Gateway) --------------------------------
# Static bearer token Claude Desktop must send. Two sources, in priority order:
#   1. GATEWAY_TOKEN_SECRET_ARN -> read from AWS Secrets Manager (prod default).
#   2. GATEWAY_AUTH_TOKEN       -> plain env var (local dev).
# If neither is set, auth is disabled (credential kind = none) — localhost only.
GATEWAY_AUTH_TOKEN = os.environ.get("GATEWAY_AUTH_TOKEN", "")
GATEWAY_TOKEN_SECRET_ARN = os.environ.get("GATEWAY_TOKEN_SECRET_ARN", "")


@functools.lru_cache(maxsize=1)
def get_auth_token() -> str:
    """Resolve the gateway bearer token (Secrets Manager, then env). Cached."""
    if GATEWAY_TOKEN_SECRET_ARN:
        import boto3

        sm = boto3.client("secretsmanager", region_name=AWS_REGION)
        return sm.get_secret_value(SecretId=GATEWAY_TOKEN_SECRET_ARN)["SecretString"]
    return GATEWAY_AUTH_TOKEN

# --- AWS / Bedrock (Gateway -> Bedrock) ---------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Optional: assume this role for Bedrock calls (leave unset to use the default
# chain / task / instance role).
GATEWAY_ASSUME_ROLE_ARN = os.environ.get("GATEWAY_ASSUME_ROLE_ARN", "")

# Bedrock expects this on every Anthropic Messages request body.
BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Claude 5 on Bedrock controls thinking via output_config.effort (low|medium|high)
# and defaults to HIGH — which fills max_tokens with thinking and can return an
# empty answer, and is slow. Clients like Claude Code either send no effort or send
# the Anthropic `thinking` block (which these models reject). We normalize both to
# output_config.effort; this is the default injected when a client specifies none.
GATEWAY_DEFAULT_EFFORT = os.environ.get("GATEWAY_DEFAULT_EFFORT", "low").strip().lower()

# When true, requests for the fleet default model (claude-fable-5) are treated as
# `auto` — the "user never changed the default" scenario, the real AZ story.
ROUTER_INTERCEPT_DEFAULT = os.environ.get("ROUTER_INTERCEPT_DEFAULT", "").lower() in (
    "1", "true", "yes",
)

# --- Model catalog (one source of truth) --------------------------------------
# Keyed by short catalog key. `system_profile` is the Bedrock cross-region system
# inference profile id used when no Application Inference Profile is available
# (local dev). `desktop_id` is the model id Claude Desktop sends. `tier` orders
# the fallback ladder (ascending = cheaper/weaker first). `param_rules` are
# enforced by sanitize_params().
MODELS: dict[str, dict] = {
    "haiku45": {
        "desktop_id": "claude-haiku-4-5",
        "system_profile": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "tier": 1, "ctx": 200_000, "vision": True,
        "supports_thinking": False,
    },
    "sonnet5": {
        "desktop_id": "claude-sonnet-5",
        "system_profile": "us.anthropic.claude-sonnet-5",
        "tier": 2, "ctx": 1_000_000, "vision": True,
        "supports_thinking": True,
    },
    "opus5": {
        "desktop_id": "claude-opus-5",
        "system_profile": "us.anthropic.claude-opus-5",
        "tier": 3, "ctx": 1_000_000, "vision": True,
        "supports_thinking": True,
    },
    "fable5": {
        "desktop_id": "claude-fable-5",
        "system_profile": "us.anthropic.claude-fable-5",
        "tier": 4, "ctx": 1_000_000, "vision": True,
        "supports_thinking": True,
        # Fable 5 rejects/needs specific sampling params (see model card).
        "param_rules": {"temperature": "1_or_unset", "top_k": "drop", "top_p": ">=0.99"},
    },
}

# The model users get when they never touch the picker (the expensive default the
# router is pitched against). Also the model the `baseline` strategy always uses.
DEFAULT_MODEL_KEY = "fable5"

# Virtual model exposed in /v1/models that routes per the router.
AUTO_MODEL_ID = "claude-auto"

# desktop model id -> catalog key
DESKTOP_TO_KEY: dict[str, str] = {m["desktop_id"]: k for k, m in MODELS.items()}

# Back-compat: desktop id -> Bedrock system profile (used by pinned/passthrough).
MODEL_ROUTES: dict[str, str] = {m["desktop_id"]: m["system_profile"] for m in MODELS.values()}

# Route keys whose model rejects the `thinking` param Desktop's effort selector
# always sends; we strip it for these before invoking.
MODELS_WITHOUT_THINKING = {k for k, m in MODELS.items() if not m["supports_thinking"]}

# Anthropic-platform-only top-level fields that Bedrock's InvokeModel rejects with
# a ValidationException ("Extra inputs are not permitted" / "not supported for this
# model"). Claude Desktop / Cowork attach these on their `?beta=true` requests, so
# we drop them before invoking or the whole request 502s. (`anthropic_beta` and
# `cache_control` ARE accepted by Bedrock, so they're intentionally not listed.)
BEDROCK_UNSUPPORTED_FIELDS = ("betas", "context_management", "container", "mcp_servers",
                              "service_tier")

# tier <-> catalog key (tiers are contiguous ints; used by the router ladder).
TIER_TO_KEY: dict[int, str] = {m["tier"]: k for k, m in MODELS.items()}
MIN_TIER = min(TIER_TO_KEY)
MAX_TIER = max(TIER_TO_KEY)


def tier_of(key: str) -> int:
    return MODELS[key]["tier"]


def key_at_tier(tier: int) -> str:
    """Catalog key for a tier, clamped into the catalog's range."""
    return TIER_TO_KEY[max(MIN_TIER, min(MAX_TIER, tier))]


# --- Application Inference Profiles (strategy x model) -> ARN ------------------
# infra/create_inference_profiles.py writes infra/inference_profiles.json:
#   {"azr-baseline-fable5": "arn:...", "azr-auto-haiku45": "arn:...", ...}
AIP_FILE = REPO_ROOT / "infra" / "inference_profiles.json"


@functools.lru_cache(maxsize=1)
def _aip_map() -> dict[str, str]:
    if AIP_FILE.exists():
        return json.loads(AIP_FILE.read_text())
    return {}


@functools.lru_cache(maxsize=1)
def bedrock_region() -> str:
    """Region to invoke Bedrock in. Application Inference Profile ARNs are
    region-pinned, so the runtime client MUST match the region the AIPs were
    created in — a mismatch makes Bedrock reject the ARN as an invalid model
    identifier. Derive it from any AIP ARN in the map; fall back to AWS_REGION
    when no profiles exist yet (system-profile ids are cross-region)."""
    for arn in _aip_map().values():
        parts = arn.split(":")
        if len(parts) > 3 and parts[3]:
            return parts[3]
    return AWS_REGION


def aip_profile_name(strategy: str, key: str) -> str:
    """Name of the Application Inference Profile for a (strategy, model)."""
    tag = {"baseline": "baseline", "auto": "auto", "sweep": "sweep",
           "pinned": "baseline"}.get(strategy, strategy)
    return f"azr-{tag}-{key}"


def invoke_target(strategy: str, key: str) -> str:
    """Bedrock modelId to invoke: the AIP ARN if we created one, else the system
    profile id. Using the AIP is what attaches the routing-strategy cost tag."""
    return _aip_map().get(aip_profile_name(strategy, key), MODELS[key]["system_profile"])


# --- Strategy resolution ------------------------------------------------------
def resolve_strategy(requested_model: str, routing_mode_header: str | None) -> str:
    """Decide the routing strategy. Precedence per plan.md §4:
      1. requested model == claude-auto        -> auto
      2. header x-routing-mode in {baseline,auto,pinned}
      3. ROUTER_INTERCEPT_DEFAULT and requested == default model -> auto
      4. otherwise                             -> pinned
    """
    if requested_model == AUTO_MODEL_ID:
        return "auto"
    mode = (routing_mode_header or "").strip().lower()
    if mode in ("baseline", "auto", "pinned", "sweep"):
        return mode
    if ROUTER_INTERCEPT_DEFAULT and requested_model == MODELS[DEFAULT_MODEL_KEY]["desktop_id"]:
        return "auto"
    return "pinned"


def requested_key(requested_model: str) -> str:
    """Catalog key for a requested desktop model id (default when unknown/auto)."""
    if requested_model == AUTO_MODEL_ID:
        return DEFAULT_MODEL_KEY
    return DESKTOP_TO_KEY.get(requested_model, DEFAULT_MODEL_KEY)


# --- Router (interim heuristic; Phase 3 replaces this with router/) -----------
def route(strategy: str, requested_model: str, payload: dict) -> tuple[str, str]:
    """Return (catalog_key, route_reason) for a request.

    baseline -> always the default model. pinned -> the requested model.
    auto     -> heuristic pick (Phase 3 swaps in the real router package).
    """
    if strategy == "baseline":
        return DEFAULT_MODEL_KEY, "baseline"
    if strategy in ("pinned", "sweep"):
        # sweep = benchmark fingerprint pass: pin the requested model but tag the
        # spend under the azr-sweep-* AIPs (invoke_target keys off `strategy`).
        return requested_key(requested_model), strategy
    return choose_auto_model(last_user_text(payload))


def choose_auto_model(prompt: str) -> tuple[str, str]:
    """Interim heuristic router for `auto`. Returns (catalog_key, reason).

    Precedence (first match wins; keywords beat length):
      1. "think"/"sure"    -> opus5    (hardest work)
      2. "python"          -> sonnet5  (coding)
      3. < 15 words        -> haiku45  (quick answers)
      4. otherwise         -> sonnet5  (default)
    """
    low = prompt.lower()
    if "think" in low or "sure" in low:
        return "opus5", "keyword:reasoning"
    if "python" in low:
        return "sonnet5", "keyword:code"
    if len(prompt.split()) < 15:
        return "haiku45", "short-prompt"
    return "sonnet5", "default"


# --- Per-model parameter sanitation -------------------------------------------
def _effort_from_thinking(thinking: dict | None) -> str:
    """Map an Anthropic `thinking` block to a Claude 5 effort level. No block (the
    common Claude Code case) -> GATEWAY_DEFAULT_EFFORT. An explicit enabled budget
    scales low/medium/high; disabled -> low (these models can't fully disable it)."""
    if not isinstance(thinking, dict):
        return GATEWAY_DEFAULT_EFFORT
    if thinking.get("type") == "disabled":
        return "low"
    budget = thinking.get("budget_tokens", 0) or 0
    if budget <= 4096:
        return "low"
    if budget <= 16384:
        return "medium"
    return "high"


def sanitize_params(key: str, body: dict) -> dict:
    """Enforce a model's param rules on a Bedrock request body (mutates + returns).

    Fable 5: temperature must be 1.0 or unset, top_p >= 0.99, no top_k. When we
    route DOWN from Fable to a cheaper model, Fable-only fields are dropped too.
    Also normalizes extended-thinking controls to the Claude 5 Bedrock scheme:
    the Anthropic `thinking` block (which these models reject) becomes
    `output_config.effort`, an explicit effort is respected, and when a client
    sends neither we inject GATEWAY_DEFAULT_EFFORT (else the model defaults to HIGH
    and can return an empty answer). Haiku supports neither knob, so both are
    dropped for it.
    """
    # Drop Anthropic-platform-only fields Bedrock rejects (Desktop's beta payloads).
    for field in BEDROCK_UNSUPPORTED_FIELDS:
        body.pop(field, None)
    # Bedrock accepts the system prompt only as the top-level `system` field and
    # rejects messages with role "system" ("role 'system' is not supported"). Some
    # Desktop/Cowork payloads put it in the messages array, so hoist those into
    # `system` (as text blocks), preserving any existing system content first.
    msgs = body.get("messages")
    if isinstance(msgs, list) and any(
        isinstance(m, dict) and m.get("role") == "system" for m in msgs
    ):
        sys_blocks: list = []
        existing = body.get("system")
        if isinstance(existing, str) and existing:
            sys_blocks.append({"type": "text", "text": existing})
        elif isinstance(existing, list):
            sys_blocks.extend(b for b in existing if isinstance(b, dict))
        kept: list = []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                c = m.get("content")
                if isinstance(c, str):
                    sys_blocks.append({"type": "text", "text": c})
                elif isinstance(c, list):
                    sys_blocks.extend(b for b in c if isinstance(b, dict))
            else:
                kept.append(m)
        body["system"] = sys_blocks
        body["messages"] = kept
    # Server tools (web_search, code_execution, computer, bash, text_editor, …) aren't
    # available on Bedrock InvokeModel; keep only client/custom tools (an input_schema
    # or type=="custom"). Drop tool_choice too if nothing usable remains.
    tools = body.get("tools")
    if isinstance(tools, list):
        kept = [t for t in tools if isinstance(t, dict)
                and ("input_schema" in t or t.get("type") == "custom")]
        if kept:
            body["tools"] = kept
        else:
            body.pop("tools", None)
            body.pop("tool_choice", None)

    thinking = body.pop("thinking", None)
    if key in MODELS_WITHOUT_THINKING:
        body.pop("output_config", None)
    elif "output_config" not in body:
        body["output_config"] = {"effort": _effort_from_thinking(thinking)}

    rules = MODELS[key].get("param_rules", {})
    if rules.get("temperature") == "1_or_unset" and body.get("temperature") not in (None, 1, 1.0):
        body["temperature"] = 1.0
    if rules.get("top_k") == "drop":
        body.pop("top_k", None)
    if rules.get("top_p", "").startswith(">=") and "top_p" in body:
        floor = float(rules["top_p"][2:])
        if body["top_p"] < floor:
            body["top_p"] = floor
    return body


# --- Fallback ladder ----------------------------------------------------------
def next_tier(key: str) -> str | None:
    """The next model UP the tier ladder (for refusal/throttle/5xx fallback), or
    None if `key` is already the top tier."""
    higher = sorted(
        (m for m in MODELS.values() if m["tier"] > MODELS[key]["tier"]),
        key=lambda m: m["tier"],
    )
    if not higher:
        return None
    return DESKTOP_TO_KEY[higher[0]["desktop_id"]]


# --- Helpers ------------------------------------------------------------------
def resolve_model(requested: str) -> str:
    """Back-compat: Bedrock model/profile id for a requested desktop model id."""
    if requested in MODEL_ROUTES:
        return MODEL_ROUTES[requested]
    if requested.startswith(("us.", "eu.", "apac.", "anthropic.", "arn:")):
        return requested
    raise KeyError(requested)


def last_user_text(payload: dict) -> str:
    """Extract the most recent user turn's text from an Anthropic Messages body."""
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
    return ""


def session_hash(payload: dict) -> str:
    """Stable, non-PII session key: hash(system_prompt[:2k] + first_user_msg[:2k]).
    Used for request-metadata `session` and (Phase 3) sticky routing."""
    system = payload.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    first_user = ""
    for msg in payload.get("messages", []):
        if msg.get("role") == "user":
            c = msg.get("content")
            first_user = c if isinstance(c, str) else json.dumps(c)
            break
    raw = (str(system)[:2000] + "\x00" + str(first_user)[:2000]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
