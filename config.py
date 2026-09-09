"""Gateway configuration and model routing table.

Loads settings from environment (.env). Credentials for Bedrock come from the
AWS default credential chain (env vars, ~/.aws profile, or an IAM role). When
GATEWAY_ASSUME_ROLE_ARN is set, the gateway assumes that role via STS instead.
"""
import functools
import os

from dotenv import load_dotenv

load_dotenv()

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

# --- Model routing ------------------------------------------------------------
# Maps the model id Claude Desktop sends -> the Bedrock model id or (recommended)
# cross-region inference profile id. First entry is the default the picker uses.
# Update these ids for your enabled Bedrock models / region prefix (us./eu./apac.).
MODEL_ROUTES: dict[str, str] = {
    "claude-sonnet-5": "us.anthropic.claude-sonnet-5",
    "claude-opus-5": "us.anthropic.claude-opus-5",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


# Virtual model exposed in the picker that routes per a prompt heuristic.
AUTO_MODEL_ID = "claude-auto"

# Route keys whose Bedrock model rejects the `thinking` param (adaptive thinking).
# Desktop's effort selector always sends `thinking`; we strip it for these.
MODELS_WITHOUT_THINKING = {"claude-haiku-4-5"}


def resolve_model(requested: str) -> str:
    """Return the Bedrock model/profile id for a requested model id."""
    if requested in MODEL_ROUTES:
        return MODEL_ROUTES[requested]
    # Pass through anything already shaped like a Bedrock id/profile/ARN.
    if requested.startswith(("us.", "eu.", "apac.", "anthropic.", "arn:")):
        return requested
    raise KeyError(requested)


def choose_auto_model(prompt: str) -> str:
    """Heuristic router for claude-auto. Returns a MODEL_ROUTES key.

    Precedence (first match wins; keywords beat length):
      1. "think" or "sure"  -> claude-opus-5   (hardest work)
      2. "python"           -> claude-sonnet-5 (coding)
      3. fewer than 15 words -> claude-haiku-4-5 (quick answers)
      4. otherwise          -> claude-sonnet-5 (default)
    """
    low = prompt.lower()
    if "think" in low or "sure" in low:
        return "claude-opus-5"
    if "python" in low:
        return "claude-sonnet-5"
    if len(prompt.split()) < 15:
        return "claude-haiku-4-5"
    return "claude-sonnet-5"


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
