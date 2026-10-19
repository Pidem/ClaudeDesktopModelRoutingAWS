#!/usr/bin/env bash
# Launch Claude Code pointed at the Claude-on-Bedrock routing gateway instead of
# talking to Bedrock directly.
#
#   ./use-gateway.sh                 # interactive Claude Code through the gateway
#   ./use-gateway.sh -p "hello"      # any claude args are passed through
#
# Run this in a FRESH top-level terminal, NOT inside an existing Claude Code
# session (a nested/child session ignores ANTHROPIC_BASE_URL and routes through
# its parent).
#
# The ASBX Claude Code build sets CLAUDE_CODE_USE_BEDROCK=1 and pins model ids,
# which force direct Bedrock and make it ignore ANTHROPIC_BASE_URL. We clear those
# so it uses the Anthropic-API path against the gateway.
#
# Override the target with GATEWAY_URL / GATEWAY_TOKEN before running.
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8787}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-localtest}"

exec env \
  -u CLAUDE_CODE_USE_BEDROCK \
  -u AWS_BEARER_TOKEN_BEDROCK \
  -u ANTHROPIC_DEFAULT_OPUS_MODEL \
  -u ANTHROPIC_DEFAULT_SONNET_MODEL \
  -u ANTHROPIC_DEFAULT_HAIKU_MODEL \
  -u ANTHROPIC_DEFAULT_FABLE_MODEL \
  ANTHROPIC_BASE_URL="$GATEWAY_URL" \
  ANTHROPIC_AUTH_TOKEN="$GATEWAY_TOKEN" \
  ANTHROPIC_MODEL="claude-auto" \
  ANTHROPIC_SMALL_FAST_MODEL="claude-haiku-4-5" \
  claude "$@"
