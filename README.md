# Claude-on-Bedrock Gateway

A minimal **Anthropic-compatible** AI gateway. Claude Desktop / Cowork sends
Anthropic Messages API requests here; the gateway routes them to Claude models
served on Amazon Bedrock and streams the responses back.

```
Claude Desktop --(Anthropic Messages API, bearer token)--> Gateway --(SigV4 / IAM role)--> Bedrock
```

## Layout

- `main.py` — FastAPI gateway: `POST /v1/messages` (stream + non-stream), `GET /v1/models`, `GET /health`.
- `config.py` — model routing table + token resolution (Secrets Manager or env).
- `Dockerfile` — container image (uv-based).
- `infra/` — AWS CDK (Python) app that deploys to **App Runner**.

## Models routed

| Desktop model id | Bedrock inference profile |
|---|---|
| `claude-sonnet-5` (default) | `us.anthropic.claude-sonnet-5` |
| `claude-opus-5` | `us.anthropic.claude-opus-5` |
| `claude-haiku-4-5` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |

Enable these models in the Bedrock console (Model access) for `us-east-1` before use.
Edit `MODEL_ROUTES` in `config.py` to change them.

## Run locally (uv)

```bash
uv sync                                   # create .venv from uv.lock
export GATEWAY_AUTH_TOKEN=testtoken       # local auth (skip Secrets Manager)
export AWS_PROFILE=your-profile AWS_REGION=us-east-1
uv run uvicorn main:app --host 127.0.0.1 --port 8787
```

Smoke test:

```bash
curl -s http://127.0.0.1:8787/v1/models -H "Authorization: Bearer $GATEWAY_AUTH_TOKEN"

curl -s http://127.0.0.1:8787/v1/messages \
  -H "Authorization: Bearer $GATEWAY_AUTH_TOKEN" -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}'
```

## Deploy to AWS App Runner (CDK)

Prereqs: Docker running, AWS creds, and a one-time `cdk bootstrap` per account/region.

```bash
cd infra
uv venv .venv && uv pip install --python .venv -r requirements.txt
source .venv/bin/activate       # Python deps for the CDK app

pnpm install                    # installs the aws-cdk CLI locally

export AWS_PROFILE=your-profile
pnpm cdk bootstrap              # first time only
pnpm cdk deploy
```

The stack creates:
- an **App Runner service** running the container (HTTPS URL, autoscaling),
- an **instance role** scoped to `bedrock:InvokeModel*` on the Claude profiles/models,
- a **Secrets Manager secret** holding the gateway bearer token (auto-generated),
  injected as `GATEWAY_TOKEN_SECRET_ARN`.

Outputs: `ServiceUrl` (your Gateway base URL) and `GatewayTokenSecretArn`.
Read the token value:

```bash
aws secretsmanager get-secret-value --secret-id <GatewayTokenSecretArn> \
  --query SecretString --output text
```

## Point Claude Desktop at it (Connection settings)

| Field | Value |
|---|---|
| **Gateway base URL** | the `ServiceUrl` output (or `http://127.0.0.1:8787` locally) |
| **Credential kind** | `token` — paste the Secrets Manager token value |
| Model discovery | on (this gateway implements `/v1/models`) |
| Model list | optional; overrides discovery, first entry = default |
| Stream idle timeout | `300` (default) |
| Custom inference headers | routing/tenant only — **no credentials** |

## Auth model (two independent hops)

1. **Desktop → Gateway**: static bearer token. In AWS it lives in Secrets Manager
   (`GATEWAY_TOKEN_SECRET_ARN`); locally, the `GATEWAY_AUTH_TOKEN` env var. If neither
   is set, auth is disabled — localhost only.
2. **Gateway → Bedrock**: AWS SigV4 via the App Runner instance role (least-privilege
   to the Claude models). Locally it uses your AWS profile / default chain. Never put
   AWS creds in Desktop.
