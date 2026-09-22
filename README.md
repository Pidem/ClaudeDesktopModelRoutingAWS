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
- `infra/migrate_to_ecs_express.py` — deploys the container to **Amazon ECS Express Mode**.

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

## Deploy to Amazon ECS Express Mode

**Amazon ECS Express Mode** stands up Fargate + ALB + autoscaling from one API call.
It has no CloudFormation/CDK support yet, so `infra/migrate_to_ecs_express.py` drives
the documented CLI/API flow idempotently: creates an ECR repo, builds+pushes the
image, ensures the three IAM roles (execution, infrastructure, and a task role scoped
to Bedrock + the token secret), then creates/updates the Express service.

```bash
# Docker running (colima ok), AWS creds set
AWS_PROFILE=your-profile uv run python infra/migrate_to_ecs_express.py
```

The script prints the service URL when it's serving. The service ARN is cached in
`infra/.ecs_express_service_arn` (gitignored) so re-runs update the service in place.

The gateway bearer token lives in a Secrets Manager secret named `GatewayToken*`
(injected as `GATEWAY_TOKEN_SECRET_ARN`). The script reuses an existing one or
creates it with a fresh 48-char token on first run. Read the token value:

```bash
aws secretsmanager get-secret-value --secret-id <GatewayToken secret ARN> \
  --query SecretString --output text
```

## Point Claude Desktop at it (Connection settings)

| Field | Value |
|---|---|
| **Gateway base URL** | the service URL the deploy script prints (or `http://127.0.0.1:8787` locally) |
| **Credential kind** | `token` — paste the Secrets Manager token value |
| Model discovery | on (this gateway implements `/v1/models`) |
| Model list | optional; overrides discovery, first entry = default |
| Stream idle timeout | `300` (default) |
| Custom inference headers | routing/tenant only — **no credentials** |

## Auth model (two independent hops)

1. **Desktop → Gateway**: static bearer token. In AWS it lives in Secrets Manager
   (`GATEWAY_TOKEN_SECRET_ARN`); locally, the `GATEWAY_AUTH_TOKEN` env var. If neither
   is set, auth is disabled — localhost only.
2. **Gateway → Bedrock**: AWS SigV4 via the ECS task role (least-privilege
   to the Claude models). Locally it uses your AWS profile / default chain. Never put
   AWS creds in Desktop.
