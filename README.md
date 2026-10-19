# Claude-on-Bedrock Gateway

A minimal **Anthropic-compatible** AI gateway. Claude Desktop / Cowork sends
Anthropic Messages API requests here; the gateway routes them to Claude models
served on Amazon Bedrock and streams the responses back.

```
Claude Desktop --(Anthropic Messages API, bearer token)--> Gateway --(SigV4 / IAM role)--> Bedrock
```

## Layout

- `main.py` — FastAPI gateway: `POST /v1/messages` (stream + non-stream), `GET /v1/models`, `GET /health`.
- `config.py` — model catalog, strategy resolution, AIP resolution, param sanitation, fallback ladder, token resolution.
- `router/` — the `auto` router (rules → sticky → features → tagger → fingerprint → utility) + `policies.yaml`.
- `cost_ledger.py` — token usage × `pricing.yaml` → CloudWatch EMF + JSONL cost estimate.
- `pricing.yaml` — per-model `$/MTok` rate card (**verify against the Bedrock pricing page**).
- `Dockerfile` — container image (uv-based).
- `infra/create_inference_profiles.py` — creates the Application Inference Profiles that tag Bedrock spend by routing strategy.
- `infra/migrate_to_ecs_express.py` — deploys the container to **Amazon ECS Express Mode**.

## Models & routing

Catalog (`config.MODELS`), cheapest → most capable:

| Desktop model id | Catalog key | Bedrock system profile | Tier |
|---|---|---|---|
| `claude-haiku-4-5` | `haiku45` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 1 |
| `claude-sonnet-5` | `sonnet5` | `us.anthropic.claude-sonnet-5` | 2 |
| `claude-opus-5` | `opus5` | `us.anthropic.claude-opus-5` | 3 |
| `claude-fable-5` | `fable5` | `us.anthropic.claude-fable-5` | 4 |

Plus the virtual **`claude-auto`** model, which routes per request.

**Strategy** is resolved per request (precedence): model id `claude-auto` → `auto`;
else header `x-routing-mode: baseline|auto|pinned`; else `ROUTER_INTERCEPT_DEFAULT=true`
maps the fleet default (`claude-fable-5`) to `auto`; else `pinned` (honor the picked model).

- **baseline** → always `fable5` (the expensive default the router is pitched against).
- **auto** → the `router/` package (below).
- **pinned** → the requested model.

### Router (`router/`, strategy = auto)

Pipeline (plan.md §5), first-turn cost ≤ ~1s, ~0 on later turns:

1. **rules** (`rules.py`) — force cheapest for auxiliary calls (title/compaction);
   floor ≥ sonnet5 for long context (>150k tok), many attachments, or Cowork
   document-skill system prompts.
2. **sticky** (`sticky.py`) — a session reuses its tier and only ever escalates
   (switching models mid-session would kill the prompt cache).
3. **features** (`features.py`) — tokens, tools, code fences, attachments, keyword hits.
4. **estimator** — `tagger.py` (Haiku judge → `{family, difficulty, needs_long_context}`,
   2s timeout → keyword heuristic) + `fingerprint.py` (per-model p̂ correctness and
   ĉ cost; SCOPE-lite kNN over `bench/anchors.parquet` when it exists, else a
   family×difficulty prior).
5. **utility** (`utility.py`) — `U(m) = p̂ − λ·norm(log ĉ)`, restricted to the policy
   floor; ties break cheaper.

`policies.yaml` holds the Arch-Router-style per-family floors and the λ presets
(`quality` 0.1 / `balanced` 0.5 / `saver` 1.5). Pick a budget per request with
header `x-routing-preset: <name>` or `x-routing-alpha: <float>`. The router never
fails a request — any internal error falls back to a keyword heuristic.

On a prompt-stage refusal, throttle, or 5xx the gateway **escalates one tier** and retries
before any bytes reach the client. Each response carries `x-routed-model`, `x-route-reason`,
and (non-streaming) `x-est-cost-usd`. Enable the models in the Bedrock console (Model access)
for `us-east-1` before use.

## Cost tracking (tag Bedrock spend by strategy)

```bash
# 1. Create the Application Inference Profiles (idempotent) and write their ARNs.
AWS_PROFILE=your-profile uv run python infra/create_inference_profiles.py
# 2. Activate tag keys routing-strategy, model-tier, project in
#    Billing -> Cost allocation tags (NOT retroactive; up to 24h to appear).
# 3. Rebuild/redeploy so the gateway picks up infra/inference_profiles.json.
```

Invoking a model through its AIP attaches the `routing-strategy` cost tag, so Cost
Explorer/CUR can compare `baseline-fable5` vs `auto-router`. Until the profiles exist the
gateway falls back to the plain system profile (no tag). The live `EstimatedCostUSD`
estimate (EMF metric in namespace `AZ/ModelRouting`, dimensions `{strategy, routed_model}`)
is emitted regardless.

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
