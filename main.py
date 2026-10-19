"""Anthropic-compatible AI gateway that routes chat to Claude models on Bedrock.

Claude Desktop / Cowork talks the Anthropic Messages API. This gateway accepts
that shape on /v1/messages, resolves a routing strategy (baseline | auto |
pinned), routes to a concrete model, invokes it through an Application Inference
Profile ARN (so Bedrock spend is tagged by routing-strategy), attaches per-request
metadata for the per-task breakdown, streams Bedrock's frames back as Anthropic
SSE, and emits a real-time cost estimate to the ledger. On refusal / throttle /
5xx it escalates one tier before any bytes reach the client. /v1/models feeds the
picker (claude-auto + the real models).
"""
from __future__ import annotations

import contextvars
import functools
import json
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

import config
import cost_ledger
import router

app = FastAPI(title="Claude-on-Bedrock Gateway")

# boto3 clients are thread-safe for calls; adaptive retries back off on Bedrock
# throttling (429) instead of failing the request outright.
_BOTO_CFG = BotoConfig(retries={"max_attempts": 4, "mode": "adaptive"})

# Per-request metadata carried into the boto3 before-sign hook so the
# X-Amzn-Bedrock-Request-Metadata header is SigV4-signed. A ContextVar lets the
# blocking invoke (run in a threadpool) see the value set on the async request.
REQ_META: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("req_meta", default=None)


def _add_meta(request, **_):
    """boto3 before-sign hook: attach the request-metadata header if one is set."""
    meta = REQ_META.get()
    if meta:
        request.headers["X-Amzn-Bedrock-Request-Metadata"] = json.dumps(meta)


def _register_meta_hook(client) -> None:
    for op in ("InvokeModel", "InvokeModelWithResponseStream"):
        client.meta.events.register(f"before-sign.bedrock-runtime.{op}", _add_meta)


@functools.lru_cache(maxsize=1)
def _default_bedrock_client():
    """Cached Bedrock client on the default chain (task role auto-refreshes)."""
    client = boto3.client("bedrock-runtime", region_name=config.bedrock_region(), config=_BOTO_CFG)
    _register_meta_hook(client)
    return client


def _bedrock_client():
    """Bedrock Runtime client — reused default-chain client, or an assumed role."""
    if config.GATEWAY_ASSUME_ROLE_ARN:
        sts = boto3.client("sts", region_name=config.bedrock_region())
        creds = sts.assume_role(
            RoleArn=config.GATEWAY_ASSUME_ROLE_ARN,
            RoleSessionName="claude-gateway",
        )["Credentials"]
        client = boto3.client(
            "bedrock-runtime",
            region_name=config.bedrock_region(),
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            config=_BOTO_CFG,
        )
        _register_meta_hook(client)
        return client
    return _default_bedrock_client()


def _check_auth(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    """Enforce the static bearer token (Desktop may send either header)."""
    expected = config.get_auth_token()
    if not expected:
        return  # auth disabled (credential kind = none)
    presented = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if presented != expected:
        raise HTTPException(status_code=401, detail="invalid gateway token")


def _haiku_invoke(client):
    """A blocking `invoke_fn(prompt) -> text` for the router's tagger, calling
    Haiku through its AIP. No request metadata is set here (REQ_META is unset
    during routing), so this housekeeping call isn't tagged as user traffic."""
    target = config.invoke_target("auto", "haiku45")

    def _fn(prompt: str) -> str:
        body = json.dumps({
            "anthropic_version": config.BEDROCK_ANTHROPIC_VERSION,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = client.invoke_model(modelId=target, body=body)
        data = json.loads(resp["body"].read())
        return "".join(b.get("text", "") for b in data.get("content", []))

    return _fn


def _to_bedrock_body(payload: dict, key: str) -> dict:
    """Anthropic Messages payload -> Bedrock InvokeModel body (as a dict).

    Strips the Anthropic-only `model`/`stream` fields, adds `anthropic_version`,
    and applies each model's parameter rules (Fable sampling, thinking support).
    """
    body = {k: v for k, v in payload.items() if k not in ("model", "stream")}
    body["anthropic_version"] = config.BEDROCK_ANTHROPIC_VERSION
    return config.sanitize_params(key, body)


def _is_retryable(err: ClientError) -> bool:
    """True for throttling / 5xx / context-length errors that a higher tier may
    survive (bigger context window, less load)."""
    code = err.response.get("Error", {}).get("Code", "")
    status = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    msg = err.response.get("Error", {}).get("Message", "").lower()
    return (
        code in ("ThrottlingException", "TooManyRequestsException",
                 "ServiceUnavailableException", "ModelErrorException",
                 "InternalServerException", "ValidationException")
        and ("context" in msg or "token" in msg or code != "ValidationException")
        or status >= 500
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    _check_auth(authorization, x_api_key)
    ids = [config.AUTO_MODEL_ID, *config.MODEL_ROUTES]
    data = [{"id": mid, "type": "model", "display_name": mid} for mid in ids]
    return {"data": data, "has_more": False}


def _estimate_tokens(payload: dict) -> int:
    """Rough token estimate (~4 chars/token) over system + messages + tools, so
    Claude Code's /v1/messages/count_tokens probe gets a usable number instead of a
    404. Only used for the client's context accounting, never for billing."""
    def _text(v) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return " ".join(_text(x) for x in v)
        if isinstance(v, dict):
            return " ".join(_text(v.get(k, "")) for k in ("text", "content", "input", "description"))
        return ""
    chars = len(_text(payload.get("system")))
    for m in payload.get("messages", []):
        chars += len(_text(m.get("content")))
    chars += len(_text(payload.get("tools")))
    return max(1, chars // 4)


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(authorization, x_api_key)
    payload = await request.json()
    return {"input_tokens": _estimate_tokens(payload)}


@app.post("/v1/messages")
async def messages(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_routing_mode: Optional[str] = Header(None),
    x_routing_alpha: Optional[str] = Header(None),
    x_routing_preset: Optional[str] = Header(None),
    x_task_id: Optional[str] = Header(None),
    x_run_id: Optional[str] = Header(None),
):
    _check_auth(authorization, x_api_key)
    payload = await request.json()

    requested = payload.get("model", "")
    strategy = config.resolve_strategy(requested, x_routing_mode)
    session = config.session_hash(payload)
    streaming = bool(payload.get("stream"))
    client = _bedrock_client()

    # Route. `auto` runs the router package (rules -> sticky -> tagger -> kNN ->
    # utility) in a worker thread so the tagger's blocking Bedrock call doesn't
    # stall the event loop; baseline/pinned are pure functions.
    route_extra: dict = {}
    if strategy == "auto":
        try:
            decision = await run_in_threadpool(
                router.route, payload,
                alpha=x_routing_alpha, preset=x_routing_preset,
                invoke_fn=_haiku_invoke(client), session=session,
            )
            key, reason = decision.model, decision.reason
            route_extra = {"family": decision.family, "difficulty": decision.difficulty,
                           "p_hat": decision.p_hat, "c_hat": decision.c_hat,
                           "lambda": decision.lam, "estimator": decision.source}
        except Exception:  # noqa: BLE001 - never fail a request on router error
            key, reason = config.choose_auto_model(config.last_user_text(payload))
            reason = f"router-error-fallback:{reason}"
    else:
        key, reason = config.route(strategy, requested, payload)

    def build_meta(routed_key: str, route_reason: str, fallback: bool) -> dict:
        # X-Amzn-Bedrock-Request-Metadata: <=16 keys, <=256-char values, no PII.
        m = {
            "strategy": strategy,
            "requested_model": requested or config.AUTO_MODEL_ID,
            "routed_model": routed_key,
            "route_reason": route_reason,
            "session": session,
        }
        if x_task_id:
            m["task_id"] = x_task_id[:256]
        if x_run_id:
            m["run_id"] = x_run_id[:256]
        if fallback:
            m["fallback"] = "true"
        return m

    # --- Non-streaming: full fallback ladder (we see stop_reason before replying).
    if not streaming:
        current, current_reason, fallback = key, reason, False
        while True:
            REQ_META.set(build_meta(current, current_reason, fallback))
            body = json.dumps(_to_bedrock_body(payload, current))
            target = config.invoke_target(strategy, current)
            try:
                resp = await run_in_threadpool(client.invoke_model, modelId=target, body=body)
            except ClientError as err:
                nxt = config.next_tier(current)
                if nxt and _is_retryable(err):
                    current, current_reason, fallback = nxt, f"fallback:{current_reason}", True
                    continue
                print(json.dumps({"bedrock_error": str(err), "routed_model": current,
                                  "strategy": strategy}), flush=True)
                raise HTTPException(status_code=502, detail=f"bedrock error: {err}") from err
            data = json.loads(resp["body"].read())
            # Prompt-stage refusal isn't billed; escalate once to a stronger tier.
            nxt = config.next_tier(current)
            if data.get("stop_reason") == "refusal" and nxt:
                current, current_reason, fallback = nxt, f"fallback:refusal", True
                continue
            cost = cost_ledger.emit(
                strategy=strategy, routed_model=current, requested_model=requested,
                reason=current_reason, usage=data.get("usage", {}), key=current,
                fallback=fallback,
                extra={"session": session, "task_id": x_task_id, **route_extra},
            )
            headers = {
                "x-routed-model": config.MODELS[current]["desktop_id"],
                "x-route-reason": current_reason,
                "x-est-cost-usd": f"{cost:.6f}",
            }
            return JSONResponse(data, headers=headers)

    # --- Streaming: only pre-stream errors can fall back (no mid-stream cascade).
    current, current_reason, fallback = key, reason, False
    while True:
        REQ_META.set(build_meta(current, current_reason, fallback))
        body = json.dumps(_to_bedrock_body(payload, current))
        target = config.invoke_target(strategy, current)
        try:
            resp = await run_in_threadpool(
                client.invoke_model_with_response_stream, modelId=target, body=body,
            )
            break
        except ClientError as err:
            nxt = config.next_tier(current)
            if nxt and _is_retryable(err):
                current, current_reason, fallback = nxt, f"fallback:{current_reason}", True
                continue
            print(json.dumps({"bedrock_error": str(err), "routed_model": current,
                              "strategy": strategy}), flush=True)
            raise HTTPException(status_code=502, detail=f"bedrock error: {err}") from err

    routed_key = current
    resp_headers = {
        "x-routed-model": config.MODELS[routed_key]["desktop_id"],
        "x-route-reason": current_reason,
    }

    def event_stream():
        # Accumulate usage across frames for the ledger: input/cache tokens land
        # in message_start, output_tokens in the final message_delta.
        usage: dict = {}
        for event in resp["body"]:
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = json.loads(chunk["bytes"])
            if data.get("type") == "message_start":
                usage.update(data.get("message", {}).get("usage", {}) or {})
            elif data.get("type") == "message_delta":
                usage.update(data.get("usage", {}) or {})
            yield f"event: {data.get('type', 'message')}\ndata: {json.dumps(data)}\n\n"
        cost_ledger.emit(
            strategy=strategy, routed_model=routed_key, requested_model=requested,
            reason=current_reason, usage=usage, key=routed_key, fallback=fallback,
            extra={"session": session, "task_id": x_task_id, **route_extra},
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=resp_headers)
