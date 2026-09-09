"""Anthropic-compatible AI gateway that routes chat to Claude models on Bedrock.

Claude Desktop / Cowork talks the Anthropic Messages API. This gateway accepts
that shape on /v1/messages, rewrites it for Bedrock (drop `model`, add
`anthropic_version`, model id -> the URL), signs with SigV4 (via boto3), and
re-emits Bedrock's streaming frames as Anthropic SSE. /v1/models feeds the picker.
"""
from __future__ import annotations

import json
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import config

app = FastAPI(title="Claude-on-Bedrock Gateway")


def _bedrock_client():
    """Bedrock Runtime client using the default chain, or an assumed role."""
    boto_cfg = BotoConfig(retries={"max_attempts": 2, "mode": "standard"})
    if config.GATEWAY_ASSUME_ROLE_ARN:
        sts = boto3.client("sts", region_name=config.AWS_REGION)
        creds = sts.assume_role(
            RoleArn=config.GATEWAY_ASSUME_ROLE_ARN,
            RoleSessionName="claude-gateway",
        )["Credentials"]
        return boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            config=boto_cfg,
        )
    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION, config=boto_cfg)


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


def _to_bedrock_body(payload: dict, routed: str) -> str:
    """Anthropic Messages payload -> Bedrock InvokeModel body."""
    body = {k: v for k, v in payload.items() if k not in ("model", "stream")}
    body["anthropic_version"] = config.BEDROCK_ANTHROPIC_VERSION
    # Some models (e.g. Haiku) reject the `thinking` param Desktop's effort
    # selector sends — drop it for those so requests don't 400 on Bedrock.
    if "thinking" in body and routed in config.MODELS_WITHOUT_THINKING:
        body.pop("thinking")
    return json.dumps(body)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    _check_auth(authorization, x_api_key)
    ids = [config.AUTO_MODEL_ID, *config.MODEL_ROUTES]
    data = [{"id": mid, "type": "model", "display_name": mid} for mid in ids]
    return {"data": data, "has_more": False}


@app.post("/v1/messages")
async def messages(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(authorization, x_api_key)
    payload = await request.json()

    requested = payload.get("model", "")
    # Virtual auto model: pick a concrete model from the prompt heuristic.
    routed = requested
    if requested == config.AUTO_MODEL_ID:
        routed = config.choose_auto_model(config.last_user_text(payload))
    try:
        bedrock_model = config.resolve_model(routed)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown model: {requested!r}")

    body = _to_bedrock_body(payload, routed)
    client = _bedrock_client()
    # Surface the routing decision for observability (visible in curl/-v).
    resp_headers = {"x-gateway-routed-model": routed}

    if not payload.get("stream"):
        resp = client.invoke_model(modelId=bedrock_model, body=body)
        return JSONResponse(json.loads(resp["body"].read()), headers=resp_headers)

    def event_stream():
        resp = client.invoke_model_with_response_stream(modelId=bedrock_model, body=body)
        for event in resp["body"]:
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = json.loads(chunk["bytes"])
            # Bedrock chunks already carry the Anthropic event shape (`type`).
            yield f"event: {data.get('type', 'message')}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=resp_headers)
