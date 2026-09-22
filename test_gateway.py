#!/usr/bin/env python3
"""Smoke-test the deployed Claude-on-Bedrock gateway.

Pulls the bearer token from Secrets Manager, then hits /health, /v1/models,
and /v1/messages (non-streaming and streaming). Exit code is non-zero if any
check fails.

Usage:
    uv run python test_gateway.py
    uv run python test_gateway.py --url https://... --model claude-opus-5

Env / defaults point at the deployed stack; override with flags or env vars.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import boto3
import httpx

# Deployed gateway URL. The ECS Express service URL is dynamic (printed by
# infra/migrate_to_ecs_express.py), so pass it via GATEWAY_URL or --url; this
# falls back to the local dev server.
DEFAULT_URL = "http://127.0.0.1:8787"
# Region the service + secret live in. Hardcoded so an ambient AWS_REGION
# (e.g. us-west-2) can't point us elsewhere; override with --region if you
# deploy to another region.
DEFAULT_REGION = "us-east-1"
# Secret name prefix for the gateway bearer token (a random suffix is appended).
SECRET_NAME_PREFIX = "GatewayToken"


def get_token(secret_arn: str, region: str) -> str:
    sm = boto3.client("secretsmanager", region_name=region)
    if not secret_arn:
        # Resolve the gateway token secret by name prefix (avoids ARN typos).
        matches = [
            s["ARN"]
            for s in sm.list_secrets(MaxResults=100).get("SecretList", [])
            if s.get("Name", "").startswith(SECRET_NAME_PREFIX)
        ]
        if not matches:
            raise SystemExit(f"no secret found with name prefix {SECRET_NAME_PREFIX!r}")
        secret_arn = matches[0]
    return sm.get_secret_value(SecretId=secret_arn)["SecretString"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("GATEWAY_URL", DEFAULT_URL))
    ap.add_argument("--secret-arn", default=os.environ.get("GATEWAY_TOKEN_SECRET_ARN", ""))
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--model", default="claude-sonnet-5")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    token = os.environ.get("GATEWAY_AUTH_TOKEN") or get_token(args.secret_arn, args.region)
    headers = {"Authorization": f"Bearer {token}", "content-type": "application/json"}
    ok = True

    with httpx.Client(timeout=120) as client:
        # 1. health
        r = client.get(f"{base}/health")
        passed = r.status_code == 200 and r.json().get("ok") is True
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] GET /health -> {r.status_code} {r.text.strip()}")

        # 2. auth is enforced
        r = client.get(f"{base}/v1/models", headers={"Authorization": "Bearer wrong"})
        passed = r.status_code == 401
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] GET /v1/models (bad token) -> {r.status_code} (expect 401)")

        # 3. model discovery
        r = client.get(f"{base}/v1/models", headers=headers)
        models = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        passed = r.status_code == 200 and args.model in models
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] GET /v1/models -> {r.status_code} {models}")

        payload = {
            "model": args.model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        }

        # 4. non-streaming message
        r = client.post(f"{base}/v1/messages", headers=headers, json=payload)
        text = ""
        if r.status_code == 200:
            text = "".join(b.get("text", "") for b in r.json().get("content", []))
        passed = r.status_code == 200 and bool(text)
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] POST /v1/messages -> {r.status_code} reply={text!r}")

        # 5. streaming message
        got_delta = False
        with client.stream("POST", f"{base}/v1/messages", headers=headers,
                           json={**payload, "stream": True}) as s:
            status = s.status_code
            for line in s.iter_lines():
                if line.startswith("data:"):
                    try:
                        evt = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") == "content_block_delta":
                        got_delta = True
        passed = status == 200 and got_delta
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] POST /v1/messages (stream) -> {status} deltas={got_delta}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
