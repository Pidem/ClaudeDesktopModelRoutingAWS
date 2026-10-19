"""Thin client that drives prompts through the deployed gateway (not Bedrock
directly), so routing, AIP tagging, and metadata all go through the real path
(plan.md §6). Captures the response text, the x-routed-model / x-est-cost-usd
headers, token usage, and latency."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8787").rstrip("/")
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", os.environ.get("GATEWAY_AUTH_TOKEN", ""))


@dataclass
class CallResult:
    ok: bool
    status: int
    text: str = ""
    routed_model: str = ""          # desktop id from x-routed-model
    route_reason: str = ""
    est_cost: float = 0.0           # x-est-cost-usd (gateway estimate)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""


def call(messages: list[dict], *, model: str, max_tokens: int,
         system: Optional[str] = None, mode: Optional[str] = None,
         preset: Optional[str] = None, alpha: Optional[str] = None,
         task_id: Optional[str] = None, run_id: Optional[str] = None,
         effort: Optional[str] = None,
         timeout: float = 120.0) -> CallResult:
    """effort sets output_config.effort (low|medium|high) on the Bedrock Claude 5
    models. These models default to HIGH adaptive thinking, which expands to fill
    max_tokens and starves the answer (empty text on hard tasks). Pin a lower
    effort for bounded thinking, latency, and cost. Haiku ignores it."""
    headers = {"content-type": "application/json"}
    if GATEWAY_TOKEN:
        headers["Authorization"] = f"Bearer {GATEWAY_TOKEN}"
    if mode:
        headers["x-routing-mode"] = mode
    if preset:
        headers["x-routing-preset"] = preset
    if alpha:
        headers["x-routing-alpha"] = alpha
    if task_id:
        headers["x-task-id"] = task_id
    if run_id:
        headers["x-run-id"] = run_id

    body: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    if effort:
        body["output_config"] = {"effort": effort}

    t0 = time.time()
    try:
        r = httpx.post(f"{GATEWAY_URL}/v1/messages", json=body, headers=headers, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return CallResult(False, 0, error=f"{type(e).__name__}: {e}", latency_s=time.time() - t0)
    dt = time.time() - t0
    if r.status_code != 200:
        return CallResult(False, r.status_code, error=r.text[:300], latency_s=dt)

    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return CallResult(
        True, 200, text=text,
        routed_model=r.headers.get("x-routed-model", ""),
        route_reason=r.headers.get("x-route-reason", ""),
        est_cost=float(r.headers.get("x-est-cost-usd", "0") or 0),
        usage=data.get("usage", {}),
        latency_s=dt,
    )
