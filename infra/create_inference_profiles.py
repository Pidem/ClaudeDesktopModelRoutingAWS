#!/usr/bin/env python3
"""Create the Application Inference Profiles (AIPs) that tag Bedrock spend by
routing strategy, and write their ARNs to infra/inference_profiles.json.

An AIP is model-specific, so there's one per (strategy x model). Invoking a model
THROUGH its AIP is what attaches the `routing-strategy` / `model-tier` cost tags
to the Bedrock line items — that's the "this is the real bill" view in the demo.

⚠️  Cost allocation tags are NOT retroactive and can take up to 24h to appear
    after you activate them. Run this, then activate the tag keys in
    Billing -> Cost allocation tags, BEFORE any benchmark traffic.

Idempotent: existing profiles (matched by name) are reused, never duplicated.

Run:  AWS_PROFILE=<profile> uv run python infra/create_inference_profiles.py
"""
from __future__ import annotations

import json
import os
import pathlib

import boto3
from dotenv import load_dotenv

# Pick up AWS_PROFILE / AWS_REGION from the repo .env (same as config.py) so the
# script works without exporting them into the shell first.
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

REGION = os.environ.get("AWS_REGION_DEPLOY", os.environ.get("AWS_REGION", "us-east-1"))
OUT_FILE = pathlib.Path(__file__).resolve().parent / "inference_profiles.json"

br = boto3.client("bedrock", region_name=REGION)
ACCT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

# System (cross-region) inference profile ARN for a given profile id.
SYS = lambda pid: f"arn:aws:bedrock:{REGION}:{ACCT}:inference-profile/{pid}"

COMMON_TAGS = [
    {"key": "project", "value": "az-model-routing-demo"},
    {"key": "env", "value": "demo"},
    {"key": "owner", "value": "pidemal"},
]

# name -> (system profile id, routing-strategy tag, model-tier tag)
PROFILES: dict[str, tuple[str, str, str]] = {
    "azr-baseline-fable5": ("us.anthropic.claude-fable-5", "baseline-fable5", "fable"),
    "azr-auto-haiku45":    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "auto-router", "haiku"),
    "azr-auto-sonnet5":    ("us.anthropic.claude-sonnet-5", "auto-router", "sonnet"),
    "azr-auto-opus5":      ("us.anthropic.claude-opus-5", "auto-router", "opus"),
    "azr-auto-fable5":     ("us.anthropic.claude-fable-5", "auto-router", "fable"),
    # fingerprint-sweep: keep "run every model on every task" traffic out of the
    # baseline-vs-auto comparison.
    "azr-sweep-haiku45":   ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "fingerprint-sweep", "haiku"),
    "azr-sweep-sonnet5":   ("us.anthropic.claude-sonnet-5", "fingerprint-sweep", "sonnet"),
    "azr-sweep-opus5":     ("us.anthropic.claude-opus-5", "fingerprint-sweep", "opus"),
    "azr-sweep-fable5":    ("us.anthropic.claude-fable-5", "fingerprint-sweep", "fable"),
}


def existing_profiles() -> dict[str, str]:
    """name -> ARN for all APPLICATION inference profiles in the account."""
    out: dict[str, str] = {}
    paginator = br.get_paginator("list_inference_profiles") if br.can_paginate(
        "list_inference_profiles") else None
    if paginator:
        for page in paginator.paginate(typeEquals="APPLICATION"):
            for p in page["inferenceProfileSummaries"]:
                out[p["inferenceProfileName"]] = p["inferenceProfileArn"]
    else:
        for p in br.list_inference_profiles(typeEquals="APPLICATION")["inferenceProfileSummaries"]:
            out[p["inferenceProfileName"]] = p["inferenceProfileArn"]
    return out


def main() -> None:
    existing = existing_profiles()
    out: dict[str, str] = {}
    for name, (src, strategy, tier) in PROFILES.items():
        if name in existing:
            print(f"[bedrock] reusing {name}")
            out[name] = existing[name]
            continue
        resp = br.create_inference_profile(
            inferenceProfileName=name,
            modelSource={"copyFrom": SYS(src)},
            tags=COMMON_TAGS + [
                {"key": "routing-strategy", "value": strategy},
                {"key": "model-tier", "value": tier},
            ],
        )
        print(f"[bedrock] created {name}")
        out[name] = resp["inferenceProfileArn"]

    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {len(out)} profile ARNs to {OUT_FILE}")
    print("Next: activate tag keys routing-strategy, model-tier, project in "
          "Billing -> Cost allocation tags (can take up to 24h to appear).")


if __name__ == "__main__":
    main()
