"""Titan text-embeddings for the SCOPE-lite kNN fingerprint (anchors.parquet and
the router's live embed_fn). Calls Bedrock directly (embeddings don't route
through the gateway). Region matches the AIPs via config.bedrock_region()."""
from __future__ import annotations

import functools
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
EMBED_DIM = int(os.environ.get("BENCH_EMBED_DIM", "1024"))


@functools.lru_cache(maxsize=1)
def _client():
    return boto3.client("bedrock-runtime", region_name=config.bedrock_region())


def embed(text: str) -> list[float]:
    body = json.dumps({"inputText": text[:8000], "dimensions": EMBED_DIM, "normalize": True})
    resp = _client().invoke_model(modelId=EMBED_MODEL, body=body)
    return json.loads(resp["body"].read())["embedding"]
