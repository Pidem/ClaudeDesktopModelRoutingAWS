"""Stage 3b — per-model correctness (p̂) and cost (ĉ) estimates.

SCOPE's idea: don't just classify "which model" — predict correctness AND cost
per model, then let a utility function trade them off. Two paths:

  * ANCHORS (preferred, Phase 4): kNN over bench anchors (anchors.parquet). For
    each model, similarity-weight its pass/cost on the nearest anchors. Adding a
    model = one new column, no retraining. Activated automatically once the file
    exists and numpy/pandas are importable.
  * PRIOR (offline default): a family×difficulty→pass table plus a rate-card cost
    estimate from the token features. Keeps the router fully functional before any
    benchmark data exists; also the calibration blend term in the anchors path.
"""
from __future__ import annotations

import functools
import pathlib
from typing import Callable, Optional

import config
import cost_ledger
from .features import Features
from .tagger import TaskTag

ANCHORS_FILE = pathlib.Path(__file__).resolve().parent.parent / "bench" / "anchors.parquet"


def _pass_prior(model_tier: int, difficulty: int) -> float:
    """Coarse P(pass) prior. A model whose tier matches the task difficulty passes
    ~0.85; capability above difficulty helps a little, below it hurts fast."""
    if model_tier >= difficulty:
        return min(0.97, 0.85 + 0.03 * (model_tier - difficulty))
    return max(0.05, 0.85 - 0.25 * (difficulty - model_tier))


def _cost_prior(key: str, features: Features, expected_output: int) -> float:
    """ĉ from the rate card and token features (this model's own token counts)."""
    usage = {"input_tokens": features.tokens_in, "output_tokens": expected_output}
    return cost_ledger.compute_cost(usage, key)


def _expected_output(family: str, policies: dict) -> int:
    fam = policies.get("families", {}).get(family, {})
    return int(fam.get("expected_output", 500))


def prior_estimates(tag: TaskTag, features: Features, policies: dict,
                    keys: list[str]) -> dict[str, tuple[float, float]]:
    exp_out = _expected_output(tag.family, policies)
    out: dict[str, tuple[float, float]] = {}
    for key in keys:
        p = _pass_prior(config.tier_of(key), tag.difficulty)
        c = _cost_prior(key, features, exp_out)
        out[key] = (p, c)
    return out


@functools.lru_cache(maxsize=1)
def _load_anchors():
    """Return (df, ok). Lazy so numpy/pandas aren't hard deps until Phase 4."""
    if not ANCHORS_FILE.exists():
        return None, False
    try:
        import pandas as pd  # noqa: PLC0415
        return pd.read_parquet(ANCHORS_FILE), True
    except Exception:  # noqa: BLE001 - missing engine / bad file -> use prior
        return None, False


def _knn_estimates(query_text: str, tag: TaskTag, features: Features, policies: dict,
                   keys: list[str], embed_fn: Callable[[str], list[float]]
                   ) -> Optional[dict[str, tuple[float, float]]]:
    """SCOPE-lite kNN over anchors. Returns None if anchors/embeddings unavailable,
    so callers fall back to the prior. anchors.parquet columns expected:
    embedding (list[float]), and pass_<key> / cost_<key> per model key."""
    df, ok = _load_anchors()
    if not ok or embed_fn is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415

        q = np.asarray(embed_fn(query_text), dtype=float)
        mat = np.vstack(df["embedding"].to_numpy())
        sims = mat @ q / ((np.linalg.norm(mat, axis=1) * np.linalg.norm(q)) + 1e-9)
        k = int(policies.get("knn_k", 8))
        idx = np.argsort(-sims)[:k]
        w = np.clip(sims[idx], 0, None)
        wsum = w.sum() + 1e-9
        out: dict[str, tuple[float, float]] = {}
        for key in keys:
            pass_col, cost_col = f"pass_{key}", f"cost_{key}"
            if pass_col not in df.columns:
                # Unknown model in the anchor set -> prior for this one.
                p = _pass_prior(config.tier_of(key), tag.difficulty)
                c = _cost_prior(key, features, _expected_output(tag.family, policies))
            else:
                p = float((w * df.iloc[idx][pass_col].to_numpy()).sum() / wsum)
                c = float((w * df.iloc[idx][cost_col].to_numpy()).sum() / wsum)
            out[key] = (p, c)
        return out
    except Exception:  # noqa: BLE001 - any numeric/data issue -> prior
        return None


def estimates(tag: TaskTag, features: Features, policies: dict, keys: list[str],
              query_text: str = "", embed_fn: Optional[Callable[[str], list[float]]] = None,
              ) -> tuple[dict[str, tuple[float, float]], str]:
    """Return ({key: (p_hat, c_hat)}, source). source is 'anchors' or 'prior'."""
    knn = _knn_estimates(query_text, tag, features, policies, keys, embed_fn) if embed_fn else None
    if knn is not None:
        return knn, "anchors"
    return prior_estimates(tag, features, policies, keys), "prior"
