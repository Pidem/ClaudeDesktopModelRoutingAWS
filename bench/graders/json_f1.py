"""JSON extraction grader (T03): field-level F1 between the model's JSON and the
gold JSON. Scalars score 1 if they match after normalization; list fields score
by set-overlap F1. Overall = mean field F1; pass at >= threshold (default 0.9)."""
from __future__ import annotations

import json
import re

from .base import GradeContext, GradeResult, extract_json


def _norm(v):
    s = str(v).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _scalar_f1(gold, got) -> float:
    return 1.0 if _norm(gold) == _norm(got) else 0.0


def _list_f1(gold: list, got) -> float:
    if not isinstance(got, list):
        got = [got]
    g, h = {_norm(x) for x in gold}, {_norm(x) for x in got}
    if not g and not h:
        return 1.0
    tp = len(g & h)
    prec = tp / (len(h) or 1)
    rec = tp / (len(g) or 1)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    gold = json.loads((ctx.gold_dir / args["gold"]).read_text())
    threshold = float(args.get("threshold", 0.9))
    got = extract_json(response_text)
    if got is None:
        return GradeResult(False, 0.0, "no parseable JSON in response")

    scores = {}
    for field, gval in gold.items():
        hval = got.get(field)
        if hval is None:
            scores[field] = 0.0
        elif isinstance(gval, list):
            scores[field] = _list_f1(gval, hval)
        else:
            scores[field] = _scalar_f1(gval, hval)
    mean = sum(scores.values()) / (len(scores) or 1)
    missed = [f for f, s in scores.items() if s < 1.0]
    return GradeResult(mean >= threshold, mean,
                       f"field F1={mean:.2f} (threshold {threshold}); weak fields: {missed}")
