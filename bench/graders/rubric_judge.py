"""LLM-judge rubric grader (T01, T10). A fixed judge model (Sonnet/Opus, set by
the harness via ctx.judge_fn) scores the response against a rubric and must-have
checklist, returning strict JSON {score: 0-1, missing: [...]}. Pass at the task's
threshold. Judge hygiene (plan.md §6): fixed judge, rubric threshold, 10% human
spot-check recommended."""
from __future__ import annotations

import json
import re

from .base import GradeContext, GradeResult

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _judge_prompt(rubric: str, must_include: list[str], response: str) -> str:
    checklist = "\n".join(f"- {m}" for m in must_include) or "- (none)"
    return (
        "You are a strict grader for a pharma task. Score the RESPONSE against the "
        "rubric and the must-include checklist. Reply with ONLY JSON:\n"
        '{"score": <float 0-1>, "missing": [<checklist items not satisfied>]}\n\n'
        f"RUBRIC:\n{rubric}\n\nMUST INCLUDE:\n{checklist}\n\n"
        f"RESPONSE:\n{response[:6000]}\n\nJSON:"
    )


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    if ctx.judge_fn is None:
        return GradeResult(False, 0.0, "no judge_fn configured")
    rubric = args.get("rubric", "Score overall quality and correctness.")
    must = args.get("must_include", [])
    threshold = float(args.get("threshold", 0.7))

    raw = ctx.judge_fn(_judge_prompt(rubric, must, response_text))
    m = _JSON.search(raw or "")
    if not m:
        return GradeResult(False, 0.0, f"judge returned no JSON: {raw[:120]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return GradeResult(False, 0.0, f"judge JSON parse error: {raw[:120]!r}")
    score = float(obj.get("score", 0.0))
    missing = obj.get("missing", [])
    return GradeResult(score >= threshold, score,
                       f"judge score={score:.2f} (threshold {threshold}); missing={missing}")
