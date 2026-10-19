"""AE triage grader (T05): exact match on the six ICH E2A seriousness booleans
and the MedDRA Preferred Term must be in the accepted list. Pass iff all
booleans correct AND PT accepted."""
from __future__ import annotations

import json

from .base import GradeContext, GradeResult, extract_json


def _norm(s: str) -> str:
    return str(s).strip().lower()


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    gold = json.loads((ctx.gold_dir / args["gold"]).read_text())
    got = extract_json(response_text)
    if got is None:
        return GradeResult(False, 0.0, "no parseable JSON in response")

    gold_flags = gold["seriousness"]
    got_flags = got.get("seriousness", {}) or {}
    correct = sum(1 for k, v in gold_flags.items()
                  if bool(got_flags.get(k)) == bool(v))
    n = len(gold_flags)

    accepted = {_norm(x) for x in gold["preferred_term_accepted"]}
    pt = _norm(got.get("preferred_term", ""))
    pt_ok = pt in accepted

    flags_ok = correct == n
    score = (correct / n) * (1.0 if pt_ok else 0.5)
    passed = flags_ok and pt_ok
    return GradeResult(passed, score,
                       f"seriousness {correct}/{n} correct; PT {got.get('preferred_term')!r} "
                       f"{'accepted' if pt_ok else 'NOT accepted'}")
