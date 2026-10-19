"""Plain-language grader (T04): Flesch-Kincaid grade level must be <= target
(default 8), AND a fixed judge confirms all key facts are preserved. FK is
computed locally (no dependency) with a vowel-group syllable heuristic."""
from __future__ import annotations

import json
import re

from .base import GradeContext, GradeResult

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def fk_grade(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    if not sentences or not words:
        return 99.0
    syll = sum(_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syll / len(words)) - 15.59


def _facts_ok(ctx: GradeContext, key_facts: list[str], response: str) -> tuple[int, list]:
    if ctx.judge_fn is None or not key_facts:
        return len(key_facts), []
    checklist = "\n".join(f"- {f}" for f in key_facts)
    prompt = (
        "Does the PATIENT SUMMARY preserve each key fact? Reply ONLY JSON "
        '{"preserved": <int count>, "missing": [<facts not present>]}.\n\n'
        f"KEY FACTS:\n{checklist}\n\nPATIENT SUMMARY:\n{response[:4000]}\n\nJSON:"
    )
    raw = ctx.judge_fn(prompt)
    m = _JSON.search(raw or "")
    if not m:
        return 0, ["judge returned no JSON"]
    obj = json.loads(m.group(0))
    return int(obj.get("preserved", 0)), obj.get("missing", [])


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    max_grade = float(args.get("max_grade", 8.0))
    key_facts = args.get("key_facts", [])

    grade_level = fk_grade(response_text)
    readable = grade_level <= max_grade
    preserved, missing = _facts_ok(ctx, key_facts, response_text)
    facts_ok = preserved >= len(key_facts)

    score = (0.5 if readable else 0.0) + 0.5 * (preserved / (len(key_facts) or 1))
    passed = readable and facts_ok
    return GradeResult(passed, score,
                       f"FK grade={grade_level:.1f} (<= {max_grade}: {readable}); "
                       f"facts {preserved}/{len(key_facts)} missing={missing}")
