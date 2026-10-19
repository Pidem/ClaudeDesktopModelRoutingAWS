"""Shared grader types and extraction helpers."""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GradeResult:
    passed: bool
    score: float          # 0..1 (graded quality; `passed` is the thresholded verdict)
    detail: str


@dataclass
class GradeContext:
    fixtures_dir: pathlib.Path
    gold_dir: pathlib.Path
    # A blocking judge call for rubric/readability graders: judge(prompt) -> text.
    judge_fn: Optional[Callable[[str], str]] = None


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def extract_code(text: str, lang: str | None = None) -> str:
    """Return the largest fenced code block (optionally matching a language tag),
    or the whole text if there are no fences."""
    blocks = _FENCE.findall(text or "")
    if not blocks:
        return (text or "").strip()
    if lang:
        typed = re.findall(rf"```{lang}\n(.*?)```", text, re.DOTALL)
        if typed:
            return max(typed, key=len).strip()
    return max(blocks, key=len).strip()


def extract_sql(text: str) -> str:
    code = extract_code(text, "sql")
    # Drop leading prose lines that aren't SQL if the block was the whole text.
    return code.strip().rstrip(";").strip()


def extract_json(text: str) -> Optional[dict]:
    """First parseable JSON object in the response (prefer a ```json block)."""
    block = None
    typed = re.findall(r"```json\n(.*?)```", text or "", re.DOTALL)
    if typed:
        block = typed[0]
    else:
        m = _JSON_OBJ.search(text or "")
        block = m.group(0) if m else None
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None
