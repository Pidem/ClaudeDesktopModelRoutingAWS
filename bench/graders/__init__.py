"""Automatic graders for the pharma benchmark (plan.md §6).

Each task in bench/tasks.yaml names a grader module here. Every module exposes:

    grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult

`grade` returns a GradeResult(passed, score 0..1, detail). `dispatch()` routes a
task to its grader. Graders are pure w.r.t. the model — they only see the model's
text response plus fixtures/gold on disk (and, for judge/rubric graders, a fixed
judge invoke_fn in the context)."""
from __future__ import annotations

import importlib

from .base import GradeContext, GradeResult

_GRADERS = ("sql", "json_f1", "triage", "pytest_grader", "numeric",
            "pptx", "rubric_judge", "readability")


def dispatch(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    name = task["grader"]
    if name not in _GRADERS:
        return GradeResult(False, 0.0, f"unknown grader {name!r}")
    mod = importlib.import_module(f"{__name__}.{name}")
    try:
        return mod.grade(task, response_text, ctx)
    except Exception as e:  # noqa: BLE001 - a grader crash is a fail, not a run abort
        return GradeResult(False, 0.0, f"grader {name} raised: {type(e).__name__}: {e}")


__all__ = ["dispatch", "GradeResult", "GradeContext"]
