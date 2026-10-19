"""Stage 3a — task tagger.

Preferred path: a Haiku 4.5 judge with the policy families in its prompt returns
strict JSON {family, difficulty, needs_long_context}. It's a cheap auxiliary call
with a hard 2s timeout; on timeout / bad JSON / no client we fall back to a
keyword+feature heuristic so the router never blocks on the tagger.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .features import Features

FAMILIES = ("sql", "deck", "summarize", "extract", "pv", "biostat",
            "code", "litreview", "design", "chat")

# Shared executor so the timeout wrapper doesn't spawn a thread per request.
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tagger")
_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class TaskTag:
    family: str
    difficulty: int          # 1..5
    needs_long_context: bool
    source: str              # "haiku" | "heuristic"


def _prompt(system_text: str, user_text: str) -> str:
    fams = ", ".join(FAMILIES)
    return (
        "You are a routing classifier. Read the user's task and reply with ONLY a "
        "JSON object, no prose. Schema:\n"
        '{"family": one of [' + fams + '], '
        '"difficulty": integer 1-5, "needs_long_context": boolean}\n'
        "difficulty: 1 trivial, 3 needs a capable model, 5 expert/agentic.\n\n"
        f"SYSTEM PROMPT (may be empty):\n{system_text[:1500]}\n\n"
        f"USER TASK:\n{user_text[:2500]}\n\nJSON:"
    )


def _parse(text: str) -> Optional[TaskTag]:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    fam = str(obj.get("family", "")).lower()
    if fam not in FAMILIES:
        return None
    try:
        diff = int(obj.get("difficulty", 3))
    except (TypeError, ValueError):
        diff = 3
    return TaskTag(fam, max(1, min(5, diff)), bool(obj.get("needs_long_context")), "haiku")


def heuristic_tag(features: Features) -> TaskTag:
    """Keyword + feature fallback. Family = strongest keyword hit; difficulty from
    size/code/tools; long context from token count."""
    hits = features.keyword_hits
    family = max(hits, key=hits.get) if hits and max(hits.values()) > 0 else "chat"
    difficulty = 1
    if features.code_fences or features.n_tools or family in ("code", "sql", "deck", "pv"):
        difficulty = 3
    if family in ("biostat", "litreview"):
        difficulty = 4
    if family == "design":
        difficulty = 5
    if features.tokens_in > 8000:
        difficulty = min(5, difficulty + 1)
    return TaskTag(family, difficulty, features.tokens_in >= 100_000, "heuristic")


def tag(features: Features, invoke_fn: Optional[Callable[[str], str]] = None,
        timeout: float = 2.0) -> TaskTag:
    """Classify a task. Uses the Haiku judge when `invoke_fn` is given, else the
    heuristic. `invoke_fn(prompt) -> completion_text` should call Haiku."""
    if invoke_fn is None:
        return heuristic_tag(features)
    fut = _EXEC.submit(invoke_fn, _prompt(features.system_text, features.user_text))
    try:
        result = _parse(fut.result(timeout=timeout))
    except (concurrent.futures.TimeoutError, Exception):  # noqa: BLE001 - degrade to heuristic
        result = None
    return result or heuristic_tag(features)
