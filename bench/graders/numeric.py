"""Numeric grader (T09): pull the model's final numeric answers for named targets
and pass iff each is within a relative tolerance of gold. The model is asked to
emit a JSON object of answers; we fall back to scanning for `key = number`."""
from __future__ import annotations

import json
import re

from .base import GradeContext, GradeResult, extract_json


def _find_number(text: str, key: str) -> float | None:
    # Match either order: "events: 508" / "N ≈ 1392" OR "508 events" / "1,392 patients".
    num = r"([0-9][0-9,\.]*)"
    for pat in (rf"{key}\D{{0,6}}{num}", rf"{num}\s+(?:total\s+)?{key}"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    gold = json.loads((ctx.gold_dir / args["gold"]).read_text())
    targets = gold["targets"]
    tol = float(gold.get("tol_rel", 0.02))

    got = extract_json(response_text) or {}
    details, n_ok = [], 0
    aliases = {"sample_size": ["sample_size", "sample size", r"\bN\b", "total N"],
               "events": ["events", "number of events", "required events"]}
    for key, gval in targets.items():
        val = got.get(key)
        if val is None:
            for alias in aliases.get(key, [key]):
                val = _find_number(response_text, alias)
                if val is not None:
                    break
        if val is None:
            details.append(f"{key}: missing")
            continue
        rel = abs(float(val) - gval) / (abs(gval) or 1)
        ok = rel <= tol
        n_ok += ok
        details.append(f"{key}: got {val} vs {gval} (rel {rel:.3f}{' OK' if ok else ' MISS'})")
    passed = n_ok == len(targets)
    return GradeResult(passed, n_ok / (len(targets) or 1), "; ".join(details))
