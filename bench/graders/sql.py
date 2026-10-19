"""SQL grader: execute the model's query and the gold query on the fixture
DuckDB, pass iff the result sets match. Order-insensitive; floats rounded so
PRR/chi-square answers compare cleanly (T02, T06)."""
from __future__ import annotations

import shutil
import tempfile

import duckdb

from .base import GradeContext, GradeResult, extract_sql


def _normalize(rows, ndigits=3):
    out = set()
    for r in rows:
        out.add(tuple(round(v, ndigits) if isinstance(v, float) else v for v in r))
    return out


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    db = ctx.fixtures_dir / args["db"]
    gold_sql = (ctx.gold_dir / args["gold"]).read_text()
    model_sql = extract_sql(response_text)
    if not model_sql:
        return GradeResult(False, 0.0, "no SQL found in response")

    # Copy the DB so a model query can't mutate the shared fixture.
    with tempfile.TemporaryDirectory() as td:
        tmp = f"{td}/copy.duckdb"
        shutil.copy(db, tmp)
        con = duckdb.connect(tmp, read_only=True)
        gold = _normalize(con.execute(gold_sql).fetchall())
        try:
            got = _normalize(con.execute(model_sql).fetchall())
        except Exception as e:  # noqa: BLE001
            return GradeResult(False, 0.0, f"model SQL error: {type(e).__name__}: {str(e)[:120]}")
        finally:
            con.close()

    if got == gold:
        return GradeResult(True, 1.0, f"exact match ({len(gold)} rows)")
    inter = len(got & gold)
    denom = len(got | gold) or 1
    jac = inter / denom
    return GradeResult(False, jac,
                       f"mismatch: gold={len(gold)} got={len(got)} overlap={inter} jaccard={jac:.2f}")
