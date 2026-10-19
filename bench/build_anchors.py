"""Turn a sweep results file into bench/anchors.parquet — the SCOPE-lite kNN
index the router's fingerprint uses. One row per (task, variant): the embedded
prompt plus pass_<key> and cost_<key> for each model (averaged over reps).

Leakage control (plan.md §6): pass --variants A,B to build the index from only
those variants so the router can be evaluated on the held-out variant C.

  uv run --group bench python bench/build_anchors.py --results bench/results/sweep1.jsonl
  uv run --group bench python bench/build_anchors.py --results bench/results/sweep1.jsonl --variants A,B --out bench/anchors_AB.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import pandas as pd
import yaml

BENCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))
sys.path.insert(0, str(BENCH))

import config  # noqa: E402
import embed as embedder  # noqa: E402

TASKS = yaml.safe_load((BENCH / "tasks.yaml").read_text())["tasks"]
TASK_BY_ID = {t["id"]: t for t in TASKS}
KEYS = list(config.MODELS.keys())          # haiku45, sonnet5, opus5, fable5


def query_text(task: dict, variant: str) -> str:
    """Mirror the router's kNN query: system[:500] + '\\n' + user text."""
    system = (task.get("system") or "")[:500]
    return f"{system}\n{task['variants'][variant]}"


def build(results_path: pathlib.Path, variants: set[str] | None, out: pathlib.Path) -> None:
    # (task_id, variant) -> key -> {"pass": [...], "cost": [...]}
    agg: dict = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"pass": [], "cost": []}))
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["strategy"] != "sweep" or not r.get("ok"):
            continue
        if variants and r["variant"] not in variants:
            continue
        key = config.DESKTOP_TO_KEY.get(r["requested_model"])
        if key is None:
            continue
        cell = agg[(r["task_id"], r["variant"])][key]
        cell["pass"].append(1.0 if r["passed"] else 0.0)
        cell["cost"].append(float(r["est_cost"]))

    rows = []
    for (task_id, variant), per_key in sorted(agg.items()):
        task = TASK_BY_ID[task_id]
        row = {"task_id": task_id, "variant": variant,
               "family": task["family"], "difficulty": task["difficulty"],
               "embedding": embedder.embed(query_text(task, variant))}
        for key in KEYS:
            c = per_key.get(key)
            row[f"pass_{key}"] = (sum(c["pass"]) / len(c["pass"])) if c and c["pass"] else None
            row[f"cost_{key}"] = (sum(c["cost"]) / len(c["cost"])) if c and c["cost"] else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"wrote {out} — {len(df)} anchors, columns: {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--variants", default=None, help="comma-sep subset, e.g. A,B (leakage control)")
    ap.add_argument("--out", default=str(BENCH / "anchors.parquet"))
    a = ap.parse_args()
    variants = {v.strip().upper() for v in a.variants.split(",")} if a.variants else None
    build(pathlib.Path(a.results), variants, pathlib.Path(a.out))


if __name__ == "__main__":
    main()
