"""Benchmark runner (plan.md §6). Drives prompts through the gateway and grades
each response automatically. Three strategies:

  sweep    : each model in --models on every prompt (mode=sweep, azr-sweep-* AIPs).
             Produces the per-(task,variant,model) pass/cost matrix -> anchors +
             the always-X baselines and the Oracle.
  baseline : always fable5 (mode=baseline) -- the expensive default we pitch against.
  auto     : the router (mode=auto) at a given --preset / --alpha.

Writes one JSON line per call to bench/results/<run-id>.jsonl.

Examples:
  uv run --group bench python bench/run.py --strategy sweep \
      --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5,claude-fable-5 \
      --reps 3 --run-id sweep1
  uv run --group bench python bench/run.py --strategy auto --preset balanced --reps 3 --run-id auto_bal
  uv run --group bench python bench/run.py --strategy baseline --reps 3 --run-id base
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import yaml

BENCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))          # repo root, for `graders`/`config`
sys.path.insert(0, str(BENCH))

import client  # noqa: E402
from graders import GradeContext, dispatch  # noqa: E402

TASKS_FILE = BENCH / "tasks.yaml"
FIXTURES = BENCH / "fixtures"
GOLD = BENCH / "gold"
RESULTS = BENCH / "results"


def load_tasks(include_optional: bool, only: set[str] | None = None) -> list[dict]:
    doc = yaml.safe_load(TASKS_FILE.read_text())
    tasks = doc["tasks"]
    tasks = [t for t in tasks if include_optional or not t.get("optional")]
    if only:
        tasks = [t for t in tasks if t["id"] in only]
    return tasks


def make_judge(model: str, effort: str = "low"):
    """A fixed-judge invoke_fn bound to one model (pinned), for rubric graders.
    Low effort + headroom so the short JSON verdict isn't eaten by thinking."""
    def _judge(prompt: str) -> str:
        res = client.call([{"role": "user", "content": prompt}], model=model,
                          max_tokens=1500, mode="pinned", effort=effort)
        return res.text if res.ok else ""
    return _judge


def variants_for(task: dict, only: str | None) -> list[tuple[str, str]]:
    items = sorted(task["variants"].items())
    if only:
        items = [(k, v) for k, v in items if k.upper() == only.upper()]
    return items


def run(args) -> None:
    only = {t.strip().upper() for t in args.tasks.split(",")} if args.tasks else None
    tasks = load_tasks(args.include_optional, only)
    RESULTS.mkdir(exist_ok=True)
    outpath = RESULTS / f"{args.run_id}.jsonl"
    models = [m.strip() for m in args.models.split(",")] if args.models else [None]

    # For sweep we iterate models; for baseline/auto the model is implied by the mode.
    if args.strategy == "sweep":
        model_iter = models
    elif args.strategy == "baseline":
        model_iter = ["claude-fable-5"]           # mode=baseline forces fable5 anyway
    else:  # auto
        model_iter = ["claude-auto"]

    n = 0
    with open(outpath, "w") as out:
        for task in tasks:
            judge = make_judge(task.get("grader_args", {}).get("judge_model", args.judge_model),
                               effort=args.effort)
            ctx = GradeContext(FIXTURES, GOLD, judge_fn=judge)
            for variant, prompt in variants_for(task, args.variant):
                for model in model_iter:
                    for rep in range(args.reps):
                        mode = args.strategy if args.strategy in ("baseline", "auto", "sweep") else None
                        req_model = model if args.strategy == "sweep" else \
                            ("claude-auto" if args.strategy == "auto" else "claude-fable-5")
                        res = client.call(
                            [{"role": "user", "content": prompt}],
                            model=req_model, max_tokens=task.get("max_tokens", 800),
                            system=task.get("system"), mode=mode,
                            preset=(args.preset if args.strategy == "auto" else None),
                            alpha=(args.alpha if args.strategy == "auto" else None),
                            task_id=task["id"], run_id=args.run_id,
                            effort=args.effort, timeout=args.timeout,
                        )
                        if res.ok:
                            g = dispatch(task, res.text, ctx)
                        else:
                            from graders import GradeResult
                            g = GradeResult(False, 0.0, f"gateway error {res.status}: {res.error[:120]}")
                        row = {
                            "run_id": args.run_id, "strategy": args.strategy,
                            "task_id": task["id"], "family": task["family"],
                            "difficulty": task["difficulty"], "variant": variant, "rep": rep,
                            "requested_model": req_model, "routed_model": res.routed_model,
                            "route_reason": res.route_reason, "preset": args.preset,
                            "effort": args.effort,
                            "passed": g.passed, "score": round(g.score, 4), "detail": g.detail,
                            "est_cost": res.est_cost, "usage": res.usage,
                            "latency_s": round(res.latency_s, 3), "ok": res.ok,
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        n += 1
                        tag = res.routed_model or req_model
                        print(f"[{n}] {task['id']}{variant} rep{rep} {args.strategy} "
                              f"-> {tag} pass={g.passed} ${res.est_cost:.5f} {g.detail[:70]}")
    print(f"\nwrote {n} rows to {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["sweep", "baseline", "auto"])
    ap.add_argument("--models", default="", help="comma-sep desktop ids (sweep only)")
    ap.add_argument("--preset", default=None, help="auto: quality|balanced|saver")
    ap.add_argument("--alpha", default=None, help="auto: raw lambda float")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variant", default=None, help="restrict to one variant A|B|C")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tasks", default=None, help="restrict to task ids, e.g. T02,T08")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="output_config.effort for thinking models (Claude 5 defaults to high, "
                         "which fills max_tokens and starves the answer); low keeps the sweep tractable")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-call HTTP timeout (s)")
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--include-optional", action="store_true", help="include T11 refusal probe")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
