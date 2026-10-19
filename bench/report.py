"""Analysis + demo artifacts (plan.md §7). Reads sweep + baseline + auto result
files and produces:
  * a per-strategy metrics table (pass rate, $/task, cost per passed task, latency
    p50/p95, fallback rate),
  * always-X / random / Oracle baselines derived from the sweep,
  * PGR (Performance Gap Recovered) for each router preset,
  * a Pareto plot (bench/report/pareto.png): $/task (log x) vs pass rate,
  * a back-of-envelope monthly-savings extrapolation.

  uv run --group bench python bench/report.py --sweep bench/results/sweep1.jsonl \
      --baseline bench/results/base.jsonl \
      --auto bench/results/auto_quality.jsonl,bench/results/auto_balanced.jsonl,bench/results/auto_saver.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

BENCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))
import config  # noqa: E402

TIER_ORDER = ["haiku45", "sonnet5", "opus5", "fable5"]      # cheapest -> priciest
REPORT_DIR = BENCH / "report"


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]


def _pctl(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def summarize(rows: list[dict], label: str) -> dict:
    passed = [r for r in rows if r["passed"]]
    costs = [r["est_cost"] for r in rows]
    lat = [r["latency_s"] for r in rows]
    total_cost = sum(costs)
    n = len(rows)
    fb = sum(1 for r in rows if "fallback" in (r.get("route_reason") or ""))
    return {
        "strategy": label, "n": n,
        "pass_rate": len(passed) / n if n else 0.0,
        "cost_per_task": total_cost / n if n else 0.0,
        "cost_per_passed": (total_cost / len(passed)) if passed else float("inf"),
        "lat_p50": _pctl(lat, 50), "lat_p95": _pctl(lat, 95),
        "fallback_rate": fb / n if n else 0.0,
    }


def sweep_baselines(sweep: list[dict]) -> tuple[dict, dict]:
    """Return ({model_key: summary}, oracle_summary) from the sweep rows."""
    by_model = collections.defaultdict(list)
    for r in sweep:
        key = config.DESKTOP_TO_KEY.get(r["requested_model"])
        if key:
            by_model[key].append(r)
    model_summ = {k: summarize(v, f"always-{k}") for k, v in by_model.items()}

    # Oracle: per (task, variant), cheapest tier that passes a majority of reps.
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sweep:
        key = config.DESKTOP_TO_KEY.get(r["requested_model"])
        if key:
            cell[(r["task_id"], r["variant"])][key].append(r)
    oracle_rows = []
    for tv, per_key in cell.items():
        chosen = None
        for key in TIER_ORDER:
            rs = per_key.get(key, [])
            if rs and sum(x["passed"] for x in rs) / len(rs) >= 0.5:
                chosen = key
                break
        if chosen is None:  # nothing passed -> take priciest, mark fail
            chosen = TIER_ORDER[-1]
        rs = per_key[chosen]
        oracle_rows.append({
            "passed": sum(x["passed"] for x in rs) / len(rs) >= 0.5,
            "est_cost": statistics.mean(x["est_cost"] for x in rs),
            "latency_s": statistics.mean(x["latency_s"] for x in rs),
            "route_reason": f"oracle:{chosen}",
        })
    return model_summ, summarize(oracle_rows, "oracle")


def pgr(auto_pass: float, cheapest_pass: float, best_pass: float) -> float:
    denom = best_pass - cheapest_pass
    return (auto_pass - cheapest_pass) / denom if denom > 1e-9 else float("nan")


def print_table(summaries: list[dict]) -> None:
    cols = ["strategy", "n", "pass_rate", "cost_per_task", "cost_per_passed",
            "lat_p50", "lat_p95", "fallback_rate"]
    w = {c: max(len(c), *(len(_fmt(s.get(c))) for s in summaries)) for c in cols}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for s in summaries:
        print(" | ".join(_fmt(s.get(c)).ljust(w[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:.4f}" if v < 1 else f"{v:.2f}"
    return str(v)


def pareto_plot(model_summ: dict, auto_summ: list[dict], oracle: dict, out: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping plot: {e})")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, s in model_summ.items():
        ax.scatter(s["cost_per_task"], s["pass_rate"], s=80, color="#888")
        ax.annotate(key, (s["cost_per_task"], s["pass_rate"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    auto_sorted = sorted(auto_summ, key=lambda s: s["cost_per_task"])
    ax.plot([s["cost_per_task"] for s in auto_sorted],
            [s["pass_rate"] for s in auto_sorted],
            "-o", color="#0a7", label="router (λ presets)")
    for s in auto_sorted:
        ax.annotate(s["strategy"].replace("auto-", "λ:"),
                    (s["cost_per_task"], s["pass_rate"]), fontsize=8,
                    color="#0a7", xytext=(4, -10), textcoords="offset points")
    ax.scatter(oracle["cost_per_task"], oracle["pass_rate"], marker="*", s=260,
               color="gold", edgecolor="k", zorder=5, label="oracle")
    ax.set_xscale("log")
    ax.set_xlabel("cost per task ($, log)")
    ax.set_ylabel("pass rate")
    ax.set_title("Cost vs quality: single models, router, oracle")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def extrapolate(model_summ: dict, auto_balanced: dict | None) -> None:
    fable = model_summ.get("fable5")
    if not fable or not auto_balanced:
        return
    per_task_saving = fable["cost_per_task"] - auto_balanced["cost_per_task"]
    pct = per_task_saving / fable["cost_per_task"] if fable["cost_per_task"] else 0
    print(f"\nExtrapolation (balanced router vs all-Fable): "
          f"${per_task_saving:.5f}/task saved ({pct*100:.1f}%). "
          f"At 1M requests/mo ≈ ${per_task_saving*1_000_000:,.0f}/mo "
          f"(replace with AZ family-mix weights when available).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--auto", default="", help="comma-sep auto result files")
    a = ap.parse_args()

    sweep = _load(a.sweep)
    model_summ, oracle = sweep_baselines(sweep)

    summaries = list(model_summ.values()) + [oracle]
    if a.baseline:
        summaries.append(summarize(_load(a.baseline), "baseline-fable5"))
    auto_summ = []
    for f in [x for x in a.auto.split(",") if x.strip()]:
        rows = _load(f)
        preset = rows[0].get("preset") or pathlib.Path(f).stem
        s = summarize(rows, f"auto-{preset}")
        auto_summ.append(s)
        summaries.append(s)

    print_table(summaries)

    cheapest = model_summ.get("haiku45", {}).get("pass_rate", 0.0)
    best = max((s["pass_rate"] for s in model_summ.values()), default=1.0)
    print("\nPGR (Performance Gap Recovered) vs always-haiku..best:")
    for s in auto_summ:
        print(f"  {s['strategy']}: PGR={pgr(s['pass_rate'], cheapest, best):.2f} "
              f"(pass {s['pass_rate']:.2f}, ${s['cost_per_task']:.5f}/task)")

    pareto_plot(model_summ, auto_summ, oracle, REPORT_DIR / "pareto.png")
    bal = next((s for s in auto_summ if "balanced" in s["strategy"]), None)
    extrapolate(model_summ, bal)


if __name__ == "__main__":
    main()
