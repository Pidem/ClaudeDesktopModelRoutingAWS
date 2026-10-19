---
marp: true
title: Claude-on-Bedrock Routing Logic
paginate: true
---

# Claude-on-Bedrock Routing

**One virtual model in, the cheapest model that can do the job out.**

Clients talk the Anthropic Messages API to a gateway. When they ask for
`claude-auto`, the router picks a concrete Claude model on Bedrock per request —
trading answer quality against cost on an explicit budget knob.

`gateway (main.py)` → `router.route()` → Bedrock (via Application Inference Profile)

---

## The model catalog

Four tiers, cheapest → strongest. The tier ordering *is* the fallback ladder.

| Tier | Key | Model | Context | Notes |
|------|-----|-------|---------|-------|
| 1 | `haiku45` | Claude Haiku 4.5 | 200K | no thinking / no `effort` |
| 2 | `sonnet5` | Claude Sonnet 5 | 1M | long-context floor |
| 3 | `opus5`  | Claude Opus 5  | 1M | hardest work |
| 4 | `fable5` | Claude Fable 5 | 1M | expensive default (baseline) |

`fable5` is what a user gets if they never touch the picker — the cost the router
is pitched against.

---

## Three strategies (resolved per request)

- **baseline** — always `fable5`. The "did nothing" cost we compare to.
- **pinned** — send exactly the requested model (passthrough).
- **auto** — run the router pipeline. Triggered by `model: claude-auto`, the
  `x-routing-mode: auto` header, or (optionally) intercepting the default model.

Only **auto** engages the routing logic in the next slides. Spend on every path is
tagged by `routing-strategy` via per-(strategy × model) Application Inference
Profiles, so cost is attributable in Cost Explorer.

---

## The auto pipeline — 5 stages

```
payload
  │
  ├─ 0. Hard rules ........ force a model, or raise a tier FLOOR   (µs)
  │
  ├─ 1. Session stickiness . reuse the session's tier, skip estimator
  │
  ├─ 2. Features ........... cheap request signals (µs–ms)
  │
  ├─ 3. Estimator .......... Haiku tagger + p̂/ĉ per model
  │
  └─ 4. Utility ............ argmax  p̂ − λ·cost  → chosen model
```

Only the **first turn** of a session pays for the estimator. Post-generation
fallback (refusal / throttle / 5xx) lives in the gateway, not here.

---

## Stage 0 — Hard rules (no model calls)

Rules either **force** a model (skip everything downstream) or raise a **floor**
the later stages must respect.

- **Auxiliary calls** → cheapest tier (`haiku45`).
  Title generation, compaction, short classification, or tiny completions
  (`max_tokens ≤ 96`, no tools, small input).
- **Long context** (≥ 150K tokens) → floor `sonnet5`.
- **Heavy attachments** (> 4 images/docs) → floor `sonnet5`.
- **Cowork document skills** (`python-pptx`, `.docx`, "presentation skill", …)
  in the system prompt → floor `sonnet5`.

Fast, deterministic, and cheap to audit.

---

## Stage 1 — Session stickiness

Switching models mid-session **kills the prompt cache** (cache is per model).

- First turn lands on a tier → cache it (in-proc TTL LRU, 30 min).
- Later turns **reuse** that tier and **skip the estimator** entirely.
- Escalate-only: a session can move *up* a tier, never back down.

Result: ≈0 added latency on follow-up turns, and the prompt cache stays warm.
(In-process today; swap for ElastiCache to scale horizontally.)

---

## Stage 2 — Features

Cheap signals extracted from the payload (all µs–ms, no model calls):

- Rough input token count, turn count, tool count, tool-result presence
- Code-fence count, image / document counts
- `max_tokens`, thinking flag
- Per-family **keyword hits** (sql, deck, code, biostat, design, …)

These feed the hard rules, the estimator's cost prior, and the tagger's offline
fallback.

---

## Stage 3 — Estimator: tag the task

**Task tagger** — a Haiku 4.5 judge classifies the request into strict JSON:

```json
{"family": "code", "difficulty": 3, "needs_long_context": false}
```

- 10 families (sql, deck, summarize, extract, pv, biostat, code, litreview,
  design, chat); difficulty 1–5.
- Hard **2 s timeout**; on timeout / bad JSON / no client it degrades to a
  keyword + feature **heuristic** — the router never blocks on the tagger.
- Each family carries a policy **floor** from `policies.yaml`
  (e.g. `design → tier 3`, `sql → tier 2`).

---

## Stage 3 — Estimator: predict per model

Don't just pick a model — predict **correctness (p̂)** and **cost (ĉ)** for
*every* model, then let utility trade them off (SCOPE-style).

- **Prior (default):** a `tier × difficulty` pass table + a rate-card cost
  estimate from token features. Works with zero benchmark data.
- **Anchors (preferred):** kNN over benchmarked examples
  (`bench/anchors.parquet`) — similarity-weight each model's real pass/cost on
  the nearest anchors. Auto-activates when the file exists.

Adding a model = one new column, no retraining.

---

## Stage 4 — Utility decision

$$ U(m) = \hat{p}(m) \;-\; \lambda \cdot \text{norm}\big(\log \hat{c}(m)\big) $$

- Cost is **log-scaled then min-max normalized** across candidates, so λ means
  the same thing regardless of the absolute price spread.
- Candidates restricted to **tier ≥ floor** (from rules + family policy).
- **argmax U**; ties break to the **cheaper** model.

**λ = the budget knob**, chosen per request:

| Preset | λ | Behavior |
|--------|-----|----------|
| `quality` | 0.1 | pay for correctness |
| `balanced` | 0.5 | default |
| `saver` | 1.5 | cost-averse |

Set via `x-routing-preset` (name) or `x-routing-alpha` (raw float).

---

## Post-generation fallback (in the gateway)

The router picks; the gateway guards the result.

- **Non-streaming:** full ladder — on throttle / 5xx / context-length error, or a
  prompt-stage **refusal**, escalate one tier and retry before any bytes reach the
  client.
- **Streaming:** only pre-stream errors can fall back (no mid-stream cascade once
  bytes are flowing).

Every response carries `x-routed-model`, `x-route-reason`, and
`x-est-cost-usd`; the cost ledger records real Bedrock token usage.

---

## Worked example

Request: *"Write a python-pptx deck summarizing this trial design."* — preset `balanced`.

1. **Rules:** system prompt names `python-pptx` → floor `sonnet5` (tier 2).
2. **Sticky:** new session → miss.
3. **Tagger:** `{family: deck, difficulty: 3}` → family floor tier 2. Floor = 2.
4. **Estimator:** p̂/ĉ for sonnet5, opus5, fable5 (haiku45 excluded by floor).
5. **Utility:** at λ=0.5, `sonnet5` wins — passes the task at a fraction of
   fable5's cost. Session pinned to tier 2 for follow-ups.

`reason = "rule:cowork-skill+deck:d3"`

---

## Why it's built this way

- **Cheap-first, escalate-on-demand** — most turns never need the top tier.
- **Explainable** — YAML policies + a scalar λ; no black-box classifier to
  retrain to change behavior.
- **Degrades gracefully** — every stage has a fallback; the router never blocks
  and never fails a request on its own error.
- **Cache-aware** — stickiness protects the per-model prompt cache.
- **Attributable** — spend tagged by strategy/model/tier for Cost Explorer.
