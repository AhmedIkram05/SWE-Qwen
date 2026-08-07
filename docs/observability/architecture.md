# SWE-Qwen Observability Architecture

Status: Phase 8 delivery | Owner: Ahmed | Version: V1 (W&B + Langfuse)

Conforms to PHASE-8-OBSERVABILITY.md (§1–§8), ADR-011 (every stage emits
structured outputs), ADR-015/016/017/018, and the master plan's V1 monitoring
decision ("W&B + Langfuse"). The metric keys cited here are those actually
emitted and registered in `observability/metrics.py` — this document matches
reality, not aspiration.

## 1. Overview

V1 observability is a dual-store setup: **W&B** holds aggregated metrics and
dashboards (the only write target for scalars), **Langfuse** (Cloud, hobby
tier) holds per-call traces of LLM generations. On top: a **structured JSON
logging** layer every component entry point shares, a **metric registry**
that pins every `{domain}/{metric}` key W&B may receive, an **SLO/error-budget
layer** over the serving metrics, **email alerts** on serving degradation, and
**deploy telemetry** from `cd.yml`. OpenTelemetry/Prometheus/Grafana are
explicitly v2 (non-goal for V1); the registry is the seam they slot into.

## 2. Dataflow

| Component | Emission site | Registry keys | Destination |
| --- | --- | --- | --- |
| Data pipeline | `data_engineering/run_pipeline.py` stage boundaries (`log_metrics`) | `data/records_ingested`, `data/records_validated`, `data/records_cleaned`, `data/pipeline_seconds` | W&B aggregates (registered; no panel in V1) |
| Training | `training/callbacks.py` WandbCallback; `training/qlora_trainer.py` finish | `train/loss`, `train/lr`, `train/grad_norm`, `train/gpu_util`, `train/epoch`, `train/step`, `train/cost_usd`; `cost/*` via `log_run_cost` | W&B — Training dashboard |
| Evaluation | `evaluation/harness.py` (`log_aggregate`, `log_eval_run`, `_log_finish_scalars`) | `eval/f2p_rate`, `eval/p2p_rate`, `eval/num_examples`, `eval/total_cost_usd`, `eval/cost_per_fix`, hierarchical `eval/{model}/{variant}/{template}/latency_p50` / `latency_p95`; `cost/*` at finish | W&B — Evaluation dashboard; Langfuse per-example traces (`trace_generation`) |
| Serving | `inference/telemetry.py` flush loop (`run_flush_loop`, `log_cold_start`) | `serve/request_count`, `serve/error_rate`, `serve/ttfb_p50_ms`, `serve/ttfb_p95_ms`, `serve/latency_p50_ms`, `serve/latency_p95_ms`, `serve/tokens_per_sec`, `serve/gpu_util`, `serve/cold_start_s`, `serve/cost_usd`, `serve/cost_per_inference_usd` | W&B — Serving dashboard; Langfuse sampled traces (`trace_request` via flush-loop drain) |
| Deploy | `.github/workflows/cd.yml` "Report deploy telemetry" → `scripts/log_deploy.py` | `deploy/status` (1=success, 0=failure), `deploy/duration_s` | W&B — Infrastructure/Cost dashboard |
| Cost | `observability/cost.py` (`estimate_cost_usd`, `log_run_cost`), called by train finish / eval finish / serving flush | `cost/cost_usd`, `cost/gpu_seconds`, `cost/rate_per_hour` | W&B — Infrastructure/Cost dashboard |

Key names are exactly `observability/metrics.py::METRIC_REGISTRY`; the
evaluation hierarchical pattern is only the `latency_p50`/`latency_p95`
segment suffixes (e.g. `eval/qwen3-14b/baseline_14b/template_v1/latency_p50`).

## 3. Dual-write contract

One event has two legal representations:

| Store | Content | Scope |
| --- | --- | --- |
| W&B | Aggregated scalars only, never per-sample raw data (free-tier cost guard, master plan risk) | Every `{domain}/{metric}` registry key; dashboards aggregate across runs |
| Langfuse | Per-call generation traces (prompt → completion → scores) | Evaluation: **every** golden example, one trace each. Serving: **sampled only** — `rec.error or random.random() >= telemetry_trace_sample_rate` (0.1 in `ServeConfig`) skips the trace; successful requests only |

Cross-links: evaluation traces carry `run_id` + `instance_id` (+ prompt
template / variant) in `trace_generation(metadata=...)`, so the same example
is addressable in both W&B (aggregate series) and Langfuse (per-call trace).
Serving traces are drained within the long-lived serving run and carry
`template_name` + timing/`output_tokens` metadata.

Trace sampling rationale: tracing every serving request would multiply Langfuse
volume and add latency; sampling 0.1 bounds volume and cost. The sampling
decision happens on the request hot path, but the trace itself is only an
O(1) bounded-deque append (`_trace_queue`, `maxlen=500`) — Langfuse IO happens
in the background flush loop (`_drain_trace_queue`), never on the hot path.
Evaluation traces are never sampled: every golden example counts toward
F2P/P2P scoring and prompt comparison, so all of them must be traceable.

## 4. Structured JSON logging

`observability/logging.py::JsonFormatter` emits one JSON object per line:
`{"ts": ISO8601-UTC, "level", "logger", "msg", ...extras}`. Extras are any
non-standard `LogRecord` attributes passed via `extra={...}` — the codebase
convention is `extra={"event": ..., "run_id": ...}`. `msg` stays %-lazy
formatted (the code already uses lazy formatting).

```json
{"ts": "2026-08-07T14:23:11.482713+00:00", "level": "INFO", "logger": "training.qlora_trainer", "msg": "training run started", "run_id": "swe-qwen-20260807-1423", "event": "train_start"}
```

Enable/disable:

| Call | Effect |
| --- | --- |
| `configure_logging(json=True)` (default, `level=logging.INFO`) | Root handler with `JsonFormatter` — production/container stdout |
| `configure_logging(json=False)` | Human-readable `%(asctime)s [%(levelname)s] %(name)s: %(message)s` — local dev |

Details:

- The ~12 `logging.basicConfig` call sites were replaced with
  `configure_logging(...)` (data_engineering, evaluation, scripts, training/
  inference container mains). No other module changes: `logging.getLogger(__name__)`
  loggers inherit the root handler automatically.
- `configure_logging` is idempotent-friendly: if the root logger already has
  handlers it re-formats them in place instead of stacking a second handler.
- Unserializable extras fall back to `str(value)` so a bad value can never
  break the log line.

## 5. Metric registry contract

`observability/metrics.py::METRIC_REGISTRY` is the single source of truth:
one namespace per domain, flat `{domain}/{metric}` keys, plus eval's
hierarchical suffix pattern `eval/{model}/{variant}/{template}/latency_p50|p95`
(mirrors harness L793). Enforcement is belt-and-braces:

| Check | Where | Failure mode |
| --- | --- | --- |
| `assert_registered(metrics)` | `log_metrics()` and `scripts/log_deploy.py::build_payload` | `KeyError` naming the offending key |
| `test_all_wandb_log_keys_registered` | CI `lint-and-test` | AST walk over `src/`, `training/`, `evaluation/`, `inference/`, `data_engineering/`, `observability/` for `wandb.log({...})` / `log_metrics({...})` key literals; any key outside `METRIC_REGISTRY` fails the PR |
| `test_panels_spec_subset_of_registry` | CI `lint-and-test` | Every key in `dashboards.py::PANELS` must be registered (`assert_panels_registered()` rechecks at build time too) |

The plan's DoD note caps each domain at ~5–7 keys (master-plan clutter guard);
the registry table *is* the cap — nothing outside it may be emitted. `serve/*`
carries the full flush-loop metric set (11 keys) and `eval` adds the two
hierarchical latency patterns as one.

To add a key: edit `METRIC_REGISTRY` (or reuse an existing key, which is
preferable), add the emission site, and let the CI contract test prove the
keys match. Removing a key breaks `build_payload` and any dashboard panel
referencing it loudly rather than silently.

Eval hierarchy: `_is_eval_hierarchical` accepts any `eval/` key with ≥ 4
segments whose last segment is `latency_p50`/`latency_p95` — the harness emits
per-repo permutation strings, and the dashboard spec only ever writes the
wildcard form `eval/*/latency_p50`.

## 6. SLO layer

`observability/slo.py` operationalizes the master plan's success criteria —
no new metric keys, derived from the existing `serve/*` collector summary:

| SLO | Target | Source |
| --- | --- | --- |
| S3: TTFB p50 | `< 500 ms` (`SLO_TARGETS["ttfb_p50_ms"] = 500.0`) | master plan S3 |
| S9: cold start | `< 10 s` (`SLO_TARGETS["cold_start_s"] = 10.0`) | master plan S9 |

Mechanics, all in the flush loop (`inference/telemetry.py::run_flush_loop`):

- **Attainment** per flush: `attainment(summary)` returns `1.0` if
  `summary[key] <= target` else `0.0`, for each SLO key present in the
  collector summary (in practice the loop tracks `ttfb_p50_ms`; `cold_start_s`
  is scored when a cold-start measurement lands in the summary, and is
  rendered against the 10 s target on the Infra dashboard).
- **Error-budget math**: budget = 1 − SLO = 1%. Consumed per flush interval =
  `(1 − attainment) × (flush_interval_s / window_s)`; `burn_rate` sums consumed
  across the trailing 3600 s window and divides by the budget.
- **Levels**: `burn_level` returns `None` (healthy) when `n_samples < 10`
  (min-sample guard against low-traffic noise), `WARN` at ≥ 1× the window
  budget, `ERROR` at ≥ 5× (budget would drain in ~6 days at that pace).
- **Alert**: `maybe_alert_burn` fires `wandb.alert` (title
  `SLO error budget burn rate {level}`) only while a W&B run is active.
- OTel v2 replaces this heuristic with proper SLO tooling (ADR-018).

## 7. Alerts

Email via `wandb.alert`, configured once in the W&B UI (project → alerts →
notification destination). Fired from the flush loop, per flush, when the
collector summary tripped a threshold *and* the window has requests
(`count > 0`):

| Condition | Alert |
| --- | --- |
| `error_rate > ServeConfig.alert_error_rate_threshold` (0.10) | `serve/error_rate above threshold` — level `error` |
| `ttfb_p95_ms > ServeConfig.alert_ttfb_p95_threshold_ms` (2000) | `serve/ttfb_p95 above threshold` — level `warn` |
| SLO burn ≥ 1× / 5× budget | `SLO error budget burn rate WARN/ERROR` (see §6) |

`wandb.alert` only fires while a run is active; this works because the serving
run is long-lived. CI eval runs are already gated by the Phase 7 quality gate,
so no alert duplication. Slack webhook is deferred to Phase 11.

## 8. Cost tracking

Estimate-first (ADR-016; no Modal usage API dependency):

```
cost_usd = gpu_seconds / 3600 × rate_per_hour
```

`rate_per_hour` is looked up by GPU type in `config/observability.yaml`
(`a10g-24gb: 1.0`, `a100-40gb: 2.0`, `a100-80gb: 2.5`, `h100-80gb: 4.0`,
`default: 2.0`), overridable by `OBSERVABILITY_RATE_PER_HOUR` (wins when set
and parseable); unparseable/missing config falls back to
`DEFAULT_RATE_PER_HOUR = 1.0`.

| Caller | Emits | Meaning |
| --- | --- | --- |
| Training finish (`qlora_trainer.py`) | `cost/cost_usd`, `cost/gpu_seconds`, `cost/rate_per_hour`, `train/cost_usd` | Train duration × rate |
| Eval finish (`harness.py::_log_finish_scalars`) | `cost/*` trio, `eval/total_cost_usd`, `eval/cost_per_fix` | Eval wall time × rate (rate from `inference_gpu` config); `cost_per_fix = total_cost_usd / max(f2p_passes, 1)` — zero F2P passes is guarded, not a div-by-zero |
| Serving flush loop | `serve/cost_usd` (uptime × rate), `serve/cost_per_inference_usd` | Container uptime since loop start × rate, divided by request count |

Every `cost/*` emission logs `rate_per_hour` alongside so each number is
auditable. The infra dashboard accumulates `cost/cost_usd` and
`cost/gpu_seconds` across runs (sum aggregation). Deploy cost is tracked by
the CI → `log_deploy.py` loop below.

## 9. Deploy telemetry

Closes ADR-011's "deployment status" requirement (ADR-018). Three steps in
`.github/workflows/cd.yml`:

1. **Record deploy start** (`if: always()`): writes `DEPLOY_START_EPOCH` —
   anchored before the deploy so even a failed install still anchors the clock.
2. **Deploy serving endpoint** (`id: deploy`): `uv run modal deploy -m inference.modal_serve`.
3. **Report deploy telemetry** (`if: always()`): maps
   `steps.deploy.outcome` → `STATUS` (1 success / 0 failure), computes
   `DURATION = now − DEPLOY_START_EPOCH`, then runs
   `uv run python scripts/log_deploy.py --status "$STATUS" --duration-s "$DURATION" --sha "${{ github.sha }}"`.

`log_deploy.py` logs the two registered keys (`deploy/status`, `deploy/duration_s`)
to a short-lived `wandb.init(project=swe-qwen, job_type="deploy")` run using the
`WANDB_API_KEY` Phase 7 already provisions in CI; `build_payload` is pure and
`assert_registered`-checked so the CI contract test covers it without
credentials. The commit SHA goes in the run `config`, not as a metric (it is
not a registered key).

`if: always()` on the report step is deliberate: a failed deploy must still
emit — the red dot on the dashboard *is* the point. The infra dashboard shows
the full loop: code push → CI gate → `deploy/status` + `deploy/duration_s` →
cold-start trend vs S9.

## 10. v2 upgrade path

OpenTelemetry + Prometheus/Grafana slot in behind the same registry
(master plan Monitoring v2: "W&B + Langfuse + OpenTelemetry"; ADR-015:
OTel coexists, it does not replace W&B/Langfuse). The registry is the seam:
OTel instrumentation emits the same canonical `{domain}/{metric}` keys, so
dashboards and alerts keep working unchanged while the transport
(W&B scalars + Langfuse traces → OpenTelemetry spans/metrics) evolves. ADR-018
also flags OTel replacing the heuristic SLO burn model with proper SLO tooling.
The JSON-log schema and the registry contract are intentionally transport-agnostic
so this migration is additive.