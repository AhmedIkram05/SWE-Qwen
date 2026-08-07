# Phase 8 — Observability & Telemetry: Implementation Plan

Status: reviewed-APPROVED | Owner: Ahmed | Date: 2026-08-07

## 1. Goal

Deliver full structured telemetry across every platform component — data pipeline, training, evaluation, inference serving, cost, and deployment — with W&B dashboards (training, evaluation, serving, infrastructure/cost), a CI-enforced metric contract, dual-write to Langfuse for prompt→completion→score tracing, serving degradation alerts, an SLO/error-budget layer over the serving metrics (master plan S3/S9), and deploy-status telemetry that closes ADR-011's "deployment status" requirement. Conforms to MASTER-PLAN.md §Phase 8 (lines 861–920), ADR-011 (every stage emits structured outputs), and the V1 monitoring decision "W&B + Langfuse" from the master plan decision table.

Non-goals (explicit, locked):

- **NO OpenTelemetry / Prometheus / Grafana** — that is v2 (8.10 deferred; master plan Monitoring v2 table). V1 = W&B + Langfuse Cloud.
- No changes to the eval *runner* semantics (harness/inference containers) — Phase 8 only adds emission at existing call sites + new `observability/` package.
- No Modal usage API dependency in the DoD path — real-spend pull is a stretch goal only (decision 3).
- No per-sample raw data in W&B (free-tier cost guard, master plan risk): W&B gets aggregates; Langfuse gets per-call traces (decision 1).
- Dashboards are built from the registered contract + seed run via **`wandb-workspaces` (Public Preview API) as code**; the manual UI build is the fallback if the preview API fights back (decision 5).

## 2. What already exists (reuse, don't rebuild)

| Piece | Where | Notes |
| --- | --- | --- |
| Serving metrics engine | `inference/telemetry.py` (183 lines, stdlib-only, lazy wandb) | `MetricsCollector` (thread-safe), `_metrics_dict` → `serve/*`, flush loop thread, `cost_per_inference()`, `log_gpu_util()`, `log_cold_start()`, `finish_wandb()`. Already wired into Wave-3 `modal_serve.py`. **8.4 is ~95% done.** |
| Training metrics | `training/callbacks.py` WandbCallback (L117–125 `wandb.log(metrics)`, checkpoint artifacts), `training/qlora_trainer.py` (init L166, log L366, artifact L396) | Keys not yet normalized to a registry; grad-norm/GPU util coverage to verify. |
| Eval metrics | `evaluation/harness.py` (eval/cost_usd L734, `eval/{key}/latency_p50/p95` L738-739, hierarchical `eval/{model}/{variant}/{template}` prefix L793), `evaluation/comparison.py` (wandb.init L346, champion selection L185) | F2P/P2P scalar key names unverified — normalization step will pin them. |
| Langfuse dependency | `pyproject.toml` `langfuse>=2.60.0` — **declared, zero usage** | No import anywhere in `*.py`. 8.7 fills the gap; no pyproject change needed (verify SDK API surface via Context7 at implementation). |
| Cost helper | `inference/telemetry.py::cost_per_inference(gpu_seconds, requests, rate_per_hour)` | Formula already exists; needs `cost/*` emission + duration capture. |
| W&B project pin | `scripts/init_wandb.py` | Re-pin `swe-qwen` (auto-deleted once — Phase 7 lesson) before dashboard setup. |
| Dashboard-as-code API | PyPI `wandb-workspaces` 0.4.4 (wandb official, **Public Preview**) | `wandb_workspaces.reports.v2` (`wr.Report`, `wr.PanelGrid`, `wr.LinePlot`/`BarPlot`/`ScalarChart`, `wr.Runset` filters) + `wandb_workspaces.workspaces` (`ws.Workspace(entity, project, sections=[ws.Section(name, panels, is_open)]).save()`) — creates the project workspace itself. Verified against official docs 2026-08-07; add to pyproject as a dev/optional dep. |
| CI plumbing | `.github/workflows/ci.yml` (ruff, mypy, pytest w/ cov ≥75), eval.yml | Registry + JSON-log tests slot into `lint-and-test`; no new workflow file. |
| GPU util probe | `inference/telemetry.py::log_gpu_util` | nvidia-smi csv, timeout 5, None on macOS — reused by training emission. |
| SLO targets | MASTER-PLAN.md success criteria: S3 TTFB p50 < 500ms, S9 cold start < 10s, scale-to-zero $0 idle | Free targets — the SLO layer (5.4b) consumes them; no new requirements invented. |
| Deploy job | `.github/workflows/cd.yml` (terraform-plan + deploy, WANDB_API_KEY already available — Phase 7) | Deploy telemetry (5.6b) hooks here; no new workflow, one added step. |

## 3. Decision log (grill-adopted)

| # | Decision | Consequence enforced in code |
| --- | --- | --- |
| 1 | **Langfuse scope = eval + sampled serving.** Eval: every golden example → one trace (prompt → completion → f2p/p2p scores). Serving: only successful requests, `sample_rate=0.1` (config), async flush, fire-and-forget — never on the request hot path. | `observability/langfuse.py`; serving traces drained by the existing telemetry flush thread |
| 2 | **Langfuse hosting = Cloud** (hobby free tier). `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST=https://cloud.langfuse.com` as GitHub + Modal secrets (Modal-hosted secrets by name, Phase 7 lesson). Missing keys ⇒ module is a silent no-op (local dev / CI). | lazy import + `if not keys: return` pattern like telemetry.py |
| 3 | **Cost = estimate-first**: `cost_usd = gpu_seconds / 3600 * rate_per_hour` with `rate_per_hour` logged alongside for auditability. Modal usage API = stretch, dropped silently if a rabbit hole. | `observability/cost.py`; no new deps |
| 4 | **"Cost per F2P point" = cost per successful fix**: `eval cost ÷ # F2P-passing golden examples` (dollar cost of one working fix), logged `cost/cost_per_fix` by the harness at finish. | harness finish hook |
| 5 | **Dashboards = metric registry + seed run + dashboards-as-code (Public Preview API).** `observability/dashboards.py` holds the PANELS spec (which registered keys on which panel); `scripts/seed_dashboards.py` logs one synthetic run with every registered key so panels render real-shaped curves; `scripts/build_dashboards.py` then creates the 4 project workspaces/reports programmatically via `wandb-workspaces` (`ws.Workspace` + `wr.PanelGrid`) from the same PANELS spec. If the preview API fights back (breaking changes, auth quirks), the manual UI build is the documented fallback — PANELS spec is the single source of truth either way. URLs documented in `docs/observability/dashboards.md` (README refactor is a later phase). | PANELS spec + seed script + `build_dashboards.py` |
| 6 | **JSON logging = full retrofit** (user decision): one `JsonFormatter` + `configure_logging()` in `observability/logging.py`; every component entry point calls it (CLI mains, train entry, serve container stdout, run_pipeline). Module-level `logging.getLogger(__name__)` calls stay untouched — they inherit the root formatter. | ~12 `logging.basicConfig` sites replaced |
| 7 | **Metric registry is the contract.** `observability/metrics.py` defines every allowed `{domain}/{metric}` key; a CI test fails if any `wandb.log` key in the repo is unregistered. Kills silent dashboard drift. | `tests/observability/test_telemetry_contract.py` |
| 8 | **Alerts = serving degradation only, W&B email.** `wandb.alert()` fired from the flush loop on `serve/error_rate > 0.10` or `serve/ttfb_p95_ms > 2000` (thresholds in `ServeConfig`). Eval degradation is already the Phase 7 CI gate — no duplication. Slack webhook deferred to Phase 11. | flush loop + ServeConfig fields |
| 9 | **SLO layer (user-approved amendment).** SLOs come from the master plan, not new ones: S3 (TTFB p50 < 500ms) and S9 (cold start < 10s). `observability/slo.py` computes trailing-window attainment + error-budget burn from the existing `MetricsCollector` summary; `wandb.alert` on burn (WARN ≥ 1× budget, ERROR ≥ 5×). No new metric keys — attainment is derived, the raw `serve/*` keys already render. | `observability/slo.py` (5.4b) |
| 10 | **Deploy-status telemetry (user-approved amendment).** ADR-011 requires "deployment status" as a structured output — the plan missed it. `scripts/log_deploy.py` runs as a step in `cd.yml` (after the deploy job, `if: always()`), emitting `deploy/status` (0/1) and `deploy/duration_s` using the WANDB_API_KEY Phase 7 already puts in CI. Infra dashboard gains a deploy-status panel + cold-start trend vs the S9 target. | `scripts/log_deploy.py` + cd.yml (5.6b) |

## 4. The telemetry contract (metric registry)

Single source of truth: `observability/metrics.py` → `METRIC_REGISTRY: dict[namespace, dict[key, description]]`. One namespace per domain, `{domain}/{metric}` flat keys (except eval's allowed hierarchical suffix `eval/{model}/{variant}/{template}/{metric}`, which mirrors harness L793).

| Namespace | Canonical keys | Source (now → after) | Status |
| --- | --- | --- | --- |
| `serve/*` | request_count, error_rate, ttfb_p50_ms, ttfb_p95_ms, latency_p50_ms, latency_p95_ms, tokens_per_sec, gpu_util, cold_start_s | `inference/telemetry.py` | exists — register as-is |
| `serve/*` (new) | cost_usd, cost_per_inference_usd | telemetry flush loop | add |
| `train/*` | loss, lr, grad_norm, gpu_util, epoch, step, cost_usd | `training/callbacks.py`, `qlora_trainer.py` | verify current keys → normalize to registry names |
| `eval/*` | f2p_rate, p2p_rate, num_examples, total_cost_usd, {key}/latency_p50, {key}/latency_p95, cost_per_fix | `evaluation/harness.py` | verify F2P/P2P key names → normalize; add cost_per_fix |
| `cost/*` | cost_usd, gpu_seconds, rate_per_hour | `observability/cost.py` | new |
| `data/*` | records_ingested, records_validated, records_cleaned, pipeline_seconds | `data_engineering/run_pipeline.py`, `cli.py` | new (small — 4 keys, DoD covers ingest/validate/clean) |
| `deploy/*` | status, duration_s | `scripts/log_deploy.py` ← cd.yml step | new (decision 10; ADR-011 deployment status) |

DoD note: 5–7 keys per domain maximum (master plan clutter guard) — the table above is the cap; nothing else may be emitted.

## 5. Changes (by file)

### 5.1 `observability/__init__.py` (new)

Exports: `configure_logging`, `log_metrics`, `estimate_cost_usd`, `trace_generation`. No side effects on import (lazy wandb/langfuse imports, same pattern as telemetry.py).

### 5.2 `observability/logging.py` (new) — task 8.1

- `JsonFormatter(logging.Formatter)`: one JSON object per line — `{"ts": ISO8601-UTC, "level", "logger", "msg", ...extras}`. Extras passed via `extra={"event": ..., "run_id": ...}`; `msg` stays %-lazy formatted (codebase already uses lazy formatting).
- `configure_logging(level=logging.INFO, json: bool = True)`: root handler with `JsonFormatter`; `json=False` keeps human-readable local dev output.
- Retrofit call sites (replace each `logging.basicConfig(...)`): `data_engineering/cli.py:273`, `evaluation/cli.py:88`, `scripts/local_e2e_smoke.py:27`, `scripts/prepare_training_data.py:33`, `scripts/run_3config_comparison.py:32`, `scripts/validate_golden.py:18`, `scripts/export_golden.py:38`, `data_engineering/run_pipeline.py` (top), `training/modal_train.py` (container stdout — Phase 6 lesson: structured logs to stdout from day one), `inference/serve.py` + `modal_serve.py` (container stdout). All other modules need **zero** changes — `getLogger(__name__)` inherits the root handler.
- Format documented in §8 of this doc (DoD: "structured logging format documented").

### 5.3 `observability/metrics.py` (new) — registry + emitter guard

- `METRIC_REGISTRY` (table in §4) + `assert_registered(keys: dict) -> None` (raises `KeyError` with the offending key).
- `log_metrics(metrics: dict)` = `assert_registered` then lazy `wandb.log` (no-op without run). **Existing call sites keep calling `wandb.log` directly** — the registry test (5.9) enforces the contract without an invasive sweep of every call site.

### 5.4 `observability/cost.py` (new) — task 8.6

- `estimate_cost_usd(gpu_seconds, rate_per_hour) = gpu_seconds/3600 * rate_per_hour` (reuses the telemetry formula; single implementation).
- `log_run_cost(wandb_run, gpu_seconds, rate_per_hour)`: logs `cost/cost_usd`, `cost/gpu_seconds`, `cost/rate_per_hour` to the active run.
- Callers: training finish (`training/qlora_trainer.py` — duration from train start/end timestamps, rate from config), harness finish (eval duration + rate → feeds `eval/cost_per_fix`), serving flush loop (`serve/cost_usd` from uptime).
- Rates: `config/observability.yaml` — `{gpu_type: rate_per_hour}` (Modal pay-per-use; documented default ~$0.50–2.00/hr per master plan cost model), overridable by env `OBSERVABILITY_RATE_PER_HOUR`. Stretch (separate step, not DoD): Modal usage GraphQL pull, dropped if it fights back.

### 5.4b `observability/slo.py` (new) — SLO layer (decision 9)

- Targets read from the master plan, hardcoded as the v1 defaults: `SLO_TARGETS = {"ttfb_p50_ms": 500, "cold_start_s": 10}` (S3, S9). No config file, no new keys — derived from the existing `serve/*` metrics (registry untouched, clutter guard intact).
- `attainment(summary) -> dict`: per SLO, `1.0` if the collector summary meets the target (ttfb_p50_ms ≤ 500; cold_start_s ≤ 10), else `0.0`. Runs on each flush.
- `burn_rate(attainment_history, window_s=3600) -> float`: error budget = 1 − SLO (99% → 1%); budget consumed per flush interval = (1 − attainment) × (interval / window); burn rate = consumed ÷ budget. v1 heuristic documented in `docs/observability/architecture.md`; OTel v2 gets proper SLO tooling.
- `maybe_alert_burn(burn_rate)`: `wandb.alert` WARN at ≥ 1× budget (100% consumed), ERROR at ≥ 5× (fast burn — budget would drain in ~6 days at this pace). Min-sample guard: no alert until ≥ 10 flush samples in the window (low-traffic noise guard, decision 9).
- Call site: telemetry flush loop (same place as decision-8 alerts). ~40 lines.

### 5.5 `observability/langfuse.py` (new) — task 8.7

- Lazy `Langfuse(public_key=..., secret_key=..., host=...)` client built from env; `_enabled()` false → all functions no-op (local dev, CI without secrets).
- `trace_generation(*, name, model, prompt, completion, metadata, scores=None)`: one Langfuse generation trace (prompt→completion) + `lf.score()` per score (f2p, p2p) with `instance_id`, `run_id`, `prompt_template`, `variant` in metadata. SDK call surface verified via Context7 at implementation (pin against installed `langfuse>=2.60.0`).
- `trace_request(rec: RequestRecord)`: serving trace (model, prompt-builder template name, ttfbs, latency, tokens) — **only for sampled non-error records** (decision 1).
- Dual-write: W&B = aggregates, Langfuse = per-call traces; cross-link by `run_id` + `instance_id` metadata (documented in §8). Prompt A/B: `evaluation/prompt_ab_test.py` already samples templates — its template names flow into trace metadata, making prompt-comparison visible in the Langfuse UI and the eval dashboard (AC-5).
- Call sites: harness per-example loop (eval traces), telemetry flush thread (sampled serving traces — drain a bounded queue, never block the collector).

### 5.6 `observability/dashboards.py` + `scripts/seed_dashboards.py` + `scripts/build_dashboards.py` (new) — task 8.5

- `dashboards.py`: `PANELS` spec — for each of the 4 dashboards, the registered keys and panel type (line/timeseries, bar, run-table, custom metric). This is the versioned artifact; both build paths are mechanical from it.
- `seed_dashboards.py`: `uv run python scripts/seed_dashboards.py --project swe-qwen` logs one synthetic run ("dashboard-seed") emitting every registered key with realistic values over ~60 synthetic steps → panels get real-shaped curves; then delete/archive the seed run afterwards (or keep tagged `seed`).
- `build_dashboards.py` (as-code path, decision 5): consumes `PANELS` and creates the 4 dashboards via `wandb-workspaces` (new optional dep) — `ws.Workspace(entity, project, sections=[ws.Section(name, panels=[wr.LinePlot(x="Step", y=[...]), ...], is_open=True)]).save()` per dashboard; `wr.LinePlotConfig.expressions` can render target lines (e.g. S9 cold-start 10s). Requires `wandb login` locally (user auth — see §7 manual list). Fallback if Public Preview API breaks: build the same panels manually in the UI from `PANELS`.
- Setup: re-pin project (`scripts/init_wandb.py`) → seed run → `build_dashboards.py` (or UI fallback) → save → paste URLs into `docs/observability/dashboards.md`.
- Panel contents (5–7 metrics each): **Training** — train/loss, train/lr, train/grad_norm, train/gpu_util, train/cost_usd; **Evaluation** — eval/f2p_rate, eval/p2p_rate, eval/*/latency_p50, eval/*/latency_p95, eval/cost_per_fix, eval/num_examples; **Serving** — serve/request_count, serve/error_rate, serve/ttfb_p50_ms, serve/ttfb_p95_ms, serve/latency_p50_ms, serve/tokens_per_sec, serve/gpu_util, serve/cost_usd; **Infrastructure/Cost** — cost/cost_usd (cumulative across runs), cost/gpu_seconds, cost/rate_per_hour, cost/cost_per_fix, serve/cold_start_s (trend vs S9 10s target line), deploy/status (step/bar per deploy), deploy/duration_s.
- URLs documented in `docs/observability/dashboards.md` (README refactor is a later phase — deliberate, per user).

### 5.6b Deploy telemetry — `scripts/log_deploy.py` (new) + cd.yml hook (decision 10)

- `log_deploy.py`: `--status 0|1 --duration_s N` → logs `deploy/status` (1 = success, 0 = failure) + `deploy/duration_s` to the pinned `swe-qwen` project via `wandb.init(project=..., job_type="deploy")` (short-lived run, reuses Phase 7's WANDB_API_KEY in CI). Keys registered in 5.3 — CI contract test covers them.
- cd.yml: one step after the deploy job, `if: always()`, passing the deploy step outcome → status + duration (`${{ steps.deploy.outcome }}`, `${{ github.sha }}` in run notes). Failure deploys still emit — the dashboard shows the red dot, that's the point.
- Infra dashboard panel (5.6) makes the whole loop visible: code push → CI gate → deploy/status + duration → cold-start trend vs S9.

### 5.7 Training + eval emission (tasks 8.2, 8.3)

- `training/callbacks.py`: rename verified WandbCallback keys to registry names (`train/loss`, `train/lr`, `train/grad_norm`, `train/step`); add `train/gpu_util` (reuse `telemetry.log_gpu_util`) at log frequency.
- `training/qlora_trainer.py`: at finish → `log_run_cost` (duration × rate).
- `evaluation/harness.py`: pin F2P/P2P scalar key names to `eval/f2p_rate`, `eval/p2p_rate`; log `eval/num_examples`; at finish → `eval/cost_per_fix = total_cost / max(f2p_passes, 1)` and `log_run_cost`. No runner-semantics changes (non-goal).
- `data_engineering/run_pipeline.py` + `cli.py`: 4 `data/*` keys (records_ingested/validated/cleaned, pipeline_seconds) at stage boundaries. DoD requires ingest/validate/clean to emit — these are the only data-pipeline emission points, and they already exist as counters in the pipeline run (verify names at implementation).

### 5.8 Serving + alerts (tasks 8.4, 8.8)

- `inference/telemetry.py` flush loop: add `serve/cost_usd` (uptime × rate) + `serve/cost_per_inference_usd` (existing helper); add `wandb.alert("serve/error_rate", ...)` when `error_rate > ServeConfig.alert_error_rate_threshold` (0.10) or `ttfb_p95_ms > ServeConfig.alert_ttfb_p95_threshold_ms` (2000), level WARN/ERROR. `wandb.alert` fires only while a run is active — the serving run is long-lived, so this works; CI eval runs are already gated by Phase 7 (no duplicate alerts).
- `inference/config.py`: add the two frozen `ServeConfig` fields (matches the existing Phase 6 config pattern).
- `inference/modal_serve.py`: unchanged code path — flush loop already runs there; wire `trace_request` sampling into it (5.5).

### 5.9 CI contract tests — `tests/observability/test_telemetry_contract.py` (new)

- `test_all_wandb_log_keys_registered`: scans `src/`, `training/`, `evaluation/`, `inference/`, `data_engineering/`, `observability/` for `wandb.log({...})` / `log_metrics({...})` key literals (AST walk, no wandb import needed) and asserts every key ∈ `METRIC_REGISTRY`. Fails the PR if anyone emits an unregistered key — dashboards can't drift silently.
- `test_json_log_parse`: `configure_logging()` → capture handler → emit INFO/WARNING records with extras → assert each line parses as JSON with `ts/level/logger/msg` + extras.
- `test_cost_formula`: `estimate_cost_usd(3600, 2.0) == 2.0`; `cost_per_fix` division guard (zero passes → no div-by-zero, logs `None`).
- `test_langfuse_noop_without_keys`: unset env → `trace_generation` returns without touching network (patched client never constructed).
- `test_slo_burn`: collector summary with ttfb_p50 > 500 → attainment 0.0, burn_rate ≥ 1 after ≥ 10 samples, alert path fires (patched `wandb.alert`); min-sample guard: < 10 samples → no alert.
- `test_deploy_payload`: `log_deploy.py` build/parse produces exactly the registered `deploy/*` keys (import the pure builder, no wandb init needed).
- `test_panels_spec_subset_of_registry`: every metric key referenced in `dashboards.py::PANELS` is a registered key in `METRIC_REGISTRY` — the as-code dashboard generator can never reference a key CI hasn't approved (dashboard drift guard extends to the as-code path).
- Slots into existing `ci.yml` `lint-and-test` (no new workflow, no new secrets — Langfuse keys never needed in CI, decision 2). CI runs this suite with `--cov=observability --cov-branch --cov-fail-under=100` — any new un-tested line in the observability package fails the PR (see §7).

## 6. Implementation order

1. **Foundation**: 5.2 logging.py + retrofit call sites (8.1) → 5.3 metrics.py registry (8.2-core) — everything else depends on these.
2. **Emission**: 5.7 training/eval/data key normalization (8.2, 8.3) → 5.4 cost.py (8.6) — dashboards need real keys first.
3. **Serving + alerts + SLO**: 5.8 (8.4, 8.8) → 5.4b (SLO burn — consumes the same collector summary, decision 9).
4. **Langfuse**: 5.5 (8.7) — independent of 1–3, parallelizable; verify SDK API via Context7 first.
5. **Dashboards + deploy telemetry**: 5.6 (8.5) as-code build via `wandb-workspaces` (UI fallback) + saved URLs (only after 2 so panels show real shapes); 5.6b `log_deploy.py` + cd.yml hook (decision 10 — independent, can land anytime after 5.3).
6. **Contract tests + docs**: 5.9 → §8 documentation (8.9).

## 7. Testing strategy

- **Coverage bar: 100% of new code.** `observability/` (logging, metrics, cost, slo, langfuse) + the pure parts of `scripts/seed_dashboards.py` / `scripts/log_deploy.py` must hit 100% line+branch coverage (`pytest --cov=observability --cov=scripts --cov-fail-under=100 --cov-branch` in CI `lint-and-test`; whole-repo floor stays ≥75% per Phase 7). The only exceptions carved out in tests: lazy wandb/langfuse client construction (patched, never real) and the `nvidia-smi` probe (returns None on macOS).
- Unit: `tests/observability/test_telemetry_contract.py` (5.9) — registry, JSON parse, cost formula, Langfuse no-op, SLO burn, deploy payload. All offline, no credentials.
- Integration: `scripts/local_e2e_smoke.py` run with `configure_logging(json=True)` — captured stdout parses line-by-line as JSON (smoke tier already exists; Phase 6 debugging lesson).
- E2E / manual (acceptance-driven): start a smoke eval → within 60s a run appears on the eval dashboard; fire a few served requests via `inference/benchmark.py` → serving dashboard updates in real time; verify a Langfuse trace with scores in the Cloud UI; confirm alert email fires when the error-rate threshold is deliberately tripped (temp test value, then restore).

## 8. Documentation (task 8.9) — `docs/observability/`

- `architecture.md`: dataflow — component → emission site → registry key → W&B dashboard / Langfuse trace; dual-write split (W&B aggregates, Langfuse traces); cross-links via `run_id` + `instance_id`; the JSON log format spec (fields, extras convention, `configure_logging` usage); v2 upgrade path note (OTel instrumentation slots in behind the same registry, master plan Monitoring v2).
- `dashboards.md`: PANELS spec (from `dashboards.py`), step-by-step UI build, saved dashboard URLs, how to interpret each panel ("cost per fix: dollar cost of one F2P-passing fix on this dataset run").

## 9. Risks & mitigations

- **Langfuse on serving path** (latency/cost): sampling (0.1), async drain in the existing flush thread, no-op without keys, never on the hot path. Trace volume bounded per run.
- **W&B free tier**: aggregates only, never per-sample raw logs (master plan risk); alert only on degradation, not volume.
- **Dashboard clutter**: 5–7 registered keys per domain cap (decision 7 + §4); registry test blocks new keys silently.
- **`wandb-workspaces` Public Preview breakage** (decision 5): API is preview-status — schema/save quirks possible. Mitigation: PANELS spec is the single source of truth, `build_dashboards.py` is isolated in one script, and the manual UI build from the same spec is the documented fallback; preview API is never a DoD dependency.
- **Key normalization breaks dashboards mid-phase**: registry lands *before* emission changes; dashboards built last; seed run verifies every registered key renders.
- **Modal usage API stretch**: explicitly non-DoD; dropped if it fights back (decision 3).
- **`wandb.alert` silent failure**: only fires with an active run; serving runs are long-lived so the condition holds; alert config (email) is a one-time W&B UI setting documented in `docs/observability/architecture.md`.
- **SLO burn alert noise at low traffic**: a handful of slow samples on a quiet endpoint could page falsely — min-sample guard (≥ 10 flush samples in the window) + WARN-before-ERROR escalation; thresholds re-tunable in `slo.py` (decision 9).
- **Deploy telemetry depends on cd.yml correctness**: `if: always()` + explicit outcome capture, or failed deploys silently vanish — covered by the `test_deploy_payload` pure-builder test and a manual check of one failed-deploy run.

## 10. DoD & acceptance mapping

| Master plan DoD | Covered by |
| --- | --- |
| 4 W&B dashboards (training, eval, serving, infra/cost) | 5.6 (as-code via `wandb-workspaces`; manual UI fallback) |
| Every component (ingest, validate, clean, train, eval, serve) emits structured JSON logs | 5.2 retrofit (incl. `data/*` emission 5.7) |
| Key metrics visible real-time on dashboards during active experiments | 5.6/5.8; verified via §7 manual E2E |
| Cost per experiment + cumulative cost in W&B | 5.4 (`cost/cost_usd` per run; cumulative panel on infra dashboard) |
| Langfuse traces LLM calls, prompt versions, eval runs; dual-write working | 5.5 |
| Structured logging format documented | §8 architecture.md |
| AC-1: run on training dashboard within 60s | 5.6/5.7; W&B live logging already in callbacks |
| AC-2: inference requests visible real-time | flush loop 5.8 |
| AC-3: cost reflects Modal + W&B spend | 5.4 (estimate-first, rate logged; API stretch) |
| AC-4: structured logs queryable (JSON, consistent fields) | 5.2 + §8 format spec + parse test |
| AC-5: Langfuse prompt→completion→eval_score traces; prompt A/B visible | 5.5 (template metadata from `prompt_ab_test.py`) |
| ADR-011: "deployment status" structured output | 5.6b (`deploy/status`, `deploy/duration_s` from cd.yml) |
| Master plan S3 (TTFB p50 < 500ms) / S9 (cold start < 10s) made operational | 5.4b (attainment + burn alerts) |

## 11. Proposed ADRs (ADR-015/016/017 appended to ADR-&-VISION.md; ADR-018 pending your OK)

- **ADR-015: Langfuse as V1 trace store (eval + sampled serving, Cloud-hosted)** — why: trace debugging, prompt versioning, eval comparisons (8.7); sampling + async drain protect the hot path; no-op without keys keeps CI/local hermetic; OTel v2 coexists (dual observability, master plan CI/CD evolution table).
- **ADR-016: Cost tracking is estimate-first** — why: `duration × rate` logged with `rate_per_hour` is auditable and dependency-free; Modal usage API is real-spend ground truth but thin-documented and non-DoD; "cost per fix" = eval cost ÷ F2P passes, chosen for recruiter/intuitive semantics.
- **ADR-017: Metric registry as the telemetry contract** — why: dashboards depend on key names, so the registry + panel spec + CI test is the versioned contract that keeps dashboards (whether generated via the `wandb-workspaces` Public Preview API or built in the UI) and code in lockstep; unregistered keys fail CI. Original claim "W&B has no dashboard-as-code API" retracted — `wandb-workspaces` 0.4.4 (Reports + Workspaces API) verified 2026-08-07; as-code is now the primary path, UI is fallback.
- **ADR-018: V1 reliability telemetry = SLO layer + deploy status** — why: ADR-011 lists "deployment status" among required structured outputs and the master plan already defines S3/S9, so the phase operationalizes both: `observability/slo.py` derives attainment + burn from the existing collector (no new keys, clutter guard) and alerts via the decision-8 `wandb.alert` channel, and `scripts/log_deploy.py` closes the CI/CD → deploy → observe loop using CI's existing WANDB_API_KEY.

## 12. Glossary (terms resolved this session — offered for CONTEXT.md)

- **telemetry namespace** — `{domain}/{metric}` flat key convention (`serve/*`, `train/*`, `eval/*`, `cost/*`, `data/*`); the unit the registry governs.
- **cost per fix** — eval cost ÷ number of F2P-passing golden examples; dollar cost of one working fix.
- **dual-write** — same event lands in W&B (aggregates) and Langfuse (per-call trace), linked by `run_id`/`instance_id`.
- **trace sampling** — tracing only a fraction of serving requests (default 0.1) to bound Langfuse volume/latency.

## 13. Effort estimate

Foundation M → Emission M → Serving/alerts/SLO S → Langfuse M → Dashboards M (as-code script; UI fallback) → Deploy telemetry S → Tests/docs S. Total ≈ 2 focused sessions of agentic implementation + one manual dashboard-building pass (SLO layer and deploy script are small additions — ~70 lines combined with tests).
