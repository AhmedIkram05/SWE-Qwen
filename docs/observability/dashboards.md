# SWE-Qwen W&B Dashboards

Status: Phase 8 delivery | Owner: Ahmed | Version: V1

The four W&B dashboards (Training, Evaluation, Serving, Infrastructure/Cost)
are defined as code in `observability/dashboards.py::PANELS` — the versioned
single source of truth. Both build paths below are mechanical from it.

## 1. PANELS spec

Every `PanelSpec` has a registered metric key (`observability/metrics.py`), a
type (`line` / `bar` / `custom`; `run-table` is reserved, unused), a title,
and optionally a `target` (horizontal reference line, e.g. the S9 10 s
cold-start limit) and `aggregate` (cross-run aggregation override).

### Training
| Metric | Type | Title | Notes |
| --- | --- | --- | --- |
| `train/loss` | line | Training Loss | |
| `train/lr` | line | Learning Rate | |
| `train/grad_norm` | line | Gradient Norm | |
| `train/gpu_util` | line | GPU Utilization | |
| `train/cost_usd` | line | Training Cost (USD) | |

### Evaluation
| Metric | Type | Title | Notes |
| --- | --- | --- | --- |
| `eval/f2p_rate` | line | Fail-to-Pass Rate | |
| `eval/p2p_rate` | line | Pass-to-Pass Rate | |
| `eval/*/latency_p50` | line | Eval Segment Latency p50 (ms) | wildcard = `eval/{model}/{variant}/{template}/latency_p50` |
| `eval/*/latency_p95` | line | Eval Segment Latency p95 (ms) | wildcard = `eval/{model}/{variant}/{template}/latency_p95` |
| `eval/cost_per_fix` | bar | Cost per Fix (USD) | |
| `eval/num_examples` | bar | Examples Evaluated | |

### Serving
| Metric | Type | Title | Notes |
| --- | --- | --- | --- |
| `serve/request_count` | line | Request Count | |
| `serve/error_rate` | line | Error Rate | |
| `serve/ttfb_p50_ms` | line | TTFB p50 (ms) | S3 SLO target 500 ms — alert threshold 2000 ms p95 |
| `serve/ttfb_p95_ms` | line | TTFB p95 (ms) | |
| `serve/latency_p50_ms` | line | Latency p50 (ms) | |
| `serve/tokens_per_sec` | line | Tokens per Second | |
| `serve/gpu_util` | line | GPU Utilization | |
| `serve/cost_usd` | line | Serving Cost (USD) | |

### Infrastructure/Cost
| Metric | Type | Title | Notes |
| --- | --- | --- | --- |
| `cost/cost_usd` | bar | Cost (USD, cumulative across runs) | `aggregate="sum"` |
| `cost/gpu_seconds` | bar | GPU-Seconds (cumulative across runs) | `aggregate="sum"` |
| `cost/rate_per_hour` | line | GPU Rate (USD/hour) | auditable alongside every cost number |
| `eval/cost_per_fix` | bar | Cost per Fix (USD) | same key as the Evaluation panel |
| `serve/cold_start_s` | line | Cold Start vs S9 Target (10s) | `target=10.0` → the S9 10 s line |
| `deploy/status` | custom | Deploy Status | one point per cd.yml run |
| `deploy/duration_s` | line | Deploy Duration (s) | |

## 2. Build paths

### (a) As code (primary, decision 5)

```bash
uv sync --extra dashboards                 # wandb-workspaces>=0.4.4 (Public Preview)
uv run python scripts/seed_dashboards.py --project swe-qwen   # [--entity] [--steps 60]
uv run --extra dashboards python scripts/build_dashboards.py --project swe-qwen   # [--entity]
```

1. **Seed run** — `seed_dashboards.py` logs one synthetic run
   (`dashboard-seed`, tagged `seed`, deterministic RNG seed 42) that emits
   *every* registered key over `--steps` steps, including the hierarchical
   eval keys under their real shape (`eval/qwen3-14b/baseline_14b/template_v1/...`)
   and a planted deploy failure at step 30 (the red dot). Panels need
   real-shaped curves to be useful; once the seed has served its purpose it
   can be archived or deleted.
2. **Build** — `build_dashboards.py` validates the spec
   (`assert_panels_registered()`), then creates one Workspace per dashboard
   (`Training`, `Evaluation`, `Serving`, `Infrastructure-Cost`) via
   `wandb_workspaces.workspaces.Workspace(...).save()`. Panel mapping is
   mechanical: `line` → `wr.LinePlot` (+ `custom_expressions=[str(target)]`
   when the spec carries a target line, e.g. the S9 10 s cold-start limit),
   `bar` → `wr.BarPlot` (`groupby_aggfunc` from `aggregate`), `custom` →
   `wr.ScalarChart`. Requires `wandb login` (or `WANDB_API_KEY`).

### (b) Manual UI fallback

The Public Preview API is not a DoD dependency. If
`wandb-workspaces` fights back (breaking schema, auth quirks), build the same
panels manually in the W&B UI from `PANELS` — the spec here is the source of
truth either way, and `test_panels_spec_subset_of_registry` keeps the panel
keys CI-approved regardless of how they were rendered.

## 3. Dashboard URIs

| Dashboard | Workspace name (as created) | URL |
| --- | --- | --- |
| Training | `Training` | https://wandb.ai/2571642-university-of-dundee/swe-qwen?nw=nfnemmwlrtl |
| Evaluation | `Evaluation` | https://wandb.ai/2571642-university-of-dundee/swe-qwen?nw=nklwebmg1q8 |
| Serving | `Serving` | https://wandb.ai/2571642-university-of-dundee/swe-qwen?nw=f3jvhf7qlz8 |
| Infrastructure/Cost | `Infrastructure-Cost` | https://wandb.ai/2571642-university-of-dundee/swe-qwen?nw=pi71byd0ynj |

Built 2026-08-07 from `PANELS` via `build_dashboards.py` (seed run
`ffq2ig4b` is the dashboard-seed data source; `iixs1n2l` is the
`log_deploy.py` probe run). Re-running `build_dashboards.py` after a
`PANELS` change creates a fresh workspace — update this table with the new
`?nw=` URLs.

## 4. How to interpret each panel

- **Training** — loss/lr/grad-norm curves show one training run each;
  `train/cost_usd` is the running estimated cost from train duration × rate.
  The plan's AC-1 (run visible within 60 s of train start) is this dashboard.
- **Evaluation** — `eval/f2p_rate` / `eval/p2p_rate` are per-group F2P/P2P
  rates (last group wins the W&B series; full per-group detail lives in the
  `eval-aggregate-{run_id}` artifact). The latency panels render one line per
  model/variant/template segment (wildcard). **Cost per fix** = eval cost ÷
  F2P-passing golden examples = the dollar cost of one working fix on this
  dataset run — eval `total_cost_usd` divided by the count of results with
  `f2p > 0.0`, zero-passes guarded so it can never be div-by-zero.
- **Serving** — real-time flush-loop aggregates (default every 60 s):
  `serve/ttfb_p50_ms` is the median time to first byte and the S3 SLO line
  (500 ms); error rate above 0.10 or TTFB p95 above 2000 ms fires the email
  alert. `serve/gpu_util` only renders when `nvidia-smi` is reachable (None on
  local macOS). Cost is uptime × rate.
- **Infrastructure/Cost** — `cost/cost_usd` and `cost/gpu_seconds` accumulate
  across runs (sum aggregation) so total platform spend is one number;
  `cost/rate_per_hour` shows the assumed rate for auditability. `deploy/status`
  is a point per cd.yml run: green for success, red for failure — the
  `if: always()` step guarantees failed deploys appear too. `serve/cold_start_s`
  trends against the S9 10 s target line (the SLO layer consumes the same
  target). Cost-per-fix appears here as the cross-cutting economic panel.

## 5. Drift protection

`test_panels_spec_subset_of_registry` (CI `lint-and-test`) asserts every
`PANELS` metric key is a registered key in `METRIC_REGISTRY`, and
`build_dashboards.py` re-runs the same check at build time. The registry and
the dashboards therefore cannot drift apart: a code rename fails the test, an
unregistered dashboard key fails the build, and the seed run proves every
registered key renders real-shaped data before the workspaces are saved.