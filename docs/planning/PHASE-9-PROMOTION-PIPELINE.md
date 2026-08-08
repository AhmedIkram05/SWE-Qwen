# Phase 9 — Champion/Challenger Promotion Pipeline: Implementation Plan

Status: reviewed-approved (grill-adopted) | Owner: Ahmed | Date: 2026-08-07

## 0. Current status — prerequisites satisfied (2026-08-08)

Pre-flight ops are DONE; implementation can start now. **No training is required
for the first cycle — a challenger is already in the lane.**

- **`MODAL_SERVE_TOKEN` minted** (random 96-hex, never stored in the repo): Modal
  secret `serve-token` + GitHub Actions secret `MODAL_SERVE_TOKEN`, the secret
  also carries `SERVING_DEFAULT_VARIANT=higher_lr_14b` (the deploy-step pin
  target). Step 0's "create the Modal Secret first" checklist item is complete.
- **Challenger tagged, no training**: `model-qwen3-14b-higher_rank_14b:v3` =
  `challenger` + `latest` on W&B (`swe-qwen`). `tag_challenger.py` /
  `run_3config_comparison.py` hook (§4.7) is for **future** cycles only.
- **W&B aliases clean**: exactly one champion — `model-qwen3-14b-higher_lr_14b:v3`
  and its `eval-champion:latest` link (both `champion`); stale `champion` on
  `baseline_14b:v8` removed.
- **Credentials/env in place**: 12 GH secrets (GCP WIF, Modal, W&B, CODECOV,
  LANGFUSE, MODAL_SERVE_TOKEN); `production` env with required-reviewers gate
  exists; repo var `RUN_MODAL_EVAL=false` (pipeline defaults to the $0
  gating-off path until flipped).
- **Only remaining prerequisite to the FIRST promotion run** (not to
  implementation, and not training): seed `gs://swe-qwen-datasets/ci/champion.json`
  via `seed_champion.py` (§4.7) — `ci/` is still EMPTY. Runs during
  implementation step 3; needs GCP auth (WIF job, or one
  `gcloud auth application-default login` locally).
- **First-cycle expectation**: `higher_rank_14b` (F2P 14.6% / P2P 61.5%)
  REJECTS vs champion (16.9% / 91.2%) — accepted; the first documented
  *rejected* decision is the correct DoD output and exercises the entire loop
  (eval → gate → decision → audit) with no promotion.

## 1. Goal

Automate ADR-007 (Champion/Challenger auto-promotion, manual deployment
decisions prohibited). A candidate checkpoint that enters the Challenger lane
is evaluated head-to-head against the current Champion on the golden eval set;
if it beats the Champion by a statistically-gated margin it is promoted,
re-deployed to the Modal endpoint, and every decision — accept, reject, or
rollback — lands as an auditable W&B decision record.

From the user's mouth: the pipeline must **not involve manually running
anything** past the GitHub approval gate "like in cd.yml". The only human act
in the entire loop is approving the deploy job's `environment: production`
gate in the Actions UI. Cadence is chained to training: when training
completes it tags the new checkpoint `challenger` and dispatches the promote
workflow; the pipeline runs the rest.

Non-goals (explicit, locked):

- NO re-training decisions here. Promotion only.
- NO code moved out of `evaluation/comparison.py`. Phase 9 **composes** the
  existing promote/normalize/significance logic — it does not rewrite it.
- NO un-bounded Modal spend: the `RUN_MODAL_EVAL` repository variable (same
  kill-switch idea as `eval.yml`, but **acts on every run — promote runs on
  `workflow_dispatch`, whose `github.ref` is `main`, so the eval.yml PR-only
  guard would never trip; promote skips unconditionally when "false"**). When
  "false", the promote
  workflow runs only free steps (validate challenger, write audit note
  `gating-off`, `modal token info`) and spends zero GPU dollars.

## 2. What already exists (reuse, don't rebuild)

| Piece | Where | Notes |
| --- | --- | --- |
| Champion re-validation | `evaluation/comparison.py::revalidate_champion` (L104) | filters `p2p_rate ≥ min_p2p` AND `f2p_rate ≥ min_f2p`, ranks by f2p desc |
| Paired bootstrap / McNemar | `evaluation/stats.py::paired_bootstrap_ci` + `mcnamar_p` | gate.py calls them DIRECTLY on per-instance vectors (candidate as arg `a` ⇒ `lo > 0` means candidate beats champion); `comparison.paired_significance` (L215) kept for **display only** |
| Registry promote + alias | `evaluation/comparison.py::promote_champion_to_registry` (L311) + `_clear_champion_alias` (L298) | lazy wandb; links `model-qwen3-14b-{variant}:latest` to `eval-champion` collection `champion` alias; never raises |
| Per-instance pairing | `evaluation/schema.py::EvalResult.instance_id` + `f2p` (0/1) | `comparison.extract_model_metrics` returns **aggregates** only — gate.py builds per-variant 0/1 vectors from `run.results` and intersects `instance_id`s itself |
| Run files self-describe | `evaluation/schema.py::EvalRun` | embeds full `EvalConfig` (dataset_run_id, tier, seed) ⇒ pair sanity-checked by assert, no external state |
| Eval tiers | `evaluation/cli.py` `eval run --mode smoke\|dev\|final` | dev = 100 instances, seed 42; `tier_sizes`/`tier_seed` in `EvalConfig` |
| Baseline-in-GCS pattern | `gs://swe-qwen-datasets/ci/smoke_baseline.json` (+ `eval.yml` `--update-baseline`) | champion.json mirrors it. `ci/` prefix is currently EMPTY (verified) — `seed_champion.py` must run before the first promote; reads are publicly readable, **writes require GCP WIF auth** |
| Modal deploy + telemetry | `cd.yml` job `deploy-modal` (L150) | `uv run modal deploy -m inference.modal_serve`, secrets `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`/`HF_TOKEN`, `scripts/log_deploy.py` |
| Env approval gate | `cd.yml` job `terraform-apply` (L112) | `environment: production` = required manual approval in Settings → Environments — the ONLY human act Phase 9 keeps |
| Metric registry contract | `observability/metrics.py` `METRIC_REGISTRY` + `tests/observability/test_telemetry_contract.py` | AST test: ANY new `wandb.log` key MUST be registered or CI fails |
| Current Champion | W&B `eval-champion` collection `champion` alias | `model-qwen3-14b-higher_lr_14b`; F2P 16.9% (Wilson CI 8.2–27.9%), P2P 91.2%, n=50 golden, promoted 2026-08-06 |
| Offline E2E | in-memory `EvalRun`/`EvalResult` fixtures | deterministic per-instance 0/1 vectors → paired gate, no GPU, CI-safe. `evaluation/local_backend.py` is **dev-box only** (live Ollama + real git/pytest per instance — hours, network-bound, non-deterministic), NOT for CI |

## 3. Decision log (grill-adopted)

| # | Decision | Consequence enforced in code |
| --- | --- | --- |
| 1 | `promotion/` **composes** `evaluation/`; no code moved, no reimplementation. `eval compare` keeps working. | `promotion/gate.py` wraps `revalidate_champion` + builds per-instance vectors calling `stats` directly (NOT `paired_significance` — it cross-run-same-variant only) |
| 2 | Promote iff: (a) floors `p2p ≥ 0.90` & `f2p ≥ 0.15` (ADR-014) pass; (b) `candidate_f2p ≥ champion_f2p + 0.05`; (c) paired-bootstrap CI lower bound of the F2P gain **> 0**; (d) `candidate_p2p ≥ champion_p2p − 0.02`. Reject reasons: `fatal-flaw` (floor/margin), `regression` (P2P drop over ceiling), `micro-gain` (margin passes but CI includes 0). | `promotion/rules.py` — pure function, defaults env-overridable (`PROMOTE_MIN_F2P_GAIN`, `PROMOTE_MAX_P2P_REGRESSION`) |
| 3 | Trigger = auto `challenger` tag + `promote.yml` dispatch from training completion. The ONLY human act = approving the deploy job's `environment: production` gate (exactly like `terraform-apply`). No CLI, no polling. | `scripts/tag_challenger.py`; hook in `scripts/run_3config_comparison.py`; `promote.yml` |
| 4 | Fresh **paired eval of BOTH models each cycle** at `dev` tier (100, seed 42). Champion re-evaluated in the same run window — fair, self-reinforcing; winner's metrics become the next baseline. Same `dataset_run_id` + `tier_seed` asserted from embedded `EvalRun.config`. | `promotion/gate.py` asserts pairing; builds per-instance vectors from `run.results` (variant-filtered, intersected); `promotion/run.py` launches 2 Modal eval runs |
| 5 | Source of truth = **`gs://swe-qwen-datasets/ci/champion.json`** (mirror of `smoke_baseline.json`: same bucket, workflow-written). W&B `champion` alias stays as human-facing pointer, synced from the record. `ci/` is EMPTY today — `seed_champion.py` (needs GCP auth) runs once before the first promote. | `promotion/registry.py` read/write + alias sync |
| 6 | Deploy = `uv run modal deploy -m inference.modal_serve` from the promote workflow's deploy job (secrets already in repo). The **variant must be pinned BEFORE deploy**: `ServeConfig` resolves `default_variant` from `SERVING_DEFAULT_VARIANT` env (fallback hardcoded); if the candidate variant is not in `ServeConfig.variants` → abort with a `config-gap` reason (v1 promotes only among trained variants). Then health probe on **`POST /v1/chat/completions`** (with `MODAL_SERVE_TOKEN`) — `GET /health` is static, a liveness pre-check only. Alias sync failure (`promote_champion_to_registry` returns `None`) aborts the deploy job. | `promotion/deploy.py` + `inference/config.py` + workflow deploy job |
| 7 | Rollback is IN SCOPE: previous champion (persisted in champion.json) is re-promoted through the SAME pipeline (decision record, deploy, probe). | `promotion/deploy.py::rollback` |
| 8 | Step 0 prerequisite: the served endpoint is PUBLIC — add Bearer-token auth (`Authorization: Bearer <MODAL_SERVE_TOKEN>`, from a **new Modal Secret**, which must exist in the Modal workspace before the first `modal deploy` or deploy fails), then one **one-time redeploy** through the new deploy code proves it before the first real promotion. No AsyncLLM fix exists; nothing else to ship. | `inference/modal_serve.py` |
| 9 | Audit trail = W&B artifact `promotion-decision-{id}` (JSON + markdown) + `promote/*` scalar namespace registered in `observability/metrics.py` (contract test enforces). `promotion-decision-*` artifacts are immutable; deploy outcome appended via artifact metadata update + `promote/deploy_status` scalar. | `promotion/audit.py` |
| 10 | Positioning for recruiters: the complete self-driving loop — auto challenger entry → paired eval → statistical gate → promotion → deploy → smoke probe → audit → rollback — where **rigor and auditability are inseparable from automation**. | everything above |
| 11 | `promote.yml` structure: `init-wandb` (reuse `scripts/init_wandb.py` pin) → `decide` (validate challenger input — variant ∈ `ServeConfig.variants`; skip all Modal work when `RUN_MODAL_EVAL=false`; paired eval; rules; decision record; sets `promote=true` output but does NOT touch champion.json) → `deploy` (`if needs.decide.outputs.promote == 'true'`, `environment: production`, `modal deploy` → probe → only after probe green: write champion.json + sync alias; on probe/alias failure: rollback previous champion through the same path). Rejected decisions exit 0 — rejection is a successful outcome, not a failure. | `.github/workflows/promote.yml` |

## 4. Changes (by file)

### 4.1 `promotion/rules.py` (NEW — task 9.2)

Pure module, no I/O, no imports of cloud deps (offline-testable).

- Constants (env-overridable): `PROMOTE_MIN_F2P_GAIN = float(os.getenv("PROMOTE_MIN_F2P_GAIN", "0.05"))`, `PROMOTE_MAX_P2P_REGRESSION = float(os.getenv("PROMOTE_MAX_P2P_REGRESSION", "0.02"))`.
- `OUTCOME_PROMOTE`, `OUTCOME_REJECT = ("fatal-flaw" | "regression" | "micro-gain")` literals.
- `decide(champion_f2p, candidate_f2p, champion_p2p, candidate_p2p, ci_lower) -> (outcome, reasons: list[str])`:
  1. floors: `candidate_f2p < 0.15 or candidate_p2p < 0.90` → `fatal-flaw`;
  2. margin: `candidate_f2p < champion_f2p + 0.05` → `fatal-flaw`;
  3. P2P ceiling: `candidate_p2p < champion_p2p − 0.02` → `regression`;
  4. significance: `ci_lower <= 0` → `micro-gain`;
  5. else `PROMOTE`, reasons `[]`.
- Exact-boundary behavior unit-tested (== is NOT a pass anywhere).

### 4.2 `promotion/gate.py` (NEW — task 9.1)

- `@dataclass PairEval`: `{champion_metrics, candidate_metrics, f2p_gain, p2p_delta, ci_lower, ci_high, mcnemar_p}`.
- `evaluate_pair(champion_run: EvalRun, candidate_run: EvalRun, config: EvalConfig) -> PairEval`:
  - assert `champion_run.config.dataset_run_id == candidate_run.config.dataset_run_id` and equal `tier_seed` (embedded config ⇒ provable pairing);
  - **build per-instance vectors itself** from `run.results`, variant-filtered, `instance_id`-intersected:
    `cand = {r.instance_id: 1.0 if r.f2p > 0 else 0.0 for r in candidate_run.results if r.variant == candidate}` (same for champion), intersect keys;
  - significance calls `stats.paired_bootstrap_ci(cand_vec, champ_vec)` **directly** — candidate is arg `a` so `lo > 0` ⇒ candidate beats champion (N=100, 10 000 boots, seed 42) — plus `stats.mcnamar_p(...)`;
  - `comparison.paired_significance` is NOT used for the gate (it compares the same variant across two runs; with different variants it silently returns "no overlap"). Display only.
- `revalidate` = thin passthrough to `comparison.revalidate_champion` (keeps floor semantics identical).

### 4.3 `promotion/registry.py` (NEW — task 9.3)

- `read_champion(path) -> ChampionRecord` / `write_champion(path, record)` — `champion.json` schema: `{"variant", "model_ref", "f2p_rate", "p2p_rate", "dataset_run_id", "tier", "seed", "promoted_at", "previous": {...or null}}` (self-referential ⇒ rollback needs no extra state).
- `sync_alias(champion_key, config)` — calls `comparison._clear_champion_alias` + `promote_champion_to_registry`; idempotent (alias is derived from champion.json, never a second source of truth).
- Lazily imports `wandb` (repo convention: offline tests pass without cloud).

### 4.4 `promotion/audit.py` (NEW — task 9.5)

- `build_decision_record(...)` → dict: `{decision_id, pipeline_version, candidate: {run_id, variant, model_ref}, incumbent: {...}, metrics: {candidate, incumbent, f2p_gain, p2p_delta, ci_lower, ci_high, mcnemar_p}, thresholds: {min_f2p_gain, max_p2p_regression, floors}, outcome, reasons, deployed: false, timestamps, git_sha}`.
- `write_decision_record(record)` — lazy wandb: `wandb.Artifact(f"promotion-decision-{record_id}", "decision")`, `.add` the JSON + a generated markdown summary; log scalars as **literal** `log_metrics({...})` with the six registered `promote/*` keys (or via `assert_registered`) — the telemetry AST walker resolves exact keys, and a templated dict would fail `test_telemetry_contract` (see 4.11).
- `note_gating_off(candidate_ref, reason)` — used when `RUN_MODAL_EVAL=false`.

### 4.5 `promotion/deploy.py` (NEW — task 9.4)

- `deploy(champion_key, config) -> subprocess.CompletedProcess` — **variant pinning first**: assert candidate variant ∈ `ServeConfig.variants` else raise `ConfigGapError` (v1 promotes only among trained variants); pin the served default to the champion via the `SERVING_DEFAULT_VARIANT` env that `inference/config.py::ServeConfig` resolves at app start (fallback: hardcoded `higher_lr_14b`), then `uv run modal deploy -m inference.modal_serve` with `MODAL_TOKEN_ID/SECRET`, `HF_TOKEN` env. The deployed app reads `SERVING_DEFAULT_VARIANT` from a Modal Secret / secret-file update performed by the deploy step — the pin is only done when the probe proves the served default matches the champion.
- `health_check(base_url, token, ttfpb_target_ms=500)` — **probes `POST /v1/chat/completions`** with `Authorization: Bearer <token>` + a short generation, measures TTFB; `GET /health` is only a cheap liveness pre-check (static, no TTFB, no auth).
- `sync_alias_or_abort(champion_key, config)` — calls `comparison.promote_champion_to_registry`; **`None` ⇒ abort deploy** (never deploy stale). Runs AFTER the probe passes.
- `rollback(previous: ChampionRecord, config)` — re-promotes `previous` through the same deploy+probe+alias path; returns rollback record (reuses the audit pipeline — symmetric and audited).
- `dry_run = True` in the unit/E2E path (never deploys for real in tests).

### 4.6 `promotion/run.py` (NEW — the `decide` job entrypoint)

`python -m promotion.run --candidate-variant <v> [--no-eval]` — the candidate's paired eval run ids are generated by `run.py` itself; there is no pre-existing `candidate_run_id` (there is nothing to "detect" — the input is a label).

- Reads `champion.json` (downloaded to the workspace by the workflow step via `gcloud storage cp`; `create_anonymous_client()` fallback for local/dev) → incumbent ref;
- Validates challenger: variant ∈ `ServeConfig.variants` and `challenger` tagged in W&B (else abort with `config-gap` / `no-challenger` record);
- Unless `--no-eval`: launches a paired `eval run --mode dev` for incumbent + candidate (2 Modal runs, est. $1–4), polls to completion;
- `evaluate_pair` → `rules.decide` → `write_decision_record`;
- On PROMOTE: writes `promote=true candidate=... champion=...` to `$GITHUB_OUTPUT` — job outputs are read from `$GITHUB_OUTPUT`, **not stdout**; plain prints would leave `needs.decide.outputs.*` empty and the deploy job would silently no-op after spending GPU. Does **NOT write champion.json** (single writer: the deploy job, after the probe; avoids a champion recorded but never deployed);
- On REJECT: prints `promote=false <reason>`, exits 0.
- `record.tier` = the `--mode` value run.py launched (**never `config.tier`** — `EvalConfig` has no `tier` field and `extra="ignore"` silently drops it; `tier` exists only as the `eval run --mode` CLI arg).

### 4.7 `scripts/tag_challenger.py` (NEW) + `scripts/run_3config_comparison.py` hook

- `tag_challenger.py` — lazy wandb; tags the finished training `model-qwen3-14b-{variant}:latest` artifact with `challenger` (+ keeps `latest`). Retry-able (idempotent).
- `run_3config_comparison.py` end-of-run: (1) `python scripts/tag_challenger.py --variant <winner>`, (2) if `gh repo variable get RUN_MODAL_EVAL` != "false": `gh workflow run promote.yml -f candidate_variant=<variant>`.
- `scripts/seed_champion.py` (NEW, one-shot) — writes the 2026-08-06 Champion (`higher_lr_14b`, F2P 0.169, P2P 0.912, dataset_run_id=expanded-repos, n=50) into champion.json so the loop has a baseline on first run; its `record.tier = "full"` (n=50 matches `tier_sizes["full"]` — `EvalConfig` has no `tier`, see 4.6). **Requires GCP credentials** (gcloud auth application-default or a workflow job with WIF) — `ci/` is empty until this runs.

### 4.8 `inference/modal_serve.py` (CHANGED — Step 0)

- Require `Authorization: Bearer <token>` on chat completions; token from Secret env `MODAL_SERVE_TOKEN` (lazy, no hardcode); 401 otherwise. Everything else untouched. Step-0 checklist includes **creating the Modal Secret first** (`modal secret create` the serve-token secret before the one-time redeploy — `modal deploy` fails if a referenced secret is absent, and adding it to `_secrets` would red cd.yml's `deploy-modal` + benchmark paths too).

### 4.9 `observability/metrics.py` (CHANGED)

- Register exact new keys in `METRIC_REGISTRY` under `promote/*`: `promote/outcome` (0/1), `promote/f2p_gain`, `promote/p2p_delta`, `promote/ci_lower`, `promote/mcnemar_p`, `promote/deploy_status`. The AST contract test fails until this is done — but note it only walks `_WALK_DIRS` (see 4.11): `"promotion"` must be added there too or the enforcement never sees the keys.

### 4.10 `.github/workflows/promote.yml` (NEW)

- `on: workflow_dispatch` (inputs: `candidate_variant` only).
- `env: DATASET_RUN_ID: expanded-repos`, `EVAL_DATASET_RUN_ID: expanded-repos` (same as eval.yml).
- Job `init-wandb` — `uv run python scripts/init_wandb.py --entity 2571642-university-of-dundee` (project pin; auto-deleted projects bite otherwise); `env WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}`.
- Job `decide` (needs init-wandb, timeout 240 min, `permissions: id-token: write` + `google-github-actions/auth@v3` + `google-github-actions/setup-gcloud@v2`, `env WANDB_API_KEY + MODAL_TOKEN_ID/SECRET + HF_TOKEN`):
  - downloads `gs://swe-qwen-datasets/ci/champion.json` (if any) to the workspace with `gcloud storage cp` — reads are public but the runner needs auth/ADC;
  - Modal kill-switch: `if [ "${{ vars.RUN_MODAL_EVAL }}" = "false" ]` → `python -m promotion.run --no-eval` writes `gating-off` audit note + `modal token info`, exits 0. **Unlike eval.yml, promote skips even on `main`** (workflow_dispatch ref is `refs/heads/main`) — intentional, this is not a PR gate;
  - else `python -m promotion.run --candidate-variant ${{ inputs.candidate_variant }}`;
  - uploads decision artifact; declares job `outputs: {promote, champion, candidate}` mapped from the run.py step (**`id: decide`** on that step — outputs only exist if the step writes `$GITHUB_OUTPUT` and the job-level `outputs:` map references it);
  - **publish decision summary for the human approver (terraform-plan parity)**: `promotion/audit.py` also writes `promotion-decision-<id>.md` to `$GITHUB_STEP_SUMMARY` AND `actions/upload-artifact` (name `promotion-decision-<id>`, retention 7). The person approving the deploy job must be able to read exactly what they are approving — same affordance cd.yml gives the terraform plan.
- Job `deploy` (needs decide): `if: needs.decide.outputs.promote == 'true'`, **name: `Deploy Champion ${{ needs.decide.outputs.champion }}`** (job names render in the "Review deployments" approval screen, so the human sees WHAT they are approving without opening anything), `environment: production` (HUMAN GATE), `timeout-minutes: 60`, `permissions: id-token: write` + GCP auth, `env WANDB_API_KEY + MODAL_TOKEN_ID/SECRET + HF_TOKEN`:
  0. first step writes the run's step summary: decision id, champion→candidate F2P/P2P before-after, reason list, artifact link — `echo`ed from the downloaded decision record, before any deploy work starts;
  1. `python -m promotion.run --deploy --variant ${{ needs.decide.outputs.champion }}` → asserts variant known + `SERVING_DEFAULT_VARIANT` pinned-aware, `modal deploy`, probe `POST /v1/chat/completions`;
  2. **only after probe green**: write `champion.json` (with `previous`) + `sync_alias_or_abort` (alias failure aborts);
  3. appends `deploy_status=success` + timestamp to the decision artifact metadata and logs `promote/deploy_status`; on probe/alias failure: `rollback` (re-promote previous) + decision record gets `deploy_status=rollback` — **AC: a variant is recorded as champion only if it is deployed and green**.
- `concurrency: promote-decision-${{ github.ref }}`, `cancel-in-progress: false`.

### 4.11 `tests/`

- `tests/test_promotion.py` (NEW — task 9.6): rule matrix — promote happy path; `fatal-flaw` via floor fail / margin fail; `regression` P2P drop; `micro-gain` CI-lower==0 boundary; exact-constant boundaries (gain exactly 0.05, regression exactly 0.02, ci_lower exactly 0.0 → all REJECT); `decide` never crashes on missing reasons; `evaluate_pair` assert fires on mismatched `dataset_run_id`/`tier_seed`; decision-record JSON shape (frozen field set); `registry.write_champion/read_champion` round-trip incl. `previous` chain; `deploy` respects `dry_run`.
- `tests/test_champion_auth.py` (NEW): endpoint 401 without token / 200 with `MODAL_SERVE_TOKEN`.
- `tests/test_promotion_e2e.py` (NEW — task 9.7): offline E2E using **in-memory `EvalRun`/`EvalResult` fixtures** (deterministic per-instance 0/1 vectors — no LLM, no Ollama, CI-safe by construction) — seed fake champion.json → candidate run vs champion run → `decide` → assert accept path prints `promote=true`; assert reject path exits 0 with a written decision record; `deploy` on `dry_run` asserts the trigger command is constructed, never executed. (`evaluation/local_backend.py` is dev-only: live Ollama + real git/pytest per instance — hours, network-bound, non-deterministic.)
- `tests/observability/test_telemetry_contract.py` (CHANGED expectation): add `"promotion"` to `_WALK_DIRS` (the AST walk only covers listed dirs — without it the contract never sees `promote/*`) + new keys list includes `promote/*`; no dashboard change needed (`test_panels_spec_subset_of_registry` only checks PANELS ⊆ registry).
- **Package integration (else the gates silently miss `promotion/`)**: add `promotion` to `[tool.setuptools.packages.find] include` (pyproject), `[tool.coverage.run] source` (pyproject), `[tool.mypy] files` (pyproject), and to the ruff (ci.yml L64), ruff-format (L67), and mypy (L70) path lists.
- **Coverage floor (user lock-in)**: `promotion/` CI gate is **`--cov-fail-under=95` line+branch** (empirically: a 6-module package with subprocess/argparse/except-paths cannot honestly hold 100% — a trivial 2-file module scored 85%/84% with the same command shape), with **100% as per-module intent, not the gate**. Run it in `ci.yml` as its own step **before** the repo-wide fast invocation (pytest-cov erases `.coverage` without `--cov-append`, so the promotion pass must not piggyback after the existing invocations or it would destroy the repo-wide file the codecov step uploads): `uv run pytest tests/test_promotion.py tests/test_promotion_e2e.py --cov=promotion --cov-branch --cov-fail-under=95` (or emit an isolated `coverage-promotion.xml`). Carve-outs enumerated up front (repo convention: `# pragma: no cover` + `exclude_lines`, cf. `observability/langfuse.py`): the `subprocess` exec + `--no-eval` host paths in `run.py`, the `modal deploy`/probe exec lines in `deploy.py` (guarded by `dry_run`), the `create_anonymous_client` fallback in `run.py`, and the `ImportError` branch of every lazy wandb import (add a test that forces `import wandb` to raise). `rules.py` (the decision core) is branch-tested: every branch exercised by `tests/test_promotion.py`, not just lines. NB: Phase 8's "100% of new code" narrative was **never CI-enforced** (ci.yml has no per-package `--cov=` invocation today) — this 95-gate IS wired into ci.yml and will actually run.

## 5. Task mapping (master-plan)

| Task | Deliverable |
| --- | --- |
| 9.1 comparison engine | `promotion/gate.py` (4.2) |
| 9.2 promotion rules | `promotion/rules.py` (4.1) |
| 9.3 W&B registry integration | `promotion/registry.py` (4.3) + `tag_challenger.py` (4.7) |
| 9.4 deployment trigger | `promotion/deploy.py` (4.5) + promote.yml deploy job (4.10) |
| 9.5 audit trail | `promotion/audit.py` (4.4) + `promote/*` registry keys (4.9) |
| 9.6 unit tests | `tests/test_promotion.py` (4.11) |
| 9.7 E2E promotion test | offline `tests/test_promotion_e2e.py` (4.11) |

## 6. Implementation order (dependency-topological)

0. **Auth + first redeploy** (Step 0, 4.8): create the `serve-token` Modal Secret first, then prove `promotion/deploy.py`'s command path before any real promotion. Redeploy once via the new path.
1. **`promotion/rules.py`** — pure, TDD (write `decide` tests first; they ARE the acceptance tests for 9.2).
2. **`promotion/gate.py`** — per-instance vectors from `run.results` + direct `stats.paired_bootstrap_ci`/`mcnamar_p` (candidate as arg `a`); validate pairing asserts.
3. **`promotion/registry.py`** — champion.json read/write/alias; `seed_champion.py` one-shot baseline.
4. **`promotion/audit.py`** + `observability/metrics.py` `promote/*` keys (contract test must pass).
5. **`promotion/deploy.py`** — deploy/probe/rollback with `dry_run`.
6. **`promotion/run.py`** — orchestration entrypoint.
7. **`scripts/tag_challenger.py`** + `run_3config_comparison.py` hook.
8. **`.github/workflows/promote.yml`** — wiring; `RUN_MODAL_EVAL` kill switch wired first.
9. **E2E test** with in-memory fixtures (no local backend); then `ci.yml` package wiring (pyproject/mypy/coverage/`_WALK_DIRS`), full suite + ruff/mypy (+ per-module coverage ≥95% for `promotion/`).
10. **Docs**: IMPLEMENTATION-LOG Phase 9 entry; README one-liner. (No CONTEXT.md — out of scope per user.)

## 7. E2E runbook

```
# first cycle — challenger already tagged, no training (higher_rank_14b; will REJECT, correct)
gh workflow run promote.yml -f candidate_variant=higher_rank_14b   # eval run ids generated by run.py
# future cycles: run_3config_comparison.py tags the new checkpoint + dispatches automatically
# → init-wandb → decide (GCP auth; paired dev eval ~$1-4, rules, decision record)
# → deploy job sits on `environment: production` → human approves in Actions UI
# → modal deploy (SERVING_DEFAULT_VARIANT pinned) → TTFB probe on /v1/chat/completions
# → probe green → champion.json + alias written → audit artifact (deploy_status=success)
# RUN_MODAL_EVAL=false ⇒ decide writes gating-off note, exits 0, $0 spent (even on main)
```

## 8. Acceptance (master-plan ACs + DoD, adapted per decisions)

- [ ] AC1 champion.json seeded (2026-08-06 champion) via `seed_champion.py`.
- [ ] AC1-authentication: endpoint 401 without `MODAL_SERVE_TOKEN`, 200 with (pre-existing gap, closed Step 0).
- [ ] AC-auto-entry: on training completion, checkpoint tagged `challenger`; promote.yml dispatched automatically (no human run).
- [ ] Pairing: same `dataset_run_id` + `tier_seed` asserted (embedded config) — matched per-instance.
- [ ] Promote iff floors ∧ margin ≥ 0.05 ∧ CI-lower > 0 ∧ P2P regression ≤ 0.02 (rules unit-tested incl. boundaries).
- [ ] On accept: Modal redeploy with `SERVING_DEFAULT_VARIANT` pinned (variant ∈ `ServeConfig.variants` or `config-gap` reject); probe green on `/v1/chat/completions`; **then** champion.json + W&B alias updated (probe-before-record); all via `environment: production` gate.
- [ ] On reject: `promotion-decision-*` artifact with reason + metrics; workflow exits 0.
- [ ] Approver clarity (terraform-plan parity): decision summary in `decide` job summary + artifact; `deploy` job name embeds the champion variant; approving = reading the decision, not blind-clicking.
- [ ] On deploy-fail: previous champion re-promoted, rollback decision artifact recorded.
- [ ] Promote path never spends a GPU dollar when `RUN_MODAL_EVAL=false` (skips even on `main`).
- [ ] `promote/*` keys registered AND `"promotion"` added to `_WALK_DIRS`; AST contract test green; full `ci.yml` (ruff, mypy, pytest) green.
- [ ] `promotion/` wired into pyproject (packages include, coverage source, mypy files) + ci.yml ruff/mypy paths; **≥95% line+branch, CI-gated** (`--cov-fail-under=95`, own step before the repo-wide invocations; 100% per-module intent with enumerated pragma carve-outs).
- [ ] DoD: unit + E2E cover ALL rule scenarios and the full accept chain; manual deploy decisions impossible by construction (only env approval gate triggers deploy).

## 9. Risks / watch-items

- **Threshold tuning**: 0.05 gain is a guess (user: adjust after 3–5 cycles). Env-overridable; each decision record stores the thresholds used ⇒ tuning is itself auditable.
- **Wide bootstrap CI at n=100**: the `CI-lower > 0` gate is deliberately strict — it makes few champs but beats champion-noise. Record `ci_high` so a "close call" is visible; no special-casing without a decision record.
- **Alias race vs. manual `eval compare`**: `eval compare` still flips the alias outside the pipeline. champion.json remains the source of truth; alias sync is idempotent and derived. Risk accepted (Phase 11 candidate: `--write-registry` default-off on compare).
- **Rejected ≠ failure**: workflow must exit 0 on reject so operator bookkeeping doesn't page on healthy signal.
- **Endpoint is public until Step 0 lands** — Step 0 is first, not last.
- **Deploy quota/timeouts**: Modal fn cap 300 min; GitHub job `timeout-minutes: 240`; `modal deploy` cold-start noted in deployment record.

## 10. Glossary (terms resolved this session)

- **Champion**: the deployed best-performing model as of the last accepted promotion. Recorded in `champion.json` (source of truth) with a W&B `champion` alias pointer.
- **Challenger**: a candidate checkpoint tagged `challenger` on the W&B artifact, eligible for a paired promotion evaluation.
- **Promotion decision**: the accept/reject verdict of `rules.decide` over a paired eval — the unit of audit (one `promotion-decision-*` artifact each).
- **Paired evaluation**: evaluating Champion and Challenger on the same golden instances in the same run window; the per-instance match is what makes bootstrap/McNemar valid.
- **Rollback**: re-promoting the previous Champion through the same audited pipeline — symmetric, not a manual revert.

## 11. Proposed ADRs (offered for ADR-&-VISION.md)

- **ADR-019 — Champion-of-record is `champion.json`; the W&B alias is a derived pointer.** Rationale: GCS JSON survives alias churn, is diffable/history-able, and mirrors the Phase 7 smoke baseline pattern; the alias is for humans.
- **ADR-020 — Promotion is chained-and-gated, never manual-CLI.** Reconciles Phase 7's "promotion stays human-triggered": the trigger is now workflow-driven, the ONLY human act is the deploy environment approval — the same trust boundary cd.yml already uses.

## 12. Effort estimate

~1.5–2 focused dev days (pure rules/gate/audit are small; workflow wiring + first redeploy dominate) + est. **$1–4 per promotion cycle** (paired dev tier) with `RUN_MODAL_EVAL=false` guaranteeing $0 at rest.
