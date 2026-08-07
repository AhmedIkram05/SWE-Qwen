# Phase 7 — CI/CD with Quality Gates: Implementation Plan

Status: reviewed-APPROVED-WITH-FIXES (all fixed) | Owner: Ahmed | Date: 2026-08-06

## 1. Goal

Wire SWE-Qwen's existing pieces (Phase 2 CI, Phase 5b eval CLI + smoke gate,
Phase 6 Modal serving, Phase 1 Terraform/OIDC) into automated CI/CD with an
eval-based quality gate that blocks regressions of the champion model — and an
absolute F2P floor — before a PR merges or a model ships. Builds on completed
phases only; all heavy lifting already exists in the eval CLI.

Non-goals (explicit, locked):

- NO auto-retraining on new data, ever. Training/promotion stay human-triggered. (Ahmed, 2026-08-06)
- No candidate-promotion machinery (compare/revalidate_champion promotion) — that is Phase 9. Phase 7's gate is regression surveillance of the current champion + a literal absolute floor.
- No infra for branch protection (manual GitHub UI setup, §6 step 1).
- No automated Modal deploy until Phase 6.8 (auth + AsyncLLM) lands; Phase 7 declares a hard dependency on 6.8 for that piece.

## 2. What already exists (reuse, don't rebuild)

| Piece | Where | Notes |
| --- | --- | --- |
| GCP OIDC (WIF) | `.github/workflows/cd.yml` job `terraform-plan` | `google-github-actions/auth@v3`, vars `GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`. SA `github_actions` has `roles/storage.admin` (+ secretAccessor, artifactregistry.writer, cloudbuild.editor, bigquery.jobUser) — **CI can read/write GCS with zero new secrets** (verified: iam/main.tf L183-188) |
| lint + typecheck + unit tests | `ci.yml` job `lint-and-test` | ruff check+format, mypy, fast+slow pytest w/ coverage, codecov upload |
| Terraform validate | `ci.yml` job `terraform-validate` | `init -backend=false` + `validate` + `fmt -check` — static gate on every PR. Plan moved to cd.yml (plan-on-PR, apply gated by `production` env) per repo conventions |
| Smoke gate | `evaluation/cli.py` `_smoke_gate` (L390-438) | fires on `--mode smoke` (call site L117-118); baseline `{output_dir}/smoke_baseline.json`; fails `rate < baseline - 0.05`; bootstrap writes+passes; update-on-pass ratchet (being fixed) |
| Eval config | `evaluation/config.py` | `EVAL_` env prefix; `min_f2p_threshold=0.15`, `min_p2p_threshold=0.90`, `ci_sample_size=50`, `ci_random_seed=42`, `tier_sizes["smoke"]=20`, `dataset_run_id`, `golden_data_path=gs://swe-qwen-datasets/datasets/{run_id}/swebench/golden.jsonl`, `wandb_entity="2571642-university-of-dundee"` |
| GCS public golden pattern | `inference/prompt_builder.py::_ensure_golden` L174 | urllib on public bucket, local cache, offline fallback — reused for baseline download |
| W&B project pin | `scripts/init_wandb.py` | `init_wandb_project("swe-qwen")` — project auto-deleted once (observed); must be re-pinned before eval |
| CD deploy target | `inference/modal_serve.py` | Modal 1.5.3 requires `-m`: **`modal deploy -m inference.modal_serve`** |

## 3. Decision log (grill-adopted)

| # | Decision | Consequence enforced in code |
| --- | --- | --- |
| 1 | Gate = champion **regression surveillance** (PR blocked if champion smoke F2P drops >5% vs baseline) **+ literal absolute floor now**: fail if `rate < config.min_f2p_threshold` (0.15) regardless of baseline. Phase 9 refines. | one-line addition in `_smoke_gate` |
| 2 | Baseline is **CD-owned**: PR evals READ-only; the `push . main` eval-gate run WRITES via `--update-baseline` (bootstrap on first main push; monotonic update thereafter). | `--update-baseline` flag; push vs PR mode at workflow level |
| 3 | Baseline file = `{"dataset_run_id": ..., "rates": {model:variant:prompt: f2p}}`. Stale `dataset_run_id` ⇒ re-bootstrap regime (no silent cross-dataset comparison). | gate compares run_id |
| 4 | Writes are monotonic: `rates[key] = max(new, prev, min_f2p_threshold)` — kills ratchet decay, guarantees baseline ≥ floor. | in gate |
| 5 | W&B stays the LoRA artifact source; `init-wandb` head job pins the project (with the harness's entity) before any artifact-resolving job. GCS champion mirror = Phase 9. | eval.yml job ordering |
| 6 | Zero changes to the eval *runner* (harness/inference/containers). Code delta = `cli.py` gate + `config.py` one value + tests + workflows. | — |
| 7 | Sampling is deterministic per fixed golden.jsonl (seed-42 `random.sample`, verified harness L1324); dataset regen ⇒ new run_id ⇒ decision 3 handles it. | — |
| 8 | **Smoke tier measures the real quality regime**: `tier_max_new_tokens["smoke"]` 2048 → **8192** (one config value). 2048 is the historically documented *truncation* regime for 14B (IMPLEMENTATION-LOG L865-866); a truncation-lottery gate would make the literal 15% floor unmeterable. Cost delta ≈ +$0.20-0.40/run, negligible for a merge-blocking gate. | config.py edit |

## 4. Changes

### 4.1 `evaluation/cli.py`

- `run` gains `--update-baseline` (Typer bool, default False).
- `_smoke_gate(eval_run, config, update_baseline: bool = False)`:
  - baseline file schema `{"dataset_run_id": ..., "rates": {...}}`; read both shapes (old flat dict tolerated via normalization), write new shape only.
  - **Absolute floor (the one-liner)**: a key fails if `rate < config.min_f2p_threshold` — independent of baseline, no tolerance.
  - Missing baseline → if not `update_baseline`: exit 1 "no baseline; rerun with --update-baseline to bootstrap"; if yes: write baseline `rates={k: max(v, floor)}`, pass.
  - Stale run_id → same as missing (re-bootstrap regime).
  - Comparison: fail if `rate < baseline[key] - _SMOKE_TOLERANCE` or `rate < min_f2p_threshold`; baseline keys absent from run still fail.
  - On pass: if `update_baseline`, `rates[key]=max(new, prev, floor)` + write; else print "read-only (baseline unchanged)".
- Call site L117-118 → pass `update_baseline`.

### 4.2 `evaluation/config.py`

- `tier_max_new_tokens["smoke"]: 2048 → 8192` (decision 8). Check/update any test asserting 2048.

### 4.3 Tests (`tests/test_eval_review_fixes.py`)

- Rewrite 2 tests for new semantics: `test_first_run_writes_baseline` → must now pass `update_baseline=True`; `test_within_tolerance_updates_baseline` → read-only default leaves file unchanged; write-mode update asserts monotonic `max(new, prev, floor)`.
- Add: bootstrap-with-flag writes floor-wrapped rates; floor blocks below-threshold despite healthy baseline; stale run_id re-bootstraps; read-only pass leaves file untouched. Corrupt/empty/missing/drop tests keep passing unchanged. (Signature compatible: new param defaults False; returns None, raises typer.Exit(1).)

### 4.4 `.github/workflows/ci.yml`

- Add `--cov-fail-under=75` to the slow-pytest step (completes 7.3 coverage check). Additive vs existing `--cov --cov-report=xml --cov-append -m "slow and not requires_credentials"` — pytest-cov combines the appended data. Calibrate on first run (one string).

### 4.5 `.github/workflows/eval.yml` (NEW — 7.4+7.5)

- Triggers: `pull_request` + `push` to `main`; `paths`: `evaluation/**, training/**, inference/**, config/**, models.yaml, qlora_variants.yaml, training/prompts/**, pyproject.toml, uv.lock, .github/workflows/eval.yml`. (Config-file changes re-validate the champion — AC-2 minimal obligation; no `templates/**` — that dir doesn't exist.)
- `env`: `DATASET_RUN_ID: expanded-repos`, `EVAL_DATASET_RUN_ID: ${{ env.DATASET_RUN_ID }}`.
- `concurrency: group: ci-eval-${{ github.ref }}, cancel-in-progress: true`.
- `timeout-minutes: 240`.
- Jobs (each: checkout, astral-sh/setup-uv@v5, setup-python@v5, uv sync --extra dev):
  1. `init-wandb` — `WANDB_API_KEY` secret → `uv run python scripts/init_wandb.py` (re-pin project; entity must match harness: pass harness's entity so project lands in the right namespace).
  2. `eval-gate` (needs init-wandb) — permissions `id-token: write, contents: read`; `google-github-actions/auth@v3` + `google-github-actions/setup-gcloud@v2` (ADC for gcloud):
     - download: `gcloud storage cp gs://swe-qwen-datasets/ci/smoke_baseline.json data/eval_results/` — non-fatal if absent (gate handles missing).
     - run — **PR**: `uv run eval run --mode smoke --models qwen3-14b:higher_lr_14b` (read-only, no flag). **push/main**: same + `--update-baseline` (CD-owned write; bootstraps on first main push). Secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (required; runner-side `WANDB_API_KEY`/`HF_TOKEN` optional — artifact loading runs inside Modal via its own `wandb-secret`/`hf-secret`, golden data is public GCS).
     - upload (results preservation, baseline excluded): `gcloud storage cp data/eval_results/ gs://swe-qwen-datasets/eval/${DATASET_RUN_ID}/ --recursive --exclude '**/smoke_baseline.json'`.
  - Both triggers gate merges; `continue-on-error: false`.

### 4.6 `.github/workflows/cd.yml` (NEW — 7.6, convention from stocklens/laad/w3c-etl)

Terraform plan lives in CD, not CI. PRs open a reviewed plan; apply is a separate job
gated by the `production` GitHub Environment (manual approval):

- Triggers: `pull_request` + `push` on main (both `paths: infra/**, inference/**, config/**, pyproject.toml, uv.lock, .github/workflows/cd.yml`) + `workflow_dispatch`.
- Job `terraform-plan` (all triggers): WIF auth → terraform init (real GCS backend) → `terraform plan -lock=false -no-color -out=tfplan -var=... env=dev`, tolerating exit 2 (`|| { rc=$?; [ $rc -eq 2 ] && true || exit $rc; }`) → `terraform show -no-color tfplan` to job summary → upload artifact `tfplan-${{ github.sha }}` (retention 7).
- Job `terraform-apply` (`needs: terraform-plan`): `if: github.event_name != 'pull_request'`; **`environment: production`** (manual approval via Settings→Environments; default blocked until reviewers approve) → downloads `tfplan-${{ github.sha }}` → `terraform apply tfplan` (applies the reviewed plan, no re-plan). Applies only ever with an approved plan.
- Job `deploy-modal` (`if: github.event_name == 'workflow_dispatch'`, needs `MODAL_TOKEN_ID/SECRET`, `HF_TOKEN`): **`uv run modal deploy -m inference.modal_serve`**. Manual until Phase 6.8 (auth + AsyncLLM); comment in-file: "flip to on-push after 6.8". NOT claimed as delivered-fully without 6.8 (hard dependency, recorded in IMPLEMENTATION-LOG).
- `concurrency: group: cd-terraform-${{ github.ref }}, cancel-in-progress: false` (never cancel mid-apply).
- Phase 9 hook comment: "if promoted → deploy" wiring is Phase 9 (promote_champion_to_registry); here deploy is explicit/manual per no-auto-promotion.

### 4.7 Secrets (7.2 + 7.7)

- Existing: `GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`, `GCP_REGION`, `CODECOV_TOKEN`.
- New (Modal has no GitHub OIDC → scoped secrets): `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `WANDB_API_KEY`, `HF_TOKEN`. GCS needs nothing new (storage.admin via WIF).
- Documented in README secrets table + IMPLEMENTATION-LOG.

### 4.8 Docs (7.9)

- `docs/IMPLEMENTATION-LOG.md`: fill Phase 7 template — tasks 7.1-7.9, decisions 1-8, ADR-009 split note (Phase 7 = champion-regression + floor; candidate-promotion = Phase 9), hard dependency on 6.8 for automated deploy.
- `docs/planning/ADR-&-VISION.md`: ADR-013 (baseline is CD-owned, write-once-per-dataset, PRs read-only) + ADR-014 (absolute quality floor in CI gate, refined in Phase 9).

## 5. Task mapping (master-plan)

| Task | Delivered by |
| --- | --- |
| 7.1 GCP OIDC | ✅ existing (terraform-plan WIF; SA storage.admin) |
| 7.2 Modal OIDC | scoped secrets §4.7 (Modal has no GH OIDC) |
| 7.3 lint→typecheck→test→coverage | ✅ existing + `--cov-fail-under=75` §4.4 |
| 7.4 eval on PR + F2P gate | eval.yml §4.5 |
| 7.5 quality-gate logic | `_smoke_gate` floor + regression §4.1 |
| 7.6 CD: tf apply → modal deploy | cd.yml §4.6 (deploy manual until 6.8) |
| 7.7 secrets | §4.7 |
| 7.8 E2E on feature branch | §6 |
| 7.9 document | §4.8 |

## 6. E2E runbook (7.8 — on a feature branch)

1. **Branch protection** (one-time, manual, needs repo Admin): Settings→Branches→main: require status checks `lint-and-test`, `eval-gate`, `terraform-validate`. If no Admin → first blocker to raise. **Do NOT enable before step 4's measurement.**
2. Apply all changes; local verification: ruff check/format, mypy, targeted pytest (gate tests).
3. Add the **5 secrets** to the repo: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `WANDB_API_KEY`, `HF_TOKEN` (only consumed by `deploy-modal`; Modal-side artifacts use its own `hf-secret`/`wandb-secret`) — plus the existing GCP 5 (`GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`, `GCP_REGION`, `CODECOV_TOKEN`). Verify with `gh secret list`.
3a. **Bootstrap the GitHub Actions SA's control-plane grant** (one-time, outside Terraform — chicken-and-egg; the SA already holds runtime roles but NOT terraform-management roles: `iam.securityAdmin`, `iam.workloadIdentityPoolAdmin`, `secretmanager.admin`, `artifactregistry.admin`, `serviceusage.apiUsageAdmin`. Without this every plan/apply 403s outside the storage bucket):
   ```bash
   gcloud projects add-iam-policy-binding <PROJECT_ID> \
     --member="serviceAccount:github-actions-dev@<PROJECT_ID>.iam.gserviceaccount.com" \
     --role=roles/owner   # or grant the five granular roles; roles/owner is the pragmatic one-shot
   ```
   The grants are mirrored in `infra/terraform/modules/iam/main.tf` (count=enable_workload_identity) so Terraform keeps them in sync thereafter.
4. **Measure before flipping protection**: run the exact CI smoke slice (seed-42 × 20 @ 8192, now decision-8 tokens) for `higher_lr_14b` once and record its F2P. If ≥0.15 → floor is metering a real regime; enable protection. If <0.15 legitimately → set `EVAL_MIN_F2P_THRESHOLD` in the workflow env to `measured − ~0.05` (calibration knob, not a code change). This step also seeds context for the bootstrap comparison.
5. Push → observe `lint-and-test` (calibrate `--cov-fail-under` if needed) + `terraform-validate`.
6. **Create the `production` GitHub Environment** (Settings→Environments→New environment `production`; add required reviewers or a wait timer so apply can't run unattended). Then a first merge to main → the push-mode `eval-gate` job bootstraps `gs://swe-qwen-datasets/ci/smoke_baseline.json` (rates maxed with floor). Verify it exists. (`gcloud storage cp` via setup-gcloud honored; if `gcloud storage` misbehaves, fall back to `storage.Client()` — project dep — which reads ADC reliably.) Verify `terraform-under-plan` applies (after manual approval) idempotently; verify `deploy-modal` only fires on `workflow_dispatch`.
7. Open a PR with a crafted regression (e.g. mutated golden path override) → confirm `eval-gate` fails. Revert → green.
8. Open a docs-only PR → confirm eval-gate does not run (paths filter).
9. Confirm PR runs leave the GCS baseline byte-identical (read-only).

## 7. Acceptance (master-plan DoD, adapted per decisions)

- ✅ PR failing lint/unit blocked (existing; now also coverage).
- ✅ PR with champion-regressed smoke F2P (>5% drop) OR below the absolute floor is blocked by `eval-gate`.
- ✅ Champion re-validated when its config changes (paths include models.yaml/qlora_variants.yaml/config).
- ✅ Merge to main triggers terraform apply (idempotent), gated by manual approval on the `production` environment — plan is reviewed in the job summary first.
- ✅ Model deploy to Modal explicit/manual until 6.8 — recorded as a deviation with a hard 6.8 dependency, not overclaimed.
- ✅ All non-GCP secrets via GitHub secrets; GCP via WIF; no long-lived creds in repo.
- ✅ Full CI/CD exercised end-to-end on a feature branch (§6).

## 8. Risks / watch-items

- **Floor vs smoke-sample variance**: mitigated twice — (a) smoke now measures at 8192 tokens (decision 8), the same regime where the champion proved 16.9%; (b) E2E step 4 measures before enabling branch protection, with `EVAL_MIN_F2P_THRESHOLD` as the calibration knob. Bootstrap stores `max(measured, floor)` so the baseline never ratchets below floor.
- **Coverage threshold guess** (75): calibrate on first run.
- **Baseline last-write-wins**: serialized by `concurrency:` on eval.yml (same-ref cancel, main merges are sequential).
- **gcloud storage vs gcloud-tooling**: setup-gcloud@v2 after auth is docs-standard; fallback `storage.Client()` if E2E shows issues.
- **Public endpoint deploy** remains unauthenticated → `deploy-modal` manual precisely until 6.8 fixes auth + AsyncLLM.
