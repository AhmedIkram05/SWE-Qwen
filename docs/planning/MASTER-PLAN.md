# Master Implementation Plan — SWE-Qwen LLM Fine-Tuning Platform

**Document Type:** Technical Master Plan (Level 3 in project hierarchy)
**Status:** Draft v1.0
**Parent Document:** `docs/ADR-&-VISION.md` — all decisions, principles, and ADRs referenced herein are authoritative. This document translates them into an implementation roadmap.
**Hierarchy Position:**

```
Research Notes (Level 1)
  └── ADR & Vision (Level 2) ← this plan reads from it, does not repeat it
        └── Master Plan (Level 3) ← this document
              ├── Phase Plans (Level 4)
              │     └── Technical Specifications (Level 5)
              │           └── Implementation (Level 6)
```

**Design Principle:** Every phase produces a working, testable vertical slice (ADR-012). No phase leaves a workstream partially complete. Each phase can be independently validated before the next begins.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Objectives & Success Criteria](#2-project-objectives--success-criteria)
3. [Guiding Principles](#3-guiding-principles)
4. [Overall Roadmap](#4-overall-roadmap)
5. [Architecture-to-Workstream Mapping](#5-architecture-to-workstream-mapping)
6. [Repository Structure Evolution](#6-repository-structure-evolution)
7. [Work Breakdown Structure](#7-work-breakdown-structure)
8. [Detailed Phase Plans](#8-detailed-phase-plans)
9. [Milestones & Phase Exit Gates](#9-milestones--phase-exit-gates)
10. [Deliverables](#10-deliverables)
11. [Dependencies](#11-dependencies)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Testing Strategy](#13-testing-strategy)
14. [Documentation Strategy](#14-documentation-strategy)
15. [CI/CD Evolution](#15-cicd-evolution)
16. [Infrastructure Evolution](#16-infrastructure-evolution)
17. [Data Lifecycle](#17-data-lifecycle)
18. [Model Lifecycle](#18-model-lifecycle)
19. [Decision Traceability](#19-decision-traceability)
20. [Future Enhancements Roadmap](#20-future-enhancements-roadmap)
21. [Definition of Done](#21-definition-of-done)
22. [Appendices](#22-appendices)

---

## 1. Executive Summary

This document is the authoritative implementation blueprint for the SWE-Qwen LLM Fine-Tuning Platform. It translates the architectural decisions in the ADR & Vision document into a sequenced, dependency-aware work plan that can be decomposed into implementation-ready phase plans.

The platform demonstrates the complete lifecycle of developing, evaluating, deploying, and maintaining a fine-tuned open-weight LLM for automated software issue resolution. The emphasis is on **LLMOps engineering quality, platform architecture, high-throughput inference serving, and end-to-end reproducibility**.

**Key decisions baked into this plan:**

| Decision | Value | Source |
| ---------- | ------- | -------- |
| Foundation model | Qwen3-14B (primary), Qwen3-30B-A3B (future) | Conversation decision + P4 pivot |
| GPU platform | Modal (standardized, Baseten removed) | Conversation decision |
| Training platform | Modal | Conversation decision |
| Cloud provider | GCP only | Conversation decision |
| API compatibility | OpenAI-compatible | Conversation decision |
| Dataset size | 8–12k validated examples | Conversation decision |
| Repository count | ~10 Python repositories | Conversation decision |
| Monitoring (V1) | W&B + Langfuse | Conversation decision |
| Monitoring (v2) | OpenTelemetry + Prometheus/Grafana | ADR-011 intent |
| Execution feedback conditioning | Deferred to v2 | Conversation decision |
| Deployment pattern | Serverless vLLM on Modal | ADR-010 (updated) |
| Delivery model | Vertical slices (ADR-012) | ADR-012 |

---

## 2. Project Objectives & Success Criteria

### 2.1 Primary Objective

Build a production-grade LLMOps platform that demonstrates the complete lifecycle of fine-tuning, evaluating, and serving an open-weight LLM for automated software issue resolution — all implemented with modern engineering practices (IaC, CI/CD, observability, automated quality gates).

### 2.2 Success Criteria

The project is successful when **all** of the following are true:

| # | Criteria | Measurable Target |
| --- | ---------- | ------------------- |
| S1 | F2P resolution rate on golden eval set | ≥ 30% (baseline-dependent) |
| S2 | Regression safety (P2P rate) | ≥ 90% |
| S3 | Inference API serves requests end-to-end | OpenAI-compatible, <500ms TTFB p50 |
| S4 | Terraform provisions full infrastructure | Single `terraform apply` creates all resources |
| S5 | CI/CD pipeline runs on every PR | Lint → test → eval → gate → deploy |
| S6 | W&B tracks full experiment lineage | Data → config → run → model artifact → registry entry |
| S7 | Champion/Challenger promotion is automated | No manual deployment decisions |
| S8 | Platform is model-agnostic in architecture | Model selection can be swapped without rearchitecting |
| S9 | Scale-to-zero inference | $0 cost when no requests, cold start <10s |
| S10 | Documented and reproducible | A new engineer can reproduce any experiment from this doc |

### 2.3 Success Metrics by Workstream

| Workstream | Primary Metric | Gate |
| ------------ | --------------- | ------ |
| Data Pipeline | All records pass schema validation + dedup + F2P heuristics | Zero duplicates, no invalid records after cleaning |
| Fine-Tuning | QLoRA convergence without OOM | Training completes within Modal GPU budget (A10G 24GB for 14B QLoRA) |
| Evaluation | F2P rate measured on golden set | Golden set results logged to W&B per run |
| Inference | OpenAI-compatible API responds to chat completions | Integration test passes against running endpoint |
| Infrastructure | Terraform plan applies cleanly | Zero manual interventions required |
| CI/CD | All quality gates pass in GitHub Actions | PR can be merged only after gates pass |
| Observability | Telemetry emitted for training + serving | Dashboards show training loss, F2P, latency, cost |

---

## 3. Guiding Principles

These principles are inherited from the ADR & Vision (Level 2) and are not repeated here in full. The Master Plan operationalizes them.

| Principle (ADR) | How This Plan Operationalizes It |
| ----------------- | ---------------------------------- |
| **Platform over Model** (Vision) | Model selection is a Phase 1 output, not a Phase 0 input. Architecture is model-agnostic by design. |
| **Reproducibility** (ADR-001, ADR-006) | Every artifact (dataset, config, checkpoint) is versioned in W&B. Experiment IDs are immutable. |
| **Automation** (ADR-008, ADR-009) | Infrastructure provisioned via Terraform. Deployment triggered by CI/CD. No manual console steps. |
| **Execution-Based Evaluation** (ADR-005) | F2P is the primary quality gate. Text similarity is secondary monitoring only. |
| **Modularity & Adaptability** (ADR-002) | Each workstream is a decoupled module with clean interfaces. Model, compute, and storage layers are independently swappable. |
| **Cost-Conscious Engineering** (ADR-002, Vision) | Modal serverless GPU ($0 idle). Terraform free-tier resources. No 24/7 infrastructure. |
| **Observability** (ADR-011) | W&B for model metrics; structured JSON logging for serving; all phases emit exit criteria metrics. |
| **Vertical Slice Delivery** (ADR-012) | Each phase produces a independently functional increment. Phases are sequenced but each stands on its own. |

---

## 4. Overall Roadmap

The project is divided into **13 phases**, sequenced to respect dependencies and deliver vertical slices at each stage.

```
Phase 1  ──▶ Phase 2  ──▶ Phase 3  ──▶ Phase 4  ──▶ Phase 5  ──▶ Phase 6  ──▶ Phase 7
Foundation   Repo Curation  Data Pipeline  Fine-Tuning  Evaluation   Inference API  CI/CD
(Setup)      (Selection)    (Engine)       (Pipeline)   (Harness)    (Serverless)   (Quality Gates)

Phase 8  ──▶ Phase 9  ──▶ Phase 10 ──▶ Phase 11 ──▶ Phase 12 ──▶ Phase 13
Observability  Promotion  Documentation  Hardening  Validation  Launch
(Instrument.)  Pipeline   (Strategy)    (Resilience) (End-to-End) (Production)
```

**Phase grouping into vertical slices:**

| Slice | Phases | Deliverable |
| ------- | -------- | ------------- |
| **Slice 1 — Data & Training** | 1–4 | Working data pipeline + trained model checkpoint |
| **Slice 2 — Evaluation** | 5–7 | Evaluation harness + inference API serving |
| **Slice 3 — Platform** | 8–10 | Full observability + automated promotion |
| **Slice 4 —Production** | 11–13 | Hardened, validated, documented platform |

Each slice produces a working, testable system that can be validated independently.

---

## 5. Architecture-to-Workstream Mapping

The ADR architecture diagram decomposes into the following workstreams. Each workstream maps to one or more phases.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Architecture Layer                       │
├──────────────────┬──────────────────┬──────────────────────────┤
│  Data Layer      │  Compute Layer   │  Serving Layer          │
├──────────────────┼──────────────────┼──────────────────────────┤
│ Workstream WS-1  │ Workstream WS-2  │ Workstream WS-3          │
│ Data Pipeline    │ Fine-Tuning     │ Inference API            │
│ (Phases 2-3)     │ (Phases 3-4)    │ (Phases 6-7)            │
├──────────────────┼──────────────────┼──────────────────────────┤
│ Workstream WS-4  │ Workstream WS-5  │ Workstream WS-6          │
│ Evaluation       │ Promotion       │ Infrastructure           │
│ (Phase 5)        │ (Phase 8-9)     │ (Phases 1, 11)          │
├──────────────────┼──────────────────┼──────────────────────────┤
│ Workstream WS-7  │ Workstream WS-8  │                          │
│ Observability    │ CI/CD           │                          │
│ (Phase 10)       │ (Phase 7, 11)   │                          │
└──────────────────┴──────────────────┴──────────────────────────┘
```

### Workstream Definitions

| ID | Name | Description | Key Deliverables |
| ---- | ------ | ------------- | ----------------- |
| WS-1 | Data Pipeline | Ingestion, validation, cleaning, versioning, splitting, archiving of issue-patch data | Python package `data_engineering/`, W&B dataset artifacts |
| WS-2 | Fine-Tuning Pipeline | QLoRA training pipeline running on Modal | Python package `training/`, W&B run artifacts, LoRA adapters |
| WS-3 | Inference API | OpenAI-compatible, serverless vLLM endpoint on Modal | Python package `inference/`, Modal app, W&B serving metrics |
| WS-4 | Evaluation Harness | Execution-based F2P evaluation against real test suites | Python package `evaluation/`, golden set, eval reports |
| WS-5 | Champion/Challenger Promotion | Automated quality gate and model promotion pipeline | Promotion logic, W&B registry integration, deployment trigger |
| WS-6 | Infrastructure (IaC) | Terraform modules for GCS, IAM, secrets | Terraform configs, CI deployment workflows |
| WS-7 | Observability | Structured telemetry for training and serving | W&B dashboards, JSON logging, metrics emission |
| WS-8 | CI/CD Pipelines | GitHub Actions for lint, test, eval, gate, deploy | `.github/workflows/` configs, OIDC auth setup |

---

## 6. Repository Structure Evolution

### 6.1 Initial Repository (Empty)

```
swe-qwen/
├── README.md
├── .gitignore
├── docs/
│   ├── ADR-&-VISION.md
│   └── RESEARCH-NOTES.txt
└── docs/planning/
    └── MASTER-PLAN.md
```

### 6.2 Post-Phase-5 (Data + Training + Eval Complete)

```
swe-qwen/
├── README.md
├── .gitignore
├── pyproject.toml
├── docs/
│   ├── ADR-&-VISION.md
│   ├── RESEARCH-NOTES.txt
│   └── planning/
│       └── MASTER-PLAN.md
├── data_engineering/
│   ├── __init__.py
│   ├── ingest.py          # GitHub API → raw JSON
│   ├── validate.py        # Schema validation
│   ├── clean.py           # Dedup, clean, normalize
│   ├── split.py           # Train/val/test splits
│   ├── version.py         # W&B dataset versioning
│   └── schema.py          # Data schema definitions
├── training/
│   ├── __init__.py
│   ├── qlora_train.py     # QLoRA training entry point
│   ├── model_config.py    # Model selection & config
│   ├── modal_train.py     # Modal training job wrapper
│   └── callbacks.py       # W&B logging callbacks
├── evaluation/
│   ├── __init__.py
│   ├── config.py
│   ├── schema.py
│   ├── patch_applier.py
│   ├── test_runner.py
│   ├── metrics.py
│   ├── harness.py                  ← F2P engine (includes runners, resume, W&B logging)
│   ├── prompt_ab_test.py
│   ├── inference.py                ← batch inference for patch gen
│   ├── comparison.py               ← re-validate P4 proxy champion
│   └── cli.py                      ← Typer CLI entrypoint
├── models/
│   └── checkpoints/       # LoRA adapter storage (gitignored)
├── .github/workflows/
│   └── ci.yml             # Basic CI (lint + test)
└── tests/
    ├── test_data.py
    ├── test_training.py
    └── test_evaluation.py
```

### 6.3 Post-Phase-10 (Full Platform)

```
swe-qwen/
├── README.md
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml     # Local development only
├── docs/
│   ├── ADR-&-VISION.md
│   ├── RESEARCH-NOTES.txt
│   └── planning/
│       └── MASTER-PLAN.md
├── infra/
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── providers.tf
│       └── modules/
│           ├── storage/
│           ├── iam/
│           └── deployment/
├── data_engineering/      # (as Phase 5)
├── training/              # (as Phase 5)
├── inference/
│   ├── __init__.py
│   ├── serve.py           # vLLM serving entry point
│   ├── modal_serve.py     # Modal deployment wrapper
│   ├── openai_compat.py   # OpenAI API adapter
│   └── schema.py          # Request/response schemas
├── evaluation/            # (as Phase 5)
├── promotion/
│   ├── __init__.py
│   ├── gate.py            # Quality gate logic
│   ├── champion.py        # Champion/Challenger logic
│   └── deploy.py          # Deployment trigger
├── observability/
│   ├── __init__.py
│   ├── logging.py         # Structured JSON logging
│   ├── metrics.py         # Prometheus-style metrics emitters
│   └── dashboards.py      # W&B dashboard config
├── models/
│   └── checkpoints/
├── .github/workflows/
│   ├── ci.yml             # Full CI with eval gates
│   └── cd.yml             # CD with promotion trigger
└── tests/
    ├── test_data.py
    ├── test_training.py
    ├── test_evaluation.py
    ├── test_inference.py
    ├── test_promotion.py
    └── test_infra.py
```

---

## 7. Work Breakdown Structure

### WBS Level 1: Phases

| WBS | Phase | Workstream(s) | Estimated Focus |
| ----- | ------- | --------------- | ----------------- |
| 1.0 | Phase 1: Foundation | WS-6 (infra setup) | Repo init, Terraform scaffolding, Modal setup, W&B project creation |
| 2.0 | Phase 2: Repository Curation | WS-1 (data) | Select & document 10 Python repos |
| 3.0 | Phase 3: Data Pipeline | WS-1 (data) | Build ingestion → validation → cleaning → versioning |
| 4.0 | Phase 4: Fine-Tuning Pipeline | WS-2 (training) | QLoRA pipeline on Modal, experiment tracking |
| 5.0 | Phase 5: Evaluation Harness | WS-4 (evaluation) | F2P harness, golden set, execution-based metrics |
| 6.0 | Phase 6: Inference API | WS-3 (serving) | OpenAI-compatible vLLM serverless API on Modal |
| 7.0 | Phase 7: CI/CD Integration | WS-8 (CI/CD) | GitHub Actions with quality gates + OIDC |
| 8.0 | Phase 8: Observability | WS-7 (observability) | Full telemetry, dashboards, structured logging |
| 9.0 | Phase 9: Promotion Pipeline | WS-5 (promotion) | Automated Champion/Challenger pipeline |
| 10.0 | Phase 10: Documentation | All | Technical docs, architecture notes, deployment guides |
| 11.0 | Phase 11: Hardening | All | Resilience, error handling, edge cases |
| 12.0 | Phase 12: End-to-End Validation | All | Full pipeline validation, acceptance testing |
| 13.0 | Phase 13: Production Launch | All | Final deployment, handoff, portfolio-ready presentation |

### WBS Level 2: Key Tasks (Sample from Phase 3)

```
3.0 Data Pipeline
├── 3.1 Design data schema (issue_id, repo, issue_body, patch diff, test_results, metadata)
├── 3.2 Build GitHub API ingestion module
│   ├── 3.2.1 Authenticate via GitHub token (env var, no hardcoded secrets)
│   ├── 3.2.2 Fetch issues with labels matching bug/fix
│   ├── 3.2.3 Resolve linked PRs for each issue
│   └── 3.2.4 Extract diff, test file changes, commit messages
├── 3.3 Build schema validation module
│   ├── 3.3.1 Define Pydantic models for each record type
│   ├── 3.3.2 Validate required fields (issue body, patch, test suite)
│   └── 3.3.3 Reject records that fail validation; log reasons
├── 3.4 Build cleaning & deduplication module
│   ├── 3.4.1 Remove duplicate issue-PR pairs (same repo, same fix)
│   ├── 3.4.2 Filter out PRs without test changes where possible
│   └── 3.4.3 Normalize patch format (unified diff)
├── 3.5 Build train/val/test split module
│   ├── 3.5.1 Stratified split by repository (prevent data leakage)
│   ├── 3.5.2 Golden eval subset: only examples with test-verified fixes
│   └── 3.5.3 Version the splits via W&B
├── 3.6 Build dataset archiving module
│   ├── 3.6.1 Upload final dataset to GCS via Terraform-managed bucket
│   ├── 3.6.2 Log dataset artifact to W&B with full lineage
│   └── 3.6.3 Generate dataset card (size, schema, source repos, quality stats)
└── 3.7 Write tests for all data pipeline modules
```

*Full WBS for all 13 phases follows the same decomposition pattern.*

---

## 8. Detailed Phase Plans

---

### Phase 1: Foundation & Infrastructure Setup

**Objective:** Establish the project scaffolding, repository structure, Terraform GCP foundation, Modal account configuration, and W&B project. This phase produces a working, empty infrastructure that subsequent phases populate.

**Why it exists:** Every other phase depends on having infrastructure, IAM, storage, and compute platform configured. This is the first vertical slice — a working, empty platform that proves the deployment path works.

**Inputs:**

- GCP project with programmatic access
- GitHub repository (this repo)
- Modal account with API key
- W&B account with API key
- GitHub OIDC configuration (prepared)

**Outputs:**

- Initialized Git repository with branch strategy
- Terraform scaffold creating GCS bucket, IAM roles, secrets
- Modal project configuration (`modal config`, `modal serve` skeleton)
- W&B project created and documented
- GitHub Actions scaffold (empty workflows)
- Project README with architecture overview

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 1.1 | Initialize Git repository with branches (`main`, `develop`, phase branches) | `.git/` configured, branch protection rules documented |
| 1.2 | Create `pyproject.toml` with project metadata, dependencies, tooling config | `pyproject.toml` with Ruff, pytest, mypy config |
| 1.3 | Write `.gitignore` excluding model checkpoints, W&B artifacts, `.env` files | `.gitignore` |
| 1.4 | Scaffold Terraform: `infra/terraform/` directory with `main.tf`, `variables.tf`, `outputs.tf`, `providers.tf` | Terraform scaffold |
| 1.5 | Implement Terraform GCS module: bucket for dataset artifacts and model checkpoints | `infra/terraform/modules/storage/` |
| 1.6 | Implement Terraform IAM module: execution roles, OIDC provider for GitHub Actions, secret management | `infra/terraform/modules/iam/` |
| 1.7 | Write `terraform plan` and validate the infrastructure graph | `terraform plan` output, no errors |
| 1.8 | Configure Modal: API key setup, `modal init`, create skeleton `modal serve` app | Modal project initialized |
| 1.9 | Create W&B project: initialize run logging, define project structure for experiment tracking | W&B project created, `wandb` initialized |
| 1.10 | Write first GitHub Actions workflow skeleton (placeholder, no real steps yet) | `.github/workflows/ci.yml` skeleton |
| 1.11 | Write project README with architecture overview, quick-start, and structure documentation | `README.md` |
| 1.12 | Write tests for Terraform outputs (validate bucket exists, IAM roles are correct) | `tests/test_infra.py` |
| 1.13 | Write tests for project scaffold (verify all expected directories and files exist) | `tests/test_scaffold.py` |

**Dependencies:** None (first phase).

**Risks:**

- Modal API key configuration issues → Mitigation: Document key setup steps in README, test with Modal ping before proceeding
- Terraform GCP provider version conflicts → Mitigation: Pin provider versions in `required_providers` block

**Definition of Done:**

- [ ] `terraform plan` succeeds without errors
- [ ] Modal serve skeleton runs without error
- [ ] W&B project exists and accepts test run
- [ ] All scaffold files present in correct locations
- [ ] `pytest` passes for all scaffold tests

**Acceptance Criteria:**

1. A new contributor can clone the repo and run `terraform plan` to see the full infrastructure plan
2. Modal serve skeleton starts without error
3. W&B accepts a test logged metric
4. GitHub Actions workflow skeleton exists and passes linting

**Expected repository state:** Scaffolded repo with Terraform, Modal, W&B, and CI configuration — no data, no models, no training yet.

---

### Phase 2: Repository Curation

**Objective:** Select, document, and prepare 10 Python repositories that will serve as the data source for the training dataset. Each repository is chosen for test quality, issue clarity, permissive license, and issue-to-PR linkage.

**Why it exists:** The data pipeline (Phase 3) depends on having well-defined source repositories. This phase ensures the source data is high-quality before any engineering effort is spent on ingestion.

**Inputs:**

- List of selection criteria (from ADR-004, Vision doc)
- GitHub API access with sufficient rate limits
- Manual curation effort (~1-2 days)

**Outputs:**

- `repos/` directory with curated repository manifest (`repos/manifest.json`)
- Per-repo documentation file with selection rationale
- Validation script that checks repo criteria programmatically

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 2.1 | Define selection criteria (Python, active, tests, permissive license, clear issue-PR linkage) | Criteria document |
| 2.2 | Identify candidate repositories using GitHub search and manual review | Candidate list (20-30 repos) |
| 2.3 | Validate each candidate against criteria programmatically | Validation script |
| 2.4 | Select final 10 repositories | Final list of 10 repos |
| 2.5 | Create `repos/manifest.json` with repo metadata (name, url, license, test command, issues) | Manifest file |
| 2.6 | Document selection rationale per repository | `repos/README.md` |
| 2.7 | Write verification script: clone each repo, check test suite installs and runs | `verify_repos.py` |
| 2.8 | Run verification script against all 10 repos | Verification log |

**Dependencies:** Phase 1 complete (Terraform not required yet, but repo structure should exist).

**Risks:**

- Selected repos may have test suites that don't run cleanly → Mitigation: Verification script in Phase 2 catches this before Phase 3 begins
- License incompatibility → Mitigation: Only MIT/Apache 2.0/licenses with commercial use permitted

**Definition of Done:**

- [ ] 10 repositories selected and documented
- [ ] `repos/manifest.json` contains all metadata
- [ ] Verification script passes for all 10 repos
- [ ] Each repo's test command is documented

**Acceptance Criteria:**

1. All 10 repositories have installable test suites that pass on a clean environment
2. Each repository has clear issue-to-PR linkage (issues reference PRs that fix them)
3. All licenses are permissive (MIT, Apache 2.0, BSD)
4. The manifest file is valid JSON and contains all required fields

**Expected repository state:** `repos/manifest.json` + per-repo docs + verification script. No data pipeline yet.

---

### Phase 3: Data Pipeline Engine

**Objective:** Build a production-grade data engineering pipeline that ingests issue-PR pairs from GitHub, validates them against a schema, cleans and deduplicates them, splits into train/val/test, and versions the dataset in W&B and GCS.

**Why it exists:** Data quality directly determines model quality (ADR-004). This phase produces the first working vertical slice: a pipeline that takes GitHub repos and produces a validated, versioned dataset ready for training.

**Inputs:**

- Repository manifest from Phase 2
- GitHub API tokens (secrets in GCP via Terraform)
- W&B API key
- GCS bucket (from Phase 1 Terraform)

**Outputs:**

- `data_engineering/` Python package (complete)
- W&B dataset artifacts with full lineage
- GCS dataset artifacts (versioned)
- Golden evaluation subset (8-12k examples with verified test-suite fixes)
- Train/validation/test split artifacts

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 3.1 | Design data schema (Pydantic models for issue records, PR records, patches, test results) | `data_engineering/schema.py` |
| 3.2 | Build GitHub API ingestion module | `data_engineering/ingest.py` |
| 3.3 | Implement schema validation with detailed error logging | `data_engineering/validate.py` |
| 3.4 | Implement cleaning & deduplication | `data_engineering/clean.py` |
| 3.5 | Implement train/val/test stratified split (by repo) | `data_engineering/split.py` |
| 3.6 | Implement golden eval subset extraction (test-verified fixes only) | `data_engineering/golden.py` |
| 3.7 | Implement W&B dataset versioning integration | `data_engineering/version.py` |
| 3.8 | Implement GCS upload with Terraform-managed bucket path | `data_engineering/archive.py` |
| 3.9 | Build dataset card generation (auto-generated summary) | `data_engineering/card.py` |
| 3.10 | Write unit tests for all pipeline modules | `tests/test_data.py` |
| 3.11 | Write integration test: full pipeline from manifest → validated dataset | Integration test |
| 3.12 | Run pipeline on 1-2 repos as proof of concept | Sample dataset artifact in W&B |
| 3.13 | Run pipeline on all 10 repos for full dataset | Complete dataset artifact |

**Dependencies:** Phase 2 complete (repo manifest required). Phase 1 complete (GCS bucket, W&B project required for archive and versioning).

**Risks:**

- GitHub API rate limits on large-scale ingestion → Mitigation: Implement exponential backoff, use GitHub App token for higher limits, batch requests
- Data quality issues in golden subset (fewer test-verified fixes than expected) → Mitigation: Expand candidate pool to 20 repos initially; prune to 10 after verifying golden set size; pad with synthetic examples only if absolutely needed
- OOM during dataset processing → Mitigation: Process in batches, stream from GitHub API, use disk-backed intermediate storage

**Definition of Done:**

- [ ] All pipeline modules implemented with unit test coverage ≥ 80%
- [ ] Full pipeline runs end-to-end on all 10 repos
- [ ] Golden eval subset contains 200+ verified examples (target), 800+ total validated examples
- [ ] Dataset artifacts versioned in W&B with full lineage
- [ ] Dataset archived to GCS
- [ ] Dataset card auto-generated

**Acceptance Criteria:**

1. `python -m data_engineering.run_pipeline --manifest repos/manifest.json` produces a complete dataset artifact in W&B and GCS
2. Golden eval subset: all records pass schema validation + F2P heuristic verification + dedup checks (zero duplicates)
3. W&B shows dataset lineage: manifest → raw → validated → cleaned → split → versioned
4. Dataset card includes: size, schema, source repos, quality stats, split ratios

**Expected repository state:** Working `data_engineering/` package, complete dataset in W&B + GCS, golden eval subset ready for Phase 5.

---

### Phase 4: Fine-Tuning Pipeline

**Objective:** Build a modular QLoRA fine-tuning pipeline that runs training jobs on Modal, integrates with W&B for full experiment tracking, and produces versioned LoRA adapter checkpoints.

**Why it exists:** This is the core ML workstream. It produces the trained model artifacts that all subsequent evaluation, serving, and promotion phases depend on.

**Inputs:**

- Validated dataset from Phase 3 (golden subset + training split)
- QLoRA configuration (rank, alpha, learning rate, batch size, epochs — initially set to recommended defaults)
- Modal compute configuration (**A10G 24GB for Qwen3-14B QLoRA** — Phase 4 pivot: 14B-only, 30B excluded due to cost)

**Outputs:**

- `training/` Python package (complete)
- W&B experiment runs with full lineage (data → config → run → model)
- LoRA adapter checkpoints (versioned in W&B)
- Training curves and metrics (loss, LR, gradient norm)

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 4.1 | Implement model selection logic (Qwen3-14B primary; qwen3-30b-a3b retained in config but **phase4_excluded**) | `training/model_config.py` |
| 4.2 | Implement QLoRA configuration (peft + bitsandbytes + transformers integration) | `training/qlora_config.py` |
| 4.3 | **Prompt engineering workstream**: design and version prompt templates for training/inference (Issue+Context → Patch); W&B artifact versioning for prompt templates; A/B test 2-3 variants in Phase 5 eval | `training/prompts/` + W&B prompt artifacts |
| 4.4 | Build training entry point with W&B integration | `training/qlora_train.py` |
| 4.5 | Implement Modal training job wrapper (modal.function decorator, GPU config, volume mounts) | `training/modal_train.py` |
| 4.6 | Add training callbacks for W&B logging (loss, metrics, checkpoints, hyperparameters) | `training/callbacks.py` |
| 4.7 | Implement checkpoint saving and versioning to W&B model registry | `training/checkpoint.py` |
| 4.8 | Add experiment resumption support (checkpoint → resume from step) | Resume capability |
| 4.9 | Write unit tests for all training modules | `tests/test_training.py` |
| 4.10 | Run baseline training on small subset (100 examples) to validate pipeline | Test run artifact |
| 4.11 | Run full training on complete dataset | Final LoRA adapter checkpoint |
| 4.12 | **Mandatory 3-config QLoRA comparison**: run three 14B-optimized configs on Modal (baseline_14b r=16/alpha=32/lr=2e-5; higher_rank_14b r=32/alpha=64/lr=2e-5; higher_lr_14b r=16/alpha=32/lr=5e-5); all three evaluated on golden F2P set in Phase 5; winner → Champion. Optuna deferred to v2 (budget >50 GPU-hrs, automated eval). **Cost: ~$10-17 total (vs ~$110-160 for 30B)** | `scripts/run_3config_comparison.py`, 3 W&B runs, comparison report |

**Dependencies:** Phase 3 complete (validated dataset required). Phase 1 complete (Modal, W&B configured).

**Risks:**

- ~~Qwen3-30B-A3B (30B total params) requires A100 40GB for QLoRA training~~ — **Phase 4 pivot eliminates this risk**
- Qwen3-14B on A10G 24GB: OOM risk low → Mitigation: Gradient checkpointing, batch_size=2, grad_accum=8, max_seq_length=8192, fallback to A100-40GB
- Training instability (loss divergence, NaNs) → Mitigation: Start with conservative hyperparameters (lr=2e-5, rank=16, warmup=10%), log all metrics to W&B for rapid diagnosis
- Modal job timeout or interruption → Mitigation: Implement checkpoint resume (4.8), use Modal's checkpointing volumes

**Definition of Done:**

- [ ] Training completes without OOM on Qwen3-14B within Modal budget (A10G 24GB)
- [ ] W&B shows complete lineage: dataset artifact → training config → run → checkpoint → model registry entry
- [ ] LoRA adapter checkpoint is loadable and produces valid outputs
- [ ] Training curve (loss) is logged to W&B and shows convergence
- [ ] All unit tests pass
- [ ] **3-config 14B-optimized comparison completed and winner selected via F2P on golden eval set**

**Acceptance Criteria:**

1. `python -m training.qlora_train --config training_config.yaml --data-dir data/` runs to completion
2. W&B run shows: loss curve, hyperparameters, dataset version, GPU utilization, estimated cost
3. Checkpoint can be loaded with `AutoModel.from_pretrained()` + PEFT adapters
4. Baseline evaluation completed: Qwen3-14B QLoRA fits on A10G 24GB (primary), A100-40GB (fallback)
5. **3-config 14B-optimized comparison: three W&B runs exist, F2P on golden set determines Champion, results logged**

**Expected repository state:** Trained LoRA adapter checkpoint in W&B + GCS, full training pipeline operational, baseline model evaluation report in W&B.

> **PIVOT NOTE:** Phase 4 now uses ONLY qwen3-14b on A10G 24GB. qwen3-30b-a3b is excluded from Phase 4 execution due to cost (~$110-160 for 3-config on H100). The 30B config is retained in `config/models.yaml` for future phases but marked `phase4_excluded: true`. All 3 mandatory comparison variants are redesigned for 14B (baseline_14b, higher_rank_14b, higher_lr_14b). Cost reduction: ~90% ($10-17 total vs $110-160). Full details in `docs/planning/PHASE-4-QLoRA-TRAINING-PIPELINE.md`.

---

### Phase 5: Evaluation Harness

**Objective:** Build an execution-based evaluation harness that measures F2P (Fail-to-Pass) resolution rate on real test suites, computes regression safety (P2P), and produces evaluation reports in W&B.

**Why it exists:** The evaluation harness is the quality gate for everything. Without it, there is no way to objectively assess whether fine-tuning improves the model, and no way to drive the Champion/Challenger promotion in Phase 9.

**Inputs:**

- Trained LoRA adapter from Phase 4
- Golden eval subset from Phase 3 (test-verified examples)
- Test suites from 10 source repositories

**Outputs:**

- `evaluation/` Python package (complete)
- F2P and P2P metrics per model/run
- Evaluation reports in W&B
- Golden eval set benchmark results (baseline vs. fine-tuned)

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 5.1 | Design evaluation schema (input: issue + context, expected: test results before/after) | `evaluation/schema.py` |
| 5.2 | Build test suite execution engine (run pytest against repo with patch applied) | `evaluation/test_runner.py` |
| 5.3 | Implement F2P computation (failing tests before fix that pass after) | `evaluation/metrics.py` |
| 5.4 | Implement P2P computation (passing tests that stay passing — regression safety) | Regression safety calc |
| 5.5 | Add golden eval subset runner (apply model-generated patch to each golden example, run tests) | `evaluation/harness.py` (consolidated) |
| 5.6 | **Add SWE-bench Verified subset evaluation runner** (download SWE-bench Verified, run model on subset, log results as secondary benchmark alongside golden set) | `evaluation/harness.py` (consolidated) |
| 5.7 | Implement W&B evaluation artifact logging (metrics, examples, diffs) | `evaluation/harness.py` (inline) |
| 5.8 | Build comparison framework (baseline model vs. fine-tuned model on same golden set) | `evaluation/comparison.py` |
| 5.9 | Run evaluation on baseline (unfine-tuned) model | Baseline metrics |
| 5.10 | Run evaluation on fine-tuned model | Fine-tuned metrics |
| 5.11 | Write unit + integration tests for evaluation modules | `tests/test_eval_unit.py`, `tests/test_eval_integration.py` |
| 5.12 | **Re-validate P4 proxy champion** with real F2P (not fresh selection) | Champion re-validation |

> **Note:** Golden, SWE-bench, and baseline runners are consolidated into `evaluation/harness.py` for brevity. W&B logging and resume logic are inline in `harness.py`. See `PHASE-5-EVALUATION-HARNESS.md` for module details.

**Dependencies:** Phase 4 complete (model checkpoint required). Phase 3 complete (golden eval subset required).

**Risks:**

- Test suite execution may fail for reasons unrelated to the patch (environment issues, flaky tests) → Mitigation: Isolate test runs in clean Docker containers per repo, retry flaky tests up to 2 times, mark tests as flaky rather than failing
- Golden set is too small for statistical significance → Mitigation: Phase 3 targets 800+ golden examples; if fewer are verified, expand the candidate pool in Phase 2
- Patch application may fail (conflicts, malformed diffs) → Mitigation: Validate patch format before application, log failure reasons, exclude from F2P calculation (not counted as false negative)

**Definition of Done:**

- [ ] F2P rate computed for baseline and fine-tuned models on golden eval set
- [ ] P2P regression rate computed for fine-tuned model
- [ ] All metrics logged to W&B per evaluation run
- [ ] Evaluation can be re-run reproducibly from the same model checkpoint and golden set
- [ ] All unit + integration tests pass

**Acceptance Criteria:**

1. `python -m evaluation.run --model checkpoint/ --golden data/golden.json` produces F2P and P2P metrics
2. F2P improvement over baseline is measurable (even if modest for V1)
3. W&B shows evaluation artifacts: metrics table, per-example pass/fail status, diffs, test output logs
4. The evaluation harness runs without human intervention (fully automated)

**Expected repository state:** Working evaluation harness, benchmark results for baseline and fine-tuned models in W&B, golden eval set validated.

---

### Phase 6: Inference API (Serverless vLLM on Modal)

**Objective:** Deploy a production-grade, OpenAI-compatible inference API using vLLM served serverlessly on Modal with scale-to-zero capabilities.

**Why it exists:** This phase delivers the production serving layer. The platform is not complete without a working, high-throughput inference endpoint that demonstrates modern LLMOps serving patterns.

**Inputs:**

- Trained LoRA adapter from Phase 4
- vLLM serving configuration (deferred to Phase 6 benchmarking)
- Modal account and compute allocation

**Outputs:**

- `inference/` Python package (complete)
- Modal app for serverless vLLM serving
- OpenAI-compatible API endpoint (chat completions, completions, embeddings)
- W&B serving metrics (latency, throughput, cost per inference)

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 6.1 | Benchmark vLLM configuration options on Modal (tensor parallelism, GPU memory utilization, max num seqs) | Benchmark report, select optimal config |
| 6.2 | Implement vLLM serving entry point with LoRA adapter loading | `inference/serve.py` |
| 6.3 | Wrap serving app as Modal function/server (scale-to-zero configuration) | `inference/modal_serve.py` |
| 6.4 | Implement OpenAI-compatible API adapter (chat/completions endpoint, request/response schemas) | `inference/openai_compat.py` |
| 6.4.1 | **Add OpenAI streaming support** (`stream: true` in chat/completions, Server-Sent Events response) — table stakes for production-grade API | Updated `inference/openai_compat.py` |
| 6.5 | Add telemetry: TTFB, tokens/sec, request count, error rate, GPU utilization | `inference/telemetry.py` |
| 6.6 | Implement request validation and error handling | Input validation, error responses |
| 6.7 | Write integration test: send chat completion request to running endpoint, verify OpenAI-compatible response format | Integration test |
| 6.8 | Benchmark latency and throughput; log to W&B | Benchmark artifact |
| 6.9 | Configure scale-to-zero: idle timeout, cold start measurement | Scale-to-zero config |

> **Phase 6 research notes — logic/method crossover from Phases 4-5 + Modal credit conservation (may be wrong).** Derived from repo inspection + manual-run experience, not a live serve run. The point of crossover is *general logic and methods*, not file reuse: Phase 6 should re-implement `inference.py`'s structure around a persistent engine, not copy its batch loop.
>
> **What crosses over as logic/method patterns:**
>
> 1. **vLLM usage pattern** from `evaluation/inference.py`: single `LLM` instance → batched `generate` with `tokenize=False` → per-model tokenizer caching. Phase 6 keeps the same shape but the engine is persistent and batching is request-driven instead of dataset-driven.
> 2. **Prompt-builder methods**: the same chat-template + golden few-shot + per-tier token-budget logic (methods, not just output). Keep method signatures stable and share the implementation so served inference and eval F2P can never drift (drift = silently re-debugging and re-evaluating).
> 3. **Modal app skeleton** from `training/modal_train.py`: app/function-decorator layout, `Secret.from_name()`, volume `create_if_missing`, `add_local_dir` last, explicit teardown (the `wandb.finish()` lesson), cache-buster strings for image invalidation, concurrency caps (the aiohttp 1.5.3 bug at 64-way). Only genuinely-new pattern: persistent `@app.cls` engine with async one-time init + warm containers.
> 4. **Config-layer pattern** from `EvalConfig`: pydantic-settings BaseSettings, env prefix, frozen — copy the *pattern* for a new `ServeConfig` (do not import `EvalConfig` itself).
> 5. **Adapter resolution method**: how `model-qwen3-14b-{variant}` is located and loaded is proven; only the served-mode registration differs (`--lora-modules` + per-request `"model": "<lora_name>"`).
>
> **Credit-conservation strategy — limit Modal debugging spend (A100 budget is the constraint):**
>
> - **Local-first before any Modal run.** Unit-test prompt building, SSE chunk assembly, and the OpenAI response schema against fixtures locally; mock vLLM with an LPU-less stub. Everything that can run without GPUs must never touch Modal in a debug loop.
> - **One boot should validate many things.** Batch debug attempts: a single preflight/integration call that exercises adapter load + one chat + one stream together — not five sequential single-purpose boots. Iterate locally until confident, then boot.
> - **Build the serving image once; do not rebuild it in the loop.** Modal caches images — only bump the cache-buster when dependencies change. Weight-load-only iterations must reuse the cached image, or every code tweak burns a full image rebuild + model load.
> - **Fail fast at boot.** Let vLLM config errors (VRAM, `max-model-len`) surface during container warmup (30-60s of A100) rather than at first request. Adds warmup checks; removes mid-request crash-debug loops.
> - **Prefer `modal serve` (hot reload) for the dev loop; `deploy` only when the endpoint is stable.** Minimizes one-shot cold boots.
> - **Pin the Modal version** until Phase 6 is done — the 1.5.3 aiohttp regression shows upgrades carry debugging cost; treat a Modal bump as a separate task.
> - **Structured logging to container stdout from day one** (`modal app logs`) — avoids blind 504/500 debugging, which is the most expensive loop of all (each guess = a boot).
> - **Reuse smoke-tier sampling** (existing tier pattern, e.g. smoke:20) for validation instead of full benchmarks during debugging; full benchmarks only as acceptance.
>
> **Serving quantization recommendation (Path A — quantized base + live LoRA, may be wrong).** Training quantization (QLoRA 4-bit NF4) is an in-memory training state, NOT a servable format — the saved artifact is bf16 LoRA adapters, so training artifacts cannot be reused for serving. Serving on `a10g-24gb` (per the GPU-sizing advice above) therefore requires a servable-quantized base. Recommended path:
>
> 1. Use a **pre-quantized base** from HF (e.g. `Qwen/Qwen3-14B-FP8` or an AWQ/GPTQ build) — zero quantization work, ~14GB (FP8) or ~7-9GB (4-bit).
> 2. Serve in vLLM on Modal with `LLM(model=<quantized_path>, quantization='fp8'|'awq', enable_lora=True, ...)` and attach the trained LoRA adapter at request time (`lora_request`, every request carries `"model": "<variant>"`). No merge, no 28GB bf16 checkpoint, no re-quantization on new training — swap adapter, keep base.
> 3. Fallback (Path B, only if served-mode LoRA registration bugs out): merge base+adapter to bf16 (one GPU pass, a few $), quantize with `llm-compressor` (FP8) or `autoawq` (AWQ, `group_size=128`), serve merged — simpler runtime but bakes the adapter in (new training = re-merge). Budget ~$5-15 for this retry if needed.

**Dependencies:** Phase 4 complete (model checkpoint required). Phase 1 complete (Modal configured).

**Risks:**

- vLLM configuration not optimal for 24GB VRAM → Mitigation: Benchmark multiple configs in Phase 6.1 before finalizing; start with conservative GPU memory utilization (0.85)
- Cold start latency exceeds 10 seconds → Mitigation: Modal's serverless platform handles this; measure and document as a known tradeoff; accept for V1
- OpenAI compatibility gaps → Mitigation: Implement core endpoints (chat/completions) with streaming support

**Definition of Done:**

- [ ] Running inference endpoint accepts OpenAI-compatible chat completion requests
- [ ] Returns valid JSON responses matching OpenAI API schema
- [ ] **Streaming support: `stream: true` returns Server-Sent Events per OpenAI spec**
- [ ] TTFB p50 < 500ms under benchmark load
- [ ] Scale-to-zero: $0 cost when idle, cold start < 15s
- [ ] W&B shows serving metrics (latency, throughput, cost)
- [ ] Integration test verifies end-to-end request/response

**Acceptance Criteria:**

1. `modal serve inference.modal_serve` starts a serverless endpoint
2. Client can call the endpoint using any OpenAI Python SDK with OpenAI-compatible configuration
3. The endpoint returns completions for code-fix prompts with measurable latency
4. **Streaming responses work: `stream: true` yields token-by-token SSE chunks**
5. Cold start time is measured and documented in W&B
6. Scale-to-zero is verified (endpoint scales down to zero after idle timeout)

**Expected repository state:** Running serverless inference API on Modal, OpenAI-compatible, scale-to-zero, full telemetry in W&B.

---

### Phase 7: CI/CD Integration with Quality Gates

**Objective:** Build GitHub Actions pipelines that validate code quality, run tests, execute evaluation benchmarks, enforce automated quality gates, trigger infrastructure deployment via Terraform using OIDC keyless authentication, and gate model promotion based on evaluation results.

**Why it exists:** CI/CD quality gates enforce the principle that code must meet standards and models must pass evaluation before deployment (ADR-009). OIDC keyless authentication eliminates long-lived cloud credentials from the repository (ADR-008).

**Inputs:**

- All previous phases complete (working code, trained models, evaluation harness, infrastructure)
- GitHub repository with OIDC configured for GCP
- W&B API key (stored as GitHub secret)
- Modal API key (stored as GitHub secret)
- GCP credentials via OIDC (no long-lived keys)

**Outputs:**

- Complete GitHub Actions workflow files
- OIDC authentication for GCP (Terraform apply from CI)
- Quality gate logic (eval F2P threshold check)
- Automated deployment trigger (promoted model → Terraform apply)

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 7.1 | Configure GitHub OIDC identity provider for GCP | GCP Workload Identity Pool/Provider, Terraform `google_iam_workload_identity_pool` |
| 7.2 | Configure GitHub OIDC identity provider for Modal | Modal team/org API token via OIDC or scoped secret |
| 7.3 | Implement CI workflow: lint (Ruff) → typecheck (mypy) → unit test (pytest) → coverage check | `.github/workflows/ci.yml` |
| 7.4 | Implement eval workflow: on PR → run evaluation harness → check F2P threshold → pass/fail gate | `.github/workflows/eval.yml` |
| 7.5 | Implement quality gate logic: F2P must exceed baseline by minimum delta to pass | Quality gate config |
| 7.6 | Implement CD workflow: on merge to main → Terraform apply (infra) → if model promoted then deploy to Modal | `.github/workflows/cd.yml` |
| 7.7 | Store secrets (W&B, Modal) as GitHub repository secrets, no hardcoded values | GitHub secrets configured |
| 7.8 | Test full CI/CD pipeline end-to-end on a feature branch | Pipeline validation |
| 7.9 | Document CI/CD architecture and workflow descriptions | CI/CD documentation |

> **Phase 7 research notes — automation advice (may be wrong).** Derived from inspecting the current repo + manual-run experience, not from a live CI run. Treat as hypotheses to validate during pre phase implementation (they change task details, not the phase's scope):
>
> 1. **The manual dev CLI is NOT the automation interface.** Tokenize prep is only reachable via the data CLI (`python -m data_engineering.cli run --run-id <id> --tokenize-model ...`) but it is CPU-only (downloads the Qwen3-14B *tokenizer*, not weights) and runs on a normal GitHub runner — $0 Modal GPU. Its only purpose is to materialize `tokenized/<run_id>/` in GCS (`tokenize._save_dataset_to_gcs`), which `modal_train.train_qlora` then pulls via the public GCS JSON API. For a retrain of an existing dataset this step is a **no-op that CI should skip**: hit the same public JSON API, check the prefix exists, and jump straight to Modal. It only needs to run when `run_id` changes or the prompt template (tokenization) changes.
> 2. **Training is invoked directly, not through any CLI wrapper**: `modal run training/modal_train.py::train_qlora --model-name qwen3-14b --variant <v> --run-id <id> --run-name <n> ...` (or by subprocess-launching `scripts/run_3config_comparison.py`). Precondition already proven manually — 2/3 comparison variants completed on Modal.
> 3. **The same run-id string is carried in three places** (tokenize call, training `--run-id`, and the eval env var). GitHub Actions does not persist shell env across jobs, so define it once at workflow level (`env: DATASET_RUN_ID: expanded-repos`) and derive per-job. The eval job MUST receive `EVAL_DATASET_RUN_ID: ${{ env.DATASET_RUN_ID }}`: it drives `EvalConfig.dataset_run_id` (env prefix `EVAL_`), which substitutes `{run_id}` into `golden_data_path` (`gs://swe-qwen-datasets/datasets/{run_id}/swebench/golden.jsonl`, cached by `_ensure_golden`). If absent, `load_examples` silently falls back to the eval resume id / errors out — a quiet failure mode, so assert it is set in the eval job. It is inert for training (nothing in `scripts/` reads it).
> 4. **GitHub Actions job timeout is 360 min by default** vs Modal fn timeout 300 min — tight enough that a long training run can be cancelled mid-flight by CI. Every job that launches a training/eval Modal call must set `timeout-minutes:` explicitly above the expected runtime.
> 5. **Secrets handoff is the new moving part** (GCS is already de-risked: the bucket is public-read because GCP org policy blocks HMAC key creation). CI needs `WANDB_API_KEY`, `HF_TOKEN`, plus the **Modal token pair** (`MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`) — a GitHub Actions `modal run` CANNOT consume the Modal-hosted `wandb-secret`/`hf-secret` secrets by name. GCP write creds for the tokenize job come from the existing WIF provider (already wired in this file for terraform-plan).
> 6. **Cost amplification:** Modal `Retries(max_retries=1)` + the orchestrator's auto-restart will auto-relaunch expensive jobs with no human watching. Add a spend/attempt guard (max N launches per invocation) before letting CI trigger training.
> 7. **W&B auto-deletes inactive projects** (observed: project `swe-qwen` was auto-deleted once, all run telemetry lost). In CI, pin/re-create the project in a run-once init job before any training/eval job depends on it.
> 8. **The per-PR eval gate (7.4) is the single most expensive and least-proven automation unit** — GPU batch inference + real per-repo test containers. Budget most debug effort there. Gate it with `paths:` filters so docs/data-only PRs never fire it, and keep samples small (config already ships `ci_sample_size=50`, `ci_random_seed=42`, tier sizes).

**Dependencies:** All previous phases complete (at minimum Phase 6 — inference API must exist to test deployment). Phase 1 complete (Terraform + OIDC must be configured).

**Risks:**

- OIDC configuration for GCP may fail on first attempt → Mitigation: Test OIDC setup in Phase 7.1 in isolation before wiring into CI/CD
- GitHub Actions workflow timeout on evaluation runs (may be long) → Mitigation: Use Modal serverless for evaluation to parallelize; set appropriate timeout limits
- Quality gate threshold too strict/flexible → Mitigation: Start with a generous threshold; tighten in Phase 12 hardening

**Definition of Done:**

- [ ] PR must pass CI (lint, typecheck, tests) before merge
- [ ] PR must pass evaluation quality gate before merge (if model changes)
- [ ] Merge to main triggers CD: Terraform apply + (if promoted) deployment to Modal
- [ ] No long-lived GCP or Modal secrets in the repository
- [ ] Full CI/CD pipeline tested end-to-end on a feature branch

**Acceptance Criteria:**

1. A PR that fails lint or unit tests is blocked from merging
2. A PR that introduces a model with F2P below threshold is blocked from merging
3. Merging to main triggers infrastructure deployment via Terraform
4. A promoted model triggers deployment to the Modal inference endpoint
5. All secrets are managed via GitHub repository secrets + OIDC, never committed

**Expected repository state:** Complete CI/CD pipeline, OIDC auth, quality gates enforced, automated deployment from merge to production.

---

### Phase 8: Observability & Telemetry

**Objective:** Implement full structured telemetry across all platform components — training, evaluation, inference serving, infrastructure — with dashboards and alerts.

**Why it exists:** ADR-011 requires every stage to emit structured outputs. Observability is critical for debugging, cost monitoring, and proving platform quality. V1 uses W&B as the primary observability layer (per decisions), with the architecture ready for OpenTelemetry extension (v2).

**Inputs:**

- W&B project configured (Phase 1)
- All running components (training, inference, evaluation)

**Outputs:**

- `observability/` Python package
- W&B dashboards (training, evaluation, serving)
- Structured JSON logging across all components
- Metrics emission (latency, throughput, cost, F2P, loss)

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 8.1 | Implement structured JSON logging utility (all modules use this) | `observability/logging.py` |
| 8.2 | Add training metrics emission (loss, LR, gradient norm, GPU util, cost) to W&B | W&B training dashboards |
| 8.3 | Add evaluation metrics emission (F2P, P2P, per-example results, latency) to W&B | W&B eval dashboards |
| 8.4 | Add inference metrics emission (TTFB, tokens/sec, request count, error rate, cost) to W&B | W&B serving dashboards |
| 8.5 | Build W&B dashboard templates (one per domain: training, eval, serving, infra) | Dashboard configs |
| 8.6 | **Cost tracking implementation**: `observability/cost.py` — pulls Modal run metadata (GPU-hours × rate) + W&B run cost estimate; logs `cost_usd` metric per run; dashboard panel: cumulative cost, cost per F2P point | Cost tracking module + W&B cost dashboard |
| 8.7 | **Langfuse integration**: add `observability/langfuse.py` — trace LLM calls, prompt versions, evaluation runs; dual-write to W&B + Langfuse; dashboard: prompt performance, trace debugging, eval comparisons | Langfuse module + dual observability stack |
| 8.8 | Implement alert configuration (W&B alerts on metric degradation) | Alert rules |
| 8.9 | Document observability architecture and how to interpret dashboards | Observability docs |
| 8.10 | (Deferred: OpenTelemetry instrumentation) | — |

**Dependencies:** Phase 6 complete (serving metrics depend on running inference). Phase 4 complete (training metrics depend on active training).

**Risks:**

- Dashboard clutter (too many metrics, hard to read) → Mitigation: Start with 5-7 key metrics per domain; expand only when specific questions arise
- W&B cost limits on free tier → Mitigation: Use W&B's free personal tier; monitor usage; log metrics efficiently (aggregate, don't log raw per-sample data)

**Definition of Done:**

- [ ] W&B has 4 dashboards: training, evaluation, serving, infrastructure/cost
- [ ] Every component (ingest, validate, clean, train, eval, serve) emits structured JSON logs
- [ ] Key metrics are visible on dashboards in real-time during active experiments
- [ ] Cost per experiment and cumulative cost tracked in W&B
- [ ] **Langfuse traces LLM calls, prompt versions, and evaluation runs; dual-write to W&B + Langfuse working**
- [ ] Structured logging format documented

**Acceptance Criteria:**

1. A new experiment run appears on the W&B training dashboard within 60 seconds of starting
2. Inference requests are visible on the serving dashboard in real time
3. Cost tracking accurately reflects Modal and W&B spend per experiment
4. Structured logs can be queried programmatically (JSON format with consistent field names)
5. **Langfuse shows prompt→completion→eval_score traces for evaluation runs; prompt A/B comparison visible in dashboard**

**Expected repository state:** Full telemetry across all pipeline stages, W&B dashboards operational, structured logging in all components.

---

### Phase 9: Champion/Challenger Promotion Pipeline

**Objective:** Implement the automated Champion/Challenger model promotion pipeline (ADR-007). New candidate models are automatically evaluated against the current champion on the golden eval set; if they win by a configurable margin, they are automatically promoted.

**Why it exists:** Manual deployment decisions are prohibited (ADR-007). This phase implements the automated quality gate that makes model promotion a fully automated, reproducible process — mirroring enterprise continuous delivery for ML models.

**Inputs:**

- Evaluation harness from Phase 5
- Current champion model checkpoint (stored in W&B model registry)
- New candidate model checkpoint (from any training run)
- Promotion rules configuration (F2P threshold, minimum improvement)

**Outputs:**

- `promotion/` Python package
- Automated promotion logic
- W&B model registry integration (champion alias updates)
- Deployment trigger (triggers Phase 6 redeployment on promotion)
- Promotion audit trail in W&B

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 9.1 | Implement Champion/Challenger comparison engine | `promotion/gate.py` |
| 9.2 | Implement promotion rules (F2P improvement threshold, P2P regression ceiling) | `promotion/rules.py` |
| 9.3 | Integrate with W&B model registry (tag champion alias, store comparison results) | W&B registry integration |
| 9.4 | Implement deployment trigger on promotion (calls Modal deploy via API or CI/CD) | `promotion/deploy.py` |
| 9.5 | Implement promotion audit trail (who promoted what, why, when — all logged to W&B) | Audit logging |
| 9.6 | Write unit tests for promotion logic | `tests/test_promotion.py` |
| 9.7 | Run end-to-end promotion test: candidate vs. champion → evaluation → promotion decision → deployment trigger | Promotion test run |

**Dependencies:** Phase 5 complete (evaluation harness). Phase 6 complete (deployment target). Phase 7 complete (CI/CD already enforces gates).

**Risks:**

- Promotion threshold set too low → noisy promotions; too high → no promotions → Mitigation: Start with conservative threshold (candidate must beat champion by ≥5% F2P); adjust after observing 3-5 promotion cycles
- Automated deployment may break if model causes regressions → Mitigation: P2P regression ceiling blocks promotion; deployment includes smoke test; rollback documented

**Definition of Done:**

- [ ] Champion/Challenger comparison runs automatically on any new model checkpoint
- [ ] Promotion decision (promote/reject) is logged to W&B with full rationale
- [ ] Successful promotion triggers automatic deployment to inference endpoint
- [ ] rejected promotions have documented reason and metrics
- [ ] Unit tests cover all promotion rule scenarios

**Acceptance Criteria:**

1. A new model checkpoint automatically enters the Challenger lane when logged to W&B
2. Evaluation runs automatically compared to current Champion
3. If F2P improves by ≥ threshold AND P2P regression ≤ ceiling → champion alias updated automatically
4. Champion alias update triggers redeployment to Modal inference endpoint
5. Full audit trail exists in W&B for every promotion decision (candidate, champion, metrics, decision, timestamp)

**Expected repository state:** End-to-end automated model promotion pipeline, champion alias tracked in W&B, deployment triggered by promotion, full audit trail.

---

### Phase 10: Documentation

**Objective:** Produce production-grade technical documentation covering the architecture, deployment guide, dataset engineering methodology, evaluation protocol, experiment tracking, and all component APIs.

**Why it exists:** The ADR specifies "Production-grade technical documentation" as a deliverable (Deliverable #8). Good documentation is also essential for a portfolio project — it demonstrates communication skills that matter in interviews.

**Inputs:**

- All completed phases (source of truth for how things work)
- ADR & Vision document (for architectural context)
- This Master Plan (for implementation context)

**Outputs:**

- Complete documentation set in `docs/`
- Architecture Decision Records reference
- Deployment guide
- API documentation
- Experiment guide
- Dataset card templates
- CONTRIBUTING.md

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 10.1 | Write architecture overview (narrative + diagram) | `docs/architecture.md` |
| 10.2 | Write deployment guide (Terraform apply, Modal deploy, CI/CD trigger) | `docs/deployment.md` |
| 10.3 | Write API reference (OpenAI-compatible endpoints, request/response schemas) | `docs/api.md` |
| 10.4 | Write experiment guide (how to run training, evaluation, promotion) | `docs/experiments.md` |
| 10.5 | Write dataset engineering guide (how the pipeline works, schema, quality criteria) | `docs/dataset.md` |
| 10.6 | Write evaluation methodology document (F2P, P2P, golden set protocol) | `docs/evaluation.md` |
| 10.7 | Write CONTRIBUTING.md (how to contribute, code style, testing) | `CONTRIBUTING.md` |
| 10.8 | Write README update (comprehensive overview, quick-start, architecture diagram) | `README.md` update |
| 10.9 | AddADR cross-reference index (which ADRs apply to which components) | `docs/adr_index.md` |
| 10.10 | Review all documentation for accuracy against actual implementation | Documentation review |

**Dependencies:** All phases 1-9 complete (documentation must reflect the actual working system).

**Risks:**

- Documentation drifts from implementation during earlier phases → Mitigation: Write documentation as each phase completes (don't defer all docs to Phase 10); Phase 10 is a review pass
- Over-documentation for a portfolio project → Mitigation: Focus on what a recruiter would want to see: architecture, evaluation methodology, and how to run/reproduce results

**Definition of Done:**

- [ ] Every component has documented API/usage in `docs/`
- [ ] Deployment can be reproduced from docs alone
- [ ] Architecture diagram is current and accurate
- [ ] Experiment reproduction steps are tested and verified
- [ ] All docs pass `markdownlint` or equivalent

**Acceptance Criteria:**

1. A new reader can understand the full architecture from `docs/architecture.md` in ≤ 15 minutes
2. A new engineer can deploy the platform end-to-end from `docs/deployment.md`
3. API documentation matches actual endpoint behavior (tested)
4. Experiment guide produces reproducible results

---

### Phase 11: Hardening & Resilience (NOT DOING, WASTE OF TIME)

**Objective:** Strengthen error handling, edge case coverage, retry logic, input validation, model fallback behavior, and operational resilience across all components.

**Why it exists:** A production-grade platform must handle failures gracefully. This phase focuses on the operational maturity that separates a demo from a real system.

**Inputs:**

- All previous phases complete
- Known failure modes discovered during testing

**Outputs:**

- Improved error handling across all modules
- Retry logic with exponential backoff
- Input validation hardening
- Model fallback chain (primary model → fallback model)
- Circuit breaker patterns for external dependencies (GitHub API, Modal, W&B)
- Edge case test coverage expanded

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 11.1 | Audit all external API calls for timeout/retry handling | Audit report |
| 11.2 | Add retry logic with exponential backoff to GitHub API calls | Updated ingest.py |
| 11.3 | Add retry logic to Modal API calls | Updated modal_train.py, modal_serve.py |
| 11.4 | Add input validation hardening to inference API | Updated openai_compat.py |
| 11.5 | Implement model fallback chain (Qwen3-14B → Qwen3-30B-A3B) | Fallback logic |
| 11.6 | Add circuit breaker for GitHub API (fail fast if rate limited) | Circuit breaker |
| 11.7 | Expand edge case test coverage (malformed patches, empty repos, missing test suites) | Expanded tests |
| 11.8 | Validate error messages are actionable and logged | Error message audit |
| 11.9 | Run failure injection tests (simulate GitHub API failure, Modal timeout, etc.) | Resilience tests |

**Dependencies:** Phases 1-10 complete.

**Risks:**

- Over-engineering resilience for a portfolio project → Mitigation: Focus on the 5 most common failure modes; don't build for every edge case

**Definition of Done:**

- [ ] All external API calls have retry + timeout + circuit breaker
- [ ] Model fallback chain works: primary model unavailable → Qwen3-14B serves
- [ ] Edge case tests pass (malformed inputs, missing data, API failures)
- [ ] Error messages are logged with sufficient context for debugging

**Acceptance Criteria:**

1. When GitHub API returns 403 (rate limit), the pipeline retries with backoff and eventually succeeds or fails with a clear error message
2. When Modal is unavailable, the inference API returns a meaningful error response (503 with retry-after hint)
3. When the primary model checkpoint fails to load, the fallback model is loaded automatically
4. All failures are logged with enough context to diagnose the root cause without reproducing the failure

---

### Phase 12: End-to-End Validation (NOT DOING, WASTE OF TIME)

**Objective:** Run the complete pipeline from data ingestion through trained model to serving API to promotion, validating that every component works together as a system.

**Why it exists:** ADR-012 (vertical slice delivery) requires each phase to produce a working increment, but only end-to-end validation proves the system works as a whole. This phase is the final quality gate before launch.

**Inputs:**

- All previous phases complete
- Full infrastructure deployed (Terraform)
- Model checkpoint from Phase 4
- Inference API from Phase 6
- CI/CD pipeline from Phase 7

**Outputs:**

- End-to-end validation report
- Performance benchmarks (latency, throughput, cost)
- Final F2P/P2P metrics on golden eval set
- Any bugs or integration issues found and fixed

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 12.1 | Run full pipeline from scratch on a clean environment | Validation run |
| 12.2 | Validate data pipeline: ingest → validate → clean → split → archive | Data pipeline validation |
| 12.3 | Validate training pipeline: dataset → training → checkpoint → W&B | Training validation |
| 12.4 | Validate evaluation: checkpoint → golden set → F2P/P2P metrics | Eval validation |
| 12.5 | Validate inference API: checkpoint → deploy → chat completion → response | Inference validation |
| 12.6 | Validate CI/CD: PR triggers → lint/test/eval/gate → merge → deploy | CI/CD validation |
| 12.7 | Validate promotion: candidate → evaluation → compare → promote → deploy | Promotion validation |
| 12.8 | Measure end-to-end latency (issue → prediction → evaluation) | Latency benchmark |
| 12.9 | Measure end-to-end cost (data ingest → training → serving → evaluation) | Cost analysis |
| 12.10 | Write validation report with findings and any fixes applied | Validation report |

**Dependencies:** All phases 1-11 complete.

**Risks:**

- Integration issues discovered late → Mitigation: This phase exists specifically to catch them; budget time for fixes
- Performance below target → Mitigation: Benchmark results are documented as baselines for the Future Enhancements roadmap

**Definition of Done:**

- [ ] Full pipeline runs end-to-end without human intervention
- [ ] F2P rate on golden eval set meets or exceeds baseline target
- [ ] Inference API responds to OpenAI-compatible requests within SLA
- [ ] CI/CD pipeline passes on clean PR
- [ ] Promotion pipeline runs successfully
- [ ] All integration issues found and fixed

**Acceptance Criteria:**

1. A new clone of the repository can be set up and the full pipeline executed from `README.md` instructions
2. The end-to-end F2P result is comparable to Phase 5 results (no regression from integration)
3. Total pipeline cost is documented and within budget expectations
4. All validation issues are resolved and retested

---

### Phase 13: Production Launch & Portfolio Presentation

**Objective:** Final production deployment, portfolio presentation package, and project handoff. The platform is considered complete and ready for review.

**Why it exists:** The project exists to demonstrate production LLMOps competency to AI Engineering recruiters. This phase ensures the project is presented at its best.

**Inputs:**

- Validated platform from Phase 12
- Portfolio presentation requirements (recruiter-facing portfolio)

**Outputs:**

- Production deployment (Modal inference endpoint live)
- Portfolio README / showcase document
- Final benchmark results package
- Project summary for CV/LinkedIn
- Final W&B project with full experiment history

**Tasks:**

| # | Task | Deliverable |
| --- | ------ | ------------- |
| 13.1 | Deploy production inference endpoint (final promoted model) | Live endpoint |
| 13.2 | Write portfolio showcase document (narrative, architecture decisions, results, learnings) | Portfolio doc |
| 13.3 | Package final benchmark results (F2P, P2P, latency, cost) | Benchmark report |
| 13.4 | Write CV/LinkedIn project summary (2-3 paragraph description) | CV summary |
| 13.5 | Tag final release in Git (v1.0.0) | Git tag |
| 13.6 | Write project wrap-up: decisions made, tradeoffs, what to improve next | Retro doc |
| 13.7 | Update README to highlight the finished project | README update |
| 13.8 | Final documentation review | Docs review |
| 13.9 | **Publish model card to Hugging Face Hub** (model description, training details, eval results, license, usage examples) | HF Hub model card |

**Dependencies:** Phase 12 complete.

**Risks:**

- Showcase document doesn't effectively communicate the project → Mitigation: Review with a peer or mentor; iterate on narrative

**Definition of Done:**

- [ ] Inference endpoint is live and serving requests
- [ ] Portfolio showcase document exists and tells a complete story
- [ ] CV summary is accurate and compelling
- [ ] Git tag v1.0.0 exists on main
- [ ] All documentation is current and accurate
- [ ] **Model card published to Hugging Face Hub**

**Acceptance Criteria:**

1. The project can be presented to an AI Engineering recruiter as a complete, working platform
2. A recruiter reading the README + portfolio doc understands the full scope and technical depth
3. The inference endpoint is accessible and demonstrated in the portfolio doc
4. All ADR cross-references are valid and complete
5. **Model card is live on HF Hub with training details, eval results, license, and usage examples**

---

## 9. Milestones & Phase Exit Gates

| Milestone | Phase | Gate Criteria |
| ----------- | ------- | --------------- |
| **M1: Scaffold** | Phase 1 complete | Terraform plan succeeds, Modal config works, W&B project created |
| **M2: Sources Ready** | Phase 2 complete | 10 repos selected, verified, documented in manifest |
| **M3: Data Ready** | Phase 3 complete | Validated dataset (8-12k examples), golden eval subset, W&B artifacts |
| **M4: Model Trained** | Phase 4 complete | LoRA checkpoint trained, W&B lineage complete, baseline evaluation done |
| **M5: Eval Working** | Phase 5 complete | F2P/P2P metrics computed for baseline and fine-tuned, logged to W&B |
| **M6: API Live** | Phase 6 complete | OpenAI-compatible endpoint serving, scale-to-zero verified, benchmarks logged |
| **M7: Pipeline Automated** | Phase 7 complete | CI/CD with quality gates, OIDC auth, automated deploy on merge |
| **M8: Observable** | Phase 8 complete | Dashboards live, structured logging in all components, cost tracking |
| **M9: Auto Promotion** | Phase 9 complete | Champion/Challenger pipeline automated, promotion triggers deployment |
| **M10: Documented** | Phase 10 complete | All docs written, reviewed, and accurate |
| **M11: Hardened** | Phase 11 complete | Error handling, retries, fallbacks, circuit breakers in place |
| **M12: Validated** | Phase 12 complete | End-to-end pipeline passes on clean environment |
| **M13: Launched** | Phase 13 complete | v1.0.0 tagged, production endpoint live, portfolio ready |

---

## 10. Deliverables

### Phase-Delivered Deliverables

| Phase | Deliverables |
| ------- | ------------- |
| 1 | Scaffolded repo, Terraform infrastructure, Modal config, W&B project, CI skeleton |
| 2 | Curated repo manifest (10 repos), verification scripts, selection rationale |
| 3 | Complete `data_engineering/` package, validated dataset, golden eval subset, W&B dataset artifacts |
| 4 | Complete `training/` package, trained LoRA checkpoint, W&B experiment runs, baseline eval report |
| 5 | Complete `evaluation/` package, F2P/P2P metrics on golden set, comparison reports |
| 6 | Complete `inference/` package, OpenAI-compatible serverless API on Modal, benchmarks |
| 7 | Complete CI/CD workflows (`.github/workflows/`), OIDC auth, quality gates |
| 8 | `observability/` package, W&B dashboards, structured logging, cost tracking |
| 9 | `promotion/` package, automated Champion/Challenger pipeline, promotion audit trail |
| 10 | Complete `docs/` (architecture, deployment, API, experiments, evaluation, dataset, contributing) |
| 11 | Error handling hardening, retry logic, fallback chain, circuit breakers, resilience tests |
| 12 | End-to-end validation report, performance benchmarks, cost analysis |
| 13 | Production deployment, portfolio showcase, CV summary, v1.0.0 release tag |

### Final Deliverable Checklist (Mapped to ADR-008/009/010/011)

- [x] Data Pipeline → ADR-004 deliverable (1)
- [x] Fine-Tuning Engine → ADR-003 deliverable (2)
- [x] Execution Evaluation Harness → ADR-005 deliverable (3)
- [x] W&B Model Registry → ADR-006 deliverable (4)
- [x] Terraform Infrastructure → ADR-008 deliverable (5)
- [x] CI/CD with Quality Gates → ADR-009 deliverable (6)
- [x] Serverless vLLM Inference API → ADR-010 deliverable (7)
- [x] Telemetry & Observability → ADR-011 deliverable (8)
- [x] Technical Documentation → Vision doc deliverable (8)

---

## 11. Dependencies

### Phase Dependencies (Hard)

```
Phase 1 ──────────────────────────────────────────────────────┐
                                                              │
Phase 2 ────▶ Phase 3 ────▶ Phase 4 ──┐                    │
                                       │                    │
Phase 5 (requires Phase 3 + 4) ◄───────┘                    │
       │                                                    │
Phase 6 (requires Phase 4) ◄───────────────────────────────┘
       │
Phase 7 (requires Phase 1 + 6) ◄───────────────────────────────┐
       │                                                        │
Phase 8 (requires Phase 4 + 6) ◄───────────────────────────────┤
       │                                                        │
Phase 9 (requires Phase 5 + 6) ◄───────────────────────────────┤
       │                                                        │
Phase 10 (requires all prior) ◄────────────────────────────────┤
       │                                                        │
Phase 11 (requires all prior) ◄────────────────────────────────┤
       │                                                        │
Phase 12 (requires all prior) ◄────────────────────────────────┤
       │                                                        │
Phase 13 (requires Phase 12) ◄─────────────────────────────────┘
```

### External Dependencies

| Dependency | Required When | Source | Notes |
| ----------- | -------------- | -------- | ------- |
| Modal API | Phase 1 onward | Modal platform | API key in GitHub secrets |
| GCP (GCS, IAM) | Phase 1 onward (Terraform) | GCP account | OIDC keyless auth |
| GitHub API | Phase 3 onward | GitHub (token) | Rate limits managed in code |
| Weights & Biases | Phase 1 onward | W&B platform | API key in GitHub secrets |
| vLLM | Phase 6 onward | vLLM (pip install) | Runs on Modal |
| QLoRA / PEFT | Phase 4 onward | HuggingFace PEFT | pip install |
| Transformers | Phase 4+ onward | HuggingFace | pip install |
| bitsandbytes | Phase 4 onward | HuggingFace | 4-bit quantization |
| pytest | Phase 1+ | pip | Testing framework |
| Ruff | Phase 1+ | pip | Linting |
| mypy | Phase 1+ | pip | Type checking |
| Pydantic | Phase 3+ | pip | Data validation |
| GitHub Actions | Phase 7 onward | GitHub | Free for public repos |
| Terraform | Phase 1+ | HashiCorp | Infrastructure provisioning |
| Python ≥ 3.11 | Phase 1+ | python.org | Runtime |

### Internal Dependencies (Cross-Workstream)

| Dependency | From → To | Impact if Missing |
| ----------- | ----------- | ------------------- |
| Dataset artifact | Phase 3 → Phase 4 | Training has no data |
| Model checkpoint | Phase 4 → Phase 5 | No model to evaluate |
| Model checkpoint | Phase 4 → Phase 6 | No model to serve |
| Evaluation harness | Phase 5 → Phase 9 | No metrics for promotion |
| Inference API | Phase 6 → Phase 9 | No deployment target |
| W&B project | Phase 1 → All | No experiment tracking anywhere |
| CI/CD | Phase 7 → Phase 9 | Promotion has no automated trigger |
| Terraform infra | Phase 1 → Phase 3, 6 | No GCS, no IAM, no secrets |

---

## 12. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
| --- | ------ | ----------- | -------- | ----------- |
| R1 | Qwen3-30B-A3B exceeds available VRAM on Modal | ~~Medium~~ Resolved | ~~High~~ N/A | **Phase 4 pivot: 14B-only, 30B excluded.** Qwen3-14B on A10G 24GB works. 30B config retained for future phases |
| R2 | Dataset yield is lower than expected (fewer valid issue-PR pairs than 8k) | Medium | Medium | Start with larger candidate pool (up to 20 repos); accept smaller dataset if quality is high; synthetic examples only as last resort |
| R3 | Modal costs unexpectedly high for training runs | Low-Medium | Medium | Use Modal's scale-to-zero for training; benchmark cost per hour pre-committed; set budget alerts; document cost per experiment |
| R4 | vLLM cold start latency too high for production feel | Medium | Low | Acceptable for V1 (scale-to-zero tradeoff); optimization deferred to v2. Document as known limitation |
| R5 | GitHub API rate limiting blocks data ingestion | Medium | High | Implement exponential backoff, batch requests, use GitHub App token (higher limits), cache responses |
| R6 | CI/CD pipeline timeout on evaluation runs | Low-Medium | Medium | Run evaluation asynchronously; use Modal for parallel test execution; increase workflow timeout limits |
| R7 | Golden eval subset too small for statistical significance | Low | Medium | Target 800+ examples in golden set; if insufficient, expand candidate pool in Phase 2; document as limitation |
| R8 | Quality gate threshold miscalibrated leading to false promotions | Low | High | Start with conservative thresholds; manual review for first 3 promotions; tighten after observing results |
| R9 | W&B free tier limits reached | Low | Low | Monitor usage; W&B free tier sufficient for V1 scale; upgrade only if needed |
| R10 | Terraform state drift between phases | Low | Medium | Run `terraform plan` before each apply; use remote state (GCS); treat state file as critical artifact |

---

## 13. Testing Strategy

### Testing Layers

| Layer | Scope | Tools | Frequency |
| ------- | ------- | ------- | ----------- |
| **Unit** | Individual functions/methods | pytest | Every PR |
| **Integration** | Module-to-module interaction | pytest + fixtures | Every PR |
| **End-to-End** | Full pipeline execution | pytest + Modal + GCP | Nightly / on release |
| **Benchmark** | Performance and quality targets | Custom scripts | Weekly during active training |
| **Resilience** | Failure modes and recovery | Chaos-style tests | Phase 11 onward |

### Test Organization

```
tests/
├── test_data.py            # Data pipeline validation
├── test_training.py        # Training pipeline logic
├── test_evaluation.py      # F2P/P2P computation
├── test_inference.py       # API request/response, OpenAI compat
├── test_promotion.py       # Champion/Challenger logic
├── test_infra.py           # Terraform output validation
├── test_resilience.py      # Error handling, retry, fallback
├── conftest.py             # Shared fixtures
├── fixtures/
│   ├── sample_issues.json  # Synthetic test data
│   ├── sample_patches.json
│   └── golden_sample.json
```

### Key Test Cases

1. **Data Pipeline:** Given 10 repos with issues and PRs, produce ≥8k validated examples with schema compliance ≥95%
2. **Training:** Given a dataset, training completes without OOM within Modal GPU budget (A10G 24GB for 14B dense)
3. **Evaluation:** Given a fine-tuned model and golden set, F2P rate is computed and logged to W&B
4. **Inference:** Given an OpenAI-compatible request, the endpoint returns a valid completion in <500ms p50
5. **Promotion:** Given a candidate model better than champion by threshold, promotion triggers deployment
6. **CI/CD:** Given a PR, all gates execute without manual intervention

### Test Quality Targets

- Unit test coverage ≥ 80% for all packages
- Integration tests run in CI on every PR
- End-to-end tests run on every release candidate

---

## 14. Documentation Strategy

### Documentation Layers

| Layer | Content | Location | Audience |
| ------- | --------- | ---------- | ---------- |
| **Arch** | System architecture, component interaction | `docs/architecture.md` | Engineers, reviewers |
| **API** | Endpoint schemas, request/response formats | `docs/api.md` | Integrators |
| **Dev** | How to run, develop, test locally | `README.md`, `docs/development.md` | Contributors |
| **Deploy** | Terraform apply steps, Modal deploy, CI/CD | `docs/deployment.md` | DevOps, reviewers |
| **Data** | Dataset schema, source repos, quality criteria | `docs/dataset.md` | ML engineers |
| **Experiments** | How to run training, hyperparams, results | `docs/experiments.md` | ML engineers |
| **Eval** | F2P/P2P methodology, golden set protocol | `docs/evaluation.md` | reviewers, QA |
| **Ops** | Troubleshooting, monitoring, alerts | `docs/operations.md` | SRE, on-call |
| **ADR** | Decision records (referenced from ADR doc) | `docs/ADR-&-VISION.md` | All |
| **Portfolio** | Narrative summary, results, showcase | `docs/portfolio.md` | Recruiters |

### Documentation Principles

- **Living docs:** Every phase updates relevant docs as it completes
- **No orphan docs:** Every doc has an owner (the phase that produces it)
- **Accuracy over completeness:** A correct short doc beats an incomplete long one
- **Code as docs:** API documentation comes from docstrings + `mkdocstrings` or equivalent
- **Recruiter-first:** The README and portfolio doc are written for AI Engineering recruiters

---

## 15. CI/CD Evolution

### CI/CD Architecture

```
GitHub PR
    │
    ▼
┌─────────────────────────────────────────────────┐
│ CI Pipeline (GitHub Actions)                    │
│ ├── Ruff lint                                   │
│ ├── mypy typecheck                              │
│ ├── pytest (unit + integration)                 │
│ ├── coverage check (≥80%)                       │
│ ├── terraform plan (validate infra changes)     │
│ └── eval gate (if model changes: F2P check)    │
│                                                 │
│ Pass → Merge allowed                            │
│ Fail → Block merge, report required             │
└─────────────────────────────────────────────────┘
    │ (merge to main)
    ▼
┌─────────────────────────────────────────────────┐
│ CD Pipeline (GitHub Actions)                    │
│ ├── Terraform apply (infra)                     │
│ ├── If model promoted: deploy to Modal          │
│ ├── W&B run: log deployment event              │
│ └── Slack/Discord notification (optional)       │
└─────────────────────────────────────────────────┘
```

### Auth Strategy

- **GitHub → GCP:** Workload Identity Federation (no long-lived GCP keys)
- **GitHub → Modal:** Scoped API token in GitHub Secrets
- **GitHub → W&B:** API key in GitHub Secrets
- **GitHub → GitHub:** Native (PAT for repo operations only if needed)

### CI/CD Evolution (V1 → v2)

| V1 (This Plan) | v2 (Future) |
| ---------------- | ------------- |
| GitHub Actions for all pipelines | Same |
| OIDC for GCP only | OIDC for GCP + Modal |
| Quality gate: F2P threshold | Quality gate: F2P + P2P + latency SLA |
| Manual promotion trigger after Phase 9 | Fully automated promotion in CD |
| No deployment rollback | Rollback via champion alias revert |
| W&B + Langfuse dual observability | W&B + Langfuse + OpenTelemetry |

---

## 16. Infrastructure Evolution

### V1 Infrastructure (Terraform)

```
Module: storage
├── GCS Bucket (datasets, checkpoints, artifacts)
├── GCS Bucket Policy (Least-privilege access)
├── GCS Lifecycle Policy (archive old versions)
└── GCS Versioning (enabled)

Module: iam
├── Service Account for GitHub Actions WIF
├── Service Account for Modal (if cross-platform)
├── IAM Policy: GCS read/write (scoped)
├── IAM Policy: W&B API calls (no GCP IAM needed for W&B)
└── Secrets in GitHub (no GCP Secret Manager needed for V1)
```

### v2 Infrastructure Additions (Future)

```
Module: monitoring
├── Cloud Monitoring dashboards (optional)
├── Alerting on cost thresholds
└── Deployment success/failure notifications

Module: networking
├── VPC for Modal (if required by Modal pricing tier)
└── Private link to GCS (optional)
```

### Infrastructure Cost Model

| Resource | V1 Cost | Notes |
| ---------- | --------- | ------- |
| GCS storage | < $1/month | Minimal dataset + checkpoint storage |
| GCS requests | < $0.10/month | Infrequent access pattern |
| Modal GPU training | Pay-per-use | ~$0.50-2.00 per training run (estimated) |
| Modal GPU inference | Pay-per-use | $0 when idle (scale-to-zero) |
| W&B tracking | Free | Personal tier sufficient for V1 |
| GitHub Actions | Free | Public repo, generous free minutes |
| Terraform state | Free | GCS-backed, no additional cost |
| **Total estimated V1 cost** | **<$50/month** | Running inference + periodic retraining |

---

## 17. Data Lifecycle

The data lifecycle follows ADR-004's defined stages, implemented across the pipeline:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ RAW DATA │────▶│ VALIDATE │────▶│ CLEAN    │────▶│ SPLIT    │────▶│ ARCHIVE  │────▶│ VERSION  │
│ (GitHub  │     │ (Schema  │     │ (Dedup,  │     │ (Train/  │     │ (GCS +   │     │ (W&B     │
│  Issues, │     │  check)  │     │  Normal- │     │  Val/Test│     │  GCS)    │     │  Dataset │
│  PRs)    │     │          │     │  ize)    │     │          │     │          │     │  Artifact)│
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
       │                │                │                │                │                │
       ▼                ▼                ▼                ▼                ▼                ▼
   GitHub API      Pydantic        Dedup by          Stratified      Terraform        W&B run
   response        validation      repo+issue        by repo         managed          linkage
                    schema          pair fingerprint  split           GCS bucket
```

### Data Quality Gates

| Stage | Gate | Action on Failure |
| ------- | ------ | ------------------- |
| Validation | All required fields present, schema valid | Reject record, log reason |
| Cleaning | No duplicate issue-PR pairs | Remove duplicate, log count |
| Splitting | No data leakage across splits (same repo stays in one split) | Re-split if leakage detected |
| Archiving | GCS upload successful, checksums match | Retry upload, fail after 3 attempts |
| Versioning | W&B artifact logged successfully | Halt pipeline, alert |

### Data Versioning

- Each pipeline run produces a uniquely versioned dataset (e.g., `dataset-v20260724-abc123`)
- W&B tracks: raw data hash → cleaned data hash → split configuration → final dataset artifact
- GCS stores: raw JSON, cleaned JSON, split artifacts, golden eval set
- Dataset card auto-generated with: record count, schema version, source repos, quality stats, split distribution

---

## 18. Model Lifecycle

### V1 Model Lifecycle (Simplified)

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│ Candidate  │────▶│  Baseline  │────▶│  Train     │────▶│  Evaluate  │────▶│  Promote  │
│ Model      │     │  Eval      │     │  QLoRA     │     │  (F2P/P2P) │     │  or Reject│
│            │     │            │     │  on Modal  │     │  vs Champion│    │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘     └──────────┬──┘
                                                                                      │
                                                                              ┌───────┴───────┐
                                                                              │  Deploy if    │
                                                                              │  Champion     │
                                                                              └───────────────┘
```

### V1 Model Lifecycle Details

| Stage | Owner | Tool | Output |
| ------- | ------- | ------ | -------- |
| Candidate Selection | Manual (Phase 4.1) | Memory + report | Model shortlist with memory fit report |
| Baseline | Training pipeline | W&B | Baseline metrics (F2P, P2P, latency, cost) on prompt-only model |
| Training | QLoRA pipeline (Modal) | W&B | LoRA adapter checkpoint |
| Evaluation | Evaluation harness (Phase 5) | W&B | F2P, P2P per example, comparison to baseline |
| Promotion Decision | Champion/Challenger pipeline (Phase 9) | W&B model registry | Champion alias updated or candidate rejected |
| Deployment | CD pipeline (Phase 7) | Modal | vLLM serverless endpoint with new model |

### Model Registry (W&B)

- Each model version is logged to W&B with aliases: `latest`, `champion`, `candidate`
- `champion` alias points to the current production model
- `candidate` alias points to the model under evaluation
- `latest` alias points to the most recent trained model regardless of performance
- Promotion = update `champion` alias in W&B → triggers CD → updates Modal endpoint

---

## 19. Decision Traceability

This section maps every implementation decision in this Master Plan back to the ADR or conversation decision that produced it.

| Master Plan Decision | ADR / Conversation Ref | Status |
| ---------------------- | ---------------------- | -------- |
| Foundation model: Qwen3-14B (primary), Qwen3-30B-A3B (future) | Conversation decision + P4 pivot | Locked |
| GPU platform: Modal only (Baseten removed) | Conversation decision + ADR-010 update | Locked |
| Training on Modal | Conversation decision | Locked |
| GCP only for cloud | Conversation decision | Locked |
| OpenAI-compatible API (with streaming) | Conversation decision | Locked |
| 8-12k dataset from ~10 repos | Conversation decision | Locked |
| W&B + Langfuse for V1 monitoring | Conversation decision + ADR-006 | Locked |
| No execution-feedback conditioning in V1 | Conversation decision | Deferred to v2 |
| QLoRA fine-tuning | ADR-003 | Locked |
| vLLM serverless serving | ADR-010 (updated: Modal only) | Locked |
| F2P primary metric | ADR-005 | Locked |
| W&B experiment tracking | ADR-006 | Locked |
| Champion/Challenger promotion | ADR-007 | Locked |
| Terraform IaC | ADR-008 | Locked |
| CI/CD with quality gates | ADR-009 | Locked |
| Structured telemetry | ADR-011 | Locked |
| Vertical slice delivery | ADR-012 | Locked |
| GitHub Actions for CI/CD | ADR-009 implementation | Locked |
| Modal for serverless inference | ADR-010 (updated) | Locked |
| GCS for artifact storage | ADR-008 implementation | Locked |
| OIDC keyless auth | ADR-008 implementation | Locked |
| Mandatory 3-config QLoRA comparison | Conversation decision | Locked |
| SWE-bench Verified evaluation tier | Conversation decision | Locked |
| HF Hub model card publication | Conversation decision | Locked |

---

## 20. Future Enhancements Roadmap

### v1.1 (After Launch)

| Enhancement | Rationale | Effort |
| ------------ | ----------- | -------- |
| OpenTelemetry instrumentation | ADR-011 intent; broader industry adoption; extensible observability | Medium |
| Prometheus + Grafana dashboards | Already used in other projects by author; deeper metrics than W&B alone | Medium |
| Automated rollback on serving errors | Champion alias revert + health check failure triggers rollback | Low |
| Dataset expansion (12k → 25k+) | More data improves F2P; engineering overhead increases | Low-Medium |

### v2 (Major)

| Enhancement | Rationale | Effort |
| ------------ | ----------- | -------- |
| Execution-feedback conditioning | Fine-tune on failures from real test runs (ADR deferred item) | High |
| Multi-agent evaluation framework | More comprehensive evaluation beyond F2P | High |
| GCP/Cross-cloud portability | ADR-002 principle (cloud-portable) | Medium |
| Multi-GPU training support | Faster training, larger models | Medium |
| Hyperparameter tuning automation (Optuna) | Bayesian optimization over QLoRA configs (budget >50 GPU-hrs) | Medium |

### v3 (Stretch)

| Enhancement | Rationale | Effort |
| ------------ | ----------- | -------- |
| RLHF/DPO for preference alignment | State-of-the-art post-training | High |
| Autonomous issue triage | End-to-end automated bug resolution | Very High |
| Multi-language support (non-Python repos) | Broader applicability | High |
| Custom evaluation benchmarks | Domain-specific eval beyond SWE-bench | Medium |

---

## 21. Definition of Done

### Per-Phase DoD

A phase is **Done** when:

1. All tasks in the phase plan are complete
2. All deliverables for the phase exist and pass acceptance criteria
3. All unit and integration tests for the phase pass
4. W&B shows complete lineage for any experiments in the phase
5. Documentation for the phase is written or updated
6. The repository is in a deployable state (vertical slice: the increment works end-to-end)
7. Phase exit gate criteria are met (see Section 9)

### Project Complete DoD

The project is **Complete** when:

1. All 13 phases are Done
2. The platform is fully deployable from a clean clone
3. All ADR traceability entries are valid
4. The inference endpoint is live and serving requests
5. The promotion pipeline is automated and verified
6. All documentation is current and accurate
7. A recruiter (or AI Engineering lead) can review the project and understand the full LLMOps value demonstrated
8. v1.0.0 is tagged in Git
9. The project demonstrates at least 7 of the 10 success criteria from Section 2.2

---

## 22. Appendices

### Appendix A: ADR Reference Index

| ADR | Decision | Referenced In |
| ----- | ---------- | --------------- |
| ADR-001 | Project Framing & Domain Focus | This plan (Scope, Objectives) |
| ADR-002 | Platform & Model Independence | This plan (Workstream design, model selection process) |
| ADR-003 | Fine-Tuning Methodology (QLoRA) | This plan (Phase 4) |
| ADR-004 | Dataset Engineering & Strategy | This plan (Phase 3, Data Lifecycle) |
| ADR-005 | Evaluation Philosophy & Metrics | This plan (Phase 5, Success Criteria) |
| ADR-006 | Experiment Tracking (W&B) | This plan (Phase 1+ monitoring) |
| ADR-007 | Champion/Challenger Promotion | This plan (Phase 9) |
| ADR-008 | IaC & Cloud Foundation | This plan (Phase 1, Terraform sections) |
| ADR-009 | CI/CD & Model Quality Gates | This plan (Phase 7) |
| ADR-010 | Model Deployment Architecture | This plan (Phase 6 — updated to Modal only) |
| ADR-011 | Observability & Telemetry | This plan (Phase 8) |
| ADR-012 | Vertical Slice Delivery | This plan (all phases, Section 4) |

### Appendix B: Glossary

| Term | Definition |
| ------ | ----------- |
| **F2P (Fail-to-Pass)** | Percentage of previously-failing tests that pass after applying a generated code patch |
| **P2P (Pass-to-Pass)** | Percentage of previously-passing tests that still pass after applying a generated code patch |
| **QLoRA** | Quantized Low-Rank Adaptation — a parameter-efficient fine-tuning method using 4-bit quantization and low-rank adapter matrices |
| **LoRA Adapter** | A small set of trainable weight matrices inserted into a pre-trained model; can be merged back or applied at inference time |
| **vLLM** | High-throughput LLM inference engine with PagedAttention, serving multiple requests efficiently |
| **Golden Eval Set** | A subset of examples where test-suite results are verified — the ground truth for evaluation |
| **Champion/Challenger** | Automated A/B testing pattern: new model (challenger) is compared against current best (champion) and promoted if it wins |
| **Scale-to-Zero** | Serverless pattern where GPU resources are released when idle, costing $0 until the next request |
| **OIDC** | OpenID Connect — a federated identity protocol allowing GitHub Actions to authenticate to GCP without long-lived credentials |
| **Vertical Slice** | A small, end-to-end functional increment that crosses all layers of the system |

### Appendix C: Modal Configuration Reference

```yaml
# Example Modal configuration for this project
modal_config:
  # Training
  training:
    image: "python:3.11-slim"
    gpu: "A10G"      # 24GB VRAM, fits Qwen3-14B QLoRA 4-bit (P4 pivot: 14B-only)
    timeout: 3600      # 1 hour max per training job
    volumes:
      - /models       # Persistent storage for checkpoints
    env:
      - WANDB_API_KEY
      - GCP_PROJECT_ID

  # Inference
  inference:
    image: "vllm/vllm-openai:latest"
    gpu: "A10G"
    timeout: 86400     # Persistent for serving
    keep_warm: false   # Scale-to-zero when idle
    ports:
      - 8000           # OpenAI-compatible API port
    env:
      - MODEL_PATH    # GCS path to LoRA adapter + base model
      - WANDB_API_KEY
```

### Appendix D: File Manifest (Expected)

This is the expected file manifest after all phases are complete:

```
swe-qwen/
├── README.md                          ← Project overview + quick-start
├── pyproject.toml                     ← Project config, dependencies, tooling
├── Dockerfile                          ← Container image for local dev
├── .gitignore
├── docs/
│   ├── ADR-&-VISION.md                ← Level 2: architectural decisions (source of truth)
│   ├── RESEARCH-NOTES.txt             ← Research reference
│   |── MASTER-PLAN.md              ← This document (Level 3)
│   ├── architecture.md                ← Architecture narrative + diagrams
│   ├── api.md                         ← API reference
│   ├── deployment.md                  ← Infrastructure + deployment guide
│   ├── dataset.md                     ← Dataset engineering methodology
│   ├── experiments.md                 ← Training + evaluation guide
│   ├── evaluation.md                  ← F2P/P2P methodology
│   └── operations.md                  ← Monitoring, troubleshooting
├── infra/
│   └── terraform/
│       ├── main.tf                    ← Terraform root config
│       ├── variables.tf               ← Terraform input variables
│       ├── outputs.tf                 ← Terraform outputs
│       ├── providers.tf               ← Provider configuration
│       └── modules/
│           ├── storage/               ← GCS bucket + policies
│           ├── iam/                   ← IAM roles + WIF + secrets
│           └── deployment/            ← Deployment resources (Modal config, etc.)
├── data_engineering/                  ← WS-1: data pipeline
│   ├── __init__.py
│   ├── ingest.py
│   ├── validate.py
│   ├── clean.py
│   ├── split.py
│   ├── version.py
│   ├── archive.py
│   ├── golden.py
│   ├── card.py
│   └── schema.py
├── training/                          ← WS-2: fine-tuning pipeline
│   ├── __init__.py
│   ├── qlora_train.py
│   ├── model_config.py
│   ├── modal_train.py
│   ├── callbacks.py
│   ├── checkpoint.py
│   └── prompts.py
├── inference/                         ← WS-3: serving layer
│   ├── __init__.py
│   ├── serve.py
│   ├── modal_serve.py
│   ├── openai_compat.py
│   ├── telemetry.py
│   └── schema.py
├── evaluation/                        ← WS-4: evaluation harness
│   ├── __init__.py
│   ├── config.py
│   ├── schema.py
│   ├── patch_applier.py
│   ├── test_runner.py
│   ├── metrics.py
│   ├── harness.py                  ← F2P engine (includes runners, resume, W&B logging)
│   ├── prompt_ab_test.py
│   ├── inference.py                ← batch inference for patch gen
│   ├── comparison.py               ← re-validate P4 proxy champion
│   └── cli.py                      ← Typer CLI entrypoint
├── promotion/                         ← WS-5: Champion/Challenger
│   ├── __init__.py
│   ├── gate.py
│   ├── champion.py
│   ├── deploy.py
│   └── rules.py
├── observability/                     ← WS-7: telemetry
│   ├── __init__.py
│   ├── logging.py
│   ├── metrics.py
│   ├── dashboards.py
│   ├── cost.py
│   └── langfuse.py
├── models/
│   └── checkpoints/                   ← LoRA adapters (gitignored)
├── repos/
│   ├── manifest.json                  ← Curated repo manifest
│   ├── README.md                      ← Selection rationale
│   └── verify_repos.py                ← Verification script
├── .github/workflows/
│   ├── ci.yml                         ← CI: lint, test, eval gate
│   ├── cd.yml                         ← CD: deploy, promote
│   └── eval.yml                       ← Eval workflow (if separate)
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_training.py
    ├── test_evaluation.py
    ├── test_inference.py
    ├── test_promotion.py
    ├── test_infra.py
    ├── test_resilience.py
    └── fixtures/
        ├── sample_issues.json
        ├── sample_patches.json
        └── golden_sample.json
```

---

*End of Master Plan — v1.0*

*This document is the authoritative implementation blueprint for the SWE-Qwen LLM Fine-Tuning Platform. All subsequent Phase Plans should be decompositions of the relevant sections in this document. No new architectural decisions should be made without updating the ADR & Vision document and cross-referencing this plan.*
