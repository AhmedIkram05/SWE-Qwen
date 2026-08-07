# Project Vision & Architectural Foundations Document for SWE-Qwen LLM Fine Tuning Project

**Title:** Architectural Decision Record (ADR) & Platform Vision
**Purpose:** This document defines the authoritative project vision, architectural principles, and foundational decisions governing the implementation of the LLMOps / AI Engineering platform. All downstream implementation planning (Master Plan, Phase Plans, Technical Specifications, and Code) must conform to the decisions defined here. The goal of this document is to define what is being built and why, not how each component will be implemented.

---

# 1. Project Vision & Positioning

Build a production-grade **LLMOps and AI Engineering platform** that demonstrates the complete lifecycle of developing, evaluating, deploying, and maintaining a fine-tuned open-weight Large Language Model for automated software issue resolution.

The emphasis of the project is on **LLMOps engineering quality, platform architecture, high-throughput inference serving, and end-to-end reproducibility**, rather than chasing state-of-the-art model benchmark scores.

The finished project represents a modern platform that mirrors cutting-edge AI Engineering practices found in modern AI startups and high-performing enterprise teams.

### Core Positioning Strategy

- **Platform over Model:** The platform is the primary product; the underlying foundation model is intentionally treated as a replaceable component.
- **Modern LLMOps Focus:** Communicates competency in modern LLM infrastructure (`vLLM`, serverless GPU orchestration, execution-based benchmarks, IaC, and automated model quality gates) rather than legacy ML pipelines or basic cloud VMs.

---

# 2. Primary Objectives & Key Competencies

The project demonstrates production competency across the following engineering disciplines:

- **AI Engineering & LLMOps:** Fine-tuning, prompt management, model registry, high-performance inference serving (`vLLM`), and automated quality gates.
- **Data Engineering:** Automated data ingestion, validation, cleaning, versioning, and golden evaluation set curation.
- **Automated & Execution-Based Evaluation:** Real-world benchmark execution and test-suite verification harnesses.
- **Cloud Infrastructure & IaC:** Automated provisioning using Terraform for cloud storage, IAM, and secrets management without manual configuration.
- **Serverless GPU Orchestration:** High-throughput, scale-to-zero model serving to maximize cost efficiency and eliminate infrastructure cold-start friction.
- **CI/CD for ML Systems:** Continuous Integration/Deployment pipelines incorporating automated model quality gates and OIDC keyless authentication.
- **Software Engineering Discipline:** Modular design, testing, observability, and reproducible environments.

---

# 3. Project Scope & Model Task

## Model Task Formulation

Rather than a simple text-to-text generation task, the model will learn:

Issue Description + Execution Feedback --> Code Patch

Including execution feedback reflects modern AI engineering workflows and produces significantly higher fix accuracy without introducing the operational complexity of full multi-agent frameworks.

## In Scope

- End-to-end dataset generation, validation, and versioning pipeline
- QLoRA parameter-efficient fine-tuning pipeline
- Full experiment tracking, artifact lineage, and model registry integration (Weights & Biases)
- Execution-based evaluation harness (testing generated code against real test suites using Fail-to-Pass metrics)
- Automated Champion / Challenger promotion pipeline
- High-performance production model deployment as a serverless `vLLM` inference API
- Infrastructure as Code (Terraform) and CI/CD quality gates (GitHub Actions)
- Real-time monitoring and observability for training and inference latency
- Production-grade technical documentation

## Out of Scope

The project will **not**:

- Train foundation models from scratch
- Build user-facing frontends, chatbots, or general-purpose coding assistants
- Implement autonomous multi-agent frameworks, RLHF, DPO, or complex reinforcement learning
- Deploy heavy, persistent 24/7 legacy GPU endpoints (e.g. unoptimized SageMaker endpoints) that incur idle costs
- Compete with commercial coding agents or optimize for benchmark leadership at the expense of engineering rigor

---

# 4. Core Engineering Principles

Every architectural decision must maximize:

1. **Reproducibility:** Every dataset, prompt, hyperparameter set, and checkpoint must be fully traceable and reproducible.
2. **Automation:** Zero manual steps in infrastructure creation, model evaluation, and deployment promotion.
3. **Execution-Based Evaluation:** Functional correctness takes absolute precedence over textual similarity metrics.
4. **Modularity & Adaptability:** The platform must remain model-agnostic and cloud-portable.
5. **Cost-Conscious Engineering:** Utilize scale-to-zero serverless GPU architecture ($0 idle cost) and free-tier cloud storage/compute without sacrificing production realism.
6. **Observability:** Every stage of the pipeline must emit structured metrics and telemetry.

---

# 5. Consolidated Architecture Decision Records (ADRs)

---

### ADR-001 — Project Framing & Domain Focus

- **Decision:** The project is an **LLMOps Platform**, not an ML research experiment. The domain is automated software issue resolution using open-source GitHub issue reports, test suites, and code patches.
- **Rationale:** Industry demands robust engineering surrounding generative models (data pipelines, evaluation, serving, CI/CD) over basic fine-tuning capability. Automated bug fixing provides objective, execution-verifiable evaluation metrics.

---

### ADR-002 — Platform & Model Independence Philosophy

- **Decision:** The platform architecture must remain strictly model-agnostic and cloud-portable. Compute and serving layers must be fully decoupled from storage and state layers.
- **Rationale:** Open-weight foundation models and GPU hosting providers evolve rapidly. Decoupling storage (GCS) from GPU inference (vLLM / Serverless) ensures the system remains flexible and resilient to vendor changes.

---

### ADR-003 — Fine-Tuning Methodology (QLoRA)

- **Decision:** Adopt Parameter-Efficient Fine-Tuning using **QLoRA** (Quantized Low-Rank Adaptation).
- **Rationale:** QLoRA is the industry standard for cost-effective, high-performance LLM adaptation. It allows production-grade fine-tuning within accessible GPU compute limits.

---

### ADR-004 — Dataset Engineering & Strategy

- **Decision:** Datasets are treated as production software assets subject to a strict lifecycle:
`Ingestion` --> `Validation` --> `Cleaning` --> `Versioning` --> `Splitting` --> `Archiving`.
The project will use a **hybrid strategy**: training data curated from real GitHub issue-PR pairs, with evaluation grounded in SWE-bench principles (including the official SWE-bench Verified subset where appropriate).
- **Rationale:** Data quality directly dictates model utility. Establishing a versioned, schema-validated data pipeline guarantees reproducibility.

---

### ADR-005 — Evaluation Philosophy & Primary Metrics

- **Decision:** Execution-based functional evaluation is the primary measure of model quality.
- **Priority Order:**
  1. **Fail-to-Pass (F2P) Resolution Rate** (Primary Metric)
  2. Regression Safety (Pass-to-Pass rate)
  3. Latency & Throughput (Time-to-First-Token & Tokens/Sec)
  4. Token Efficiency / Inference Cost
  5. Text Similarity Metrics (ROUGE / BERTScore — monitored as secondary signals only)
- **Rationale:** Production software engineering requires working code that passes test suites, making text-matching metrics insufficient for evaluating code correctness.

---

### ADR-006 — Experiment Tracking & Artifact Management

- **Decision:** Standardize on **Weights & Biases (W&B)** for experiment tracking, prompt versioning, artifact storage, and model registry management.
- **Rationale:** A central, managed tracking platform enforces 100% reproducibility across datasets, configurations, evaluation outputs, and model lineage without adding self-hosted infrastructure overhead.

---

### ADR-007 — Champion / Challenger Model Promotion

- **Decision:** Model promotion into production will be governed by an automated **Champion / Challenger** pipeline. Manual deployment decisions are prohibited.
- **Rationale:** Automated quality gates prevent regression and mirror enterprise continuous delivery practices for ML models.

---

### ADR-008 — Infrastructure as Code (IaC) & Cloud Foundation

- **Decision:** Provision all cloud storage (GCS), IAM policies, OIDC keyless authentication, and environment secrets using **Terraform**. Manual cloud console creation is prohibited.
- **Rationale:** Ensures complete infrastructure reproducibility, version-controlled cloud configuration, and secure keyless authentication via GitHub Actions.

---

### ADR-009 — CI/CD & Model Quality Gates

- **Decision:** CI/CD pipelines (via **GitHub Actions**) must validate both software integration and model performance before triggering deployment.
- **Rationale:** In LLMOps, passing unit tests for python code is necessary but insufficient; candidate models must pass execution-based benchmark quality gates before promotion.

---

### ADR-010 — Model Deployment Architecture & Serving Stack

- **Decision:** Deploy candidate models as a serverless, high-throughput **Inference API** utilizing **vLLM** hosted on a serverless GPU platform, **Modal**.
- **Rationale:** `vLLM` provides industry-standard PagedAttention and high token throughput. Serverless GPU orchestration provides automatic scale-to-zero capabilities ($0 cost when idle), eliminates GCP GPU quota approval friction, and demonstrates modern AI engineering architecture.

---

### ADR-011 — Observability & Telemetry

- **Decision:** Every stage of the system must emit structured, measurable outputs (training loss, evaluation F2P rate, API serving latency, Time-To-First-Token, cost per inference, deployment status).
- **Rationale:** Production operations require full visibility into both model drift/performance and platform health.

---

### ADR-012 — Vertical Slice Delivery

- **Decision:** The platform will be implemented as a sequence of independently functional vertical slices. Each phase must produce a working, testable increment rather than partially completing multiple workstreams.

- **Rationale:** This reduces integration risk, enables continuous validation, and ensures the repository remains deployable throughout development.

---

### ADR-013 — CD-Owned Smoke Baseline

- **Decision:** The smoke-eval baseline is a CI artifact owned by the deploy branch (main): PR runs read it read-only, and only a push to main may write it (one write per `dataset_run_id`, monotonic `max(new, prev, floor)`), stored at `gs://swe-qwen-datasets/ci/smoke_baseline.json`.
- **Rationale:** PRs competing for the baseline caused last-write-wins churn; making it CD-owned and run-id-keyed keeps the regression comparison stable and auditable for the dataset it was calibrated on.

---

### ADR-014 — Absolute Quality Floor in CI Gate

- **Decision:** The CI smoke gate fails on `f2p < min_f2p_threshold` (default 0.15) regardless of the stored baseline, in addition to the relative regression check.
- **Rationale:** A model sinking below an absolute minimum is unacceptable even if no baseline regression is measured; Phase 9 refines thresholding for candidate promotion, this floor guards the champion in the meantime.

---

### ADR-015 — Langfuse as V1 Trace Store (Eval + Sampled Serving, Cloud-Hosted)

- **Decision:** Langfuse (Cloud, hobby tier) is the V1 trace store alongside W&B: every evaluation run traces each golden example as prompt → completion → f2p/p2p scores, and the serving path traces only successful requests at `sample_rate=0.1`, drained asynchronously by the telemetry flush thread — never on the request hot path. Missing `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` makes the module a silent no-op so CI and local dev stay hermetic. OpenTelemetry (v2) coexists rather than replaces this (master plan Monitoring v2: W&B + Langfuse + OTel).
- **Rationale:** Dual observability (W&B aggregates, Langfuse per-call traces) gives prompt-versioning, trace debugging, and eval comparisons with bounded trace volume and zero serving latency risk; Cloud hosting matches the V1 managed-platform pattern (W&B, Modal) and avoids owning Postgres/Redis ops.

---

### ADR-016 — Cost Tracking Is Estimate-First

- **Decision:** Cost is estimated as `gpu_seconds / 3600 × rate_per_hour`, with `rate_per_hour` logged alongside `cost/cost_usd` and `cost/gpu_seconds` so every number is auditable; the Modal usage API is a stretch goal, not a DoD requirement. "Cost per F2P point" is defined as **cost per successful fix**: eval cost ÷ number of F2P-passing golden examples (`cost/cost_per_fix`).
- **Rationale:** Duration × rate is dependency-free, reproducible from run metadata, and accurate to the master plan's cost model (~$0.50–2.00/hr Modal pay-per-use); the real-spend API is thin-documented and would put the phase at risk for marginal accuracy gain. Per-fix semantics were chosen because it is the intuitive, communicable dollar cost of a working fix.

---

### ADR-017 — Metric Registry as the Telemetry Contract

- **Decision:** `observability/metrics.py` holds the single registry of allowed `{domain}/{metric}` keys (max 5–7 per domain), enforced by a CI test that fails any unregistered `wandb.log` key. Dashboards are generated **as code** via the official `wandb-workspaces` library (Reports + Workspaces API, `wandb_workspaces.reports.v2` + `wandb_workspaces.workspaces`) from the versioned PANELS spec in `observability/dashboards.py`, after a seed run emits every registered key so panels render real-shaped data; the manual UI build from the same spec is the documented fallback (the API is Public Preview, not a DoD dependency). Dashboard URLs are documented in `docs/observability/dashboards.md`.
- **Rationale:** Dashboards depend on key names, so an ad-hoc key regime means silent dashboard drift; the registry + CI test makes the telemetry contract versioned and reviewable, the seed run guarantees every panel renders real-shaped data, and the as-code generation (verified working via `wandb-workspaces` 0.4.4, 2026-08-07 — this ADR's earlier "no dashboard-as-code API" claim is retracted) keeps the dashboards themselves versioned in the repo. The preview-status API is isolated behind one script so the spec, not the API, is the source of truth.

---

### ADR-018 — V1 Reliability Telemetry: SLO Layer + Deploy Status

- **Decision:** The phase operationalizes two things the vision already promised but the task list didn't cover: (a) an SLO layer (`observability/slo.py`) deriving attainment and error-budget burn from the existing serving `MetricsCollector` against the master plan's success criteria S3 (TTFB p50 < 500ms) and S9 (cold start < 10s), alerting via `wandb.alert` (WARN ≥ 1× budget, ERROR ≥ 5×, min 10 samples per window to suppress low-traffic noise); (b) deploy-status telemetry (`scripts/log_deploy.py`, a `if: always()` step in `cd.yml`) emitting `deploy/status` and `deploy/duration_s` using the WANDB_API_KEY already provisioned in CI, feeding the infrastructure dashboard. Neither adds new metric keys beyond `deploy/*` — the SLO layer is derived from registered `serve/*` keys.
- **Rationale:** ADR-011 explicitly requires "deployment status" among the stage outputs and the master plan already defines S3/S9 targets, so leaving both unobserved would be a gap between the vision and the delivered telemetry. Deriving SLOs from the existing collector (rather than a new pipeline) respects the clutter guard and the no-new-deps rule; the CI-owned deploy step makes the code-push → gate → deploy → observe loop visible end to end. OTel v2 replaces the heuristic burn model with proper SLO tooling.

---

# 6. Target High-Level Platform Architecture

```
                      [ GitHub Source Data ]
                                │
                                ▼
                    [ Data Ingestion Engine ]
                                │
                                ▼
                   [ Data Validation & Schema ]
                                │
                                ▼
                 [ Dataset Versioning (W&B/Cloud) ]
                                │
                                ▼
                  [ QLoRA Training Pipeline ] ──┐
                                │               │
                                ▼               │
                   [ Experiment Tracking ] ◄────┤ (Metadata & Artifacts)
                                │               │
                                ▼               │
                 [ Execution Evaluation Harness ]
                                │
                                ▼
                  [ Automated Quality Gate ]
                                │
                  (Pass) ───────┴─────── (Fail)
                    │                      │
                    ▼                      ▼
            [ Model Registry ]       [ Reject Candidate ]
                    │
                    ▼
          [ Terraform / Cloud IaC ] ──► (Storage, Secrets, IAM)
                    │
                    ▼
       [ Serverless GPU Deployment ] (vLLM / Modal)
                    │
                    ▼
       [ Scale-to-Zero Inference API ]
                    │
                    ▼
        [ Telemetry & Observability ]
```

---

# 7. Deferred Implementation Decisions

To maintain architectural adaptability, specific execution details are intentionally postponed until the benchmarking phase:

- **Foundation Model:** Selection criteria established (open-weight, instruction-tuned, strong coding baseline, QLoRA compatible, permissive license). Specific selection deferred until baseline evaluations are complete.
- **Target Repositories:** Selection criteria established (Python ecosystem, fast automated test suites, permissive licenses, clear issue-to-PR linkage). Specific repository list deferred.
- **Fine-Tuning Hyperparameters:** Specific rank, alpha, learning rate, batch size, context window length, and scheduler deferred to experimental execution.
- **Inference Runtime Engine Config:** Exact `vLLM` container parameters (tensor parallelism, GPU memory utilization fraction, max num seqs) deferred to serving performance benchmarks.

---

# 8. Project Deliverables

Upon completion, the project will yield:

1. **Reproducible Data Pipeline:** Python-based ingestion, validation, and transformation pipeline with dataset versioning.
2. **Fine-Tuning Engine:** Modular QLoRA training pipeline supporting multi-GPU execution with full tracking.
3. **Execution-Based Evaluation Harness:** Containerized benchmark runner computing Fail-to-Pass (F2P) resolution metrics.
4. **Model Registry & Tracking System:** Full W&B integration showing artifact lineage from raw data to model artifacts.
5. **Terraform Infrastructure:** Production IaC scripts provisioning cloud storage, IAM policies, and deployment credentials.
6. **Automated CI/CD Workflows:** GitHub Actions pipelines running code linting, tests, evaluation verification, and automated cloud deployment via OIDC.
7. **High-Performance Serverless API Endpoint:** Scalable `vLLM` inference endpoint with scale-to-zero capabilities and telemetry logging.
8. **Technical Documentation:** Architecture notes, deployment guides, and experimental benchmark results analysis.

---

# 9. Planning Hierarchy & Next Steps

This document serves as **Level 2** in the project design hierarchy:

```
                  Research Notes
                        │
                        ▼
  Architecture Decision Record / Vision (This Document)
                        │
                        ▼
          Master Implementation Plan
                        │
                        ▼
             Detailed Phase Plans
                        │
                        ▼
             Technical Specifications
                        │
                        ▼
                   Implementation
```

The next document we have to create is the **Master Plan**, which should not repeat these decisions. Instead, it should translate them into a concrete implementation roadmap (workstreams, milestones, dependencies, deliverables, acceptance criteria, risks, and phase sequencing). That separation keeps the ADR stable while allowing the implementation plan to evolve as technologies and tooling change.
