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
- **Rationale:** Open-weight foundation models and GPU hosting providers evolve rapidly. Decoupling storage (AWS/GCP) from GPU inference (vLLM / Serverless) ensures the system remains flexible and resilient to vendor changes.

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

- **Decision:** Provision all cloud storage (S3/GCS), IAM policies, OIDC keyless authentication, and environment secrets using **Terraform**. Manual cloud console creation is prohibited.
- **Rationale:** Ensures complete infrastructure reproducibility, version-controlled cloud configuration, and secure keyless authentication via GitHub Actions.

---



### ADR-009 — CI/CD & Model Quality Gates

- **Decision:** CI/CD pipelines (via **GitHub Actions**) must validate both software integration and model performance before triggering deployment.
- **Rationale:** In LLMOps, passing unit tests for python code is necessary but insufficient; candidate models must pass execution-based benchmark quality gates before promotion.

---



### ADR-010 — Model Deployment Architecture & Serving Stack

- **Decision:** Deploy candidate models as a serverless, high-throughput **Inference API** utilizing **vLLM** hosted on a serverless GPU platform (e.g., **Modal** or **Baseten**).
- **Rationale:** `vLLM` provides industry-standard PagedAttention and high token throughput. Serverless GPU orchestration provides automatic scale-to-zero capabilities ($0 cost when idle), eliminates AWS GPU quota approval friction, and demonstrates modern AI engineering architecture.

---



### ADR-011 — Observability & Telemetry

- **Decision:** Every stage of the system must emit structured, measurable outputs (training loss, evaluation F2P rate, API serving latency, Time-To-First-Token, cost per inference, deployment status).
- **Rationale:** Production operations require full visibility into both model drift/performance and platform health.

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
       [ Serverless GPU Deployment ] (vLLM / Modal / Baseten)
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