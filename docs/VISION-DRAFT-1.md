# Architecture Decision Record (ADR)

## Project Foundation & Architectural Decisions

**Status:** Accepted
**Purpose:** This document defines the architectural principles, project vision, and long-term decisions that govern the implementation of the AI Engineering / LLMOps project. It serves as the authoritative source for all future planning. The Master Plan, implementation phases, and technical specifications must conform to the decisions defined here.

---

# 1. Project Vision

Build a production-grade AI Engineering / LLMOps platform that demonstrates the complete lifecycle of developing, evaluating, deploying, and maintaining a fine-tuned open-weight Large Language Model for automated software issue resolution.

The emphasis of the project is not on producing the highest-scoring model, but on demonstrating production-quality engineering practices across the entire LLM lifecycle.

The finished project should represent the type of system that could realistically exist within an AI Engineering organisation rather than an academic machine learning experiment.

---

# 2. Primary Objectives

The project aims to demonstrate competency across the following disciplines:

* AI Engineering
* LLMOps
* Machine Learning Operations
* Cloud Infrastructure
* Infrastructure as Code
* CI/CD for ML Systems
* Data Engineering
* Software Engineering
* Production Deployment
* Automated Evaluation
* Experiment Tracking
* Model Lifecycle Management

---

# 3. Success Criteria

Success is measured by the engineering quality of the platform rather than achieving state-of-the-art benchmark performance.

The completed system should demonstrate:

* Fully automated training pipeline
* Reproducible experiments
* Automated dataset generation and validation
* Execution-based evaluation
* Automated model promotion
* Cloud deployment
* Infrastructure as Code
* CI/CD integration
* Comprehensive monitoring
* Complete reproducibility

---

# 4. Project Scope

## In Scope

* Dataset generation pipeline
* Dataset validation
* Dataset versioning
* Fine-tuning pipeline
* Experiment tracking
* Model registry
* Automated evaluation
* Model deployment
* CI/CD
* Infrastructure as Code
* Cloud infrastructure
* Monitoring
* Documentation

## Out of Scope

The project is **not** intended to:

* Train frontier foundation models
* Compete with commercial coding agents
* Build a general-purpose coding assistant
* Conduct novel ML research
* Optimise for benchmark leadership

The objective is engineering excellence rather than research novelty.

---

# 5. Core Engineering Principles

Every architectural decision should maximise:

* Reproducibility
* Automation
* Maintainability
* Observability
* Scalability
* Modularity
* Reliability
* Engineering quality

The project should prioritise production engineering over benchmark optimisation.

---

# 6. Architectural Decision Records

---

## ADR-001 — Project Type

### Decision

The project is an **AI Engineering / LLMOps platform**, not a machine learning research project.

### Rationale

Employers primarily assess engineering capability rather than research contributions for entry-level AI Engineering roles.

---

## ADR-002 — Platform Philosophy

### Decision

The platform is the product.

The underlying language model is replaceable.

### Rationale

The architecture should remain useful as open-weight models evolve.

---

## ADR-003 — Domain Selection

### Decision

The project domain will be automated software issue resolution using GitHub issue reports and accepted code fixes.

### Rationale

This provides:

* objective evaluation
* real-world data
* execution-based verification
* minimal overlap with existing portfolio projects
* strong interview discussion opportunities

---

## ADR-004 — Evaluation Philosophy

### Decision

Execution-based evaluation is the primary measure of model quality.

Evaluation priority:

1. Functional correctness
2. Regression safety
3. Latency
4. Throughput
5. Text similarity metrics

### Rationale

Production systems care about working software rather than textual similarity.

---

## ADR-005 — Dataset Philosophy

### Decision

Datasets are treated as production assets.

The project will implement a complete dataset lifecycle.

Dataset lifecycle:

* Data ingestion
* Validation
* Cleaning
* Filtering
* Versioning
* Training split generation
* Golden evaluation set generation
* Archiving

### Rationale

Dataset engineering is a core competency within AI Engineering.

---

## ADR-006 — Experiment Management

### Decision

Every experiment must be fully reproducible.

Experiments must version:

* datasets
* prompts
* configurations
* hyperparameters
* metrics
* checkpoints
* trained models
* evaluation outputs

---

## ADR-007 — Model Lifecycle

### Decision

The platform will support a complete model lifecycle.

Lifecycle:

Training

↓

Evaluation

↓

Quality Gates

↓

Model Registry

↓

Deployment

↓

Monitoring

↓

Future Retraining

---

## ADR-008 — Champion / Challenger Strategy

### Decision

Models are promoted through an automated Champion / Challenger workflow.

Deployment must never rely upon manual judgement alone.

---

## ADR-009 — Infrastructure Philosophy

### Decision

Use managed cloud services whenever they remove undifferentiated operational complexity.

Infrastructure should focus engineering effort on the AI pipeline rather than infrastructure maintenance.

---

## ADR-010 — Infrastructure as Code

### Decision

All cloud infrastructure will be provisioned through Infrastructure as Code.

Manual infrastructure creation is prohibited.

---

## ADR-011 — CI/CD Philosophy

### Decision

Every production deployment must be automated.

The deployment pipeline should enforce quality gates before promotion.

High-level workflow:

Training

↓

Evaluation

↓

Quality Gates

↓

Deployment

---

## ADR-012 — Cloud Philosophy

### Decision

The architecture should remain cloud-portable where practical.

Cloud providers are implementation details rather than architectural dependencies.

---

## ADR-013 — Model Independence

### Decision

The platform must remain model-agnostic.

Changing the foundation model should require minimal architectural change.

Model selection should not influence the overall system design.

---

## ADR-014 — Cost Philosophy

### Decision

The project should maximise engineering quality while remaining executable using free-tier or low-cost infrastructure whenever practical.

This prevents unnecessary infrastructure complexity and demonstrates cost-conscious engineering.

---

## ADR-015 — Observability

### Decision

Every stage of the pipeline should expose measurable outputs.

This includes:

* training metrics
* evaluation metrics
* latency
* throughput
* deployment status
* experiment history
* model lineage

---

# 7. High-Level System Architecture

The platform should be viewed as a complete engineering pipeline.

```
                 GitHub

                    │

          Dataset Ingestion

                    │

          Quality Validation

                    │

        Dataset Versioning

                    │

          Training Dataset

                    │

             Fine-Tuning

                    │

       Experiment Tracking

                    │

      Execution Evaluation

                    │

      Champion Selection

                    │

      Automated Deployment

                    │

      Production Endpoint

                    │

          Monitoring

                    │

       Future Retraining
```

The fine-tuning process is only one component within the wider platform.

---

# 8. Deferred Implementation Decisions

The following decisions are intentionally postponed until implementation planning.

These are implementation details rather than architectural decisions.

## Foundation Model

Selection criteria:

* Open-weight
* Strong coding capability
* PEFT compatible
* Efficient fine-tuning
* Suitable for free-tier GPU resources
* Actively maintained
* Strong community ecosystem

---

## Target Repositories

Selection criteria:

* Permissive licence
* Active maintenance
* High-quality issue history
* Linked issue-to-PR workflow
* Fast automated tests
* Python ecosystem
* Manageable dependency graph

---

## Dataset Size

Will be determined through experimentation and available compute resources.

Dataset quality takes precedence over dataset volume.

---

## Fine-Tuning Configuration

To be selected during implementation:

* LoRA configuration
* QLoRA configuration
* Learning rate
* Batch size
* Epoch count
* Context length
* Optimiser
* Scheduler

---

## Quantisation Strategy

Will be selected after benchmarking deployment requirements.

---

## Deployment Stack

The serving infrastructure will be chosen after evaluating:

* latency
* cost
* maintainability
* operational complexity
* cloud compatibility

---

## Experiment Tracking Platform

Tool selection remains flexible provided it supports:

* experiment tracking
* model registry
* artifact versioning
* reproducibility

---

## Cloud Services

Specific services will be selected during implementation planning based upon architectural requirements rather than vendor preference.

---

# 9. Open Questions

The following questions remain unresolved and will be answered during Master Plan creation:

* Which open benchmark or dataset should form the basis of training?
* Should the project incorporate agentic inference alongside fine-tuning?
* What deployment strategy best balances cost and realism?
* What evaluation thresholds should be required for automatic promotion?
* What monitoring strategy should be implemented for production endpoints?
* How should retraining and future model updates be orchestrated?

---

# 10. Master Plan Constraints

The implementation plan must satisfy the following requirements:

* Infrastructure as Code only
* Fully reproducible experiments
* Automated CI/CD
* Automated deployment
* Automated evaluation
* Execution-based quality gates
* Versioned datasets
* Versioned models
* Model registry
* Cloud deployment
* Monitoring
* Production-quality documentation

No implementation decision should violate the architectural principles defined within this document.

---

# 11. Future Planning Documents

This ADR forms the foundation for the remaining planning hierarchy.

```
Research Notes
        │
        ▼
Architecture Decision Record (This Document)
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

All future planning documents should derive from and remain consistent with the architectural decisions established here.
