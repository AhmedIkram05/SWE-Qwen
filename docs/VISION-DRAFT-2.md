# LLMOps Fine-Tuning Project – Architecture & Planning Decisions (Pre-Implementation)

## Purpose

This document captures the architectural decisions that must be agreed before implementation begins. It intentionally avoids implementation-specific details (hyperparameters, framework configuration, infrastructure minutiae, etc.) so the project remains adaptable as the LLM ecosystem evolves.

The goal of this document is to define **what is being built and why**, not **how each component will be implemented**.

---

# 1. Project Vision

Build a production-style **LLMOps platform** that fine-tunes an open-weight coding language model for software bug resolution, evaluates it using execution-based benchmarks inspired by SWE-bench, and deploys it using modern MLOps infrastructure and CI/CD.

The emphasis of the project is **LLMOps engineering**, not simply model fine-tuning.

The finished project should demonstrate the complete lifecycle of a modern generative AI system:

* Dataset engineering
* Parameter-efficient fine-tuning
* Experiment tracking
* Reproducible evaluation
* Model registry
* CI/CD
* Infrastructure as Code
* Cloud deployment
* Continuous model validation

---

# 2. Primary Objectives

The project should demonstrate the ability to:

* Build reproducible LLM training pipelines
* Fine-tune modern open-weight language models efficiently
* Design rigorous execution-based evaluation frameworks
* Build production-style MLOps infrastructure
* Deploy and manage generative AI models in AWS
* Automate deployment using Infrastructure as Code
* Build CI/CD pipelines around model quality rather than only software quality

The project should be positioned as an **LLMOps Engineering project**, not simply a machine learning project.

---

# 3. Success Criteria

The project is considered successful if it demonstrates:

* An end-to-end reproducible LLMOps pipeline
* Measurable improvement over the baseline model
* Execution-based evaluation rather than text-only evaluation
* Automated infrastructure provisioning
* Automated deployment pipeline
* Experiment reproducibility
* Production-quality documentation

---

# 4. Project Scope

## Core Problem

Improve a code-focused open-weight language model's ability to resolve real software issues by fine-tuning it on issue-resolution examples and evaluating its ability to produce working fixes.

---

## Model Task

The model will learn:

> Issue description + execution feedback → code patch

rather than

> Issue description → code patch

Including execution feedback better reflects modern AI engineering workflows while remaining achievable without building a fully autonomous coding agent.

---

## Explicitly Out of Scope

The project is **not** intended to build:

* a general-purpose coding assistant
* a chatbot
* an autonomous coding agent
* a multi-agent framework
* reinforcement learning systems
* RLHF
* DPO
* Kubernetes infrastructure

Keeping the scope focused allows greater engineering quality in the chosen domain.

---

# 5. High-Level System Architecture

```
Dataset Collection
        │
        ▼
Dataset Validation & Curation
        │
        ▼
QLoRA Fine-Tuning Pipeline
        │
        ▼
Experiment Tracking
        │
        ▼
Execution-Based Evaluation
        │
        ▼
Model Registry
        │
        ▼
AWS Deployment
        │
        ▼
CI/CD Quality Gates
```

The project should be viewed as an engineering platform rather than a collection of scripts.

---

# 6. Architecture Decision Records

---

# ADR-1 — Project Framing

## Decision

Build an end-to-end LLMOps platform.

## Rationale

The engineering surrounding modern language models has become significantly more valuable than demonstrating model training alone.

The project should communicate expertise in:

* reproducibility
* deployment
* experimentation
* infrastructure
* evaluation
* automation

rather than only fine-tuning.

---

# ADR-2 — Training Methodology

## Decision

Use Parameter-Efficient Fine-Tuning (QLoRA).

## Rationale

QLoRA is the current industry standard for adapting open-weight language models efficiently.

It demonstrates practical engineering skills while remaining achievable on freely available GPU resources.

---

# ADR-3 — Dataset Strategy

## Decision

Adopt a hybrid dataset strategy.

Training data should consist of carefully curated issue-resolution examples collected from suitable open-source repositories.

Evaluation should use a curated benchmark inspired by SWE-bench principles, with an official SWE-bench Verified subset used where practical and licensing permits.

## Why

This provides:

* realistic training data
* trustworthy evaluation
* reproducibility
* benchmark credibility

while avoiding dependence on a single dataset.

---

# ADR-4 — Repository Selection

## Decision

Repositories will be selected later using predefined criteria rather than fixed names.

Selection criteria:

* permissive licensing
* active maintenance
* clear issue → merged PR linkage
* manageable dependency graph
* fast automated test suites
* strong test coverage
* suitable repository size

The exact repositories remain an implementation decision.

---

# ADR-5 — Model Selection

## Decision

The base model will not be selected until baseline benchmarking has been completed.

Selection criteria:

* open-weight
* instruction tuned
* code capable
* actively maintained
* PEFT compatible
* suitable for QLoRA
* achievable on free GPU compute
* strong inference performance
* permissive licensing

The ecosystem evolves rapidly, therefore locking a specific model during planning would unnecessarily reduce future flexibility.

---

# ADR-6 — Evaluation Methodology

## Decision

Execution-based evaluation will be the primary evaluation strategy.

Primary metric:

* Fail-to-Pass (F2P) Resolution Rate

Secondary metrics:

* ROUGE
* BERTScore
* latency
* throughput
* inference cost

Execution-based correctness is significantly more meaningful than text similarity for software engineering tasks.

---

# ADR-7 — Experiment Tracking

## Decision

Weights & Biases will be used.

The platform should manage:

* experiment tracking
* datasets
* prompt versioning
* artifacts
* model registry
* model comparison

Hosted experiment tracking reduces infrastructure complexity while aligning with current LLM engineering workflows.

---

# ADR-8 — Infrastructure

## Decision

Infrastructure should remain intentionally lightweight.

Core services:

* AWS
* Terraform
* GitHub Actions
* OIDC authentication
* S3
* SageMaker

Infrastructure should demonstrate modern deployment practices without introducing unnecessary operational complexity.

---

# ADR-9 — Deployment

## Decision

Deploy an inference API rather than a user-facing application.

The deployment objective is to demonstrate production model serving, not frontend development.

Deployment implementation details (container technology, inference runtime, quantization format, endpoint configuration) will remain implementation decisions until benchmarking is complete.

---

# ADR-10 — CI/CD

## Decision

CI/CD should validate both software quality and model quality.

Pipelines should include:

* infrastructure validation
* automated testing
* evaluation execution
* deployment gates
* model promotion

The deployment pipeline should treat model quality as a first-class deployment requirement.

---

# 7. Deferred Decisions

The following decisions intentionally remain open until implementation.

## Model

* exact base model
* model version
* context length
* tokenizer

---

## Fine-Tuning

* LoRA rank
* alpha
* learning rate
* epochs
* optimizer
* scheduler
* sequence length
* batch size
* gradient accumulation
* checkpoint frequency

---

## Dataset

* exact repositories
* final dataset size
* filtering heuristics
* prompt templates
* dataset schema

---

## Evaluation

* performance targets
* latency budgets
* acceptance thresholds
* benchmark split sizes

Targets should be established from measured baselines rather than arbitrary values.

---

## Deployment

* inference runtime
* quantization format
* container framework
* endpoint configuration
* CPU vs GPU serving strategy
* autoscaling configuration

These should be determined after model benchmarking.

---

# 8. Guiding Principles

The project should prioritise:

1. Engineering quality over model size.
2. Reproducibility over experimentation speed.
3. Execution-based evaluation over text-based evaluation.
4. Simplicity over unnecessary architectural complexity.
5. Production engineering practices over research novelty.
6. Adaptability to future model releases.
7. Clear separation between planning decisions and implementation decisions.

---

# 9. Deliverables

The completed project should produce:

* End-to-end LLMOps pipeline
* Reproducible dataset pipeline
* Fine-tuned open-weight model
* Automated evaluation framework
* Execution-based benchmark harness
* Experiment tracking platform
* Model registry
* Terraform infrastructure
* AWS deployment
* GitHub Actions CI/CD pipeline
* Comprehensive documentation
* Architecture documentation
* Deployment documentation
* Results analysis
* Public portfolio repository

---

# 10. Positioning

This project should be positioned as a demonstration of modern **LLMOps Engineering** rather than simply fine-tuning a language model.

The strongest signals it should communicate are:

* production AI engineering
* reproducible experimentation
* infrastructure automation
* execution-based evaluation
* cloud deployment
* software engineering discipline
* modern MLOps practices

Model choice is intentionally treated as an implementation detail. The enduring value of the project comes from the engineering architecture and reproducible LLMOps workflow rather than any specific foundation model.
