# SWE-Qwen

> A production-grade, model-agnostic LLMOps platform that turns **20,477 SWE-bench software issues** into a **17,456-example training corpus**, fine-tunes **3 QLoRA variants of Qwen3-14B on Modal A100-80GB GPU(s)**, evaluates them with **execution-based fail-to-pass / pass-to-pass testing inside real SWE-bench Docker images** (50-instance CI gate, Wilson CIs, McNemar + paired-bootstrap significance), gates every promotion behind a **statistical champion/challenger flow**, and serves the winner through an **OpenAI-compatible, scale-to-zero inference API with per-request LoRA adapters** — all orchestrated by **Terraform IaC on Google Cloud**, tracked end-to-end in **Weights & Biases**, and gated by **4 GitHub Actions workflows**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&labelColor=000000&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&labelColor=000000&logo=pytorch" />
  <img src="https://img.shields.io/badge/QLoRA-Transformers-8B5CF6?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Unsloth-Fast_Training-FF6B6B?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/FastAPI-OpenAI_API-009688?style=for-the-badge&labelColor=000000&logo=fastapi" />
  <img src="https://img.shields.io/badge/Modal-Serverless-007FFF?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/vLLM-Serving-5A67D8?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Qwen3-14B-7C3AED?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Terraform-IaC-844EBA?style=for-the-badge&labelColor=000000&logo=terraform" />
  <img src="https://img.shields.io/badge/Google_Cloud-GCS-4285F4?style=for-the-badge&labelColor=000000&logo=googlecloud" />
  <img src="https://img.shields.io/badge/Weights_%26_Biases-Experiments-FFBE00?style=for-the-badge&labelColor=000000&logo=weightsandbiases" />
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&labelColor=000000&logo=githubactions" />
  <img src="https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?style=for-the-badge&labelColor=000000&logo=huggingface" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-2D9CDB?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/bitsandbytes-NF4_Quant-DC143C?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/TRL-SFT_Trainer-FF6F61?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/HF_Datasets-Arrow-6DA8FF?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Typer-CLI-FFA629?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Pydantic-Schemas-3B82F6?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Uvicorn-ASGI-4A4A55?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/SSE-Streaming-10B981?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/Docker-SWE--bench-2496ED?style=for-the-badge&labelColor=000000&logo=docker" />
  <img src="https://img.shields.io/badge/Langfuse-LLM_Tracing-BB8FF7?style=for-the-badge&labelColor=000000" />
  <img src="https://img.shields.io/badge/BigQuery-Analytics-669DF6?style=for-the-badge&labelColor=000000&logo=googlebigquery" />
  <img src="https://img.shields.io/badge/pytest-1%2C456_Tests-0A9EDC?style=for-the-badge&labelColor=000000&logo=pytest" />
</p>

<p align="center">
  <a href="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/ci.yml"><img src="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/cd.yml"><img src="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/cd.yml/badge.svg" /></a>
  <a href="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/eval.yml"><img src="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/eval.yml/badge.svg" /></a>
  <a href="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/promote.yml"><img src="https://github.com/AhmedIkram05/SWE-Qwen/actions/workflows/promote.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/AhmedIkram05/SWE-Qwen"><img src="https://img.shields.io/codecov/c/github/AhmedIkram05/SWE-Qwen" /></a>
</p>

<p align="center">
  <img src="assets/media/data-pipeline-volume.png" width="700" alt="One data-pipeline run · 20,477 issues in → tokenized out" />
  <br/>
  <em>One real data-pipeline run: 20,477 raw SWE-bench records → 17,456 cleaned → 15,011 / 1,556 / 889 train / val / test → 2,313 golden set bypassing training → 14,833 tokenized training examples.</em>
</p>

SWE-Qwen is an **automated software-issue-resolution platform**: it ingests real GitHub issues + gold patch PRs from SWE-bench, cleans them into a high-signal training corpus, fine-tunes open-weight LLMs with QLoRA, and measures success the only way that matters — **does the generated patch actually flip failing tests to passing inside the real repository?** Every layer is reproducible, observable, and CI-gated.

---

<details>
<summary><b>Table of Contents</b> (click to expand)</summary>

- [Architecture Overview](#architecture-overview)
  - [End-to-end flow](#end-to-end-flow)
- [Engineering Highlights](#engineering-highlights)
- [Key Metrics at a Glance](#key-metrics-at-a-glance)
- [Demos (the system, run)](#demos-the-system-run)
  - [Evaluation results](#evaluation-results)
  - [Training curves](#training-curves)
  - [Inference demo](#inference-demo)
  - [Data pipeline run](#data-pipeline-run)
  - [Every subsystem is one command](#every-subsystem-is-one-command)
- [Anatomy of a Real Run (the system was up)](#anatomy-of-a-real-run-the-system-was-up)
- [Component Deep Dives](#component-deep-dives)
  - [1. Data Engineering (`data_engineering/`)](#1-data-engineering-data_engineering)
  - [2. QLoRA Training (`training/`)](#2-qlora-training-training)
  - [3. Evaluation Harness (`evaluation/`)](#3-evaluation-harness-evaluation)
  - [4. Promotion & Registry (`promotion/`)](#4-promotion--registry-promotion)
  - [5. Inference Serving (`inference/`)](#5-inference-serving-inference)
  - [6. Observability (`observability/`)](#6-observability-observability)
- [Design Decisions](#design-decisions)
- [Testing Strategy](#testing-strategy)
- [Infrastructure (Terraform)](#infrastructure-terraform)
- [CI/CD Pipeline](#cicd-pipeline)
- [Security Model](#security-model)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
  - [Running Tests](#running-tests)
  - [Production Deployment](#production-deployment)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Related Projects](#related-projects)

</details>

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph DATA["DATA ENGINEERING · data_engineering/"]
        direction LR
        ING["Ingest<br/>20,477 raw instances<br/>402 SWE-bench repos"]
        VAL["Validate<br/>20,470 ✓ · 7 rejected"]
        CLEAN["Clean<br/>17,456 ✓ · 3,014 dropped"]
        SPLIT["Split by repo<br/>15,011 train · 1,556 val · 889 test<br/>46 repos (37/5/4)"]
        GOLD["Golden set<br/>2,313 held out<br/>never touches training"]
        TOK["Tokenize<br/>Qwen3-14B @ 8,192 ctx<br/>14,833 train examples"]
        ING --> VAL --> CLEAN --> SPLIT --> GOLD
        CLEAN --> TOK
    end
    subgraph TRAIN["QLORA TRAINING · training/ · Modal A100-80GB"]
        direction LR
        BASE["baseline_14b<br/>r16 · α32 · lr2e-5 · 2×8"]
        RANK["higher_rank_14b<br/>r32 · α64 · lr2e-5 · 1×16"]
        LR["higher_lr_14b<br/>r16 · α32 · lr5e-5 · 2×8"]
    end
    subgraph EVAL["EXECUTION-BASED EVALUATION · evaluation/"]
        direction LR
        SAMPLE["Sample 100 golden<br/>seed 42"]
        DOCKER["Materialize repo<br/>official SWE-bench image"]
        PATCH["Generate patch<br/>LoRA variant"]
        TEST["Run FAIL_TO_PASS + PASS_TO_PASS<br/>30 s/test · flaky retries"]
        STATS["Statistics<br/>Wilson CI · McNemar · paired bootstrap"]
        TEST --> STATS
    end
    subgraph SERVE["INFERENCE SERVING · inference/ · Modal scale-to-zero"]
        direction LR
        API["FastAPI<br/>POST /v1/chat/completions<br/>SSE streaming"]
        VLLM["vLLM + AWQ 4-bit<br/>per-request LoRA swap"]
    end
    OBS["OBSERVABILITY · observability/<br/>Weights & Biases · Langfuse (10% trace) · GCP Logging"]
    DATA --> TRAIN
    TRAIN --> SAMPLE
    TRAIN --> API
    EVAL --> PROMO1["Promotion gate<br/>F2P ≥ 15% · CI lower bound > 0<br/>champion vs challenger"]
    PROMO1 --> API
    API --> OBS
    EVAL --> OBS
    TRAIN --> OBS

    classDef src fill:#2a2a52,color:#fff,stroke:#1e1e4f
    classDef proc fill:#3b82f6,color:#fff,stroke:#1e40af
    classDef ml fill:#8b5cf6,color:#fff,stroke:#5b21b6
    classDef evl fill:#f59e0b,color:#1f2937,stroke:#b45309
    classDef srv fill:#10b981,color:#fff,stroke:#065f46
    classDef obs fill:#ef4444,color:#fff,stroke:#991b1b
    class ING,VAL,CLEAN,SPLIT,GOLD,TOK proc
    class BASE,RANK,LR ml
    class SAMPLE,DOCKER,PATCH,TEST,STATS evl
    class API,VLLM srv
    class OBS,PROMO1 obs
    class DATA,TRAIN,EVAL,SERVE src
```

### End-to-end flow

`github issues + PR patches → schema-validated IssueRecords → quality-cleaned corpus → repo-stratified splits → QLoRA fine-tuning (3 variants, A100-80GB) → golden-set execution-based evaluation → statistical compare → champion LoRA adapter → OpenAI-compatible serverless inference → per-request adapter inference → Langfuse + W&B telemetry`

Every stage is a first-class, independently runnable step with typed schemas (`pydantic`), deterministic seeds, artifact versioning (W&B + GCS), and CI gates (4 workflows).

---

## Engineering Highlights

| Area | Decision | Why |
| ---- | -------- | --- |
| Model-agnostic core | Qwen via `config/models.yaml` registry | Swap any open-weight model config without touching pipeline code; model identity travels as plain data |
| Storage decoupled | GCS `gs://swe-qwen-datasets` = single source of truth | Pipeline + training + eval all read the same artifacts; public-readable for leak-free Modal training |
| Training compute | Modal serverless A100-80GB (timeout 5h, retries) | No GPU fleet to babysit; spot-like cost, object-storage resume |
| Serving compute | Modal scale-to-zero + vLLM AWQ + FlashAttention | $0.00 when idle; per-request LoRA via `LoRARequest` swaps adapters without engine restarts |
| No Dockerfile | Modal Images (`debian-slim` + torch 2.11 + unsloth + flash-attn 2.8.3) | Images are code, build once per `uv.lock`, no docker daemon in CI |
| Efficient fine-tuning | QLoRA 4-bit NF4 via Unsloth (Xformers + FlashAttention 2.8.3) | ~3–4 h for a 14B variant on A100-80GB; 3 variants trained on one GPU budget |
| Secrets hygiene | GCP Workload Identity Federation (OIDC) | `iam.disableServiceAccountKeyCreation` blocks long-lived SA keys; GitHub OIDC → short-lived tokens only |
| Real evaluation | Execution-based F2P/P2P inside official SWE-bench Docker images | A passing unit test is not a fix; a patch that flips `FAIL_TO_PASS` and keeps `PASS_TO_PASS` is |
| Statistical promotions | Wilson CI + McNemar + paired bootstrap before a champion moves | Prevent advertising a "win" that is a coin flip on a 20–50 instance sample |
| Self-certification impossible | `eval.yml`: PRs read `smoke_baseline.json`, only `main` writes it | The model cannot approve its own regression |
| Secrets in-cloud | GCP Secret Manager + GitHub Secrets | No credentials in repo; `.env.example` documents shape only |
| Observability | W&B artifacts/runs + Langfuse (10% traced) + GCP Logging + wandb-workspaces dashboards-as-code | Every action auditable; dashboards versioned in repo |

---

## Key Metrics at a Glance

| Category | Metric | Value |
| -------- | ------ | ----- |
| **Dataset** | Raw SWE-bench instances ingested | **20,477** across 402 repos |
| | Records that passed schema validation | 20,470 (7 rejected) |
| | High-quality corpus after cleaning | **17,456** (3,014 dropped) |
| | Train / val / test split (repo-stratified) | **15,011 / 1,556 / 889** (46 repos · 37/5/4) |
| | Golden set (bypassed training) | **2,313** from verified + test + dev |
| | Tokenized training examples (Qwen3-14B, max 8,192) | 14,833 |
| | Cleaned-out noise | 12 binary, 1,212 non-Python, 1,149 oversized patches, 726 duplicates |
| **Training** | QLoRA variants fine-tuned (A100-80GB) | 3 (`baseline_14b`, `higher_rank_14b`, `higher_lr_14b`) |
| | Adapter checkpoints shipped to W&B | `model-qwen3-14b-{variant}` × 3 |
| | Train loss / runtime (higher_rank_14b, 1 epoch) | 0.5843 final · 4,214 s (~1.2 h) |
| **Evaluation** | Execution-based harness (real SWE-bench containers) | 2,313 golden pool · 100-sample released run |
| | Final F2P — champion (`higher_rank_14b`) vs base Qwen3-14B | **17.20%** vs 2.46% (**7.0×**, 95% CI 11.1–25.8%) |
| | Final P2P / avg latency (champion) | 90.10% · 8.92 s/instance |
| | Cost ceiling for a full compare | **$30** for 2 runs × 4 models (100 golden instances) |
| **Serving** | OpenAI-compatible endpoint | `POST /v1/chat/completions` |
| | Adapter switching | Per-request LoRA, zero engine restarts |
| | Idle cost | $0.00 (scale-to-zero) |
| **Quality** | Test suite (offline) | **1,456 passed · 1 skipped · 5 deselected** in ~3 min |
| | Lint / type-check | `ruff check` clean · `mypy` strict on 42 source files |
| | Infrastructure as code | 100% Terraform (storage + IAM + project roots) |

> **Final results** On the **100-instance golden set**, the promoted **`higher_rank_14b`** champion scores **17.20% F2P (95% Wilson CI 11.1–25.8%)** with **90.10% P2P** at **8.92 s/instance** — up from the base Qwen3-14B's **2.46% F2P / 28.54% P2P** (**7.0× F2P gain**, +61.6pt P2P, McNemar p < 1e-6, paired-bootstrap 95% CI lower bound > 0). Full table verbatim in [assets/results.txt](assets/results.txt); methodology in [docs/evaluation.md](docs/evaluation.md). The champion adapter ships on the Hugging Face Hub: **[`ahmedikram/SWE-Qwen-qwen3-14b-higher_rank_14b`](https://huggingface.co/ahmedikram/SWE-Qwen-qwen3-14b-higher_rank_14b)**.

---

## Demos (the system, run)

Everything below was captured against **live systems** — the real GCS bucket, the real Modal volumes, the real Weights & Biases artifacts, the real trained adapters, and a real running inference server. Headline media first; the supporting evidence is dispersed across the component deep-dives further down.

### Evaluation results

<p align="center">
  <img src="assets/media/eval-f2p-p2p.png" width="680" alt="Execution-based eval · F2P / P2P from assets/results.txt" />
  <br/><em>Source of truth: <code>assets/results.txt</code> — <code>evaluation.cli compare</code> output (100 golden instances per variant, est. cost $30).
</p>

### Training curves

<p align="center">
  <img src="assets/media/training-loss-curve.png" width="680" alt="Real loss curve from Modal training log" />
  <br/><em>Real training loss + learning rate extracted from `logs/modal-higher_rank_14b-20260806-030415.log` — QLoRA on Modal A100-80GB, the champion variant, final loss 0.5843 after 1 epoch.</em>
</p>

### Inference demo

<p align="center">
  <img src="assets/media/serve-demo.png" width="560" alt="Live uvicorn serve + /health + chat + error handling" />
  <br/><em>Live server: `/health` → `StubEngine`, real `chatcmpl-998dfe608954` completion, 401 without bearer token, `model_not_found` envelope. The first request per model pulled the 1.4 GB LoRA adapter from W&B into cache.</em>
</p>

### Data pipeline run

<p align="center">
  <img src="assets/media/cli-data-pipeline.png" width="560" alt="python -m data_engineering.cli --help" />
  <br/><em>`python -m data_engineering.cli --help` — one command runs the whole pipeline.</em>
</p>

<p align="center">
  <img src="assets/media/data-pipeline-run.png" width="640" alt="real pipeline run transcript" />
  <br/><em>The actual run (run id `expanded-repos`, `assets/data-eng.txt`): 20,477 ingested → 20,470 validated (7 rejected) → 17,456 cleaned → 15,011/1,556/889 split + 2,313 golden → 14,833 tokenized — W&B artifacts, manifest hash, and the GCS round-trip in one transcript.</em>
</p>

### Every subsystem is one command

<p align="center">
  <img src="assets/media/cli-eval.png" width="440" alt="python -m evaluation.cli --help" />
  <img src="assets/media/cli-train.png" width="440" alt="python -m training.qlora_train --help" />
  <br/><em>Evaluation and training each expose a single Typer CLI — the pipeline, the training and the eval all run off the same repo, same configs.</em>
</p>

---

## Anatomy of a Real Run (the system was up)

Everything in this README corresponds to at least one artifact still on the live GCS bucket, Modal, or W&B. Reconstructed timeline (real timestamps):

| Stage | Object / Event | Real artifact |
| ----- | -------------- | ------------- |
| `21:32:46` | Ingest wrote `raw.jsonl` | `796,629,716` B in `datasets/expanded-repos/swebench/` |
| `21:35:19` | Validation wrote `validated.jsonl` | `796,618,866` B (7 schema errors → `validation_errors.jsonl`, 848 B) |
| `21:36:04` | Cleaning wrote `cleaned.jsonl` | `223,394,448` B (3,014 dropped) |
| after | Split wrote `train/val/test` | 184.2 MB / 25.6 MB / 13.6 MB |
| after | `golden.jsonl` | 48,983,535 B (2,313 instances) |
| `07-31 → 08-06 00:33` | Tokenized train splits → arrow files | 25 MB + 374 MB + 327 MB in `tokenized/expanded-repos/` |
| `2026-08-06` | QLoRA training logs | `logs/modal-higher_rank_14b-20260806-030415.log` → loss 0.5843, 1 epoch |
| `2026-08-06 01:38` | StubEngine pulled LoRA artifact from W&B | 1,399.16 MB · 40 files · `./artifacts/` |
| live | `compare run_baseline,run_golden` (100-instance golden sample) | `assets/results.txt` — est. cost **$30** |
| live | Modal volumes | `serve-model-cache` (08-06 17:07), `eval-repo-cache`, `eval-test-cache`, `eval-model-cache`, `swe-qwen-data`, `swe-qwen-models` |
| live | GCS bucket totals | 437 dataset run dirs + 322 tokenized run dirs |

In short: the corpus <u>was built</u>, <u>three adapters were trained and shipped to W&B</u>, <u>evaluation executed real software-engineering tests</u>, and <u>an OpenAI-compatible server answered requests</u> — with the evidence above as the receipt.

---

## Component Deep Dives

Each subsystem below is independently runnable, CI-gated, and covered end-to-end in `docs/`.

### 1. Data Engineering (`data_engineering/`)

The data layer turns raw GitHub issue + PR dumps into a tokenized, `pydantic`-typed training corpus — reproducibly, with every stage versioned. The central type is `IssueRecord` (`data_engineering/schema.py`): 14 typed fields covering identity, the issue body, the gold patch (+ parsed hunks), test results, PR context, changed files, and SWE-bench metadata — validated on ingestion so downstream stages never see malformed rows.

**Under the hood — every stage in one command (`python -m data_engineering.cli run`)**:

| Stage | What actually happens | Live gate |
| ----- | --------------------- | --------- |
| `ingest` | Reads SWE-bench from the local Hugging-Face datasets cache (`data/swe_bench/`), fanning out with parallel workers (max 32, batch 50, `--max-issues` cap) → `raw.jsonl` | 20,477 |
| `validate` | Builds `IssueRecord` (pydantic): `patch_diff` must parse as a `unidiff.PatchSet` **or** match `---`/`+++`/`@@` diff headers; field-level errors → `validation_errors.jsonl` | 20,470 ✓ · 7 rejected |
| `clean` | Six counted gates (no test files · patch > 500 lines · binary diffs · non-Python · empty body · no F2P signal), then exact + semantic dedup → `cleaned.jsonl` | 17,456 ✓ · 726 dup |
| `split` | By-repo 80/10/10 (`--train-ratio 0.8`) — a whole repo goes to one split to stop cross-repo leakage | 15,011 / 1,556 / 889 · 46 repos |
| `golden` | Carves the held-out eval set **before** tokenization from verified + test + dev slices (`GoldenSet{records, f2p_verified_count, source_split}`) | 2,313 |
| `tokenize` | Qwen3-14B tokenizer, `max_length=8192` (cap 32768), SFT `packing=true` → arrow datasets via `datasets` | 14,833 train · 2,304 golden |

Every run is hash-pinned in a `manifest.json` + `dataset_card.md`, artifacts are versioned in W&B (`dataset-cleaned:v8`-era tags) and mirrored to `gs://swe-qwen-datasets/datasets/{run_id}/`; `python -m data_engineering.cli config` dumps the effective `DataPipelineConfig` for reproduction.

<p align="center">
  <img src="assets/media/gcs-artifacts.png" width="560" alt="gs://swe-qwen-datasets expanded-repos artifact tree" />
  <br/><em>Live GCS: `datasets/expanded-repos/swebench/*.jsonl` — 8 objects, 1.95 GiB total (raw 796 MB → cleaned 223 MB …).</em>
</p>

```mermaid
flowchart LR
    A["raw.jsonl"] --> B["validate.py<br/>IssueRecord pydantic schema<br/>errors → validation_errors"]
    B --> C["clean.py<br/>6 quality gates"]
    C --> D["split.py<br/>by-repo 80/10/10"]
    D --> E["golden.py<br/>verified+test+dev → golden"]
    C --> F["tokenize.py<br/>Qwen3-14B · 8,192 ctx"]
    D --> F
    style A fill:#2a2a52,color:#fff
    style B fill:#3b82f6,color:#fff
    style C fill:#3b82f6,color:#fff
    style D fill:#3b82f6,color:#fff
    style E fill:#f59e0b,color:#1f2937
    style F fill:#8b5cf6,color:#fff
```

| Parameter | Setting | Rationale |
| --------- | ------- | --------- |
| Max patch lines | 500 (`--max-patch-lines`) | reject model-unanswerable mega-diffs |
| Language gate | Python only | consistent, testable corpus for v1 |
| Binary / empty files | dropped | no junk tokens in training |
| Duplicates | exact + semantic (726 removed) | no data inflation |
| Split | per-repo 80/10/10, `--train-ratio 0.8` | prevent repo leakage between splits |
| Golden set | carved from verified + test + dev | eval oracle never sees training data |
| Tokenization | Qwen3-14B tokenizer, `max_length=8192` (max 32768) | fits LoRA context; SFT packing enabled |

**Live run numbers** (run id `expanded-repos`): ingest **20,477** → validate **20,470** (7 schema errors) → clean **17,456** (net −3,014: 12 binary diffs, 1,212 non-Python records, 1,149 oversized patches, 726 duplicates) → split **15,011 / 1,556 / 889** (46 repos) → golden **2,313** → tokenized **14,833 / 1,550 / 885 / 2,304** (train/val/test/golden).

**Reproducibility:** any stage can be re-run independently (`--stages ingest,validate,clean`, `--resume-from validated|cleaned`), every run is hash-pinned in a `manifest.json` + `dataset_card.md`, and artifacts are versioned in W&B and mirrored to GCS. See [docs/dataset.md](docs/dataset.md).

### 2. QLoRA Training (`training/`)

```mermaid
flowchart LR
    A["data/tokenized/"] --> B["modal_train.py<br/>Modal app swe-qwen-training-v2"]
    B --> C["qlora_trainer.py<br/>Unsloth + FlashAttention 2.8"]
    C --> D["W&B artifact<br/>model-qwen3-14b-{variant}"]
    D --> E["models/comparisons/expanded-repos/"]
    style A fill:#2a2a52,color:#fff
    style B fill:#8b5cf6,color:#fff
    style C fill:#8b5cf6,color:#fff
    style D fill:#7c3aed,color:#fff
    style E fill:#10b981,color:#fff
```

| Variant | LoRA rank α | lr | batch × grad-accum | notes |
| ------- | ----------- | -- | ------------------ | ----- |
| `baseline_14b` | r16 · α32 · dropout 0.0 | 2e-5 | 2 × 8 | needs dropout 0.0 for Unsloth fast patching (~2×) |
| `higher_rank_14b` | r32 · α64 | 2e-5 | 1 × 16 | more trainable params, same budget |
| `higher_lr_14b` | r16 · α32 · dropout 0.05 | 5e-5 | 2 × 8 | higher LR + dropout, same rank |

Shared: **1 epoch**, `max_seq_length=4096` (longest in corpus), bf16, `paged_adamw_8bit`, cosine, warmup 0.03, weight decay 0.01, `max_grad_norm=1.0`, gradient checkpointing, `packing=true`, `eval_strategy="no"` (eval OOMs on A10G — evaluation is a separate step by design). GPU: **A100-80GB**, timeout 18,000 s (5 h), 1 retry. Full 19K-run ≈ 3–4 h.

**How a training run happens:**

1. `modal run training/modal_train.py::train_qlora` boots an **image-as-code** Modal app: `debian-slim` + torch 2.11 (cu126) + `transformers>=5.5` + `unsloth[colab-new]` + flash-attn 2.8.3 cu126 wheel, with `wandb-secret` / `hf-secret` and the `swe-qwen-models` volume attached.
2. The tokenized corpus is pulled from the **public** GCS bucket (`tokenized/{run_id}/`) using the stdlib `urllib` JSON API — deliberately *not* `CloudBucketMount`, because the GCP org policy forbids HMAC service-account keys (`iam.disableServiceAccountKeyCreation`).
3. `unsloth_factory` loads the base Qwen3-14B 4-bit NF4 and applies fast-attention patches; `qlora_trainer.py` runs the variant block from `config/qlora_variants.yaml` with `packing=true`, gradient checkpointing, and `paged_adamw_8bit`.
4. Every 10 steps it logs loss / grad-norm / lr; checkpoints save every 500 steps (last 3 kept); `eval_strategy="no"` because evaluation is the *separate* execution-based step in the next section.
5. On completion the adapter uploads to W&B (`model-qwen3-14b-{variant}`) and lands in `models/comparisons/{run_id}/{variant}/` with `adapter_model.safetensors`, `chat_template.jinja`, tokenizer + `training_args`.

**Why these three variants** — a deliberate one-GPU ablation: `r16/α32 @ lr2e-5` (baseline), `r32/α64` (more trainable parameters), `r16/α32 @ lr5e-5` + dropout (faster adaptation). `scripts/run_3config_comparison.py` trains all three sequentially on the same corpus so the subsequent eval compares *configurations, not data*. Resume mid-run: `training/resume.py` locates the newest adapter; `--resume` continues from the last checkpoint.

<p align="center">
  <img src="assets/media/modal-volumes-gcs.png" width="560" alt="modal volume list + bucket counts" />
  <br/><em>Live Modal volumes (6: `serve-model-cache`, `eval-repo-cache`, `eval-test-cache`, `eval-model-cache`, `swe-qwen-data`, `swe-qwen-models`) and the GCS bucket: 437 dataset run dirs, 322 tokenized run dirs.</em>
</p>

<p align="center">
  <img src="assets/media/trained-adapters.png" width="560" alt="models/comparisons/expanded-repos adapters + checkpoints" />
  <br/><em>Real trained artifacts on disk: 3 adapters with `adapter_model.safetensors`, `chat_template.jinja`, tokenizer, `training_args.bin` + checkpoints.</em>
</p>

**One-command trio:**

```bash
python -m data_engineering.cli run --run-id expanded-repos --tokenize-model qwen3-14b --tokenize-max-length 8192
modal run training/modal_train.py::train_qlora --model-name qwen3-14b --variant baseline_14b --run-id expanded-repos
python scripts/run_3config_comparison.py --run-id expanded-repos --max-train-samples 3000   # + --force-retrain · drop the flag for the full 14,833-example corpus
```

**Prompts are versioned components** (`training/prompts/`): `system.j2`, `user.j2`, `assistant.j2`, `chat.j2` — Jinja2 templates shared with inference (`inference/prompt_builder.py`), so a prompt change between two experiments is **attributable and auditable** (not a silent confounder). `evaluation.cli run_prompt_ab` runs A/B prompt-template comparisons (`--sample 200`) with the same paired significance machinery.

See [docs/experiments.md](docs/experiments.md) for the full training/experiment loop.

<p align="center">
  <img src="assets/media/modal-training.png" alt="Modal training run" width="560"/>
  <br/><em>Drop `modal-training.png` into `assets/media/` — a real training run in the Modal UI / logs.</em>
</p>

### 3. Evaluation Harness (`evaluation/`)

The harness runs **real code**: materialize `repo@base_sha` with the per-instance official SWE-bench Docker image, generate a patch with your model, **apply** it (3-strategy fallback, method recorded), then execute `FAIL_TO_PASS` and `PASS_TO_PASS` inside the container. Every cell is an `EvalResult` (`evaluation/schema.py`): instance identity, the generated patch + how it was applied, per-test outcomes, per-instance F2P/P2P scores, latency, and error — the full audit trail.

```mermaid
flowchart TB
    A["--split golden · seed 42"] --> B["tier:<br/>smoke 20 · dev 100 · final 500 · full 50"]
    B --> C["Materialize repo@base_sha<br/>official image, cached volume"]
    C --> D["Generate patch<br/>per-variant GPU task"]
    D --> E["Apply: git apply → gnu patch --fuzz → unidiff"]
    E --> F["Test: F2P → P2P<br/>30s/test · 300s/repo · ≤2 retries"]
    F --> G["Score: F2P / P2P / flaky"]
    G --> H["Stats: Wilson CI · McNemar · paired bootstrap"]
    H --> I{"gate"}
    I -- pass --> J["champion"]
    I -- fail --> K["stay"]
    style A fill:#f59e0b,color:#1f2937
    style B fill:#f59e0b,color:#1f2937
    style C fill:#f59e0b,color:#1f2937
    style D fill:#f59e0b,color:#1f2937
    style E fill:#f59e0b,color:#1f2937
    style F fill:#f59e0b,color:#1f2937
    style G fill:#f59e0b,color:#1f2937
    style H fill:#f59e0b,color:#1f2937
    style I fill:#ef4444,color:#fff
    style J fill:#10b981,color:#fff
    style K fill:#64748b,color:#fff
```

| Setting | Value | Why |
| ------- | ----- | --- |
| Tier sizes | smoke 20 · dev 100 · final 500 · full 50 | cost-proportional confidence |
| Determinstic subset | `tier_seed = 42` | comparable runs |
| `max_parallel` | 16 (64 broke Modal 1.5.3 aiohttp) | proven ceiling |
| `max_new_tokens` | 8,192 (smoke tier) | 2,048 truncated patches mid-diff (~75% budget goes to out-loud reasoning) |
| Patch application | recorded `method_used` | "patch failed" ≠ "patch wrong" |
| Resume | `--resume run_id` | don't re-burn GPU mid-run |
| Cost | `--sample 100` full compare ≈ **$30** (2 runs × 4 models) | see `assets/results.txt` |

**Under the hood:**

- `evaluation.cli run` fans each `model:variant` out to its **own Modal GPU task** (maximum `--max-parallel 16` — proven ceiling after 64 broke Modal 1.5.3's aiohttp), with the golden set resolved from `EvalConfig.golden_data_path` (`gs://swe-qwen-datasets/datasets/{run_id}/swebench/golden.jsonl`).
- Each instance materializes `repo@base_sha` in the `eval-repo-cache` volume with its **official SWE-bench Docker image**; the harness applies the *test patch* to establish the fail-state, then the model's patch via the 3-strategy applier (`git apply` → `gnu patch --fuzz` → `unidiff`), recording `method_used` on every cell.
- `test_runner` executes `FAIL_TO_PASS` then `PASS_TO_PASS` inside the container — 30 s/test, 300 s/repo, ≤ 2 retries so infra flapping never silently flips a score; each cell is an `EvalResult` (patch + how it applied, per-test outcomes, F2P/P2P, latency, error).
- Runs **resume** (`--resume run_id`), persist under `data/eval_results/` with `cost_usd` billing, and stream per-example + aggregate rows to the W&B `swe-qwen` project.
- `compare` re-aggregates on the **paired** instances, reports `F2PMetrics` (rates + 95% Wilson CIs + latency + per-repo breakdown), then McNemar + paired bootstrap across runs — the numbers behind `assets/results.txt`.

The released reference run (100 golden instances/model) is reproduced from `assets/results.txt` in [docs/evaluation.md](docs/evaluation.md) — with the champion `higher_rank_14b` clearing the promotion gate (F2P 17.20% ≥ 15%, P2P 90.10% ≥ 90%).

### 4. Promotion & Registry (`promotion/`)

```mermaid
flowchart LR
    A["challenger eval<br/>+ champion eval"] --> B["paired compare<br/>McNemar + bootstrap"]
    B --> C{"gate<br/>F2P ≥ 15% · P2P ≥ 90%<br/>gain CI LB > 0 · P2P drop ≤ 2pt"}
    C -- pass --> D["registry eval-champion<br/>W&B decision record"]
    C -- fail --> E["rejected · audit trail"]
    style A fill:#f59e0b,color:#1f2937
    style B fill:#ef4444,color:#fff
    style C fill:#ef4444,color:#fff
    style D fill:#10b981,color:#fff
    style E fill:#64748b,color:#fff
```

Promotion is a **decision with a paper trail**, never a merge. `promotion/` splits the concern: `rules.py` (the criteria), `gate.py` (the verdict), `registry.py` (W&B `eval-champion`), `audit.py` (the human-readable record), `deploy.py` (the hand-off).

**The four conditions, checked on the same paired sample as the compare:**

1. **Absolute floors** — challenger F2P ≥ 15% *and* P2P ≥ 90% (`min_f2p_threshold`, `min_p2p_threshold`): a model below the floor is not deployable even if it "beat" a weaker champion.
2. **The gain is real, not noise** — paired-bootstrap 95% CI lower bound on the F2P delta must be *strictly > 0* (the released `higher_rank_14b` vs base run reports McNemar `p < 1e-6`).
3. **No regression** — P2P may not drop more than 2 points across the paired set; an offensive win that breaks other tests is not a win.
4. **Silent promotions are rejected** — if a challenger can't clear its own confidence interval, `gate.py` keeps the incumbent and writes the rejection for the audit trail.

**Who remembers:** the champion of record is `gs://swe-qwen-datasets/ci/champion.json` (read by `eval.yml` baselines and the dashboards); `registry.py` appends the full decision record to the W&B `eval-champion` collection; deployment is gated to the `production` environment, and any step can be **dry-run** with `RUN_MODAL_EVAL=false` — the gate re-scores the last logged numbers at $0.

```bash
python -m evaluation.cli compare --run_ids run_baseline,run_golden --promote-to-registry    # local gate + promote
gh workflow run promote.yml -f candidate_variant=higher_rank_14b                            # CI champion/challenger
```

### 5. Inference Serving (`inference/`)

OpenAI-compatible API surface — **your client code doesn't change**. `POST /v1/chat/completions` (stream + non-stream), `GET /health`; models resolve as `qwen3-14b`, `qwen3-14b:{variant}`, bare `{variant}`, or W&B artifact name; errors are faithful OpenAI envelopes (401/422/404/500).

| Endpoint | Auth | Behavior |
| -------- | ---- | -------- |
| `POST /v1/chat/completions` | Bearer (`MODAL_SERVE_TOKEN`, constant-time compare, fail-closed) | chat completion, SSE streaming (`data: [DONE]`) |
| `GET /health` | open | `{status, model, engine}` |

Engines: **VLLMEngine** (`SERVING_STUB=0`; AWQ int4 `Qwen/Qwen3-14B-AWQ`, `enable_lora=True`, `max_lora_rank=64`, gpu_mem 0.85, 16 max seqs, `LoRARequest(lora_int_id=1)` per request) and **StubEngine** (default, deterministic local dev). Prompt assembly inserts the Qwen3 `no_think` soft-switch (`/no_think\n### Response`) for LoRA models. Full API reference: [docs/api.md](docs/api.md).

**Wire format** — OpenAI-compatible (`model_config = {"extra": "ignore"}`), see the full schemas in [docs/api.md](docs/api.md):

```json
// POST /v1/chat/completions · Authorization: Bearer $MODAL_SERVE_TOKEN
{ "model": "qwen3-14b:higher_rank_14b", "messages": [{ "role": "user", "content": "Explain QLoRA in one sentence" }],
  "temperature": 0.1, "top_p": 0.95, "max_tokens": 512, "stream": false }
```

Streaming returns SSE `data: {json}\n\n` frames — role chunk → content chunks → final (`finish_reason:"stop"`) → `data: [DONE]`. Errors are faithful envelopes: `401 {"detail":"invalid or missing bearer token"}`, `422 invalid_request_error`, `404 model_not_found`, `500 server_error`.

**Under the hood:**

1. **Auth first.** Every completion checks the Bearer token against `MODAL_SERVE_TOKEN` with `hmac.compare_digest`, fail-closed (401 before any prompt touches the model); `/health` stays open. Per-request order: auth → pydantic validation (422) → model resolution (404) → engine call.
2. **Model resolution** (`openai_compat.resolve_engine_model`): `qwen3-14b` = base model, no adapter; `qwen3-14b:{variant}` / bare `{variant}` / `model-qwen3-14b-{variant}` = LoRA. Requesting the same name as the base returns the base without an adapter.
3. **VLLMEngine** holds a process-singleton `_LLM_CACHE` keyed by the serving Hf id — one AWQ int4 `Qwen/Qwen3-14B-AWQ` process with `enable_lora=True`, `max_lora_rank=64`, and a per-request `LoRARequest(lora_name, lora_int_id=1, lora_path)`. The **first request for a variant pulls the LoRA weights from W&B** (demonstrated: 1,399.16 MB / 40 files) and caches them on the `serve-model-cache` volume — serverless adapters, zero pre-provisioning.
4. **Prompt assembly differs by model type** — LoRA models are conditioned on the raw `### Response -> patch` continuation with the Qwen3 `no_think` soft-switch (`/no_think\n### Response`); the base model gets the full chat template with `enable_thinking=False` — a 14B that "thinks out loud" burns ~75% of the budget before diffing.
5. **The wire stays OpenAI**: `chatcmpl-{12 hex}` ids, `choices`/`usage`, SSE `data: [DONE]`, word-chunk streaming with TTFB recorded at the first chunk; every request emits a `RequestRecord` (ts · model · stream · ttfb_ms · latency_ms · tokens · error) to the observability layer.

<p align="center">
  <img src="assets/media/modal-qwen-server.png" alt="Modal serving endpoint" width="560"/>
  <br/><em>Modal serving endpoint — the vLLM + LoRA server sits at zero/cold until a request scales it up; per-request GPU billing, no idle cost.</em>
</p>

<p align="center">
  <img src="assets/media/inference-demo.png" alt="Streaming inference demo" width="560"/>
  <br/><em>Live streaming — `/v1/chat/completions` with `curl -N` over SSE: role chunk → content chunks → `finish_reason:"stop"` → `data: [DONE]`.</em>
</p>

### 6. Observability (`observability/`)

Every layer of the platform phones home, and the dashboards that visualize it are versioned in this repo.

- **Weights & Biases** — two projects: `swe-qwen-data` (every pipeline artifact `raw → validated → cleaned → train/val/test → golden → tokenized`, hash-pinned manifest, `dataset_card.md`) and `swe-qwen` (training runs with live loss curves, eval runs with per-example rows + aggregates + `cost_usd`, and the `eval-champion` registry collection).
- **Langfuse** — 10% of LLM requests traced (`telemetry_trace_sample_rate`): prompts, responses, latency — bounded cost, full audit of the sampled wire.
- **GCP Logging / structured logs** — every inference request is a `RequestRecord` (`ts · model · stream · ttfbs_ms · latency_ms · output_tokens · error · error_type · status`), so latency regressions are queryable, not anecdotal.
- **Dashboards as code** — `scripts/build_dashboards.py` + `scripts/seed_dashboards.py` (`wandb-workspaces`) keep the W&B dashboards in git; `docs/observability/architecture.md` + `dashboards.md` document the layout.
- **Cost** — `observability/cost.py` folds Modal + GCS spend into each `EvalRun.cost_usd` (the `$30` figure in `assets/results.txt` is the sum of the two runs' recorded spend).

<p align="center">
  <img src="assets/media/langfuse.png" alt="Langfuse trace" width="560"/>
  <br/><em>Langfuse — one request traced end-to-end (prompt → generated patch → latency), sampled at 10% (`telemetry_trace_sample_rate`).</em>
</p>

<p align="center">
  <img src="assets/media/w%26b-dashboards.gif" alt="W&B dashboards-as-code" width="480"/>
  <br/><em>W&B workspaces — dashboards as code: `scripts/build_dashboards.py` + `scripts/seed_dashboards.py` (wandb-workspaces) keep the layout in git, not in a browser tab.</em>
</p>

---

## Design Decisions

1. **Execution over unit-test proxies.** The only score that matters is patching the repo and running the real tests — everything else is a proxy.
2. **QLoRA + Unsloth over full fine-tuning.** 1.76% of params trained 4-bit NF4; A100-80GB instead of 8×A100s; 3 variants in the cost of 1.
3. **Modal over VMs/Dockerfiles.** Image-as-code + scale-to-zero + per-request GPU is the cheapest correct abstraction for bursty train/test/serve.
4. **GCS as the data backbone, W&B as the artifact registry.** Public-read GCS sidesteps Modal's CloudBucketMount HMAC limit (`iam.disableServiceAccountKeyCreation` on the project).
5. **CI-gated evaluation** (`eval.yml` + `promote.yml`) so "did the model regress?" is answered on every PR, not quarterly.
6. **Typed-everything** (`pydantic` schemas for data, eval, and OpenAI wire format) — failures are structured, not stringly-typed.

---

## Testing Strategy

| Layer | Tooling | Coverage of |
| ----- | ------- | ----------- |
| Unit + integration | pytest (**1,456 passed**, 1 skipped, 5 deselected, ~3 min) | every package: data_engineering, evaluation, inference, training, promotion, observability, scripts |
| Lint / format | ruff (line-length 100, strict rule set) | `All checks passed` |
| Type checking | mypy (strict-ish) | 42 source files, `Success: no issues found` |
| Coverage | pytest-cov (`--cov`, branch=true) + Codecov | 8 packages |
| Model regression | `eval.yml` smoke gate (20-instance F2P vs baseline, `_SMOKE_TOLERANCE=0.05`, PRs read / main writes) | catches real model-quality regressions per PR |

<p align="center">
  <img src="assets/media/pytest-summary.png" width="480" alt="pytest summary — 1,456 passed" />
</p>

> The full offline suite runs green in ~3 minutes (`pytest`, 5 deselected = Modal/GCP/W&B integration tests).

Run locally:

```bash
uv sync --extra dev
ruff check . && ruff format --check .
mypy data_engineering/ evaluation/ scripts/
pytest -m "not requires_modal and not requires_gcp and not requires_wandb and not requires_credentials"
```

---

## Infrastructure (Terraform)

```text
gs://swe-qwen-datasets           <- private GCS, dedicated 3-module layout
├─ terraform/
│  ├─ modules/storage/            (bucket + lifecycle + IAM)
│  └─ modules/iam/                (Workload Identity Federation pool/provider)
├─ providers.tf · main.tf · variables.tf
```

- **State & storage as code**: bucket naming/lifecycle/IAM parameterized (`dataset_bucket_name`, `model_bucket_name`, `enable_workload_identity=true`).
- **Workload Identity Federation** for GitHub Actions (`gcp_project_id`, OIDC `github-actions-pool-dev`) — no service-account keys allowed by org policy.
- **CI/CD wiring**: `cd.yml` plans on PRs, applies to `production` on push-relevant-changes (paths: `infra/**`, `inference/**`, `config/**`, `pyproject.toml`, `uv.lock`), then `modal deploy`.
- **Eval volumes**: `eval-repo-cache` + `eval-test-cache`; training models → `swe-qwen-models`; inference adapters → `serve-model-cache`.

---

## CI/CD Pipeline

| Workflow | Job | Key config |
| -------- | --- | ---------- |
| `ci.yml` | lint + typecheck + tests | ruff · mypy · pytest (offline markers) · paths-ignore `**.md`, `docs/**` · concurrency cancel-in-progress |
| `cd.yml` | infra plan / apply + Modal deploy | `pull_request` plan · `push main` apply · env `production` |
| `eval.yml` | SWE-bench smoke gate | 20-instance F2P vs `smoke_baseline.json` · PRs read / main writes |
| `promote.yml` | champion/challenger promotion | paired eval · 4-condition statistical gate · W&B decision record · optional dry-run |

<p align="center">
  <img src="assets/media/ci.png" alt="ci.yml run" width="400"/>
  <img src="assets/media/eval.png" alt="eval.yml run" width="400"/>
  <br/><em>Left: `ci.yml` — ruff + mypy + the full pytest suite. Right: `eval.yml` — the SWE-bench smoke gate on PRs (baseline read-only for PRs, updated on main).</em>
</p>
<p align="center">
  <img src="assets/media/cd-deploy.png" alt="cd.yml run" width="400"/>
  <img src="assets/media/promote.png" alt="promote.yml run" width="400"/>
  <br/><em>Left: `cd.yml` — Terraform plan/apply + Modal deploy, gated to production. Right: `promote.yml` — champion-vs-challenger statistical promotion with the 4-condition gate.</em>
</p>

---

## Security Model

| Layer | Mechanism |
| ----- | --------- |
| Cloud auth | GCP Workload Identity Federation (OIDC, short-lived tokens) — org policy blocks SA keys |
| Machine access | Modal Secrets (`wandb-secret`, `hf-secret`) |
| API auth | Bearer token, constant-time compare, fail-closed, 401 on missing |
| Inference isolation | private Modal net, per-request LoRA, no static credentials |
| Data | private GCS bucket; VPC-SC-ready |
| CI secrets | GitHub Secrets (WIF provider, GCP ids, Modal tokens, W&B key) |

<p align="center">
  <img src="assets/media/fail-closed.png" alt="401 fail-closed demo" width="440"/>
  <br/><em>Drop `fail-closed.png` into `assets/media/` — the API refusing a request without a token.</em>
</p>

---

## Getting Started

### Prerequisites

- Python 3.11+ (`uv` recommended), a GitHub account for Actions
- Modal account + token; Weights & Biases account + API key
- Google Cloud project (for Terraform + GCS) — or skip infra by pointing at the public bucket
- Docker (only if you run the eval harness or serve locally with vLLM)

### Quick Start

```bash
# 1. Clone + deps
git clone git@github.com:AhmedIkram05/SWE-Qwen.git && cd SWE-Qwen
uv sync --extra dev --extra training --extra eval --extra inference

# 2. Configure
cp .env.example .env            # add MODAL_TOKEN_ID, WANDB_API_KEY, GITHUB_TOKEN, GCP_PROJECT_ID
source .venv/bin/activate

# 3. Provision infra (needs gcloud auth) — or skip: bucket already has public data
gcloud auth login
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform apply \
  -var gcp_project_id=$GCP_PROJECT_ID -var gcp_region=europe-west2 \
  -var dataset_bucket_name=swe-qwen-datasets -var model_bucket_name=swe-qwen-models \
  -var enable_workload_identity=true
export GCP_WIF_PROVIDER="projects/1001461381543/locations/global/workloadIdentityPools/github-actions-pool-dev/providers/github-provider-dev"
gcloud iam workload-identity-pools create-cred-config "$GCP_WIF_PROVIDER" \
  --service-account="swe-qwen-github@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --output-file=gha-wif.json

# 4. Build the dataset (run the real pipeline)
python -m data_engineering.cli run --run-id expanded-repos \
  --tokenize-model qwen3-14b --tokenize-max-length 8192

# 5. Train 3 QLoRA variants on Modal (A100-80GB)
modal run training/modal_train.py::train_qlora --model-name qwen3-14b --variant baseline_14b --run-id expanded-repos
modal run training/modal_train.py::train_qlora --model-name qwen3-14b --variant higher_rank_14b --run-id expanded-repos
modal run training/modal_train.py::train_qlora --model-name qwen3-14b --variant higher_lr_14b --run-id expanded-repos
# ...or one shot:
python scripts/run_3config_comparison.py --run-id expanded-repos --max-train-samples 3000   # --force-retrain

# 6. Evaluate (baseline + variants on the same seeded sample)
export EVAL_DATASET_RUN_ID=expanded-repos
python -m evaluation.cli run --split golden --sample 100 --models qwen3-14b:baseline --resume run_baseline
python -m evaluation.cli run --split golden --sample 100 \
  --models qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b --resume run_golden

# 7. Compare (statistics + optional promotion)
python -m evaluation.cli compare --run_ids run_baseline,run_golden --promote-to-registry

# 8. Serve the champion (OpenAI-compatible, scale-to-zero)
modal serve inference.modal_serve
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model":"Qwen/Qwen3-14B-AWQ","engine":"VLLMEngine"}
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $MODAL_SERVE_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b:higher_rank_14b","messages":[{"role":"user","content":"Explain QLoRA in one sentence"}]}'
```

### Configuration

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Modal auth | – |
| `WANDB_API_KEY` | Weights & Biases | – |
| `GITHUB_TOKEN` | GitHub (registry/CI) | – |
| `GCP_PROJECT_ID` / `GCP_REGION` | Terraform / GCS | – |
| `EVAL_DATASET_RUN_ID` | eval golden-set pointer | `expanded-repos` |
| `SERVING_STUB` | `0` → vLLM, else stub engine | stub |

GitHub Actions secrets: `GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `WANDB_API_KEY`, `GITHUB_TOKEN`. Training config lives in `config/qlora_variants.yaml` + `config/models.yaml`.

### Running Tests

See [Testing Strategy](#testing-strategy) — full offline suite is **~3 minutes**.

### Production Deployment

```bash
# Infra + serving
terraform -chdir=infra/terraform apply -var gcp_project_id=$GCP_PROJECT_ID ...   # then
modal deploy inference.modal_serve                                              # scale-to-zero serverless

# Promote a candidate (CI): 
gh workflow run promote.yml -f candidate_variant=baseline_14b                   # + RUN_MODAL_EVAL=false to dry-run
```

---

## Documentation

| Doc | What it covers |
| --- | -------------- |
| [docs/api.md](docs/api.md) | OpenAI-compatible endpoints, request/response schemas, SSE, errors, model resolution |
| [docs/experiments.md](docs/experiments.md) | Full loop: dataset → train → eval → compare → promote → serve |
| [docs/dataset.md](docs/dataset.md) | Pipeline stages, schema, quality gates, reproducibility |
| [docs/evaluation.md](docs/evaluation.md) | F2P/P2P methodology, statistics, golden-set protocol |
| [docs/benchmarks.md](docs/benchmarks.md) | **Final benchmark report** — measured F2P/P2P/latency/cost, champion selection |
| [docs/observability/architecture.md](docs/observability/architecture.md) | System architecture, telemetry flows, dashboards layout |
| [docs/observability/dashboards.md](docs/observability/dashboards.md) | Dashboards-as-code: `scripts/build_dashboards.py` + `scripts/seed_dashboards.py` |
| [docs/planning/](docs/planning/) | Phase-by-phase engineering plans (ADR & vision, phases 2–9 design docs) |
| [docs/IMPLEMENTATION-LOG.md](docs/IMPLEMENTATION-LOG.md) | Chronological record of every stage built, run and evaluated |

---

## Related Projects

- **[LAAD](https://github.com/AhmedIkram05/LAAD)** — Apache Kafka · Flink · MLflow · RAG anomaly-detection platform (sister LLMOps/MLEng README style)
- **[DevSync](https://github.com/AhmedIkram05/DevSync)** — developer collaboration platform (REST + CI/CD)
- **[W3C ETL Pipeline](https://github.com/AhmedIkram05/w3c-etl-pipeline)** — batch ETL: 153 K IIS rows in 45 s → Power BI
- **[StockLens](https://github.com/AhmedIkram05/StockLens)** — algorithmic trading + ML pipelines

---

<p align="center"><b>SWE-Qwen</b> — SWE-bench → QLoRA → execution-based eval → statistical promotion → OpenAI-compatible serving.<br/>Built with Python · PyTorch · Modal · Terraform · Google Cloud · W&B · GitHub Actions.<br/>MIT © Ahmed Ikram</p>
