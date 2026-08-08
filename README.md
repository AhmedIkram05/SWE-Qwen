# SWE-Qwen

> A production-grade, model-agnostic LLMOps platform: data pipeline, QLoRA fine-tuning, execution-based evaluation, and serverless inference for automated software issue resolution

SWE-Qwen is an end-to-end LLMOps platform — data engineering → fine-tuning → execution-based SWE-bench evaluation → OpenAI-compatible serverless serving — built around a **replaceable base model** (currently Qwen3-14B). The platform is the product, not the model: storage (GCS) is fully decoupled from compute (Modal serverless GPUs), base models are a single entry in `config/models.yaml`, and every stage is versioned in W&B with automated Champion/Challenger promotion.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         SWE-Qwen Platform                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   GitHub    │    │   Modal     │    │     W&B     │              │
│  │  Actions    │───▶│   GPUs      │───▶│  Tracking   │              │
│  │   (CI/CD)   │    │ (Training)  │    │  (Metrics)  │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│        │                   │                   │                    │
│        ▼                   ▼                   ▼                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    GCP Infrastructure                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │   GCS    │  │  IAM     │  │  Secret  │  │  Artifact│     │    │
│  │  │ Buckets  │  │  (WIF)   │  │ Manager  │  │ Registry │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  |_____________________________________________________________|    │
|                                                                     |
└─────────────────────────────────────────────────────────────────────┘
```

## Credentials Required

This project requires **all** of the following before you can run infrastructure or training:

| Service | What You Need | How To Get It |
| --------- | -------------- | --------------- |
| **GCP** | GCP project with billing + programmatic access | `gcloud auth application-default login` (install [gcloud CLI](https://cloud.google.com/sdk/docs/install) first) |
| **Modal** | Modal account + API token | `modal setup` after `pip install modal` |
| **W&B** | W&B account + API key | Sign up at [wandb.ai](https://wandb.ai), then `wandb login` |
| **GitHub** | Personal access token (classic, `repo` scope) | [GitHub Settings → Tokens](https://github.com/settings/tokens) |
| **GitHub OIDC** | WIF provider configured (for CI) | Created by `terraform apply` — see infra setup below |

Terraform uses GCS backend (`swe-qwen-terraform-state`) — you **must** have the `gcloud` CLI installed and authenticated. Without it, `terraform init/plan/apply/graph/output` will all fail with auth errors.

## Quick Start

### Prerequisites

- Python 3.11+
- gcloud CLI installed & authenticated (`gcloud auth application-default login`)
- Modal account (`modal setup`)
- Weights & Biases account (`wandb login`)
- GCP project with billing enabled
- Terraform >= 1.6.0

### Infrastructure Setup

```bash
cd infra/terraform

# Initialize Terraform
terraform init

# Plan infrastructure
terraform plan \
  -var="gcp_project_id=YOUR_PROJECT_ID" \
  -var="gcp_region=us-central1" \
  -var="environment=dev" \
  -var="repository_name=SWE-Qwen" \
  -var="repository_owner=YOUR_GITHUB_USERNAME"

# Apply infrastructure
terraform apply \
  -var="gcp_project_id=YOUR_PROJECT_ID" \
  -var="gcp_region=us-central1" \
  -var="environment=dev" \
  -var="repository_name=SWE-Qwen" \
  -var="repository_owner=YOUR_GITHUB_USERNAME"
```

### Configure Secrets

After infrastructure is deployed, add secrets to GCP Secret Manager:

```bash
# Modal token
echo -n "YOUR_MODAL_TOKEN" | gcloud secrets versions add modal-api-token --data-file=-

# W&B API key
echo -n "YOUR_WANDB_API_KEY" | gcloud secrets versions add wandb-api-key --data-file=-

# GitHub token
echo -n "YOUR_GITHUB_TOKEN" | gcloud secrets versions add github-token --data-file=-
```

### GitHub Actions Secrets

Add these secrets in **GitHub → Settings → Secrets and variables → Actions** for CI workflows to work:

| Secret | Value | Used By |
| -------- | ------- | --------- |
| `GCP_PROJECT_ID` | `project-7f2bbd9a-c5f0-48d2-b08` | terraform-plan, docker-build |
| `GCP_REGION` | `us-central1` | terraform-plan, docker-build |
| `GCP_WIF_PROVIDER` | `projects/1001461381543/locations/global/workloadIdentityPools/github-actions-pool-dev/providers/github-provider-dev` | terraform-plan, docker-build |
| `GCP_SERVICE_ACCOUNT` | `github-actions-dev@project-7f2bbd9a-c5f0-48d2-b08.iam.gserviceaccount.com` | terraform-plan, docker-build |
| `MODAL_TOKEN_ID` | Your Modal token ID | modal-deploy |
| `MODAL_TOKEN_SECRET` | Your Modal token secret | modal-deploy |
| `CODECOV_TOKEN` | Your Codecov token | test (coverage upload) |
| `WANDB_API_KEY` | _(optional, add in Phase 4)_ | future training eval |

> **Note:** The GCP org policy `iam.disableServiceAccountKeyCreation` blocks long-lived SA keys. All auth uses Workload Identity Federation (OIDC) — no `GCP_SA_KEY` needed.

### Initialize W&B Project

```bash
python scripts/init_wandb.py \
  --project swe-qwen \
  --entity YOUR_WANDB_ENTITY \
  --create-sweep \
  --setup-registries
```

### Run the Full Pipeline (Data → Training → Evaluation)

The complete manual pipeline from data engineering through evaluation:

```bash
# 1. Data engineering: ingest → validate → clean → split → version (tokenized for training)
python -m data_engineering.cli run --run-id expanded-repos \
  --tokenize-model qwen3-14b --tokenize-max-length 8192

# 2. Train the three QLoRA variants (baseline_14b / higher_rank_14b / higher_lr_14b) and compare them
export EVAL_DATASET_RUN_ID=expanded-repos
python scripts/run_3config_comparison.py --run-id expanded-repos --max-train-samples 3000

# Force a full retrain (ignore cached state):
python scripts/run_3config_comparison.py --run-id expanded-repos --force-retrain --max-train-samples 3000

# 3. Evaluation: baseline vs. fine-tuned variants on the golden set (execution-based F2P / P2P)
export EVAL_DATASET_RUN_ID=expanded-repos
python -m evaluation.cli run --split golden --sample 50 \
  --models qwen3-14b:baseline --resume run_baseline
python -m evaluation.cli run --split golden --sample 50 \
  --models qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b \
  --resume run_golden
python -m evaluation.cli compare --run_ids run_baseline,run_golden
```

- Evaluation is **execution-based**: generated patches run against real test
  suites inside the official per-instance SWE-bench containers, measuring
  Fail-to-Pass resolution (F2P) and Pass-to-Pass regression (P2P) with
  statistical rigor (Wilson CIs, McNemar, paired bootstrap).
- `--split` accepts `golden` (whole golden set; `--sample 0` for all) or
  `swebench_verified`. Subset selection is seeded.
- After `compare`, a champion (best F2P above thresholds) is promoted to the
  W&B Registry `eval-champion` collection.
- Expected cost: a full smoke run is ~$0.15–0.30 on Modal (cold starts and
  image pulls dominate the first run).

### Serve the Model (Serverless, OpenAI-Compatible)

```bash
modal serve inference.modal_serve      # dev (hot reload, scale-to-zero)
modal deploy inference.modal_serve     # production endpoint
```

Serving runs vLLM on Modal (A10G) with per-request LoRA adapters over a
pre-quantized base model, OpenAI-compatible `/v1/chat/completions` (streaming
SSE included), TTFB p50 < 500 ms, $0 cost when idle, and W&B telemetry
(latency, throughput, cost per inference).

## Project Structure

```
SWE-Qwen/
├── .github/workflows/          # GitHub Actions CI/CD pipeline
├── config/
│   ├── models.yaml             # Model registry (single source of truth for base model + variants)
│   └── qlora_variants.yaml     # QLoRA training configurations
├── data_engineering/           # Data pipeline: ingest → validate → clean → split → version → archive
│   ├── cli.py                  # Typer CLI (`python -m data_engineering.cli run ...`)
│   ├── run_pipeline.py         # Pipeline orchestration
│   ├── swebench_ingest.py      # SWE-bench ingestion + golden extraction
│   └── ...                     # validate / clean / split / version / archive / golden / card / tokenize
├── training/                   # QLoRA fine-tuning (Modal)
│   ├── modal_train.py          # Modal training entry point
│   ├── qlora_trainer.py / qlora_config.py / callbacks.py / resume.py
│   └── prompts/                # Prompt templates (shared with inference via prompt_builder)
├── evaluation/                 # Execution-based evaluation harness
│   ├── cli.py                  # `python -m evaluation.cli run|compare|run_golden|run_swebench`
│   ├── harness.py              # F2P / P2P engine (runners, resume, W&B logging)
│   ├── test_runner.py          # Test execution in official SWE-bench containers
│   └── stats.py                # Wilson CIs, McNemar, paired bootstrap
├── inference/                  # Serverless inference API (vLLM on Modal)
│   ├── serve.py                # FastAPI app + StubEngine/VLLMEngine abstraction
│   ├── modal_serve.py          # Modal deployment (scale-to-zero)
│   ├── openai_compat.py        # OpenAI-compatible request/response + SSE streaming
│   ├── prompt_builder.py       # Shared prompt logic (also re-exported by evaluation)
│   └── telemetry.py            # TTFB / latency / cost-per-inference metrics → W&B
├── infra/terraform/            # GCS, IAM, OIDC (workload identity), secrets — all IaC
├── scripts/
│   ├── run_3config_comparison.py  # Train + compare the three QLoRA variants
│   ├── preflight_serve.py         # Serving preflight (GPU smoke before deploy)
│   └── init_wandb.py              # W&B project initialization
├── data/                       # Datasets + golden eval set
├── tests/                      # Unit + integration tests
├── pyproject.toml              # Python project configuration
└── docs/planning/              # ADR & Vision, Master Plan, phase plans
```

## Configuration

### Environment Variables

| Variable | Description | Required |
| ---------- | ------------- | ---------- |
| `MODAL_TOKEN` | Modal API token | Yes |
| `WANDB_API_KEY` | Weights & Biases API key | Yes |
| `GITHUB_TOKEN` | GitHub personal access token | Yes |
| `GCP_PROJECT_ID` | GCP project ID | For Terraform |
| `GCP_REGION` | GCP region (default: us-central1) | For Terraform |

### Terraform Variables

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `gcp_project_id` | GCP Project ID | - |
| `gcp_region` | GCP region | `us-central1` |
| `environment` | Environment name | `dev` |
| `repository_name` | GitHub repo name | `SWE-Qwen` |
| `repository_owner` | GitHub org/user | `ahmedikram` |
| `dataset_bucket_name` | GCS dataset bucket | Auto-generated |
| `model_bucket_name` | GCS model bucket | Auto-generated |
| `enable_workload_identity` | Enable WIF for GitHub | `true` |

## Training Configuration

QLoRA hyperparameters live in `config/qlora_variants.yaml` (single source of truth,
consumed by `training/qlora_config.py`). Three 14B-optimized variants are trained and
compared via F2P on the golden set (see `run_3config_comparison.py`):

| Variant | Tuning |
| ---------- | -------------------------------------------- |
| `baseline_14b` | r=16, α=32, lr=2e-5 (template; others inherit) |
| `higher_rank_14b` | r=32, α=64, lr=2e-5 |
| `higher_lr_14b` | r=16, α=32, lr=5e-5 |

Training runs on Modal (A10G 24 GB) with Unsloth, 1 epoch, `max_seq_length=4096`.
The winner is promoted to the W&B Registry as Champion after evaluation.

## Why No Dockerfile?

This project does **not** use a Dockerfile. Instead, Modal handles containerization:

- **Modal Image** (`modal.Image.debian_slim()` in `inference/modal_serve.py`) defines the container in pure Python — apt packages, pip installs, CUDA toolkit, etc.
- Modal manages GPU-optimized base images (CUDA 12.6, flash-attn, etc.) with build caching and automatic rebuilds on code changes.
- Training and inference run on Modal's serverless GPU infrastructure — no Docker daemon, no registry push, no container orchestration needed.

A **Dockerfile is not needed** because:

1. Modal's `base_image` + `gpu_image` replace the image build step entirely
2. Modal volumes (`modal.Volume`) persist datasets and checkpoints, replacing bind mounts
3. The CI `docker-build` job exists **only** as an optional path for those who want Artifact Registry deployment (e.g., for alternative serving setups)

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Lint & Typecheck** - Ruff + MyPy
2. **Tests** - pytest with coverage
3. **Terraform Validate** - fmt, validate
4. **Terraform Plan** - PR preview (on PRs)
5. **Modal Deploy** - Deploy app on main branch
6. **Docker Build** - Build & push to Artifact Registry

## Monitoring & Observability

- **W&B**: Training metrics, hyperparameters, model artifacts
- **GCP Logging**: Modal function logs, infrastructure logs
- **GCP Monitoring**: GPU utilization, training progress
- **Artifact Registry**: Docker images, model checkpoints

## Security

- Workload Identity Federation for GitHub Actions (no long-lived keys)
- Secret Manager for API keys
- Least-privilege IAM roles
- Private GCS buckets (public read only in dev)
- VPC-SC ready (configure via Terraform)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run CI locally: `ruff check . && mypy . && pytest tests/`
5. Submit PR

## Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- W&B: Project dashboard for training runs
