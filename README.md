# SWE-Qwen

**Fine-tuning Qwen3-30B-A3B (MoE) for Software Engineering tasks using LoRA on Modal GPUs**

SWE-Qwen is a specialized code generation model fine-tuned from **Qwen3-30B-A3B** (MoE — 30B total, 3B active params) for software engineering tasks including code generation, bug fixing, repository understanding, and automated code review. Falls back to **Qwen3-14B** if VRAM constraints require.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SWE-Qwen Platform                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   GitHub    │    │   Modal     │    │     W&B     │             │
│  │  Actions    │───▶│   GPUs      │───▶│  Tracking   │             │
│  │   (CI/CD)   │    │ (Training)  │    │  (Metrics)  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                   │                   │                   │
│        ▼                   ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    GCP Infrastructure                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │   GCS    │  │  IAM     │  │  Secret  │  │  Artifact│    │   │
│  │  │ Buckets  │  │  (WIF)   │  │ Manager  │  │ Registry │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Credentials Required

This project requires **all** of the following before you can run infrastructure or training:

| Service | What You Need | How To Get It |
|---------|--------------|---------------|
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

### Local Development

```bash
# Clone and setup
git clone https://github.com/ahmedikram/SWE-Qwen.git
cd SWE-Qwen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linters
ruff check .
ruff format --check .
mypy src/
```

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
|--------|-------|---------|
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

### Deploy Modal App

```bash
modal deploy modal_app.py
```

### Run Training

```bash
# Local test
modal run modal_app.py::hello_modal

# Full training on Modal GPUs
modal run modal_app.py::train_swe_qwen \
  --model_name Qwen/Qwen3-30B-A3B \  # or Qwen/Qwen3-14B
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### Serve Model

```bash
# Deploy inference server
modal deploy modal_app.py::serve_swe_qwen \
  --model_path /models/swe-qwen-finetuned
```

## Project Structure

```
SWE-Qwen/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── infra/
│   └── terraform/
│       ├── main.tf             # Root module
│       ├── variables.tf        # Input variables
│       ├── providers.tf        # Provider configuration
│       └── modules/
│           ├── storage/        # GCS buckets module
│           │   ├── main.tf
│           │   ├── variables.tf
│           │   └── outputs.tf
│           └── iam/            # IAM & WIF module
│               ├── main.tf
│               ├── variables.tf
│               └── outputs.tf
├── modal_app.py                # Modal application (training + serving)
├── pyproject.toml              # Python project configuration
├── scripts/
│   └── init_wandb.py          # W&B project initialization
├── src/
│   └── swe_qwen/              # Main package (to be implemented)
├── tests/
│   ├── test_infra.py          # Terraform output tests
│   └── test_scaffold.py       # Project scaffold tests
└── README.md
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MODAL_TOKEN` | Modal API token | Yes |
| `WANDB_API_KEY` | Weights & Biases API key | Yes |
| `GITHUB_TOKEN` | GitHub personal access token | Yes |
| `GCP_PROJECT_ID` | GCP project ID | For Terraform |
| `GCP_REGION` | GCP region (default: us-central1) | For Terraform |

### Terraform Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `gcp_project_id` | GCP Project ID | - |
| `gcp_region` | GCP region | `us-central1` |
| `environment` | Environment name | `dev` |
| `repository_name` | GitHub repo name | `SWE-Qwen` |
| `repository_owner` | GitHub org/user | `ahmedikram` |
| `dataset_bucket_name` | GCS dataset bucket | Auto-generated |
| `model_bucket_name` | GCS model bucket | Auto-generated |
| `enable_workload_identity` | Enable WIF for GitHub | `true` |

## Training Configuration

Default hyperparameters in `modal_app.py::train_swe_qwen`:

```python
model_name = "Qwen/Qwen3-30B-A3B"  # fallback: Qwen/Qwen3-14B
num_epochs = 3
batch_size = 4
learning_rate = 2e-4
lora_rank = 64
lora_alpha = 128
lora_dropout = 0.05
use_4bit = True
use_flash_attn = True
gradient_accumulation_steps = 4
max_seq_length = 4096
warmup_ratio = 0.03
weight_decay = 0.01
```

## Why No Dockerfile?

This project does **not** use a Dockerfile. Instead, Modal handles containerization:

- **Modal Image** (`modal.Image.debian_slim()` in `modal_app.py`) defines the container in pure Python — apt packages, pip installs, CUDA toolkit, etc.
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
4. **Modal Deploy** - Deploy app on main branch
5. **Docker Build** - Build & push to Artifact Registry

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
4. Run CI locally: `ruff check . && mypy src/ && pytest tests/`
5. Submit PR

## Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- W&B: Project dashboard for training runs