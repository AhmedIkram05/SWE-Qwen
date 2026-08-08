# Experiments Guide — Train, Evaluate, Promote

> SWE-Qwen's experiment loop is **one command per phase**: build the dataset, train three QLoRA variants on serverless A100s, evaluate them execution-style against the golden set, then promote a statistically-proven champion. This guide is the how-to; `docs/evaluation.md` is the methodology.

**Typical full loop (~4–6 hours on Modal)**

```text
data-pipeline run ──▶ run_3config_comparison ──▶ eval run ──▶ eval compare ──▶ promote.yml
     ~20K examples          3 × QLoRA (~3-4h)         F2P/P2P                 champion gate
```

---

## 1. Prerequisites & Environment

```bash
# Install all extras (GPU training + eval + serving extras)
uv sync --extra dev --extra training --extra eval --extra inference

# Auth (one time)
gcloud auth application-default login
modal setup
wandb login

# Environment (see README → Configuration)
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
export WANDB_API_KEY=...
```

**W&B layout** (used by every phase):

| Object | Name |
| ------ | ---- |
| Entity | `2571642-university-of-dundee` |
| Data project | `swe-qwen-data` (dataset artifacts: `dataset-raw:v8`, `validated:v8`, `cleaned:v8`, `train:v23`, `val:v18`, `test:v14`, `golden:v23`, `validation_errors:v2`) |
| Run project | `swe-qwen` (training runs, eval runs, model artifacts `model-qwen3-14b-{variant}`) |
| Registry | `eval-champion` (promoted variants) |

---

## 2. Build the Dataset

All stages are one Typer CLI. Data docs: [`docs/dataset.md`](docs/dataset.md).

```bash
# Full pipeline: ingest → validate → clean → split → golden → tokenize
python -m data_engineering.cli run \
  --run-id expanded-repos \
  --tokenize-model qwen3-14b \
  --tokenize-max-length 8192

# Re-run only from a checkpoint (e.g. after tweaking the cleaner)
python -m data_engineering.cli run --run-id expanded-repos --resume-from cleaned --stages split,golden,tokenize

# Inspect the effective config
python -m data_engineering.cli config
```

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--swe-bench-dir` | `data/swe_bench` | Raw SWE-bench JSONL source |
| `--max-issues` | `2000` | Cap issues ingested (0 = all) |
| `--max-patch-lines` | `500` | Reject degenerate patches |
| `--resume-from` | — | `validated` \| `cleaned` — skip completed stages |
| `--stages` | all | Comma list, e.g. `split,golden` |
| `--train-ratio` / `--val-ratio` / `--test-ratio` | 0.8 / 0.1 / 0.1 | Split proportions (by repo) |
| `--tokenize-model` | `qwen3-14b` | Tokenizer for the train split |
| `--tokenize-max-length` | `4096` | Max sequence length (8,192 used for training run) |
| `--augment-codecontests` / `--augment-codealpaca` | off | Optional external augmentation |

The run prints a JSON summary: `{run_id, manifest_hash, stats, gcs_paths, wandb_artifacts}`. Remember the `run_id` — it feeds eval and training.

---

## 3. Train QLoRA Variants

### 3.1 Config is YAML, not code

`config/qlora_variants.yaml` is the single source of truth. Add a variant there and both training and the comparison script pick it up:

```yaml
baseline_14b:
  lora: {r: 16, alpha: 32, dropout: 0.0}
  training:
    lr: 2.0e-5           # 1 epoch, cosine, wd 0.01
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 8   # effective batch 16
    max_grad_norm: 1.0
    bf16: true
    gradient_checkpointing: true
    max_seq_length: 4096
    packing: true
```

### 3.2 Direct training on Modal

```bash
# Train one variant on A100-80GB (timeout 5h, retries 1)
modal run training/modal_train.py::train_qlora \
  --model-name qwen3-14b --variant baseline_14b \
  --run-id expanded-repos --max-train-samples 5000
```

What happens under the hood: Modal builds a `python:3.11-slim` image (torch 2.11.0 cu126 + transformers + peft + trl + Unsloth + flash-attn 2.8.3), mounts the shared models volume, downloads the `tokenized/{run_id}` split from the **public** GCS bucket, trains 1 epoch, saves checkpoints every 500 steps (keep ≤3), and logs to W&B. Returns `{status, wandb_run_id, artifact_name}`.

> **Why public GCS instead of a Modal volume mount?** The GCP org blocks service-account HMAC keys (`iam.disableServiceAccountKeyCreation`), so Modal can't use `CloudBucketMount`. The bucket's objects are public-read in dev; training fetches via stdlib urllib from the GCS JSON API. Production moves this behind a signed URL.

### 3.3 Train + compare all three variants

```bash
export EVAL_DATASET_RUN_ID=expanded-repos
python scripts/run_3config_comparison.py --run-id expanded-repos --max-train-samples 3000
# Force full retrain (ignore cache):
python scripts/run_3config_comparison.py --run-id expanded-repos --force-retrain --max-train-samples 3000
```

### 3.4 Resume / thin checkpoints

```bash
python -m training.qlora_train --model-name qwen3-14b --variant baseline_14b \
  --resume data/tokenized/expanded-repos/  --wandb-project swe-qwen
```

Checkpoint hygiene: `save_strategy` keeps the last 3; use `training/resume.py` to locate the newest adapter for serving.

---

## 4. Evaluate

Evaluation is execution-based — see [`docs/evaluation.md`](docs/evaluation.md) for the methodology. All commands come from `evaluation/` CLI (`python -m evaluation.cli`).

```bash
export EVAL_DATASET_RUN_ID=expanded-repos

# 1. Baseline (base Qwen3-14B, no LoRA)
python -m evaluation.cli run --split golden --sample 100 \
  --models qwen3-14b:baseline --resume run_baseline

# 2. Fine-tuned variants, same split & seed
python -m evaluation.cli run --split golden --sample 100 \
  --models qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b \
  --resume run_golden

# 3. Compare + optionally promote winner
python -m evaluation.cli compare --run_ids run_baseline,run_golden
```

### Eval tiers

| `--mode` | Sample | Split | Typical cost |
| -------- | ------ | ----- | ------------ |
| `smoke` | 20 | `swebench_verified` | ~$0.15–0.30 (CI gate) |
| `dev` | 100 | `swebench_verified` | — |
| `final` | 500 | `swebench_verified` | — |
| `full` | 50 | `golden` (or `--sample` for a sub-sample) | released run: 100-instance sample of the 2,313-pool, **$30** for 4 variants compare |

`--sample` overrides; `--ci-mode` runs the CI F2P gate; `--update-baseline` writes the CD-owned `smoke_baseline.json` (main-only). `--backend modal` (default) or `--backend local` with `--ollama-model qwen2.5-coder:7b` for offline debugging.

### Cost controls

- `--sample 100` full 4-variant compare ≈ **$30** (released run); smaller samples scale roughly linearly.
- Cold starts dominate the first run — the EvalConfig caches repos (`eval-repo-cache`) and test images (`eval-test-cache`) on Modal volumes.
- Historical: `results.txt` shows the 2-run × 4-model comparison at **$30 total** (est.) — the harness is cheap per example (~$0.05 per instance per model).

---

## 5. Promote a Champion

### 5.1 Manual promotion (via the CLI gate)

`eval compare --run_ids run_baseline,run_golden --promote-to-registry` writes the winner to the W&B Registry `eval-champion` collection with a full decision record.

### 5.2 CI promotion (recommended)

```bash
# Paired challenger-vs-champion eval on the same dev subset
gh workflow run promote.yml -f candidate_variant=higher_rank_14b

# $0 dry run — skip Modal eval, gate on the latest logged numbers only
gh workflow run promote.yml -f candidate_variant=higher_rank_14b \
  # (set repo variable RUN_MODAL_EVAL=false)
```

The gate (in `promotion/gate.py`):

1. **F2P absolute floor**: `min_f2p_threshold = 0.15` (15%)
2. **Relative gain**: champion must exceed challenger's F2P by **≥5 points** (or beat on all paired metrics)
3. **CI lower bound > 0**: Wilson 95% CI of the F2P gain must not include zero (McNemar significance)
4. **P2P safeguard**: no more than **2 points** Pass-to-Pass regression vs champion

All pass → decision recorded (W&B + `promotion/audit.py`), champion updated, then deployment behind the `production` GitHub Environment approval.

---

## 6. Serve the Champion

```bash
modal deploy inference.modal_serve
curl -s https://<workspace>.modal.run/v1/chat/completions \
  -H "Authorization: Bearer $MODAL_SERVE_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b:baseline_14b","messages":[{"role":"user","content":"Write a prime sieve."}]}'
```

See [`docs/api.md`](docs/api.md) for the full wire format.

---

## 7. Recipe: Onboarding a New Base Model

1. Add the model to `config/models.yaml` (hf_id, quantization, serving_hf_id, gpu, context_window, target_modules).
2. Re-tokenize: `python -m data_engineering.cli tokenize --run-id <id> --model-name my-model --max-seq-length 8192`.
3. Train variants per §3, evaluate per §4, promote per §5.
4. Update `serve.py`'s engine if quantization differs (AWQ vs GPTQ vs none).

No code changes elsewhere — the platform is the product.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
| ------- | ----------- |
| Modal eval 500s / aiohttp errors | `max_parallel` in EvalConfig: 64 broke Modal 1.5.3 — keep ≤16 |
| Patched outputs truncated mid-diff | Increased context budget: 2048 tokens too small once reasoning is removed — `max_new_tokens` 8192 default |
| Training eval OOM on A10G | By design: `eval_strategy='no'`; evaluate separately |
| Terraform auth failure | `iam.disableServiceAccountKeyCreation` blocks keys — use WIF, never `GCP_SA_KEY` |
| Stub rather than real model on `uvicorn` | `SERVING_STUB=0` for vLLM; default is the deterministic stub |