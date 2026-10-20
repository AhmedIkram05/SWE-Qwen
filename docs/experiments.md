# Experiments Guide — Train, Evaluate, Promote

> SWE-Qwen's experiment loop is **one command per phase**: build the dataset, train QLoRA variants on serverless A100s, evaluate them execution-style against the golden set, then promote a statistically-proven champion. This guide is the how-to; `docs/evaluation.md` is the methodology. Five QLoRA variants are configured in `config/qlora_variants.yaml`; three have been trained + evaluated so far.

**Typical full loop**

```text
data-pipeline run ──▶ run_3config_comparison ──▶ eval run ──▶ eval compare ──▶ promote.yml
     ~20K examples          3 × QLoRA (~1.2 h each)      F2P/P2P                 champion gate
```

A recorded 1-epoch 14B variant run (champion `higher_rank_14b`) took **4,214 s (~1.2 h)**, well under the 5 h Modal timeout.

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

All stages are one Typer CLI — `python -m data_engineering.cli` (console script `data-pipeline`) with three subcommands: `run`, `tokenize`, `config`. Data docs: [`docs/dataset.md`](dataset.md).

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
| `--output` / `-o` | `data/` | Output directory for pipeline artifacts |
| `--max-patch-lines` | `500` | Reject degenerate patches |
| `--min-golden` | `100` (from config) | Minimum golden examples target |
| `--resume-from` | — | `validated` \| `cleaned` — skip completed stages |
| `--stages` | all | Comma list, e.g. `split,golden` |
| `--run-id` / `--run-name` | auto | Run ID override / descriptive W&B run name |
| `--train-ratio` / `--val-ratio` / `--test-ratio` | 0.8 / 0.1 / 0.1 | Split proportions (by repo) |
| `--bigquery` / `--no-bigquery` | off | BigQuery repo-stat enrichment |
| `--augment-codecontests` / `--augment-codealpaca` | off | Optional external augmentation (train split only) |
| `--max-train-examples` | `30000` | Cap total training size after augmentation |
| `--tokenize-model` | registry default | Tokenizer model for the tokenize stage |
| `--tokenize-max-length` | `4096` | `min(registry context_window, trainer max_seq_length)`; the recorded run passed `8192` — ≤ the A100-80GB train window, so nothing is truncated |

The run prints a JSON summary: `{run_id, manifest_hash, stats, gcs_paths, wandb_artifacts, tokenized_paths}` and writes `manifest.json` + `dataset_card.md` alongside the stage files. Remember the `run_id` — it feeds eval and training.

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
    packing: false       # packing concatenates examples across the -100 prompt boundaries and corrupts the loss mask
```

Variants only override what they specify: `training/qlora_config.py` deep-merges `baseline_14b` under every variant at load time (there is no `inherits:` key in the YAML).

**Configured variants** — 5 in `config/qlora_variants.yaml`, 3 trained + evaluated:

| Variant | LoRA | LR | Epochs | Status |
| ------- | ---- | -- | ------ | ------ |
| `baseline_14b` | r16/α32 | 2e-5 | 1 | evaluated |
| `higher_rank_14b` | r32/α64 | 2e-5 | 1 | evaluated — **champion** |
| `higher_lr_14b` | r16/α32 | 5e-5 | 1 | evaluated |
| `efficient_14b` | r8/α16 | 5e-5 | 2 | quick-debugging tier |
| `strong_14b` | r64/α128 | 1e-4 | 3 | staged production candidate — no eval results yet |

### 3.2 Direct training on Modal

```bash
# Train one variant on A100-80GB (timeout 5h, retries 1)
modal run training/modal_train.py::train_qlora \
  --model-name qwen3-14b --variant baseline_14b \
  --run-id expanded-repos --max-train-samples 5000
```

What happens under the hood: Modal builds the image as code — `modal.Image.debian_slim(python_version="3.11")`, no Dockerfile — with torch 2.11 (cu126), `transformers>=5.5,<5.6`, peft/trl/bitsandbytes, `unsloth[colab-new]` from git, the flash-attn 2.8.3 cu126 wheel, and `xformers>=0.0.29`. The job runs with the `wandb-secret` + `hf-secret` Modal secrets, mounts the shared `swe-qwen-models` volume, downloads the `tokenized/{run_id}` split from the **public** GCS bucket, trains 1 epoch, saves a checkpoint every **10 steps** (`save_steps: 10`, keep last 3 — a 1-epoch 14B run totals ~16–76 steps, so the old 500-step interval produced zero intermediate checkpoints), and logs to W&B. Returns `{status, wandb_run_id, artifact_name}`.

**GPU reality:** on Modal (A100-80GB) the GPU override adapts the YAML's base shape to the hardware — `max_seq_length` 8,192 and batch 8×2 (effective 16, same product as the YAML's 2×8). Tokenize/train alignment is intentional: the pipeline tokenizes at 4,096 by default (`min(registry context_window, trainer max_seq_length)`) while the A100-80GB trainer window is 8,192, so tokenized data is never longer than the training window (pinned by `tests/test_contract_config.py`; the recorded run tokenized at 8,192).

Recorded champion run (`higher_rank_14b`): **4,214 s (~1.2 h)**, 1 epoch, final loss **0.5843** — well under the 5 h (18,000 s) Modal timeout.

> **Why public GCS instead of a Modal volume mount?** The GCP org blocks service-account HMAC keys (`iam.disableServiceAccountKeyCreation`), so Modal can't use `CloudBucketMount`. The bucket's objects are public-read in dev; training fetches via stdlib urllib from the GCS JSON API. Production moves this behind a signed URL.

### 3.3 Train + compare all three variants

```bash
export EVAL_DATASET_RUN_ID=expanded-repos
python scripts/run_3config_comparison.py --run-id expanded-repos --max-train-samples 3000
# Force full retrain (ignore cache):
python scripts/run_3config_comparison.py --run-id expanded-repos --force-retrain --max-train-samples 3000
```

`run_3config_comparison.py` is resumable — state persists in `scripts/.pipeline-state.json` — reconciles with W&B before reusing cached runs, supports `--dry-run`, and tags the winner `challenger` → `champion` via W&B before dispatching `promote.yml`.

> **`scripts/f2p_proxy.py` is DRY_RUN_ONLY.** The module hard-codes `DRY_RUN_ONLY = True` and every score dict carries `"dry_run": True`. Its proxy ranking can invert relative to true F2P and must never drive promotion.

### 3.4 Resume / thin checkpoints

```bash
python -m training.qlora_train --model-name qwen3-14b --variant baseline_14b \
  --resume data/expanded-repos_tokenized/  --wandb-project swe-qwen
```

Checkpoint hygiene: `save_strategy` keeps the last 3; use `training/resume.py` to locate the newest adapter for serving.

---

## 4. Evaluate

Evaluation is execution-based — see [`docs/evaluation.md`](evaluation.md) for the methodology. All commands come from `evaluation/` CLI (`python -m evaluation.cli`).

```bash
export EVAL_DATASET_RUN_ID=expanded-repos

# 1. Baseline (base Qwen3-14B, no LoRA)
python -m evaluation.cli run --split golden --sample 100 \
  --models qwen3-14b:baseline --resume run_baseline

# 2. Fine-tuned variants, same split & seed
python -m evaluation.cli run --split golden --sample 100 \
  --models qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b \
  --resume run_golden

# 3. Compare (promotes the revalidated winner best-effort)
python -m evaluation.cli compare --run_ids run_baseline,run_golden
```

### Eval tiers

| `--mode` | Sample | Split |
| -------- | ------ | ----- |
| `smoke` | 20 | `swebench_verified` |
| `dev` | 100 | `swebench_verified` |
| `final` | 500 | `swebench_verified` |
| `full` | 5,000 | `golden` (or `--sample` for a sub-sample) |

Tier subsets are deterministic (`tier_seed 42`). Cost is recorded per `EvalRun` — `cost_usd` = inference GPU-min at $0.0417/min + test vCPU at 2 × $0.008/hr — so treat any recorded total as run-specific, not a reproducible flat rate.

`--sample` overrides the tier size. **`--ci-mode` only sets the sample size** — the actual CI F2P gate runs with `--mode smoke` (20-instance smoke tier, 5% absolute F2P-drop tolerance vs the stored baseline). `--update-baseline` writes the CD-owned `smoke_baseline.json` (main-only). `--backend modal` (default) or `--backend local` with `--ollama-model qwen2.5-coder:7b` for offline debugging. Tuning knobs such as `max_parallel` and `max_new_tokens` are `EvalConfig` fields set via `EVAL_*` env vars — there are no CLI flags for them.

### Cost controls

- Cost is recorded per `EvalRun` (`cost_usd` = inference GPU-min at $0.0417/min + test vCPU at 2 × $0.008/hr); smaller samples scale down accordingly.
- Cold starts dominate the first run — the EvalConfig caches repos (`eval-repo-cache`) and test images (`eval-test-cache`) on Modal volumes.
- Historical: `results.txt` records the 2-run × 4-model comparison — the harness is cheap per example (~$0.05 per instance per model).

---

## 5. Promote a Champion

### 5.1 Manual promotion (via the CLI gate)

`python -m evaluation.cli compare --run_ids run_baseline,run_golden` promotes best-effort on **every** compare run: it revalidates the best run against the gates (`revalidate_champion`) and, on success, writes it to the W&B Registry `eval-champion` collection with a full decision record (`promote_champion_to_registry`). There is no `--promote-to-registry` flag — the promotion step is built into `compare` and runs best-effort unconditionally.

### 5.2 CI promotion (recommended)

```bash
# Paired challenger-vs-champion eval on the same dev subset
gh workflow run promote.yml -f candidate_variant=higher_rank_14b

gh workflow run promote.yml -f candidate_variant=higher_rank_14b \
  # (set repo variable RUN_MODAL_EVAL=false)
```

The gate (in `promotion/gate.py`):

1. **F2P absolute floor**: `min_f2p_threshold = 0.15` (15%)
2. **Relative gain**: the challenger must beat the champion's F2P by **≥5 points** (`PROMOTE_MIN_F2P_GAIN = 0.05`; all four gates must pass — there is no alternate path)
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

See [`docs/api.md`](api.md) for the full wire format.

---

## 7. Recipe: Onboarding a New Base Model

1. Add the model to `config/models.yaml` (hf_id, quantization, serving_hf_id, gpu, context_window, target_modules).
2. Re-tokenize: `python -m data_engineering.cli tokenize --run-id <id> --model-name my-model --max-seq-length 8192`.
3. Train variants per §3, evaluate per §4, promote per §5.
4. Serve — deployment is registry-driven: `serving_hf_id` + `serving_quantization` per model in `config/models.yaml` (AWQ int4 `Qwen/Qwen3-14B-AWQ` for `qwen3-14b`). No code edits needed to swap models.

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
