# Phase 4 Implementation Plan: QLoRA Training Pipeline

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Draft v1.2 (post-pivot: 14B-only)
**Parent Document:** `docs/planning/MASTER-PLAN.md`
**Dependencies:** Phase 1 complete (Modal, W&B, GCS infra), Phase 3 complete (JSONL dataset splits in `data/`)

> **PIVOT NOTE (v1.2):** Phase 4 now uses **ONLY qwen3-14b** on A10G 24GB. qwen3-30b-a3b is excluded from Phase 4 execution due to cost (~$110-160 for 3-config on H100). The 30B config is retained in `models.yaml` for future phases but marked `phase4_excluded: true`. All 3 mandatory comparison variants are redesigned for 14B (baseline_14b, higher_rank_14b, higher_lr_14b). Cost reduction: ~90% ($10-17 total vs $110-160).
>
> **Deviation Note:** This plan uses `training/` at repo root (matching Master Plan Appendix D), not `src/training/`. Existing `src/swe_qwen/modal_app.py` (Phase 1 skeleton) will be superseded by the modular pipeline; `modal_app.py.train_swe_qwen` serves as reference for smoke testing. See Notes section for full deviation log.

## Overview
Build a complete QLoRA fine-tuning pipeline for **Qwen3-14B (primary, Modal A10G 24GB)** using 4-bit NF4 quantization. Includes model config registry, QLoRA factory, prompt templates, class-based trainer, Modal GPU dispatch, W&B checkpoint callbacks, hybrid resume, and **3-config mandatory 14B-optimized comparison** with F2P proxy evaluation.

## Requirements (from MASTER-PLAN Phase 4 + Grilling Answers + Pivot)
- 4.1 Model config registry (models.yaml with GPU mapping) — Q1:C → **Updated: qwen3-14b primary, qwen3-30b-a3b retained but phase4_excluded**
- 4.2 QLoRA config factory `build_qlora_config(variant, model_name)` — Q2:C → **Updated: 14B-optimized variants**
- 4.3 Prompt engineering with Jinja2 + W&B Artifact versioning — Q3:A
- 4.4 Class-based `QLoRATrainer` — Q4:B
- 4.5 Modal train wrapper with dynamic GPU selection — Q5:C → **Updated: A10G 24GB primary**
- 4.6 W&B checkpoint callback with artifact logging — Q6:B
- 4.7 Hybrid resume (Modal volume → W&B artifact fallback) — Q7:C
- 4.8 Pre-tokenized .arrow shards from Phase 3 tokenize.py — Q8:B
- 4.9 Unit tests: config, trainer smoke (tiny model), full pipeline mock — Q10:C
- 4.10 Baseline training (100 examples)
- 4.11 Full training
- 4.12 **MANDATORY**: 3-config 14B-optimized QLoRA comparison → F2P on golden → Champion

## Architecture Changes

### New Files
| File | Purpose |
|------|---------|
| `config/models.yaml` | Model registry: name, hf_id, active_params, total_params, gpu_mapping, quantization (**updated: qwen3-14b primary, qwen3-30b-a3b phase4_excluded**) |
| `config/qlora_variants.yaml` | **3 mandatory 14B-optimized variants: baseline_14b, higher_rank_14b, higher_lr_14b** (deprecated 30B variants retained in file) |
| `training/qlora_config.py` | Factory `build_qlora_config(variant, model_name)` → `(LoraConfig, TrainingArguments)` |
| `training/prompts/` | Jinja2 templates: `system.j2`, `user.j2`, `assistant.j2`, `chat.j2` |
| `training/prompt_loader.py` | Load/render templates, log to W&B Artifacts |
| `training/qlora_trainer.py` | Class `QLoRATrainer`: model, data, tokenizer, callbacks, train(), resume() |
| `training/qlora_train.py` | Thin CLI wrapper: imports `QLoRATrainer`, parses CLI args, calls `train()`. Enables `python -m training.qlora_train` per Master Plan acceptance criteria |
| `training/callbacks.py` | `WandbCheckpointCallback` + `WandbLoggingCallback` |
| `training/resume.py` | Hybrid resume logic: local volume → W&B artifact |
| `data_engineering/tokenize.py` | **NEW**: Tokenize Phase 3 JSONL → .arrow shards (Phase 3 outputs JSONL) |
| `training/modal_train.py` | Modal entrypoint: dynamic GPU from models.yaml, variant param (**A10G 24GB primary**) |
| `training/conftest.py` | Shared fixtures: temp dirs, mock config, sample tokenized data |
| `tests/test_qlora_config.py` | Config factory tests |
| `tests/test_qlora_trainer_smoke.py` | Smoke test with tiny model (e.g., TinyLlama) |
| `tests/test_training_pipeline_mock.py` | Full pipeline mock test |
| `scripts/run_3config_comparison.py` | Orchestrates 3 14B variants → F2P eval → champion selection |

### Modified Files
| File | Change |
|------|--------|
| `pyproject.toml` | Add `jinja2>=3.1.0` to deps; add `[project.scripts]` entry point `train = "training.qlora_train:app"` (points to `training/qlora_train.py` CLI wrapper) |
| `data_engineering/tokenize.py` | Ensure outputs .arrow shards (already done in Phase 3) |
| `src/swe_qwen/modal_app.py` | **Deprecated** — existing `train_swe_qwen` function serves as reference only; Phase 4 replaces with modular `training/modal_train.py` + `QLoRATrainer` |
| `README.md` / docs | Update with Phase 4 usage |

---

## Implementation Steps

### Phase 4A: Config Registry & QLoRA Factory (Tasks 4.1, 4.2)

#### Step 0: Infrastructure Precondition Check
- **File**: N/A (verification only)
- **Action**: Verify Phase 1 infrastructure is operational before any code work
- **Checks**:
  1. Modal connectivity: `modal run src.swe_qwen.modal_app::hello_modal` returns `{"status": "ok"}`
  2. W&B connectivity: `wandb.init(project="swe-qwen")` succeeds
  3. GCS bucket accessible: `gsutil ls gs://{bucket}/` succeeds (from Terraform output)
  4. GPU quota: Modal can provision A100 40GB (check `modal quota`)
  5. Secrets exist: `modal secret list` shows `wandb-api-key`, `hf-secret`, `github-token`
- **Dependencies**: Phase 1 complete
- **Risk**: Low — fast check, fails fast if infra missing

#### Step 1: Create `config/models.yaml`
- **File**: `config/models.yaml`
- **Action**: Define model registry with GPU mapping (**PIVOT: qwen3-14b primary, qwen3-30b-a3b phase4_excluded**)
- **Schema**:
```yaml
models:
  qwen3-14b:
    hf_id: "Qwen/Qwen3-14B"
    active_params: 14_000_000_000
    total_params: 14_000_000_000
    quantization: "nf4"
    compute_dtype: "bfloat16"
    gpu_mapping:
      primary: "a10g-24gb"
      fallback: "a100-40gb"
    context_window: 32768
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

  # Retained for future phases (Phase 5+ re-evaluation, Phase 6+ serving)
  # NOT used in Phase 4 training pipeline
  qwen3-30b-a3b:
    hf_id: "Qwen/Qwen3-30B-A3B"
    active_params: 3_000_000_000
    total_params: 30_000_000_000
    quantization: "nf4"
    compute_dtype: "bfloat16"
    gpu_mapping:
      primary: "h100-80gb"
      fallback: "a100-80gb"
    context_window: 32768
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    phase4_excluded: true
```
- **Dependencies**: None
- **Risk**: Low — static config

#### Step 2: Create `config/qlora_variants.yaml`
- **File**: `config/qlora_variants.yaml`
- **Action**: Define 3 mandatory 14B-optimized comparison variants (**PIVOT: baseline_14b, higher_rank_14b, higher_lr_14b**)
- **Schema**:
```yaml
variants:
  baseline_14b:
    lora:
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules: null  # resolved from model config at runtime
      bias: "none"
      task_type: "CAUSAL_LM"
    training:
      learning_rate: 2.0e-5
      num_train_epochs: 3
      per_device_train_batch_size: 2
      gradient_accumulation_steps: 8
      warmup_ratio: 0.03
      lr_scheduler_type: "cosine"
      weight_decay: 0.01
      max_grad_norm: 1.0
      fp16: false
      bf16: true
      optim: "paged_adamw_8bit"
      logging_steps: 10
      save_strategy: "steps"
      save_steps: 500
      eval_strategy: "steps"
      eval_steps: 500
      save_total_limit: 3
      load_best_model_at_end: true
      metric_for_best_model: "eval_loss"
      greater_is_better: false
      report_to: "wandb"
      ddp_find_unused_parameters: false
      gradient_checkpointing: true
      dataloader_num_workers: 2
      remove_unused_columns: false
      packing: true
      max_seq_length: 8192
      dataloader_pin_memory: false

  higher_rank_14b:
    lora:
      r: 32
      lora_alpha: 64
      lora_dropout: 0.05
      # target_modules, bias, task_type inherit from baseline_14b
    training:
      learning_rate: 2.0e-5
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 16
      # all other training args inherit from baseline_14b

  higher_lr_14b:
    lora:
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      # target_modules, bias, task_type inherit from baseline_14b
    training:
      learning_rate: 5.0e-5
      # all other training args inherit from baseline_14b

  efficient_14b:
    lora:
      r: 8
      lora_alpha: 16
      lora_dropout: 0.05
      # target_modules, bias, task_type inherit from baseline_14b
    training:
      learning_rate: 5.0e-5
      per_device_train_batch_size: 2
      gradient_accumulation_steps: 8
      max_seq_length: 8192
      num_train_epochs: 2
      # inherits: warmup_ratio, lr_scheduler_type, weight_decay, max_grad_norm, bf16, optim, logging_steps, save_strategy, save_steps, eval_strategy, eval_steps, save_total_limit, load_best_model_at_end, metric_for_best_model, greater_is_better, report_to, ddp_find_unused_parameters, gradient_checkpointing, dataloader_num_workers, remove_unused_columns, packing, dataloader_pin_memory
```
- **Dependencies**: Step 1 (target_modules reference)
- **Risk**: Low — static config
- **Dependencies**: Step 1 (target_modules reference)
- **Risk**: Low — static config

#### Step 3: Implement `training/qlora_config.py`
- **File**: `training/qlora_config.py`
- **Action**: Factory function `build_qlora_config(variant: str, model_name: str) -> tuple[LoraConfig, TrainingArguments]`
- **Details**:
  - Load `models.yaml` and `qlora_variants.yaml`
  - Merge variant config with model-specific `target_modules`
  - Return instantiated `LoraConfig` and `TrainingArguments`
  - Validate variant exists, model exists
  - Type hints throughout
- **Dependencies**: Step 1, Step 2
- **Risk**: Medium — merging logic must handle inheritance correctly

---

### Phase 4B: Prompt Engineering (Task 4.3)

#### Step 4: Create Jinja2 Prompt Templates
- **Dir**: `training/prompts/`
- **Files**:
  - `system.j2`: System prompt with `{{ task_description }}`, `{{ language }}`, `{{ style_guide }}`
  - `user.j2`: User prompt with `{{ issue_title }}`, `{{ issue_body }}`, `{{ context_files }}`, `{{ test_files }}`
  - `assistant.j2`: Assistant response format with `{{ analysis }}`, `{{ plan }}`, `{{ code_changes }}`
  - `chat.j2`: Chat template combining above with `{% for message in messages %}` loop
- **Dependencies**: None
- **Risk**: Low — template design

#### Step 5: Implement `training/prompt_loader.py`
- **File**: `training/prompt_loader.py`
- **Action**:
  - `PromptLoader` class: load templates from `prompts/` dir
  - `render(template_name: str, **kwargs) -> str`
  - `log_to_wandb_artifact(run_id: str, version: str)` — upload rendered templates as W&B Artifact type `prompt_template`
  - `load_from_wandb_artifact(run_id: str, version: str) -> PromptLoader` — restore for reproducibility
- **Dependencies**: Step 4
- **Risk**: Medium — W&B artifact API

> **Note on Prompt A/B Testing**: MASTER-PLAN 4.3 specifies "A/B test 2-3 variants in Phase 5 eval". Phase 4 delivers the template infrastructure (Jinja2 + W&B Artifact versioning) to enable this. The A/B runner (`prompt_ab_test.py`) is explicitly deferred to **Phase 5** (Evaluation Harness) where it will run controlled experiments against the golden eval set.

---

### Phase 4C: Tokenized Data Loading (Task 4.8)

#### Step 6: Implement `data_engineering/tokenize.py` (NEW — Phase 3 outputs JSONL, not .arrow)
- **File**: `data_engineering/tokenize.py` (new module, Phase 3 did not include tokenization)
- **Action**: Read Phase 3 JSONL splits (`train.jsonl`, `val.jsonl`, `test.jsonl`, `golden.jsonl`), tokenize with model tokenizer, save as HuggingFace `.arrow` shards
- **Details**:
  - Load `IssueRecord` from JSONL
  - Render prompt using `PromptLoader` (Step 5) → `chat.j2` template with issue + context
  - Tokenize: `tokenizer(prompt, truncation=True, max_length=context_window, padding=False)`
  - Create `DatasetDict` with `train`/`val`/`test`/`golden` splits
  - Save via `dataset.save_to_disk(output_dir)` → produces `.arrow` shards + `dataset_dict.json`
  - **Columns**: `input_ids`, `attention_mask`, `labels` (labels = input_ids for causal LM)
  - **Helper**: `load_tokenized_shards(data_dir: str) -> DatasetDict`
- **Dependencies**: Steps 1, 3, 5 (model config for context_window, qlora_config for tokenizer, prompt_loader for rendering)
- **Risk**: Medium — tokenization must match training format exactly; max_length from `models.yaml.context_window`

---

### Phase 4D: QLoRA Trainer Class (Task 4.4)

#### Step 7: Implement `training/qlora_trainer.py`
- **File**: `training/qlora_trainer.py`
- **Action**: Class `QLoRATrainer`
- **Signature**:
```python
class QLoRATrainer:
    def __init__(
        self,
        model_name: str,  # key from models.yaml
        variant: str,  # key from qlora_variants.yaml
        data_dir: str,  # path to tokenized .arrow shards
        output_dir: str,  # local output dir (Modal volume)
        wandb_project: str,
        wandb_entity: str,
        run_name: str | None = None,
        resume_from_checkpoint: str | None = None,
    ): ...
```
- **Methods**:
  - `setup_model_and_tokenizer()` — 4-bit NF4 via `BitsAndBytesConfig`, `AutoModelForCausalLM.from_pretrained`, `get_peft_model`
  - `setup_data()` — load tokenized shards, create `DataLoader`
  - `setup_callbacks()` — instantiate `WandbCheckpointCallback`, `WandbLoggingCallback`
  - `train()` — `trainer.train(resume_from_checkpoint=...)`
  - `resume(checkpoint_path: str)` — delegate to `resume_training()`
  - `save_model()` — save adapter + tokenizer
- **Internal**: Uses `trl.SFTTrainer` (committed — refactors existing `src/swe_qwen/modal_app.py::train_swe_qwen` SFTTrainer usage into proper class). SFTTrainer handles data collation, formatting, and packing internally. No custom `Trainer` subclass needed for V1.
- **Dependencies**: Steps 1, 2, 3, 5, 6 (model config, qlora config, prompt loader, tokenization)
- **Risk**: High — core training logic, model loading, PEFT integration

#### Step 8: Implement `training/callbacks.py`
- **File**: `training/callbacks.py`
- **Classes**:
  - `WandbCheckpointCallback(TrainerCallback)`:
    - `on_save`: upload checkpoint dir as W&B Artifact type `model_checkpoint`, metadata: `step`, `epoch`, `eval_loss`, `variant`, `model_name`
    - Artifact name: `{{model_name}}-{{variant}}-checkpoint-{{step}}`
  - `WandbLoggingCallback(TrainerCallback)`:
    - `on_log`: log metrics to W&B
    - `on_train_begin`: log config (model, variant, prompt version)
- **Dependencies**: Step 7
- **Risk**: Medium — W&B artifact API, callback timing

---

### Phase 4E: Hybrid Resume (Task 4.7)

#### Step 9: Implement `training/resume.py`
- **File**: `training/resume.py`
- **Action**: `resolve_checkpoint_path(resume_spec: str, local_volume_path: Path, wandb_run_path: str) -> str`
- **Logic**:
  1. If `resume_spec` is local path and exists → return it
  2. If `resume_spec` is W&B artifact ref (`entity/project/artifact:vN`) → download via `wandb.Api().artifact().download()` to local volume → return local path
  3. If `resume_spec == "latest"` → query W&B for latest `model_checkpoint` artifact for this run → download → return
  4. Raise if not found
- **Dependencies**: Step 8 (artifact naming convention)
- **Risk**: Medium — W&B API, network fallback

---

### Phase 4F: Modal Training Wrapper (Task 4.5)

> **Existing modal_app.py:** `src/swe_qwen/modal_app.py` has a `train_swe_qwen` function (Phase 1 skeleton). This step supersedes it. `modal_app.py.train_swe_qwen` will be kept as a reference/smoke test helper but is **not** the production path. Phase 4's `training/modal_train.py` is the canonical entry point and delegates to `QLoRATrainer`.

#### Step 10: Create `training/modal_train.py`
- **File**: `training/modal_train.py` (Modal entrypoint; supersedes `src/swe_qwen/modal_app.py`)
- **Action**: Modal function `train_qlora` with dynamic GPU selection
- **Signature**:
```python
@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu=GPU_CONFIG,  # resolved at call time
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def train_qlora(
    model_name: str = "qwen3-14b",
    variant: str = "baseline_14b",
    data_volume: str = "training-data",
    run_name: str | None = None,
    resume: str | None = None,
): ...
```
- **GPU Resolution**: Read `models.yaml`, pick `gpu_mapping.primary` (or fallback if OOM), map to Modal GPU string via **GPU Mapping Table**:

| `models.yaml` value | Modal GPU Spec |
|---------------------|----------------|
| `a100-40gb` | `modal.gpu.A100(count=1, size="40GB")` |
| `a10g-24gb` | `modal.gpu.A10G(count=1)` |
| `a100-80gb` | `modal.gpu.A100(count=1, size="80GB")` |
| `h100-80gb` | `modal.gpu.H100(count=1)` |

- **Volume Mount**: `/vol` for checkpoints, `/data` for tokenized shards
- **Delegation**: Instantiate `QLoRATrainer`, call `train()`
- **Dependencies**: Steps 1, 7, 9
- **Risk**: High — Modal GPU dynamics, volume mounting, secrets

---

### Phase 4G: Testing (Task 4.9)

#### Step 11: `tests/test_qlora_config.py`
- **Tests**:
  - `test_build_baseline_14b_config()` — returns correct LoraConfig + TrainingArguments
  - `test_build_higher_rank_14b_config()` — r=32, alpha=64
  - `test_build_higher_lr_14b_config()` — lr=5e-5
  - `test_invalid_variant_raises()`
  - `test_invalid_model_raises()`
  - `test_target_modules_merged_from_model_config()`
- **Dependencies**: Step 3
- **Risk**: Low

#### Step 11b: `tests/training/conftest.py`
- **Content**: Shared fixtures for training tests
  - `tmp_output_dir` — temp directory for checkpoints
  - `mock_config` — pre-loaded `models.yaml` + `qlora_variants.yaml` data
  - `sample_tokenized_data` — small pre-tokenized dataset dict for smoke tests
  - `mock_wandb_run` — patched W&B run for callback tests
- **Dependencies**: Step 3, Step 6
- **Risk**: Low

#### Step 12: `tests/test_qlora_trainer_smoke.py`
- **Tests**:
  - `test_trainer_init()` — instantiates without error
  - `test_setup_model_tokenizer_tiny()` — use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (or `hf-internal-testing/tiny-random-LlamaForCausalLM`), 4-bit load, PEFT wrap
  - `test_one_train_step()` — single batch forward/backward
- **Dependencies**: Step 7
- **Risk**: Medium — requires GPU (Modal test or CI GPU runner)

#### Step 13: `tests/test_training_pipeline_mock.py`
- **Tests**:
  - `test_full_pipeline_mock()` — mock `QLoRATrainer.train()`, verify callbacks called, artifacts logged
  - `test_resume_local()` — local checkpoint path resolves
  - `test_resume_wandb_fallback()` — mock W&B artifact download
- **Dependencies**: Steps 7, 8, 9
- **Risk**: Low — mocked

---

### Phase 4H: 3-Config Comparison & Champion Selection (Task 4.12)nn#### Step 14: Create `scripts/run_3config_comparison.py`n- **File**: `scripts/run_3config_comparison.py`n- **Action**: Orchestration script (**PIVOT: 14B-optimized variants**)n- **Flow**:n  1. For each variant in `[baseline_14b, higher_rank_14b, higher_lr_14b]`:n     - Launch `modal_train.py` via `modal.run()` or subprocessn     - Wait for completion, capture W&B run IDn  2. For each completed run:n     - Download adapter from W&B artifactn     - Run F2P evaluation on `golden.jsonl` (Phase 3 proxy: test file overlap + fix keywords)n     - Compute F2P scoren  3. Select champion: highest F2Pn  4. Promote champion adapter to W&B Registry as `champion` aliasn  5. Output summary JSON with scores, winnern- **Dependencies**: Steps 7, 10, Phase 3 `golden.jsonl`, `evaluation/f2p_proxy.py` (new or reuse Phase 3)n- **Risk**: High u2014 orchestration, multiple Modal runs, evaluation logicn- **Post-Phase-5 Re-evaluation**: Champion selected by proxy F2P must be re-evaluated with full Phase 5 evaluation harness (`evaluation/harness.py`). Step 14 champion promotion to W&B Registry alias `champion` is provisional until Phase 5 validates. See Notes.n- **Sequencing Note**: Master Plan 4.12 assigns golden F2P evaluation to Phase 5; proxy F2P here enables Phase 4 champion selection. Full F2P re-evaluation in Phase 5 supersedes proxy results.
---

### Phase 4I: Training Execution (Tasks 4.10, 4.11)

#### Step 15: Baseline Training Run (100 examples)
- **Action**: Run `modal_train.py` with `variant=baseline`, `data_dir` pointing to 100-example subset
- **Verify**: Loss decreases, checkpoints saved to W&B, no OOM
- **Dependencies**: Step 10
- **Risk**: Medium — first real GPU run

#### Step 16: Full Training Run
- **Action**: Run all 3 variants on full dataset via `run_3config_comparison.py`
- **Verify**: All 3 complete, F2P scores computed, champion promoted
- **Dependencies**: Step 14, Step 15
- **Risk**: High — long-running, GPU costs, potential OOM

---

## Testing Strategy

| Test File | Scope | Type | Dependencies |
|-----------|-------|------|--------------|
| `test_qlora_config.py` | Config factory | Unit | Step 3 |
| `test_qlora_trainer_smoke.py` | Trainer init, model load, 1 step | Integration (GPU) | Step 7 |
| `test_training_pipeline_mock.py` | Callbacks, resume, artifact flow | Unit (mocked) | Steps 7-9 |
| `run_3config_comparison.py` dry-run | Orchestration logic | Integration (mocked Modal) | Step 14 |

**Run commands**:
```bash
# Unit tests (no GPU)
pytest tests/test_qlora_config.py tests/test_training_pipeline_mock.py -v

# Smoke test (requires GPU - run on Modal)
modal run tests/test_qlora_trainer_smoke.py

# Full comparison (manual trigger)
python scripts/run_3config_comparison.py
```

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OOM on A10G 24GB with Qwen3-14B 4-bit | Low | High | Gradient checkpointing, batch_size=2, grad_accum=8, max_seq_length=8192, fallback to A100-40GB |
| PEFT + bitsandbytes version incompatibility | Low | High | Pin versions in pyproject.toml, test smoke first |
| W&B artifact download fails on resume | Medium | Medium | Retry logic, local volume primary, clear error messages |
| Modal GPU type string mapping incorrect | Low | High | Map in models.yaml, test GPU detection in modal_train.py |
| F2P proxy doesn't correlate with real quality | Medium | Medium | Phase 5 adds real eval; champion selected in Step 14 is **provisional** — Phase 5 must re-evaluate and confirm |
| Long training time (>8h per variant) | Medium | Medium | Checkpoint every 500 steps, resume support, early stopping on eval_loss plateau |

---

## Success Criteria

- [ ] `config/models.yaml` and `config/qlora_variants.yaml` created and validated
- [ ] `build_qlora_config(variant, model_name)` returns correct configs for all 3 variants
- [ ] Prompt templates in `prompts/` render correctly, logged to W&B Artifacts
- [ ] `QLoRATrainer` loads 4-bit model, applies PEFT, trains 1 step (smoke test passes)
- [ ] `WandbCheckpointCallback` uploads checkpoints as artifacts with metadata
- [ ] Hybrid resume: local → W&B fallback works in tests
- [ ] `modal_train.py` launches on Modal, selects correct GPU from models.yaml
- [ ] All unit tests pass (`pytest tests/test_qlora_config.py tests/test_training_pipeline_mock.py`)
- [ ] Smoke test passes on Modal GPU (`modal run tests/test_qlora_trainer_smoke.py`)
- [ ] Baseline 100-example run completes, checkpoints in W&B
- [ ] **MANDATORY**: 3-config comparison runs, F2P scores computed, champion promoted to W&B Registry

---

## Implementation Order (Dependency-Topological)

0. **Step 0** → Infrastructure precondition check (Modal, W&B, GCS)
1. **Step 1** → `config/models.yaml`
2. **Step 2** → `config/qlora_variants.yaml`
3. **Step 3** → `training/qlora_config.py`
4. **Step 4** → `training/prompts/*.j2`
5. **Step 5** → `training/prompt_loader.py`
6. **Step 6** → `data_engineering/tokenize.py` (NEW: Phase 3 JSONL → .arrow)
7. **Step 7** → `training/qlora_trainer.py` (core)
8. **Step 8** → `training/callbacks.py`
9. **Step 9** → `training/resume.py`
10. **Step 10** → `training/modal_train.py`
11. **Step 11** → `tests/test_qlora_config.py`
11b. **Step 11b** → `tests/training/conftest.py`
12. **Step 12** → `tests/test_qlora_trainer_smoke.py`
13. **Step 13** → `tests/test_training_pipeline_mock.py`
14. **Step 14** → `scripts/run_3config_comparison.py`
15. **Step 15** → Baseline 100-ex run (manual Modal)
16. **Step 16** → Full 3-config comparison (manual)

---

## Notes

### Vertical Slicing
- **Vertical slice principle (ADR-012)**: Steps 0-10 deliver a working trainer; 11-13 verify; 14-16 execute comparison
- **Champion promotion**: Winner alias in W&B Model Registry enables Phase 5 serving (provisional — see below)
- **Prompt versioning**: W&B Artifact per run ensures reproducibility for Phase 5 A/B
- **Prompt A/B runner**: Deferred to Phase 5 (`evaluation/prompt_ab_test.py`) per MASTER-PLAN 4.3

### F2P Proxy & Re-Evaluation
- Step 14 uses proxy F2P (test file overlap + fix keywords). **Champion selected by proxy is provisional.**
- Phase 5 evaluation harness (`evaluation/harness.py`) **must** re-evaluate the champion with full F2P computation against real test suites.
- After Phase 5 re-evaluation, the W&B Registry `champion` alias is updated if proxy and full F2P agree; otherwise the best-performing model on real F2P is promoted.

### Master Plan Appendix D Deviations
This plan intentionally deviates from the file layout in MASTER-PLAN.md Appendix D. Deviations and rationale:

| Master Plan Expects | This Plan Creates | Rationale |
|---------------------|-------------------|-----------|
| `training/qlora_train.py` | `training/qlora_trainer.py` (class) + `training/qlora_train.py` (CLI stub) | Class-based design supports `train()` and `resume()` as methods; thin CLI wrapper preserves Master Plan acceptance criterion `python -m training.qlora_train` |
| `training/model_config.py` | `config/models.yaml` + `training/qlora_config.py` | YAML config is cleaner for model registry updates without code changes; factory pattern for LoraConfig/TrainingArguments |
| `training/modal_train.py` | `training/modal_train.py` | ✅ Matches (was `modal_train.py` at root, moved to `training/`) |
| `training/callbacks.py` | `training/callbacks.py` | ✅ Matches |
| `training/checkpoint.py` | Merged into `training/callbacks.py` + `training/resume.py` | Checkpoint saving is a callback concern; loading/resolution belongs in resume module. Cleaner separation than separate file |
| `training/prompts.py` | `training/prompts/*.j2` + `training/prompt_loader.py` | Directory of templates + loader class is more maintainable than single prompts.py |
| `training/` at root | `training/` at root | ✅ Matches (was `src/training/`, corrected to match Master Plan) |
| `training/qlora_compare.py` | `scripts/run_3config_comparison.py` | Orchestration script for 3 variants lives in `scripts/` (not training package) to avoid coupling CLI orchestration with training library |
| *(not in manifest)* `config/` directory | `config/models.yaml` + `config/qlora_variants.yaml` | New top-level `config/` directory stores model registry and QLoRA variant YAML files; replaces monolithic `training/model_config.py` for cleaner separation; not shown in Master Plan Appendix D manifest |

### Existing `src/swe_qwen/modal_app.py`
- Phase 1 skeleton with `train_swe_qwen` using SFTTrainer.
- Phase 4 supersedes it with modular `training/modal_train.py` → `QLoRATrainer`.
- `modal_app.py.train_swe_qwen` is kept as reference for smoke testing (GPU path validation) but is **not** the production path.
- After Phase 4 stabilizes, `modal_app.py` can be cleaned up (Phase 11 hardening).

### Dependencies
- **New dependency**: `jinja2>=3.1.0` for prompt templates (add to `pyproject.toml`)
- Existing deps from Phase 1 (`pyproject.toml`) already include: `transformers>=5.14.0`, `peft>=0.19.0`, `bitsandbytes>=0.49.0`, `trl>=1.9.0`, `accelerate>=1.14.0`, `datasets>=5.0.0`, `wandb>=0.28.0`, `modal>=1.5.0`, `pyyaml>=6.0.1` — no additional ML deps needed
