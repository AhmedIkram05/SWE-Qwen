# Implementation Plan: Unsloth Integration for 2-5x Faster QLoRA Training

## Overview
Integrate Unsloth's optimized Triton kernels into the existing SWE-Qwen QLoRA training pipeline (Phase 4) to achieve 2-5x faster training with 60-74% less VRAM, while maintaining full TRL/PEFT compatibility for a mechanical fallback.

## Requirements
- 2-5x training speedup on H100/A100 via Unsloth Triton kernels
- 60-74% VRAM reduction enabling larger batch sizes or larger models on same hardware
- Full backward compatibility: zero breaking changes to existing config YAMLs, Modal entrypoints, or training scripts
- Mechanical fallback to plain TRL + PEFT + bitsandbytes if Unsloth fails
- Support for Qwen3-14B and Qwen3-30B-A3B on Modal H100/A100 80GB
- Compatibility with existing config/models.yaml and config/qlora_variants.yaml

## Architecture Changes

| Component | Current | With Unsloth |
|-----------|---------|--------------|
| Model loading | `AutoModelForCausalLM.from_pretrained` + `prepare_model_for_kbit_training` | `FastLanguageModel.from_pretrained` (4-bit/8-bit via `load_in_4bit`) |
| LoRA wrapping | `get_peft_model(model, LoraConfig(...))` | `FastLanguageModel.get_peft_model(model, ...)` |
| Trainer | `SFTTrainer` (TRL) | `SFTTrainer` (TRL) — **unchanged**, just pass Unsloth-wrapped model |
| Config | `config/qlora_variants.yaml` + `config/models.yaml` | **Unchanged** — map via factory |
| Modal entrypoint | `training/modal_train.py` | **Unchanged** — toggle via env var |

## Implementation Steps

### Phase 1: Add Unsloth Dependency & Factory (Low Risk)
**File: `training/unsloth_factory.py` (NEW)**
```python
# Factory: try Unsloth first, fallback to TRL+PEFT+BnB
# Maps config/models.yaml + config/qlora_variants.yaml -> (model, peft_model)
# Env var: UNSLOTH_ENABLED=1 (default: 1 for H100/A100, 0 to force fallback)
```

**File: `pyproject.toml` / `requirements.txt`**
- Add `unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git` (or pinned release once 2026.x lands)
- Keep existing `transformers`, `peft`, `bitsandbytes`, `trl` for fallback

**File: `training/qlora_config.py`**
- Add `build_model_and_peft(config, qlora_variant)` that delegates to `unsloth_factory`
- Preserve existing `LoraConfig` + `TrainingArguments` factory interface

### Phase 2: Update QLoRATrainer to Accept Pre-Wrapped Model (Low Risk)
**File: `training/qlora_trainer.py`**
- Modify `QLoRATrainer.__init__` to accept optional `model` and `peft_model` args
- If provided, skip internal model loading; else fall back to current logic
- No change to `train()` method signature

### Phase 3: Wire Modal Entrypoint (Low Risk)
**File: `training/modal_train.py`**
- Read `UNSLOTH_ENABLED` env var (default `1` on H100/A100)
- Pass through to `QLoRATrainer` via config factory
- No other changes — existing Modal config, volumes, secrets unchanged

### Phase 4: Config Mapping & Validation (Medium Risk)
**File: `training/unsloth_factory.py`**
- Map `config/models.yaml` fields → `FastLanguageModel.from_pretrained` kwargs:
  - `model_name` → `model_name`
  - `load_in_4bit` / `load_in_8bit` → `load_in_4bit` / `load_in_8bit`
  - `torch_dtype` → `dtype` (map `torch.bfloat16` → `"bfloat16"`)
  - `device_map` → `device_map` (default `"auto"`)
  - `attn_implementation` → `attn_implementation` (default `"flash_attention_2"` on H100)
  - `max_seq_length` → `max_seq_length` (from `qlora_variants.yaml` or trainer args)
- Map `config/qlora_variants.yaml` LoRA fields → `FastLanguageModel.get_peft_model` kwargs:
  - `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `bias`, `task_type`
  - Add `use_gradient_checkpointing="unsloth"` (Unsloth default, saves 30% VRAM)
  - Set `padding_free=False` explicitly (avoids known collision with `assistant_only_loss`)

### Phase 5: Fallback Mapping Table (Low Risk)
**File: `training/unsloth_factory.py`**
```python
FALLBACK_MAP = {
    "FastLanguageModel.from_pretrained": "AutoModelForCausalLM.from_pretrained + prepare_model_for_kbit_training",
    "FastLanguageModel.get_peft_model": "get_peft_model(model, LoraConfig(...))",
    "FastLanguageModel.for_inference": "model.eval() + model.merge_and_unload() if needed",
}
```
- On any Unsloth exception → log warning, fall back using mapping
- Identical `LoraConfig` + `TrainingArguments` passed to `SFTTrainer` in both paths

### Phase 6: Testing & Validation (Medium Risk)
- Unit test: `training/test_unsloth_factory.py` — factory returns model+peft on both paths
- Integration test: `training/test_unsloth_integration.py` — end-to-end 1-epoch run on tiny model (e.g., Qwen2.5-0.5B) with both `UNSLOTH_ENABLED=1` and `=0`
- Modal smoke test: run 1 epoch on A100 80GB via `modal run training/modal_train.py --model qwen3-14b --epochs 1`
- Benchmark: compare tokens/sec and peak VRAM vs baseline (expect 2-5x speedup, 60-74% VRAM drop)

## Testing Strategy
| Test | Scope | Command |
|------|-------|---------|
| Unit | Factory returns (model, peft_model) on both paths | `pytest training/test_unsloth_factory.py` |
| Integration | 1-epoch tiny model end-to-end | `pytest training/test_unsloth_integration.py` |
| Modal Smoke | Real GPU, 1 epoch Qwen3-14B | `modal run training/modal_train.py --model qwen3-14b --epochs 1` |
| Benchmark | Tokens/sec, peak VRAM vs baseline | Manual compare logs |

## Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Unsloth version conflict with transformers/trl | Medium | Pin `unsloth` to tested commit; fallback is mechanical |
| `assistant_only_loss` + `padding_free` collision | Known (2026.7.x) | Explicit `padding_free=False` in factory; test messages-shaped datasets |
| Modal H100/A100 driver mismatch | Low | Pin `bitsandbytes` to 0.49+; Unsloth handles CUDA kernels |
| Config mapping drift | Low | Centralize mapping in `unsloth_factory.py`; unit test round-trip |
| Merge/export differences | Low | Unsloth `for_inference` + `merge_and_unload` maps to PEFT equivalent |

## Success Criteria
- [ ] `UNSLOTH_ENABLED=1` trains Qwen3-14B on A100 80GB in ≤50% baseline time
- [ ] `UNSLOTH_ENABLED=1` trains Qwen3-30B-A3B on H100 80GB in ≤40% baseline time
- [ ] Peak VRAM ≤40% of baseline (enables 2x batch size or larger model)
- [ ] `UNSLOTH_ENABLED=0` produces bitwise-identical checkpoints to current pipeline
- [ ] All existing config YAMLs work unchanged
- [ ] Modal entrypoint unchanged except env var
- [ ] Unit + integration tests pass in CI

## Rollout Plan
1. **Phase 1-3** (this PR): Factory + trainer modification + Modal wiring behind `UNSLOTH_ENABLED=1`
2. **Phase 4-5** (follow-up): Config mapping + fallback table + unit tests
3. **Phase 6** (follow-up): Integration tests + Modal benchmarks on A100/H100
4. **Default flip**: After validation, flip `UNSLOTH_ENABLED` default to `1` for H100/A100
