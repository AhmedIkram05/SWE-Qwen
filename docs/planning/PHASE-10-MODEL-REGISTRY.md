# PHASE-10: Model Registry as Platform

**Status:** Planned
**Goal:** Any open-weight causal LM on Hugging Face runs the full pipeline — tokenize → QLoRA train → execution-based eval → statistical promotion → OpenAI-compatible serve — without editing tracked config, by adding it via CLI.

**Scope ceiling (honest):** any open-weight autoregressive HF model with a chat template. Not: closed APIs, non-causal architectures. QLoRA + SWE-bench eval + vLLM serving all assume a causal LM.

---

## Step 0 — Shared registry loader (the real bulk of the diff)

Today ~8 call sites each do their own `yaml.safe_load` of `config/models.yaml`. The user overlay only works if every reader goes through ONE merged loader. Without this, "add a model → all stages pick it up" is false.

**New module: `registry/`** (add `registry*` to `[tool.setuptools.packages.find]` include list).

- `registry/loader.py` — `load_models() -> dict[str, ModelSpec]`:
  - Merge order: `config/models.yaml` (tracked, generated) ← `config/models.user.yaml` (user, gitignored) overlay. Deep-merge **per model key**, not whole-file replace — overlay overrides/extends keys of an existing model, or adds new models.
  - `ModelSpec` pydantic model validating required fields (`hf_id`, `context_window`, `target_modules`), with defaults (`gpu_mapping` fallback → `a10g-24gb`, `serving_hf_id: null`, `serving_quantization: null`).
  - In-memory `@lru_cache` on the merged dict keyed by mtime; single reload path.
- Migrate every reader to call the loader:
  - `training/qlora_config.py` (`_MODELS_PATH` load, ~L25)
  - `data_engineering/tokenize.py` (`context_window` reads, ~L181, L302-304)
  - `data_engineering/run_pipeline.py` (`tokenize_model`, ~L635)
  - `inference/prompt_builder.py` (`resolve_hf_id()`, L363-379)
  - `inference/config.py` (`registry_serving_hf_id()`, L73-84) — redirect to loader, keep method name as compat
  - `scripts/prepare_training_data.py` (L244-250)
  - `scripts/run_3config_comparison.py` (L756, auto gpu resolve)
  - `evaluation/inference.py` rides on `prompt_builder` — free once that migrates
  - Hardcoded defaults in `data_engineering/config.py`, `evaluation/config.py`, `data_engineering/cli.py`, `evaluation/cli.py` resolve registry-first, Qwen-default-last (Step 2)

**Tests (`tests/registry/test_loader.py`):** overlay wins on conflict, deep-merge per key, missing user file ok, invalid entry → validation error naming the model, mtime cache invalidation.

## Step 1 — `model add` CLI

**`registry/cli.py`** (typer, `python -m registry.cli`; console script `model = "registry.cli:app"`):

```
model add meta-llama/Llama-3.1-8B-Instruct --name llama-31-8b --context 8192 --target-modules auto
```

Validation, in order (cheap HTTP + local; no GPU):
1. HF reachability: HEAD `https://huggingface.co/api/models/{id}` (network failure → error, don't write)
2. `ContextWindow` probe: read `config.json` `max_position_embeddings`; `--context 0` = use that
3. Chat-template detection: `tokenizer_config.json` has non-empty `chat_template` → warn (+ `--allow-no-template` to override) — SWE-bench prompt building needs it
4. Thinking support: `tokenizer_config.json` contains `enable_thinking`/`chat_template_continue_generation` markers → sets `prompt_behavior.no_think: false` (default true for new models)
5. `--target-modules auto` (default): `peft.utils.other.find_all_linear_names` from the model config's architecture (Llama/Mistral → q/v/k/o/gate/up/down; fallbacks for other archs)

Writes ONLY to `config/models.user.yaml`. Refuses to clobber an existing key unless `--force`. Output the merged spec and the stages that will pick it up.

`model list` — merged registry table (source: tracked vs user). That's the whole CLI; `delete`/`edit` can be YAML edits for now.

## Step 2 — De-hardcode, registry-first

Resolve from registry, keep the phase's Qwen defaults as **last-resort** defaults, no implicit fallback to a specific family:

- `inference/config.py`: `base_model` (L23), `serving_hf_id` (L26), `quantization` (L27), `variants` (L28), `lora_artifact_pattern` (L29), `default_variant` (L32) → registry-derived; artifact pattern = `model-{key}-{variant}`; `variants`/`default_variant` empty-safe for a fresh model (serve base model, no LoRA — already a supported path in `openai_compat.resolve_engine_model`)
- `data_engineering/config.py`: `tokenize_model` (L49) registry-first
- `evaluation/config.py`: `baseline_model` (L30), `lora_artifact_pattern` (L33)
- CLI typer defaults (`evaluation/cli.py` L42/160/185, `data_engineering/cli.py` L156-158/225-227): default = first registry key or env, keep `qwen3-14b` only as literal when registry absent

## Step 3 — Quantization-aware serving

`inference/serve.py` `VLLMEngine._ensure_engine` currently passes `quantization=config.quantization` unconditionally (`'awq'`). Change:

- `serving_hf_id` present → current behaviour (AWQ/FP8 path, `serving_quantization`)
- `serving_hf_id: null` (new models) → vLLM `LLM(model=registry.hf_id, enable_lora=True, max_lora_rank=...)`, omit quantization, bf16/fp16 — LoRA adapters serve on top of the plain base
- Validate at config load (not vLLM boot): `serving_quantization` set but no `serving_hf_id` → error

This is what makes arbitrary models serveable without a pre-quantized base existing. Note the two distinct YAML keys: `quantization: nf4` (training QLoRA) vs `serving_quantization` (serving) — never cross-map them.

## Step 4 — Per-model prompt behaviour (gate both Qwen-specifics)

New registry key `prompt_behavior: {no_think: bool}` (Qwen entries: `true`). Gate BOTH:

1. `/no_think\n` insertion — `serve.py:198` (`_build_prompt` LoRA path) and `evaluation/inference.py:348` (`_generate_patches_batch_body` wrap)
2. `apply_chat_template(..., enable_thinking=False)` — `serve.py:216` and `prompt_builder.py:89` (`no_think_wrap`). This kwarg raises on non-Qwen tokenizers; it's the second Qwen-gate the string-only view misses.

Registry flag false → plain template apply, no string insert. `prompt_builder.no_think_wrap(hf_id, ...)` reads the flag; all callers route through it.

**Update pinned tests:**
- `tests/inference/test_serve_extra.py` L142-152 (asserts exact `/no_think` prompt strings) — parametrize per flag
- `tests/evaluation/test_eval_inference_coverage.py` L549/593/621 (monkeypatch `_no_think_wrap`), L643-666 (`enable_thinking=False` asserts)
- `tests/inference/test_prompt_builder_extra.py` L57/66/291, `test_openai_compat.py` L26, `test_config.py` L28, `tests/training/test_qlora_config.py` L250-251, `tests/training/conftest.py` L21 (models.yaml fixture)

## Step 5 — Training fallback: verify, don't build

`training/unsloth_factory.py` already has Unsloth → TRL+PEFT+bitsandbytes fallback on exception or `UNSLOTH_ENABLED=0`. Claim-5 work is mostly verification: run `build_model_and_peft` on a non-Unsloth-family tiny model (SmolLM2-135M, llama-arch) offline.

One real fix rides along: `_build_fallback` (L246-247) hardcodes `load_in_4bit=True` + quant type `fp4`, ignoring `model_cfg.quantization`. But that force is a **documented workaround** — nf4 causes CUDA illegal memory access on A10G with Qwen3-14B (comment L242-244). Naively honoring the registry would resurrect the crash. Fix: `_build_fallback` reads `model_cfg.quantization` via a small map (nf4/fp4 → 4-bit with that quant type, int8 → 8-bit, none → plain bf16 load), and the workaround moves into the registry data as an explicit per-model override (`fallback_quantization: fp4`, with the comment) on the qwen3-14b entry. Same edit session as Step 9-5's `max_memory` budget fix — same function, L246-250.

## Step 6 — Promotion layer + CI workflows (the gate is model-locked too)

The promotion pipeline and CI are Qwen-coupled in three real places:

- `promotion/run.py:355` — `base_model = serve.base_model` feeds the whole decide flow (`candidate_model_ref = f"{base_model}:{candidate}"` at L367/399/423/452, eval `--models` at L274, `champion_key` at L551). Resolution order: new `--candidate-model` CLI arg → champion record's `model_ref` base (champion.json is already downloaded before decide runs — free) → `ServeConfig.base_model` last resort. Goes hand-in-hand with a new `candidate_model` input on `.github/workflows/promote.yml`.
- `promotion/deploy.py:135` — liveness probe payload `"model": "qwen3-14b"` literal → champion record's model base.
- `.github/workflows/eval.yml:145` — smoke gate runs `--models qwen3-14b:higher_rank_14b` literally → derive from the champion record (`ci/champion.json`, downloaded next to `smoke_baseline.json`) via `model_ref`, or an `eval run --models default` resolution against the registry.

Already agnostic: `promotion/registry.py` (records are JSON-data, `model_ref` is a field; only docstrings show Qwen), the `champion.json` schema itself (`variant` + `model_ref` carry everything), `cd.yml`/`ci.yml` (no model content).

## Step 7 — Scripts + dev defaults sweep

- `scripts/tag_challenger.py:31` — `f"model-qwen3-14b-{args.variant}"` literal → `EvalConfig.lora_artifact_pattern` (config already carries it; one line).
- `scripts/preflight_serve.py:27,29` — `_MODEL_BASE`/`_MODEL_LORA` literals → `--model`/`--model-lora` args defaulting to registry.
- `scripts/seed_dashboards.py:32` — `_SEGMENT = "qwen3-14b/baseline_14b/template_v1"` → `--model`/`--variant` args.
- `scripts/init_wandb.py` — W&B project bootstrap literals (`model_family: "qwen"` L68, model/fallback_model L189-190, tags, description) → `--model-family`/`--model` args with current values as defaults (one-shot infra script).
- `scripts/seed_champion.py:36` — the seeded record is a historical snapshot, left as-is; add `--model-ref`/`--variant` overrides for future cycles.
- `scripts/local_e2e_smoke.py:118`, `evaluation/local_backend.py:32` (ollama `qwen2.5-coder:7b`), `evaluation/prompt_ab_test.py:41` — dev-mode defaults → registry-derived, literal last resort.

## Step 8 — Deployment path (mostly already agnostic)

The deploy chain is well-positioned: the variant pin rides `SERVING_DEFAULT_VARIANT` env (`promotion/deploy.py:72`), the engine resolves model + quantization from the registry at container start (`modal_serve._build_smoke`, `VLLMEngine(ServeConfig())`), and rollback re-deploys `previous.model_ref` (data-driven). Steps 0–3 carry deployment for free. Four remaining gaps:

- `inference/modal_serve.py:121` — `gpu="A10G:1"` hardcoded on both the smoke step and the serving class. A 14B fp16 model does not fit A10G — **the** deployment blocker for arbitrary models. Resolve GPU from the registry `gpu_mapping` via env (`SERVING_GPU`, default registry fallback) at module import; pass the string into `@app.cls(gpu=...)` and the image smoke.
- `.github/workflows/cd.yml:186` — `modal deploy -m inference.modal_serve` runs unpinned, so it deploys whatever `ServeConfig` defaults to. `ServeConfig` already reads `SERVING_BASE_MODEL` via pydantic-settings (no code needed) — set `SERVING_BASE_MODEL` + `SERVING_GPU` (+ `SERVING_DEFAULT_VARIANT` where intended) in the workflow.
- Class name `QwenServer` → rename (cosmetic; the Modal app name `swe-qwen-serving` stays — infra identity).
- Ops note: the `serve-model-cache` volume holds the current model's quantized base and the cache-buster line (`serve-cache-bust-v1`) must change with it — the new model's base downloads once on first boot (28 GB-class). No code.

The probe payload (`promotion/deploy.py:135`) is fixed in Step 6.

## Step 9 — Audit pass (post-sweep findings)

Result of a full-repo literal sweep (`(?i)qwen` across all source, plus `4096`/`max_model_len` coupling greps, `.github/`, and the scripts inventory). Five gaps the original steps missed, folded in here:

1. **Registry default model** — today ~15 CLI/config defaults are the literal `"qwen3-14b"`. Give the registry a designated default (one entry flagged `default: true`) and a `registry.default_model_key()` helper; every default resolves through it: `training/qlora_train.py:24`, `training/qlora_config.py:129,225`, `training/qlora_trainer.py:61`, `training/modal_train.py:197,332`, `training/local_cli.py:18`, `data_engineering/tokenize.py:290,407`, `data_engineering/cli.py:156,225`, `data_engineering/run_pipeline.py:635`, `evaluation/cli.py:160`, `evaluation/prompt_ab_test.py:41`, `evaluation/local_backend.py:32`, `inference/benchmark.py:224`, `scripts/prepare_training_data.py:243`, `scripts/run_3config_comparison.py:753`, `scripts/local_e2e_smoke.py:118`. One mechanism, not per-site bespoke defaults.
2. **`inference/prompt_builder.py:38` `_DEFAULT_HF_ID = "Qwen/Qwen3-14B"`** — silent Qwen fallback when the registry is missing or a key is unknown (`resolve_hf_id` L377-386). New behavior: unknown key → error; missing models.yaml in serving/eval → error. Flips pinned tests `tests/inference/test_prompt_builder_extra.py:283,287` and `tests/evaluation/test_eval_inference_coverage.py:368-395` (they assert the Qwen fallback today).
3. **Context-length couplings** — `inference/config.py:40,49,51` (`max_model_len`, `default_max_tokens`, `max_tokens_cap`, all 4096) and `data_engineering/config.py:50` + `cli.py:161` (`tokenize_max_length` 4096) default from the registry `context_window` (env-overridable). `tests/inference/test_config.py:16` updated. `inference/benchmark.py:144` stays a sweep-tool knob.
4. **`scripts/run_3config_comparison.py:69`** — `_artifacts_path` hardcodes `f"model-qwen3-14b-{variant}"` → use `EvalConfig.lora_artifact_pattern` (same one-line fix as `tag_challenger.py:31`).
5. **Training device budget static** — `training/unsloth_factory.py:160,250` `max_memory={0: "18GiB", "cpu": "32GiB"}` is A10G-shaped regardless of the registry `gpu_mapping`. Derive budget from the resolved GPU tier (folded into Step 5 verification).

Verified NOT model-coupled (no work): `unsloth_factory._fix_eos_token` (generic eos candidate list + decode-from-id last resort; Qwen mentions are comments only); `training/prompts/chat.j2` `### Response` (repo-owned training-contract marker, not a model token); `config/observability.yaml` (gpu-type rate keys only); CI workflows (exactly 4 exist: cd/ci/eval/promote; no Makefile); every remaining qwen mention is infra identity (wandb projects, GCS bucket `swe-qwen-datasets`, Modal app names), docstrings, `docs/`, or the baked `promotion/MODEL_CARD.md` (regenerates from the artifact pattern).

## Verification

- `pytest tests/registry tests/inference tests/evaluation -m "not requires_credentials"` after each step
- Smoke end-to-end (no Modal, no real GPU): `model add` a tiny model → `data-pipeline tokenize` → eval in local mode → `serve` with `SERVING_STUB=1`, all mocked GPU paths
- ruff + mypy (repo gates: line-length 100, double quotes, mypy files list must include `registry/`)

## Out of scope

Modal runs, real training compute, closed-API models, non-autoregressive architectures, multi-node serving.

**Infra-identity names deliberately NOT renamed** — they name the platform, not the model family: GCS bucket `swe-qwen-datasets`, W&B project `swe-qwen` and entity, Modal app names, `swe-qwen-codecov`. Renaming them is churn with zero agnosticism value; any model rides on the same infra.

## Effort

3 days. Step 0 is the bulk; steps 2–4 are default-punning and one branch each; steps 6–7 are a dozen one-liners plus two workflow edits and one CLI argument.
