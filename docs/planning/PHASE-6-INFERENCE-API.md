# Phase 6 Implementation Plan: Inference API — Serverless vLLM on Modal

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Draft v1.0
**Parent Document:** `docs/planning/MASTER-PLAN.md`
**Dependencies:** Phase 4 complete (trained LoRA adapters as W&B artifacts + local checkpoints), Phase 1 complete (Modal configured, secrets), Phase 5 complete (eval harness + `evaluation/inference.py` serve-shaped inference pattern), `config/qlora_variants.yaml` (variant registry)

---

## 1. Objective

Deploy a **production-grade, OpenAI-compatible inference API** served **serverlessly on Modal with scale-to-zero**:

- `modal serve inference.modal_serve` starts the endpoint; any OpenAI SDK client works with an OpenAI-compatible base URL
- vLLM serves a **pre-quantized base** (Path A: FP8, AWQ fallback) with **live LoRA attachment** per request on **A10G-24GB**
- Endpoint scope: **`/v1/chat/completions`** (non-streaming + streaming SSE) only — embeddings and legacy `/v1/completions` explicitly out of scope for V1
- Telemetry (TTFB, tokens/sec, request count, error rate, GPU util, cost per inference) logged to **W&B**
- 6.1 config sweep selects `gpu_memory_utilization` × `max_num_seqs`; 6.8 benchmark proves TTFB p50 < 500ms under load; cold start measured and documented
- Prompt-building logic **shared** between served inference and eval F2P so they never drift

---

## 2. Inputs (from completed phases)

| Source | Artifact | Location |
| -------- | ---------- | ---------- |
| Phase 4 | Trained LoRA adapters (baseline_14b, higher_rank_14b, higher_lr_14b; efficient_14b debug) | W&B artifacts `model-qwen3-14b-{variant}` (type `model_checkpoint`) + `models/checkpoints/{variant}/` local copies |
| Phase 4 | Prompt templates (system.j2, user.j2, chat.j2) | `training/prompts/*.j2` via `training.prompt_loader.PromptLoader` |
| Phase 3 | Golden few-shot set | `data/golden.jsonl` (baked) + public GCS `gs://swe-qwen-datasets/datasets/{run_id}/swebench/golden.jsonl` |
| Phase 5 | Serve-shaped vLLM pattern: singleton `LLM`, `enable_lora=True`, `max_lora_rank=64`, `gpu_memory_utilization=0.85`, per-request `LoRARequest`, `_no_think_wrap`, `resolve_hf_id`, `resolve_adapter_path` | `evaluation/inference.py` |
| Phase 5 | Modal app skeleton: image build, volume, secrets, cache-buster, env vars (`VLLM_USE_FLASHINFER_SAMPLER=0`, `HF_HOME`, `VLLM_CACHE_ROOT`), concurrency cap | `training/modal_train.py`, `evaluation/inference.py` |
| Phase 5 | Config pattern (pydantic-settings, env prefix, frozen) | `evaluation/config.py` (`EvalConfig`) |
| Phase 1 | Modal account, secrets `wandb-secret`, `hf-secret` | Modal |
| Phase 4 | Model registry | `config/models.yaml` (`qwen3-14b` → `Qwen/Qwen3-14B`, context 32768) |

---

## 3. Decisions Locked (grilled, accepted)

| # | Decision | Choice |
| --- | ---------- | -------- |
| D1 | Serving stack | **Path A**: pre-quantized base (try `Qwen/Qwen3-14B-FP8` first; AWQ/GPTQ build as documented fallback) + live LoRA per request. No merge, no bf16 on A100. Bounded fallback budget ~$5–15. |
| D2 | GPU tier | **A10G-24GB only** for the production endpoint. |
| D3 | Endpoint scope | **`/v1/chat/completions` only** (non-stream + stream). No `/v1/completions`, no `/v1/embeddings` in V1. |
| D4 | Warm containers | **keep_warm=0** — true scale-to-zero per DoD/ADR-010. First request after idle pays cold start (measured, documented). |
| D5 | Prompt sharing | **Extract shared module** `inference/prompt_builder.py`; `evaluation/inference.py` imports it and re-exports for backward compat. |
| D6 | Telemetry target | **W&B only** (per master plan + ADR-011). No Langfuse in V1. |
| D7 | API auth | **Modal web endpoint auth** (Modal-issued Bearer token). No custom auth code. |
| D8 | 6.1 sweep scope | **Minimal sweep**: `gpu_memory_utilization {0.85, 0.90}` × `max_num_seqs {8, 16, 32}` on A10G only, plus the FP8-vs-AWQ LoRA-compat probe. Report + pick winner. |

### Decided without grilling (veto if wrong)

- `ServeConfig` env prefix **`SERVING_`** (mirrors `EVAL_`).
- Default `max_model_len=16384` (32768 KV cache does not fit 24GB with FP8 14B weights; knob for sweep).
- Request `model` accepts: `qwen3-14b`, `qwen3-14b:{variant}`, bare `{variant}`, full artifact `model-qwen3-14b-{variant}`; no variant ⇒ base model without LoRA. Unknown model ⇒ OpenAI-style 404 error body.
- pyproject `serve` console script retargeted to `inference.serve:app` (local dev, stub engine). Remote path is `modal serve inference.modal_serve`.
- New Modal volume **`serve-model-cache`** (kept separate from `eval-model-cache`).
- Benchmark report → `docs/planning/SERVING-BENCHMARK-REPORT.md`.
- Concurrency cap: `allow_concurrent_inputs=16` (Phase 5's aiohttp/64-way lesson; vLLM queues internally).

---

## 4. Module Structure (flat, matching `data_engineering/` / `evaluation/`)

```
inference/
├── __init__.py
├── config.py              # ServeConfig (pydantic-settings, SERVING_ prefix, frozen)
├── prompt_builder.py      # SHARED prompt logic — extracted from evaluation/inference.py (pure, no modal/vllm imports)
├── openai_compat.py       # OpenAI request/response/chunk Pydantic schemas + model mapping + SSE assembly (pure)
├── serve.py               # FastAPI app factory create_app(engine): routes, validation, errors, streaming, telemetry hooks
├── telemetry.py           # per-request metrics (TTFB, tok/s, latency, errors), W&B flush, GPU util, cost calc
├── modal_serve.py         # Modal App: image, volume, secrets, @app.cls persistent engine + ASGI serving, scale-to-zero
└── benchmark.py           # sweep (6.1 engine-config matrix) + benchmark (6.8 HTTP-level) → W&B + report
```

**Consolidation rationale:** vLLM engine wrapper lives in `serve.py` (not a separate `engine.py`) — it is the `LLM` singleton + `LoRARequest` attach from `evaluation/inference.py`, moved verbatim; `openai_compat.py` is pure schema/assembly so all OpenAI-format logic is unit-testable with zero GPU. `benchmark.py` doubles as the 6.1 sweep driver and the 6.8 acceptance benchmark (two subcommands, shared measurement core).

---

## 5. Detailed Module Specifications

### 5.1 `inference/config.py`

```python
class ServeConfig(BaseSettings):
    """Serving config. Copy the EvalConfig pattern; do NOT import EvalConfig."""
    model_config = SettingsConfigDict(env_prefix="SERVING_", env_file=".env", extra="ignore", frozen=True)

    # Model / adapter registry
    base_model: str = "qwen3-14b"                      # registry key from config/models.yaml
    serving_hf_id: str = "Qwen/Qwen3-14B-FP8"          # pre-quantized base (AWQ fallback: "Qwen/Qwen3-14B-AWQ")
    quantization: str = "fp8"                          # "fp8" | "awq"
    variants: tuple[str, ...] = ("baseline_14b", "higher_rank_14b", "higher_lr_14b")  # served LoRA variants
    lora_artifact_pattern: str = "model-qwen3-14b-{variant}"
    default_variant: str = "baseline_14b"

    # Engine (6.1 sweep knobs; defaults = Phase 5 eval-proven)
    gpu_memory_utilization: float = 0.85               # 0.85 | 0.90
    max_num_seqs: int = 16                             # 8 | 16 | 32
    max_lora_rank: int = 64                            # REQUIRED: higher_rank_14b is rank 32 (vLLM default 16 kills EngineCore)
    max_model_len: int = 16384                         # 32768 does not fit 24GB with FP8; sweep knob
    enforce_eager: bool = False                        # serving: keep CUDA graphs (eval used eager=True to save ~150s boot)

    # Sampling defaults (Phase 5-proven values; request params override)
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.15                   # no penalty → ~1000x "```" fence degeneracy
    default_max_tokens: int = 4096                     # hard cap = max_model_len - prompt
    max_tokens_cap: int = 8192

    # Modal / W&B
    gpu_type: str = "a10g-24gb"                        # A10G:1
    modal_volume: str = "serve-model-cache"
    wandb_entity: str = "2571642-university-of-dundee"
    wandb_project: str = "swe-qwen"
    idle_timeout_seconds: int = 600                    # scale-to-zero idle window (documented, measured)
    max_concurrent_requests: int = 16                  # allow_concurrent_inputs; 64-way broke modal 1.5.3 aiohttp

    # Telemetry
    telemetry_flush_interval_seconds: int = 60         # in-container W&B flush cadence
    gpu_util_sample_interval_seconds: int = 5
```

Add `serving_hf_id` (+ optional `serving_quantization`) to `config/models.yaml` `qwen3-14b` entry so the registry stays the single source of truth.

### 5.2 `inference/prompt_builder.py` — SHARED (extracted from `evaluation/inference.py`)

Move verbatim from `evaluation/inference.py` (pure, no `modal`/`vllm` imports — locally testable):

- `resolve_hf_id(model_name)` — models.yaml registry key → hf_id; contains "/" → as-is; default `Qwen/Qwen3-14B`
- `resolve_adapter_path(variant, config)` — local `models/checkpoints/{variant}` (adapter_config.json check) → W&B artifact `model-qwen3-14b-{variant}:latest` → `None` (baseline fallback)
- `_TOKENIZER_CACHE` + `no_think_wrap(hf_id, prompt)` — `/no_think\n` soft switch before `### Response` for LoRA adapters (chat-wrap breaks the training contract → repetition loops); `enable_thinking=False` for chat template
- `_ensure_golden(dataset_run_id)` — public GCS download (urllib only), local cache, baked-file fallback
- `golden_patches(repo, exclude_instance_id, max_examples=2, max_lines=150)` — same-repo gold few-shot, no leakage, capped index
- `render_patch_prompt(example, template_name="chat", include_file_contents=False, example_patches=None)` — via `training.prompt_loader.PromptLoader` (system.j2 + user.j2, `render_chat`)
- `_fetch_raw_file(repo, base_sha, path)` (`@lru_cache(4096)`, best-effort) + `_file_snippets(...)` — code context
- `_GOLDEN_PATH = data/golden.jsonl`

**Compatibility contract:** `evaluation/inference.py` keeps `from inference.prompt_builder import (...) as _` + module-level re-export aliases (same names) so `scripts/`, `tests/`, and `debug_eval_one.py` imports keep working. Zero behavior change; run the eval smoke suite to prove it (see 6.7 in test strategy).

### 5.3 `inference/openai_compat.py` — pure OpenAI-format logic

Pydantic schemas (subset of the OpenAI spec, strict):

- `ChatCompletionRequest`: `model: str`, `messages: list[ChatMessage]` (roles system/user/assistant, content str — non-str content parts rejected with 422), `temperature: float = 0.1`, `top_p: float = 0.95`, `max_tokens: int | None`, `stream: bool = False`, `stop: str | list[str] | None`, `repetition_penalty: float | None = None` (non-standard passthrough, maps to engine), optional `swe_bench: dict | None` (non-standard extension — server-side prompt assembly via prompt_builder for SWE-bench style requests; ignored by strict OpenAI clients)
- `ChatCompletionResponse`: `id`, `object="chat.completion"`, `created` (unix int), `model` (echoed request model), `choices[0] = {index, message{role="assistant", content}, finish_reason}`, `usage {prompt_tokens, completion_tokens, total_tokens}`
- `ChatCompletionChunk`: `id`, `object="chat.completion.chunk"`, `created`, `model`, `choices[0] = {index, delta{role?|content?}, finish_reason}`
- `OpenAIErrorBody`: `{"error": {"message", "type", "param", "code"}}`

Functions (all pure, unit-tested):

- `resolve_engine_model(request_model, config) -> tuple[hf_id | None, lora_path | None, lora_name | None]` — mapping table in 3.3; unknown ⇒ raise `ModelNotFoundError`
- `build_messages(request)` → `(system_prompt, messages)` or, when `swe_bench` present, `render_patch_prompt(example=swe_bench, ...)`
- `assemble_response(...)` → `ChatCompletionResponse` (usage from engine output)
- `iter_chunks(...)` → yields `data: {json}\n\n` strings: first chunk `delta={"role":"assistant"}`, content chunks, final chunk `delta={}, finish_reason="stop"`, then `data: [DONE]\n\n` — SSE per OpenAI spec
- `error_response(status, message, type_, code)` → `OpenAIErrorBody` JSON

### 5.4 `inference/serve.py` — FastAPI app factory + vLLM engine

- `class Engine` (Protocol): `def generate(messages, *, model, max_tokens, temperature, top_p, stop, repetition_penalty) -> GenerationResult` where `GenerationResult = {text, prompt_tokens, completion_tokens}`
- `class VLLMEngine(Engine)` — **moved verbatim from `evaluation/inference.py`** (singleton `_LLM_CACHE` + `threading.Lock`, one LLM per base; `LLM(model=serving_hf_id, quantization=config.quantization, enable_lora=True, max_lora_rank=64, gpu_memory_utilization=..., max_num_seqs=..., max_model_len=...)`; adapter attach per request via `LoRARequest(lora_name=f"qwen3-14b-{variant}", lora_int_id=1, lora_path=...)`; `tokenize=False` batching). Sync `generate` runs in FastAPI's threadpool (vLLM sync engine is thread-safe and batches internally — verify under concurrency in 6.1).
- `class StubEngine(Engine)` — deterministic fake (echo/hash of prompt, simulated token pacing for stream tests). No GPU, no vLLM import. Used by local `serve` script + all unit tests.
- `create_app(engine: Engine, config: ServeConfig | None = None) -> FastAPI`:
  - `POST /v1/chat/completions` — validate → telemetry.start(request) → non-stream: single `assemble_response`; stream: `EventSourceResponse(iter_chunks(...))` (sse-starlette) with `X-Accel-Buffering: no`-style streaming headers; TTFB = first chunk send
  - Guard clauses: unknown model → 404 `OpenAIErrorBody`; validation errors → 422 OpenAI-style body; engine exceptions → 500 `OpenAIErrorBody` (log traceback; no leak of internals)
  - Telemetry middleware per request: latency, TTFB (first-byte timestamp), output tokens, error flag
  - `GET /health` → `{"status": "ok", "model": ..., "engine": ...}` (also used by Modal warmup check)
- Module-level `app = create_app(...)` for `uvicorn`/local `serve` script (engine resolved from env: stub when `SERVING_STUB=1`)

### 5.5 `inference/telemetry.py`

- `RequestRecord`: timestamp, model/variant, stream flag, TTFB ms, generation ms, output tokens, tokens/sec, error (bool + type), status
- `MetricsCollector`: thread-safe append; `summary()` → aggregates (count, error rate, TTFB p50/p95, tok/s mean, latency p50/p95)
- **W&B metric namespace (single convention, used by both telemetry flush and benchmark):** `serve/request_count`, `serve/error_rate`, `serve/ttfb_p50_ms`, `serve/ttfb_p95_ms`, `serve/latency_p50_ms`, `serve/latency_p95_ms`, `serve/tokens_per_sec`, `serve/gpu_util`, `serve/cost_per_inference`, `serve/cold_start_s`
- W&B flush loop: every `telemetry_flush_interval_seconds` (background thread) `wandb.log({...})` of rolling aggregates; **explicit `wandb.finish()` on shutdown** (Modal kills container before atexit — Phase 4 lesson)
- GPU util: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits` sampled at `gpu_util_sample_interval_seconds` in the flush loop
- Cost per inference: `(container wall seconds × gpu_rate) / requests` — gpu_rate from config constant (A10G hourly), logged per benchmark run (ADR-011)
- Cold-start record: benchmark writes `{cold_start_seconds, idle_before_request}` to W&B

### 5.6 `inference/modal_serve.py` — Modal App (6.2 + 6.3)

```python
app = modal.App("swe-qwen-serving")

image = modal.Image.debian_slim(python_version="3.11") \
    .apt_install("git", "wget", "curl") \
    .pip_install("vllm>=0.26.0", "fastapi>=0.115.0", "uvicorn>=0.34.0",
                 "sse-starlette>=2.2.0", "wandb>=0.28.0") \
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"}) \      # FlashInfer sampler JIT-compiles nvcc; no CUDA toolkit in container
    .env({"HF_HOME": "/models"}) \                     # volume-cached quantized base
    .env({"VLLM_CACHE_ROOT": "/models/vllm-cache"}) \  # persist CUDA graph cache → fast cold starts after first boot
    .run_commands("echo 'serve-cache-bust-v1'") \      # bump ONLY when deps change (credit conservation)
    .add_local_dir("inference", remote_path="/inference", copy=True) \
    .add_local_dir("training", remote_path="/training", copy=True) \    # prompts/*.j2 for PromptLoader
    .add_local_dir("config", remote_path="/config", copy=True) \
    .add_local_file("data/golden.jsonl", remote_path="/data/golden.jsonl", copy=True)  # last, as in modal_train.py

serve_volume = modal.Volume.from_name("serve-model-cache", create_if_missing=True)
_secrets = [modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")]  # hf-secret: Qwen3-14B is gated

@app.cls(gpu="A10G:1", image=image, volumes={"/models": serve_volume},
         secrets=_secrets, allow_concurrent_inputs=16, timeout=1800)
class QwenServer:
    @modal.build()               # boot-time fail fast: engine init errors (VRAM, max_model_len) surface in warmup, not at first request
    def build(self): ...         # VLLMEngine() construction + warmup generate

    @modal.enter()               # persistent engine, one-time async init
    def enter(self): self._engine = VLLMEngine(ServeConfig())

    @modal.asgi_app()            # persistent engine + FastAPI in one container
    def web(self): return create_app(self._engine)
```

- **Scale-to-zero:** no `keep_warm` — containers die when idle; first request after idle boots fresh (measured cold start).
- **Module-import registration:** engine/web registered at import time (Modal 1.5.x lazy-registration gotcha — `modal serve`/deploy entry must import this module top-level).
- **No vLLM registry image** (`vllm/vllm-openai:latest` is unusable on Modal — no python in PATH, exit 127); rebuild from debian_slim exactly as eval does.
- **Pin modal** to the version pinned for Phase 5 until Phase 6 done (aiohttp 1.5.3 regression lesson).

### 5.7 `inference/benchmark.py` (6.1 + 6.8)

Two subcommands sharing a measurement core (openai SDK client):

- `sweep` (6.1): for each config in `gpu_memory_utilization {0.85, 0.90}` × `max_num_seqs {8, 16, 32}` (+ FP8-vs-AWQ probe when Path A initial probe fails): one Modal run booting `VLLMEngine(config)` + N synthetic chat generations in-container (engine-level TTFB/tok-s/VRAM), in-process metrics, no HTTP. ~6 boots × ~5 min ≈ 30 A10G-min ≈ <$1. Emits config table → `docs/planning/SERVING-BENCHMARK-REPORT.md` + W&B. Also verifies `max_lora_rank=64` + rank-32 adapter attach + concurrent `generate` from 16 threads.
- `benchmark` (6.8): against the **deployed** endpoint over HTTP (env: `MODAL_WEB_URL`, `MODAL_WEB_TOKEN`): warm container, synthetic chat requests (smoke-tier swebench instances from `data/golden.jsonl`), concurrency ramp 1→8→16; measures TTFB p50/p95 (acceptance: <500ms p50), tokens/sec, error rate, throughput, cost per inference; logs W&B summary + report section. Includes cold-start run: wait > idle window → single request, `cold_start = first_byte_latency - warmed_ttfb`, document in W&B + report.
- CLI: `python -m inference.benchmark sweep|benchmark --config ...`

---

## 6. Implementation Steps

### Phase A: Foundation — local-first, zero GPU (6.2 logic, 6.4, 6.4.1, 6.6)

Everything here runs on the laptop; Modal never touched in the debug loop (credit conservation).

1. **Packaging** (`pyproject.toml`)
   - Action: add `inference*` to `packages.find` include list; add `inference` to `mypy` files list and coverage source; retarget `[project.scripts] serve` → `inference.serve:app`; add `pytest` markers if needed (reuse `requires_modal`; no new markers)
   - Why: module must be importable, typed, covered, and the pre-registered `serve` script must point at a real ASGI app
   - Dependencies: none · Risk: Low

2. **`inference/config.py` + `config/models.yaml` serving entry** — `ServeConfig` per 5.1; add `serving_hf_id: "Qwen/Qwen3-14B-FP8"` (+ `serving_quantization: fp8`) to `models: qwen3-14b:`
   - Dependencies: step 1 · Risk: Low

3. **`inference/prompt_builder.py` extraction + `evaluation/inference.py` re-export shim**
   - Action: move the functions in 5.2 verbatim; in `evaluation/inference.py` replace bodies with imports + module-level re-exports (same names/signatures)
   - Why: D5 — one source of truth for prompt logic; eval compatibility contract keeps Phase 5 green
   - Dependencies: step 2 · Risk: **Medium** (eval regression) — mitigate: run `pytest tests/ -k "eval or inference"` + `scripts/debug_eval_one.py` local probe after extraction

4. **`inference/openai_compat.py`** — schemas, `resolve_engine_model`, `assemble_response`, `iter_chunks`, `error_response` per 5.3 (pure)
   - Dependencies: step 2 · Risk: Low

5. **`inference/serve.py` + `StubEngine`** — `create_app(engine)`: `/v1/chat/completions` non-stream + SSE stream (`EventSourceResponse`), guard clauses, OpenAI error bodies, `/health`; module-level `app`
   - Dependencies: steps 3, 4 · Risk: Low

6. **`inference/telemetry.py`** — `MetricsCollector` + W&B flush loop + GPU-util sampler + cost calc per 5.5 (W&B calls no-op when no run)
   - Dependencies: step 5 · Risk: Low

7. **Unit tests** (local, fake modal via existing `conftest.py`):
   - `tests/test_prompt_builder.py` — moved functions + eval re-export compat
   - `tests/test_openai_compat.py` — model mapping (all 4 accepted forms + unknown → 404), response/chunk schema shape, SSE chunk sequence (`role` chunk → content chunks → finish chunk → `[DONE]`), error body
   - `tests/test_serve.py` — FastAPI `TestClient` + `StubEngine`: 200 schema, stream event stream, 422 validation, unknown-model 404, engine-error 500 OpenAI body, `/health`
   - `tests/test_telemetry.py` — TTFB/tok-s aggregation, error-rate math, flush payload
   - `tests/test_config.py` — `SERVING_` env overrides
   - Dependencies: steps 4–6 · Risk: Low

### Phase B: Modal serving (6.2 + 6.3 + 6.7 preflight)

1. **`inference/modal_serve.py`** — Modal App per 5.6: image (cache-buster v1), `serve-model-cache` volume, `QwenServer` cls (`@modal.build` warmup fail-fast, `@modal.enter` engine init, `@modal.asgi_app`)
   - Dependencies: Phase A · Risk: Medium (image build once; only bump cache-buster on dep changes)

2. **Remote validation boot (ONE boot validates many things — credit conservation)**
   - Action: `modal serve inference.modal_serve`, then a single preflight script **`scripts/preflight_serve.py`** (openai SDK against the `modal serve` URL, env: `MODAL_WEB_URL`, `MODAL_WEB_TOKEN`): adapter load (rank-32 variant) + one non-stream chat + one stream + `/health` — not five separate boots
   - Verify FP8+LoRA works; if vLLM errors on LoRA-over-FP8 → switch `serving_hf_id`/`quantization` to AWQ build (bounded ~$5–15 fallback), re-run this step
   - Dependencies: step 8 · Risk: **High** (the D1 gamble — Path A working is the critical unknown)
   - Fail-fast rule: engine config errors surface in `@modal.build` warmup (30–60s of A10G), never at first request

### Phase C: 6.1 config sweep

 1. **`inference/benchmark.py sweep`** — matrix per 5.7; pick winner (`gpu_memory_utilization` × `max_num_seqs`), verify 16-thread concurrent `generate` + rank-32 attach; write `docs/planning/SERVING-BENCHMARK-REPORT.md` + W&B
    - Dependencies: step 9 (Path A proven) · Risk: Medium ($ <1 budget; smoke-tier prompts only)

### Phase D: Deploy + integration test (6.7)

 1. **Deploy** — `modal deploy inference.modal_serve` (only when endpoint stable); record web URL + Modal web token
 2. **Integration test** — `tests/test_serving_integration.py` (`@pytest.mark.requires_modal`, skipped unless `MODAL_WEB_URL`/`MODAL_WEB_TOKEN` set): openai SDK client → non-stream chat (verify `id/object/model/choices[0].message.content/usage` against OpenAI schema), stream=True (token-by-token SSE, `[DONE]`), unknown model → OpenAI-style 404, `/health`
    - Dependencies: step 11 · Risk: Low (skipped in CI without env)

### Phase E: 6.8 benchmark + scale-to-zero + W&B telemetry (6.5, 6.8, 6.9)

 1. **`benchmark.py benchmark`** — HTTP-level on deployed endpoint: TTFB p50 < 500ms acceptance gate, tokens/sec, error rate, cost per inference → W&B + report; cold-start measurement after idle window (documented, not fought — DoD risk note accepts >10s for V1)
 2. **Scale-to-zero verification** — idle → container terminated (Modal logs), confirm no GPU spend at idle ($0 idle per ADR-010), cold start < 15s target measured and logged; adjust `idle_timeout_seconds` if Modal default doesn't fit
    - Dependencies: step 12 · Risk: Low

### Phase F: Wrap

 1. **DoD checklist + report** — update this doc's status, fill SERVING-BENCHMARK-REPORT.md, note deferred decisions (embeddings, Langfuse, CI-driven modal deploy needing `MODAL_TOKEN_ID/SECRET` GitHub secrets — Phase 7 research note)
 2. **CI smoke wiring** — ensure `ci.yml` runs Phase A unit tests (no modal); integration stays manual/on-demand (`requires_modal`)

---

## 7. Testing Strategy

- **Unit (local, CI)**: Phase A step 7 suite — prompt_builder compat, OpenAI schema/SSE assembly, serve routes with `StubEngine`, telemetry math, config env overrides. Fake modal from `tests/conftest.py` already covers `modal` imports.
- **Integration (`requires_modal`)**: step 12 — openai SDK end-to-end against deployed endpoint, non-stream + stream + error shapes.
- **Benchmark**: step 10 (engine configs) + step 13 (HTTP acceptance) with smoke-tier sampling during debugging; full runs only as acceptance (credit conservation).
- **Regression**: after prompt_builder extraction, run the eval smoke suite (`pytest -m "not requires_modal"` + a local `debug_eval_one.py` probe) to prove the re-export shim.

## 8. Risks & Mitigations

- **Risk: vLLM LoRA-on-FP8 unsupported/buggy** (D1 gamble) — Mitigation: pre-planned AWQ fallback (bounded $5–15), probed in the single validation boot (step 9) before any sweep; GPTQ as last resort
- **Risk: cold start > 15s** — accepted for V1 per master plan; measure + document in W&B; mitigate via `VLLM_CACHE_ROOT` persistence (CUDA graph cache survives across boots) + HF weights cached on `serve-model-cache` volume; FP8 (~14GB) loads faster than bf16 (~28GB)
- **Risk: 32768 context not servable on 24GB** — default `max_model_len=16384`; sweep knob; document in report
- **Risk: eval regression from prompt_builder extraction** — re-export shim, same signatures, eval smoke suite after extraction
- **Risk: engine concurrency misbehavior (sync `LLM.generate` under FastAPI threadpool)** — verified in sweep step 10 (16 threads); cap `allow_concurrent_inputs=16` (Phase 5 aiohttp lesson)
- **Risk: silent prompt drift between eval and serving** — single shared `prompt_builder.py` (D5); eval re-exports are aliases, not copies
- **Risk: Modal lazy-registration / same-tag function-override gotchas** — engine/web registered at import time; one function per GPU tier
- **Risk: image rebuild churn (credit/cost)** — cache-buster bumped only on dep changes; weight-load-only iterations reuse cached image
- **Risk: GPU spend from open endpoint** — Modal web endpoint auth (D7) + scale-to-zero

## 9. Success Criteria (Phase 6 DoD)

- [ ] `modal serve inference.modal_serve` starts endpoint; `modal deploy` endpoint stable
- [ ] Any OpenAI SDK client works with `base_url=<web url>` + bearer token; non-stream chat returns valid OpenAI-schema JSON
- [ ] `stream: true` yields token-by-token SSE chunks ending in `data: [DONE]` per OpenAI spec
- [ ] TTFB p50 < 500ms under benchmark load (warm container)
- [ ] Scale-to-zero: $0 idle; cold start measured, documented in W&B + report
- [ ] 6.1 sweep report + selected config (`gpu_memory_utilization` × `max_num_seqs`) in `docs/planning/SERVING-BENCHMARK-REPORT.md`
- [ ] W&B serving metrics: TTFB, tokens/sec, request count, error rate, GPU util, cost per inference (ADR-005/011)
- [ ] Integration test (`requires_modal`) passes end-to-end; Phase A unit suite green in CI
- [ ] Eval F2P prompt path unchanged (re-export shim + smoke suite green)
- [ ] `inference/` complete: config, prompt_builder, openai_compat, serve, telemetry, modal_serve, benchmark — lint/typecheck clean (ruff 100-col, mypy)
