# API Reference — OpenAI-Compatible Inference

> SWE-Qwen serves an **OpenAI wire-compatible** API on Modal (`POST /v1/chat/completions`) with optional SSE streaming and **per-request LoRA adapter switching**. Any OpenAI SDK or `curl` works out of the box.

- Base URL: `https://<your-workspace>.modal.run` (dev: `https://<your-workspace>--swe-qwen-serve.modal.run`)
- Auth: Bearer token (`MODAL_SERVE_TOKEN`), compared with `hmac.compare_digest`, read lazily per request, fail-closed (missing env var or missing header → `401`).
- Implementation: `inference/openai_compat.py` (wire format) + `inference/serve.py` (FastAPI app).

---

## 1. Endpoints

| Method | Path | Auth | Description |
| ------ | ---- | ---- | ----------- |
| `POST` | `/v1/chat/completions` | Bearer | Chat completion; streaming via SSE when `"stream": true` |
| `GET` | `/health` | none | Liveness; returns serving model + engine |

`/health` response:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-14B-AWQ",
  "engine": "VLLMEngine"
}
```

---

## 2. Request

### `POST /v1/chat/completions`

```json
{
  "model": "qwen3-14b:higher_rank_14b",
  "messages": [
    { "role": "system", "content": "You are an expert software engineer." },
    { "role": "user", "content": "Fix this failing test.\n\n```python\ndef test_reverse():\n    assert reverse([1, 2, 3]) == [3, 2, 1]\n```\n" }
  ],
  "temperature": 0.1,
  "top_p": 0.95,
  "max_tokens": 1024,
  "stream": false,
  "stop": ["```"],
  "repetition_penalty": 1.05
}
```

### Fields

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `model` | `string` | — | Required. See [Model resolution](#model-resolution) |
| `messages` | `array<{role, content}>` | — | Required. `role` ∈ `system \| user \| assistant` |
| `temperature` | `number` | `0.1` | Sampling temperature |
| `top_p` | `number` | `0.95` | Nucleus sampling |
| `max_tokens` | `int` \| `null` | `null` | Max completion tokens; `null` → server default, capped by `max_tokens_cap` (registry context window) |
| `stream` | `boolean` | `false` | When `true`, respond with SSE chunks |
| `stop` | `string` \| `array<string>` \| `null` | `null` | Stop sequences |
| `repetition_penalty` | `number` \| `null` | `null` | Non-standard passthrough to vLLM |
| `swe_bench` | `object` \| `null` | `null` | Non-standard: server-side SWE-bench issue assembly (used by the eval harness) |

Unknown/extra fields are **ignored** (`model_config.extra = "ignore"`), so newer OpenAI SDKs keep working.

### Model resolution

The `model` field resolves through `resolve_engine_model` in this priority order:

| Value | Result |
| ----- | ------ |
| `qwen3-14b` | The configured **default variant** (the champion LoRA adapter); falls back to the raw base model with **no adapter** only when `default_variant` is unset |
| `qwen3-14b:<variant>` | Base + LoRA adapter for `<variant>` |
| `model-qwen3-14b-<variant>` | W&B artifact name form → base + LoRA adapter for `<variant>` |
| `<variant>` (bare) | Base + LoRA adapter for `<variant>` (must exist in `config.variants`) |

Unknown model or unknown variant → **404** `model_not_found`. If a known variant's adapter file cannot be resolved, the server logs a warning and serves the base model without an adapter (`lora_path=None`).

---

## 3. Responses

### 3.1 Non-streaming (`200 OK`)

```json
{
  "id": "chatcmpl-1f2e3d4a5b6c",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "Qwen/Qwen3-14B-AWQ",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "diff --git a/tests/test_reverse.py ..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 214, "completion_tokens": 96, "total_tokens": 310 }
}
```

### 3.2 Streaming (`200 OK`, SSE)

`stream: true` returns an SSE stream of `data:` frames. Sequence:

1. Role chunk: `{"object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}`
2. Content chunks: `{"object":"chat.completion.chunk","choices":[{"delta":{"content":"diff "}}]}`
3. Final chunk: `{"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}`
4. Termination: `data: [DONE]`

Each chunk is flushed as a complete `data: <json>\n\n` frame (SSE via `sse-starlette`). Content is chunked on whitespace-delimited segments (not token-level); TTFB is recorded at the first chunk.

### 3.3 Errors

`401` uses a FastAPI-style `detail` body — **not** the OpenAI envelope, and 401s are **not** recorded in telemetry:

```json
{ "detail": "invalid or missing bearer token" }
```

All other errors use the OpenAI shape:

```json
{
  "error": {
    "message": "model 'qwen3-14b:does-not-exist' not found",
    "type": "invalid_request_error",
    "param": null,
    "code": "model_not_found"
  }
}
```

| HTTP | Shape | Type | Trigger |
| ---- | ----- | ---- | ------- |
| `401` | FastAPI `detail` | — | Missing/invalid Bearer token (fail-closed, constant-time compare); not recorded in telemetry |
| `422` | OpenAI envelope | `invalid_request_error` | Payload fails Pydantic validation |
| `404` | OpenAI envelope | `invalid_request_error` (code `model_not_found`) | Unresolvable `model` string |
| `500` | OpenAI envelope | `server_error` | Engine/generation failure |

Request IDs use the OpenAI convention: `chatcmpl-` + 12 hex chars.

---

## 4. Engines

| Engine | Selection | Behaviour |
| ------ | --------- | --------- |
| `VLLMEngine` | `SERVING_STUB=0` | vLLM `LLM` over `Qwen/Qwen3-14B-AWQ` (AWQ int4), `enable_lora=True`, `max_lora_rank=64`, `gpu_memory_utilization=0.85`, `max_num_seqs=16`, `max_model_len` = registry `context_window` (32768 for `qwen3-14b`), `enforce_eager=False`. The `quantization` kwarg is applied only when the registry entry has `serving_hf_id` (`qwen3-14b` → AWQ int4 `Qwen/Qwen3-14B-AWQ`). Per-request `LoRARequest(lora_int_id=1, lora_path=...)`; the engine is cached process-singleton keyed by engine model id (not per variant) |
| `StubEngine` | default unless `SERVING_STUB=0` | Deterministic fake: `stub[{lora_name or "base"}]: ...{prompt[-40:]}` — no GPU, used by tests and `uvicorn inference.serve:app` |

### Prompt assembly

- **LoRA requests**: the raw `### Response -> patch` continuation prompt — system prompt + `\n\n` + user text — with the **first** `### Response` marker rewritten to `/no_think\n### Response` when the registry's `prompt_behavior.no_think` flag is set (the Qwen3 no-think soft switch, applied at inference only).
- **Base requests**: the tokenizer's chat template with `enable_thinking=False` (only for thinking-flagged models).

`/no_think` is **never** inserted by training or evaluation — the contract constant `NO_THINK_POLICY = "inference_wrapper_only"` pins it to the inference wrapper.

### Adapter resolution

For a request with a variant, the server checks `models/checkpoints/{variant}` locally first, then downloads the W&B artifact `model-{key}-{variant}:latest`. The first request per variant pulls the adapter (recorded pull: 1,399.16 MB / 40 files) and caches it on the `serve-model-cache` Modal volume — zero engine restarts.

---

## 5. Example Clients

### cURL

```bash
curl -s https://<workspace>.modal.run/v1/chat/completions \
  -H "Authorization: Bearer $MODAL_SERVE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b:baseline_14b",
       "messages":[{"role":"user","content":"Write a pytest for a queue implementation."}]}'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="https://<workspace>.modal.run/v1", api_key="modaltoken")
resp = client.chat.completions.create(
    model="qwen3-14b:higher_rank_14b",
    messages=[{"role": "user", "content": "Explain what F2P means in SWE-bench."}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

### Langchain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3-14b:baseline_14b",
    base_url="https://<workspace>.modal.run/v1",
    api_key="modaltoken",
)
```

---

## 6. Deployment & Local Dev

```bash
# Local: FastAPI with StubEngine (no GPU, deterministic)
uvicorn inference.serve:app --reload          # SERVING_STUB defaults to 1

# Modal dev: hot reload, scale-to-zero
modal serve inference.modal_serve

# Modal prod: deploy a persistent endpoint
modal deploy inference.modal_serve

# Preflight GPU smoke before deploy
python scripts/preflight_serve.py
```

Deployment is wired into CI: `cd.yml` deploys the serving app on merge to `main` (Modal). The `EvaluatorConfig`-adjacent `ServeConfig` reads env vars with a `SERVING_` prefix (see `inference/serve.py`).

---

## 7. Telemetry

Each request emits a `RequestRecord` (`inference/telemetry.py`):

| Field | Meaning |
| ------ | ------- |
| `ts` | `time.time()` at request completion |
| `model` | The `model` string as sent by the client |
| `stream` | Whether the request streamed |
| `ttfbs_ms` | Time to first streamed byte (streaming only) |
| `latency_ms` | End-to-end request latency |
| `output_tokens` | Completion tokens |
| `error` / `error_type` / `status` | Failure classification (404/422/500; 401 is not recorded) |

Metrics flow to **W&B** (latency/cost per inference) with **Langfuse** tracing: errors are traced 100%, successes are sampled at `telemetry_trace_sample_rate` (default 0.1).
