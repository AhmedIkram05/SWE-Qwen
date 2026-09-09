"""FastAPI serving app: OpenAI-compatible chat completions over vLLM.

``create_app`` wires the pure wire-format logic (``openai_compat``) and the
shared prompt builder (``prompt_builder``) to a pluggable engine: the real
``VLLMEngine`` (lazy vLLM import — Linux-only, never imported on macOS) or a
deterministic ``StubEngine`` for local dev and tests.

Local dev: ``uvicorn inference.serve:app`` (module-level ``app``) uses the
stub engine unless ``SERVING_STUB=0`` selects the vLLM engine.  Sync route
handlers run in FastAPI's threadpool; vLLM's sync engine is thread-safe and
batches internally.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import random
import threading
import time
from collections.abc import Iterator
from typing import Any, Protocol

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sse_starlette import EventSourceResponse

from inference import prompt_builder
from inference.config import ServeConfig
from inference.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelNotFoundError,
    assemble_response,
    build_messages,
    create_request_id,
    error_response,
    iter_chunks,
    resolve_engine_model,
)
from inference.telemetry import MetricsCollector, RequestRecord, add_trace_record
from observability.logging import configure_logging
from registry.loader import load_models

logger = logging.getLogger(__name__)


class GenerationResult:
    """One engine generation: text plus token accounting."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class Engine(Protocol):
    """Minimal generation contract implemented by both engines."""

    def generate(  # noqa: PLR0913
        self,
        prompt: str,
        *,
        lora: tuple[str, str] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
        repetition_penalty: float,
    ) -> GenerationResult: ...


# ── vLLM engine ────────────────────────────────────────────────────────────
# ponytail: process-level singleton keyed by the engine model id only (NOT
# variant).  vLLM LLM() is expensive to construct (model load + CUDA graph
# capture); ONE engine serves all variants via per-request LoRARequest.  Same
# shape as evaluation/inference.py's _get_llm (the proven Phase 5 pattern).
_LLM_CACHE: dict[str, Any] = {}
_LLM_LOCK = threading.Lock()


class VLLMEngine:
    """Real inference engine backed by vLLM (lazy import; Linux-only)."""

    def __init__(self, config: ServeConfig) -> None:
        self.config = config

    def _engine_spec(self) -> tuple[str, str | None]:
        """Resolve ``(engine model id, quantization)`` for this config.

        Resolution is env > registry > fallback via the resolved
        ``ServeConfig`` fields (never the raw registry — env precedence must
        survive into the engine + singleton cache key).  ``serving_hf_id``
        non-empty → the pre-quantized base (AWQ/FP8 path).  Empty → the plain
        base model (registry ``hf_id``) with no quantization kwarg — LoRA
        adapters serve on top of the unquantized base (native bf16/fp16).
        Unknown ``base_model`` → KeyError (fail loud, no silent fallback).
        """
        spec = load_models()[self.config.base_model]
        if self.config.serving_hf_id:
            return self.config.serving_hf_id, self.config.quantization
        return spec.hf_id, None

    def _ensure_engine(self) -> Any:
        """Return the process-singleton vLLM LLM for this config's model."""
        from vllm import LLM

        key, quantization = self._engine_spec()
        with _LLM_LOCK:
            if key in _LLM_CACHE:
                return _LLM_CACHE[key]
            llm_kwargs: dict[str, Any] = {
                "model": key,
                "enable_lora": True,  # allow both LoRA and non-LoRA generations
                "max_lora_rank": self.config.max_lora_rank,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_num_seqs": self.config.max_num_seqs,
                "max_model_len": self.config.max_model_len,
                "enforce_eager": self.config.enforce_eager,
            }
            if quantization:
                # Only the pre-quantized base path (validated against vLLM at
                # load time); the plain base gets vLLM's native dtype.
                llm_kwargs["quantization"] = quantization
            llm = LLM(**llm_kwargs)
            _LLM_CACHE[key] = llm
            return llm

    def generate(  # noqa: PLR0913
        self,
        prompt: str,
        *,
        lora: tuple[str, str] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
        repetition_penalty: float,
    ) -> GenerationResult:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        llm = self._ensure_engine()
        lora_request = (
            LoRARequest(lora_name=lora[0], lora_int_id=1, lora_path=lora[1])
            if lora is not None
            else None
        )
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            repetition_penalty=repetition_penalty,
        )
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        output = outputs[0]
        return GenerationResult(
            text=output.outputs[0].text,
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
        )


class StubEngine:
    """Deterministic fake engine for local dev and tests. No GPU, no vLLM.

    Generates canned text derived from the prompt; used by ``uvicorn
    inference.serve:app`` by default so local dev never touches vLLM.
    """

    def generate(  # noqa: PLR0913
        self,
        prompt: str,
        *,
        lora: tuple[str, str] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
        repetition_penalty: float,
    ) -> GenerationResult:
        model_tag = lora[0] if lora is not None else "base"
        logger.info("StubEngine generate (model=%s)", model_tag)
        text = f"stub[{model_tag}]: ...{prompt[-40:]}"
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )


# ── Prompt assembly ────────────────────────────────────────────────────────


def _build_prompt(
    hf_id: str,
    system_prompt: str,
    user_text: str,
    lora: tuple[str, str] | None,
) -> str:
    """Assemble the engine prompt.

    LoRA adapters were trained on the raw "### Response -> patch"
    continuation; chat-wrapping breaks that contract (repetition loops — eval
    lesson).  The Qwen3 no-think gates (``/no_think`` insertion and
    ``enable_thinking=False``) key off the registry's
    ``prompt_behavior.no_think`` flag inside ``prompt_builder``, shared with
    the eval harness.
    """
    if lora is not None:
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(prompt_builder.apply_no_think_raw(hf_id, user_text))
        return "\n\n".join(parts)
    return prompt_builder.chat_wrap(hf_id, system_prompt, user_text)


# ── App factory ────────────────────────────────────────────────────────────

# Module-level collector: shared by every create_app() instance so the Wave 3
# flush loop / benchmark can read the same records.
_collector = MetricsCollector()


def _record_and_trace(
    rec: RequestRecord, config: ServeConfig, template_name: str | None = None
) -> None:
    """Record into the shared collector; sample successful requests for Langfuse.

    Phase 8 decision 1: a sampled (non-error) request is queued for the
    Langfuse drain — one bounded deque append, O(1), never blocks, never
    raises, so the serving hot path is unaffected.  ``template_name`` is the
    prompt-builder template (None here: the serving path never renders
    ``prompt_builder.render_prompt`` templates).
    """
    _collector.record(rec)
    if rec.error or random.random() >= config.telemetry_trace_sample_rate:
        return
    add_trace_record(
        model=rec.model,
        template_name=template_name,
        ttfbs_ms=rec.ttfbs_ms,
        latency_ms=rec.latency_ms,
        output_tokens=rec.output_tokens,
    )


def _default_engine() -> Engine:
    """Stub engine by default; ``SERVING_STUB=0`` selects the vLLM engine."""
    if os.environ.get("SERVING_STUB", "1") == "0":
        return VLLMEngine(ServeConfig())
    return StubEngine()


def _is_authorized(authorization: str | None) -> bool:
    """Exact ``Bearer <MODAL_SERVE_TOKEN>`` check; fail-closed when unset.

    Token is read lazily per request (no module-level caching) so tests can
    set/unset the env per test; comparison is constant-time.
    """
    expected = os.environ.get("MODAL_SERVE_TOKEN")
    if expected is None or authorization is None:
        return False
    scheme, _, token = authorization.partition(" ")
    return scheme == "Bearer" and hmac.compare_digest(token, expected)


def _stream_gen(  # noqa: PLR0913, PLR0917
    engine: Engine,
    request: ChatCompletionRequest,
    config: ServeConfig,
    hf_id: str,
    lora: tuple[str, str] | None,
    max_tokens: int,
    request_id: str,
    created: int,
    t0: float,
) -> Iterator[str]:
    """SSE generator: build prompt, generate, stream word chunks, telemetry.

    Generation errors are streamed to the client as an OpenAI-style error
    frame followed by ``[DONE]``; telemetry is recorded exactly once in the
    ``finally`` block (TTFB = time of the first yielded chunk).
    """
    ttfbs_ms: float | None = None
    output_tokens = 0
    error_type: str | None = None
    try:
        system_prompt, user_text = build_messages(request)
        prompt = _build_prompt(hf_id, system_prompt, user_text, lora)
        result = engine.generate(
            prompt,
            lora=lora,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            repetition_penalty=request.repetition_penalty
            if request.repetition_penalty is not None
            else config.repetition_penalty,
        )
        # ponytail: full token-streaming via a future engine stream API;
        # V1 streams word chunks of the completed text.
        pieces = result.text.split() or [""]
        for frame in iter_chunks(request_id, created, request.model, iter(pieces)):
            if ttfbs_ms is None:
                ttfbs_ms = (time.perf_counter() - t0) * 1000.0
            # iter_chunks yields full "data: ...\n\n" SSE frames; sse-starlette
            # adds its own "data: " framing, so hand it the raw payloads.
            yield frame.removeprefix("data: ").rstrip("\n")
        output_tokens = result.completion_tokens
    except RuntimeError:
        # Client disconnect / request cancellation: the ASGI server closes the
        # in-flight generator and raises INTO the suspended yield.  Expected
        # teardown (e.g. Modal's queue cancels requests arriving during cold
        # boot) — swallow it, no error frames (the client is gone), keep the
        # container alive.  Telemetry records it as "cancelled".
        error_type = "cancelled"
        logger.debug("streaming request cancelled (model=%s)", request.model)
    except Exception:
        error_type = "engine_error"
        logger.exception("streaming generation failed for model=%s", request.model)
        _, body = error_response(500, "internal generation failure", "server_error")
        with contextlib.suppress(Exception):
            yield json.dumps(body, ensure_ascii=False)
            yield "[DONE]"
    finally:
        _record_and_trace(
            RequestRecord(
                ts=time.time(),
                model=request.model,
                stream=True,
                ttfbs_ms=ttfbs_ms,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                output_tokens=output_tokens,
                error=error_type is not None,
                error_type=error_type,
                status=500 if error_type is not None else 200,
            ),
            config,
        )


def create_app(engine: Engine, config: ServeConfig | None = None) -> FastAPI:
    """Build the FastAPI app around *engine*.

    Routes: ``POST /v1/chat/completions`` (stream + non-stream, OpenAI wire
    format) and ``GET /health``.  Errors map to OpenAI-style bodies: 422 for
    invalid requests, 404 for unknown models, 500 for engine failures.
    """
    if config is None:
        config = ServeConfig()
    app = FastAPI(title="swe-qwen-inference", version="0.1.0")

    def _record(  # noqa: PLR0913
        model: str,
        *,
        stream: bool,
        ttfbs_ms: float | None,
        output_tokens: int,
        error: bool,
        error_type: str | None,
        status: int,
        t0: float,
    ) -> None:
        _record_and_trace(
            RequestRecord(
                ts=time.time(),
                model=model,
                stream=stream,
                ttfbs_ms=ttfbs_ms,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                output_tokens=output_tokens,
                error=error,
                error_type=error_type,
                status=status,
            ),
            config,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": config.serving_hf_id,
            "engine": engine.__class__.__name__,
        }

    @app.post("/v1/chat/completions", response_model=None)
    def chat_completions(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> ChatCompletionResponse | EventSourceResponse | JSONResponse:
        """OpenAI-compatible chat completions (stream + non-stream)."""
        # Phase 9: bearer auth on this route only; /health stays open (static
        # liveness for infra probes). Check runs before payload validation so
        # the stream and non-stream paths are both gated.
        if not _is_authorized(authorization):
            return JSONResponse(
                status_code=401, content={"detail": "invalid or missing bearer token"}
            )
        t0 = time.perf_counter()
        try:
            request = ChatCompletionRequest.model_validate(payload)
        except ValidationError:
            _, body = error_response(422, "invalid request body", "invalid_request_error")
            _record(
                "unknown",
                stream=False,
                ttfbs_ms=None,
                output_tokens=0,
                error=True,
                error_type="validation_error",
                status=422,
                t0=t0,
            )
            return JSONResponse(status_code=422, content=body)

        try:
            hf_id, lora_name, lora_path = resolve_engine_model(request.model, config)
        except ModelNotFoundError:
            _, body = error_response(
                404,
                f"model {request.model!r} not found",
                "invalid_request_error",
                code="model_not_found",
            )
            _record(
                request.model,
                stream=False,
                ttfbs_ms=None,
                output_tokens=0,
                error=True,
                error_type="model_not_found",
                status=404,
                t0=t0,
            )
            return JSONResponse(status_code=404, content=body)

        lora = (lora_name, lora_path) if lora_name is not None and lora_path is not None else None
        max_tokens = min(request.max_tokens or config.default_max_tokens, config.max_tokens_cap)
        request_id = create_request_id()
        created = int(time.time())

        if request.stream:
            return EventSourceResponse(
                _stream_gen(
                    engine, request, config, hf_id, lora, max_tokens, request_id, created, t0
                ),
                headers={"Cache-Control": "no-cache"},
            )

        try:
            system_prompt, user_text = build_messages(request)
            prompt = _build_prompt(hf_id, system_prompt, user_text, lora)
            result = engine.generate(
                prompt,
                lora=lora,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                repetition_penalty=request.repetition_penalty
                if request.repetition_penalty is not None
                else config.repetition_penalty,
            )
        except Exception:
            logger.exception("generation failed for model=%s", request.model)
            _, body = error_response(500, "internal generation failure", "server_error")
            _record(
                request.model,
                stream=False,
                ttfbs_ms=None,
                output_tokens=0,
                error=True,
                error_type="engine_error",
                status=500,
                t0=t0,
            )
            return JSONResponse(status_code=500, content=body)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record(
            request.model,
            stream=False,
            ttfbs_ms=elapsed_ms,
            output_tokens=result.completion_tokens,
            error=False,
            error_type=None,
            status=200,
            t0=t0,
        )
        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
        return assemble_response(request_id, created, request.model, result.text, usage)

    return app


# Module-level app for `uvicorn inference.serve:app` and the `serve` console
# script (local dev; stub unless SERVING_STUB=0).
configure_logging()
app = create_app(_default_engine(), ServeConfig())
