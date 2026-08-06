"""OpenAI-compatible wire schemas and pure formatting logic for the serving API.

Pure stdlib + pydantic only — no fastapi/vllm/modal imports — so the wire
format is unit-testable on macOS and importable anywhere the serving stack
runs.  The non-standard extensions (``repetition_penalty`` passthrough,
``swe_bench`` server-side prompt assembly) are ignored by strict OpenAI
clients and map to engine knobs / shared prompt building server-side.

Model resolution follows the Phase 6 decision: a request ``model`` accepts
``qwen3-14b``, ``qwen3-14b:{variant}``, bare ``{variant}``, or the W&B
artifact name ``model-qwen3-14b-{variant}``; no variant means the base model
without a LoRA adapter.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from inference import prompt_builder
from inference.config import ServeConfig

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """One chat turn as sent by an OpenAI client."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI ``/v1/chat/completions`` request body.

    Non-standard passthroughs: ``repetition_penalty`` (engine knob) and
    ``swe_bench`` (server-side SWE-bench prompt assembly via the shared
    ``prompt_builder``; ignored by strict OpenAI clients).
    """

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    repetition_penalty: float | None = None
    swe_bench: dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class ChatResponseMessage(BaseModel):
    """Assistant message in a non-streaming completion response."""

    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    """One completion choice (the API always returns exactly one)."""

    index: int
    message: ChatResponseMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    """Non-streaming completion response (``object="chat.completion"``)."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict[str, int]


class ChatChunkChoice(BaseModel):
    """One streaming chunk choice (delta holds role/content/finish_reason)."""

    index: int
    delta: dict[str, str]
    finish_reason: str | None


class ChatCompletionChunk(BaseModel):
    """Streaming completion chunk (``object="chat.completion.chunk"``)."""

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatChunkChoice]


ErrorBody = dict[str, Any]


class ModelNotFoundError(Exception):
    """Raised when a request ``model`` string matches no served model."""

    def __init__(self, model_name: str, status: int = 404) -> None:
        super().__init__(f"model {model_name!r} not found")
        self.model_name = model_name
        self.status = status


def create_request_id() -> str:
    """A short unique OpenAI-style request id (``chatcmpl-`` + hex)."""
    return "chatcmpl-" + uuid.uuid4().hex[:12]


def resolve_engine_model(
    request_model: str, config: ServeConfig
) -> tuple[str, str | None, str | None]:
    """Map a request ``model`` string to ``(hf_id, lora_name, lora_path)``.

    Accepted forms (see module docstring): the base model key, a variant
    suffix after ``:``, a bare variant, or the ``model-{base}-{variant}`` W&B
    artifact name.  An unknown model (or an unknown variant) raises
    ``ModelNotFoundError``; a known variant whose adapter cannot be resolved
    falls back to the base model with a warning.

    Args:
        request_model: The ``model`` field from a chat completion request.
        config: Serving config (variant registry + adapter resolution).

    Returns:
        ``(hf_id, lora_name, lora_path)``; lora fields are None for base.
    """
    base = config.base_model
    variant: str | None = None
    if request_model == base:
        pass
    elif request_model.startswith(f"{base}:"):
        variant = request_model[len(base) + 1 :]
    elif request_model.startswith(f"model-{base}-"):
        variant = request_model[len(f"model-{base}-") :]
    elif request_model in config.variants:
        variant = request_model
    else:
        raise ModelNotFoundError(request_model)

    if variant is not None and variant not in config.variants:
        raise ModelNotFoundError(request_model)

    hf_id = prompt_builder.resolve_hf_id(config.base_model)
    if variant is None:
        return hf_id, None, None
    lora_name = f"{base}-{variant}"  # matches eval's LoRARequest naming
    # resolve_adapter_path is typed for EvalConfig but only reads the
    # duck-typed fields ServeConfig mirrors (lora_artifact_pattern, wandb_*).
    lora_path = prompt_builder.resolve_adapter_path(variant, cast(Any, config))
    if lora_path is None:
        logger.warning("no LoRA adapter available for variant %s — serving base model", variant)
    return hf_id, lora_name, lora_path


def build_messages(request: ChatCompletionRequest) -> tuple[str, str]:
    """Split a chat request into ``(system_prompt, user_text)``.

    With ``swe_bench`` present the prompt is assembled server-side from the
    SWE-bench example dict via the shared ``prompt_builder`` (the OpenAI
    messages are ignored; system prompt is empty).  Otherwise the first
    system message is the system prompt and the remaining user/assistant
    turns are joined with blank lines.
    """
    if request.swe_bench:
        from evaluation.schema import EvalInput

        return "", prompt_builder.render_patch_prompt(
            example=EvalInput.from_swebench_record(request.swe_bench)
        )
    system_prompt = ""
    user_parts: list[str] = []
    for message in request.messages:
        if message.role == "system":
            system_prompt = message.content
        else:
            user_parts.append(message.content)
    return system_prompt, "\n\n".join(user_parts)


def assemble_response(
    request_id: str,
    created: int,
    model: str,
    content: str,
    usage: dict[str, int],
) -> ChatCompletionResponse:
    """Assemble a non-streaming completion response."""
    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


def iter_chunks(
    request_id: str, created: int, model: str, content_iter: Iterator[str]
) -> Iterator[str]:
    """Yield OpenAI SSE frames for one streaming completion.

    Frame order: a role chunk (``delta={"role": "assistant"}``), one content
    chunk per piece from ``content_iter`` (``delta={"content": piece}``), a
    final chunk (``delta={}``, ``finish_reason="stop"``), then
    ``data: [DONE]``.  Every yielded item is a complete SSE frame
    (``data: {json}\\n\\n``, json with ``ensure_ascii=False``).
    """

    def frame(chunk: ChatCompletionChunk) -> str:
        return f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"

    yield frame(
        ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[ChatChunkChoice(index=0, delta={"role": "assistant"}, finish_reason=None)],
        )
    )
    for piece in content_iter:
        yield frame(
            ChatCompletionChunk(
                id=request_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[ChatChunkChoice(index=0, delta={"content": piece}, finish_reason=None)],
            )
        )
    yield frame(
        ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[ChatChunkChoice(index=0, delta={}, finish_reason="stop")],
        )
    )
    yield "data: [DONE]\n\n"


def error_response(
    status: int, message: str, error_type: str, code: str | None = None
) -> tuple[int, ErrorBody]:
    """Build an OpenAI-style error body: ``{"error": {message, type, param, code}}``."""
    return status, {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
