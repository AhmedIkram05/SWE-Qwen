"""Batch patch generation for the SWE-Qwen evaluation harness.

Runs vLLM (optionally with a LoRA adapter) in a Modal container to generate
candidate fix patches for a batch of evaluation instances.  Prompts are
rendered with the shared ``training.prompt_loader.PromptLoader`` templates.

Prompt construction was extracted to ``inference.prompt_builder`` (Phase 6,
Wave 1) so served inference and eval F2P share one implementation; every
moved name is re-exported here so scripts/tests/harness imports keep working
unchanged.  vLLM is only imported inside the Modal function because it is
Linux-only.

The LLM engine is cached per (model, variant) so the container re-uses the
same loaded model across ``generate_patches_batch`` calls instead of
cold-starting vLLM (engine init + 28 GB model download + CUDA graph capture)
on every invocation.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

# Phase 6 Wave 1: prompt logic extracted to inference.prompt_builder (shared
# with the serving API).  Re-exported here so every existing name stays
# importable from evaluation.inference — scripts, harness, and tests keep
# working unchanged; tests monkeypatch these names on THIS module, which
# prompt_builder honors via its _eval() bridge.
import inference.prompt_builder as _prompt_builder

DEFAULT_TEMPLATE = _prompt_builder.DEFAULT_TEMPLATE
_DEFAULT_HF_ID = _prompt_builder._DEFAULT_HF_ID
_DIFF_FILE_RE = _prompt_builder._DIFF_FILE_RE
_GOLDEN_INDEX = _prompt_builder._GOLDEN_INDEX
_GOLDEN_PATH = _prompt_builder._GOLDEN_PATH
_GOLDEN_SOURCE = _prompt_builder._GOLDEN_SOURCE
_PATH_RE = _prompt_builder._PATH_RE
_PROMPTS_DIR = _prompt_builder._PROMPTS_DIR
_TOKENIZER_CACHE = _prompt_builder._TOKENIZER_CACHE
_ensure_golden = _prompt_builder._ensure_golden
_fetch_raw_file = _prompt_builder._fetch_raw_file
_files_from_diff = _prompt_builder._files_from_diff
_file_snippets = _prompt_builder._file_snippets
_golden_patches = _prompt_builder._golden_patches
_no_think_wrap = _prompt_builder._no_think_wrap
fetch_raw_file = _prompt_builder.fetch_raw_file
file_snippets = _prompt_builder.file_snippets
golden_patches = _prompt_builder.golden_patches
no_think_wrap = _prompt_builder.no_think_wrap
render_patch_prompt = _prompt_builder.render_patch_prompt
resolve_adapter_path = _prompt_builder.resolve_adapter_path
resolve_hf_id = _prompt_builder.resolve_hf_id

if TYPE_CHECKING:
    from evaluation.schema import EvalInput

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = _REPO_ROOT / "training"
_EVAL_DIR = _REPO_ROOT / "evaluation"
_CONFIG_DIR = _REPO_ROOT / "config"
_INFERENCE_DIR = _REPO_ROOT / "inference"

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("swe-qwen-eval-inference")

# Phase 5 plan doc said Image.from_registry("vllm/vllm-openai:latest"), but that
# image is unusable on Modal: it has no `python` binary in PATH (pip install
# fails with exit 127) and Modal cannot detect its Python version, so
# FunctionCreate fails. Rebuilt from debian_slim per training/modal_train.py
# precedent; vllm>=0.26.0 ships manylinux wheels, so no build-essential needed.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install("vllm>=0.26.0", "peft", "wandb", "trl")
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc at startup; the
    # Modal container has no CUDA toolkit. vLLM 0.26 honors
    # VLLM_USE_FLASHINFER_SAMPLER=0 to fall back to the torch sampler.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    # Cache models into the Modal volume so container restarts reuse the
    # 28 GB model instead of re-downloading from Hugging Face.
    .env({"HF_HOME": "/models"})
    # Persist vLLM's torch.compile + CUDA graph caches on the volume too.
    # Without this every container recompiles from scratch: 54-107s of
    # torch.compile + ~35s graph capture per container (observed 4x in one
    # dev-tier run). Cache dirs on the ephemeral /root are lost per restart.
    .env({"VLLM_CACHE_ROOT": "/models/vllm-cache"})
    .add_local_dir(str(_TRAINING_DIR), remote_path="/root/training", copy=True)
    .add_local_dir(str(_EVAL_DIR), remote_path="/root/evaluation", copy=True)
    .add_local_dir(str(_CONFIG_DIR), remote_path="/root/config", copy=True)
    # Phase 6 Wave 1: evaluation.inference now imports inference.prompt_builder
    # at module level, so the container needs the package baked in.
    .add_local_dir(str(_INFERENCE_DIR), remote_path="/root/inference", copy=True)
)
# Gold patch diffs for few-shot prompting (see _golden_patches). Baked only
# when present: data/golden.jsonl is gitignored (CI has no data dir — image
# build used to FileNotFoundError), and the container fetches golden from GCS
# at runtime (_ensure_golden), so the bake is just an air-gap fallback.
if (_REPO_ROOT / "data" / "golden.jsonl").is_file():
    vllm_image = vllm_image.add_local_file(
        str(_REPO_ROOT / "data" / "golden.jsonl"), remote_path="/root/data/golden.jsonl"
    )

model_volume = modal.Volume.from_name("eval-model-cache", create_if_missing=True)

# ── GPU type mapping ──────────────────────────────────────────────────────
# GPU type hardcoded to A10G:1; change here + EvalConfig.gpu_type to switch to A100-80GB.
_GPU_MAP: dict[str, str] = {
    "a10g-24gb": "A10G:1",
    "a100-80gb": "A100-80GB",
}

# ── LLM singleton cache ───────────────────────────────────────────────────
# ponytail: process-level singleton keyed by model_name only (NOT variant).
# vLLM LLM() is expensive to construct (28 GB model load + CUDA graph capture).
# Cache ONE LLM per base model; all variants share it via per-call LoRA requests.
# vLLM 0.26+ loads LoRA adapters on-demand: set enable_lora=True and pass
# lora_request or None per generate call.  This avoids 3× model loads when
# evaluating 3 variants on the same base model.
_LLM_CACHE: dict[str, Any] = {}
_LLM_LOCK = threading.Lock()


def _get_llm(
    model_name: str,
    adapter_path: str | None = None,
    eager: bool = False,
) -> Any:
    """Return a cached vLLM ``LLM`` instance for *model_name* (cached once).

    First call loads the model (expensive). Subsequent calls (any variant)
    return the same instance.  ``enable_lora=True`` so both baseline (None)
    and LoRA variants work on the same loaded model.

    Args:
        model_name: ``models.yaml`` key or full Hugging Face ID.
        adapter_path: Ignored at load time (LoRA loaded per-call via
            ``lora_request`` on ``llm.generate()``).
        eager: Skip torch.compile + CUDA graph capture (saves ~150s of cold
            start for probe/smoke batches).  Only honoured on the first load;
            a cached engine is reused as-is so a container never holds two
            LLM instances for one model (27.5 GiB each — OOM risk).

    Returns:
        A ``vllm.LLM`` instance (cached after first creation).
    """
    from vllm import LLM

    with _LLM_LOCK:
        if model_name in _LLM_CACHE:
            return _LLM_CACHE[model_name]

        llm = LLM(
            model=resolve_hf_id(model_name),
            enable_lora=True,  # allow both LoRA and non-LoRA generations
            gpu_memory_utilization=0.85,
            # higher_rank_14b adapter is rank 32; vLLM defaults max_lora_rank
            # to 16 and kills the EngineCore with "LoRA rank 32 is greater
            # than max_lora_rank 16" when the adapter is first loaded.
            max_lora_rank=64,
            enforce_eager=eager,
        )
        # ponytail: one LLM per base model; LoRA adapters loaded on-demand
        # per generate() call via lora_request param.
        _LLM_CACHE[model_name] = llm
        return llm


# ── Pure helpers (no Modal / vLLM) ────────────────────────────────────────────
# extract_patch lives here; the other pure helpers (_files_from_diff, fetch
# + snippets, golden loaders, render_patch_prompt, resolve_hf_id,
# resolve_adapter_path) moved to inference.prompt_builder (re-exported above).


def extract_patch(text: str) -> str:
    """Extract a unified diff from model output.

    Handles reasoning text before the diff, multiple code blocks, and fence variants.
    """
    # Try ```diff fenced blocks — take LAST (model often reasons first, diff last)
    diffs: list[str] = re.findall(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if diffs:
        patch = diffs[-1].strip()
        if patch:
            patch = patch + "\n"
            if patch.startswith("diff --git") or patch.startswith("---") or patch.startswith("+++"):
                return patch
    # Try any fenced block — take LAST
    blocks: list[str] = re.findall(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        patch = blocks[-1].strip()
        if patch:
            patch = patch + "\n"
            if patch.startswith("diff --git") or patch.startswith("---") or patch.startswith("+++"):
                return patch
    # No fences: scan for diff --git anywhere, take from there
    idx = text.rfind("diff --git")
    if idx >= 0:
        patch = text[idx:].strip()
        return patch + "\n" if patch else ""
    # Last resort: return as-is (with trailing newline)
    patch = text.strip()
    return patch + "\n" if patch else ""


# ── Lazy-registered inference functions (one per GPU config) ──────────────

# ponytail: Modal 1.5.x requires ``gpu`` at decoration time; we register one
# function per GPU tier lazily so the harness can switch via config.inference_gpu.
# A100-80GB is required for 14B bf16; A10G-24GB fits ≤7B quantized models.
_inference_fns: dict[str, Any] = {}


def _get_inference_fn(gpu: str) -> Any:
    """Return the pre-registered inference function for *gpu*.

    Functions are registered at module import time (after
    ``_generate_patches_batch_body``, see the dict at the bottom of this
    module), BEFORE the harness enters ``app.run()``. Modal 1.5.x hydrates
    only functions registered at run-entry; registering lazily on a running
    app leaves the handle unhydrated and the first ``.remote()`` raises
    ExecutionError. Registering the module-level body also avoids the
    closure-serialization InvalidError.

    Args:
        gpu: Modal GPU spec (e.g. ``"A100-80GB"``, ``"A10G:1"``).
    """
    fn = _inference_fns.get(gpu)
    if fn is None:
        raise ValueError(f"no inference function registered for GPU {gpu!r}")
    return fn


def generate_patches_batch(  # noqa: PLR0913, PLR0917
    model_name: str,
    variant: str,
    prompt_template: str,
    examples: list[EvalInput],
    max_new_tokens: int = 8192,
    temperature: float = 0.1,
    top_p: float = 0.95,
    dataset_run_id: str | None = None,
) -> list[str]:
    """Generate candidate fix patches for a batch of eval examples.

    Dispatches to the GPU tier configured by ``EvalConfig.inference_gpu``.
    See ``_generate_patches_batch_body`` for full docs.
    """
    from evaluation.config import EvalConfig

    config = EvalConfig()
    gpu = _GPU_MAP.get(config.inference_gpu, "A100-80GB")
    fn = _get_inference_fn(gpu)
    return fn.remote(  # type: ignore[no-any-return]
        model_name,
        variant,
        prompt_template,
        examples,
        max_new_tokens,
        temperature,
        top_p,
        dataset_run_id,
    )


# ── Core generation body (shared by all GPU tiers) ────────────────────────


def _generate_patches_batch_body(  # noqa: PLR0913, PLR0917
    model_name: str,
    variant: str,
    prompt_template: str,
    examples: list[EvalInput],
    max_new_tokens: int = 8192,
    temperature: float = 0.1,
    top_p: float = 0.95,
    dataset_run_id: str | None = None,
) -> list[str]:
    """Generate candidate fix patches for a batch of eval examples (body).

    1. Resolve the LoRA adapter for ``variant`` (``None`` for baseline).
    2. Load the model (``resolve_hf_id(model_name)``) with vLLM, enabling LoRA
       if an adapter exists.
    3. Render prompts from the eval inputs via ``PromptLoader``.
    4. Generate with the given sampling parameters.
    5. Strip markdown code fences from completions.

    Args:
        model_name: ``models.yaml`` key (e.g. ``"qwen3-14b"``).
        variant: Trained variant key (adapter lookup; baseline if unresolved).
        prompt_template: Template name for ``render_patch_prompt``.
        examples: Eval instances to generate patches for.
        max_new_tokens: Maximum completion length.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        dataset_run_id: Pipeline run id — the few-shot golden patches are
            fetched from that run's GCS golden (``_ensure_golden``), keeping
            GCS the single source of truth instead of the image-baked file.

    Returns:
        List of patch strings, same order as ``examples``.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    from evaluation.config import EvalConfig

    config = EvalConfig()
    if dataset_run_id:
        # GCS golden is the source of truth; rebuild the few-shot index from it.
        global _GOLDEN_SOURCE, _GOLDEN_INDEX  # noqa: PLW0603
        src = _ensure_golden(dataset_run_id)
        if src != _GOLDEN_SOURCE:
            _GOLDEN_SOURCE = src
            _GOLDEN_INDEX = None
    adapter_path = resolve_adapter_path(variant, config)
    if adapter_path:
        logger.info("using LoRA adapter for variant %s: %s", variant, adapter_path)
    else:
        logger.info("no LoRA adapter for variant %s — using base model %s", variant, model_name)

    # ponytail: engine compile pays off for big batches; probe/smoke tiers
    # (<32 prompts) get enforce_eager instead — ~150s less cold start per run
    # (~$0.14 saved).  First load wins; a cached engine is never rebuilt.
    eager = len(examples) < 32  # noqa: PLR2004
    llm = _get_llm(model_name, eager=eager)
    hf_id = resolve_hf_id(model_name)
    rendered = [
        render_patch_prompt(example, template_name=prompt_template, include_file_contents=True)
        for example in examples
    ]
    # Adapters were trained on raw "### Response -> patch" continuation
    # (tokenize.format_training_prompt); chat-wrapping that breaks the contract
    # and produced repetition loops. Only the untrained base model gets the
    # no-think wrap, to stop it rambling "Okay, let's see..." preamble.
    prompts = [
        _no_think_wrap(hf_id, p)
        if adapter_path is None
        # Qwen3 soft switch: "/no_think" as the last user-turn line suppresses
        # thinking even in raw continuation, where enable_thinking can't reach.
        else p.replace("### Response", "/no_think\n### Response", 1)
        for p in rendered
    ]
    # ponytail: no repetition_penalty => decoding degeneracy (the model fell
    # into a ~1000x "```" fence loop eating the whole 8192 budget). 1.15 breaks
    # self-repetition; bump to ~1.3 if loops persist.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.15,
    )
    # ponytail: vLLM 0.26+ loads LoRA on-demand via lora_request on generate().
    # The cached LLM only needs enable_lora=True; the adapter is loaded first use.
    lora_req = (
        LoRARequest(
            lora_name=f"{model_name}-{variant}",
            lora_int_id=1,
            lora_path=adapter_path,
        )
        if adapter_path is not None
        else None
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
    return [extract_patch(output.outputs[0].text) for output in outputs]


# Testing backward-compat: .local() calls the body directly (no Modal container)
generate_patches_batch.local = _generate_patches_batch_body  # type: ignore[attr-defined]


# ── Modal function registration (module level, BEFORE app.run() entry) ────
# Modal 1.5.x hydrates only functions registered at run-entry
# (_create_all_objects snapshots local_state.functions when app.run() is
# entered). Register one function per GPU tier at import time, sharing the
# module-level body, so harness._ensure_app_running hydrates them. Registering
# lazily on a running app (as this file did before) leaves the handle
# unhydrated and the first .remote() raises ExecutionError.
# Distinct wrapper names per tier: registering the SAME function object under
# one tag makes Modal override the server-side spec by registration order —
# silent wrong-GPU risk; the wrapper also keeps each tier in global scope.


def _infer_a10g(*args: Any, **kwargs: Any) -> Any:
    """A10G-24GB tier (≤7B quantized models)."""
    return _generate_patches_batch_body(*args, **kwargs)


def _infer_a100(*args: Any, **kwargs: Any) -> Any:
    """A100-80GB tier (14B bf16)."""
    return _generate_patches_batch_body(*args, **kwargs)


_inference_fns = {
    "A10G:1": app.function(
        image=vllm_image,
        gpu="A10G:1",
        volumes={"/models": model_volume},
        timeout=1800,  # 8K-token generation at eager-mode ~20-40 tok/s needs >10 min
        secrets=[
            modal.Secret.from_name("wandb-secret"),
            modal.Secret.from_name("hf-secret"),
        ],
    )(_infer_a10g),
    "A100-80GB": app.function(
        image=vllm_image,
        gpu="A100-80GB",
        volumes={"/models": model_volume},
        timeout=1800,  # 8K-token generation at eager-mode ~20-40 tok/s needs >10 min
        secrets=[
            modal.Secret.from_name("wandb-secret"),
            modal.Secret.from_name("hf-secret"),
        ],
    )(_infer_a100),
}
