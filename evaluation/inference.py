"""Batch patch generation for the SWE-Qwen evaluation harness.

Runs vLLM (optionally with a LoRA adapter) in a Modal container to generate
candidate fix patches for a batch of evaluation instances. Prompts are
rendered with the shared ``training.prompt_loader.PromptLoader`` templates.

The core logic (``render_patch_prompt``, ``extract_patch``,
``resolve_adapter_path``, ``resolve_hf_id``) is plain Python and importable
without Modal; vLLM is only imported inside the Modal function because it is
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

if TYPE_CHECKING:
    from evaluation.config import EvalConfig
    from evaluation.schema import EvalInput

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = _REPO_ROOT / "training"
_EVAL_DIR = _REPO_ROOT / "evaluation"
_CONFIG_DIR = _REPO_ROOT / "config"
_PROMPTS_DIR = _TRAINING_DIR / "prompts"

DEFAULT_TEMPLATE = "chat"
_DEFAULT_HF_ID = "Qwen/Qwen3-14B"

_DIFF_FILE_RE = re.compile(r"^diff --git a/\S+ b/(\S+)", re.MULTILINE)


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
    .add_local_dir(str(_TRAINING_DIR), remote_path="/root/training", copy=True)
    .add_local_dir(str(_EVAL_DIR), remote_path="/root/evaluation", copy=True)
    .add_local_dir(str(_CONFIG_DIR), remote_path="/root/config", copy=True)
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
            enforce_eager=eager,
        )
        # ponytail: one LLM per base model; LoRA adapters loaded on-demand
        # per generate() call via lora_request param.
        _LLM_CACHE[model_name] = llm
        return llm


# ── Pure helpers (no Modal / vLLM) ────────────────────────────────────────────


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


def _files_from_diff(patch: str) -> list[str]:
    """Extract the changed file paths (``b/`` side) from a unified diff."""
    if not patch:
        return []
    return [match.group(1) for match in _DIFF_FILE_RE.finditer(patch)]


def render_patch_prompt(
    example: EvalInput,
    template_name: str = DEFAULT_TEMPLATE,
    template_dir: str | Path | None = None,
) -> str:
    """Render the inference prompt for one eval example.

    Args:
        example: Eval input (issue, repo, test names, test patch).
        template_name: One of ``chat`` (default), ``system``, ``user``,
            ``assistant`` — see ``training/prompts/*.j2``.
        template_dir: Prompt template directory (defaults to
            ``training/prompts/`` next to the repo checkout).

    Returns:
        The rendered prompt string.
    """
    from training.prompt_loader import PromptLoader

    loader = PromptLoader(template_dir=template_dir or _PROMPTS_DIR)
    metadata: dict[str, Any] = example.metadata or {}
    language = str(metadata.get("language") or "Python")
    task_description = str(
        metadata.get("task_description") or "Fix the bug described in the issue below."
    )
    style_guide = str(
        metadata.get("style_guide")
        or "Follow the repository's existing code style and test conventions."
    )
    context_files: list[str] = [str(f) for f in metadata.get("context_files", [])]
    test_files = _files_from_diff(example.test_patch)

    if template_name == "chat":
        system_prompt = loader.render(
            "system",
            language=language,
            task_description=task_description,
            style_guide=style_guide,
        )
        user_prompt = loader.render(
            "user",
            issue_title=example.instance_id,
            issue_body=example.issue_body,
            repo_name=example.repo,
            repo_domain=example.repo_domain,
            context_files=context_files,
            test_files=test_files,
        )
        return loader.render_chat(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

    if template_name == "system":
        return loader.render(
            "system",
            language=language,
            task_description=task_description,
            style_guide=style_guide,
        )

    if template_name == "user":
        return loader.render(
            "user",
            issue_title=example.instance_id,
            issue_body=example.issue_body,
            repo_name=example.repo,
            repo_domain=example.repo_domain,
            context_files=context_files,
            test_files=test_files,
        )

    if template_name == "assistant":
        logger.warning(
            "assistant template is a training-target template; rendering with empty sections"
        )
        return loader.render("assistant", analysis="", plan="", code_changes="")

    raise ValueError(
        f"unknown prompt template {template_name!r}; available: {loader.available_templates}"
    )


def resolve_hf_id(model_name: str = "qwen3-14b") -> str:
    """Resolve a ``config/models.yaml`` registry key to a Hugging Face model ID.

    Args:
        model_name: Key from ``models.yaml``, or a full ``owner/model`` ID.

    Returns:
        The Hugging Face model ID; a value containing ``/`` is returned as-is.
    """
    if "/" in model_name:
        return model_name
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not available — cannot resolve %s from models.yaml", model_name)
        return _DEFAULT_HF_ID
    path = _REPO_ROOT / "config" / "models.yaml"
    if not path.is_file():
        logger.warning("model registry not found at %s — using default %s", path, _DEFAULT_HF_ID)
        return _DEFAULT_HF_ID
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models: dict[str, Any] = data.get("models", {}) or {}
    entry: dict[str, Any] = models.get(model_name, {}) or {}
    hf_id: str = entry.get("hf_id", _DEFAULT_HF_ID) or _DEFAULT_HF_ID
    return hf_id


def resolve_adapter_path(variant: str, config: EvalConfig | None = None) -> str | None:
    """Resolve the LoRA adapter directory for a variant.

    Checks ``models/checkpoints/{variant}`` locally first, then downloads the
    W&B artifact ``{entity}/{project}/model-qwen3-14b-{variant}:latest``.
    Any failure returns ``None`` so the baseline model path always works.

    Args:
        variant: Variant key (e.g. ``"baseline_14b"``).
        config: Eval config (defaults to env-driven ``EvalConfig()``).

    Returns:
        Absolute path to the adapter directory, or ``None`` if unavailable.
    """
    from evaluation.config import EvalConfig

    if config is None:
        config = EvalConfig()
    local_dirs = (
        _REPO_ROOT / "models" / "checkpoints" / variant,
        Path("models/checkpoints") / variant,
    )
    for candidate in local_dirs:
        if (candidate / "adapter_config.json").is_file():
            logger.info("found local LoRA adapter for %s at %s", variant, candidate)
            return str(candidate)
    try:
        import wandb

        api = wandb.Api()
        artifact_name = config.lora_artifact_pattern.format(variant=variant)
        artifact_ref = f"{config.wandb_entity}/{config.wandb_project}/{artifact_name}:latest"
        artifact = api.artifact(artifact_ref)
        local_dir = artifact.download()
        logger.info("downloaded LoRA adapter %s to %s", artifact_name, local_dir)
        return str(local_dir)
    except Exception:
        logger.warning(
            "no LoRA adapter found for variant %s — falling back to base model",
            variant,
            exc_info=True,
        )
        return None


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
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> list[str]:
    """Generate candidate fix patches for a batch of eval examples.

    Dispatches to the GPU tier configured by ``EvalConfig.inference_gpu``.
    See ``_generate_patches_batch_body`` for full docs.
    """
    from evaluation.config import EvalConfig

    config = EvalConfig()
    gpu = _GPU_MAP.get(config.inference_gpu, "A100-80GB")
    fn = _get_inference_fn(gpu)
    return fn.remote(
        model_name,
        variant,
        prompt_template,
        examples,
        max_new_tokens,
        temperature,
        top_p,
    )


# ── Core generation body (shared by all GPU tiers) ────────────────────────


def _generate_patches_batch_body(  # noqa: PLR0913, PLR0917
    model_name: str,
    variant: str,
    prompt_template: str,
    examples: list[EvalInput],
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.95,
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

    Returns:
        List of patch strings, same order as ``examples``.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    from evaluation.config import EvalConfig

    config = EvalConfig()
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
    prompts = [render_patch_prompt(example, template_name=prompt_template) for example in examples]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
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
        timeout=600,
        secrets=[
            modal.Secret.from_name("wandb-secret"),
            modal.Secret.from_name("hf-secret"),
        ],
    )(_infer_a10g),
    "A100-80GB": app.function(
        image=vllm_image,
        gpu="A100-80GB",
        volumes={"/models": model_volume},
        timeout=600,
        secrets=[
            modal.Secret.from_name("wandb-secret"),
            modal.Secret.from_name("hf-secret"),
        ],
    )(_infer_a100),
}
