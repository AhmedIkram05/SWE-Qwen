"""Batch patch generation for the SWE-Qwen evaluation harness.

Runs vLLM (optionally with a LoRA adapter) in a Modal container to generate
candidate fix patches for a batch of evaluation instances. Prompts are
rendered with the shared ``training.prompt_loader.PromptLoader`` templates.

The core logic (``render_patch_prompt``, ``extract_patch``,
``resolve_adapter_path``, ``resolve_hf_id``) is plain Python and importable
without Modal; vLLM is only imported inside the Modal function because it is
Linux-only.
"""

from __future__ import annotations

import logging
import re
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
    .pip_install("vllm>=0.26.0", "peft", "wandb", "trl", "bitsandbytes>=0.49.2")
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc at startup; the
    # Modal container has no CUDA toolkit. vLLM 0.26 honors
    # VLLM_USE_FLASHINFER_SAMPLER=0 to fall back to the torch sampler.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(str(_TRAINING_DIR), remote_path="/root/training", copy=True)
    .add_local_dir(str(_EVAL_DIR), remote_path="/root/evaluation", copy=True)
    .add_local_dir(str(_CONFIG_DIR), remote_path="/root/config", copy=True)
)

model_volume = modal.Volume.from_name("eval-model-cache", create_if_missing=True)


# ── Pure helpers (no Modal / vLLM) ────────────────────────────────────────────


def extract_patch(text: str) -> str:
    """Strip markdown code fences from a model completion.

    Args:
        text: Raw completion text (e.g. ``"```diff\\n+line\\n```"``).

    Returns:
        The patch text with any surrounding ``` fence (and language tag)
        removed; the raw completion is returned otherwise.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


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


# ── Modal function ────────────────────────────────────────────────────────────


@app.function(
    image=vllm_image,
    gpu="A10G:1",  # A10G = 24GB; uses config.gpu_type mapping if needed
    volumes={"/models": model_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
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
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from evaluation.config import EvalConfig

    config = EvalConfig()
    adapter_path = resolve_adapter_path(variant, config)
    if adapter_path:
        logger.info("using LoRA adapter for variant %s: %s", variant, adapter_path)
    else:
        logger.info("no LoRA adapter for variant %s — using base model %s", variant, model_name)

    llm = LLM(
        model=resolve_hf_id(model_name),
        quantization="bitsandbytes",  # in-flight NF4 4-bit, matches QLoRA-trained base
        enable_lora=adapter_path is not None,
        gpu_memory_utilization=0.85,
    )
    prompts = [render_patch_prompt(example, template_name=prompt_template) for example in examples]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )
    lora_request = (
        LoRARequest(lora_name=f"{model_name}-{variant}", lora_int_id=1, lora_path=adapter_path)
        if adapter_path is not None
        else None
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [extract_patch(output.outputs[0].text) for output in outputs]
