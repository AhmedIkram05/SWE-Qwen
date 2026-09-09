"""Modal App: serverless vLLM serving with scale-to-zero (Phase 6, Wave 3b).

Image construction mirrors ``evaluation/inference.py``'s ``vllm_image`` and
``training/modal_train.py``: debian_slim py3.11, apt git/wget/curl, pip_install
order, ``.env()`` calls, a cache-buster via ``run_commands``, and
``add_local_dir`` last (Modal's default workdir is /root, so the packages
import).  Do NOT use the vllm/vllm-openai registry image — unusable on Modal
(no ``python`` in PATH; pip fails with exit 127).

Scale-to-zero: no ``keep_warm`` / ``min_containers`` — idle containers
terminate per Modal's default scaledown window after
``ServeConfig.idle_timeout_seconds``; the ``serve-model-cache`` volume (FP8
base + vLLM cache) pays for cold starts.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from fastapi import FastAPI

from inference.config import ServeConfig
from inference.serve import VLLMEngine, create_app
from observability.logging import configure_logging
from registry.loader import default_model_key, load_raw_models

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Serving GPU resolution ─────────────────────────────────────────────────────
# Registry gpu_mapping tier labels → Modal GPU spec strings. Mirrors
# training/qlora_config.GPU_MAP — duplicated, not imported: that module pulls
# peft/trl, which the serving image never installs.
_GPU_TIER_TO_SPEC: dict[str, str] = {
    "a100-40gb": "A100:1",
    "a10g-24gb": "A10G:1",
    "a100-80gb": "A100-80GB",
    "h100-80gb": "H100:1",
}

_DEFAULT_GPU_SPEC = "A10G:1"


def gpu_spec_for_tier(tier: str | None) -> str:
    """Map a registry ``gpu_mapping`` tier label to its Modal GPU spec string.

    Unknown labels pass through verbatim (assumed to already be Modal specs);
    a missing tier is the A10G default.
    """
    if not tier:
        return _DEFAULT_GPU_SPEC
    return _GPU_TIER_TO_SPEC.get(tier.lower(), tier)


def resolve_serving_gpu(repo_root: Path | None = None) -> str:
    """Resolve the Modal GPU spec for serving.

    ``SERVING_GPU`` env wins verbatim; otherwise the serving model's registry
    ``gpu_mapping`` primary tier (the model is ``SERVING_BASE_MODEL`` env or the
    registry's default) is mapped to a spec; otherwise A10G.  Import-time safe:
    any registry failure (missing models.yaml, unknown model, no default)
    falls back to A10G so the Modal decorators below never fail to import.
    """
    env_override = os.environ.get("SERVING_GPU", "").strip()
    if env_override:
        return env_override
    tier: str | None = None
    try:
        raw = load_raw_models(repo_root)
        base = os.environ.get("SERVING_BASE_MODEL", "").strip() or default_model_key(repo_root)
        mapping = (raw.get(base) or {}).get("gpu_mapping") or {}
        tier = mapping.get("primary") or mapping.get("fallback")
    except Exception:  # import must never raise; A10G is the invariant fallback
        tier = None
    return gpu_spec_for_tier(tier)


# Modal decorators evaluate their gpu= at import — resolve once, up front.
_SERVING_GPU = resolve_serving_gpu()


def _build_smoke() -> None:
    """Boot-time fail-fast: engine init errors (VRAM, max_model_len) surface
    during image build, not at first request.

    Runs as an ``Image.run_function`` build step — modal 1.5.3 has no
    ``@modal.build()`` hook (verified: ``hasattr(modal, "build")`` is False) —
    with the serving class's GPU/volume/secrets so the smoke is representative.
    The FP8 base lives on ``serve_volume`` (HF_HOME=/models), so rebuilds
    reuse it instead of re-downloading 28 GB.
    """
    engine = VLLMEngine(ServeConfig())
    engine.generate(
        "ping",
        lora=None,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        stop=None,
        repetition_penalty=1.0,
    )


app = modal.App("swe-qwen-serving")

serve_volume = modal.Volume.from_name("serve-model-cache", create_if_missing=True)

# Kept private on purpose (a from-import would trip PLC2701); benchmark.py
# reuses these via ``modal_serve.image`` / ``modal_serve.serve_volume`` /
# ``modal_serve._secrets``.  hf-secret: Qwen3-14B is gated on the Hub.
# serve-token: MODAL_SERVE_TOKEN bearer auth for the chat endpoint (Phase 9).
_secrets = [
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("hf-secret"),
    modal.Secret.from_name("serve-token"),
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install(
        "vllm>=0.26.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "sse-starlette>=2.2.0",
        "wandb>=0.28.0",
        # ServeConfig (inference/config.py) imports pydantic-settings at module
        # level; vllm 0.26.0 does NOT pull it transitively (verified against
        # PyPI metadata), so without it the container dies importing
        # inference.serve.  pyyaml/pydantic/jinja2 come via vllm /
        # fastapi[standard].
        "pydantic-settings>=2.7.0",
    )
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc at startup; the
    # container has no CUDA toolkit. vLLM 0.26 honors this env to fall back to
    # the torch sampler.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    # Volume-cached quantized base (FP8): restarts reuse /models instead of
    # re-downloading the 28 GB model from Hugging Face.
    .env({"HF_HOME": "/models"})
    # Persist vLLM's CUDA graph cache across cold starts; without this every
    # container re-captures graphs (~35s observed in Phase 5).
    .env({"VLLM_CACHE_ROOT": "/models/vllm-cache"})
    # Bump ONLY when deps change (forces a rebuild; the rebuild reruns the
    # smoke step below).
    .run_commands("echo 'serve-cache-bust-v1'")
    .add_local_dir(str(_REPO_ROOT / "inference"), remote_path="/root/inference", copy=True)
    .add_local_dir(str(_REPO_ROOT / "training"), remote_path="/root/training", copy=True)
    # prompts/*.j2 for PromptLoader (swe_bench requests)
    .add_local_dir(str(_REPO_ROOT / "config"), remote_path="/root/config", copy=True)
    # merged model registry (registry/loader.py — ServeConfig + resolve_hf_id)
    .add_local_dir(str(_REPO_ROOT / "registry"), remote_path="/root/registry", copy=True)
    .add_local_dir(str(_REPO_ROOT / "observability"), remote_path="/root/observability", copy=True)
)
# data/golden.jsonl is gitignored and absent in CI; the container fetches
# golden from GCS (_ensure_golden), so bake the fallback only when present —
# a missing file must not break image builds (used to FileNotFoundError).
if (_REPO_ROOT / "data" / "golden.jsonl").is_file():
    image = image.add_local_file(
        str(_REPO_ROOT / "data" / "golden.jsonl"),
        remote_path="/root/data/golden.jsonl",
        copy=True,
    )
# Build-time fail-fast must run last: it needs /root/inference baked in.
image = image.run_function(
    _build_smoke,
    gpu=_SERVING_GPU,
    volumes={"/models": serve_volume},
    secrets=_secrets,
    timeout=1800,
)


@app.cls(
    gpu=_SERVING_GPU,
    image=image,
    volumes={"/models": serve_volume},
    secrets=_secrets,
    timeout=1800,
)
@modal.concurrent(max_inputs=16)  # 1.5.3 replacement for asgi_app allow_concurrent_inputs
class ModelServer:
    """Serverless serving class: one vLLM engine per container.

    Fail-fast at container warmup: ``enter`` constructs the engine, so init
    errors (VRAM, max_model_len) kill the container before any request is
    served.  Modal 1.5.3 has no ``@modal.build()`` hook — the image-level
    ``_build_smoke`` step covers the build-time half.
    """

    _engine: VLLMEngine

    @modal.enter()
    def enter(self) -> None:
        configure_logging()
        self._engine = VLLMEngine(ServeConfig())

    @modal.asgi_app()
    def web(self) -> FastAPI:
        return create_app(self._engine)
