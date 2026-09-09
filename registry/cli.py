"""Typer CLI for the model registry.

Usage::

    model list
    model add meta-llama/Llama-3.1-8B-Instruct --name llama-31-8b --context 8192
    model add Qwen/Qwen3-14B --name qwen3-14b --context 0 --force
    model add nvidia/Llama-3.1-Nemotron-... --name nemotron --allow-no-template

``add`` validates over plain HTTP (HF API reachability + repo file probes) and
writes ONLY to ``config/models.user.yaml`` (gitignored); the tracked
``config/models.yaml`` is never touched.  No GPU, no model download.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import typer
import yaml

from registry.loader import ModelSpec, _load_yaml, load_raw_models

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="model",
    help="Model registry CLI: list the merged registry, add user-overlay models.",
    no_args_is_help=True,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_client = httpx.Client(
    follow_redirects=True, timeout=15.0, headers={"User-Agent": "swe-qwen-registry-cli"}
)

# Standard attention(q/k/v/o) + MLP(gate/up/down) set — what peft's
# find_all_linear_names returns for these decoder-only families.
_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_ATTN_MLP_ARCHES: tuple[str, ...] = (
    "llama",
    "mistral",
    "mixtral",
    "qwen",
    "gemma",
    "phi",
    "starcoder2",
    "olmo",
    "exaone",
    "granite",
    "nemotron",
)

_THINKING_MARKERS = ("enable_thinking", "chat_template_continue_generation")

_STAGES = ("tokenize", "train", "eval", "serve")


@app.command()
def add(
    hf_id: str = typer.Argument(
        ..., help="Hugging Face repo id, e.g. meta-llama/Llama-3.1-8B-Instruct"
    ),
    name: str = typer.Option(..., "--name", help="registry key the pipelines resolve"),
    context: int = typer.Option(
        0, "--context", help="context window; 0 = probe the model's max_position_embeddings"
    ),
    target_modules: str = typer.Option(
        "auto",
        "--target-modules",
        help="'auto' or a comma-separated list, e.g. q_proj,v_proj",
    ),
    allow_no_template: bool = typer.Option(
        False,
        "--allow-no-template",
        help="add the model even if its tokenizer_config.json has no chat_template",
    ),
    force: bool = typer.Option(False, "--force", help="overlay an existing registry key"),
) -> None:
    """Validate *hf_id* over HF, then add it to the USER overlay (``config/models.user.yaml``).

    The tracked ``config/models.yaml`` is never touched. Network is used only
    for the HF API reachability check and the config/tokenizer file probes.
    """
    repo_root = _REPO_ROOT
    merged = load_raw_models(repo_root)
    if name in merged and not force:
        typer.echo(
            f"error: registry key {name!r} already exists; pass --force to overlay it", err=True
        )
        raise typer.Exit(code=1)

    _check_reachable(hf_id)
    config_json = _fetch_json(_resolve_url(hf_id, "config.json"), f"config.json for {hf_id!r}")
    tokenizer_json = _fetch_json(
        _resolve_url(hf_id, "tokenizer_config.json"), f"tokenizer_config.json for {hf_id!r}"
    )

    context_window = _resolve_context_window(context, config_json, hf_id)
    _check_chat_template(tokenizer_json, hf_id, allow_no_template)
    modules = _parse_target_modules(target_modules, config_json.get("model_type"))

    entry: dict[str, Any] = {
        "hf_id": hf_id,
        "context_window": context_window,
        "target_modules": modules,
        "prompt_behavior": {"no_think": not _has_thinking_support(tokenizer_json)},
    }
    path = _write_user_entry(repo_root, name, entry)
    spec = ModelSpec.model_validate(entry)
    typer.echo(f"added {name!r} ({hf_id}) to {path.relative_to(repo_root)}")
    typer.echo(yaml.safe_dump({"models": {name: spec.model_dump()}}, sort_keys=False), nl=False)
    typer.echo(f"stages picking this up: {', '.join(_STAGES)}")


@app.command("list")
def list_models() -> None:
    """Print the merged registry (tracked + user overlay) as a table."""
    repo_root = _REPO_ROOT
    base: dict[str, Any] = _load_yaml(repo_root / "config" / "models.yaml").get("models") or {}
    rows = [
        (
            key,
            str(entry.get("hf_id", "")),
            str(entry.get("context_window", "")),
            "tracked" if key in base else "user",
        )
        for key, entry in load_raw_models(repo_root).items()
    ]
    if not rows:
        typer.echo("registry is empty")
        return
    headers = ("key", "hf_id", "context_window", "source")
    widths = [max(len(str(row[index])) for row in rows + [headers]) for index in range(4)]
    typer.echo("  ".join(headers[index].ljust(widths[index]) for index in range(4)))
    for row in rows:
        typer.echo("  ".join(str(row[index]).ljust(widths[index]) for index in range(4)))


# ── Validation ────────────────────────────────────────────────────────────────


def _api_models_url(hf_id: str) -> str:
    return f"https://huggingface.co/api/models/{hf_id}"


def _resolve_url(hf_id: str, filename: str) -> str:
    return f"https://huggingface.co/{hf_id}/resolve/main/{filename}"


def _check_reachable(hf_id: str) -> None:
    """HEAD the HF API; any network problem or non-2xx → error + Exit(1)."""
    url = _api_models_url(hf_id)
    try:
        response = _client.head(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        typer.echo(f"error: huggingface.co unreachable for {hf_id!r}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _fetch_json(url: str, what: str) -> dict[str, Any]:
    try:
        response = _client.get(url)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        typer.echo(f"error: failed to fetch {what}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(payload, dict):
        typer.echo(f"error: {what} is not a JSON object", err=True)
        raise typer.Exit(code=1)
    return payload


def _resolve_context_window(context: int, config_json: dict[str, Any], hf_id: str) -> int:
    """``--context 0`` resolves to the model's ``max_position_embeddings``."""
    if context > 0:
        return context
    max_pos = config_json.get("max_position_embeddings")
    if not isinstance(max_pos, int) or max_pos <= 0:
        typer.echo(
            f"error: {hf_id!r} config.json has no usable max_position_embeddings "
            "and --context is 0",
            err=True,
        )
        raise typer.Exit(code=1)
    return max_pos


def _check_chat_template(
    tokenizer_json: dict[str, Any], hf_id: str, allow_no_template: bool
) -> None:
    """SWE-bench prompt building needs a chat_template; warn, then gate on the flag."""
    if tokenizer_json.get("chat_template"):
        return
    typer.echo(
        f"warning: {hf_id!r} has no chat_template; SWE-bench prompt building needs one",
        err=True,
    )
    if allow_no_template:
        return
    typer.echo("error: pass --allow-no-template to add the model anyway", err=True)
    raise typer.Exit(code=1)


def _has_thinking_support(tokenizer_json: dict[str, Any]) -> bool:
    """Thinking-mode markers (Qwen3-style) in keys or the chat_template text."""
    if any(marker in tokenizer_json for marker in _THINKING_MARKERS):
        return True
    template = tokenizer_json.get("chat_template")
    text = template if isinstance(template, str) else ""
    return any(marker in text for marker in _THINKING_MARKERS)


def _parse_target_modules(value: str, model_type: Any) -> list[str]:
    if value.strip().lower() == "auto":
        return _auto_target_modules(model_type)
    modules = [part.strip() for part in value.split(",") if part.strip()]
    if not modules:
        raise typer.BadParameter(f"expected 'auto' or a comma-separated list, got {value!r}")
    return modules


def _auto_target_modules(model_type: Any) -> list[str]:
    """Offline stand-in for peft's find_all_linear_names (no model loaded).

    Attention+MLP families get the standard 7-module set; otherwise peft's own
    per-arch mapping decides; unknown archs fall back to the same default with
    a warning (target_modules is one YAML edit away).
    """
    arch = str(model_type or "").lower()
    if arch.startswith(_ATTN_MLP_ARCHES):
        return list(_DEFAULT_TARGET_MODULES)
    try:
        from peft.utils.other import (
            TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as _PEFT_MAPPING,
        )
    except ImportError:
        logger.warning(
            "peft not importable; --target-modules auto falls back to the default 7-module list"
        )
        return list(_DEFAULT_TARGET_MODULES)
    if arch in _PEFT_MAPPING:
        return [str(module) for module in _PEFT_MAPPING[arch]]
    logger.warning(
        "architecture %r not in peft's mapping; --target-modules auto falls back "
        "to the default 7-module list",
        arch,
    )
    return list(_DEFAULT_TARGET_MODULES)


def _write_user_entry(repo_root: Path, name: str, entry: dict[str, Any]) -> Path:
    """Add/replace *entry* under ``models.<name>`` in the user overlay file."""
    path = repo_root / "config" / "models.user.yaml"
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            typer.echo(f"error: {path} is not a YAML mapping; refusing to overwrite it", err=True)
            raise typer.Exit(code=1)
        data = loaded
    data.setdefault("models", {})[name] = entry
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    app()
