"""Single merged loading path for the model registry.

Every reader of ``config/models.yaml`` routes through this module so the user
overlay (``config/models.user.yaml``, gitignored) applies everywhere:

* ``config/models.yaml``       — tracked: the base registry
* ``config/models.user.yaml``  — user overlay; deep-merged per model key
  (overrides/extends fields of an existing model, or adds new models).
  Missing file is fine.

The full merged dict per model is the source of truth; ``ModelSpec`` validates
the required fields on top of it without dropping the extras.  The merged raw
result is memoized with ``functools.lru_cache`` keyed by (repo root, mtime of
each file) so file rewrites invalidate the cache without any manual clearing.
"""

from __future__ import annotations

import copy
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"
_BASE_MODELS_FILE = "models.yaml"
_USER_MODELS_FILE = "models.user.yaml"

_DEFAULT_GPU = "a10g-24gb"
_DEFAULT_GPU_MAPPING: dict[str, str] = {"primary": _DEFAULT_GPU, "fallback": _DEFAULT_GPU}

_MISSING_MTIME = -1  # cache-key marker for an absent file


class ModelSpec(BaseModel):
    """One validated model registry entry.

    Required fields are validated; every other key in the YAML entry
    (``active_params``, ``phase4_excluded``, ``prompt_behavior``, ...) rides
    along in ``model_extra`` — the merged dict from ``load_models`` data is
    the source of truth.
    """

    model_config = ConfigDict(extra="allow")

    hf_id: str
    context_window: int
    target_modules: list[str]
    gpu_mapping: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_GPU_MAPPING))
    serving_hf_id: str | None = None
    serving_quantization: str | None = None


# ── Loading ───────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}  # type: ignore[no-any-return]


def _mtime_ns(path: Path) -> int:
    """mtime as nanoseconds; ``_MISSING_MTIME`` when the file is absent."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return _MISSING_MTIME


@lru_cache(maxsize=32)
def _merged_raw(
    repo_root: Path, base_mtime_ns: int, user_mtime_ns: int
) -> dict[str, dict[str, Any]]:
    """Load + deep-merge the two registry files (memoized on file identity +
    mtimes; callers must not mutate the returned value).
    """
    base_data = _load_yaml(repo_root / "config" / _BASE_MODELS_FILE)
    base: dict[str, Any] = copy.deepcopy(base_data.get("models") or {})
    merged: dict[str, dict[str, Any]] = {}
    for key, value in base.items():
        merged[str(key)] = copy.deepcopy(value)

    if user_mtime_ns >= 0:
        user_data = _load_yaml(repo_root / "config" / _USER_MODELS_FILE)
        overlay: dict[str, Any] = user_data.get("models") or {}
        for key, value in overlay.items():
            key_str = str(key)
            if key_str in merged and isinstance(merged[key_str], dict) and isinstance(value, dict):
                _deep_merge(merged[key_str], value)
            else:
                merged[key_str] = copy.deepcopy(value)
    return merged


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """In-place recursive merge of *overlay* into *base*."""
    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = copy.deepcopy(value)


def load_raw_models(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Merged raw model entries (no validation), deep-copied for the caller.

    Used by readers that must stay lenient today (e.g. ``resolve_hf_id``'s
    fallback contract) — a minimal or partial entry is still visible.
    """
    root = repo_root or _REPO_ROOT
    merged = _merged_raw(
        root,
        _mtime_ns(root / "config" / _BASE_MODELS_FILE),
        _mtime_ns(root / "config" / _USER_MODELS_FILE),
    )
    return {key: copy.deepcopy(entry) for key, entry in merged.items()}


def load_models(repo_root: Path | None = None) -> dict[str, ModelSpec]:
    """Merged registry, validated.

    Raises:
        ValidationError: an entry is missing a required field — the error
            title names the failing model key.
        OSError: the tracked registry file does not exist.
    """
    specs: dict[str, ModelSpec] = {}
    for key, entry in load_raw_models(repo_root).items():
        try:
            specs[key] = ModelSpec.model_validate(entry)
        except ValidationError as exc:
            raise ValidationError.from_exception_data(
                f"models registry entry {key!r}",
                cast(
                    "list[Any]",
                    exc.errors(include_url=False, include_context=False),
                ),
            ) from exc
    return specs


def get_model_config(model_name: str, repo_root: Path | None = None) -> dict[str, Any]:
    """Return the merged entry for *model_name* as a dict (validated + extras)."""
    models = load_models(repo_root)
    if model_name not in models:
        raise KeyError(f"Unknown model {model_name!r}. Available: {list(models.keys())}")
    return models[model_name].model_dump()


def default_model_key(repo_root: Path | None = None) -> str:
    """Registry key flagged ``default: true`` in the merged (incl. user) config.

    Raises:
        KeyError: no model is flagged as default.
    """
    for key, entry in load_raw_models(repo_root).items():
        if entry.get("default") is True:
            return key
    raise KeyError("no default model flagged in the registry (expected `default: true`)")
