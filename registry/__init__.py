"""Model registry: single merged loading path for ``config/models.yaml``."""

from registry.loader import ModelSpec, default_model_key, load_models

__all__ = ["ModelSpec", "default_model_key", "load_models"]
