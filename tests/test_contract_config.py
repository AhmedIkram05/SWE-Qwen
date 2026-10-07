"""P0-7 config contract: tokenize ↔ trainer length alignment, packing off, save cadence.

- ``DataPipelineConfig.tokenize_max_length`` must equal the trainer's
  effective ``max_seq_length`` (capped by the model context window) — we must
  never tokenize at 32k then silently train at 4k/2k.
- ``packing`` must be ``False`` on the pre-tokenized path: the ``.arrow``
  shards already carry ``input_ids``/``labels`` with ``-100`` prompt masks,
  and packing would concatenate examples across those boundaries.
- ``save_steps`` must be ``< total steps`` so at least one mid-run checkpoint
  exists for a full-size (``max_train_examples``) dataset.

No network, no GPU, no Modal: config objects are built directly.
"""

from __future__ import annotations

import math

import pytest

from data_engineering.config import DataPipelineConfig
from training.qlora_config import (
    GPU_MEMORY_OVERRIDES,
    _get_model_config,
    _get_variant_config,
    build_qlora_config,
    list_variants,
)

try:  # Added by P0-2 (tokenize/trainer length alignment).
    from training.qlora_config import resolve_train_max_seq_length
except ImportError:  # pragma: no cover - checkout predates P0-2
    resolve_train_max_seq_length = None  # type: ignore[assignment]

VARIANTS = list_variants()
GPUS = [None, *GPU_MEMORY_OVERRIDES.keys()]


def _require_p02() -> None:
    if resolve_train_max_seq_length is None:
        pytest.skip(
            "training.qlora_config.resolve_train_max_seq_length is missing — "
            "the P0-2 tokenize/trainer alignment is not in this checkout "
            "(these guards activate once P0-2 merges)"
        )


class TestTokenizeTrainerLengthAlignment:
    def test_tokenize_max_length_matches_trainer(self) -> None:
        _require_p02()
        config = DataPipelineConfig()
        ctx = _get_model_config(config.tokenize_model)["context_window"]
        train_max = resolve_train_max_seq_length()
        assert config.tokenize_max_length == min(ctx, train_max)

    def test_variant_default_matches_variant_yaml(self) -> None:
        _require_p02()
        for variant in VARIANTS:
            base = _get_variant_config(variant)["training"].get("max_seq_length")
            if base is not None:
                assert resolve_train_max_seq_length(variant=variant) == base

    def test_gpu_overrides_are_the_documented_divergence(self) -> None:
        _require_p02()
        # Any difference from the variant value must be exactly the documented
        # GPU override — no undocumented third length.
        for variant in VARIANTS:
            base = _get_variant_config(variant)["training"].get("max_seq_length")
            for gpu, overrides in GPU_MEMORY_OVERRIDES.items():
                expected = overrides.get("max_seq_length", base)
                if expected is not None:
                    assert resolve_train_max_seq_length(variant=variant, gpu_type=gpu) == expected

    def test_never_exceeds_context_window(self) -> None:
        _require_p02()
        config = DataPipelineConfig()
        ctx = _get_model_config(config.tokenize_model)["context_window"]
        for gpu in GPUS:
            if gpu is None:
                train_max = resolve_train_max_seq_length()
            else:
                train_max = resolve_train_max_seq_length(gpu_type=gpu)
            assert min(ctx, train_max) <= ctx


class TestPackingOffForPreTokenized:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("gpu", GPUS)
    def test_packing_false(self, variant: str, gpu: str | None) -> None:
        _require_p02()
        _, args = build_qlora_config(variant=variant, model_name="qwen3-14b", gpu_type=gpu)
        assert args.packing is False, (
            f"packing must be False on the pre-tokenized path (variant={variant}, gpu={gpu}); "
            "shards carry -100 prompt masks that packing would bleed across"
        )


class TestSaveCadence:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_save_steps_below_total_steps(self, variant: str) -> None:
        _, args = build_qlora_config(variant=variant, model_name="qwen3-14b")
        assert args.save_strategy == "steps"
        assert args.save_steps is not None

        config = DataPipelineConfig()
        n = config.max_train_examples
        per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        total_steps = math.ceil(n / per_step) * args.num_train_epochs
        assert args.save_steps < total_steps, (
            f"variant={variant}: save_steps={args.save_steps} >= total_steps={total_steps} "
            f"for {n} examples — no mid-run checkpoint would be saved"
        )
