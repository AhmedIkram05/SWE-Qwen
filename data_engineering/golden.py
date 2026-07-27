"""Golden eval subset extraction and verification.

Builds a ``GoldenSet`` from the pipeline splits, with F2P verification stats.
"""

from __future__ import annotations

import logging

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import GoldenSet, Splits
from data_engineering.split import extract_golden

logger = logging.getLogger(__name__)


def build_golden_set(
    splits: Splits,
    min_size: int,
    source_split: str = "test",
) -> GoldenSet:
    """Build a golden eval subset from pipeline splits.

    Args:
        splits: Complete train/val/test splits.
        min_size: Minimum number of golden examples required.
        source_split: Which split to source golden examples from
            (``"test"`` or ``"all"``). Default ``"test"``.

    Returns:
        A ``GoldenSet`` with verified F2P records.
    """
    golden_records = extract_golden(splits, min_size, source_split)

    golden_set = GoldenSet(
        records=golden_records,
        f2p_verified_count=len(golden_records),
        source_split=source_split,
    )

    logger.info(
        "Golden set built: %d records from '%s' split",
        len(golden_records),
        source_split,
    )

    return golden_set


def build_golden_set_from_config(
    splits: Splits,
    config: DataPipelineConfig,
) -> GoldenSet:
    """Convenience wrapper that reads golden params from *config*."""
    return build_golden_set(
        splits,
        min_size=config.min_golden_examples,
        source_split=config.golden_source_split,
    )
