"""Shared pytest fixtures for data_engineering tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_engineering.config import DataPipelineConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_issues_dicts() -> list[dict]:
    with (FIXTURES / "sample_issues.json").open() as f:
        return json.load(f)


@pytest.fixture
def sample_invalid_dicts() -> list[dict]:
    with (FIXTURES / "sample_invalid_issues.json").open() as f:
        return json.load(f)


@pytest.fixture
def config() -> DataPipelineConfig:
    return DataPipelineConfig(
        max_patch_lines=500,
        test_directories=["tests/", "test/"],
    )


@pytest.fixture
def patch_diff() -> str:
    return "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n-x=1\n+x=1\n+y=2\n"
