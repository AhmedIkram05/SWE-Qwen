"""Root conftest: early setup for all test suites."""

from __future__ import annotations

import os

# ponytail: global lock — hf_xet Rust library panics when another trace
# subscriber is already set. HuggingFace recommend HF_XET_DISABLE=1 for
# environments like test suites where tracing is not needed.
# Upgrade when hf_xet supports setting a subscriber without panic.
os.environ.setdefault("HF_XET_DISABLE", "1")

import pytest


@pytest.fixture(autouse=True)
def _disable_modal_batch() -> None:
    """Disable Modal batch path → forces fallback to per-job _run_tests.

    Existing integration tests monkeypatch ``_run_tests`` but not the Modal
    batch path. This fixture ensures ``_run_tests_batch_modal`` always raises,
    so ``_run_tests_batch`` falls back to ``_run_tests_batch_fallback``
    (which calls the monkeypatched ``_run_tests`` per job).
    """
    import evaluation.harness

    evaluation.harness._run_tests_batch_modal = _raise_modal_disabled  # type: ignore[assignment]


def _raise_modal_disabled(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
    """Placeholder that always raises — Modal batch disabled in tests."""
    raise RuntimeError("Modal batch disabled in tests")
