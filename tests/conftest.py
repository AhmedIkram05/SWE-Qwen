"""Root conftest: early setup for all test suites."""

from __future__ import annotations

import os

# ponytail: global lock — hf_xet Rust library panics when another trace
# subscriber is already set. HuggingFace recommend HF_XET_DISABLE=1 for
# environments like test suites where tracing is not needed.
# Upgrade when hf_xet supports setting a subscriber without panic.
os.environ.setdefault("HF_XET_DISABLE", "1")

# Prevent Modal from connecting to cloud at import time.  The ``modal.App``
# constructor sends a registration heartbeat on construction (Modal >= 1.0).
# Every test file that imports ``evaluation.test_runner`` or
# ``evaluation.inference`` triggers this at module level.  The fixture below
# replaces ``modal.App`` with a no-op stand-in that stores name/label but
# never touches the network.
from typing import Any  # noqa: E402

import modal  # noqa: E402


class _FakeModalApp:
    """Stand-in for ``modal.App``: records name, deco-erases for local exec."""

    def __init__(self, name: str | None = None, label: str | None = None) -> None:
        self._name = name
        self._label = label

    def function(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Return a passthrough decorator that calls the wrapped body directly."""
        _ = args, kwargs

        def _deco(fn: Any) -> Any:
            fn._modal_function = True  # mark so tests can detect
            return fn

        return _deco

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """No-op — never enters Modal cloud."""
        _ = args, kwargs


# Patch before any evaluation.* module is imported.
modal.App = _FakeModalApp  # type: ignore[misc]

import pytest


@pytest.fixture(autouse=True)
def _disable_modal_batch() -> None:
    """Disable Modal execution paths → forces fallback routing.

    Existing integration tests monkeypatch ``_run_tests`` but not the Modal
    batch/swebench paths. This fixture ensures ``_run_tests_batch_modal`` and
    ``_run_tests_swebench`` always raise, so ``run_batch`` falls back to
    ``_run_tests_batch`` → ``_run_tests_batch_fallback`` (which calls the
    monkeypatched ``_run_tests`` per job).
    """
    import evaluation.harness

    evaluation.harness._run_tests_batch_modal = _raise_modal_disabled  # type: ignore[assignment]
    evaluation.harness._run_tests_swebench = _raise_modal_disabled  # type: ignore[assignment]


def _raise_modal_disabled(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
    """Placeholder that always raises — Modal batch disabled in tests."""
    raise RuntimeError("Modal batch disabled in tests")
