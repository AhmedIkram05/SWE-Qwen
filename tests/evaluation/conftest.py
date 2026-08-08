"""Test-suite egress gate: never send real Langfuse traces from unit tests.

LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY may be present in the local shell
(e.g. exported to run a live probe). Without this fixture every test that
exercises ``EvaluationHarness.run_batch`` would egress a real trace to the
user's Langfuse cloud project. The trace functions are no-op'd at the public
API the harness calls, so behaviour under test is unchanged.
"""

import pytest
import tests.evaluation  # noqa: F401  (anchor package; documents intent)


@pytest.fixture(autouse=True)
def _block_langfuse_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "observability.langfuse.trace_generation",
        lambda *args, **kwargs: None,
        raising=False,
    )