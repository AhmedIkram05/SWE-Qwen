"""Test-suite egress gate: never send real Langfuse traces from unit tests.

LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY may be present in the local shell
(e.g. exported to run a live probe). Without this fixture every test that
drains the serving trace queue (``inference/telemetry._drain_trace_queue``)
would egress a real ``serve/request`` trace to the user's Langfuse cloud
project. The trace function is no-op'd at the public API the drain calls.
"""

import pytest


@pytest.fixture(autouse=True)
def _block_langfuse_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "observability.langfuse.trace_request",
        lambda *args, **kwargs: None,
        raising=False,
    )