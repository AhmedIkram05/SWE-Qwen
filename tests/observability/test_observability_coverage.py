"""Phase 8 coverage top-up for the observability package (offline unit tests).

Drives the remaining branches of the four core modules — metrics, cost,
logging, langfuse — that the §5.9 contract tests leave uncovered. wandb and
langfuse SDK clients are fakes in ``sys.modules``; nothing here imports the
real SDK, touches the network, or reads credentials.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import types
from typing import Any

import pytest

from observability import cost as cost_mod
from observability import langfuse as langfuse_mod
from observability.logging import JsonFormatter, configure_logging
from observability.metrics import (
    _is_eval_hierarchical,
    _is_registered,
    assert_registered,
    log_metrics,
)

pytestmark = pytest.mark.unit


# ── metrics ─────────────────────────────────────────────────────────────────


def test_log_metrics_noop_without_run_then_logs(monkeypatch):
    """log_metrics is a no-op without an active run; logs registered keys only."""
    logged: list[dict[str, Any]] = []

    class _FakeWandb:
        run: object | None = None

        @staticmethod
        def log(data: dict[str, Any]) -> None:
            logged.append(data)

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    payload: dict[str, float | int | str | None] = {
        "serve/request_count": 1,
        "serve/error_rate": 0.25,
    }
    log_metrics(payload)
    assert logged == []

    _FakeWandb.run = object()
    log_metrics(payload)
    assert logged == [payload]


def test_assert_registered_rejects_unregistered_key():
    assert_registered({"train/loss": 1.0})
    with pytest.raises(KeyError, match=r"unregistered metric key: 'nope/x'"):
        assert_registered({"nope/x": 1})


def test_is_registered_and_hierarchical_edge_branches():
    assert _is_registered("no_separator") is False  # no "/" -> not even looked up
    assert _is_eval_hierarchical("not_eval/a/b/latency_p50") is False  # not eval/
    assert _is_eval_hierarchical("eval/a") is False  # fewer than 3 segments
    assert _is_eval_hierarchical("eval/a/b/other") is False  # suffix not allowlisted
    assert _is_eval_hierarchical("eval/m/v/t/latency_p50") is True


# ── cost ────────────────────────────────────────────────────────────────────


def test_rate_per_hour_from_config(monkeypatch, tmp_path):
    cfg = tmp_path / "observability.yaml"
    cfg.write_text("rates:\n  a10g-24gb: 1.0\n  default: 2.0\n", encoding="utf-8")
    monkeypatch.setattr(cost_mod, "_CONFIG_PATH", cfg)
    monkeypatch.delenv("OBSERVABILITY_RATE_PER_HOUR", raising=False)

    assert cost_mod.rate_per_hour_from_config("a10g-24gb") == 1.0
    assert cost_mod.rate_per_hour_from_config("unknown-gpu") == 2.0  # falls to default
    assert cost_mod.rate_per_hour_from_config() == 2.0

    monkeypatch.setenv("OBSERVABILITY_RATE_PER_HOUR", "3.5")
    assert cost_mod.rate_per_hour_from_config() == 3.5  # env override wins

    monkeypatch.setenv("OBSERVABILITY_RATE_PER_HOUR", "nope")
    assert cost_mod.rate_per_hour_from_config("a10g-24gb") == 1.0  # unparseable -> config


def test_rate_per_hour_from_config_fallbacks(monkeypatch, tmp_path):
    monkeypatch.delenv("OBSERVABILITY_RATE_PER_HOUR", raising=False)

    monkeypatch.setattr(cost_mod, "_CONFIG_PATH", tmp_path / "missing.yaml")
    assert cost_mod.rate_per_hour_from_config() == cost_mod.DEFAULT_RATE_PER_HOUR

    bad_shape = tmp_path / "bad-shape.yaml"
    bad_shape.write_text("no-rates-key: true\n", encoding="utf-8")
    monkeypatch.setattr(cost_mod, "_CONFIG_PATH", bad_shape)
    assert cost_mod.rate_per_hour_from_config() == cost_mod.DEFAULT_RATE_PER_HOUR

    bad_value = tmp_path / "bad-value.yaml"
    bad_value.write_text("rates:\n  default: not-a-number\n", encoding="utf-8")
    monkeypatch.setattr(cost_mod, "_CONFIG_PATH", bad_value)
    assert cost_mod.rate_per_hour_from_config() == cost_mod.DEFAULT_RATE_PER_HOUR


def test_log_run_cost():
    logged: list[dict[str, Any]] = []

    class _Run:
        def log(self, data: dict[str, Any]) -> None:
            logged.append(data)

    cost_mod.log_run_cost(None, 3600.0, 2.0)
    assert logged == []

    cost_mod.log_run_cost(_Run(), 3600.0, 2.0)
    assert logged == [{"cost/cost_usd": 2.0, "cost/gpu_seconds": 3600.0, "cost/rate_per_hour": 2.0}]


# ── logging ─────────────────────────────────────────────────────────────────


def _reset_root_logger() -> tuple[list[logging.Handler], int]:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    root.handlers.clear()
    return saved


def _restore_root_logger(saved: tuple[list[logging.Handler], int]) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.handlers.extend(saved[0])
    root.setLevel(saved[1])


def test_configure_logging_plain_and_stream():
    saved = _reset_root_logger()
    try:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, json=False, stream=stream)
        logging.getLogger("test.plain").warning("plain %s", "message")
        rendered = stream.getvalue()
        assert "[WARNING]" in rendered
        assert "plain message" in rendered
    finally:
        _restore_root_logger(saved)


def test_configure_logging_default_stdout(monkeypatch):
    saved = _reset_root_logger()
    try:
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        configure_logging(level=logging.INFO)
        logging.getLogger("test.stdout").info("via stdout")
        assert "via stdout" in out.getvalue()
    finally:
        _restore_root_logger(saved)


def test_json_formatter_non_serializable_extra():
    record = logging.LogRecord(
        name="test.weird",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    vars(record)["weird"] = object()  # not JSON-serializable -> str() fallback
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "WARNING"
    assert payload["msg"] == "boom"
    assert payload["weird"].startswith("<object object at")


# ── langfuse ────────────────────────────────────────────────────────────────


def _clear_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(langfuse_mod, "_client", None)


def test_get_client_one_missing_key(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    assert langfuse_mod._get_client() is None  # secret key missing -> no SDK import


def test_get_client_init_failure_is_swallowed(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    class _Boom:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            raise RuntimeError("sdk init failed")

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=_Boom))
    assert langfuse_mod._get_client() is None
    assert langfuse_mod._client is None  # not cached, next call retries


def test_trace_generation_no_observation(monkeypatch):
    _clear_langfuse_env(monkeypatch)

    class _FakeClient:
        def start_observation(self, **kwargs: object) -> None:
            _ = kwargs

    monkeypatch.setattr(langfuse_mod, "_client", _FakeClient())
    langfuse_mod.trace_generation(
        name="n", model="m", prompt="p", completion="c", metadata={"run_id": "r"}
    )


def test_trace_generation_sdk_failure_swallowed(monkeypatch):
    _clear_langfuse_env(monkeypatch)

    class _FakeClient:
        def start_observation(self, **kwargs: object) -> None:
            _ = kwargs
            raise RuntimeError("network down")

    monkeypatch.setattr(langfuse_mod, "_client", _FakeClient())
    langfuse_mod.trace_generation(
        name="gen",
        model="m",
        prompt="p",
        completion="c",
        metadata={"run_id": "r"},
        scores={"f2p": 1.0},
    )


def test_trace_request_no_observation_and_failure(monkeypatch):
    _clear_langfuse_env(monkeypatch)

    class _NoneObs:
        def start_observation(self, **kwargs: object) -> None:
            _ = kwargs

    class _BoomObs:
        def start_observation(self, **kwargs: object) -> None:
            _ = kwargs
            raise RuntimeError("network down")

    monkeypatch.setattr(langfuse_mod, "_client", _NoneObs())
    langfuse_mod.trace_request(
        model="m", template_name=None, ttfbs_ms=None, latency_ms=1.0, output_tokens=5
    )

    monkeypatch.setattr(langfuse_mod, "_client", _BoomObs())
    langfuse_mod.trace_request(
        model="m", template_name="tpl", ttfbs_ms=1.0, latency_ms=2.0, output_tokens=5
    )
