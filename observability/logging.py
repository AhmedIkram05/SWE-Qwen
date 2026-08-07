"""Structured JSON logging for every SWE-Qwen component (Phase 8, task 8.1).

One JSON object per line: ``{"ts": ISO8601-UTC, "level", "logger", "msg",
...extras}``. ``configure_logging`` is the single root-logger setup point that
replaces every ``logging.basicConfig`` call site; modules keep their plain
``logging.getLogger(__name__)`` loggers and inherit the root handler.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

# Standard LogRecord attributes that carry their own semantics; only custom
# attributes passed via extra={...} should land in the JSON object as extras.
_STANDARD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts/level/logger/msg plus extra attributes."""

    def format(self, record: logging.LogRecord) -> str:
        # Stdlib Formatter sets record.message; downstream consumers (caplog
        # assertions, chained handlers) rely on it even when we don't use it.
        record.message = record.getMessage()
        data: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS:
                continue
            data[key] = value
        try:
            return json.dumps(data, ensure_ascii=False, sort_keys=False)
        except (TypeError, ValueError):
            return json.dumps(
                {key: str(value) for key, value in data.items()},
                ensure_ascii=False,
                sort_keys=False,
            )


def configure_logging(
    level: int = logging.INFO, json: bool = True, stream: TextIO | None = None
) -> None:
    """Configure the root logger with one handler; idempotent-friendly.

    Reuses an existing root handler (e.g. one installed by an earlier
    ``basicConfig``) instead of stacking duplicates; ``json=False`` keeps
    human-readable local-dev output. Logs to stdout by default (Phase 6
    container-logging lesson; code-review N1 matched the contract).
    """
    formatter: logging.Formatter
    if json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
        )
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(level)
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    root.addHandler(handler)
