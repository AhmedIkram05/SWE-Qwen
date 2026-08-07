"""Validate golden.jsonl can be loaded into EvalInput without errors.

Run::

    python scripts/validate_golden.py [--path data/expanded-repos/swebench/golden.jsonl]

Exits 0 only when every record loads successfully.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from evaluation.schema import EvalInput
from observability.logging import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate(path: str) -> int:
    """Validate all records in a golden.jsonl file. Returns exit code."""
    p = Path(path)
    if not p.is_file():
        logger.error("file not found: %s", p)
        return 1

    errors = 0
    total = 0
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.exception("line %d: JSON decode error", line_no)
            errors += 1
            continue

        try:
            inp = EvalInput.model_validate(record)
        except Exception:
            pass
        else:
            # Direct EvalInput format
            _check_input(inp, line_no)
            continue

        try:
            inp = EvalInput.from_swebench_record(record)
        except Exception:
            logger.exception(
                "line %d (issue=%s): EvalInput construction failed",
                line_no,
                record.get("issue_id", "?"),
            )
            errors += 1
            continue

        _check_input(inp, line_no)

    if errors:
        logger.error("FAILED: %d/%d records have errors", errors, total)
    else:
        logger.info("OK: all %d records validated successfully", total)
    return 1 if errors else 0


def _check_input(inp: EvalInput, line_no: int) -> None:
    """Sanity-check a constructed EvalInput."""
    issues: list[str] = []
    if not inp.instance_id:
        issues.append("missing instance_id")
    if not inp.repo:
        issues.append("missing repo")
    if not inp.base_sha:
        issues.append("missing base_sha")
    if not inp.test_patch:
        issues.append("missing test_patch")
    if not inp.fail_to_pass and not inp.pass_to_pass:
        issues.append("both fail_to_pass and pass_to_pass empty")
    if issues:
        logger.warning("line %d (%s): %s", line_no, inp.instance_id, "; ".join(issues))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate golden.jsonl records")
    parser.add_argument(
        "--path",
        default="data/expanded-repos/swebench/golden.jsonl",
        help="Path to golden.jsonl",
    )
    args = parser.parse_args()
    sys.exit(validate(args.path))


if __name__ == "__main__":
    main()
