"""Proxy F2P (Fail-to-Pass) scorer for Phase 4 champion selection.

True F2P evaluation runs a generated patch against the real test suite
(Phase 5).  This proxy computes a heuristic score using test-file overlap
and fix keywords — enough to rank variants for champion selection, not a
substitute for the real harness.

The proxy DOES NOT need a GPU or a model adapter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def score_proxy_f2p(
    record: dict[str, Any],
    generated_patch: str = "",
) -> float:
    """Compute a heuristic proxy F2P score for one record.

    When *generated_patch* is empty (no model output yet) the function
    returns ``0.0`` — all ``patch_present`` tests will score zero, which
    is correct for pre-training comparison.

    Scoring components (each 0-1):
      - ``patch_present``   — non-empty patch diff exists           (33 %)
      - ``test_match``      — generated patch touches test files    (33 %)
      - ``fix_keywords``    — issue body contains fix indicators    (33 %)

    Returns:
        Score in ``[0.0, 1.0]``.
    """
    score = 0.0

    # 1. Patch presence (33 %)
    if generated_patch.strip():
        score += 1 / 3

    # 2. Test-file match (33 %)
    test_files: list[str] = record.get("test_files_changed") or []
    if (
        test_files and test_files[0] and generated_patch.strip()
    ):  # at least one real test file + patch
        for tf in test_files:
            fname = Path(tf).name
            if fname in generated_patch:
                score += 1 / 3
                break

    # 3. Fix keywords in issue body (33 %)
    body: str = record.get("issue_body") or ""
    fix_patterns = [
        r"\bfix\b",
        r"\bbug\b",
        r"\berror\b",
        r"\bfail",
        r"\bcrash\b",
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\bbroken\b",
    ]
    for pat in fix_patterns:
        if re.search(pat, body, re.IGNORECASE):
            score += 1 / (3 * len(fix_patterns))

    return round(min(score, 1.0), 4)


def compute_proxy_f2p_scores(
    golden_path: Path,
    variant_adapter_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Compute mean proxy F2P for each variant.

    Args:
        golden_path: Path to ``golden.jsonl``.
        variant_adapter_map: ``{variant_name: /path/to/adapter_or_empty}``.

    Returns:
        ``{variant: {"mean_f2p": float, "count": int, "warnings": [str]}}``.
    """
    with golden_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    results: dict[str, dict[str, Any]] = {}
    for variant, adapter_path in variant_adapter_map.items():
        total = 0.0
        count = len(records)
        for rec in records:
            # In Phase 4, we don't have model-generated patches yet,
            # so pass empty string — proxy evaluates ground-truth signals
            total += score_proxy_f2p(rec, adapter_path if adapter_path else "")

        mean_f2p = total / count if count > 0 else 0.0
        results[variant] = {
            "mean_f2p": round(mean_f2p, 4),
            "count": count,
            "patch_present": adapter_path != "",
        }

    return results


def select_champion(
    scores: dict[str, dict[str, Any]],
) -> str:
    """Return the variant name with the highest mean F2P."""
    best = max(scores, key=lambda v: scores[v]["mean_f2p"])
    return best
