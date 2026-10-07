"""P0-4: f2p_proxy is dry-run only; champion seed matches assets/results.txt.

Unit tests, no W&B API / network — ``wandb.Api`` is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.f2p_proxy import (
    DRY_RUN_ONLY,
    compute_proxy_f2p_scores,
    select_champion,
)
from scripts.seed_champion import build_record

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _results_txt_champion() -> tuple[str, float, float]:
    """Parse the ``[champion]`` row of ``assets/results.txt``."""
    text = (_PROJECT_ROOT / "assets" / "results.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "[champion]" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        # ['', model, variant, total, f2p%, f2p_ci, p2p%, latency, flaky, note, '']
        return (
            cells[2],
            round(float(cells[4].rstrip("%")) / 100, 4),
            round(float(cells[6].rstrip("%")) / 100, 4),
        )
    raise AssertionError("no [champion] row found in assets/results.txt")


@pytest.mark.unit
class TestProxyDryRun:
    """``scripts/f2p_proxy.py`` must be unusable for promotion."""

    def test_proxy_module_marked_dry_run_only(self):
        import scripts.f2p_proxy as mod

        assert DRY_RUN_ONLY is True
        assert "DRY-RUN ONLY" in mod.__doc__
        assert "evaluation.cli compare" in mod.__doc__

    def test_proxy_scores_tagged_dry_run(self, tmp_path, mocker):
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n', encoding="utf-8")
        api = mocker.patch("wandb.Api").return_value
        api.default_entity = "test-entity"
        run = mocker.MagicMock()
        run.state = "finished"
        run.created_at = "2026-01-01"
        run.summary = {"train/loss": 0.5}
        # variant_a scores; variant_b has no runs (warning path).
        api.runs.side_effect = lambda *a, **k: (
            [run] if a and a[1] == {"config.variant": "variant_a"} else []
        )

        result = compute_proxy_f2p_scores(golden, {"variant_a": "a", "variant_b": "b"})

        assert result["variant_a"]["dry_run"] is True
        assert result["variant_b"]["dry_run"] is True

    def test_proxy_select_champion_docstring_requires_eval_cli(self):
        doc = select_champion.__doc__ or ""
        assert "evaluation.cli compare" in doc
        assert "dry-run" in doc.lower()


@pytest.mark.unit
class TestChampionSeed:
    """``scripts/seed_champion.py`` must seed the real champion."""

    def test_champion_seed_matches_results_txt(self):
        variant, f2p, p2p = _results_txt_champion()
        record = build_record("2026-08-06T12:00:00+00:00")
        assert record.variant == variant
        assert record.f2p_rate == f2p
        assert record.p2p_rate == p2p
        assert record.dataset_run_id == "expanded-repos"

    def test_champion_seed_model_ref(self):
        record = build_record("2026-08-06T12:00:00+00:00")
        assert record.model_ref == "qwen3-14b:higher_rank_14b"
