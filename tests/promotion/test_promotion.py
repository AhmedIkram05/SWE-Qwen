"""Unit tests for the promotion pipeline (Phase 9).

Covers the rule matrix (task 9.2), the champion registry (task 9.3, spec
§4.3), the one-shot seeding script (task 9.3, spec §4.7) and the
deploy/probe/alias/rollback path (task 9.4, spec §4.5).  The exact
boundary behavior: ``==`` is not a pass anywhere — margin and P2P ceiling
survive exact equality (strict ``<``) while the significance gate fails at
``ci_lower == 0.0`` (``<=``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.config import EvalConfig
from evaluation.metrics import aggregate_metrics
from evaluation.schema import EvalResult, EvalRun, F2PMetrics, PatchApplicationResult
from evaluation.stats import paired_bootstrap_ci
from inference.config import ServeConfig
from promotion import audit
from promotion import deploy as deploy_mod
from promotion import run as run_mod
from promotion.deploy import (
    ConfigGapError,
    ProbeError,
    assert_variant_known,
    deploy,
    health_check,
    rollback,
    sync_alias_or_abort,
)
from promotion.gate import PairEval, evaluate_pair, revalidate_champion
from promotion.registry import (
    ChampionRecord,
    from_dict,
    read_champion,
    sync_alias,
    to_dict,
    write_champion,
)
from promotion.rules import (
    OUTCOME_PROMOTE,
    OUTCOME_REJECT,
    PROMOTE_MAX_P2P_REGRESSION,
    PROMOTE_MIN_F2P_GAIN,
    REASON_FATAL_FLAW,
    REASON_MICRO_GAIN,
    REASON_REGRESSION,
    decide,
)

pytestmark = pytest.mark.unit

# Champion from seed_champion.py (2026-08-06): F2P 0.169, P2P 0.912.
CHAMPION_F2P = 0.169
CHAMPION_P2P = 0.912

# Each paired run evaluates exactly one variant (spec §4.2): the champion is
# the incumbent baseline, the candidate is the challenger.
CANDIDATE_VARIANT = "higher_lr_14b"


class TestDecide:
    """Rule matrix for ``promotion.rules.decide`` (spec §4.1, L189)."""

    def test_promote_happy_path(self):
        outcome, reasons = decide(0.169, 0.25, 0.912, 0.93, ci_lower=0.05)
        assert outcome == "promote"
        assert reasons == []

    def test_fatal_flaw_f2p_below_floor(self):
        outcome, reasons = decide(0.169, 0.10, 0.912, 0.93, ci_lower=0.05)
        assert outcome == "reject"
        assert reasons == ["fatal-flaw"]

    def test_fatal_flaw_p2p_below_floor(self):
        outcome, reasons = decide(0.169, 0.25, 0.912, 0.80, ci_lower=0.05)
        assert outcome == "reject"
        assert reasons == ["fatal-flaw"]

    def test_fatal_flaw_below_margin(self):
        # Gain 0.021 < PROMOTE_MIN_F2P_GAIN (0.05); floors pass.
        outcome, reasons = decide(0.169, 0.19, 0.912, 0.93, ci_lower=0.05)
        assert outcome == "reject"
        assert reasons == ["fatal-flaw"]

    def test_regression_p2p_drop_beyond_ceiling(self):
        # champion_p2p=0.912 would make the ceiling (0.892) unreachable while
        # the 0.90 floor passes — use 0.95 so a 0.022 drop (> 0.02) exists with
        # floors and margin both passing (see report: spec vector is
        # self-contradictory; this preserves its intent).
        outcome, reasons = decide(0.169, 0.25, 0.95, 0.928, ci_lower=0.05)
        assert outcome == "reject"
        assert reasons == ["regression"]

    def test_micro_gain_ci_lower_zero(self):
        # ci_lower == 0.0 fails the significance gate (<=); margin passes.
        outcome, reasons = decide(0.169, 0.25, 0.912, 0.93, ci_lower=0.0)
        assert outcome == "reject"
        assert reasons == ["micro-gain"]

    def test_micro_gain_negative_ci(self):
        outcome, reasons = decide(0.169, 0.25, 0.912, 0.93, ci_lower=-0.01)
        assert outcome == "reject"
        assert reasons == ["micro-gain"]

    def test_exact_constant_boundaries_all_reject(self):
        # Boundary values are COMPUTED, not decimal literals: 0.219 is not
        # exactly 0.169 + 0.05 in IEEE-754, so a literal would trip the margin
        # check.  Computing candidate == champion + constant makes the strict
        # comparisons evaluate the identical float on both sides.  The P2P side
        # uses champion_p2p=0.95, not CHAMPION_P2P: a 0.02 drop from 0.912 lands
        # at 0.892, below the 0.90 floor, making the ceiling-at-== state
        # unreachable (the floor check fires first — see spec-vector note on
        # test_regression_p2p_drop_beyond_ceiling).
        # Result: margin check (strict <) and P2P ceiling check (strict <) both
        # PASS at exact equality, and ci_lower == 0.0 fails significance (<=).
        # Documents that == is not a pass anywhere: still no promotion.
        candidate_f2p = CHAMPION_F2P + PROMOTE_MIN_F2P_GAIN  # gain exactly 0.05
        champion_p2p = 0.95
        candidate_p2p = champion_p2p - PROMOTE_MAX_P2P_REGRESSION  # drop exactly 0.02
        outcome, reasons = decide(
            CHAMPION_F2P, candidate_f2p, champion_p2p, candidate_p2p, ci_lower=0.0
        )
        assert outcome == "reject"
        assert reasons == ["micro-gain"]

    def test_extreme_values_never_crash(self):
        for args in [
            (0.0, 0.0, 0.0, 0.0, -1.0),  # all zeros, negative CI
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0, 1.0),  # all perfect
            (0.5, -0.1, 0.9, 1.2, 0.0),  # out-of-range rates
        ]:
            outcome, reasons = decide(*args)
            assert isinstance(outcome, str)
            assert isinstance(reasons, list)
            assert reasons is not None

    def test_outcome_constants_are_literals(self):
        assert OUTCOME_PROMOTE == "promote"
        assert OUTCOME_REJECT == "reject"
        assert REASON_FATAL_FLAW == "fatal-flaw"
        assert REASON_REGRESSION == "regression"
        assert REASON_MICRO_GAIN == "micro-gain"


# ── in-memory EvalRun/EvalResult fixtures (no LLM, no Ollama, no network) ──
# Mirrors the fixture pattern in tests/evaluation/test_eval_comparison_extra.py.


def _result(
    instance_id: str,
    f2p: float,
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    p2p: float = 1.0,
) -> EvalResult:
    """Minimal valid ``EvalResult`` for a single instance."""
    return EvalResult(
        instance_id=instance_id,
        repo="owner/repo",
        model_name=model,
        variant=variant,
        prompt_template="chat",
        generated_patch="",
        patch_application=PatchApplicationResult(success=True, method_used="git_apply"),
        tests_before=[],
        tests_after=[],
        f2p=f2p,
        p2p=p2p,
        latency_seconds=1.0,
        timestamp=datetime.now(UTC),
    )


def _make_run(run_id: str, config: EvalConfig, results: list[EvalResult]) -> EvalRun:
    """In-memory completed ``EvalRun`` with derived models_evaluated/aggregate."""
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=sorted({f"{r.model_name}:{r.variant}" for r in results}),
        results=results,
        aggregate=[aggregate_metrics(results)] if results else [],
        status="completed",
    )


def _pair_config() -> EvalConfig:
    """Default paired-eval window (dev tier: expanded-repos, seed 42)."""
    return EvalConfig(dataset_run_id="expanded-repos", tier_seed=42)


def _metrics(model: str, variant: str, f2p_rate: float, p2p_rate: float) -> F2PMetrics:
    """Minimal ``F2PMetrics`` with the fields revalidate_champion reads."""
    return F2PMetrics(
        model_name=model,
        variant=variant,
        prompt_template="chat",
        total_examples=10,
        successful_patches=10,
        f2p_rate=f2p_rate,
        f2p_count=10,
        p2p_rate=p2p_rate,
        p2p_count=10,
        avg_latency=1.0,
        flaky_test_rate=0.0,
        per_repo_breakdown={},
    )


class TestEvaluatePair:
    """``promotion.gate.evaluate_pair`` + revalidate passthrough (spec §4.2).

    Deterministic in-memory 0/1 F2P vectors: the paired bootstrap uses a fixed
    seed (42), so every CI/McNemar assertion is exact, not probabilistic.
    """

    def test_evaluate_pair_happy_path(self):
        # Candidate solves a strict superset: champion i1..i3, candidate
        # i1..i7 on 10 shared instances.  F2P 0.7 vs 0.3, b10=4 (i4..i7),
        # b01=0 — a clean, deterministic candidate win.
        config = _pair_config()
        champion_results = [_result(f"i{index}", f2p=1.0) for index in range(1, 4)] + [
            _result(f"i{index}", f2p=0.0) for index in range(4, 11)
        ]
        candidate_results = [
            _result(f"i{index}", f2p=1.0, variant=CANDIDATE_VARIANT) for index in range(1, 8)
        ] + [_result(f"i{index}", f2p=0.0, variant=CANDIDATE_VARIANT) for index in range(8, 11)]
        pair = evaluate_pair(
            _make_run("champion-run", config, champion_results),
            _make_run("candidate-run", config, candidate_results),
            config,
        )

        assert isinstance(pair, PairEval)
        assert pair.f2p_gain == pytest.approx(0.4)
        assert pair.p2p_delta == pytest.approx(0.0)
        # b01=0, b10=4 -> 2 * P(Bin(4, 0.5) <= 0) = 0.125
        assert pair.mcnemar_p == pytest.approx(0.125)
        assert pair.ci_lower > 0.0
        assert pair.ci_high > pair.ci_lower
        assert pair.champion_metrics is not None
        assert pair.candidate_metrics is not None

    def test_evaluate_pair_counts_both_discordant_directions(self):
        # Champion solves i1..i5, candidate solves i3..i9: both discordant
        # directions occur — b01=2 (i1,i2 champion-only), b10=4 (i6..i9
        # candidate-only) -> 2 * P(Bin(6, 0.5) <= 2) = 0.6875.
        config = _pair_config()
        champion_results = [_result(f"i{index}", f2p=1.0) for index in range(1, 6)] + [
            _result(f"i{index}", f2p=0.0) for index in range(6, 11)
        ]
        candidate_results = [
            _result(f"i{index}", f2p=1.0, variant=CANDIDATE_VARIANT) for index in range(3, 10)
        ] + [_result(f"i{index}", f2p=0.0, variant=CANDIDATE_VARIANT) for index in (1, 2, 10)]
        pair = evaluate_pair(
            _make_run("champion-run", config, champion_results),
            _make_run("candidate-run", config, candidate_results),
            config,
        )
        assert pair.f2p_gain == pytest.approx(0.2)
        assert pair.mcnemar_p == pytest.approx(0.6875)

    def test_evaluate_pair_asserts_same_dataset_run_id(self):
        champion = _make_run("champion-run", _pair_config(), [_result("i1", f2p=1.0)])
        other_window = EvalConfig(dataset_run_id="other-repos", tier_seed=42)
        candidate = _make_run(
            "candidate-run", other_window, [_result("i1", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        with pytest.raises(AssertionError, match="dataset_run_id"):
            evaluate_pair(champion, candidate, _pair_config())

    def test_evaluate_pair_asserts_same_tier_seed(self):
        champion = _make_run("champion-run", _pair_config(), [_result("i1", f2p=1.0)])
        other_window = EvalConfig(dataset_run_id="expanded-repos", tier_seed=7)
        candidate = _make_run(
            "candidate-run", other_window, [_result("i1", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        with pytest.raises(AssertionError, match="tier_seed"):
            evaluate_pair(champion, candidate, _pair_config())

    def test_evaluate_pair_pairs_only_the_instance_intersection(self):
        # Champion-only instance i5 must stay out of the paired vectors: the
        # bootstrap CI is recomputed here over the intersection (i1..i4) and
        # must match the gate's exactly (same seed-42 vectors, so equality is
        # deterministic, not approximate).
        config = _pair_config()
        champion_results = [
            _result("i1", f2p=1.0),
            _result("i2", f2p=1.0),
            _result("i3", f2p=0.0),
            _result("i4", f2p=0.0),
            _result("i5", f2p=1.0),  # champion-only
        ]
        candidate_results = [
            _result("i1", f2p=1.0, variant=CANDIDATE_VARIANT),
            _result("i2", f2p=1.0, variant=CANDIDATE_VARIANT),
            _result("i3", f2p=1.0, variant=CANDIDATE_VARIANT),
            _result("i4", f2p=0.0, variant=CANDIDATE_VARIANT),
        ]
        pair = evaluate_pair(
            _make_run("champion-run", config, champion_results),
            _make_run("candidate-run", config, candidate_results),
            config,
        )

        shared = ["i1", "i2", "i3", "i4"]
        champ_vec = [1.0, 1.0, 0.0, 0.0]
        cand_vec = [1.0, 1.0, 1.0, 0.0]
        expected_lo, expected_hi, _ = paired_bootstrap_ci(cand_vec, champ_vec)
        assert pair.ci_lower == expected_lo
        assert pair.ci_high == expected_hi
        # Aggregates still cover each run's full instance set — intersection
        # applies to the significance vectors only, not the aggregate rates.
        assert pair.champion_metrics is not None
        assert pair.champion_metrics.total_examples == 5
        assert pair.candidate_metrics is not None
        assert pair.candidate_metrics.total_examples == 4

    @pytest.mark.parametrize("empty_side", ["champion", "candidate"])
    def test_evaluate_pair_raises_on_empty_run(self, empty_side):
        config = _pair_config()
        champion_results = [] if empty_side == "champion" else [_result("i1", f2p=1.0)]
        candidate_results = (
            [] if empty_side == "candidate" else [_result("i1", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        champion = _make_run("champion-run", config, champion_results)
        candidate = _make_run("candidate-run", config, candidate_results)
        with pytest.raises(ValueError, match=f"{empty_side}-run"):
            evaluate_pair(champion, candidate, config)

    def test_evaluate_pair_raises_when_instances_disjoint(self):
        config = _pair_config()
        champion = _make_run("champion-run", config, [_result("i1", f2p=1.0)])
        candidate = _make_run(
            "candidate-run", config, [_result("i9", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        with pytest.raises(ValueError, match="no shared instances"):
            evaluate_pair(champion, candidate, config)

    def test_evaluate_pair_missing_aggregate_metrics_defaults_gain_zero(self):
        # Defensive path (spec §4.2): when a run's aggregate key diverges from
        # its results (simulated by a stale models_evaluated entry), metrics
        # fields are None and gains fall back to 0.0 instead of fabricating
        # a gain.  Vectors still pair (variant matches) — only the aggregate
        # lookup misses.
        config = _pair_config()
        champion = _make_run("champion-run", config, [_result("i1", f2p=1.0)])
        champion.models_evaluated = ["qwen3-14b:baseline_14b"]
        champion.results[0].model_name = "other-model"  # aggregate key diverges
        candidate = _make_run(
            "candidate-run", config, [_result("i1", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        pair = evaluate_pair(champion, candidate, config)
        assert pair.champion_metrics is None
        assert pair.candidate_metrics is not None
        assert pair.f2p_gain == 0.0
        assert pair.p2p_delta == 0.0

    def test_evaluate_pair_rejects_runs_with_multiple_variants(self):
        config = _pair_config()
        results = [
            _result("i1", f2p=1.0),
            _result("i1", f2p=1.0, variant=CANDIDATE_VARIANT),
        ]
        candidate = _make_run(
            "candidate-run", config, [_result("i1", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        with pytest.raises(ValueError, match="exactly one"):
            evaluate_pair(_make_run("champion-run", config, results), candidate, config)

    def test_evaluate_pair_derives_variant_from_results(self):
        # models_evaluated empty -> variant derived from run.results.
        config = _pair_config()
        champion = _make_run("champion-run", config, [_result("i1", f2p=1.0)])
        champion.models_evaluated = []
        candidate = _make_run(
            "candidate-run", config, [_result("i1", f2p=1.0, variant=CANDIDATE_VARIANT)]
        )
        pair = evaluate_pair(champion, candidate, config)
        assert pair.champion_metrics is not None
        # Identical vectors -> zero gain: CI collapses to 0, no discordant pairs.
        assert pair.ci_lower == 0.0
        assert pair.ci_high == 0.0
        assert pair.mcnemar_p == 1.0

    def test_pair_eval_is_frozen(self):
        pair = PairEval(
            champion_metrics=None,
            candidate_metrics=None,
            f2p_gain=0.0,
            p2p_delta=0.0,
            ci_lower=0.0,
            ci_high=0.0,
            mcnemar_p=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            pair.f2p_gain = 1.0  # type: ignore[misc]

    def test_revalidate_champion_passthrough_picks_passing_winner(self):
        passing = _metrics("qwen3-14b", CANDIDATE_VARIANT, f2p_rate=0.25, p2p_rate=0.92)
        failing = _metrics("qwen3-14b", "baseline_14b", f2p_rate=0.10, p2p_rate=0.95)
        winner = revalidate_champion(
            {f"qwen3-14b:{CANDIDATE_VARIANT}": passing, "qwen3-14b:baseline_14b": failing},
            proxy_champion="baseline_14b",
            min_f2p=0.15,
            min_p2p=0.90,
        )
        assert winner == (f"qwen3-14b:{CANDIDATE_VARIANT}", passing)

    def test_revalidate_champion_passthrough_none_when_all_fail(self):
        all_failing = {
            "qwen3-14b:low-f2p": _metrics("qwen3-14b", "low-f2p", f2p_rate=0.10, p2p_rate=0.95),
            "qwen3-14b:low-p2p": _metrics("qwen3-14b", "low-p2p", f2p_rate=0.20, p2p_rate=0.80),
        }
        assert (
            revalidate_champion(
                all_failing, proxy_champion="baseline_14b", min_f2p=0.15, min_p2p=0.90
            )
            is None
        )


class TestRegistry:
    """``promotion.registry`` champion.json source of truth (spec §4.3).

    Schema keys in spec order (stable, not alphabetized); ``previous`` is
    self-referential so rollback needs no extra state (spec decision 7).
    """

    SCHEMA_KEYS = [
        "variant",
        "model_ref",
        "f2p_rate",
        "p2p_rate",
        "dataset_run_id",
        "tier",
        "seed",
        "promoted_at",
        "previous",
    ]

    def _record(self, **overrides) -> ChampionRecord:
        """2026-08-06 Champion (spec §4.7 values), overridable per field."""
        fields = {
            "variant": "higher_lr_14b",
            "model_ref": "qwen3-14b:higher_lr_14b",
            "f2p_rate": 0.169,
            "p2p_rate": 0.912,
            "dataset_run_id": "expanded-repos",
            "tier": "full",
            "seed": 42,
            "promoted_at": "2026-08-06T12:00:00+00:00",
            "previous": None,
        }
        fields.update(overrides)
        return ChampionRecord(**fields)

    def test_round_trip_write_read(self, tmp_path):
        path = tmp_path / "champion.json"
        record = self._record()
        write_champion(path, record)
        assert read_champion(path) == record
        assert read_champion(path).previous is None

    def test_previous_chain_round_trip(self, tmp_path):
        # three-level chain: champion -> previous -> previous.previous (leaf)
        leaf = self._record(variant="baseline_14b")
        middle = self._record(variant="higher_rank_14b", previous=leaf)
        champion = self._record(previous=middle)
        path = tmp_path / "champion.json"
        write_champion(path, champion)

        restored = read_champion(path)
        assert restored == champion
        previous = restored.previous
        assert previous is not None
        assert previous == middle
        grandprevious = previous.previous
        assert grandprevious is not None
        assert grandprevious == leaf
        assert grandprevious.previous is None

    def test_to_dict_schema_shape_and_order(self):
        data = to_dict(self._record())
        assert list(data) == self.SCHEMA_KEYS  # spec field order
        assert set(data) == set(self.SCHEMA_KEYS)  # exactly the 9 keys
        assert data["f2p_rate"] == 0.169
        assert data["previous"] is None
        assert from_dict(data) == self._record()

    def test_from_dict_nested_previous(self):
        data = to_dict(self._record(variant="v2", previous=self._record()))
        record = from_dict(data)
        assert record.variant == "v2"
        previous = record.previous
        assert previous is not None
        assert previous.variant == "higher_lr_14b"
        assert previous.previous is None

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="champion record not found"):
            read_champion(tmp_path / "no-champion.json")

    def test_read_malformed_json_raises(self, tmp_path):
        path = tmp_path / "champion.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed JSON"):
            read_champion(path)

    def test_read_missing_field_raises(self, tmp_path):
        path = tmp_path / "champion.json"
        path.write_text(json.dumps({"variant": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="model_ref"):
            read_champion(path)

    def test_read_bad_type_raises(self, tmp_path):
        path = tmp_path / "champion.json"
        path.write_text(json.dumps(to_dict(self._record(f2p_rate="0.169"))), encoding="utf-8")
        with pytest.raises(ValueError, match="f2p_rate"):
            read_champion(path)

    def test_read_non_object_json_raises(self, tmp_path):
        path = tmp_path / "champion.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read_champion(path)

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "champion.json"
        write_champion(path, self._record())
        assert path.is_file()

    def test_write_output_is_valid_json_with_nested_previous(self, tmp_path):
        path = tmp_path / "champion.json"
        write_champion(path, self._record(variant="v2", previous=self._record()))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["variant"] == "v2"
        assert data["previous"]["variant"] == "higher_lr_14b"
        assert data["previous"]["previous"] is None
        # round-trip through the parsed dict reproduces the record exactly
        assert from_dict(data) == read_champion(path)

    @pytest.fixture
    def _fake_comparison(self):
        """Fake ``comparison`` module replacing ``promotion.registry.comparison``."""

        def make(summary: str | None):
            calls: list[str] = []

            def clear(api, config):  # noqa: ARG001
                calls.append("clear")

            def promote(key, config):  # noqa: ARG001
                calls.append("promote")
                return summary

            return SimpleNamespace(
                _clear_champion_alias=clear, promote_champion_to_registry=promote
            ), calls

        return make

    def test_sync_alias_clear_then_promote_call_order(self, monkeypatch, _fake_comparison):
        fake, calls = _fake_comparison(
            "W&B Registry: champion alias -> model-qwen3-14b-higher_lr_14b"
        )
        monkeypatch.setattr("promotion.registry.comparison", fake)
        # hermetic wandb: fake module object so no network/credentials needed
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda timeout: object()))

        result = sync_alias("qwen3-14b:higher_lr_14b", EvalConfig())
        assert result == "W&B Registry: champion alias -> model-qwen3-14b-higher_lr_14b"
        assert calls == ["clear", "promote"]  # stale alias cleared first, then link

    def test_sync_alias_returns_none_when_promotion_fails(self, monkeypatch, _fake_comparison):
        fake, _ = _fake_comparison(None)
        monkeypatch.setattr("promotion.registry.comparison", fake)
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda timeout: object()))

        assert sync_alias("qwen3-14b:higher_lr_14b", EvalConfig()) is None

    def test_sync_alias_lazy_wandb_import_never_raises(self, monkeypatch):
        # sys.modules["wandb"] = None makes `import wandb` raise ImportError:
        # sync_alias must degrade to None (lazy import, spec §4.3) — the
        # comparison module is not even reached.
        monkeypatch.setitem(sys.modules, "wandb", None)
        monkeypatch.setattr("promotion.registry.comparison", SimpleNamespace())
        assert sync_alias("qwen3-14b:higher_lr_14b", EvalConfig()) is None


class TestSeedChampion:
    """``scripts/seed_champion.py`` — one-shot 2026-08-06 baseline (spec §4.7)."""

    def _run(self, tmp_path, mocker):
        from scripts import seed_champion  # lazy: mirrors tests/scripts precedent

        out = tmp_path / "champion.json"
        mocker.patch("sys.argv", ["seed_champion", "--output", str(out)])
        return seed_champion.main(), out

    def test_seed_writes_exact_champion_values(self, tmp_path, mocker):
        code, out = self._run(tmp_path, mocker)
        assert code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["variant"] == "higher_lr_14b"
        assert data["model_ref"] == "qwen3-14b:higher_lr_14b"
        assert data["f2p_rate"] == 0.169
        assert data["p2p_rate"] == 0.912
        assert data["dataset_run_id"] == "expanded-repos"
        assert data["tier"] == "full"
        assert data["seed"] == 42
        assert data["previous"] is None

    def test_seed_promoted_at_is_utc_timestamp(self, tmp_path, mocker):
        _, out = self._run(tmp_path, mocker)
        promoted = datetime.fromisoformat(
            json.loads(out.read_text(encoding="utf-8"))["promoted_at"]
        )
        assert promoted.tzinfo == UTC

    def test_main_accepts_argv_and_returns_zero(self, tmp_path):
        from scripts import seed_champion  # lazy: mirrors tests/scripts precedent

        out = tmp_path / "champion.json"
        assert seed_champion.main(["--output", str(out)]) == 0
        assert out.is_file()


# ── audit trail (spec §4.4, decision 9) ─────────────────────────────────────


class TestAudit:
    """``promotion.audit`` — decision record, markdown, W&B telemetry.

    Field-set assertions pin the frozen spec §4.4 schema; the wandb uploads
    are exercised with a fake wandb module (offline) and with
    ``sys.modules["wandb"] = None`` to prove the never-raise degradation.
    """

    def _record(self, **overrides) -> dict:
        """Minimal promote decision record, overridable per field."""
        deployed = overrides.pop("deployed", False)
        fields: dict[str, Any] = {
            "decision_id": "dec-1",
            "pipeline_version": "9.0.0",
            "candidate_run_id": "run-cand",
            "candidate_variant": CANDIDATE_VARIANT,
            "candidate_model_ref": "qwen3-14b:higher_lr_14b",
            "champion_run_id": "run-champ",
            "champion_variant": "baseline_14b",
            "champion_model_ref": "qwen3-14b:baseline_14b",
            "candidate_f2p": 0.25,
            "champion_f2p": 0.169,
            "candidate_p2p": 0.93,
            "champion_p2p": 0.912,
            "f2p_gain": 0.081,
            "p2p_delta": 0.018,
            "ci_lower": 0.02,
            "ci_high": 0.14,
            "mcnemar_p": 0.031,
            "outcome": OUTCOME_PROMOTE,
            "reasons": [],
            "git_sha": "abc123",
        }
        fields.update(overrides)
        record = audit.build_decision_record(**fields)
        record["deployed"] = deployed  # deploy job flips this post-build
        return record

    def test_build_decision_record_frozen_field_set(self):
        record = self._record()
        assert set(record) == {
            "decision_id",
            "pipeline_version",
            "candidate",
            "incumbent",
            "metrics",
            "thresholds",
            "outcome",
            "reasons",
            "deployed",
            "timestamps",
            "git_sha",
        }
        assert set(record["candidate"]) == {"run_id", "variant", "model_ref"}
        assert set(record["incumbent"]) == {"run_id", "variant", "model_ref"}
        assert set(record["metrics"]) == {
            "candidate",
            "incumbent",
            "f2p_gain",
            "p2p_delta",
            "ci_lower",
            "ci_high",
            "mcnemar_p",
        }
        assert set(record["metrics"]["candidate"]) == {"f2p_rate", "p2p_rate"}
        assert set(record["metrics"]["incumbent"]) == {"f2p_rate", "p2p_rate"}
        assert set(record["thresholds"]) == {"min_f2p_gain", "max_p2p_regression", "floors"}
        assert set(record["thresholds"]["floors"]) == {"f2p", "p2p"}
        assert set(record["timestamps"]) == {"created_utc"}

    def test_build_decision_record_values_from_rules_constants(self):
        record = self._record()
        assert record["decision_id"] == "dec-1"
        assert record["pipeline_version"] == "9.0.0"
        assert record["candidate"] == {
            "run_id": "run-cand",
            "variant": CANDIDATE_VARIANT,
            "model_ref": "qwen3-14b:higher_lr_14b",
        }
        assert record["incumbent"]["variant"] == "baseline_14b"
        assert record["metrics"]["candidate"]["f2p_rate"] == 0.25
        assert record["metrics"]["incumbent"]["p2p_rate"] == 0.912
        assert record["metrics"]["f2p_gain"] == 0.081
        assert record["outcome"] == OUTCOME_PROMOTE
        assert record["reasons"] == []
        assert record["deployed"] is False
        assert record["git_sha"] == "abc123"
        # thresholds come from promotion.rules constants, not re-hardcoded
        assert record["thresholds"]["min_f2p_gain"] == PROMOTE_MIN_F2P_GAIN == 0.05
        assert record["thresholds"]["max_p2p_regression"] == PROMOTE_MAX_P2P_REGRESSION == 0.02
        assert record["thresholds"]["floors"] == {"f2p": 0.15, "p2p": 0.90}
        created = datetime.fromisoformat(record["timestamps"]["created_utc"])
        assert created.tzinfo == UTC

    def test_render_markdown_summary(self):
        md = audit.render_markdown(self._record())
        assert "# Promotion decision: dec-1" in md
        assert f"`{OUTCOME_PROMOTE}`" in md
        assert CANDIDATE_VARIANT in md
        assert "baseline_14b" in md
        assert "+0.081" in md  # f2p_gain formatted as signed
        assert "+0.018" in md  # p2p_delta formatted as signed
        assert "0.020" in md  # ci_lower
        assert "0.0310" in md  # mcnemar_p
        assert "Min F2P gain: 0.050" in md
        assert "Floors: F2P 0.15, P2P 0.90" in md

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            (
                {"outcome": OUTCOME_PROMOTE, "deployed": True},
                {
                    "promote/outcome": 1.0,
                    "promote/f2p_gain": 0.081,
                    "promote/p2p_delta": 0.018,
                    "promote/ci_lower": 0.02,
                    "promote/mcnemar_p": 0.031,
                    "promote/deploy_status": 1.0,
                },
            ),
            (
                {"outcome": OUTCOME_REJECT, "deployed": False, "reasons": ["micro-gain"]},
                {
                    "promote/outcome": 0.0,
                    "promote/f2p_gain": 0.081,
                    "promote/p2p_delta": 0.018,
                    "promote/ci_lower": 0.02,
                    "promote/mcnemar_p": 0.031,
                    "promote/deploy_status": 0.0,
                },
            ),
        ],
    )
    def test_log_decision_metrics_exact_six_literal_keys(self, monkeypatch, overrides, expected):
        calls: list[dict] = []
        monkeypatch.setattr(audit, "log_metrics", calls.append)
        audit.log_decision_metrics(self._record(**overrides))
        assert calls == [expected]
        # literal insertion order is the registry order — pinned by the AST walk
        assert list(calls[0]) == list(expected)

    def test_write_decision_record_degrades_without_wandb(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        assert audit.write_decision_record(self._record(), entity="e", project="p") is None

    def test_write_decision_record_uploads_artifact(self, monkeypatch):
        seen: dict[str, list] = {"init": [], "add_file": [], "log_artifact": [], "finish": []}

        class FakeArtifact:
            def __init__(self, name, type):
                self.name = name
                seen["init"].append(("artifact", name, type))

            def add_file(self, path, name=None):
                seen["add_file"].append((path, name))
                return self

        class FakeWandb:
            run = None  # no live run: log_metrics is a no-op

            def __init__(self):
                self.Artifact = FakeArtifact

            def init(self, **kwargs):
                seen["init"].append(kwargs)
                return self

            def log_artifact(self, artifact):
                seen["log_artifact"].append(artifact.name)

            def finish(self):
                seen["finish"].append(True)

        monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
        name = audit.write_decision_record(self._record(), entity="my-entity", project="my-project")
        assert name == "promotion-decision-dec-1"
        assert seen["init"][0] == {
            "project": "my-project",
            "entity": "my-entity",
            "job_type": "promotion-decision",
            "name": "decision-dec-1",
            "reinit": "finish_previous",
        }
        assert seen["init"][1] == ("artifact", "promotion-decision-dec-1", "decision")
        assert sorted(n for _p, n in seen["add_file"]) == ["decision.json", "decision.md"]
        # temp dir cleaned up after add_file copied the files into the artifact
        assert all(not Path(p).exists() for p, _n in seen["add_file"])
        assert seen["log_artifact"] == ["promotion-decision-dec-1"]
        assert seen["finish"] == [True]

    def test_note_gating_off_logs_reject_decision(self, monkeypatch):
        seen: dict[str, list] = {"init": [], "finish": []}
        metric_calls: list[dict] = []

        class FakeWandb:
            run = None  # no live run: log_metrics is a no-op

            def __init__(self):
                self.config_updates: list[dict] = []
                self.config = SimpleNamespace(update=self.config_updates.append)

            def init(self, **kwargs):
                seen["init"].append(kwargs)
                return self

            def finish(self):
                seen["finish"].append(True)

        monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
        monkeypatch.setattr(audit, "log_metrics", metric_calls.append)
        result = audit.note_gating_off(CANDIDATE_VARIANT, "eval-disabled", entity="e", project="p")
        assert result == f"gating-off-{CANDIDATE_VARIANT}"
        assert seen["init"] == [
            {
                "project": "p",
                "entity": "e",
                "job_type": "promotion-decision",
                "name": f"gating-off-{CANDIDATE_VARIANT}",
                "reinit": "finish_previous",
            }
        ]
        assert metric_calls == [
            {
                "promote/outcome": 0.0,
                "promote/f2p_gain": 0.0,
                "promote/p2p_delta": 0.0,
                "promote/ci_lower": 0.0,
                "promote/mcnemar_p": 0.0,
                "promote/deploy_status": 0.0,
            }
        ]
        assert seen["finish"] == [True]

    def test_note_gating_off_never_raises_without_wandb(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        assert (
            audit.note_gating_off(CANDIDATE_VARIANT, "eval-disabled", entity="e", project="p")
            is None
        )


class TestDeploy:
    """``promotion.deploy`` — variant-pinned deploy, probe, alias, rollback (spec §4.5).

    Nothing here touches the network or a shell: ``subprocess.run``, ``httpx``
    and ``sync_alias`` are monkeypatched at the module attribute, and dry-run
    never executes anything (plan §4.11: exec/network lines are
    ``# pragma: no cover``).
    """

    CMD = ["uv", "run", "modal", "deploy", "-m", "inference.modal_serve"]
    KEY = "qwen3-14b:higher_lr_14b"
    SECRETS = {"MODAL_TOKEN_ID": "t", "MODAL_TOKEN_SECRET": "s", "HF_TOKEN": "h"}

    @staticmethod
    def _config() -> ServeConfig:
        return ServeConfig()

    @staticmethod
    def _previous() -> ChampionRecord:
        """2026-08-06 Champion (spec §4.7 values) — the incumbent to roll back to."""
        return ChampionRecord(
            variant="higher_lr_14b",
            model_ref="qwen3-14b:higher_lr_14b",
            f2p_rate=0.169,
            p2p_rate=0.912,
            dataset_run_id="expanded-repos",
            tier="full",
            seed=42,
            promoted_at="2026-08-06T12:00:00+00:00",
            previous=None,
        )

    @staticmethod
    def _stub_httpx(
        liveness: int = 200, chat: int = 200, elapsed: float = 0.42, chat_text: str = ""
    ) -> Any:
        """Stand-in ``httpx`` module: ``codes.OK`` + scriptable get/post."""

        class _FakeCodes:
            OK = 200

        class _FakeResponse:
            def __init__(self, status_code: int, elapsed: float, text: str) -> None:
                self.status_code = status_code
                self.elapsed = elapsed
                self.text = text

        class _FakeHttpx:
            codes = _FakeCodes

            def get(self, url: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG001
                return _FakeResponse(liveness, 0.0, "ok")

            def post(self, url: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG001
                return _FakeResponse(chat, elapsed, chat_text)

        return _FakeHttpx()

    # ── variant pinning (spec decision 6: config-gap abort) ────────────────

    @pytest.mark.parametrize("variant", ["baseline_14b", "higher_rank_14b", "higher_lr_14b"])
    def test_assert_variant_known_accepts_trained_variants(self, variant):
        assert_variant_known(variant, self._config())

    def test_assert_variant_known_rejects_untrained_variant(self):
        with pytest.raises(ConfigGapError, match="ghost_variant"):
            assert_variant_known("ghost_variant", self._config())

    def test_deploy_unknown_variant_aborts_before_exec(self, monkeypatch):
        def _must_not_run(*_args, **_kwargs) -> None:
            raise AssertionError("config-gap deploy must not execute subprocess")

        monkeypatch.setattr(deploy_mod.subprocess, "run", _must_not_run)
        with pytest.raises(ConfigGapError, match="ghost_variant"):
            deploy("qwen3-14b:ghost_variant", self._config(), env=self.SECRETS)

    # ── deploy (spec §4.5) ─────────────────────────────────────────────────

    def test_deploy_dry_run_never_executes(self, monkeypatch):
        def _must_not_run(*_args, **_kwargs) -> None:
            raise AssertionError("dry-run deploy must not execute subprocess")

        monkeypatch.setattr(deploy_mod.subprocess, "run", _must_not_run)
        result = deploy(self.KEY, self._config(), dry_run=True)
        assert result.returncode == 0
        assert result.stdout == "(dry-run)"
        assert result.stderr == ""
        assert result.args == self.CMD

    def test_deploy_env_pins_variant_and_carries_secrets(self):
        merged = deploy_mod._deploy_env("higher_lr_14b", self.SECRETS)
        assert merged["SERVING_DEFAULT_VARIANT"] == "higher_lr_14b"
        for name, value in self.SECRETS.items():
            assert merged[name] == value

    def test_deploy_env_pin_applied_last_so_it_wins(self):
        merged = deploy_mod._deploy_env("higher_lr_14b", {"SERVING_DEFAULT_VARIANT": "wrong"})
        assert merged["SERVING_DEFAULT_VARIANT"] == "higher_lr_14b"

    def test_deploy_real_path_runs_modal_with_pinned_env(self, monkeypatch):
        calls: list[tuple[list[str], dict[str, Any]]] = []
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="deployed", stderr="")

        def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append((cmd, kwargs))
            return fake

        monkeypatch.setattr(deploy_mod.subprocess, "run", _fake_run)
        result = deploy(self.KEY, self._config(), env=self.SECRETS)
        assert result.stdout == "deployed"
        cmd, kwargs = calls[0]
        assert cmd == self.CMD
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        env = kwargs["env"]
        assert env["SERVING_DEFAULT_VARIANT"] == "higher_lr_14b"
        for name, value in self.SECRETS.items():
            assert env[name] == value

    # ── health probe (spec decision 6: liveness pre-check + chat TTFB) ─────

    def test_health_check_returns_elapsed_on_green(self, monkeypatch):
        monkeypatch.setattr(deploy_mod, "httpx", self._stub_httpx())
        assert health_check("http://serve.modal.run", "t0k") == 0.42

    def test_health_check_raises_on_liveness_non_200(self, monkeypatch):
        monkeypatch.setattr(deploy_mod, "httpx", self._stub_httpx(liveness=503))
        with pytest.raises(ProbeError, match="503"):
            health_check("http://serve.modal.run", "t0k")

    def test_health_check_raises_on_chat_non_200(self, monkeypatch):
        monkeypatch.setattr(
            deploy_mod, "httpx", self._stub_httpx(chat=401, chat_text="unauthorized")
        )
        with pytest.raises(ProbeError, match="401"):
            health_check("http://serve.modal.run", "t0k")

    def test_health_check_raises_on_transport_exception(self, monkeypatch):
        class _BrokenHttpx:
            def get(self, url: str, **kwargs: Any) -> None:  # noqa: ARG001
                raise RuntimeError("conn refused")

        monkeypatch.setattr(deploy_mod, "httpx", _BrokenHttpx())
        with pytest.raises(ProbeError, match="conn refused"):
            health_check("http://serve.modal.run", "t0k")

    # ── alias sync (spec decision 6: alias failure aborts) ─────────────────

    def test_sync_alias_or_abort_raises_when_alias_sync_fails(self, monkeypatch):
        monkeypatch.setattr(deploy_mod, "sync_alias", lambda _key, _config: None)
        with pytest.raises(RuntimeError, match="alias sync failed"):
            sync_alias_or_abort(self.KEY, EvalConfig())

    def test_sync_alias_or_abort_passes_when_alias_sync_ok(self, monkeypatch):
        monkeypatch.setattr(deploy_mod, "sync_alias", lambda _key, _config: "linked")
        sync_alias_or_abort(self.KEY, EvalConfig())  # must not raise

    # ── rollback (spec decision 7: same pipeline, no extra state) ──────────

    def test_rollback_dry_run_never_executes(self, monkeypatch):
        def _must_not_run(*_args, **_kwargs) -> None:
            raise AssertionError("dry-run rollback must not execute subprocess")

        monkeypatch.setattr(deploy_mod.subprocess, "run", _must_not_run)
        result = rollback(self._previous(), self._config(), dry_run=True)
        assert result["outcome"] == "rollback-dry-run"
        assert result["variant"] == "higher_lr_14b"
        assert result["champion_key"] == self.KEY
        assert result["deployed"] is False
        assert result["probe_ok"] is False
        assert result["alias_synced"] is False
        assert datetime.fromisoformat(str(result["triggered_at"])).tzinfo == UTC

    def test_rollback_real_path_repromotes_then_probes_and_syncs(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            deploy_mod.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ok", stderr=""
            ),
        )

        def _fake_probe(base_url: str, token: str) -> float:
            calls.append(f"probe:{base_url}:{token}")
            return 0.5

        monkeypatch.setattr(deploy_mod, "health_check", _fake_probe)
        monkeypatch.setattr(deploy_mod, "sync_alias", lambda _key, _config: "linked")
        result = rollback(
            self._previous(),
            self._config(),
            env={"MODAL_WEB_URL": "http://serve.modal.run", "MODAL_SERVE_TOKEN": "t0k"},
        )
        assert result["outcome"] == "rollback"
        assert result["champion_key"] == self.KEY
        assert result["deployed"] is True
        assert result["probe_ok"] is True
        assert result["alias_synced"] is True
        assert calls == ["probe:http://serve.modal.run:t0k"]
        assert datetime.fromisoformat(str(result["triggered_at"])).tzinfo == UTC


class TestLaunchEvictions:
    """``promotion.run`` eviction orchestration (spec §4.6 step 3).

    Spawn/poll/reap are exercised with fakes: Popen is monkeypatched at the
    module attribute, the poll loop runs against a scriptable
    ``load_all_eval_runs`` with ``EVAL_TIMEOUT_S`` collapsed, and nothing
    ever touches a shell or the network.
    """

    CMD = ["uv", "run", "eval", "run"]

    @staticmethod
    def _cmd(variant: str, run_id: str, mode: str = "expanded-repos") -> list[str]:
        return TestLaunchEvictions.CMD + [
            "--mode",
            mode,
            "--models",
            f"qwen3-14b:{variant}",
            "--resume",
            run_id,
        ]

    class _FakePopen:
        """Scriptable stand-in: running/exited + optional wait timeout."""

        def __init__(
            self,
            cmd: list[str],
            running: bool = True,
            timeout_on_wait: bool = False,
            **kwargs: Any,
        ) -> None:
            self.args = cmd
            self.kwargs = kwargs
            self._running = running
            self._timeout_on_wait = timeout_on_wait
            self.terminated = False
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self._running else 0

        def terminate(self) -> None:
            self.terminated = True
            self._running = False

        def wait(self, timeout: float = 0) -> int:
            if self._timeout_on_wait:
                raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout)
            self._running = False
            return 0

    # ── spawn (spec §4.6 step 3: one subprocess per pair, in parallel) ─────

    def test_launch_evals_spawns_one_subprocess_per_pair(self, monkeypatch):
        spawned: list[tuple[list[str], dict[str, Any]]] = []

        def _fake_popen(cmd: list[str], **kwargs: Any) -> TestLaunchEvictions._FakePopen:
            spawned.append((cmd, kwargs))
            return TestLaunchEvictions._FakePopen(cmd, **kwargs)

        monkeypatch.setattr(run_mod.subprocess, "Popen", _fake_popen)
        procs = run_mod._launch_evals(
            [("baseline_14b", "run-1"), ("higher_lr_14b", "run-2")],
            mode="expanded-repos",
            base_model="qwen3-14b",
        )

        assert [args for args, _kwargs in spawned] == [
            self._cmd("baseline_14b", "run-1"),
            self._cmd("higher_lr_14b", "run-2"),
        ]
        assert len(procs) == 2
        for _args, kwargs in spawned:
            assert kwargs["stdout"] is subprocess.DEVNULL
            assert kwargs["stderr"] is subprocess.DEVNULL

    # ── poll (spec §4.6 step 3: completed/failed/absent/timeout) ───────────

    def _config(self) -> EvalConfig:
        return _pair_config()

    def test_wait_for_pair_returns_runs_when_both_completed(self, monkeypatch):
        champion = _make_run("run-1", self._config(), [])
        candidate = _make_run("run-2", self._config(), [])
        monkeypatch.setattr(run_mod, "load_all_eval_runs", lambda _ids, _cfg: [champion, candidate])
        sleeps: list[float] = []
        monkeypatch.setattr(run_mod.time, "sleep", sleeps.append)

        pair, reason = run_mod._wait_for_pair("run-1", "run-2", self._config())

        assert reason == "ok"
        assert pair == (champion, candidate)
        assert sleeps == []  # both present and completed: no polling round

    def test_wait_for_pair_fails_fast_on_failed_status(self, monkeypatch):
        champion = _make_run("run-1", self._config(), []).model_copy(update={"status": "failed"})
        candidate = _make_run("run-2", self._config(), [])
        monkeypatch.setattr(run_mod, "load_all_eval_runs", lambda _ids, _cfg: [champion, candidate])

        pair, reason = run_mod._wait_for_pair("run-1", "run-2", self._config())

        assert pair is None
        assert reason == "eval-failed"

    def test_wait_for_pair_polls_while_runs_absent_then_times_out(self, monkeypatch):
        monkeypatch.setattr(run_mod, "load_all_eval_runs", lambda _ids, _cfg: [])
        sleeps: list[float] = []
        monkeypatch.setattr(run_mod.time, "sleep", sleeps.append)
        monkeypatch.setattr(run_mod, "EVAL_TIMEOUT_S", 0.001)

        pair, reason = run_mod._wait_for_pair("run-1", "run-2", self._config())

        assert pair is None
        assert reason == "eval-timeout"
        assert sleeps  # at least one full poll round ran before the deadline

    # ── reap (best-effort cleanup after the poll) ──────────────────────────

    def test_reap_terminates_still_running_procs(self):
        procs: list[Any] = [TestLaunchEvictions._FakePopen([], running=True)]
        run_mod._reap(procs)
        assert procs[0].terminated is True

    def test_reap_skips_terminate_for_exited_procs(self):
        procs: list[Any] = [TestLaunchEvictions._FakePopen([], running=False)]
        run_mod._reap(procs)
        assert procs[0].terminated is False
        assert procs[0].poll_count == 1

    def test_reap_swallows_wait_timeout(self):
        procs: list[Any] = [TestLaunchEvictions._FakePopen([], running=True, timeout_on_wait=True)]
        run_mod._reap(procs)  # must not raise
        assert procs[0].terminated is True

    # ── GITHUB_OUTPUT (spec §4.6 step 6) ───────────────────────────────────

    def test_write_github_output_appends_when_env_set(self, tmp_path, monkeypatch):
        out = tmp_path / "github_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        run_mod._write_github_output(["promote=true", "decision_id=dec-1"])
        assert out.read_text(encoding="utf-8") == "promote=true\ndecision_id=dec-1\n"

    def test_write_github_output_noop_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        run_mod._write_github_output(["promote=true"])  # must not raise
        assert list(tmp_path.iterdir()) == []
