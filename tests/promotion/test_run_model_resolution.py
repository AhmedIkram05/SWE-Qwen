"""Offline unit tests for the promotion model-base resolution (Step 6).

The decide/deploy flows build every ``model:variant`` ref from one pure
function; these tests pin the 1-2-3 priority order (``--candidate-model`` →
champion record's ``model_ref`` base → ``ServeConfig.base_model`` fallback)
with no config, GCS, or network.
"""

from __future__ import annotations

from promotion.registry import ChampionRecord
from promotion.run import resolve_candidate_model_base


def _record(model_ref: str) -> ChampionRecord:
    return ChampionRecord(
        variant=model_ref.partition(":")[2],
        model_ref=model_ref,
        f2p_rate=0.172,
        p2p_rate=0.901,
        dataset_run_id="expanded-repos",
        tier="full",
        seed=42,
        promoted_at="2026-08-06T12:00:00+00:00",
        previous=None,
    )


def test_explicit_candidate_model_wins():
    base = resolve_candidate_model_base(
        "llama-31-8b", _record("qwen3-14b:higher_rank_14b"), "qwen3-14b"
    )
    assert base == "llama-31-8b"


def test_champion_model_ref_base_when_no_arg():
    base = resolve_candidate_model_base(None, _record("llama-31-8b:higher_rank_14b"), "qwen3-14b")
    assert base == "llama-31-8b"


def test_serve_fallback_is_last_resort():
    assert resolve_candidate_model_base("", None, "qwen3-14b") == "qwen3-14b"


def test_empty_candidate_model_is_not_explicit():
    # The workflow passes ``--candidate-model ""`` when the input is blank;
    # an empty string must behave exactly like the omitted flag, so the
    # champion's base (not the serve fallback) wins.
    base = resolve_candidate_model_base("", _record("llama-31-8b:higher_rank_14b"), "qwen3-14b")
    assert base == "llama-31-8b"


def test_model_ref_without_colon_yields_whole_value():
    assert resolve_candidate_model_base(None, _record("some-model"), "fallback") == "some-model"
