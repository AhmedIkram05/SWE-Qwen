"""Gap-filling tests for ``evaluation.comparison``.

Covers the W&B promotion/load paths, empty/missing-run branches, blank
``instance_id`` handling, and the JSONL parse fallbacks that the existing
tests (``test_eval_integration.py`` / ``test_eval_unit.py``) leave on the
table. No real W&B / Modal / network calls — `wandb` is faked in
``sys.modules`` when a function does a lazy ``import wandb``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from evaluation import comparison
from evaluation.comparison import (
    _as_run,
    _clear_champion_alias,
    _load_wandb_run,
    _parse_run_file,
    extract_model_metrics,
    load_all_eval_runs,
    paired_significance,
    promote_champion_to_registry,
    revalidate_champion,
    revalidate_proxy_champion,
)
from evaluation.config import EvalConfig
from evaluation.metrics import aggregate_metrics
from evaluation.schema import EvalResult, EvalRun, F2PMetrics, PatchApplicationResult


def _cfg(tmp_path: Path, **updates: Any) -> EvalConfig:
    return EvalConfig(
        checkpoint_dir=tmp_path / "ckpt",
        output_dir=tmp_path / "out",
        golden_data_path=str(tmp_path / "golden.jsonl"),
        **updates,
    )


def _result(
    *,
    instance_id: str = "inst-1",
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    f2p: float = 0.5,
    p2p: float = 0.95,
    timestamp: datetime | None = None,
) -> EvalResult:
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
        timestamp=timestamp or datetime.now(UTC),
    )


def _canned_run(tmp_path: Path, results: list[EvalResult] | None = None) -> EvalRun:
    results = results if results is not None else [_result(f2p=0.5)]
    config = _cfg(tmp_path)
    return EvalRun(
        run_id="stub-run",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=["qwen3-14b:baseline_14b"],
        results=results,
        aggregate=[aggregate_metrics(results)],
        status="completed",
    )


# ── load_all_eval_runs / extract_model_metrics ───────────────────────────


def test_load_all_eval_runs_skips_missing(tmp_path, caplog):
    config = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        runs = load_all_eval_runs(["ghost-run"], config)
    assert runs == []
    assert "ghost-run not found locally or in W&B" in caplog.text


def test_extract_model_metrics_keeps_blank_instance_ids(tmp_path):
    # instance_id = "" is falsy → the dedup guard is skipped entirely.
    run = _canned_run(
        tmp_path,
        results=[_result(instance_id="", f2p=0.6), _result(instance_id="", f2p=0.4)],
    )
    merged = extract_model_metrics([run])
    assert len(merged) == 1
    assert merged["qwen3-14b:baseline_14b"].total_examples == 2


def test_extract_model_metrics_dedups_shared_instances(tmp_path):
    results = [
        _result(instance_id="inst-1", model="qwen3-14b", f2p=0.5),
        _result(instance_id="inst-1", model="qwen3-14b", f2p=0.9),  # last wins
        _result(instance_id="inst-2", model="qwen3-14b", f2p=0.7),
    ]
    merged = extract_model_metrics([_canned_run(tmp_path, results=results)])
    assert merged["qwen3-14b:baseline_14b"].total_examples == 2


# ── revalidate_champion / revalidate_proxy_champion ──────────────────────


def test_revalidate_champion_no_candidate_returns_none(tmp_path):
    metrics = cast(
        "dict[str, F2PMetrics]",
        {
            "m:a": SimpleNamespace(p2p_rate=0.5, f2p_rate=0.5),  # p2p below ceiling
            "m:b": SimpleNamespace(p2p_rate=0.95, f2p_rate=0.05),  # f2p below floor
        },
    )
    assert revalidate_champion(metrics, "a", min_f2p=0.15, min_p2p=0.90) is None


def test_revalidate_proxy_champion_uses_p4_proxy(tmp_path, mocker):
    run = _canned_run(tmp_path)
    mocker.patch.object(comparison, "load_all_eval_runs", return_value=[run])
    mocker.patch.object(comparison, "proxy_champion_from_f2p_proxy", return_value="baseline_14b")
    got = revalidate_proxy_champion(
        _cfg(tmp_path),
        ["stub-run"],
        golden_path=Path("golden.jsonl"),
        variant_adapter_map={"baseline_14b": "adapter-ref"},
    )
    assert got == ("qwen3-14b:baseline_14b", run.aggregate[0])


def test_revalidate_proxy_champion_falls_back_when_nothing_metrics(tmp_path, mocker, caplog):
    mocker.patch.object(comparison, "load_all_eval_runs", return_value=[])
    with caplog.at_level("WARNING"):
        assert revalidate_proxy_champion(_cfg(tmp_path), ["stub-run"]) is None
    assert "nothing to revalidate" in caplog.text


# ── paired_significance ───────────────────────────────────────────────────


def test_paired_significance_no_shared_variants(tmp_path):
    a = _canned_run(tmp_path, results=[_result(instance_id="")])
    b = _canned_run(
        tmp_path,
        results=[_result(instance_id="", variant="lora_x", f2p=0.3)],
    )
    out = paired_significance(a, b)
    assert "no variant evaluated in both runs" in out


# ── _clear_champion_alias ─────────────────────────────────────────────────


def test_clear_champion_alias_removes_and_saves(mocker, caplog, tmp_path):
    member = SimpleNamespace(aliases=["champion", "latest"], name="art-1", save=mocker.Mock())
    api = SimpleNamespace(
        artifact_collection=lambda name: SimpleNamespace(
            artifacts=[member, SimpleNamespace(aliases=["latest"], name="art-2")]
        )
    )
    _clear_champion_alias(api, _cfg(tmp_path))  # config unused on success path
    assert member.aliases == ["latest"]
    member.save.assert_called_once()


def test_clear_champion_alias_failure_is_swallowed(mocker, caplog, tmp_path):
    api = SimpleNamespace(artifact_collection=mocker.Mock(side_effect=RuntimeError("boom")))
    with caplog.at_level("WARNING"):
        _clear_champion_alias(api, _cfg(tmp_path))  # must not raise
    assert "failed to clear previous W&B champion alias" in caplog.text


# ── promote_champion_to_registry ──────────────────────────────────────────


class _FakeWandb:
    """Stub wandb whose Api/init behaviour is configured per-test."""

    def __init__(
        self,
        *,
        artifact: object | None = None,
        artifact_raises: Exception | None = None,
        run: object | None = None,
        init_raises: Exception | None = None,
    ) -> None:
        self._artifact = artifact
        self._artifact_raises = artifact_raises
        self._run = run
        self._init_raises = init_raises
        self.artifact_calls: list[str] = []

    class Api:
        def __init__(self, _inner, timeout: int) -> None:
            self._inner = _inner

        def artifact(self, qualified: str):
            self._inner.artifact_calls.append(qualified)
            if self._inner._artifact_raises is not None:
                raise self._inner._artifact_raises
            return self._inner._artifact

    def init(self, **kwargs: object):
        if self._init_raises is not None:
            raise self._init_raises
        return self._run


def _install_fake_wandb(mocker, fake: _FakeWandb) -> None:
    module = SimpleNamespace(Api=lambda timeout: fake.Api(fake, timeout), init=fake.init)
    mocker.patch.dict(sys.modules, {"wandb": module})


def test_promote_champion_success(mocker, tmp_path):
    run = mocker.Mock()
    artifact = SimpleNamespace(link_artifact=mocker.Mock(), aliases=[], download=mocker.Mock())
    fake = _FakeWandb(artifact=artifact, run=run)
    _install_fake_wandb(mocker, fake)
    config = _cfg(tmp_path, lora_artifact_pattern="lora-swe-qwen-{variant}")
    out = promote_champion_to_registry("qwen3-14b:baseline_14b", config)
    assert out == "W&B Registry: champion alias -> lora-swe-qwen-baseline_14b"
    assert fake.artifact_calls == [
        f"{config.wandb_entity}/{config.wandb_project}/lora-swe-qwen-baseline_14b:latest"
    ]
    run.link_artifact.assert_called_once_with(artifact, "eval-champion", aliases=["champion"])
    run.finish.assert_called_once()


def test_promote_champion_artifact_missing(mocker, tmp_path, caplog):
    _install_fake_wandb(mocker, _FakeWandb(artifact_raises=RuntimeError("no artifact")))
    with caplog.at_level("WARNING"):
        assert promote_champion_to_registry("m:v", _cfg(tmp_path)) is None
    assert "not found in W&B — skipping promotion" in caplog.text


def test_promote_champion_init_failure(mocker, tmp_path, caplog):
    _install_fake_wandb(mocker, _FakeWandb(artifact=object(), init_raises=RuntimeError("auth")))
    with caplog.at_level("WARNING"):
        assert promote_champion_to_registry("m:v", _cfg(tmp_path)) is None
    assert "W&B champion promotion failed" in caplog.text


def test_promote_champion_empty_key_returns_none(tmp_path):
    assert promote_champion_to_registry("", _cfg(tmp_path)) is None


def test_promote_champion_no_wandb(mocker, tmp_path, caplog):
    mocker.patch.dict(sys.modules, {"wandb": None})  # forces ImportError on import
    with caplog.at_level("WARNING"):
        assert promote_champion_to_registry("m:v", _cfg(tmp_path)) is None
    assert "wandb not installed" in caplog.text


# ── _load_wandb_run / _parse_run_file / _as_run ───────────────────────────


def test_load_wandb_run_success_list(tmp_path, mocker):
    config = _cfg(tmp_path)
    results = [_result(instance_id="a", f2p=0.5, timestamp=datetime.now(UTC))]
    write_me = "\n".join(r.model_dump_json() for r in results)
    download_dir = tmp_path / "dl"
    download_dir.mkdir()
    (download_dir / "01.jsonl").write_text(write_me)
    artifact = SimpleNamespace(download=lambda: str(download_dir))
    _install_fake_wandb(mocker, _FakeWandb(artifact=artifact))
    run = _load_wandb_run("run-1", config)
    assert run is not None
    assert run.run_id == "run-1"
    assert run.results[0].instance_id == "a"
    assert run.aggregate[0].f2p_rate == 0.5


def test_load_wandb_run_success_single_run_dump(tmp_path, mocker):
    config = _cfg(tmp_path)
    canned = _canned_run(tmp_path)
    download_dir = tmp_path / "dl2"
    download_dir.mkdir()
    (download_dir / "run.json").write_text(canned.model_dump_json())
    artifact = SimpleNamespace(download=lambda: str(download_dir))
    _install_fake_wandb(mocker, _FakeWandb(artifact=artifact))
    got = _load_wandb_run("run-1", config)
    assert got is not None and got.run_id == "stub-run"  # raw dump returned as-is


def test_load_wandb_run_artifact_unavailable(mocker, tmp_path, caplog):
    _install_fake_wandb(mocker, _FakeWandb(artifact_raises=RuntimeError("no")))
    with caplog.at_level("WARNING"):
        assert _load_wandb_run("run-1", _cfg(tmp_path)) is None
    assert "W&B artifact eval-results-run-1 unavailable" in caplog.text


def test_load_wandb_run_empty_artifact(mocker, tmp_path, caplog):
    download_dir = tmp_path / "dl3"
    download_dir.mkdir()
    _install_fake_wandb(
        mocker, _FakeWandb(artifact=SimpleNamespace(download=lambda: str(download_dir)))
    )
    with caplog.at_level("WARNING"):
        assert _load_wandb_run("run-1", _cfg(tmp_path)) is None
    assert "contained no results" in caplog.text


def test_parse_run_file_jsonl_fallback(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        "\n".join(_result(instance_id=f"i-{n}", f2p=n / 10).model_dump_json() for n in (1, 2))
    )
    parsed = _parse_run_file(path)
    assert isinstance(parsed, list) and len(parsed) == 2


def test_parse_run_file_jsonl_first_line_run(tmp_path, mocker):
    path = tmp_path / "r.jsonl"
    canned = _canned_run(tmp_path)
    path.write_text(canned.model_dump_json() + "\n" + _result(instance_id="solo").model_dump_json())
    parsed = _parse_run_file(path)
    assert isinstance(parsed, EvalRun) and parsed.run_id == "stub-run"


def test_parse_run_file_bad_line_returns_none(tmp_path, caplog):
    path = tmp_path / "r.jsonl"
    path.write_text('{"not": }')
    with caplog.at_level("WARNING"):
        assert _parse_run_file(path) is None
    assert "run file" in caplog.text


def test_parse_run_file_unreadable(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert _parse_run_file(tmp_path / "missing.json") is None
    assert "unreadable" in caplog.text


def test_parse_run_file_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("  \n")
    assert _parse_run_file(path) is None


def test_parse_run_file_single_result_dict(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(_result(instance_id="solo").model_dump_json())
    parsed = _parse_run_file(path)
    assert isinstance(parsed, list) and parsed[0].instance_id == "solo"


def test_parse_run_file_invalid_schema(tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"results": [], "nope": True}))
    with caplog.at_level("WARNING"):
        assert _parse_run_file(path) is None
    assert "failed schema validation" in caplog.text


def test_as_run_aggregates_by_group(tmp_path):
    config = _cfg(tmp_path)
    results = [
        _result(instance_id="a", variant="baseline_14b", f2p=0.4, timestamp=datetime.now(UTC)),
        _result(instance_id="b", variant="baseline_14b", f2p=0.8, timestamp=datetime.now(UTC)),
        _result(instance_id="c", variant="lora_x", f2p=0.6, timestamp=datetime.now(UTC)),
    ]
    run = _as_run("grp-run", config, results)
    assert run.models_evaluated == ["qwen3-14b:baseline_14b", "qwen3-14b:lora_x"]
    grouped = {g.variant: g for g in run.aggregate}
    assert len(grouped) == 2
    assert grouped["baseline_14b"].f2p_rate == pytest.approx(0.6)
    assert grouped["lora_x"].f2p_rate == pytest.approx(0.6)
