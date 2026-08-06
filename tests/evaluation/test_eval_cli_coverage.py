"""Coverage-completion tests for ``evaluation.cli``.

Runs the Typer app through ``CliRunner`` with every command/mode/flags, plus
direct unit tests for the parse/smoke-gate/proxy helpers. Heavy components
(``EvaluationHarness``, comparison functions) are monkeypatched to return
canned values; the ``__main__`` block is never executed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from evaluation import cli
from evaluation.cli import (
    _dispatch,
    _echo_interrupt,
    _parse_adapter_map,
    _parse_model_pairs,
    _parse_prompts,
    _resolve_proxy_champion,
    _run_best_f2p,
    _smoke_gate,
    _write_baseline,
)
from evaluation.cli import (
    app as cli_app,
)
from evaluation.config import EvalConfig
from evaluation.schema import EvalRun, F2PMetrics


def _cfg(tmp_path: Path, **updates: Any) -> EvalConfig:
    return EvalConfig(
        checkpoint_dir=tmp_path / "ckpt",
        output_dir=tmp_path / "out",
        golden_data_path=str(tmp_path / "golden.jsonl"),
        **updates,
    )


def _metrics(
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    prompt: str = "chat",
    f2p: float = 0.5,
    p2p: float = 0.95,
) -> F2PMetrics:
    return F2PMetrics(
        model_name=model,
        variant=variant,
        prompt_template=prompt,
        total_examples=2,
        successful_patches=1,
        f2p_rate=f2p,
        f2p_count=1,
        p2p_rate=p2p,
        p2p_count=2,
        avg_latency=1.0,
        flaky_test_rate=0.0,
        per_repo_breakdown={},
    )


def _canned_run(config: EvalConfig, run_id: str = "stub-run", f2p: float = 0.5) -> EvalRun:
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=["qwen3-14b:baseline_14b"],
        results=[],
        aggregate=[_metrics(f2p=f2p)],
        status="completed",
        cost_usd=0.0,
    )


class _StubHarness:
    """Stand-in for ``evaluation.harness.EvaluationHarness``."""

    calls: list[tuple[Any, ...]] = []
    f2p = 0.5

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def run_golden(self, pairs, prompt_templates=None, sample=0, run_id=None) -> EvalRun:
        self.calls.append(("golden", pairs, prompt_templates, sample, run_id))
        return _canned_run(self.config, "stub-run", self.f2p)

    def run_swebench_verified(self, pairs, sample=0, run_id=None) -> EvalRun:
        # Record with same structure as run_golden: (split, pairs, prompt_templates, sample, run_id)
        self.calls.append(("swebench", pairs, None, sample, run_id))
        return _canned_run(self.config, "stub-run", self.f2p)

    def run_baseline(self, model=None, sample=0, run_id=None) -> EvalRun:
        self.calls.append(("baseline", model, sample, run_id))
        return _canned_run(self.config, "stub-run", self.f2p)


@pytest.fixture
def stub_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _StubHarness.calls = []
    _StubHarness.f2p = 0.5
    config = _cfg(tmp_path)
    monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
    monkeypatch.setattr(cli, "EvalConfig", lambda: config)
    return config


def _runner() -> CliRunner:
    return CliRunner()


# ── run command: mode resolution / flags ──────────────────────────────────


class TestRunCommand:
    def test_default_run_swebench_sample_50(self, stub_harness):
        # Bare `run` must be cheap: swebench_verified, 50 samples, never the
        # full 2820-example golden set (regression: Aug 2026 $30 accident).
        result = _runner().invoke(cli_app, ["run"])
        assert result.exit_code == 0, result.output
        assert "run_id: stub-run" in result.output
        assert _StubHarness.calls[0][0] == "swebench"
        assert _StubHarness.calls[0][3] == 50

    def test_mode_smoke_resolves_tier(self, stub_harness):
        # first run on a fresh output dir must bootstrap the baseline
        result = _runner().invoke(cli_app, ["run", "--mode", "smoke", "--update-baseline"])
        assert result.exit_code == 0, result.output
        assert _StubHarness.calls[0][0] == "swebench"
        assert _StubHarness.calls[0][3] == stub_harness.tier_sizes["smoke"]
        # smoke gate ran -> baseline written
        assert (stub_harness.output_dir / "smoke_baseline.json").is_file()

    def test_mode_smoke_requires_update_baseline_on_first_run(self, stub_harness):
        # PR-style run with no baseline present must refuse, not mint one.
        result = _runner().invoke(cli_app, ["run", "--mode", "smoke"])
        assert result.exit_code == 1
        assert "SMOKE GATE FAIL" in result.output
        assert "--update-baseline" in result.output
        assert not (stub_harness.output_dir / "smoke_baseline.json").exists()

    @pytest.mark.parametrize("mode", ["dev", "final", "full"])
    def test_mode_tiers(self, stub_harness, mode):
        result = _runner().invoke(cli_app, ["run", "--mode", mode])
        assert result.exit_code == 0, result.output
        call = _StubHarness.calls[0]
        assert call[3] == stub_harness.tier_sizes[mode]
        if mode == "full":
            assert call[0] == "golden"
        else:
            assert call[0] == "swebench"

    def test_unknown_mode_exits_2(self, stub_harness):
        result = _runner().invoke(cli_app, ["run", "--mode", "bogus"])
        assert result.exit_code == 2
        assert "unknown mode" in result.output

    def test_empty_prompts_exits_2(self, stub_harness):
        result = _runner().invoke(cli_app, ["run", "--prompts", ""])
        assert result.exit_code == 2
        assert "at least one prompt template" in result.output

    def test_ci_mode_sets_sample(self, stub_harness):
        result = _runner().invoke(cli_app, ["run", "--ci-mode"])
        assert result.exit_code == 0, result.output
        assert _StubHarness.calls[0][3] == stub_harness.ci_sample_size

    def test_resume_flag_passed(self, stub_harness):
        result = _runner().invoke(cli_app, ["run", "--resume", "run-abc"])
        assert result.exit_code == 0, result.output
        assert _StubHarness.calls[0][4] == "run-abc"

    def test_split_swebench_verified(self, stub_harness):
        result = _runner().invoke(cli_app, ["run", "--split", "swebench_verified"])
        assert result.exit_code == 0, result.output
        assert _StubHarness.calls[0][0] == "swebench"

    def test_backend_local_patches_harness(self, stub_harness, monkeypatch):
        patched: list[str] = []

        def _fake_patch(**kwargs: Any) -> None:
            patched.append(kwargs["ollama_model"])

        monkeypatch.setattr(cli, "_patch_harness_backend", _fake_patch)
        result = _runner().invoke(
            cli_app, ["run", "--backend", "local", "--ollama-model", "llama:7b"]
        )
        assert result.exit_code == 0, result.output
        assert patched == ["llama:7b"]

    def test_keyboard_interrupt_exits_130(self, stub_harness, monkeypatch):
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(_StubHarness, "run_swebench_verified", _boom)
        result = _runner().invoke(cli_app, ["run", "--resume", "run-abc"])
        assert result.exit_code == 130
        assert "interrupted" in result.output
        assert "--resume run-abc" in result.output

    def test_smoke_gate_fails_on_drop(self, stub_harness):
        runner = _runner()
        first = runner.invoke(cli_app, ["run", "--mode", "smoke", "--update-baseline"])
        assert first.exit_code == 0, first.output

        _StubHarness.f2p = 0.3  # 0.5 -> 0.3 = 0.20 drop > 0.05 tolerance
        second = runner.invoke(cli_app, ["run", "--mode", "smoke"])
        assert second.exit_code == 1
        assert "SMOKE GATE FAIL" in second.output


# ── smoke gate internals (new --update-baseline contract) ────────────────


class TestSmokeGate:
    @staticmethod
    def _raw_baseline(config: EvalConfig, content: str) -> Path:
        path = config.output_dir / "smoke_baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_no_aggregate_exits(self, stub_harness):
        run = _canned_run(stub_harness)
        run.aggregate = []
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(run, stub_harness, update_baseline=True)
        assert exc.value.exit_code == 1

    def test_corrupt_baseline_exits(self, stub_harness):
        self._raw_baseline(stub_harness, "{not json")
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(_canned_run(stub_harness), stub_harness)
        assert exc.value.exit_code == 1

    def test_corrupt_rates_exits(self, stub_harness):
        self._raw_baseline(
            stub_harness,
            json.dumps({"rates": {f"{stub_harness.dataset_run_id}": "nan!"}}),
        )
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(_canned_run(stub_harness), stub_harness)
        assert exc.value.exit_code == 1

    def test_legacy_flat_baseline_read_only_pass(self, stub_harness, capsys):
        self._raw_baseline(stub_harness, json.dumps({"qwen3-14b:baseline_14b:chat": 0.4}))
        # flat legacy file -> normalized run_id "" -> matches default dataset_run_id
        _smoke_gate(_canned_run(stub_harness), stub_harness)
        assert "baseline unchanged (read-only)" in capsys.readouterr().out

    def test_legacy_flat_baseline_update_merges(self, stub_harness):
        path = self._raw_baseline(stub_harness, json.dumps({"qwen3-14b:baseline_14b:chat": 0.4}))
        _smoke_gate(_canned_run(stub_harness), stub_harness, update_baseline=True)
        baseline = json.loads(path.read_text())
        assert baseline["dataset_run_id"] == stub_harness.dataset_run_id
        assert baseline["rates"]["qwen3-14b:baseline_14b:chat"] == max(
            0.5, 0.4, stub_harness.min_f2p_threshold
        )

    def test_missing_variant_from_run_exits(self, stub_harness):
        path = stub_harness.output_dir / "smoke_baseline.json"
        _write_baseline(
            path,
            stub_harness.dataset_run_id,
            {
                "qwen3-14b:baseline_14b:chat": 0.5,
                "qwen3-14b:lora_x:chat": 0.4,
            },
        )
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(_canned_run(stub_harness), stub_harness)
        assert exc.value.exit_code == 1

    def test_below_floor_exits(self, stub_harness):
        path = stub_harness.output_dir / "smoke_baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_baseline(
            path,
            stub_harness.dataset_run_id,
            {"qwen3-14b:baseline_14b:chat": 0.5},
        )
        cfg = stub_harness.model_copy(update={"min_f2p_threshold": 0.9})
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(_canned_run(cfg), cfg)
        assert exc.value.exit_code == 1

    def test_update_rewrites_baseline(self, stub_harness):
        runner = _runner()
        first = runner.invoke(cli_app, ["run", "--mode", "smoke", "--update-baseline"])
        assert first.exit_code == 0, first.output
        second = runner.invoke(cli_app, ["run", "--mode", "smoke", "--update-baseline"])
        assert second.exit_code == 0, second.output
        assert "baseline updated" in second.output

    def test_update_merge_adds_new_keys(self, stub_harness):
        path = stub_harness.output_dir / "smoke_baseline.json"
        _write_baseline(path, stub_harness.dataset_run_id, {"qwen3-14b:baseline_14b:chat": 0.3})
        run = _canned_run(stub_harness)
        run.aggregate = [
            _metrics(),  # in baseline
            _metrics(variant="lora_x", f2p=0.8),  # brand-new key
        ]
        _smoke_gate(run, stub_harness, update_baseline=True)
        baseline = json.loads(path.read_text())
        assert baseline["rates"]["qwen3-14b:baseline_14b:chat"] == max(
            0.5, 0.3, stub_harness.min_f2p_threshold
        )
        assert baseline["rates"]["qwen3-14b:lora_x:chat"] == max(
            0.8, 0.0, stub_harness.min_f2p_threshold
        )


# ── other run commands ────────────────────────────────────────────────────


class TestOtherRunCommands:
    def test_run_golden_command(self, stub_harness):
        result = _runner().invoke(
            cli_app,
            ["run-golden", "--models", "qwen3-14b:baseline_14b", "--resume", "run-1"],
        )
        assert result.exit_code == 0, result.output
        assert _StubHarness.calls[0][0] == "golden"
        assert _StubHarness.calls[0][3] == 50

    def test_run_golden_keyboard_interrupt(self, stub_harness, monkeypatch):
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(_StubHarness, "run_golden", _boom)
        result = _runner().invoke(cli_app, ["run-golden"])
        assert result.exit_code == 130
        assert "--resume <run_id>" in result.output

    def test_run_swebench_command(self, stub_harness):
        result = _runner().invoke(cli_app, ["run-swebench", "--sample", "5"])
        assert result.exit_code == 0, result.output
        assert _StubHarness.calls[0][0] == "swebench"
        assert _StubHarness.calls[0][3] == 5

    def test_run_swebench_keyboard_interrupt(self, stub_harness, monkeypatch):
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(_StubHarness, "run_swebench_verified", _boom)
        result = _runner().invoke(cli_app, ["run-swebench"])
        assert result.exit_code == 130

    def test_run_baseline_command(self, stub_harness):
        result = _runner().invoke(cli_app, ["run-baseline", "--sample", "7"])
        assert result.exit_code == 0, result.output
        call = _StubHarness.calls[0]
        assert call[0] == "baseline"
        assert call[1] == "Qwen/Qwen3-14B"
        assert call[2] == 7

    def test_run_baseline_keyboard_interrupt(self, stub_harness, monkeypatch):
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(_StubHarness, "run_baseline", _boom)
        result = _runner().invoke(cli_app, ["run-baseline"])
        assert result.exit_code == 130

    def test_run_prompt_ab_with_templates(self, stub_harness, monkeypatch, tmp_path):
        config = stub_harness
        canned = _canned_run(config, "ab-run")

        def _fake_ab(cfg, **kwargs):
            assert kwargs["model"] == "qwen3-14b"
            assert kwargs["templates"] == ["chat", "cot"]
            return canned

        monkeypatch.setattr("evaluation.prompt_ab_test.run_prompt_ab_test", _fake_ab)
        result = _runner().invoke(
            cli_app, ["run-prompt-ab", "--templates", "chat,cot", "--sample", "3"]
        )
        assert result.exit_code == 0, result.output
        assert "run_id: ab-run" in result.output

    def test_run_prompt_ab_defaults_to_all_templates(self, stub_harness, monkeypatch):
        config = stub_harness
        canned = _canned_run(config, "ab-run")
        captured: dict[str, Any] = {}

        def _fake_ab(cfg, **kwargs):
            captured.update(kwargs)
            return canned

        monkeypatch.setattr("evaluation.prompt_ab_test.run_prompt_ab_test", _fake_ab)

        class _Loader:
            @property
            def available_templates(self) -> list[str]:
                return ["t-one", "t-two"]

        monkeypatch.setattr("training.prompt_loader.PromptLoader", _Loader)
        result = _runner().invoke(cli_app, ["run-prompt-ab"])
        assert result.exit_code == 0, result.output
        assert captured["templates"] == ["t-one", "t-two"]
        assert captured["sample"] == 200

    def test_run_prompt_ab_keyboard_interrupt(self, stub_harness, monkeypatch):
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr("evaluation.prompt_ab_test.run_prompt_ab_test", _boom)
        result = _runner().invoke(cli_app, ["run-prompt-ab"])
        assert result.exit_code == 130


# ── _dispatch ─────────────────────────────────────────────────────────────


class TestDispatch:
    def test_swebench_split(self, monkeypatch, tmp_path):
        monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
        _StubHarness.calls = []
        run = _dispatch("swebench_verified", [("m", "v")], ["chat"], 0, None, _cfg(tmp_path))
        assert run.run_id == "stub-run"
        assert _StubHarness.calls[0][0] == "swebench"

    def test_golden_split(self, monkeypatch, tmp_path):
        monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
        _StubHarness.calls = []
        run = _dispatch("golden", [("m", "v")], ["chat"], 0, "resume-1", _cfg(tmp_path))
        assert run.run_id == "stub-run"
        assert _StubHarness.calls[0][0] == "golden"
        assert _StubHarness.calls[0][4] == "resume-1"

    def test_golden_sample_zero_capped(self, monkeypatch, tmp_path):
        # sample=0 on golden must never mean "all 2820" — capped at tier_sizes["full"].
        monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
        _StubHarness.calls = []
        cfg = _cfg(tmp_path)
        _dispatch("golden", [("m", "v")], ["chat"], 0, None, cfg)
        assert _StubHarness.calls[0][3] == cfg.tier_sizes["full"]

    def test_golden_explicit_sample_kept(self, monkeypatch, tmp_path):
        monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
        _StubHarness.calls = []
        _dispatch("golden", [("m", "v")], ["chat"], 100, None, _cfg(tmp_path))
        assert _StubHarness.calls[0][3] == 100

    def test_unknown_split(self, monkeypatch, tmp_path):
        monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
        with pytest.raises(typer.BadParameter, match="unknown split"):
            _dispatch("bogus", [], ["chat"], 0, None, _cfg(tmp_path))

    def test_local_backend_calls_patch(self, monkeypatch, tmp_path):
        patched: list[str] = []

        def _fake_patch(**kwargs: Any) -> None:
            patched.append(kwargs["ollama_url"])

        monkeypatch.setattr(cli, "_patch_harness_backend", _fake_patch)
        monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
        _StubHarness.calls = []
        _dispatch(
            "golden",
            [("m", "v")],
            ["chat"],
            0,
            None,
            _cfg(tmp_path),
            backend="local",
            ollama_model="m",
            ollama_url="http://x",
        )
        assert patched == ["http://x"]


# ── parse helpers ─────────────────────────────────────────────────────────


class TestParseHelpers:
    def test_parse_model_pairs_valid(self):
        assert _parse_model_pairs("a:b,c:d") == [("a", "b"), ("c", "d")]

    def test_parse_model_pairs_skips_blank(self):
        assert _parse_model_pairs("a:b,, c:d ,") == [("a", "b"), ("c", "d")]
        assert _parse_model_pairs("") == []

    def test_parse_model_pairs_bad(self):
        with pytest.raises(typer.BadParameter, match="model:variant"):
            _parse_model_pairs("qwen3-14b")
        with pytest.raises(typer.BadParameter, match="model:variant"):
            _parse_model_pairs(":variant")

    def test_parse_prompts(self):
        assert _parse_prompts("chat, cot ") == ["chat", "cot"]
        assert _parse_prompts("") == []
        assert _parse_prompts(" , ") == []

    def test_parse_adapter_map_valid(self):
        assert _parse_adapter_map("a=b,c=d") == {"a": "b", "c": "d"}
        assert _parse_adapter_map("") == {}
        assert _parse_adapter_map(" , ") == {}

    def test_parse_adapter_map_bad(self):
        with pytest.raises(typer.BadParameter, match="variant=adapter"):
            _parse_adapter_map("a")
        with pytest.raises(typer.BadParameter, match="variant=adapter"):
            _parse_adapter_map("=b")


class TestResolveProxyChampion:
    def test_proxy_disabled_returns_none(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        assert _resolve_proxy_champion(False, None, None) is None

    def test_proxy_default_fallback(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        assert _resolve_proxy_champion(True, None, None) == "baseline_14b"

    def test_missing_counterpart_raises(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        with pytest.raises(typer.BadParameter, match="must be provided together"):
            _resolve_proxy_champion(True, "golden.jsonl", None)
        with pytest.raises(typer.BadParameter, match="must be provided together"):
            _resolve_proxy_champion(True, None, "a=b")

    def test_both_given_uses_proxy_scorer(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        calls: list[tuple[Path, dict[str, str]]] = []

        def _fake_proxy(golden_path: Path, adapter_map: dict[str, str]) -> str:
            calls.append((golden_path, adapter_map))
            return "proxy-champ"

        monkeypatch.setattr(cli, "proxy_champion_from_f2p_proxy", _fake_proxy)
        got = _resolve_proxy_champion(True, "data/golden.jsonl", "a=b,c=d")
        assert got == "proxy-champ"
        assert calls == [(Path("data/golden.jsonl"), {"a": "b", "c": "d"})]


class TestRunBestF2p:
    def test_max_across_aggregate(self, tmp_path):
        config = _cfg(tmp_path)
        run = _canned_run(config, "r")
        run.aggregate = [_metrics(f2p=0.2), _metrics(f2p=0.9)]
        assert _run_best_f2p(run) == pytest.approx(0.9)

    def test_empty_aggregate_zero(self, tmp_path):
        config = _cfg(tmp_path)
        run = _canned_run(config, "r")
        run.aggregate = []
        assert _run_best_f2p(run) == 0.0


# ── _smoke_gate ───────────────────────────────────────────────────────────


class TestSmokeGateDirect:
    def test_empty_aggregate_fails(self, tmp_path):
        config = _cfg(tmp_path)
        run = _canned_run(config, "r")
        run.aggregate = []
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(run, config)
        assert exc.value.exit_code == 1

    def test_missing_variant_fails(self, tmp_path):
        config = _cfg(tmp_path)
        (config.output_dir).mkdir(parents=True, exist_ok=True)
        # Baseline has a variant that won't be in the current run
        (config.output_dir / "smoke_baseline.json").write_text(
            json.dumps({"qwen3-14b:variant_a:chat": 0.5, "qwen3-14b:variant_b:chat": 0.6})
        )
        # Run only has variant_a, missing variant_b
        run = _canned_run(config, "r", f2p=0.5)
        run.aggregate = [_metrics(model="qwen3-14b", variant="variant_a", f2p=0.5)]
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(run, config)
        assert exc.value.exit_code == 1


# ── compare command ───────────────────────────────────────────────────────


def _compare_runs(config: EvalConfig, cost_a: float = 4.0, cost_b: float = 2.0) -> list[EvalRun]:
    a = _canned_run(config, "run_a")
    b = _canned_run(config, "run_b")
    a.cost_usd = cost_a
    b.cost_usd = cost_b
    return [a, b]


class TestCompareCommand:
    def test_no_runs_exits_1(self, monkeypatch, tmp_path):
        config = _cfg(tmp_path)
        monkeypatch.setattr(cli, "EvalConfig", lambda: config)
        monkeypatch.setattr(cli, "load_all_eval_runs", lambda ids, cfg: [])
        monkeypatch.setattr(cli, "extract_model_metrics", lambda runs: {})

        result = _runner().invoke(cli_app, ["compare", "--run_ids", "missing"])
        assert result.exit_code == 1
        assert "no runs loaded" in result.output

    def test_champion_promoted_echoed(self, monkeypatch, tmp_path):
        config = _cfg(tmp_path)
        runs = _compare_runs(config)
        metrics = {"qwen3-14b:baseline_14b": _metrics()}
        monkeypatch.setattr(cli, "EvalConfig", lambda: config)
        monkeypatch.setattr(cli, "load_all_eval_runs", lambda ids, cfg: runs)
        monkeypatch.setattr(cli, "extract_model_metrics", lambda runs: metrics)
        monkeypatch.setattr(
            cli, "compare_and_report", lambda m, proxy_champion=None: "COMPARE_TABLE"
        )
        monkeypatch.setattr(
            cli,
            "revalidate_champion",
            lambda m, pc, f, p: ("qwen3-14b:baseline_14b", _metrics()),
        )
        monkeypatch.setattr(
            cli,
            "promote_champion_to_registry",
            lambda key, cfg: "W&B Registry: champion alias -> model-qwen3-14b-baseline_14b",
        )
        monkeypatch.setattr(cli, "paired_significance", lambda a, b: "SIG_BLOCK")

        result = _runner().invoke(cli_app, ["compare", "--run_ids", "run_a,run_b"])

        assert result.exit_code == 0, result.output
        assert "COMPARE_TABLE" in result.output
        assert "W&B Registry: champion alias" in result.output
        assert "SIG_BLOCK" in result.output
        assert "est. total cost across 2 run(s): $6.00" in result.output

    def test_no_champion_no_echo(self, monkeypatch, tmp_path):
        config = _cfg(tmp_path)
        runs = _compare_runs(config, cost_a=0.0, cost_b=0.0)
        metrics = {"qwen3-14b:baseline_14b": _metrics(f2p=0.05, p2p=0.5)}
        monkeypatch.setattr(cli, "EvalConfig", lambda: config)
        monkeypatch.setattr(cli, "load_all_eval_runs", lambda ids, cfg: runs)
        monkeypatch.setattr(cli, "extract_model_metrics", lambda runs: metrics)
        monkeypatch.setattr(cli, "compare_and_report", lambda m, proxy_champion=None: "TABLE2")
        monkeypatch.setattr(cli, "revalidate_champion", lambda m, pc, f, p: None)
        promote_calls: list[tuple[str, EvalConfig]] = []
        monkeypatch.setattr(
            cli,
            "promote_champion_to_registry",
            lambda key, cfg: promote_calls.append((key, cfg)) or None,
        )
        monkeypatch.setattr(cli, "paired_significance", lambda a, b: "SIG")

        result = _runner().invoke(cli_app, ["compare", "--run_ids", "run_a,run_b"])

        assert result.exit_code == 0, result.output
        assert "W&B Registry" not in result.output
        assert promote_calls == []
        # zero total cost -> no cost echo
        assert "est. total cost" not in result.output

    def test_single_run_skips_significance(self, monkeypatch, tmp_path):
        config = _cfg(tmp_path)
        runs = [_canned_run(config, "run_only")]
        monkeypatch.setattr(cli, "EvalConfig", lambda: config)
        monkeypatch.setattr(cli, "load_all_eval_runs", lambda ids, cfg: runs)
        monkeypatch.setattr(cli, "extract_model_metrics", lambda runs: {"k": _metrics()})
        monkeypatch.setattr(cli, "compare_and_report", lambda m, proxy_champion=None: "TABLE3")
        monkeypatch.setattr(cli, "revalidate_champion", lambda m, pc, f, p: None)
        sig_calls: list[Any] = []
        monkeypatch.setattr(
            cli, "paired_significance", lambda a, b: sig_calls.append((a, b)) or "SIG"
        )

        result = _runner().invoke(cli_app, ["compare", "--run_ids", "run_only"])
        assert result.exit_code == 0, result.output
        assert sig_calls == []

    def test_proxy_champion_via_golden(self, monkeypatch, tmp_path):
        config = _cfg(tmp_path)
        runs = [_canned_run(config, "run_x")]
        metrics = {"qwen3-14b:baseline_14b": _metrics()}
        monkeypatch.setattr(cli, "EvalConfig", lambda: config)
        monkeypatch.setattr(cli, "load_all_eval_runs", lambda ids, cfg: runs)
        monkeypatch.setattr(cli, "extract_model_metrics", lambda runs: metrics)
        monkeypatch.setattr(cli, "compare_and_report", lambda m, proxy_champion=None: "TABLE4")
        seen: list[str] = []

        def _fake_revalidate(metrics, proxy_champion, min_f2p, min_p2p):
            seen.append(proxy_champion)

        monkeypatch.setattr(cli, "revalidate_champion", _fake_revalidate)
        monkeypatch.setattr(cli, "proxy_champion_from_f2p_proxy", lambda gp, am: "proxy-champ")

        result = _runner().invoke(
            cli_app,
            [
                "compare",
                "--run_ids",
                "run_x",
                "--golden-path",
                "data/golden.jsonl",
                "--variant-adapter-map",
                "a=b",
            ],
        )
        assert result.exit_code == 0, result.output
        assert seen == ["proxy-champ"]

    def test_golden_without_adapter_exits_2(self, stub_harness, monkeypatch):
        # Provide a run with metrics so CLI doesn't exit early, then test golden/adapter constraint
        monkeypatch.setattr(cli, "load_all_eval_runs", lambda ids, cfg: [_canned_run(stub_harness)])
        monkeypatch.setattr(
            cli, "extract_model_metrics", lambda runs: {"qwen3-14b:baseline_14b": _metrics()}
        )
        result = _runner().invoke(
            cli_app, ["compare", "--run_ids", "run_x", "--golden-path", "data/golden.jsonl"]
        )
        assert result.exit_code == 2, result.output
        # typer wraps long error text across lines, so assert the stable fragment
        assert "must be provided" in result.output


# ── app-level metadata ────────────────────────────────────────────────────


class TestAppMetadata:
    def test_no_args_is_help(self):
        result = _runner().invoke(cli_app, [])
        # Typer returns exit code 2 when showing help with no_args_is_help=True
        assert result.exit_code == 2
        assert "Usage:" in result.output

    def test_help_flag(self):
        result = _runner().invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "compare" in result.output


def test_echo_interrupt_message() -> None:
    with pytest.raises(typer.Exit) as exc:
        _echo_interrupt("run-42")
    assert exc.value.exit_code == 130
