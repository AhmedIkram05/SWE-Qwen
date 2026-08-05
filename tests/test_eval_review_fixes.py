"""Review-round-2 regression tests: smoke gate, cost model, paired
significance, instance dedupe, swebench image naming, reset failure."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from evaluation.config import EvalConfig
from evaluation.schema import EvalInput, EvalResult, EvalRun, PatchApplicationResult

# ── helpers ─────────────────────────────────────────────────────────────────


def _cfg(tmp_path: Path) -> EvalConfig:
    return EvalConfig().model_copy(update={"output_dir": tmp_path})


def _result(
    instance_id: str,
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    f2p: float = 1.0,
    latency: float = 0.0,
) -> EvalResult:
    return EvalResult(
        instance_id=instance_id,
        repo="django/django",
        model_name=model,
        variant=variant,
        prompt_template="chat",
        generated_patch="",
        patch_application=PatchApplicationResult(success=True, method_used="git_apply", error=None),
        tests_before=[],
        tests_after=[],
        f2p=f2p,
        p2p=1.0,
        latency_seconds=latency,
        timestamp=datetime.now(UTC),
        error=None,
    )


def _run(run_id: str, results: list[EvalResult]) -> EvalRun:

    from evaluation.comparison import extract_model_metrics

    metrics = extract_model_metrics(
        [
            EvalRun(
                run_id=run_id,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                config=EvalConfig(),
                models_evaluated=[],
                results=results,
                aggregate=[],
                status="completed",
            )
        ]
    )
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=EvalConfig(),
        models_evaluated=sorted({f"{r.model_name}:{r.variant}" for r in results}),
        results=results,
        aggregate=list(metrics.values()),
        status="completed",
    )


# ── _smoke_gate ──────────────────────────────────────────────────────────────


def _smoke_gate(run: EvalRun, cfg: EvalConfig) -> None:
    from evaluation.cli import _smoke_gate

    return _smoke_gate(run, cfg)


class TestSmokeGate:
    def test_first_run_writes_baseline(self, tmp_path):
        cfg = _cfg(tmp_path)
        run = _run("smoke-1", [_result("inst-a", f2p=0.5), _result("inst-b", f2p=1.0)])
        _smoke_gate(run, cfg)
        baseline = json.loads((cfg.output_dir / "smoke_baseline.json").read_text())
        assert baseline["qwen3-14b:baseline_14b:chat"] == pytest.approx(0.75)

    def test_drop_beyond_tolerance_exits_1(self, tmp_path):
        cfg = _cfg(tmp_path)
        (cfg.output_dir / "smoke_baseline.json").write_text(
            json.dumps({"qwen3-14b:baseline_14b:chat": 0.80})
        )
        run = _run("smoke-2", [_result("inst-a", f2p=0.5), _result("inst-b", f2p=0.5)])
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(run, cfg)
        assert exc.value.exit_code == 1

    def test_within_tolerance_updates_baseline(self, tmp_path):
        cfg = _cfg(tmp_path)
        (cfg.output_dir / "smoke_baseline.json").write_text(
            json.dumps({"qwen3-14b:baseline_14b:chat": 0.80})
        )
        run = _run("smoke-3", [_result("inst-a", f2p=1.0), _result("inst-b", f2p=1.0)])
        _smoke_gate(run, cfg)
        baseline = json.loads((cfg.output_dir / "smoke_baseline.json").read_text())
        assert baseline["qwen3-14b:baseline_14b:chat"] == pytest.approx(1.0)

    def test_corrupt_baseline_exits_1(self, tmp_path):
        cfg = _cfg(tmp_path)
        (cfg.output_dir / "smoke_baseline.json").write_text("{not json")
        run = _run("smoke-4", [_result("inst-a")])
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(run, cfg)
        assert exc.value.exit_code == 1

    def test_empty_aggregate_exits_1(self, tmp_path):
        cfg = _cfg(tmp_path)
        empty = _run("smoke-5", [])
        empty.aggregate = []
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(empty, cfg)
        assert exc.value.exit_code == 1
        assert not (cfg.output_dir / "smoke_baseline.json").exists()

    def test_missing_variant_from_run_exits_1(self, tmp_path):
        cfg = _cfg(tmp_path)
        (cfg.output_dir / "smoke_baseline.json").write_text(
            json.dumps({"qwen3-14b:baseline_14b:chat": 0.80})
        )
        run = _run("smoke-6", [_result("inst-a", variant="other_14b", f2p=1.0)])
        with pytest.raises(typer.Exit) as exc:
            _smoke_gate(run, cfg)
        assert exc.value.exit_code == 1


# ── estimate_run_cost ────────────────────────────────────────────────────────


class TestEstimateRunCost:
    def _cost(self, results):
        from evaluation.harness import estimate_run_cost

        return estimate_run_cost(results)

    def test_empty(self):
        cost = self._cost([])
        assert cost == {"inference_usd": 0.0, "tests_usd": 0.0, "total_usd": 0.0}

    def test_zero_latency_tests_only(self):
        cost = self._cost([_result("a"), _result("b")])
        # n × 1.5 min/60 × 2 vCPU × $0.008/vCPU-hr
        assert cost["inference_usd"] == 0.0
        assert cost["tests_usd"] == pytest.approx(2 * (1.5 / 60) * 2 * 0.008)
        assert cost["total_usd"] == pytest.approx(cost["tests_usd"])

    def test_latency_amortized(self):
        r = _result("a")
        r.latency_seconds = 120.0
        cost = self._cost([r])
        # 120s/60 × $0.0417/min (A100-80GB, C3 rate fix)
        assert cost["inference_usd"] == pytest.approx(2.0 * 0.0417)


# ── paired_significance ──────────────────────────────────────────────────────


class TestPairedSignificance:
    def _paired(self, a: EvalRun, b: EvalRun) -> str:
        from evaluation.comparison import paired_significance

        return paired_significance(a, b)

    def test_same_variant_overlapping_instances(self):
        a = _run("run-a", [_result("inst-1", f2p=1.0), _result("inst-2", f2p=0.0)])
        b = _run("run-b", [_result("inst-1", f2p=0.0), _result("inst-2", f2p=1.0)])
        out = self._paired(a, b)
        assert "qwen3-14b:baseline_14b" in out
        assert "McNemar p=" in out
        assert "(n=2)" in out

    def test_disjoint_variants_skipped(self):
        a = _run("run-a", [_result("inst-1")])
        b = _run("run-b", [_result("inst-1", variant="other_14b")])
        assert "no variant evaluated in both runs" in self._paired(a, b)

    def test_shared_variant_disjoint_instances(self):
        a = _run("run-a", [_result("inst-1")])
        b = _run("run-b", [_result("inst-2")])
        assert "no overlapping instances" in self._paired(a, b)

    def test_not_self_comparison_across_variants(self):
        # Two runs with different variants must NOT pair against each other.
        a = _run("run-a", [_result("inst-1", variant="baseline_14b", f2p=1.0)])
        b = _run("run-b", [_result("inst-1", variant="higher_rank_14b", f2p=0.0)])
        out = self._paired(a, b)
        assert "no variant evaluated in both runs" in out


# ── extract_model_metrics dedupe ─────────────────────────────────────────────


class TestExtractModelMetricsDedupe:
    def test_shared_instances_counted_once(self):
        from evaluation.comparison import extract_model_metrics

        run_a = _run("run-a", [_result("inst-1", f2p=1.0), _result("inst-2", f2p=0.5)])
        run_b = _run("run-b", [_result("inst-1", f2p=1.0), _result("inst-3", f2p=0.0)])
        metrics = extract_model_metrics([run_a, run_b])
        assert metrics["qwen3-14b:baseline_14b"].total_examples == 3


# ── swebench image naming + reset failure ────────────────────────────────────


class _FakeImage:
    def pip_install(self, *args, **kwargs):
        return self


class TestSwebenchImageNaming:
    def test_munge_instance_id(self):
        from evaluation.test_runner import munge_instance_id

        assert munge_instance_id("django__django-10554") == "django_1776_django-10554"

    def test_swebench_image_name(self, monkeypatch):
        from evaluation import test_runner

        captured: dict[str, str] = {}

        def fake_from_registry(ref: str, **kwargs):
            captured["ref"] = ref
            return _FakeImage()

        monkeypatch.setattr(
            test_runner.modal.Image, "from_registry", staticmethod(fake_from_registry)
        )
        test_runner.swebench_image("django__django-10554")
        assert captured["ref"] == "swebench/sweb.eval.x86_64.django_1776_django-10554:latest"


class TestResetToBaseRaises:
    def test_missing_object_falls_back_to_head(self, tmp_path):
        """Snapshot images don't contain base_sha; reset falls back to HEAD
        (which is the snapshot) instead of raising."""
        from evaluation.test_runner import _reset_to_base

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
        (repo / "f.txt").write_text("x")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
        (repo / "f.txt").write_text("dirty")
        _reset_to_base(repo, "deadbeef" * 5)
        assert (repo / "f.txt").read_text() == "x"


# ── run_example_from_output error + latency ──────────────────────────────────


class TestRunExampleFromOutput:
    def _example(self) -> EvalInput:
        return EvalInput(
            instance_id="inst-1",
            repo="django/django",
            issue_body="fix bug",
            base_sha="abc123",
            head_sha="def456",
            test_patch="",
            fail_to_pass=["test_x"],
            pass_to_pass=[],
            repo_domain="python",
        )

    def test_error_and_latency_flow_through(self):
        from evaluation.harness import EvaluationHarness

        harness = EvaluationHarness(EvalConfig())
        output = {
            "repo": "django/django",
            "base_sha": "abc123",
            "tests_before": [],
            "tests_head": [],
            "tests_after": [],
            "patch_application": {"success": True, "method_used": "git_apply", "error": None},
            "ground_truth": {},
            "error": "ground truth F2P<100% (env drift or missing image?)",
        }
        result = harness.run_example_from_output(
            self._example(),
            "qwen3-14b",
            "baseline_14b",
            "chat",
            "patch",
            output,
            latency_seconds=42.0,
        )
        assert result.error == output["error"]
        assert result.latency_seconds == 42.0


# ── gap features: champion registry, latency percentiles ─────────────────────


class TestPromoteChampionToRegistry:
    def test_returns_none_without_wandb(self, monkeypatch):
        from evaluation import comparison

        monkeypatch.setitem(sys.modules, "wandb", None)
        assert (
            comparison.promote_champion_to_registry("qwen3-14b:baseline_14b", EvalConfig()) is None
        )

    def test_returns_none_for_empty_key(self):
        from evaluation.comparison import promote_champion_to_registry

        assert promote_champion_to_registry("", EvalConfig()) is None


class TestLatencyPercentiles:
    def test_known_list(self):
        from evaluation.harness import latency_percentiles

        results = [_result(f"inst-{i}", latency=float(v)) for i, v in enumerate([1, 2, 3, 4, 5])]
        out = latency_percentiles(results)
        assert out["qwen3-14b/baseline_14b/chat"] == {"p50": 3.0, "p95": 5.0}

    def test_empty(self):
        from evaluation.harness import latency_percentiles

        assert latency_percentiles([]) == {}

    def test_zero_latencies_excluded(self):
        from evaluation.harness import latency_percentiles

        assert latency_percentiles([_result("inst-1"), _result("inst-2")]) == {}
