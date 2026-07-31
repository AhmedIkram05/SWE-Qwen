"""Integration tests for the evaluation harness (Phase 5 acceptance #9, #12).

Covers the harness orchestration (``evaluation.harness``), comparison
(``evaluation.comparison``), CLI (``evaluation.cli``) and prompt A/B testing
(``evaluation.prompt_ab_test``) with mocked executors. Everything is offline:
patch generation and test running are replaced via the module-level
indirection functions ``evaluation.harness._generate_patches`` and
``evaluation.harness._run_tests``; W&B, Modal and GCS are never touched.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluation.cli import app as cli_app
from evaluation.comparison import (
    compare_and_report,
    extract_model_metrics,
    load_all_eval_runs,
    revalidate_champion,
)
from evaluation.config import EvalConfig
from evaluation.harness import EvaluationHarness, WandbLogger
from evaluation.prompt_ab_test import run_prompt_ab_test
from evaluation.schema import (
    EvalInput,
    EvalResult,
    EvalRun,
    F2PMetrics,
    PatchApplicationResult,
)
from evaluation.schema import (
    TestResult as _TestResult,  # noqa: N813 — avoid pytest Test* collection
)

FIX_PATCH = "+def fix():\n    return 1\n"

PASSING_PAYLOAD = {
    "tests_before": [
        {"name": "tests.test_fix", "status": "failed", "duration": 0.1},
        {"name": "tests.test_stays", "status": "passed", "duration": 0.1},
    ],
    "tests_after": [
        {"name": "tests.test_fix", "status": "passed", "duration": 0.1},
        {"name": "tests.test_stays", "status": "passed", "duration": 0.1},
    ],
    "patch_application": {
        "success": True,
        "method_used": "git_apply",
        "files_modified": ["app.py"],
    },
}


def _fake_generate(model: str, variant: str, template: str, examples: list[EvalInput]) -> list[str]:
    """Stand-in for ``_generate_patches``: one canned patch per example."""
    return [FIX_PATCH] * len(examples)


def _fake_generate_empty(
    model: str, variant: str, template: str, examples: list[EvalInput]
) -> list[str]:
    return []


def _fake_run_tests(example: EvalInput, patch: str, cfg: EvalConfig) -> dict[str, Any]:
    """Stand-in for ``_run_tests``: a passing before/after payload."""
    return PASSING_PAYLOAD


@pytest.fixture
def config(tmp_path: Path) -> EvalConfig:
    """Eval config rooted in tmp_path — no env, no dotenv, no GCS."""
    return EvalConfig(
        checkpoint_dir=tmp_path / "ckpt",
        output_dir=tmp_path / "out",
        golden_data_path=str(tmp_path / "golden.jsonl"),
        wandb_log_per_example=True,
        wandb_log_aggregate=True,
    )


def _input_record(
    instance_id: str, repo: str = "owner/repo", verified: bool | None = None
) -> dict[str, Any]:
    """An EvalInput-shaped JSONL record (model_validate-able)."""
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": repo,
        "issue_body": f"Fix the bug in {instance_id}",
        "base_sha": "a1b2c3",
        "head_sha": "d4e5f6",
        "test_patch": "--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-x\n+y\n",
        "fail_to_pass": ["tests.test_fix"],
        "pass_to_pass": ["tests.test_stays"],
        "repo_domain": "web",
    }
    if verified is not None:
        record["metadata"] = {"is_verified": verified}
    return record


def _issue_record(instance_id: str) -> dict[str, Any]:
    """An IssueRecord dump (from_swebench_record path, not model_validate-able)."""
    return {
        "issue_id": instance_id,
        "repo": "owner/repo",
        "issue_body": f"Fix the bug in {instance_id}",
        "patch_diff": "+def fix():\n    return 1\n",
        "test_results": {"failed": [], "passed": ["tests.test_stays"], "errored": []},
        "repo_domain": "web",
        "metadata": {
            "base_sha": "a1b2c3",
            "head_sha": "d4e5f6",
            "test_patch": "--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-x\n+y\n",
            "source_split": "golden",
            "has_test_patch": True,
            "is_verified": True,
            "instance_id": instance_id,
        },
    }


def _write_golden(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _make_input(instance_id: str = "demo-1", repo: str = "owner/repo") -> EvalInput:
    return EvalInput.model_validate(_input_record(instance_id, repo))


def _make_result(
    instance_id: str = "inst-1",
    repo: str = "owner/repo",
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    prompt: str = "chat",
    f2p: float = 1.0,
    p2p: float = 1.0,
    error: str | None = None,
) -> EvalResult:
    return EvalResult(
        instance_id=instance_id,
        repo=repo,
        model_name=model,
        variant=variant,
        prompt_template=prompt,
        generated_patch=FIX_PATCH,
        patch_application=PatchApplicationResult(success=True, method_used="git_apply"),
        tests_before=[
            _TestResult(name="tests.test_fix", status="failed", duration=0.1),
            _TestResult(name="tests.test_stays", status="passed", duration=0.1),
        ],
        tests_after=[
            _TestResult(name="tests.test_fix", status="passed", duration=0.1),
            _TestResult(name="tests.test_stays", status="passed", duration=0.1),
        ],
        f2p=f2p,
        p2p=p2p,
        latency_seconds=1.0,
        timestamp=datetime.now(UTC),
        error=error,
    )


def _make_metrics(model: str, variant: str, f2p: float, p2p: float) -> F2PMetrics:
    return F2PMetrics(
        model_name=model,
        variant=variant,
        prompt_template="chat",
        total_examples=10,
        successful_patches=3,
        f2p_rate=f2p,
        f2p_count=3,
        p2p_rate=p2p,
        p2p_count=9,
        avg_latency=2.5,
        flaky_test_rate=0.0,
        per_repo_breakdown={},
    )


class _FakeArtifact:
    """Minimal ``wandb.Artifact`` stand-in: records its name, touches no disk."""

    def __init__(self, name: str, type: str, metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata or {}

    def add_file(self, path: str) -> None:
        pass

    def wait(self, timeout: int = 0) -> None:
        pass


class _FakeWandb:
    """Minimal ``wandb`` module stand-in: records init/log/artifact calls."""

    def __init__(self) -> None:
        self.run = None
        self.init_calls: list[dict[str, Any]] = []
        self.log_calls: list[dict[str, Any]] = []
        self.artifact_names: list[str] = []

    def init(self, **kwargs: Any) -> None:
        self.init_calls.append(kwargs)
        self.run = object()

    def log(self, scalars: dict[str, Any]) -> None:
        self.log_calls.append(scalars)

    def Artifact(  # noqa: N802 — mirrors the real wandb.Artifact API
        self,
        name: str,
        type: str,
        metadata: dict[str, Any] | None = None,
    ) -> _FakeArtifact:
        self.artifact_names.append(name)
        return _FakeArtifact(name, type, metadata)

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        pass


def _make_run(run_id: str, config: EvalConfig, results: list[EvalResult]) -> EvalRun:
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=sorted({f"{r.model_name}:{r.variant}" for r in results}),
        results=results,
        aggregate=[],
        status="completed",
    )


def _write_run_file(config: EvalConfig, run_id: str, run: EvalRun) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / f"{run_id}.json").write_text(
        json.dumps(run.model_dump(mode="json")) + "\n", encoding="utf-8"
    )


# ── harness: Modal failure containment ──────────────────────────────────


def test_ensure_app_running_marks_failed_after_enter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``app.run()`` marks the app failed; later calls skip it.

    Without containment every later example re-enters ``app.run()``, which
    re-triggers AppCreate + image build and eventually hits Modal's
    app-create rate limit. The failed set makes the first error the last
    attempt.
    """
    monkeypatch.setattr("evaluation.harness._APP_RUN_STACKS", {})
    monkeypatch.setattr("evaluation.harness._APP_RUN_FAILED", set())
    from evaluation.harness import _APP_RUN_FAILED, _ensure_app_running

    class _BoomApp:
        def __init__(self) -> None:
            self.run_calls = 0

        def run(self) -> Any:
            self.run_calls += 1
            raise RuntimeError("image build failed")

    app = _BoomApp()

    with pytest.raises(RuntimeError, match="image build failed"):
        _ensure_app_running(app)

    assert app in _APP_RUN_FAILED

    _ensure_app_running(app)  # failed apps short-circuit without re-entering

    assert app.run_calls == 1
    assert app in _APP_RUN_FAILED


def test_generate_patches_raises_when_modal_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_generate_patches`` fails fast (no ``.remote()``) once Modal is disabled."""
    remote_calls: list[tuple[Any, ...]] = []

    class _FakeBatch:
        @staticmethod
        def remote(*args: Any) -> list[str]:
            remote_calls.append(args)
            return []

    import evaluation.inference

    monkeypatch.setattr(evaluation.inference, "app", "inference-app")
    monkeypatch.setattr(evaluation.inference, "generate_patches_batch", _FakeBatch())
    monkeypatch.setattr("evaluation.harness._APP_RUN_FAILED", {"inference-app"})

    from evaluation.harness import _generate_patches

    with pytest.raises(RuntimeError, match="Modal disabled"):
        _generate_patches("qwen3-14b", "baseline_14b", "chat", [])

    assert remote_calls == []


# ── harness: run_example ──────────────────────────────────────────────────


def test_run_example_full_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EvalConfig(
        checkpoint_dir=Path("data/ckpt"),
        output_dir=Path("data/out"),
        golden_data_path="data/golden.jsonl",
    )
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    result = harness.run_example(_make_input(), "qwen3-14b", "baseline_14b", "chat")

    assert isinstance(result, EvalResult)
    assert result.f2p == 1.0
    assert result.p2p == 1.0
    assert result.generated_patch == FIX_PATCH
    assert result.patch_application.success
    assert result.patch_application.files_modified == ["app.py"]
    assert result.error is None
    assert result.timestamp.tzinfo is not None
    assert result.latency_seconds >= 0.0


def test_run_example_patch_application_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = EvaluationHarness(
        EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    )
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    failed_payload = {
        "tests_before": [{"name": "tests.test_fix", "status": "failed", "duration": 0.1}],
        "tests_after": [{"name": "tests.test_fix", "status": "failed", "duration": 0.1}],
        "patch_application": {
            "success": False,
            "method_used": "failed",
            "error": "git apply failed",
        },
        "error": "git apply failed",
    }

    def _failed_tests(example: EvalInput, patch: str, cfg: EvalConfig) -> dict[str, Any]:
        return failed_payload

    monkeypatch.setattr("evaluation.harness._run_tests", _failed_tests)

    result = harness.run_example(_make_input(), "qwen3-14b", "baseline_14b", "chat")

    assert result.patch_application.success is False
    assert result.error is not None
    assert "git apply failed" in result.error
    assert result.f2p == 0.0


def test_run_example_generation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = EvaluationHarness(
        EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    )

    def _boom(model: str, variant: str, template: str, examples: list[EvalInput]) -> list[str]:
        raise RuntimeError("inference backend down")

    monkeypatch.setattr("evaluation.harness._generate_patches", _boom)
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    result = harness.run_example(_make_input(), "qwen3-14b", "baseline_14b", "chat")

    assert result.error == "RuntimeError: inference backend down"
    assert result.generated_patch == ""
    assert result.patch_application.success is False
    assert result.f2p == 0.0
    assert result.p2p == 0.0


def test_run_example_empty_generation_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = EvaluationHarness(
        EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    )
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate_empty,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    result = harness.run_example(_make_input(), "qwen3-14b", "baseline_14b", "chat")

    assert result.generated_patch == ""
    assert result.f2p == 1.0


# ── harness: run_batch / resume ───────────────────────────────────────────


def test_run_batch_aggregates_and_returns(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = EvaluationHarness(config)
    examples = [_make_input("demo-1"), _make_input("demo-2")]
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    results = harness.run_batch(examples, "qwen3-14b", "baseline_14b", "chat", "test-run")

    assert len(results) == 2
    assert {r.instance_id for r in results} == {"demo-1", "demo-2"}
    assert all(r.f2p == 1.0 for r in results)
    checkpoint_dir = config.checkpoint_dir / "test-run"
    assert checkpoint_dir.is_dir()
    checkpoints = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoints) == 1
    assert "owner_repo__qwen3-14b__baseline_14b__chat" in checkpoints[0].name


def test_resume_skips_completed_repos(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = EvaluationHarness(config)
    examples = [_make_input("demo-1"), _make_input("demo-2")]
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    calls: list[str] = []

    def _counting_tests(example: EvalInput, patch: str, cfg: EvalConfig) -> dict[str, Any]:
        calls.append(example.instance_id)
        return PASSING_PAYLOAD

    monkeypatch.setattr("evaluation.harness._run_tests", _counting_tests)

    first = harness.run_batch(examples, "qwen3-14b", "baseline_14b", "chat", "test-run")
    assert len(first) == 2
    assert len(calls) == 2

    second = harness.run_batch(examples, "qwen3-14b", "baseline_14b", "chat", "test-run")

    assert len(second) == 2
    assert len(calls) == 2  # completed repos short-circuit: no re-execution
    assert {r.instance_id for r in second} == {"demo-1", "demo-2"}


# ── harness: run_golden / swebench_verified ───────────────────────────────


def test_run_golden_produces_evalrun(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_golden(Path(config.golden_data_path), [_input_record("g-1"), _input_record("g-2")])
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)

    run = harness.run_golden([("qwen3-14b", "baseline_14b")], sample=0, run_id="test-run")

    assert isinstance(run, EvalRun)
    assert run.run_id == "test-run"
    assert run.status == "completed"
    assert run.models_evaluated == ["qwen3-14b:baseline_14b"]
    assert len(run.results) == 2
    assert len(run.aggregate) == 1
    assert run.aggregate[0].f2p_rate == pytest.approx(1.0)
    assert run.completed_at is not None


def test_run_swebench_verified_filters(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_golden(
        Path(config.golden_data_path),
        [
            _input_record("verified-1", verified=True),
            _input_record("not-verified-1", verified=False),
        ],
    )
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)

    run = harness.run_swebench_verified(
        [("qwen3-14b", "baseline_14b")], sample=0, run_id="verified-run"
    )

    assert len(run.results) == 1
    assert run.results[0].instance_id == "verified-1"


def test_load_examples_local_path_without_run_id(tmp_path: Path) -> None:
    _write_golden(tmp_path / "golden.jsonl", [_issue_record("exp-1"), _issue_record("exp-2")])
    config = EvalConfig(golden_data_path=str(tmp_path / "golden.jsonl"))

    examples = EvaluationHarness(config).load_examples("golden")

    assert len(examples) == 2
    assert [ex.instance_id for ex in examples] == ["exp-1", "exp-2"]
    assert all(ex.test_patch == "--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-x\n+y\n" for ex in examples)
    assert examples[0].metadata["source_split"] == "golden"


def test_load_examples_placeholder_requires_run_id() -> None:
    config = EvalConfig()  # default golden_data_path contains {run_id}

    with pytest.raises(ValueError, match=r"\{run_id\} placeholder"):
        EvaluationHarness(config).load_examples("golden")


def test_load_examples_placeholder_substituted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EvalConfig(golden_data_path="data/x/{run_id}/golden.jsonl")
    calls: list[str] = []

    def _fake_read(path: str) -> str:
        calls.append(path)
        return json.dumps(_input_record("abc-1")) + "\n"

    monkeypatch.setattr("evaluation.harness._read_text", _fake_read)

    examples = EvaluationHarness(config).load_examples("golden", run_id="abc")

    assert calls == ["data/x/abc/golden.jsonl"]
    assert [ex.instance_id for ex in examples] == ["abc-1"]


# ── WandbLogger: offline safety ───────────────────────────────────────────


def test_wandb_logging_safe_without_run(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_golden(Path(config.golden_data_path), [_input_record("g-1"), _input_record("g-2")])
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)

    run = harness.run_golden([("qwen3-14b", "baseline_14b")], run_id="wandb-safe")
    WandbLogger(config).log_eval_run(run, config)  # must not raise

    artifact_path = config.output_dir / f"{run.run_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in run.results),
        encoding="utf-8",
    )
    lines = artifact_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(EvalResult.model_validate(json.loads(line)) for line in lines)


def test_wandb_logger_inits_run_on_first_log(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First log call lazily inits one W&B run with the config's entity/project."""
    fake = _FakeWandb()
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: fake)

    logger = WandbLogger(config)
    metrics = [_make_metrics("qwen3-14b", "baseline_14b", 0.5, 0.9)]
    logger.log_aggregate(metrics, "init-run")
    logger.log_aggregate(metrics, "init-run")

    assert len(fake.init_calls) == 1  # one run per run_id, no re-init on repeat
    init_kwargs = fake.init_calls[0]
    assert init_kwargs["entity"] == config.wandb_entity
    assert init_kwargs["project"] == config.wandb_project
    assert init_kwargs["name"] == "init-run"  # run name is the run_id itself
    assert init_kwargs["reinit"] is True
    # artifact logged once per call, naming unchanged
    assert fake.artifact_names == ["eval-aggregate-init-run", "eval-aggregate-init-run"]


def test_wandb_logger_reinits_for_new_run_id(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run_id in the same process starts a fresh W&B run."""
    fake = _FakeWandb()
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: fake)

    logger = WandbLogger(config)
    metrics = [_make_metrics("qwen3-14b", "baseline_14b", 0.5, 0.9)]
    logger.log_aggregate(metrics, "run-a")
    logger.log_aggregate(metrics, "run-b")

    assert [call["name"] for call in fake.init_calls] == ["run-a", "run-b"]


def test_wandb_logger_disabled_after_init_failure(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed init disables W&B for the logger permanently, without raising."""
    init_calls: list[str] = []

    class _FailingWandb:
        run = None

        @staticmethod
        def init(**kwargs: Any) -> None:
            init_calls.append(kwargs["name"])
            raise RuntimeError("offline: no W&B credentials")

    monkeypatch.setattr("evaluation.harness._wandb_or_none", _FailingWandb)

    logger = WandbLogger(config)
    metrics = [_make_metrics("qwen3-14b", "baseline_14b", 0.5, 0.9)]
    logger.log_aggregate(metrics, "offline-run")
    logger.log_aggregate(metrics, "offline-run")

    assert init_calls == ["offline-run"]  # second call short-circuits


# ── comparison ────────────────────────────────────────────────────────────


def test_comparison_local_run_files(config: EvalConfig) -> None:
    results = [
        _make_result("inst-1", variant="baseline_14b", f2p=0.3, p2p=0.95),
        _make_result("inst-2", variant="higher_lr_14b", f2p=0.3, p2p=0.8),
    ]
    _write_run_file(config, "run1", _make_run("run1", config, results))

    runs = load_all_eval_runs(["run1"], config)

    assert len(runs) == 1
    assert runs[0].run_id == "run1"
    assert len(runs[0].results) == 2
    metrics = extract_model_metrics(runs)
    assert set(metrics) == {"qwen3-14b:baseline_14b", "qwen3-14b:higher_lr_14b"}


def test_comparison_loads_harness_persisted_run(config: EvalConfig) -> None:
    results = [_make_result("inst-1", f2p=0.5, p2p=0.95)]
    run = _make_run("persisted-run", config, results)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "persisted-run.json").write_text(
        json.dumps(run.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    runs = load_all_eval_runs(["persisted-run"], config)

    assert len(runs) == 1
    assert runs[0].run_id == "persisted-run"
    assert runs[0].status == "completed"
    assert len(runs[0].results) == 1
    assert runs[0].results[0].instance_id == "inst-1"


def test_comparison_loads_jsonl_results(config: EvalConfig) -> None:
    results = [
        _make_result("inst-1", f2p=0.5, p2p=0.95),
        _make_result("inst-2", f2p=1.0, p2p=1.0),
    ]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "jsonl-run.json").write_text(
        "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in results),
        encoding="utf-8",
    )

    runs = load_all_eval_runs(["jsonl-run"], config)

    assert len(runs) == 1
    assert runs[0].run_id == "jsonl-run"
    assert len(runs[0].results) == 2


def test_revalidate_champion_gates(config: EvalConfig) -> None:
    champion = _make_metrics("qwen3-14b", "baseline_14b", f2p=0.3, p2p=0.95)
    higher = _make_metrics("qwen3-14b", "higher_rank_14b", f2p=0.5, p2p=0.95)
    low_p2p = _make_metrics("qwen3-14b", "higher_lr_14b", f2p=0.3, p2p=0.8)
    below_floor = _make_metrics("qwen3-14b", "below_floor_14b", f2p=0.1, p2p=0.95)
    all_metrics = {
        "qwen3-14b:baseline_14b": champion,
        "qwen3-14b:higher_rank_14b": higher,
        "qwen3-14b:higher_lr_14b": low_p2p,
        "qwen3-14b:below_floor_14b": below_floor,
    }

    winner = revalidate_champion(all_metrics, "baseline_14b", min_f2p=0.15, min_p2p=0.90)

    assert winner is not None
    assert winner[0] == "qwen3-14b:higher_rank_14b"  # highest F2P among gate-passers
    assert winner[1].f2p_rate == 0.5
    assert (
        revalidate_champion({"qwen3-14b:below_floor_14b": below_floor}, "baseline_14b", 0.15, 0.90)
        is None
    )


def test_compare_and_report_markdown() -> None:
    metrics = {
        "qwen3-14b:baseline_14b": _make_metrics("qwen3-14b", "baseline_14b", 0.3, 0.95),
        "qwen3-14b:higher_rank_14b": _make_metrics("qwen3-14b", "higher_rank_14b", 0.5, 0.95),
        "qwen3-14b:higher_lr_14b": _make_metrics("qwen3-14b", "higher_lr_14b", 0.3, 0.8),
        "qwen3-14b:below_floor_14b": _make_metrics("qwen3-14b", "below_floor_14b", 0.1, 0.95),
    }

    md = compare_and_report(metrics, proxy_champion="baseline_14b")

    header = "| model | variant | total | f2p_rate | p2p_rate | avg_latency | flaky_rate | note |"
    assert header in md
    assert "qwen3-14b" in md
    assert "higher_rank_14b" in md
    assert "[champion]" in md
    assert "[proxy-champion]" in md
    assert "[rejected: p2p<90%]" in md
    assert "[rejected: f2p<15%]" in md
    champion_row = [line for line in md.splitlines() if "[champion]" in line]
    assert len(champion_row) == 1
    assert "higher_rank_14b" in champion_row[0]


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_run_command(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_golden(Path(config.golden_data_path), [_input_record("g-1")])
    run = _make_run(
        "test-run",
        config,
        [_make_result("inst-1", f2p=0.3, p2p=0.95)],
    )
    run.aggregate = [_make_metrics("qwen3-14b", "baseline_14b", 0.3, 0.95)]

    calls: dict[str, Any] = {}

    class _StubHarness:
        def __init__(self, cfg: EvalConfig) -> None:
            self.config = cfg

        def run_golden(
            self,
            pairs: list[tuple[str, str]],
            prompt_templates: list[str] | None = None,
            sample: int = 0,
            run_id: str | None = None,
        ) -> EvalRun:
            calls["pairs"] = pairs
            calls["templates"] = prompt_templates
            calls["sample"] = sample
            calls["run_id"] = run_id
            return run

    monkeypatch.setattr("evaluation.harness.EvaluationHarness", _StubHarness)
    monkeypatch.setattr("evaluation.cli.EvalConfig", lambda: config)

    result = CliRunner().invoke(
        cli_app,
        ["run", "--models", "qwen3-14b:baseline_14b", "--sample", "2", "--resume", "test-run"],
    )

    assert result.exit_code == 0, result.output
    assert "run_id: test-run" in result.output
    assert calls["pairs"] == [("qwen3-14b", "baseline_14b")]
    assert calls["templates"] == ["chat"]
    assert calls["sample"] == 2
    assert calls["run_id"] == "test-run"


def test_cli_compare_offline(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    results = [_make_result("inst-1", f2p=0.3, p2p=0.95)]
    _write_run_file(config, "run1", _make_run("run1", config, results))
    monkeypatch.setattr("evaluation.cli.EvalConfig", lambda: config)

    result = CliRunner().invoke(cli_app, ["compare", "--run_ids", "run1"])

    assert result.exit_code == 0, result.output
    assert "| model | variant | total | f2p_rate | p2p_rate |" in result.output
    assert "qwen3-14b" in result.output


# ── prompt A/B ────────────────────────────────────────────────────────────


def test_prompt_ab_runs_templates(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_golden(
        Path(config.golden_data_path),
        [_input_record("g-1"), _input_record("g-2"), _input_record("g-3")],
    )
    monkeypatch.setattr(
        "evaluation.harness._generate_patches",
        _fake_generate,
    )
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)

    run = run_prompt_ab_test(
        config,
        model="qwen3-14b",
        variant="baseline_14b",
        templates=["chat"],
        sample=2,
        run_id="ab-run",
    )

    assert run.run_id == "ab-run"
    assert run.status == "completed"
    assert len(run.results) == 2  # sample=2 caps the golden set per template
    assert len(run.aggregate) == 1  # one aggregate per template evaluated
    assert run.aggregate[0].prompt_template == "chat"
    assert run.aggregate[0].model_name == "qwen3-14b"
    assert run.aggregate[0].variant == "baseline_14b"
    assert run.aggregate[0].f2p_rate == pytest.approx(1.0)


# ── harness: batching ────────────────────────────────────────────────────────


def test_run_batch_generates_once_per_repo(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_batch`` calls ``_generate_patches`` once for ALL repos, not once per repo."""
    _write_golden(
        Path(config.golden_data_path),
        [
            _input_record("demo-1", repo="owner/repo-a"),
            _input_record("demo-2", repo="owner/repo-a"),
            _input_record("demo-3", repo="owner/repo-b"),
        ],
    )
    harness = EvaluationHarness(config)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)

    generate_calls: list[list[EvalInput]] = []

    def _tracking_generate(
        model: str, variant: str, template: str, examples: list[EvalInput]
    ) -> list[str]:
        generate_calls.append(list(examples))
        return [FIX_PATCH] * len(examples)

    monkeypatch.setattr("evaluation.harness._generate_patches", _tracking_generate)
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    harness.run_golden([("qwen3-14b", "baseline_14b")], run_id="batch-test")

    # Single call for ALL examples across ALL repos
    assert len(generate_calls) == 1
    # One call has all 3 examples
    assert len(generate_calls[0]) == 3
    repos_in_call = {ex.repo for ex in generate_calls[0]}
    assert repos_in_call == {"owner/repo-a", "owner/repo-b"}


def test_run_example_with_generated_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_example`` skips ``_generate_patches`` when ``generated_patch`` is provided."""
    harness = EvaluationHarness(
        EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    )

    generate_calls: list[tuple[str, str, str, list[EvalInput]]] = []

    def _tracking_generate(
        model: str, variant: str, template: str, examples: list[EvalInput]
    ) -> list[str]:
        generate_calls.append((model, variant, template, list(examples)))
        return [FIX_PATCH] * len(examples)

    monkeypatch.setattr("evaluation.harness._generate_patches", _tracking_generate)
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    example = _make_input()
    result = harness.run_example(
        example,
        "qwen3-14b",
        "baseline_14b",
        "chat",
        generated_patch="+pre-generated-fix\n",
    )

    # _generate_patches must NOT be called
    assert generate_calls == []
    assert result.generated_patch == "+pre-generated-fix\n"
    assert result.f2p == 1.0


def test_run_example_without_generated_patch_calls_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy path: ``run_example`` calls ``_generate_patches`` when no patch provided."""
    harness = EvaluationHarness(
        EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    )
    monkeypatch.setattr("evaluation.harness._generate_patches", _fake_generate)
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    result = harness.run_example(_make_input(), "qwen3-14b", "baseline_14b", "chat")

    assert result.generated_patch == FIX_PATCH


# ── harness: batch test execution ─────────────────────────────────────────


def test_run_example_from_output_builds_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_example_from_output`` builds EvalResult from pre-fetched output."""
    harness = EvaluationHarness(
        EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    )
    example = _make_input()
    output = {
        "tests_before": [
            {"name": "tests.test_fix", "status": "failed", "duration": 0.1},
            {"name": "tests.test_stays", "status": "passed", "duration": 0.1},
        ],
        "tests_after": [
            {"name": "tests.test_fix", "status": "passed", "duration": 0.1},
            {"name": "tests.test_stays", "status": "passed", "duration": 0.1},
        ],
        "patch_application": {
            "success": True,
            "method_used": "git_apply",
            "files_modified": ["app.py"],
        },
    }

    result = harness.run_example_from_output(
        example, "qwen3-14b", "baseline_14b", "chat", FIX_PATCH, output
    )

    assert isinstance(result, EvalResult)
    assert result.f2p == 1.0
    assert result.p2p == 1.0
    assert result.generated_patch == FIX_PATCH
    assert result.patch_application.success
    assert result.latency_seconds == 0.0  # batch path: no per-example timing
    assert result.error is None


def test_run_batch_calls_batch_tests(config: EvalConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """3 examples in same repo → 1 batch call to Modal (or 3 fallback calls)."""
    _write_golden(
        Path(config.golden_data_path),
        [
            _input_record("demo-1", repo="owner/repo"),
            _input_record("demo-2", repo="owner/repo"),
            _input_record("demo-3", repo="owner/repo"),
        ],
    )
    harness = EvaluationHarness(config)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)
    monkeypatch.setattr("evaluation.harness._generate_patches", _fake_generate)

    # Mock Modal batch to fail → triggers fallback to _run_tests
    def _boom_modal(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Modal not available")

    monkeypatch.setattr("evaluation.harness._run_tests_batch_modal", _boom_modal)

    per_job_calls: list[str] = []

    def _counting_tests(example: EvalInput, patch: str, cfg: EvalConfig) -> dict[str, Any]:
        per_job_calls.append(example.repo)
        return PASSING_PAYLOAD

    monkeypatch.setattr("evaluation.harness._run_tests", _counting_tests)

    results = harness.run_batch(
        harness.load_examples("golden"), "qwen3-14b", "baseline_14b", "chat", "batch-test"
    )

    assert len(results) == 3
    # Fallback path: 3 per-job calls when Modal batch fails
    assert len(per_job_calls) == 3


def test_run_batch_uses_modal_batch_when_available(
    config: EvalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Modal batch succeeds, only 1 batch call for all examples in repo."""
    _write_golden(
        Path(config.golden_data_path),
        [
            _input_record("demo-1", repo="owner/repo"),
            _input_record("demo-2", repo="owner/repo"),
            _input_record("demo-3", repo="owner/repo"),
        ],
    )
    harness = EvaluationHarness(config)
    monkeypatch.setattr("evaluation.harness._wandb_or_none", lambda: None)
    monkeypatch.setattr("evaluation.harness._generate_patches", _fake_generate)

    batch_calls: list[tuple[str, str, str | None, list[dict[str, Any]]]] = []

    def _fake_modal_batch(
        repo: str,
        base_sha: str,
        test_patch: str | None,
        test_jobs: list[dict[str, Any]],
        cfg: EvalConfig,
    ) -> list[dict[str, Any]]:
        batch_calls.append((repo, base_sha, test_patch, test_jobs))
        return [PASSING_PAYLOAD for _ in test_jobs]

    monkeypatch.setattr("evaluation.harness._run_tests_batch_modal", _fake_modal_batch)
    # _run_tests should NOT be called when batch succeeds
    monkeypatch.setattr("evaluation.harness._run_tests", _fake_run_tests)

    results = harness.run_batch(
        harness.load_examples("golden"), "qwen3-14b", "baseline_14b", "chat", "batch-test"
    )

    assert len(results) == 3
    assert len(batch_calls) == 1  # ONE batch call for 3 examples
    assert len(batch_calls[0][3]) == 3  # 3 jobs in the batch
    assert batch_calls[0][0] == "owner/repo"


def test_run_tests_batch_produces_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake 3 jobs via modal path, assert 3 result dicts returned."""
    from evaluation.harness import _run_tests_batch

    cfg = EvalConfig(checkpoint_dir=Path("data/ckpt"), output_dir=Path("data/out"))
    test_jobs = [
        {"generated_patch": f"patch-{i}", "fail_to_pass": ["t1"], "pass_to_pass": ["t2"]}
        for i in range(3)
    ]

    batch_args: list[Any] | None = None

    def _fake_modal(
        repo: str,
        base_sha: str,
        test_patch: str | None,
        jobs: list[dict[str, Any]],
        config: EvalConfig,
    ) -> list[dict[str, Any]]:
        nonlocal batch_args
        batch_args = [repo, base_sha, test_patch, jobs]
        return [PASSING_PAYLOAD for _ in jobs]

    monkeypatch.setattr("evaluation.harness._run_tests_batch_modal", _fake_modal)

    results = _run_tests_batch("owner/repo", "abc123", None, test_jobs, cfg)

    assert len(results) == 3
    assert batch_args is not None
    assert batch_args[0] == "owner/repo"
    assert batch_args[1] == "abc123"
    assert len(batch_args[3]) == 3
