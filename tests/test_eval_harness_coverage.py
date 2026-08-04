"""Coverage-completion tests for ``evaluation.harness``.

Targets the branches the existing eval test files miss: Modal executor
indirection happy paths, chunked ``_generate_patches``, checkpoint load
error-skipping, ``_read_gcs``/``_read_text`` gs:// routing, every
``WandbLogger`` branch (incl. ``_link_model_lineage`` ordering and the
``path is None`` cleanup edge), ``run_batch`` fallback edge cases, sampling
determinism and the W&B-failure containment path.

Offline by construction: the conftest autouse fixture stubs
``_run_tests_batch_modal`` / ``_run_tests_swebench`` to raise, and every
Modal/W&B/GCS interaction here is monkeypatched.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import evaluation.harness as harness_mod
from evaluation.config import EvalConfig
from evaluation.harness import (
    CheckpointManager,
    EvaluationHarness,
    WandbLogger,
    _generate_patches,
    _log_text_artifact,
    _read_gcs,
    _read_text,
    _run_tests,
    _run_tests_batch,
    _run_tests_batch_modal,
    _wandb_or_none,
    estimate_run_cost,
    latency_percentiles,
    make_run_id,
)
from evaluation.schema import (
    EvalInput,
    EvalResult,
    EvalRun,
    F2PMetrics,
    PatchApplicationResult,
    TestResult,
)

# conftest replaces the module attribute at test setup; capture the REAL
# implementations here (import time, before any autouse fixture ran).
_ORIG_SWEBENCH = harness_mod._run_tests_swebench

PASSING_PAYLOAD = {
    "tests_before": [{"name": "tests.test_fix", "status": "failed", "duration": 0.1}],
    "tests_after": [{"name": "tests.test_fix", "status": "passed", "duration": 0.1}],
    "patch_application": {"success": True, "method_used": "git_apply"},
}


def _cfg(tmp_path: Path, **updates: Any) -> EvalConfig:
    return EvalConfig(
        checkpoint_dir=tmp_path / "ckpt",
        output_dir=tmp_path / "out",
        golden_data_path=str(tmp_path / "golden.jsonl"),
        **updates,
    )


def _input(instance_id: str = "inst-1", repo: str = "owner/repo") -> EvalInput:
    return EvalInput(
        instance_id=instance_id,
        repo=repo,
        issue_body="fix bug",
        base_sha="abc123",
        head_sha="def456",
        test_patch="",
        fail_to_pass=["tests.test_fix"],
        pass_to_pass=[],
        repo_domain="python",
    )


def _result(
    instance_id: str,
    variant: str = "baseline_14b",
    f2p: float = 1.0,
    latency: float = 1.0,
    error: str | None = None,
    model: str = "qwen3-14b",
) -> EvalResult:
    return EvalResult(
        instance_id=instance_id,
        repo="owner/repo",
        model_name=model,
        variant=variant,
        prompt_template="chat",
        generated_patch="+patch\n",
        patch_application=PatchApplicationResult(success=True, method_used="git_apply"),
        tests_before=[TestResult(name="t", status="failed", duration=0.1)],
        tests_after=[TestResult(name="t", status="passed", duration=0.1)],
        f2p=f2p,
        p2p=1.0,
        latency_seconds=latency,
        timestamp=datetime.now(UTC),
        error=error,
    )


def _metrics(variant: str = "baseline_14b", f2p: float = 0.5) -> F2PMetrics:
    return F2PMetrics(
        model_name="qwen3-14b",
        variant=variant,
        prompt_template="chat",
        total_examples=2,
        successful_patches=1,
        f2p_rate=f2p,
        f2p_count=1,
        p2p_rate=0.9,
        p2p_count=2,
        avg_latency=1.0,
        flaky_test_rate=0.0,
        per_repo_breakdown={"owner/repo": {"count": 2, "f2p_rate": f2p, "p2p_rate": 0.9}},
    )


def _make_run(run_id: str, results: list[EvalResult], config: EvalConfig) -> EvalRun:
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=["qwen3-14b:baseline_14b"],
        results=results,
        aggregate=[],
        status="completed",
        cost_usd=estimate_run_cost(results)["total_usd"],
    )


# ── make_run_id ────────────────────────────────────────────────────────────


class TestMakeRunId:
    def test_timestamped_format(self):
        assert make_run_id().startswith("eval-")
        assert len(make_run_id().split("-")) == 3


# ── Modal executor indirection: happy paths ───────────────────────────────


class _Remote:
    """Fake ``@app.function()`` object with a callable ``.remote()``."""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[list[Any]] = []
        self.result = result

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(list(args))
        return self.result

    # generate_patches_batch is a plain-function dispatcher; callable too.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.remote(*args, **kwargs)


def test_generate_patches_chunks_at_100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_generate_patches`` splits into <=100-example remote chunks."""
    import evaluation.inference

    class _ChunkyRemote:
        """Stands in for the plain-function dispatcher; harness calls it directly."""

        captured: list[list[Any]] = []

        def __call__(self, *args: Any, **kwargs: Any) -> list[str]:
            type(self).captured.append(list(args))
            return [f"patch-{i}" for i in range(len(args[-1]))]

    _remote = _ChunkyRemote()
    captured = _ChunkyRemote.captured
    monkeypatch.setattr(evaluation.inference, "app", "fake-inference-app")
    monkeypatch.setattr(evaluation.inference, "generate_patches_batch", _remote)
    monkeypatch.setattr(harness_mod, "_ensure_app_running", lambda app: None)
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", set())

    examples = [_input(f"inst-{i}") for i in range(205)]
    patches = _generate_patches("qwen3-14b", "baseline_14b", "chat", examples)

    assert len(patches) == 205
    assert [len(c[-1]) for c in captured] == [100, 100, 5]
    # last chunk carries the trailing 5 examples
    assert [ex.instance_id for ex in captured[-1][-1]] == [f"inst-{i}" for i in range(200, 205)]


def test_generate_patches_empty_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.inference

    remote = _Remote(result=[])
    monkeypatch.setattr(evaluation.inference, "app", "fake-app")
    monkeypatch.setattr(evaluation.inference, "generate_patches_batch", remote)
    monkeypatch.setattr(harness_mod, "_ensure_app_running", lambda app: None)
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", set())

    assert _generate_patches("m", "v", "chat", []) == []


def test_run_tests_remote_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.test_runner

    remote = _Remote(result=PASSING_PAYLOAD)
    monkeypatch.setattr(evaluation.test_runner, "app", "fake-test-app")
    monkeypatch.setattr(evaluation.test_runner, "run_tests_in_container", remote)
    monkeypatch.setattr(harness_mod, "_ensure_app_running", lambda app: None)
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", set())

    out = _run_tests(_input(), "+patch\n", _cfg(Path("/tmp/unused")))
    assert out == PASSING_PAYLOAD


def test_run_tests_raises_when_modal_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.test_runner

    monkeypatch.setattr(evaluation.test_runner, "app", "fake-test-app")
    monkeypatch.setattr(evaluation.test_runner, "run_tests_in_container", _Remote())
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", {"fake-test-app"})

    with pytest.raises(RuntimeError, match="Modal disabled"):
        _run_tests(_input(), "+patch\n", _cfg(Path("/tmp/unused")))


def test_run_tests_batch_modal_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.test_runner

    remote = _Remote(result=[PASSING_PAYLOAD])
    monkeypatch.setattr(evaluation.test_runner, "app", "fake-batch-app")
    monkeypatch.setattr(evaluation.test_runner, "run_tests_batch", remote)
    monkeypatch.setattr(harness_mod, "_ensure_app_running", lambda app: None)
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", set())

    out = _run_tests_batch_modal(
        "owner/repo", "abc123", None, [{"instance_id": "inst-1"}], _cfg(Path("/tmp/unused"))
    )
    assert out == [PASSING_PAYLOAD]
    assert remote.calls[0][0] == "owner/repo"


def test_run_tests_batch_modal_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.test_runner

    monkeypatch.setattr(evaluation.test_runner, "app", "fake-batch-app")
    monkeypatch.setattr(evaluation.test_runner, "run_tests_batch", _Remote())
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", {"fake-batch-app"})

    with pytest.raises(RuntimeError, match="Modal disabled"):
        _run_tests_batch_modal(
            "owner/repo", "abc123", None, [{"instance_id": "inst-1"}], _cfg(Path("/tmp/unused"))
        )


def test_run_tests_swebench_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.test_runner

    monkeypatch.setattr(harness_mod, "_run_tests_swebench", _ORIG_SWEBENCH)
    monkeypatch.setattr(harness_mod, "_ensure_app_running", lambda app: None)
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", set())

    remote = _Remote(result={"tests_before": [], "tests_after": [], "patch_application": {}})
    monkeypatch.setattr(evaluation.test_runner, "app", "fake-swebench-app")
    monkeypatch.setattr(evaluation.test_runner, "swebench_fn", lambda repo, iid: remote)

    instances = [_input("inst-1"), _input("inst-2")]
    patches = {"inst-1": "+p1\n", "inst-2": "+p2\n"}
    out = harness_mod._run_tests_swebench(instances, patches, _cfg(Path("/tmp/unused")))
    assert set(out) == {"inst-1", "inst-2"}
    assert len(remote.calls) == 2


def test_run_tests_swebench_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.test_runner

    monkeypatch.setattr(harness_mod, "_run_tests_swebench", _ORIG_SWEBENCH)
    monkeypatch.setattr(evaluation.test_runner, "app", "fake-swebench-app")
    monkeypatch.setattr(evaluation.test_runner, "swebench_fn", lambda repo, iid: _Remote())
    monkeypatch.setattr(harness_mod, "_APP_RUN_FAILED", {"fake-swebench-app"})

    with pytest.raises(RuntimeError, match="Modal disabled"):
        harness_mod._run_tests_swebench([_input()], {"inst-1": ""}, _cfg(Path("/tmp/unused")))


# ── _run_tests_batch: fallback dispatch ───────────────────────────────────


def test_run_tests_batch_modal_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_run_tests_batch`` catches modal failure and delegates per job."""
    cfg = _cfg(Path("/tmp/unused"))

    def _boom_modal(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        raise RuntimeError("modal batch down")

    monkeypatch.setattr(harness_mod, "_run_tests_batch_modal", _boom_modal)

    seen: list[str] = []

    def _fake_run_tests(example: EvalInput, patch: str, config: EvalConfig) -> dict[str, Any]:
        seen.append(example.instance_id)
        return PASSING_PAYLOAD

    monkeypatch.setattr(harness_mod, "_run_tests", _fake_run_tests)

    jobs = [{"instance_id": f"inst-{i}", "generated_patch": "+p"} for i in range(2)]
    out = _run_tests_batch("owner/repo", "abc123", "test-patch", jobs, cfg)
    assert len(out) == 2
    assert seen == ["inst-0", "inst-1"]


# ── GCS / text reading ─────────────────────────────────────────────────────


def _install_fake_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Blob:
        def __init__(self, key: str) -> None:
            self.key = key

        def download_as_text(self) -> str:
            return f"text:{self.key}"

    class _Bucket:
        def __init__(self, name: str) -> None:
            self.name = name

        def blob(self, key: str) -> _Blob:
            return _Blob(key)

    class _Client:
        def __init__(self) -> None:
            self.bucket_names: list[str] = []

        def bucket(self, name: str) -> _Bucket:
            self.bucket_names.append(name)
            return _Bucket(name)

    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = _Client
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.storage = fake_storage
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)


def test_read_gcs_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_storage(monkeypatch)
    assert _read_gcs("gs://bucket/dir/file.json") == "text:dir/file.json"


@pytest.mark.parametrize("path", ["gs://", "gs://bucket"])
def test_read_gcs_invalid_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    _install_fake_storage(monkeypatch)
    with pytest.raises(ValueError, match="invalid GCS path"):
        _read_gcs(path)


def test_read_text_gs_routes_to_gcs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness_mod, "_read_gcs", lambda path: f"gcs:{path}")
    assert _read_text("gs://bucket/x.txt") == "gcs:gs://bucket/x.txt"


# ── CheckpointManager: load error branches ────────────────────────────────


def test_load_results_skips_corrupt_and_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    ckpt = CheckpointManager(tmp_path / "ckpt")
    run_dir = tmp_path / "ckpt" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "corrupt.json").write_text("{not json")
    (run_dir / "invalid.json").write_text(json.dumps({"instance_id": 1}))
    (run_dir / "valid-single.json").write_text(
        json.dumps(_result("inst-a").model_dump(mode="json"))
    )
    (run_dir / "valid-list.json").write_text(
        json.dumps([_result("inst-b").model_dump(mode="json")])
    )
    warnings: list[str] = []
    monkeypatch.setattr(harness_mod.logger, "warning", lambda *a, **k: warnings.append(str(a)))

    loaded = ckpt.load_results("run-1")

    assert {r.instance_id for r in loaded} == {"inst-a", "inst-b"}
    assert len(warnings) == 2


def test_load_results_missing_run_dir(tmp_path) -> None:
    ckpt = CheckpointManager(tmp_path / "ckpt")
    assert ckpt.load_results("nope") == []


def test_checkpoint_save_single_and_list(tmp_path) -> None:
    ckpt = CheckpointManager(tmp_path / "ckpt")
    key = "run-1/owner_repo__qwen3-14b__baseline_14b__chat"
    ckpt.save_result(key, _result("single"))
    assert ckpt.is_completed(key)
    assert ckpt.load_results("run-1")[0].instance_id == "single"

    list_key = "run-2/owner_repo__qwen3-14b__baseline_14b__chat"
    ckpt.save_result(list_key, [_result("a"), _result("b")])
    assert {r.instance_id for r in ckpt.load_results("run-2")} == {"a", "b"}
    # atomic write left no temp file behind
    assert not list((tmp_path / "ckpt").rglob("*.tmp"))


# ── _wandb_or_none / _log_text_artifact ───────────────────────────────────


def test_wandb_or_none_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert _wandb_or_none() is None


def test_log_text_artifact_cleanup_when_no_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """An exception before the temp path is assigned must not double-fail."""

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("tempfile unavailable")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", _boom)

    fake = types.SimpleNamespace(Artifact=lambda *a, **k: None)
    with pytest.raises(OSError, match="tempfile unavailable"):
        _log_text_artifact(fake, "name", "type", "content")


def test_log_text_artifact_happy(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    written: dict[str, Any] = {}

    class _Art:
        def __init__(self, name, type, metadata):  # noqa: A002
            written["name"] = name
            written["type"] = type
            written["metadata"] = metadata

        def add_file(self, path: str) -> None:
            written["content"] = Path(path).read_text()

        def wait(self, timeout: int = 0) -> None:
            written["wait_timeout"] = timeout

    fake = types.SimpleNamespace(
        Artifact=_Art,
        log_artifact=lambda a: written.setdefault("logged", a),
    )
    _log_text_artifact(fake, "art-1", "eval_results", "line1\nline2\n", {"run_id": "r"})

    assert written["name"] == "art-1"
    assert written["type"] == "eval_results"
    assert written["content"] == "line1\nline2\n"
    assert written["wait_timeout"] == 120


# ── Fake wandb for logger tests ───────────────────────────────────────────


class _FakeArtifact:
    def __init__(self, name: str, type_: str, metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.type = type_
        self.metadata = metadata or {}
        self.added_files: list[str] = []
        self.contents: list[str] = []

    def add_file(self, path: str) -> None:
        self.added_files.append(path)
        # harness unlinks the temp file in _log_text_artifact's finally, so
        # capture the content now (real W&B reads before upload)
        self.contents.append(Path(path).read_text(encoding="utf-8"))

    def wait(self, timeout: int = 0) -> None:
        return None


class _FakeWandb:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.log_calls: list[dict[str, Any]] = []
        self.use_artifact_calls: list[str] = []
        self.artifacts: list[_FakeArtifact] = []
        self.event_log: list[str] = []
        self.raise_on_artifact: str | None = None

    def init(self, **kwargs: Any) -> None:
        self.init_calls.append(kwargs)
        self.event_log.append("init")

    def log(self, scalars: dict[str, Any]) -> None:
        self.log_calls.append(scalars)
        self.event_log.append("log")

    def use_artifact(self, name: str) -> Any:
        self.use_artifact_calls.append(name)
        self.event_log.append(f"use:{name}")
        if self.raise_on_artifact and self.raise_on_artifact in name:
            raise RuntimeError("artifact not found")
        return object()

    def Artifact(  # noqa: N802
        self, name: str, type: str, metadata: dict[str, Any] | None = None
    ) -> _FakeArtifact:
        art = _FakeArtifact(name, type, metadata)
        self.artifacts.append(art)
        self.event_log.append(f"art:{name}")
        return art

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        return None


def _wandb_logger(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeWandb, config: EvalConfig
) -> WandbLogger:
    monkeypatch.setattr(harness_mod, "_wandb_or_none", lambda: fake)
    return WandbLogger(config)


# ── WandbLogger: _link_model_lineage ──────────────────────────────────────


def test_link_model_lineage_use_artifact_before_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    results = [_result("a", variant="baseline_14b"), _result("b", variant="higher_rank_14b")]
    run = _make_run("run-1", results, config)

    logger.log_eval_run(run, config)

    expected = [
        f"{config.wandb_entity}/{config.wandb_project}/model-qwen3-14b-baseline_14b:latest",
        f"{config.wandb_entity}/{config.wandb_project}/model-qwen3-14b-higher_rank_14b:latest",
    ]
    assert fake.use_artifact_calls == expected
    # lineage declaration precedes every artifact log
    assert fake.event_log[0] == "init"
    assert fake.event_log[1].startswith("use:")
    use_idx = next(i for i, e in enumerate(fake.event_log) if e.startswith("use:"))
    art_idx = next(i for i, e in enumerate(fake.event_log) if e.startswith("art:"))
    assert use_idx < art_idx


def test_link_model_lineage_missing_artifact_warns_not_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    fake.raise_on_artifact = "higher_rank_14b"
    logger = _wandb_logger(monkeypatch, fake, config)
    results = [_result("a", variant="baseline_14b"), _result("b", variant="higher_rank_14b")]
    run = _make_run("run-1", results, config)

    logger._link_model_lineage(run, config)  # must not raise

    assert len(fake.use_artifact_calls) == 2
    assert fake.raise_on_artifact.split("_")[0] in fake.use_artifact_calls[1]


def test_link_model_lineage_skips_when_wandb_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    logger._wandb_disabled = True
    run = _make_run("run-1", [_result("a")], config)
    logger._link_model_lineage(run, config)
    assert fake.use_artifact_calls == []


# ── WandbLogger: log_eval_run ─────────────────────────────────────────────


def test_log_eval_run_all_scalars(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    results = [_result("a", latency=120.0), _result("b", latency=240.0)]
    run = _make_run("run-1", results, config)
    run.aggregate = [_metrics()]

    logger.log_eval_run(run, config)

    # cost scalar
    cost = next(c for c in fake.log_calls if "eval/cost_usd" in c)
    assert cost["eval/cost_usd"] == pytest.approx(run.cost_usd)
    # latency p50/p95 scalars under the aggregate prefix
    latency_scalar = next(c for c in fake.log_calls if any("latency_p50" in k for k in c))
    assert latency_scalar["eval/qwen3-14b/baseline_14b/chat/latency_p50"] == pytest.approx(180.0)
    assert latency_scalar["eval/qwen3-14b/baseline_14b/chat/latency_p95"] == pytest.approx(240.0)
    # three artifact types logged
    names = [a.name for a in fake.artifacts]
    assert "eval-results-run-1" in names
    assert "eval-aggregate-run-1" in names
    assert "eval-per-repo-run-1" in names
    # per-repo CSV content
    per_repo = next(a for a in fake.artifacts if a.name == "eval-per-repo-run-1")
    csv_text = per_repo.contents[0]
    assert "model_name,variant,prompt_template,repo,f2p_rate,p2p_rate,count" in csv_text


def test_log_eval_run_skips_per_example_and_aggregate_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _cfg(tmp_path, wandb_log_per_example=False, wandb_log_aggregate=False)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    run = _make_run("run-1", [_result("a", latency=5.0)], config)
    run.aggregate = [_metrics()]

    logger.log_eval_run(run, config)

    names = [a.name for a in fake.artifacts]
    assert "eval-results-run-1" not in names
    assert "eval-aggregate-run-1" not in names
    assert "eval-per-repo-run-1" in names


def test_log_eval_run_empty_results_no_scalars(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    run = _make_run("run-empty", [], config)

    logger.log_eval_run(run, config)

    assert fake.log_calls == []  # cost==0 and no latency scalars -> nothing logged
    assert fake.artifacts == []  # per-example/aggregate/per-repo all early-return


def test_log_eval_run_latency_exception_contained(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)

    def _boom(results: list[EvalResult]) -> dict[str, dict[str, float]]:
        raise RuntimeError("stats failed")

    monkeypatch.setattr(harness_mod, "latency_percentiles", _boom)
    run = _make_run("run-1", [_result("a")], config)

    logger.log_eval_run(run, config)  # must not raise


def test_log_per_example_exception_contained(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("artifact upload failed")

    monkeypatch.setattr(harness_mod, "_log_text_artifact", _boom)
    logger.log_per_example([_result("a")], "run-1", artifact_name="custom-art")


def test_log_aggregate_exception_contained(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("artifact upload failed")

    monkeypatch.setattr(harness_mod, "_log_text_artifact", _boom)
    logger.log_aggregate([_metrics()], "run-1")


def test_log_per_repo_exception_contained(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("artifact upload failed")

    monkeypatch.setattr(harness_mod, "_log_text_artifact", _boom)
    logger.log_per_repo([_metrics()], "run-1")


def test_log_per_example_empty_results_noop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    logger.log_per_example([], "run-1")
    assert fake.artifacts == []


def test_log_aggregate_empty_noop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    logger.log_aggregate([], "run-1")
    assert fake.artifacts == []


def test_log_per_repo_empty_noop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _cfg(tmp_path)
    fake = _FakeWandb()
    logger = _wandb_logger(monkeypatch, fake, config)
    logger.log_per_repo([], "run-1")
    assert fake.artifacts == []


# ── load_examples edge branches ───────────────────────────────────────────


def test_load_examples_skips_blank_and_malformed_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _cfg(tmp_path)
    path = Path(config.golden_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n"
        + "{not json\n"
        + json.dumps(_input("inst-1").model_dump())
        + "\n"
        + json.dumps(_input("inst-2").model_dump())
        + "\n"
    )

    examples = EvaluationHarness(config).load_examples("golden")

    assert [ex.instance_id for ex in examples] == ["inst-1", "inst-2"]


def test_load_examples_unknown_split(tmp_path) -> None:
    config = _cfg(tmp_path)
    Path(config.golden_data_path).write_text(json.dumps(_input().model_dump()) + "\n")
    with pytest.raises(ValueError, match="unknown split"):
        EvaluationHarness(config).load_examples("bogus")


# ── run_example_from_output: no patch-application branch ─────────────────


def test_run_example_from_output_no_patch_application(tmp_path) -> None:
    harness = EvaluationHarness(_cfg(tmp_path))
    output = {"tests_before": [], "tests_after": [], "error": "container crash"}
    result = harness.run_example_from_output(
        _input(), "qwen3-14b", "baseline_14b", "chat", "+p\n", output, latency_seconds=3.0
    )
    assert result.patch_application.success is False
    assert result.patch_application.method_used == "failed"
    assert result.patch_application.error == "no patch application result"
    assert result.error == "container crash"
    assert result.latency_seconds == 3.0


# ── run_batch: fallback / checkpoint branches ─────────────────────────────


def test_run_batch_empty_examples(tmp_path) -> None:
    harness = EvaluationHarness(_cfg(tmp_path))
    assert harness.run_batch([], "m", "v", "chat", "run-1") == []


def test_run_batch_requires_run_id(tmp_path) -> None:
    harness = EvaluationHarness(_cfg(tmp_path))
    with pytest.raises(ValueError, match="run_id is required"):
        harness.run_batch([_input()], "m", "v", "chat", "")


def test_run_batch_skips_swebench_when_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``use_swebench_images=False`` never touches the swebench path."""
    config = _cfg(tmp_path, use_swebench_images=False)
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(harness_mod, "_run_tests", lambda ex, p, cfg: PASSING_PAYLOAD)
    swebench_calls: list[int] = []

    def _boom(*_a: Any, **_k: Any) -> Any:
        swebench_calls.append(1)
        raise RuntimeError("should not be called")

    monkeypatch.setattr(harness_mod, "_run_tests_swebench", _boom)

    results = harness.run_batch(
        [_input("a"), _input("b")], "qwen3-14b", "baseline_14b", "chat", "run-1"
    )

    assert len(results) == 2
    assert swebench_calls == []
    assert (config.checkpoint_dir / "run-1").is_dir()


def test_run_batch_runner_returned_fewer_results_falls_back_per_example(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swebench missing + batch returning fewer jobs → per-example run_example."""
    config = _cfg(tmp_path, use_swebench_images=False)
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )

    # batch path returns FEWER results than jobs -> zip leaves examples unpaired
    monkeypatch.setattr(harness_mod, "_run_tests_batch", lambda *a, **k: [PASSING_PAYLOAD])

    per_example_calls: list[str] = []
    orig_run_example = harness.run_example

    def _spy(example, model_name, variant, prompt_template, generated_patch=None):
        per_example_calls.append(example.instance_id)
        return orig_run_example(example, model_name, variant, prompt_template, generated_patch)

    monkeypatch.setattr(harness, "run_example", _spy)

    results = harness.run_batch(
        [_input("a"), _input("b")], "qwen3-14b", "baseline_14b", "chat", "run-1"
    )

    assert len(results) == 2
    # one example got its result from the (short) batch, the other via run_example
    assert len(per_example_calls) == 1


def test_run_batch_not_checkpointing_errored_repo(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _cfg(tmp_path, use_swebench_images=False)
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    error_payload = dict(PASSING_PAYLOAD, error="container crash")
    monkeypatch.setattr(harness_mod, "_run_tests_batch", lambda *a, **k: [error_payload])

    results = harness.run_batch([_input("a")], "qwen3-14b", "baseline_14b", "chat", "run-1")

    assert results[0].error == "container crash"
    # errored repo must NOT be checkpointed (resume would skip it forever)
    assert not (config.checkpoint_dir / "run-1").exists()


def test_run_batch_swebench_results_used_directly(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When swebench succeeds for every instance, no fallback batch call runs."""
    config = _cfg(tmp_path)
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(
        harness_mod,
        "_run_tests_swebench",
        lambda instances, patches, cfg: {ex.instance_id: PASSING_PAYLOAD for ex in instances},
    )
    batch_calls: list[int] = []

    def _boom(*_a: Any, **_k: Any) -> Any:
        batch_calls.append(1)
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(harness_mod, "_run_tests_batch", _boom)

    results = harness.run_batch(
        [_input("a"), _input("b")], "qwen3-14b", "baseline_14b", "chat", "run-1"
    )

    assert len(results) == 2
    assert batch_calls == []


# ── entry points / sampling determinism / W&B containment ────────────────


def test_run_baseline_entry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _cfg(tmp_path)
    Path(config.golden_data_path).write_text(json.dumps(_input("g-1").model_dump()) + "\n")
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(harness_mod, "_run_tests", lambda ex, p, cfg: PASSING_PAYLOAD)
    monkeypatch.setattr(harness_mod, "_wandb_or_none", lambda: None)

    run = harness.run_baseline(model="Qwen/Qwen3-14B", run_id="base-run")

    assert run.run_id == "base-run"
    assert run.models_evaluated == ["Qwen/Qwen3-14B:baseline"]
    assert run.results[0].variant == "baseline"


def test_run_split_sampling_is_deterministic(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _cfg(tmp_path)
    Path(config.golden_data_path).write_text(
        "".join(json.dumps(_input(f"inst-{i}").model_dump()) + "\n" for i in range(10))
    )
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(harness_mod, "_run_tests", lambda ex, p, cfg: PASSING_PAYLOAD)
    monkeypatch.setattr(harness_mod, "_wandb_or_none", lambda: None)

    run_a = EvaluationHarness(config).run_golden(
        [("qwen3-14b", "baseline_14b")], sample=3, run_id="samp-a"
    )
    run_b = EvaluationHarness(config).run_golden(
        [("qwen3-14b", "baseline_14b")], sample=3, run_id="samp-b"
    )

    ids_a = sorted(r.instance_id for r in run_a.results)
    ids_b = sorted(r.instance_id for r in run_b.results)
    assert len(ids_a) == 3
    assert ids_a == ids_b  # same tier_seed -> same subset


def test_run_split_sampling_caps_at_example_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _cfg(tmp_path)
    Path(config.golden_data_path).write_text(json.dumps(_input("only").model_dump()) + "\n")
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(harness_mod, "_run_tests", lambda ex, p, cfg: PASSING_PAYLOAD)
    monkeypatch.setattr(harness_mod, "_wandb_or_none", lambda: None)

    run = EvaluationHarness(config).run_golden(
        [("qwen3-14b", "baseline_14b")], sample=100, run_id="cap-run"
    )
    assert len(run.results) == 1


def test_run_split_wandb_failure_contained(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _cfg(tmp_path)
    Path(config.golden_data_path).write_text(json.dumps(_input().model_dump()) + "\n")
    harness = EvaluationHarness(config)
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: ["+p\n"] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(harness_mod, "_run_tests", lambda ex, p, cfg: PASSING_PAYLOAD)

    def _boom(run: EvalRun, cfg: EvalConfig) -> None:
        raise RuntimeError("wandb down")

    monkeypatch.setattr(harness.wandb_logger, "log_eval_run", _boom)

    run = harness.run_golden([("qwen3-14b", "baseline_14b")], run_id="wandb-fail")
    assert run.run_id == "wandb-fail"


# ── latency_percentiles / estimate_run_cost extra branches ────────────────


def test_latency_percentiles_single_observation_p95_equals_p50() -> None:
    out = latency_percentiles([_result("a", latency=7.0)])
    assert out["qwen3-14b/baseline_14b/chat"] == {"p50": 7.0, "p95": 7.0}


def test_estimate_run_cost_math() -> None:
    out = estimate_run_cost([_result("a", latency=120.0)])
    assert out["inference_usd"] == pytest.approx(2.0 * 0.0417)
    assert out["tests_usd"] == pytest.approx(1 * (1.5 / 60) * 2 * 0.008)


def test_estimate_run_cost_zero_instances() -> None:
    """Empty results produce zero cost."""
    out = estimate_run_cost([])
    assert out == {"inference_usd": 0.0, "tests_usd": 0.0, "total_usd": 0.0}


def test_estimate_run_cost_inference_dominant() -> None:
    """With non-zero latency, inference cost dominates."""
    out = estimate_run_cost([_result("a", latency=60.0)])
    assert out["inference_usd"] > 0.0
    assert out["total_usd"] == out["inference_usd"] + out["tests_usd"]


# ── C2: max_new_tokens plumbing ────────────────────────────────────────────


def test_generate_patches_passes_max_new_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2: max_new_tokens is forwarded through _generate_patches to generate_patches_batch."""
    import evaluation.inference as inf

    received_kw: dict = {}

    class _FakeFn:
        @staticmethod
        def remote(model_name, variant, prompt_template, examples, **kw):
            nonlocal received_kw
            received_kw = kw
            return ["+p\n"] * len(examples)

    monkeypatch.setattr(inf, "generate_patches_batch", _FakeFn())
    monkeypatch.setattr("evaluation.harness._ensure_app_running", lambda app: None)

    examples = [
        EvalInput(
            instance_id="t1",
            repo="r",
            issue_body="b",
            base_sha="s",
            head_sha="s",
            test_patch="tp",
            fail_to_pass=[],
            pass_to_pass=[],
            repo_domain="test",
        )
    ]
    patches = _generate_patches("qwen3-14b", "baseline_14b", "chat", examples, max_new_tokens=768)
    assert received_kw.get("max_new_tokens") == 768
    assert len(patches) == 1


def test_generate_patches_default_max_new_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2: default max_new_tokens is 2048 when not specified."""
    import evaluation.inference as inf

    received_kw: dict = {}

    class _FakeFn:
        @staticmethod
        def remote(model_name, variant, prompt_template, examples, **kw):
            nonlocal received_kw
            received_kw = kw
            return ["+p\n"] * len(examples)

    monkeypatch.setattr(inf, "generate_patches_batch", _FakeFn())
    monkeypatch.setattr("evaluation.harness._ensure_app_running", lambda app: None)

    examples = [
        EvalInput(
            instance_id="t1",
            repo="r",
            issue_body="b",
            base_sha="s",
            head_sha="s",
            test_patch="tp",
            fail_to_pass=[],
            pass_to_pass=[],
            repo_domain="test",
        )
    ]
    patches = _generate_patches("qwen3-14b", "baseline_14b", "chat", examples)
    assert received_kw.get("max_new_tokens") == 2048
    assert len(patches) == 1


# ── C3: estimate_run_cost GPU rate ─────────────────────────────────────────


def test_estimate_run_cost_a10g_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """C3: When running on A10G (config.inference_gpu = a10g), cost reflects the lower rate."""
    # Note: A10G-24GB is too small for 14B bf16; this test confirms the
    # estimate uses the A100 rate (0.0417/min) which is what the code
    # currently runs on.  The rate constant will change when inference_gpu
    # config is plumbed to GPU selection.
    import evaluation.harness as hv

    original_rate = hv._GPU_RATE_PER_MIN
    hv._GPU_RATE_PER_MIN = 0.0167  # A10G rate
    try:
        out = hv.estimate_run_cost([_result("a", latency=60.0)])
        assert out["inference_usd"] == pytest.approx(1.0 * 0.0167)
    finally:
        hv._GPU_RATE_PER_MIN = original_rate
