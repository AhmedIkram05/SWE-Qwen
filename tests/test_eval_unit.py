"""Unit tests for the evaluation harness foundation modules.

Covers: schema validation, F2P/P2P metric math, patch application
(git apply + unidiff fallback on real unified-diff fixtures) and the
``evaluation.test_runner.classify_test_outcomes`` contract (implemented
by Agent B; skipped until that module exists).
"""

from __future__ import annotations

import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.config import EvalConfig
from evaluation.metrics import aggregate_metrics, compute_f2p
from evaluation.patch_applier import (
    _gnu_patch_supported,
    _repair_hunk_headers,
    apply_patch,
    apply_patch_git,
    apply_patch_unidiff,
)
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

try:
    from evaluation.test_runner import classify_test_outcomes as _classify_impl
except ImportError:
    _classify_impl = None

# ── Patch fixtures (real unified diffs) ──────────────────────────────────

GREETING_ORIG = 'def greet(name):\n    return f"Hello {name}"\n'
GREETING_EXPECTED = 'def greet(name):\n    return f"Hello {name}!"\n'

GIT_DIFF = """\
diff --git a/greeting.py b/greeting.py
--- a/greeting.py
+++ b/greeting.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"Hello {name}"
+    return f"Hello {name}!"
"""

NEW_FILE_DIFF = """\
diff --git a/newfile.py b/newfile.py
new file mode 100644
--- /dev/null
+++ b/newfile.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""

REMOVE_FILE_DIFF = """\
--- gone.py
+++ /dev/null
@@ -1 +0,0 @@
-x
"""

LINES_ORIG = "a1\na2\na3\nb1\nb2\nc1\nc2\nc3\n"
LINES_EXPECTED = "a1\nA2\nX\na3\nb1\nb2\nc1\nC2\nc3\n"

MULTI_HUNK_DIFF = """\
--- lines.py
+++ lines.py
@@ -1,3 +1,4 @@
 a1
-a2
+A2
+X
 a3
@@ -6,3 +6,3 @@
 c1
-c2
+C2
 c3
"""

MISSING_FILE_DIFF = """\
--- missing.py
+++ missing.py
@@ -1,2 +1,2 @@
 old1
-old2
+new2
"""


# ── Helpers ──────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _classify(attempts: list[str]) -> str:
    """Call ``classify_test_outcomes``, skipping contract tests if module missing."""
    impl = _classify_impl
    assert impl is not None, "evaluation.test_runner not implemented (Agent B)"
    return impl(attempts)


def _rev_parse(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real git repository with one committed file at HEAD."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main", "-q")
    _git(repo, "config", "user.email", "eval@test.local")
    _git(repo, "config", "user.name", "Eval Test")
    (repo / "greeting.py").write_text(GREETING_ORIG)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _make_result(
    repo: str = "owner/repo",
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    prompt: str = "chat",
    f2p: float = 1.0,
    p2p: float = 1.0,
    latency: float = 1.0,
    patch_ok: bool = True,
    flaky_test: bool = False,
) -> EvalResult:
    tests = [_TestResult(name="t1", status="passed", duration=0.1)]
    if flaky_test:
        tests.append(_TestResult(name="t2", status="flaky", duration=0.2))
    return EvalResult(
        instance_id="inst-1",
        repo=repo,
        model_name=model,
        variant=variant,
        prompt_template=prompt,
        generated_patch=GIT_DIFF,
        patch_application=PatchApplicationResult(
            success=patch_ok, method_used="git_apply" if patch_ok else "failed"
        ),
        tests_before=tests,
        tests_after=tests,
        f2p=f2p,
        p2p=p2p,
        latency_seconds=latency,
        timestamp=datetime.now(),
    )


# ── schema: TestResult ────────────────────────────────────────────────────


class TestTestResult:
    def test_defaults(self) -> None:
        tr = _TestResult(name="test_a", status="passed", duration=0.1)
        assert tr.output == ""
        assert tr.retry_count == 0

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            _TestResult(name="test_a", status="bogus", duration=0.1)  # type: ignore[arg-type]


# ── schema: PatchApplicationResult ────────────────────────────────────────


class TestPatchApplicationResult:
    def test_defaults(self) -> None:
        par = PatchApplicationResult(success=True, method_used="git_apply")
        assert par.error is None
        assert par.files_modified == []


# ── schema: EvalInput ─────────────────────────────────────────────────────


class TestEvalInput:
    def test_metadata_defaults_to_empty(self) -> None:
        inp = EvalInput(
            instance_id="i1",
            repo="r",
            issue_body="body",
            base_sha="a",
            head_sha="b",
            test_patch="",
            fail_to_pass=["t1"],
            pass_to_pass=[],
            repo_domain="unknown",
        )
        assert inp.metadata == {}

    def test_from_swebench_record_issue_record_layout(self) -> None:
        record = {
            "issue_id": "django__django-1000",
            "repo": "django/django",
            "issue_body": "Fix the bug",
            "patch_diff": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n+y\n",
            "test_results": {
                "passed": ["tests.test_a", "tests.test_b"],
                "failed": ["tests.test_c"],
                "errored": [],
            },
            "repo_domain": "web_framework",
            "metadata": {
                "instance_id": "django__django-1000",
                "base_sha": "abc123",
                "head_sha": "def456",
                "test_patch": "--- a/tests.py\n+++ b/tests.py\n@@ -1 +1 @@\n-x\n+y\n",
                "version": "1.0",
            },
        }
        inp = EvalInput.from_swebench_record(record)
        assert inp.instance_id == "django__django-1000"
        assert inp.repo == "django/django"
        assert inp.issue_body == "Fix the bug"
        assert inp.base_sha == "abc123"
        assert inp.head_sha == "def456"
        assert inp.test_patch.startswith("--- a/tests.py")
        assert inp.fail_to_pass == ["tests.test_c"]
        assert inp.pass_to_pass == ["tests.test_a", "tests.test_b"]
        assert inp.repo_domain == "web_framework"
        assert inp.metadata["version"] == "1.0"

    def test_from_swebench_record_raw_swebench_layout(self) -> None:
        record = {
            "instance_id": "astropy__astropy-100",
            "repo": "astropy/astropy",
            "problem_statement": "Issue text",
            "patch": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n+y\n",
            "test_patch": "--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-x\n+y\n",
            "FAIL_TO_PASS": "test_a, test_b",
            "PASS_TO_PASS": "test_c",
            "base_commit": "aaaa",
            "environment_setup_commit": "bbbb",
        }
        inp = EvalInput.from_swebench_record(record)
        assert inp.instance_id == "astropy__astropy-100"
        assert inp.issue_body == "Issue text"
        assert inp.base_sha == "aaaa"
        assert inp.head_sha == "bbbb"
        assert inp.fail_to_pass == ["test_a", "test_b"]
        assert inp.pass_to_pass == ["test_c"]
        assert inp.repo_domain == "unknown"


# ── schema: EvalResult / EvalRun / EvalConfig ─────────────────────────────


class TestEvalResult:
    def test_roundtrip(self) -> None:
        er = _make_result()
        restored = EvalResult.model_validate(er.model_dump())
        assert restored.instance_id == er.instance_id
        assert restored.patch_application.success
        assert restored.f2p == er.f2p
        assert restored.p2p == er.p2p
        assert restored.timestamp == er.timestamp


class TestEvalRun:
    def _run(self, status: str) -> EvalRun:
        return EvalRun(
            run_id="run-1",
            started_at=datetime.now(),
            config=EvalConfig(),
            models_evaluated=["qwen3-14b"],
            results=[],
            aggregate=[],
            status=status,  # type: ignore[arg-type]
        )

    def test_valid_status(self) -> None:
        run = self._run("running")
        restored = EvalRun.model_validate(run.model_dump())
        assert restored.run_id == "run-1"
        assert restored.completed_at is None
        assert restored.config.ci_sample_size == EvalConfig().ci_sample_size

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._run("exploded")


class TestEvalConfig:
    def test_defaults(self) -> None:
        cfg = EvalConfig()
        assert (
            cfg.golden_data_path == "gs://swe-qwen-datasets/datasets/{run_id}/swebench/golden.jsonl"
        )
        assert cfg.dataset_run_id == ""
        assert cfg.swebench_verified_filter == "metadata.is_verified==true"
        assert cfg.baseline_model == "Qwen/Qwen3-14B"
        assert cfg.wandb_entity == "2571642-university-of-dundee"
        assert cfg.lora_artifact_pattern == "model-qwen3-14b-{variant}"
        assert cfg.modal_volumes == {
            "repo_cache": "eval-repo-cache",
            "test_cache": "eval-test-cache",
        }
        assert cfg.gpu_type == "a10g-24gb"
        assert cfg.test_timeout_seconds == 30
        assert cfg.max_retries == 2
        assert cfg.min_f2p_threshold == 0.15
        assert cfg.min_p2p_threshold == 0.90
        assert cfg.ci_sample_size == 50
        assert cfg.checkpoint_dir == Path("data/eval_checkpoints")
        assert cfg.output_dir == Path("data/eval_results")

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVAL_CI_SAMPLE_SIZE", "10")
        monkeypatch.setenv("EVAL_CHECKPOINT_DIR", "data/custom_checkpoints")
        cfg = EvalConfig()
        assert cfg.ci_sample_size == 10
        assert cfg.checkpoint_dir == Path("data/custom_checkpoints")


# ── metrics: compute_f2p ──────────────────────────────────────────────────


def _tr(name: str, status: str) -> _TestResult:
    return _TestResult(name=name, status=status, duration=0.1)  # type: ignore[arg-type]


class TestComputeF2P:
    def test_basic_flip_and_hold(self) -> None:
        before = [_tr("t1", "failed"), _tr("t2", "passed")]
        after = [_tr("t1", "passed"), _tr("t2", "passed")]
        f2p_rate, p2p_rate, f2p_count, p2p_count = compute_f2p(before, after, ["t1"], ["t2"])
        assert f2p_rate == 1.0
        assert p2p_rate == 1.0
        assert f2p_count == 1
        assert p2p_count == 1

    def test_errored_before_counts_as_failed(self) -> None:
        before = [_tr("t1", "errored")]
        after = [_tr("t1", "passed")]
        f2p_rate, _, f2p_count, _ = compute_f2p(before, after, ["t1"], [])
        assert f2p_rate == 1.0
        assert f2p_count == 1

    def test_partial_flip_rate(self) -> None:
        before = [_tr("t1", "failed"), _tr("t2", "failed"), _tr("t3", "failed")]
        after = [_tr("t1", "passed"), _tr("t2", "failed"), _tr("t3", "passed")]
        f2p_rate, _, f2p_count, _ = compute_f2p(before, after, ["t1", "t2", "t3"], [])
        assert f2p_rate == pytest.approx(2 / 3)
        assert f2p_count == 2

    def test_flaky_before_excluded_from_denominator(self) -> None:
        before = [_tr("t1", "flaky"), _tr("t2", "failed")]
        after = [_tr("t1", "passed"), _tr("t2", "passed")]
        f2p_rate, _, f2p_count, _ = compute_f2p(before, after, ["t1", "t2"], [])
        assert f2p_rate == 1.0  # only t2 counts: 1/1
        assert f2p_count == 1

    def test_skipped_after_excluded_from_denominator(self) -> None:
        before = [_tr("t1", "failed")]
        after = [_tr("t1", "skipped")]
        f2p_rate, _, f2p_count, _ = compute_f2p(before, after, ["t1"], [])
        assert f2p_rate == 0.0
        assert f2p_count == 0

    def test_test_not_run_excluded(self) -> None:
        f2p_rate, _, f2p_count, _ = compute_f2p([], [], ["ghost"], [])
        assert f2p_rate == 0.0
        assert f2p_count == 0

    def test_empty_pass_to_pass_returns_1(self) -> None:
        _, p2p_rate, _, p2p_count = compute_f2p([], [], ["t1"], [])
        assert p2p_rate == 1.0
        assert p2p_count == 0

    def test_empty_fail_to_pass_returns_0(self) -> None:
        f2p_rate, _, f2p_count, _ = compute_f2p([], [], [], [])
        assert f2p_rate == 0.0
        assert f2p_count == 0

    def test_regression_breaks_p2p(self) -> None:
        before = [_tr("t2", "passed")]
        after = [_tr("t2", "failed")]
        _, p2p_rate, _, p2p_count = compute_f2p(before, after, [], ["t2"])
        assert p2p_rate == 0.0
        assert p2p_count == 0


# ── metrics: aggregate_metrics ────────────────────────────────────────────


class TestAggregateMetrics:
    def test_happy_path(self) -> None:
        results = [
            _make_result(repo="a/b", f2p=1.0, p2p=1.0, latency=1.0),
            _make_result(repo="a/b", f2p=0.5, p2p=1.0, latency=3.0),
            _make_result(repo="c/d", f2p=0.0, p2p=0.5, latency=2.0, patch_ok=False),
        ]
        m = aggregate_metrics(results)
        assert isinstance(m, F2PMetrics)
        assert m.model_name == "qwen3-14b"
        assert m.variant == "baseline_14b"
        assert m.prompt_template == "chat"
        assert m.total_examples == 3
        assert m.successful_patches == 2
        assert m.f2p_rate == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3
        assert m.f2p_count == 2
        assert m.p2p_rate == pytest.approx(0.8333, abs=1e-3)  # (1.0 + 1.0 + 0.5) / 3
        assert m.p2p_count == 3
        assert m.avg_latency == pytest.approx(2.0)
        assert m.per_repo_breakdown["a/b"]["count"] == 2
        assert m.per_repo_breakdown["a/b"]["f2p_rate"] == pytest.approx(0.75)
        assert m.per_repo_breakdown["c/d"]["p2p_rate"] == pytest.approx(0.5)
        assert m.per_repo_breakdown["c/d"]["count"] == 1

    def test_flaky_rate(self) -> None:
        m = aggregate_metrics([_make_result(flaky_test=True)])
        # 2 tests before + 2 after = 4 runs; 1 flaky in each list → 2/4
        assert m.flaky_test_rate == pytest.approx(0.5)

    def test_errored_results_excluded_from_rates(self) -> None:
        # ponytail: errored results (error is not None) carried p2p=1.0 from
        # the "nothing regressed by definition" rule; an all-errored run must
        # not show p2p_rate 100%. total_examples stays honest (includes them).
        ok = _make_result(repo="a/b", f2p=1.0, p2p=1.0)
        errored = _make_result(repo="a/b", f2p=0.0, p2p=1.0, patch_ok=False).model_copy(
            update={"error": "'Connection' object has no attribute '_transport'"}
        )
        m = aggregate_metrics([ok, errored])
        assert m.total_examples == 2
        assert m.f2p_rate == pytest.approx(1.0)  # not (1.0+0.0)/2
        assert m.p2p_rate == pytest.approx(1.0)
        assert m.p2p_count == 1
        assert m.per_repo_breakdown["a/b"]["count"] == 1

    def test_all_errored_rates_zero(self) -> None:
        errored = _make_result().model_copy(update={"error": "boom"})
        m = aggregate_metrics([errored])
        assert m.total_examples == 1
        assert m.f2p_rate == 0.0
        assert m.p2p_rate == 0.0
        assert m.p2p_count == 0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate_metrics([])


# ── patch_applier: git apply ──────────────────────────────────────────────


class TestApplyPatchGit:
    def test_success(self, git_repo: Path) -> None:
        base_sha = _rev_parse(git_repo)
        result = apply_patch_git(git_repo, GIT_DIFF, base_sha)
        assert result.success
        assert result.method_used == "git_apply"
        assert result.error is None
        assert result.files_modified == ["greeting.py"]
        assert (git_repo / "greeting.py").read_text() == GREETING_EXPECTED

    def test_invalid_base_sha_fails(self, git_repo: Path) -> None:
        result = apply_patch_git(git_repo, GIT_DIFF, "0" * 40)
        assert not result.success
        assert result.method_used == "failed"
        assert result.error is not None
        assert "checkout" in result.error

    def test_patch_that_does_not_apply_fails(self, git_repo: Path) -> None:
        base_sha = _rev_parse(git_repo)
        result = apply_patch_git(git_repo, MISSING_FILE_DIFF, base_sha)
        assert not result.success
        assert result.error is not None

    def test_empty_patch_fails(self, git_repo: Path) -> None:
        result = apply_patch_git(git_repo, "", _rev_parse(git_repo))
        assert not result.success
        assert result.error == "patch is empty"


# ── patch_applier: unidiff fallback ───────────────────────────────────────


class TestApplyPatchUnidiff:
    def test_new_file_creation(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        result = apply_patch_unidiff(target, NEW_FILE_DIFF)
        assert result.success
        assert result.method_used == "unidiff_fallback"
        assert result.files_modified == ["newfile.py"]
        assert (target / "newfile.py").read_text() == "line1\nline2\nline3\n"

    def test_multi_hunk_offsets(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        (target / "lines.py").write_text(LINES_ORIG)
        result = apply_patch_unidiff(target, MULTI_HUNK_DIFF)
        assert result.success
        assert (target / "lines.py").read_text() == LINES_EXPECTED

    def test_removed_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        (target / "gone.py").write_text("x\n")
        result = apply_patch_unidiff(target, REMOVE_FILE_DIFF)
        assert result.success
        assert not (target / "gone.py").exists()

    def test_modified_file_missing_raises_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        result = apply_patch_unidiff(target, MISSING_FILE_DIFF)
        assert not result.success
        assert result.error is not None
        assert "missing" in result.error

    def test_garbage_patch_fails(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        result = apply_patch_unidiff(target, "this is not a diff")
        assert not result.success
        assert result.error is not None

    def test_empty_patch_fails(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        result = apply_patch_unidiff(target, "")
        assert not result.success
        assert result.error == "patch is empty"


# ── patch_applier: apply_patch orchestration ──────────────────────────────


class TestApplyPatch:
    def test_git_apply_preferred_in_repo(self, git_repo: Path) -> None:
        result = apply_patch(git_repo, GIT_DIFF, _rev_parse(git_repo))
        assert result.success
        assert result.method_used == "git_apply"

    @pytest.mark.skipif(not _gnu_patch_supported(), reason="GNU patch (not BSD/Apple's) required")
    def test_gnu_fuzz_rescues_drifted_context(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")
        (repo / "drift.py").write_text("\n".join([f"line{i}" for i in range(1, 9)]) + "\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "base")
        # Hunk context says line3 but the file actually has line2 -> context drift.
        # git apply rejects it; GNU patch --fuzz=5 can still place the hunk.
        patch = (
            "diff --git a/drift.py b/drift.py\n"
            "--- a/drift.py\n"
            "+++ b/drift.py\n"
            "@@ -3,3 +3,4 @@\n"
            " line3\n"
            "+line-new\n"
        )
        result = apply_patch(repo, patch, _rev_parse(repo))
        assert result.success
        assert result.method_used == "gnu_patch_fuzz"
        assert "line-new" in (repo / "drift.py").read_text()

    def test_falls_back_to_unidiff_outside_git_repo(self, tmp_path: Path) -> None:
        workdir = tmp_path / "plain"
        workdir.mkdir()
        (workdir / "greeting.py").write_text(GREETING_ORIG)
        result = apply_patch(workdir, GIT_DIFF, "deadbeef")
        assert result.success
        assert result.method_used == "unidiff_fallback"
        assert (workdir / "greeting.py").read_text() == GREETING_EXPECTED

    def test_total_failure_returns_failed(self, tmp_path: Path) -> None:
        workdir = tmp_path / "plain"
        workdir.mkdir()
        (workdir / "greeting.py").write_text(GREETING_ORIG)
        result = apply_patch(workdir, MISSING_FILE_DIFF, "deadbeef")
        assert not result.success
        assert result.method_used == "failed"
        assert result.error is not None
        assert "git apply" in result.error
        assert "unidiff" in result.error


# ── patch_applier: LLM hunk-header repair ──────────────────────────────────


class TestRepairHunkHeaders:
    """LLM patches often miscount hunk line numbers; repair must fix them."""

    def test_recounts_single_hunk(self) -> None:
        # Header claims -10,99 +10,99 but body has 2 context + 1 added line.
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,99 +10,99 @@ def bar():\n"
            "     a = 1\n"
            "+    b = 2\n"
            "     c = 3\n"
        )
        fixed = _repair_hunk_headers(patch)
        assert "@@ -10,2 +10,3 @@ def bar():" in fixed

    def test_keeps_second_hunk_and_file_markers(self) -> None:
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,7 +1,7 @@\n"
            " a\n"
            "+b\n"
            " c\n"
            "diff --git a/bar.py b/bar.py\n"
            "--- a/bar.py\n"
            "+++ b/bar.py\n"
            "@@ -5,50 +5,50 @@\n"
            " x\n"
            "+y\n"
            " z\n"
        )
        fixed = _repair_hunk_headers(patch)
        assert "@@ -1,2 +1,3 @@" in fixed
        assert "@@ -5,2 +5,3 @@" in fixed
        assert fixed.count("diff --git") == 2

    def test_apply_patch_recovers_from_bad_header(self, git_repo: Path) -> None:
        base_sha = _rev_parse(git_repo)
        # GIT_DIFF with a miscounted hunk header: -6,6 -> -6,99 and +6,10 -> +6,99
        bad = GIT_DIFF.replace("@@ -6,6 +6,10 @@", "@@ -6,99 +6,99 @@")
        result = apply_patch(git_repo, bad, base_sha)
        assert result.success, result.error
        assert (git_repo / "greeting.py").read_text() == GREETING_EXPECTED


# ── test_runner contract (Agent B) ────────────────────────────────────────


@pytest.mark.skipif(
    _classify_impl is None,
    reason="evaluation.test_runner not implemented yet (Agent B)",
)
class TestClassifyTestOutcomes:
    def test_single_passed(self) -> None:
        assert _classify(["passed"]) == "passed"

    def test_all_passed(self) -> None:
        assert _classify(["passed", "passed"]) == "passed"

    def test_single_failed(self) -> None:
        assert _classify(["failed"]) == "failed"

    def test_all_failed(self) -> None:
        assert _classify(["failed", "failed"]) == "failed"

    def test_all_errored(self) -> None:
        assert _classify(["errored", "errored"]) == "failed"

    def test_passed_failed_mix_is_flaky(self) -> None:
        assert _classify(["passed", "failed"]) == "flaky"

    def test_retry_mix_is_flaky(self) -> None:
        assert _classify(["failed", "passed", "failed"]) == "flaky"

    def test_inconsistent_failures_are_flaky(self) -> None:
        assert _classify(["failed", "errored"]) == "flaky"

    def test_all_skipped(self) -> None:
        assert _classify(["skipped", "skipped"]) == "skipped"

    def test_skipped_passed_mix_is_flaky(self) -> None:
        assert _classify(["skipped", "passed"]) == "flaky"

    def test_empty_attempts_are_failed(self) -> None:
        assert _classify([]) == "failed"


# ── harness: Modal app lifecycle ──────────────────────────────────────────


def test_ensure_app_running_enters_once_per_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ensure_app_running`` opens ``app.run()`` once per app; repeats no-op.

    Re-entry is the critical invariant: Modal 1.5.3 raises ``InvalidError``
    if ``app.run()`` is entered while already running, so the singleton must
    never re-enter an app's context (offline: fake apps, no Modal calls).
    """
    monkeypatch.setattr("evaluation.harness._APP_RUN_STACKS", {})
    from evaluation.harness import _APP_RUN_STACKS, _ensure_app_running

    class _FakeRun:
        enters = 0

        def __enter__(self) -> _FakeRun:
            type(self).enters += 1
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    class _FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self) -> _FakeRun:
            return _FakeRun()

    app_a, app_b = _FakeApp("inference"), _FakeApp("test-runner")
    _ensure_app_running(app_a)
    _ensure_app_running(app_a)  # repeat call must not re-enter
    _ensure_app_running(app_b)

    assert _FakeRun.enters == 2  # once per app, not once per call
    assert len(_APP_RUN_STACKS) == 2


# ── inference: LLM singleton cache ─────────────────────────────────────────


def test_get_llm_returns_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_get_llm`` returns the same LLM instance on repeated calls for the same key."""
    import evaluation.inference

    evaluation.inference._LLM_CACHE.clear()

    call_count = 0

    class _FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        def load_lora_adapter(self, req: object) -> None:
            pass

    class _FakeLoRARequest:
        def __init__(self, **kwargs: object) -> None:
            pass

    # Mock vllm at sys.modules level (import happens inside _get_llm)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = _FakeLoRARequest
    fake_lora.request = fake_lora_request

    monkeypatch.setattr(evaluation.inference, "resolve_hf_id", lambda _: "fake/model")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_request)

    from evaluation.inference import _get_llm

    llm1 = _get_llm("qwen3-14b")
    llm2 = _get_llm("qwen3-14b")

    assert llm1 is llm2
    assert call_count == 1  # LLM() constructed only once
    assert len(evaluation.inference._LLM_CACHE) == 1


def test_get_llm_same_model_same_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: same base model => one LLM instance regardless of variant."""
    import evaluation.inference

    evaluation.inference._LLM_CACHE.clear()

    call_count = 0

    class _FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        def load_lora_adapter(self, req: object) -> None:
            pass

    class _FakeLoRARequest:
        def __init__(self, **kwargs: object) -> None:
            pass

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = _FakeLoRARequest
    fake_lora.request = fake_lora_request

    monkeypatch.setattr(evaluation.inference, "resolve_hf_id", lambda _: "fake/model")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_request)

    from evaluation.inference import _get_llm

    # Same model_name, same LLM (cached)
    llm_a = _get_llm("qwen3-14b")
    llm_b = _get_llm("qwen3-14b")

    assert llm_a is llm_b
    assert call_count == 1
    assert len(evaluation.inference._LLM_CACHE) == 1


# ── test_runner: batch function ───────────────────────────────────────────


def test_run_tests_batch_single_job() -> None:
    """``run_tests_batch`` exists as a Modal Function."""
    from evaluation.test_runner import run_tests_batch

    assert run_tests_batch is not None
    assert hasattr(run_tests_batch, "remote")


# ── Baseline cache (T3) ────────────────────────────────────────────────────


def test_baseline_cache_miss_when_no_file(tmp_path: Path) -> None:
    """Loading a cache for an unregistered instance returns None."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._BASELINE_CACHE_DIR
    cache_root = tmp_path / "test_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    tr_mod._BASELINE_CACHE_DIR = cache_root
    try:
        result = tr_mod._load_baseline_cache("inst-1", "abc123")
        assert result is None
    finally:
        tr_mod._BASELINE_CACHE_DIR = original


def test_baseline_cache_hit_when_matching_base_sha(tmp_path: Path) -> None:
    """Cache hit returns the stored dict when base_sha matches."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._BASELINE_CACHE_DIR
    cache_root = tmp_path / "test_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    tr_mod._BASELINE_CACHE_DIR = cache_root
    try:
        data = {
            "version": 2,
            "base_sha": "abc123",
            "tests_before": [],
            "tests_head": [],
            "ground_truth": {},
        }
        tr_mod._save_baseline_cache("inst-1", data)
        assert (cache_root / "inst-1.json").is_file()

        loaded = tr_mod._load_baseline_cache("inst-1", "abc123")
        assert loaded is not None
        assert loaded["base_sha"] == "abc123"
    finally:
        tr_mod._BASELINE_CACHE_DIR = original


def test_baseline_cache_miss_on_base_sha_mismatch(tmp_path: Path) -> None:
    """Cache with different base_sha returns None (stale)."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._BASELINE_CACHE_DIR
    cache_root = tmp_path / "test_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    tr_mod._BASELINE_CACHE_DIR = cache_root
    try:
        tr_mod._save_baseline_cache(
            "inst-1",
            {"base_sha": "oldsha", "tests_before": [], "tests_head": [], "ground_truth": {}},
        )
        loaded = tr_mod._load_baseline_cache("inst-1", "newsha")
        assert loaded is None
    finally:
        tr_mod._BASELINE_CACHE_DIR = original


def test_baseline_cache_corrupt_json(tmp_path: Path) -> None:
    """Corrupt JSON file returns None, not an exception."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._BASELINE_CACHE_DIR
    cache_root = tmp_path / "test_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "bad-inst.json").write_text("{{not json", encoding="utf-8")
    tr_mod._BASELINE_CACHE_DIR = cache_root
    try:
        result = tr_mod._load_baseline_cache("bad-inst", "abc")
        assert result is None
    finally:
        tr_mod._BASELINE_CACHE_DIR = original


def test_baseline_cache_missing_base_sha_field(tmp_path: Path) -> None:
    """Cache with no base_sha field returns None (no match)."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._BASELINE_CACHE_DIR
    cache_root = tmp_path / "test_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "no-sha.json").write_text('{"tests_before": []}', encoding="utf-8")
    tr_mod._BASELINE_CACHE_DIR = cache_root
    try:
        result = tr_mod._load_baseline_cache("no-sha", "abc")
        assert result is None  # base_sha missing → can't match
    finally:
        tr_mod._BASELINE_CACHE_DIR = original


# ── Repo-verified markers (T4) ──────────────────────────────────────────────


def test_repo_not_verified_initially(tmp_path: Path) -> None:
    """Fresh repo returns False for _is_repo_verified."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._VERIFIED_DIR
    tr_mod._VERIFIED_DIR = tmp_path / "test_cache" / "_verified"
    tr_mod._VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        assert not tr_mod._is_repo_verified("owner/repo")
    finally:
        tr_mod._VERIFIED_DIR = original


def test_mark_repo_verified_creates_file(tmp_path: Path) -> None:
    """After _mark_repo_verified, _is_repo_verified returns True."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._VERIFIED_DIR
    tr_mod._VERIFIED_DIR = tmp_path / "test_cache" / "_verified"
    tr_mod._VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        assert not tr_mod._is_repo_verified("a/b")
        tr_mod._mark_repo_verified("a/b")
        assert tr_mod._is_repo_verified("a/b")
    finally:
        tr_mod._VERIFIED_DIR = original


def test_repo_slashes_mapped_to_underscores(tmp_path: Path) -> None:
    """Repo names with slashes are stored with underscores in the marker filename."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._VERIFIED_DIR
    tr_mod._VERIFIED_DIR = tmp_path / "test_cache" / "_verified"
    tr_mod._VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tr_mod._mark_repo_verified("owner/repo")
        assert tr_mod._verified_marker_path("owner/repo").name == "owner_repo"
        assert tr_mod._verified_marker_path("a/b/c").name == "a_b_c"
    finally:
        tr_mod._VERIFIED_DIR = original


def test_repos_independent_verified_status(tmp_path: Path) -> None:
    """Verifying one repo does not mark another."""
    import evaluation.test_runner as tr_mod

    original = tr_mod._VERIFIED_DIR
    tr_mod._VERIFIED_DIR = tmp_path / "test_cache" / "_verified"
    tr_mod._VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tr_mod._mark_repo_verified("repo-a")
        assert tr_mod._is_repo_verified("repo-a")
        assert not tr_mod._is_repo_verified("repo-b")
    finally:
        tr_mod._VERIFIED_DIR = original
