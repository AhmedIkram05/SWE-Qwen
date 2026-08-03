"""Comprehensive coverage tests for ``evaluation.test_runner``.

Targets the pure logic (classification, -k building, report parsing), the
git helpers (against real ``git`` repos in tmp dirs), ``_run_pytest_once``
(via a fake pytest subprocess so no network/real test deps), ``_execute_instance``
and the Modal function bodies (executed locally via ``.local()`` with all
filesystem/network seams monkeypatched).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from evaluation import test_runner as tr
from evaluation.schema import PatchApplicationResult, TestResult

# ── helpers ────────────────────────────────────────────────────────────────


def _test_entry(nodeid: str, outcome: str, duration: float = 0.1) -> dict:
    """A pytest-json-report style test dict."""
    return {
        "nodeid": nodeid,
        "outcome": outcome,
        "setup": {"duration": duration},
        "call": {"duration": duration, "longrepr": None, "crash": None},
        "teardown": {"duration": duration},
    }


def _report(entries: list[dict]) -> dict:
    return {"tests": entries}


def _attempt_dict(status: str, duration: float = 0.5, output: str = "") -> dict:
    return {"status": status, "duration": duration, "output": output}


def _fake_pytest(
    monkeypatch,
    tmp_path: Path,
    reports: list[dict | None],
    stdout_lines: tuple[str, ...] = (),
    exitcode: int = 0,
) -> list[str]:
    """Install a fake ``pytest`` python interpreter for ``_run_pytest_once``.

    ``reports[i]`` is the JSON report written on the i-th invocation
    (``None`` = write no report file → exercises the stdout fallback).
    Returns the ``python_cmd`` list to pass as ``_run_pytest_once``'s
    ``python_cmd``. Sets ``FAKE_PYTEST_STATE`` in the environment so the
    fake subprocess inherits it.
    """
    script = tmp_path / "fake_pytest.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            state = os.environ["FAKE_PYTEST_STATE"]
            counter = state + ".counter"
            n = 0
            if os.path.exists(counter):
                n = int(open(counter).read())
            open(counter, "w").write(str(n + 1))
            data = json.load(open(state))
            report = data["reports"][min(n, len(data["reports"]) - 1)]
            argv = sys.argv[1:]
            out = None
            if "--json-report-file" in argv:
                out = argv[argv.index("--json-report-file") + 1]
            if out and report is not None:
                json.dump(report, open(out, "w"))
            for line in data.get("stdout", []):
                print(line)
            sys.exit(data.get("exitcode", 0))
            """
        )
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"reports": reports, "stdout": list(stdout_lines), "exitcode": exitcode})
    )
    monkeypatch.setenv("FAKE_PYTEST_STATE", str(state))
    return [sys.executable, str(script)]


def _git_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a tiny real git repo; return (repo_path, HEAD sha)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, sha


def _fake_apply(success: bool, method: str = "git_apply", error: str | None = None):
    return PatchApplicationResult(
        success=success,
        method_used=method
        if method in ("git_apply", "unidiff_fallback", "failed", "none")
        else "failed",
        error=error,
    )


# ── classify_test_outcomes ──────────────────────────────────────────────────


class TestClassifyTestOutcomes:
    def test_empty_is_failed(self):
        assert tr.classify_test_outcomes([]) == "failed"

    def test_all_passed(self):
        assert tr.classify_test_outcomes(["passed", "passed"]) == "passed"

    def test_all_failed(self):
        assert tr.classify_test_outcomes(["failed", "failed"]) == "failed"

    def test_all_errored_is_failed(self):
        assert tr.classify_test_outcomes(["errored", "errored"]) == "failed"

    def test_all_skipped(self):
        assert tr.classify_test_outcomes(["skipped", "skipped"]) == "skipped"

    def test_mixed_pass_fail_is_flaky(self):
        assert tr.classify_test_outcomes(["failed", "passed"]) == "flaky"

    def test_mixed_fail_errored_is_flaky(self):
        assert tr.classify_test_outcomes(["failed", "errored"]) == "flaky"

    def test_single_unknown_status_is_failed(self):
        assert tr.classify_test_outcomes(["whatever"]) == "failed"


# ── _quote_k_name / _build_k_expression ─────────────────────────────────────


class TestQuoteKName:
    def test_plain_name_unchanged(self):
        assert tr._quote_k_name("test_foo") == "test_foo"

    def test_node_id_strips_file_path(self):
        assert tr._quote_k_name("tests/foo.py::test_bar") == "test_bar"

    def test_space_stripped(self):
        # pytest 7's -k parser has no quoted strings: any character outside
        # the ident set aborts parsing, so spaces are stripped (names still
        # match via substring).
        assert tr._quote_k_name("test with space") == "testwithspace"

    def test_quotes_and_backslash_stripped(self):
        # pytest's -k parser rejects embedded quotes/backslashes (no escape
        # support), so they are stripped rather than escaped.
        assert tr._quote_k_name('te"st\\x') == "testx"

    def test_parametrize_suffix_bare(self):
        # Brackets are part of the ident grammar, so no quoting is needed.
        assert tr._quote_k_name("test_x[1]") == "test_x[1]"


class TestBuildKExpression:
    def test_single(self):
        assert tr._build_k_expression(["test_a"]) == "test_a"

    def test_multiple(self):
        assert tr._build_k_expression(["test_a", "test_b"]) == "test_a or test_b"

    def test_empty(self):
        assert tr._build_k_expression([]) == ""


# ── _failure_text ───────────────────────────────────────────────────────────


class TestFailureText:
    def test_no_info_returns_empty(self):
        assert tr._failure_text({}) == ""

    def test_longrepr_string(self):
        entry = {"call": {"longrepr": "assert 1 == 2"}}
        assert tr._failure_text(entry) == "assert 1 == 2"

    def test_longrepr_dict_with_reprcrash(self):
        entry = {
            "call": {"longrepr": {"reprcrash": {"path": "a.py", "lineno": 7, "message": "boom"}}}
        }
        assert tr._failure_text(entry) == "a.py:7: boom"

    def test_longrepr_dict_without_reprcrash(self):
        entry = {"call": {"longrepr": {"repr_traceback": "tb text"}}}
        assert tr._failure_text(entry) == "tb text"

    def test_longrepr_dict_without_traceback(self):
        entry = {"call": {"longrepr": {"nested": True}}}
        assert tr._failure_text(entry) == str({"nested": True})

    def test_non_dict_phase_skipped(self):
        entry = {"call": "not a dict", "setup": {"longrepr": "setup fail"}}
        assert tr._failure_text(entry) == "setup fail"

    def test_crash_dict_without_longrepr(self):
        entry = {"call": {"crash": {"message": "crash msg"}}}
        assert tr._failure_text(entry) == "crash msg"

    def test_crash_dict_no_message(self):
        entry = {"call": {"crash": {"other": 1}}}
        assert tr._failure_text(entry) == str({"other": 1})

    def test_non_dict_crash_returns_empty(self):
        entry = {"call": {"crash": "just a string"}}
        assert tr._failure_text(entry) == ""


# ── _attempt_from_report_test ───────────────────────────────────────────────


class TestAttemptFromReportTest:
    def test_duration_summed_across_phases(self):
        entry = {
            "outcome": "passed",
            "setup": {"duration": 0.1},
            "call": {"duration": 0.2},
            "teardown": {"duration": 0.3},
        }
        a = tr._attempt_from_report_test(entry)
        assert a["status"] == "passed"
        assert a["duration"] == pytest.approx(0.6)

    def test_unknown_outcome_maps_to_errored(self):
        entry = {"outcome": "weird", "call": {"longrepr": "x"}}
        a = tr._attempt_from_report_test(entry)
        assert a["status"] == "errored"
        assert a["output"] == "x"

    def test_xpassed_maps_to_passed(self):
        a = tr._attempt_from_report_test({"outcome": "xpassed"})
        assert a["status"] == "passed"

    def test_xfailed_maps_to_failed(self):
        a = tr._attempt_from_report_test({"outcome": "xfailed"})
        assert a["status"] == "failed"

    def test_error_maps_to_errored(self):
        a = tr._attempt_from_report_test({"outcome": "error"})
        assert a["status"] == "errored"

    def test_missing_duration_is_zero(self):
        a = tr._attempt_from_report_test({"outcome": "skipped"})
        assert a["duration"] == 0.0
        assert a["status"] == "skipped"


# ── _load_json_report ───────────────────────────────────────────────────────


class TestLoadJsonReport:
    def test_valid_dict(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"tests": []}))
        assert tr._load_json_report(p) == {"tests": []}

    def test_missing_file(self, tmp_path):
        assert tr._load_json_report(tmp_path / "nope.json") is None

    def test_corrupt_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert tr._load_json_report(p) is None

    def test_non_dict_json(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text(json.dumps([1, 2, 3]))
        assert tr._load_json_report(p) is None


# ── _attempts_from_report ───────────────────────────────────────────────────


class TestAttemptsFromReport:
    def test_no_tests_key(self):
        by_name, nodeids = tr._attempts_from_report({}, ["test_a"])
        assert by_name == {}
        assert nodeids == []

    def test_tests_not_a_list(self):
        by_name, nodeids = tr._attempts_from_report({"tests": "nope"}, ["test_a"])
        assert by_name == {}
        assert nodeids == []

    def test_empty_tests_with_names_logs_warning(self, caplog):
        with caplog.at_level("WARNING", logger="evaluation.test_runner"):
            by_name, nodeids = tr._attempts_from_report({"tests": []}, ["test_a"])
        assert by_name == {}
        assert nodeids == []
        assert "collected no tests" in caplog.text

    def test_exact_nodeid_match(self):
        entry = _test_entry("tests/a.py::test_b", "passed")
        by_name, nodeids = tr._attempts_from_report(_report([entry]), ["tests/a.py::test_b"])
        assert nodeids == ["tests/a.py::test_b"]
        assert by_name["tests/a.py::test_b"]["status"] == "passed"

    def test_suffix_match(self):
        entry = _test_entry("tests/a.py::test_b", "failed")
        by_name, _ = tr._attempts_from_report(_report([entry]), ["test_b"])
        assert by_name["test_b"]["status"] == "failed"

    def test_nodeid_contained_in_name(self):
        entry = _test_entry("tests/a.py::test_b", "passed")
        by_name, _ = tr._attempts_from_report(_report([entry]), ["mod::tests/a.py::test_b"])
        assert "mod::tests/a.py::test_b" in by_name

    def test_unmatched_name_absent(self):
        entry = _test_entry("tests/a.py::test_b", "passed")
        by_name, _ = tr._attempts_from_report(_report([entry]), ["test_zzz"])
        assert by_name == {}

    def test_duplicate_nodeids_use_first(self):
        entries = [
            {"nodeid": "tests/a.py::t", "outcome": "passed"},
            {"nodeid": "tests/a.py::t", "outcome": "failed"},
        ]
        by_name, nodeids = tr._attempts_from_report({"tests": entries}, ["tests/a.py::t"])
        assert by_name["tests/a.py::t"]["status"] == "passed"
        assert nodeids == ["tests/a.py::t", "tests/a.py::t"]


# ── _parse_stdout_report ────────────────────────────────────────────────────


class TestParseStdoutReport:
    def test_parses_expected_format(self):
        out = tr._parse_stdout_report("passed tests/a.py::t1\nfailed tests/a.py::t2\n")
        assert out is not None
        assert out["tests/a.py::t1"]["status"] == "passed"
        assert out["tests/a.py::t2"]["status"] == "failed"
        assert out["tests/a.py::t1"]["output"] == "passed tests/a.py::t1"

    def test_skips_unknown_and_garbage_lines(self):
        out = tr._parse_stdout_report("passed tests/a.py::t1\nBOGUS x\nnot a status\n")
        assert list(out or {}) == ["tests/a.py::t1"]

    def test_empty_returns_none(self):
        assert tr._parse_stdout_report("") is None

    def test_no_valid_lines_returns_none(self):
        assert tr._parse_stdout_report("random noise\n") is None


# ── _run_pytest_once ────────────────────────────────────────────────────────


class TestRunPytestOnce:
    def test_stdout_report_success(self, tmp_path, monkeypatch):
        cmd = _fake_pytest(
            monkeypatch,
            tmp_path,
            [None],
            stdout_lines=("passed tests/a.py::test_b",),
        )
        attempts, nodeids = tr._run_pytest_once(tmp_path, ["test_b"], timeout=30, python_cmd=cmd)
        assert nodeids == ["tests/a.py::test_b"]
        assert attempts["tests/a.py::test_b"]["status"] == "passed"

    def test_json_report_read_when_unlink_fails(self, tmp_path, monkeypatch, caplog):
        cmd = _fake_pytest(
            monkeypatch,
            tmp_path,
            [_report([_test_entry("tests/a.py::test_b", "passed", duration=0.2)])],
            stdout_lines=(),
        )
        orig_unlink = Path.unlink

        def raisy(self, *a, **k):
            if str(self).endswith(".json"):
                raise OSError("boom")
            return orig_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", raisy)
        with caplog.at_level("WARNING", logger="evaluation.test_runner"):
            attempts, nodeids = tr._run_pytest_once(
                tmp_path, ["test_b"], timeout=30, python_cmd=cmd
            )
        assert nodeids == ["tests/a.py::test_b"]
        assert attempts["test_b"]["status"] == "passed"
        assert attempts["test_b"]["duration"] == pytest.approx(0.6)
        assert "Failed to cleanup report file" in caplog.text

    def test_no_report_and_no_stdout_errored(self, tmp_path, monkeypatch):
        cmd = _fake_pytest(monkeypatch, tmp_path, [None], stdout_lines=("noise",))
        attempts, nodeids = tr._run_pytest_once(tmp_path, ["t1"], timeout=30, python_cmd=cmd)
        assert attempts == {"t1": tr._errored_attempt("pytest produced no report")}
        assert nodeids == []

    def test_nonzero_exit_stdout_still_parsed(self, tmp_path, monkeypatch):
        cmd = _fake_pytest(
            monkeypatch,
            tmp_path,
            [None],
            stdout_lines=("failed tests/a.py::t1",),
            exitcode=3,
        )
        attempts, _ = tr._run_pytest_once(tmp_path, ["t1"], timeout=30, python_cmd=cmd)
        assert attempts["tests/a.py::t1"]["status"] == "failed"

    def test_timeout_expired_returns_errored(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(["cmd"], 5)

        monkeypatch.setattr(tr.subprocess, "run", boom)
        attempts, nodeids = tr._run_pytest_once(tmp_path, ["t1", "t2"], timeout=30)
        assert nodeids == []
        assert set(attempts) == {"t1", "t2"}
        assert all(a["status"] == "errored" for a in attempts.values())
        assert "timed out" in attempts["t1"]["output"]

    def test_missing_pytest_binary_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tr._run_pytest_once(
                tmp_path, ["t1"], timeout=30, python_cmd=["/nonexistent/pytest-binary"]
            )

    def test_venv_detection_uses_parent_venv(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "proj"
        repo_dir.mkdir(parents=True)
        fake = _fake_pytest(
            monkeypatch,
            tmp_path,
            [None],
            stdout_lines=("passed tests/a.py::t1",),
        )
        state = tmp_path / "state.json"
        script = fake[1]
        venv_python = tmp_path / "repos" / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text(
            "#!/usr/bin/env python3\n"
            "import os, runpy, sys\n"
            f"os.environ['FAKE_PYTEST_STATE'] = {str(state)!r}\n"
            f"sys.argv[0] = {str(script)!r}\n"
            f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
        )
        venv_python.chmod(0o755)
        attempts, _ = tr._run_pytest_once(repo_dir, ["t1"], timeout=30)
        assert attempts["tests/a.py::t1"]["status"] == "passed"

    def test_approaching_modal_timeout_warns(self, tmp_path, monkeypatch, caplog):
        cmd = _fake_pytest(
            monkeypatch,
            tmp_path,
            [None],
            stdout_lines=("passed tests/a.py::t1",),
        )
        calls = {"n": 0}

        def slow_time():
            calls["n"] += 1
            return 1000.0 if calls["n"] > 1 else 0.0

        monkeypatch.setattr(time, "time", slow_time)
        with caplog.at_level("WARNING", logger="evaluation.test_runner"):
            attempts, _ = tr._run_pytest_once(tmp_path, ["t1"], timeout=30, python_cmd=cmd)
        assert attempts["tests/a.py::t1"]["status"] == "passed"
        assert "approaching limit" in caplog.text

    def test_cleanup_failure_is_logged_not_raised(self, tmp_path, monkeypatch, caplog):
        cmd = _fake_pytest(
            monkeypatch,
            tmp_path,
            [None],
            stdout_lines=("passed tests/a.py::t1",),
        )
        orig_unlink = Path.unlink

        def raisy(self, *a, **k):
            if str(self).endswith(".json"):
                raise OSError("boom")
            return orig_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", raisy)
        with caplog.at_level("WARNING", logger="evaluation.test_runner"):
            attempts, _ = tr._run_pytest_once(tmp_path, ["t1"], timeout=30, python_cmd=cmd)
        assert attempts["tests/a.py::t1"]["status"] == "passed"


# ── collect_test_results ────────────────────────────────────────────────────


class TestCollectTestResults:
    def test_happy_path_all_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tr,
            "_run_pytest_once",
            lambda *a, **k: (
                {"test_a": _attempt_dict("passed"), "test_b": _attempt_dict("passed")},
                ["test_a", "test_b"],
            ),
        )
        results = tr.collect_test_results(tmp_path, ["test_a", "test_b"])
        assert len(results) == 2
        assert results[0].status == "passed"
        assert results[0].retry_count == 0

    def test_retry_flaky_escalation(self, tmp_path, monkeypatch):
        states = iter(
            [
                ({"test_a": _attempt_dict("failed", 0.5, "o0")}, ["test_a"]),
                ({"test_a": _attempt_dict("passed", 0.4, "o1")}, ["test_a"]),
            ]
        )

        def fake_run(*a, **k):
            return next(states)

        monkeypatch.setattr(tr, "_run_pytest_once", fake_run)
        results = tr.collect_test_results(tmp_path, ["test_a"], max_retries=2)
        assert len(results) == 1
        assert results[0].status == "flaky"
        assert results[0].retry_count == 1
        assert results[0].output == "o1"

    def test_retry_still_failed(self, tmp_path, monkeypatch):
        states = iter(
            [
                ({"test_a": _attempt_dict("failed")}, ["test_a"]),
                ({"test_a": _attempt_dict("failed")}, ["test_a"]),
            ]
        )
        monkeypatch.setattr(tr, "_run_pytest_once", lambda *a, **k: next(states))
        results = tr.collect_test_results(tmp_path, ["test_a"], max_retries=1)
        assert results[0].status == "failed"
        assert results[0].retry_count == 1

    def test_missing_attempt_uses_missing_sentinel(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "_run_pytest_once", lambda *a, **k: ({}, []))
        results = tr.collect_test_results(tmp_path, ["ghost"])
        assert results[0].status == "failed"
        assert "not collected" in results[0].output

    def test_full_suite_mode(self, tmp_path, monkeypatch):
        entry = _test_entry("tests/a.py::t1", "failed")
        per_run = {"tests/a.py::t1": tr._attempt_from_report_test(entry)}
        monkeypatch.setattr(tr, "_run_pytest_once", lambda *a, **k: (per_run, ["tests/a.py::t1"]))
        results = tr.collect_test_results(tmp_path, [])
        assert len(results) == 1
        assert results[0].name == "tests/a.py::t1"
        assert results[0].status == "failed"

    def test_zero_tests_empty_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "_run_pytest_once", lambda *a, **k: ({}, []))
        assert tr.collect_test_results(tmp_path, []) == []

    def test_empty_strings_filtered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tr, "_run_pytest_once", lambda *a, **k: ({"t1": _attempt_dict("passed")}, ["t1"])
        )
        results = tr.collect_test_results(tmp_path, ["", "t1"])
        assert [r.name for r in results] == ["t1"]

    def test_duration_prefers_positive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tr,
            "_run_pytest_once",
            lambda *a, **k: ({"t1": _attempt_dict("passed", 0.0, "o")}, ["t1"]),
        )
        results = tr.collect_test_results(tmp_path, ["t1"])
        assert results[0].duration == 0.0

    def test_timeout_truncation_warns(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            tr, "_run_pytest_once", lambda *a, **k: ({"t1": _attempt_dict("passed")}, ["t1"])
        )
        calls = {"n": 0}

        def fake_time():
            calls["n"] += 1
            return 1000.0 if calls["n"] > 1 else 0.0

        monkeypatch.setattr(time, "time", fake_time)
        with caplog.at_level("WARNING", logger="evaluation.test_runner"):
            results = tr.collect_test_results(tmp_path, ["t1"], max_retries=3)
        assert results == []
        assert "taking too long" in caplog.text


# ── _run_git ────────────────────────────────────────────────────────────────


class TestRunGit:
    def test_success(self, tmp_path):
        repo, _ = _git_repo(tmp_path)
        proc = tr._run_git(repo, "log", "--oneline")
        assert proc.returncode == 0

    def test_failure_check_false_returns_rc(self, tmp_path):
        proc = tr._run_git(tmp_path, "status")
        assert proc.returncode != 0

    def test_failure_check_true_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="git status failed"):
            tr._run_git(tmp_path, "status", check=True)


# ── _clone_repo ─────────────────────────────────────────────────────────────


class TestCloneRepo:
    def test_already_cached(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "o" / "r"
        (repo_dir / ".git").mkdir(parents=True)
        calls = []
        monkeypatch.setattr(tr, "_run_git", lambda *a, **k: calls.append(a) or None)
        tr._clone_repo("o/r", repo_dir)
        assert calls == []

    def test_fresh_clone(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "cache" / "r"
        recorded = {}

        def fake_git(cwd, *args, check=False):
            recorded["cwd"] = cwd
            recorded["args"] = args
            return subprocess.CompletedProcess(["git"], 0, "", "")

        monkeypatch.setattr(tr, "_run_git", fake_git)
        tr._clone_repo("o/r", repo_dir)
        assert recorded["cwd"] == tmp_path / "cache"
        assert recorded["args"] == ("clone", "--quiet", "https://github.com/o/r.git", "r")

    def test_clone_failure_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tr,
            "_run_git",
            lambda *a, **k: subprocess.CompletedProcess(["git"], 128, "", "auth error"),
        )
        with pytest.raises(RuntimeError, match="clone of o/r failed"):
            tr._clone_repo("o/r", tmp_path / "cache" / "r")


# ── _ensure_checked_out ─────────────────────────────────────────────────────


class TestEnsureCheckedOut:
    def test_at_base_sha_and_lock_removed(self, tmp_path):
        repo, sha = _git_repo(tmp_path)
        lock = repo / ".git" / "index.lock"
        lock.write_text("stale")
        tr._ensure_checked_out(repo, sha)
        assert not lock.exists()

    def test_needs_fetch_and_retry(self, tmp_path, monkeypatch):
        repo, _ = _git_repo(tmp_path)
        calls: list[list] = []

        def fake_git(cwd, *args, **kwargs):
            calls.append(list(args))
            if args[0] == "checkout":
                if len([c for c in calls if c[0] == "checkout"]) == 1:
                    return subprocess.CompletedProcess(["git"], 1, "", "unknown revision")
                return subprocess.CompletedProcess(["git"], 0, "", "")
            if args[0] == "fetch":
                return subprocess.CompletedProcess(["git"], 0, "", "")
            return subprocess.CompletedProcess(["git"], 0, "", "")

        monkeypatch.setattr(tr, "_run_git", fake_git)
        tr._ensure_checked_out(repo, "beef")
        checkout_calls = [c for c in calls if c[0] == "checkout"]
        assert len(checkout_calls) == 2
        assert any(c[0] == "fetch" for c in calls)

    def test_fetch_then_still_fails_raises(self, tmp_path, monkeypatch):
        repo, _ = _git_repo(tmp_path)

        def fake_git(cwd, *args, **kwargs):
            if args[0] == "fetch":
                return subprocess.CompletedProcess(["git"], 0, "", "")
            return subprocess.CompletedProcess(["git"], 128, "", "nope")

        monkeypatch.setattr(tr, "_run_git", fake_git)
        with pytest.raises(RuntimeError, match="failed to checkout beef"):
            tr._ensure_checked_out(repo, "beef")


# ── _reset_to_base ──────────────────────────────────────────────────────────


class TestResetToBase:
    def test_clean_reset(self, tmp_path):
        repo, sha = _git_repo(tmp_path)
        (repo / "f.txt").write_text("dirty")
        tr._reset_to_base(repo, sha)
        assert (repo / "f.txt").read_text() == "x"

    def test_dirty_with_untracked_files_cleaned(self, tmp_path):
        repo, sha = _git_repo(tmp_path)
        (repo / "f.txt").write_text("dirty")
        (repo / "untracked.txt").write_text("junk")
        tr._reset_to_base(repo, sha)
        assert (repo / "f.txt").read_text() == "x"
        assert not (repo / "untracked.txt").exists()

    def test_unknown_sha_raises(self, tmp_path):
        repo, _ = _git_repo(tmp_path)
        with pytest.raises(RuntimeError, match="git reset --hard"):
            tr._reset_to_base(repo, "deadbeef" * 5)


# ── _install_repo ───────────────────────────────────────────────────────────


class TestInstallRepo:
    def test_marker_fast_path(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "demo"
        repo_dir.mkdir(parents=True)
        cache = tmp_path / "cache"
        venv_dir = cache / "demo" / ".venv"
        marker = venv_dir / "demo.installed"
        marker.parent.mkdir(parents=True)
        marker.touch()
        calls = []
        monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: calls.append(a) or None)
        orig_prefix = sys.prefix
        orig_path = os.environ.get("PATH", "")
        try:
            tr._install_repo(repo_dir, cache_dir=cache)
            assert calls == []
            assert sys.prefix == str(venv_dir)
            assert os.environ["PATH"].startswith(str(venv_dir / "bin") + os.pathsep)
        finally:
            sys.prefix = orig_prefix
            os.environ["PATH"] = orig_path

    def test_fresh_install(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "demo"
        repo_dir.mkdir(parents=True)
        cache = tmp_path / "cache"
        calls: list[list] = []

        def fake_run(cmd, *a, **k):
            calls.append(list(cmd))
            if "-m" in cmd and "venv" in cmd:
                Path(cmd[-1]).mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(tr.subprocess, "run", fake_run)
        orig_path = os.environ.get("PATH", "")
        try:
            tr._install_repo(repo_dir, cache_dir=cache)
        finally:
            os.environ["PATH"] = orig_path
        assert any("-m" in c and "venv" in c for c in calls)
        assert any(c[0].endswith("pip") and "install" in c and "-e" in c for c in calls)
        marker = cache / "demo" / ".venv" / "demo.installed"
        assert marker.exists()

    def test_pip_failure_uses_setup_py(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "demo"
        repo_dir.mkdir(parents=True)
        (repo_dir / "setup.py").write_text("")
        cache = tmp_path / "cache"
        calls: list[list] = []

        def fake_run(cmd, *a, **k):
            calls.append(list(cmd))
            if "-m" in cmd and "venv" in cmd:
                Path(cmd[-1]).mkdir(parents=True)
            if cmd[0].endswith("pip"):
                return subprocess.CompletedProcess(cmd, 1, "", "pip boom")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(tr.subprocess, "run", fake_run)
        orig_path = os.environ.get("PATH", "")
        try:
            tr._install_repo(repo_dir, cache_dir=cache)
        finally:
            os.environ["PATH"] = orig_path
        assert any(c[0].endswith("python") and "setup.py" in c and "develop" in c for c in calls)
        assert (cache / "demo" / ".venv" / "demo.installed").exists()

    def test_pip_failure_no_setup_py(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "demo"
        repo_dir.mkdir(parents=True)
        cache = tmp_path / "cache"
        calls: list[list] = []

        def fake_run(cmd, *a, **k):
            calls.append(list(cmd))
            if "-m" in cmd and "venv" in cmd:
                Path(cmd[-1]).mkdir(parents=True)
            if cmd[0].endswith("pip"):
                return subprocess.CompletedProcess(cmd, 1, "", "pip boom")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(tr.subprocess, "run", fake_run)
        orig_path = os.environ.get("PATH", "")
        try:
            tr._install_repo(repo_dir, cache_dir=cache)
        finally:
            os.environ["PATH"] = orig_path
        assert not any("setup.py" in c for c in calls)

    def test_venv_creation_failure_raises(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "demo"
        repo_dir.mkdir(parents=True)

        def boom(cmd, *a, **k):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(tr.subprocess, "run", boom)
        with pytest.raises(subprocess.CalledProcessError):
            tr._install_repo(repo_dir, cache_dir=tmp_path / "cache")

    def test_pip_install_timeout_raises(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repos" / "demo"
        repo_dir.mkdir(parents=True)
        calls = {"n": 0}

        def boom(cmd, *a, **k):
            calls["n"] += 1
            if calls["n"] > 1:  # first is venv creation
                raise subprocess.TimeoutExpired(cmd, 999)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(tr.subprocess, "run", boom)
        with pytest.raises(subprocess.TimeoutExpired):
            tr._install_repo(repo_dir, cache_dir=tmp_path / "cache")


# ── _activate_venv ──────────────────────────────────────────────────────────


class TestActivateVenv:
    def test_prepends_venv_bin(self, tmp_path, monkeypatch):
        venv = tmp_path / "venv"
        venv.mkdir(parents=True)
        orig = os.environ.get("PATH", "")
        try:
            tr._activate_venv(venv)
            assert os.environ["PATH"].startswith(str(venv / "bin") + os.pathsep)
        finally:
            os.environ["PATH"] = orig


# ── munge / swebench_image ──────────────────────────────────────────────────


class TestMungeAndImage:
    def test_munge_double_underscore(self):
        assert tr.munge_instance_id("django__django-10554") == "django_1776_django-10554"

    def test_munge_no_op_without_double_underscore(self):
        assert tr.munge_instance_id("plain_id") == "plain_id"


# ── _get_volumes ────────────────────────────────────────────────────────────


class TestGetVolumes:
    def test_custom_volume_names(self, monkeypatch):
        import evaluation.config

        captured: list[str] = []

        def fake_from_name(name, **kwargs):
            captured.append(name)
            return object()

        class FakeCfg:
            modal_volumes = {"repo_cache": "vol-a", "test_cache": "vol-b"}

        monkeypatch.setattr(evaluation.config, "EvalConfig", FakeCfg)
        monkeypatch.setattr(tr.modal.Volume, "from_name", staticmethod(fake_from_name))
        tr._get_volumes()
        assert captured == ["vol-a", "vol-b"]

    def test_default_volume_names_when_keys_missing(self, monkeypatch):
        import evaluation.config

        captured: list[str] = []

        def fake_from_name(name, **kwargs):
            captured.append(name)
            return object()

        class FakeCfg:
            modal_volumes = {}

        monkeypatch.setattr(evaluation.config, "EvalConfig", FakeCfg)
        monkeypatch.setattr(tr.modal.Volume, "from_name", staticmethod(fake_from_name))
        tr._get_volumes()
        assert captured == ["eval-repo-cache", "eval-test-cache"]


# ── _execute_instance ───────────────────────────────────────────────────────


class TestExecuteInstance:
    def _setup(self, monkeypatch, tmp_path, *, f2p=1.0):
        import evaluation.metrics
        import evaluation.patch_applier

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)
        resets: list[str] = []
        monkeypatch.setattr(tr, "_reset_to_base", lambda d, s: resets.append(s))
        state = {"n": 0}

        def fake_collect(repo_path, names, timeout=30, max_retries=0, python_cmd=None):
            if max_retries == 0 and state["n"] == 0:
                state["n"] = 1
                return [TestResult(name="test_a", status="failed", duration=0.1)]
            return [TestResult(name="test_a", status="passed", duration=0.1)]

        monkeypatch.setattr(tr, "collect_test_results", fake_collect)
        monkeypatch.setattr(
            evaluation.metrics,
            "compute_f2p",
            lambda b, h, f, p: (f2p, 1.0, 1 if f2p else 0, 1),
        )

        def fake_apply(repo_path, patch, base_sha, skip_checkout=False):
            return _fake_apply(True)

        monkeypatch.setattr(evaluation.patch_applier, "apply_patch", fake_apply)
        return repo_dir, resets

    def test_success(self, tmp_path, monkeypatch):
        repo_dir, resets = self._setup(monkeypatch, tmp_path)
        result = tr._execute_instance(
            repo_dir,
            "sha",
            "--- a/t\n+++ b/t\n",
            "--- a/s\n+++ b/s\n",
            ["test_a"],
            [],
            timeout=30,
            max_retries=2,
        )
        assert result["repo"] == str(repo_dir)
        assert result["base_sha"] == "sha"
        assert result["error"] is None
        assert result["ground_truth"] == {"f2p": 1.0, "p2p": 1.0, "warning": False}
        assert len(result["tests_after"]) == 1
        assert result["tests_after"][0]["status"] == "passed"
        assert result["patch_application"]["success"] is True
        assert resets == ["sha", "sha"]

    def test_testbed_repo_name(self, tmp_path, monkeypatch):
        repo_dir = Path("/testbed")
        resets = []
        monkeypatch.setattr(tr, "_reset_to_base", lambda d, s: resets.append(s))
        state = {"n": 0}

        def fake_collect(*a, **k):
            state["n"] += 1
            if state["n"] == 1:
                return [TestResult(name="test_a", status="failed", duration=0.1)]
            return [TestResult(name="test_a", status="passed", duration=0.1)]

        monkeypatch.setattr(tr, "collect_test_results", fake_collect)
        import evaluation.metrics
        import evaluation.patch_applier

        monkeypatch.setattr(evaluation.metrics, "compute_f2p", lambda b, h, f, p: (1.0, 1.0, 1, 1))
        monkeypatch.setattr(
            evaluation.patch_applier, "apply_patch", lambda *a, **k: _fake_apply(True)
        )
        result = tr._execute_instance(
            repo_dir, "sha", "tp", "gp", ["test_a"], [], timeout=30, max_retries=2
        )
        assert result["repo"] == "testbed"

    def test_no_test_patch(self, tmp_path, monkeypatch):
        repo_dir, _ = self._setup(monkeypatch, tmp_path)
        result = tr._execute_instance(
            repo_dir, "sha", None, "gp", ["test_a"], [], timeout=30, max_retries=2
        )
        assert result["tests_head"] == []
        assert result["ground_truth"] == {}
        assert result["error"] is None
        assert len(result["tests_after"]) == 1

    def test_gold_patch_flow_ordering(self, tmp_path, monkeypatch):
        import evaluation.patch_applier

        repo_dir, resets = self._setup(monkeypatch, tmp_path)
        applied: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            evaluation.patch_applier,
            "apply_patch",
            lambda repo_path, patch, base_sha, skip_checkout=False: (
                applied.append((patch, skip_checkout)) or _fake_apply(True)
            ),
        )
        result = tr._execute_instance(
            repo_dir,
            "sha",
            "test_patch",
            "generated_patch",
            ["test_a"],
            [],
            timeout=30,
            max_retries=2,
            gold_patch="gold_patch",
        )
        # order: test_patch → collect before → gold_patch → collect head →
        # reset → test_patch → generated → collect after
        assert applied == [
            ("test_patch", True),
            ("gold_patch", True),
            ("test_patch", True),
            ("generated_patch", True),
        ]
        assert resets == ["sha", "sha"]
        assert result["ground_truth"] == {"f2p": 1.0, "p2p": 1.0, "warning": False}
        assert len(result["tests_before"]) == 1
        assert len(result["tests_head"]) == 1
        assert result["tests_after"][0]["status"] == "passed"

    def test_test_patch_apply_failed(self, tmp_path, monkeypatch):
        import evaluation.patch_applier

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)
        resets = []
        monkeypatch.setattr(tr, "_reset_to_base", lambda d, s: resets.append(s))
        monkeypatch.setattr(
            tr,
            "collect_test_results",
            lambda *a, **k: [TestResult(name="test_a", status="failed", duration=0.1)],
        )
        import evaluation.metrics

        monkeypatch.setattr(evaluation.metrics, "compute_f2p", lambda *a: (0.0, 1.0, 0, 1))
        monkeypatch.setattr(
            evaluation.patch_applier,
            "apply_patch",
            lambda *a, **k: _fake_apply(False, "failed", "apply boom"),
        )
        result = tr._execute_instance(
            repo_dir, "sha", "tp", "gp", ["test_a"], [], timeout=30, max_retries=2
        )
        assert result["tests_head"] == []
        assert result["ground_truth"] == {}
        # patch_failure semantics: tests_after reports errored "patch did not
        # apply" entries instead of running pytest on a broken tree
        assert result["tests_after"] == [
            {
                "name": "test_a",
                "status": "errored",
                "duration": 0.0,
                "output": "patch did not apply",
                "retry_count": 0,
            }
        ]

    def test_ground_truth_f2p_below_threshold(self, tmp_path, monkeypatch):
        repo_dir, _ = self._setup(monkeypatch, tmp_path, f2p=0.5)
        result = tr._execute_instance(
            repo_dir, "sha", "tp", "gp", ["test_a"], [], timeout=30, max_retries=2
        )
        assert result["error"] == "ground truth F2P<100% (env drift or missing image?)"
        assert result["tests_after"] == []
        assert result["ground_truth"]["warning"] is True
        assert result["patch_application"] == {}

    def test_no_generated_patch(self, tmp_path, monkeypatch):
        repo_dir, _ = self._setup(monkeypatch, tmp_path)
        result = tr._execute_instance(
            repo_dir, "sha", "tp", None, ["test_a"], [], timeout=30, max_retries=2
        )
        assert result["tests_after"] == []
        assert result["patch_application"] == {
            "success": False,
            "method_used": "failed",
            "error": "no generated patch provided",
            "files_modified": [],
        }

    def test_generated_patch_apply_failed(self, tmp_path, monkeypatch):
        import evaluation.patch_applier

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)
        resets = []
        monkeypatch.setattr(tr, "_reset_to_base", lambda d, s: resets.append(s))
        monkeypatch.setattr(
            tr,
            "collect_test_results",
            lambda *a, **k: [TestResult(name="test_a", status="failed", duration=0.1)],
        )
        import evaluation.metrics

        monkeypatch.setattr(evaluation.metrics, "compute_f2p", lambda *a: (1.0, 1.0, 1, 1))
        monkeypatch.setattr(
            evaluation.patch_applier,
            "apply_patch",
            lambda *a, **k: _fake_apply(False, "failed", "generated boom"),
        )
        result = tr._execute_instance(
            repo_dir, "sha", "tp", "gp", ["test_a"], [], timeout=30, max_retries=2
        )
        # patch_failure semantics: tests_after reports errored "patch did not
        # apply" entries instead of running pytest on a broken tree
        assert result["tests_after"] == [
            {
                "name": "test_a",
                "status": "errored",
                "duration": 0.0,
                "output": "patch did not apply",
                "retry_count": 0,
            }
        ]
        assert result["patch_application"]["error"] == "generated boom"

    def test_reset_first_false(self, tmp_path, monkeypatch):
        repo_dir, resets = self._setup(monkeypatch, tmp_path)
        result = tr._execute_instance(
            repo_dir,
            "sha",
            "tp",
            "gp",
            ["test_a"],
            [],
            timeout=30,
            max_retries=2,
            reset_first=False,
        )
        assert result["error"] is None
        assert resets == ["sha"]


# ── run_tests_in_container (via .local()) ───────────────────────────────────


class TestRunTestsInContainer:
    def _setup(self, monkeypatch, *, f2p=1.0, prep_ok=True, apply_ok=True):
        import evaluation.metrics
        import evaluation.patch_applier

        monkeypatch.setattr(tr, "_clone_repo", lambda *a, **k: None)
        monkeypatch.setattr(tr, "_ensure_checked_out", lambda *a, **k: None)
        monkeypatch.setattr(tr, "_install_repo", lambda *a, **k: None)
        monkeypatch.setattr(tr, "_reset_to_base", lambda *a, **k: None)

        state = {"n": 0}

        def fake_collect(repo_path, names, timeout=30, max_retries=0, **kw):
            if max_retries == 0 and state["n"] == 0:
                state["n"] = 1
                return [TestResult(name="t", status="failed", duration=0.1)]
            return [TestResult(name="t", status="passed", duration=0.1)]

        monkeypatch.setattr(tr, "collect_test_results", fake_collect)
        monkeypatch.setattr(evaluation.metrics, "compute_f2p", lambda b, h, f, p: (f2p, 1.0, 1, 1))

        if not prep_ok:

            def bad_clone(*a, **k):
                raise RuntimeError("clone failed")

            monkeypatch.setattr(tr, "_clone_repo", bad_clone)

        def fake_apply(repo_path, patch, base_sha, skip_checkout=False):
            return _fake_apply(
                apply_ok,
                "failed" if not apply_ok else "git_apply",
                None if apply_ok else "apply boom",
            )

        monkeypatch.setattr(evaluation.patch_applier, "apply_patch", fake_apply)

    def test_happy_path(self, monkeypatch):
        self._setup(monkeypatch)
        result = tr.run_tests_in_container.local(
            "o/r",
            "sha",
            test_patch="tp",
            generated_patch="gp",
            fail_to_pass=["t"],
            pass_to_pass=[],
        )
        assert result["error"] is None
        assert result["ground_truth"] == {"f2p": 1.0, "p2p": 1.0, "warning": False}
        assert len(result["tests_after"]) == 1
        assert result["repo"].endswith("o/r")

    def test_repo_prep_failure(self, monkeypatch):
        self._setup(monkeypatch, prep_ok=False)
        result = tr.run_tests_in_container.local(
            "o/r", "sha", test_patch="tp", generated_patch="gp", fail_to_pass=["t"]
        )
        assert result["error"] == "clone failed"
        assert result["tests_before"] == []
        assert result["ground_truth"] == {}

    def test_no_test_patch(self, monkeypatch):
        self._setup(monkeypatch)
        result = tr.run_tests_in_container.local(
            "o/r", "sha", test_patch=None, generated_patch="gp", fail_to_pass=["t"]
        )
        assert result["ground_truth"] == {}
        assert len(result["tests_after"]) == 1

    def test_ground_truth_f2p_below_threshold(self, monkeypatch):
        self._setup(monkeypatch, f2p=0.5)
        result = tr.run_tests_in_container.local(
            "o/r", "sha", test_patch="tp", generated_patch="gp", fail_to_pass=["t"]
        )
        assert result["error"] == "ground truth F2P<100% (env drift or missing image?)"
        assert result["tests_after"] == []

    def test_no_generated_patch(self, monkeypatch):
        self._setup(monkeypatch)
        result = tr.run_tests_in_container.local(
            "o/r", "sha", test_patch="tp", generated_patch=None, fail_to_pass=["t"]
        )
        assert result["patch_application"]["error"] == "no generated patch provided"
        assert result["tests_after"] == []

    def test_generated_patch_apply_failed(self, monkeypatch):
        self._setup(monkeypatch, apply_ok=False)
        result = tr.run_tests_in_container.local(
            "o/r", "sha", test_patch="tp", generated_patch="gp", fail_to_pass=["t"]
        )
        # patch_failure semantics: tests_after reports errored "patch did not
        # apply" entries instead of running pytest on a broken tree
        assert result["tests_after"] == [
            {
                "name": "t",
                "status": "errored",
                "duration": 0.0,
                "output": "patch did not apply",
                "retry_count": 0,
            }
        ]
        assert result["patch_application"]["error"] == "apply boom"


# ── _run_swebench_instance_body ─────────────────────────────────────────────


class TestSwebenchInstanceBody:
    def test_probe_success(self, monkeypatch):
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(tr.subprocess, "run", fake_run)
        captured = {}

        def fake_exec(*a, **k):
            captured["args"] = a
            captured["kwargs"] = k
            return {"repo": "/testbed", "base_sha": a[1], "error": None}

        monkeypatch.setattr(tr, "_execute_instance", fake_exec)
        result = tr._run_swebench_instance_body(
            "django__django-10554",
            "sha",
            "tp",
            "gold",
            "gp",
            ["t"],
            [],
            timeout=30,
            max_retries=2,
        )
        assert len(calls) == 1  # only the probe
        assert result["repo"] == "django__django-10554"
        assert captured["kwargs"]["python_cmd"] == ["conda", "run", "-n", "testbed", "python"]
        assert captured["kwargs"]["gold_patch"] == "gold"

    def test_probe_failure_installs_plugins(self, monkeypatch):
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(list(cmd))
            if "import pytest_jsonreport" in " ".join(cmd):
                return subprocess.CompletedProcess(cmd, 1, "", "not installed")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(tr.subprocess, "run", fake_run)
        captured_kwargs: dict = {}

        def fake_exec(*a, **k):
            captured_kwargs.update(k)
            return {"repo": "/testbed", "base_sha": a[1], "error": None}

        monkeypatch.setattr(tr, "_execute_instance", fake_exec)
        result = tr._run_swebench_instance_body("inst", "sha", "tp", "gold", "gp", ["t"], [])
        assert len(calls) == 2
        assert "pip" in calls[1] and "install" in calls[1]
        assert result["error"] is None
        assert captured_kwargs["gold_patch"] == "gold"

    def test_plugin_setup_timeout_error(self, monkeypatch):
        def boom(cmd, *a, **k):
            raise subprocess.TimeoutExpired(cmd, 120)

        monkeypatch.setattr(tr.subprocess, "run", boom)
        result = tr._run_swebench_instance_body("inst", "sha")
        assert result["error"].startswith("plugin setup failed:")
        assert result["tests_before"] == []

    def test_plugin_setup_pip_failure_error(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(cmd, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return subprocess.CompletedProcess(cmd, 1, "", "")
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(tr.subprocess, "run", fake_run)
        result = tr._run_swebench_instance_body("inst", "sha")
        assert result["error"].startswith("plugin setup failed:")


# ── swebench_fn ─────────────────────────────────────────────────────────────


class TestSwebenchFn:
    def test_caches_one_function_per_repo(self, monkeypatch):
        monkeypatch.setattr(tr, "_swebench_fns", {})
        fn1 = tr.swebench_fn("django/django", "django__django-10554")
        fn2 = tr.swebench_fn("django/django", "django__django-10555")
        assert fn1 is fn2
        assert set(tr._swebench_fns) == {"django/django"}

    def test_new_repo_registers_new_function(self, monkeypatch):
        monkeypatch.setattr(tr, "_swebench_fns", {})
        fn1 = tr.swebench_fn("django/django", "django__django-10554")
        fn2 = tr.swebench_fn("sympy/sympy", "sympy__sympy-100")
        assert fn1 is not fn2


# ── run_tests_batch (via .local()) ──────────────────────────────────────────


class TestRunTestsBatch:
    def _setup(self, monkeypatch, *, prep_ok=True, f2p=1.0):
        import evaluation.metrics
        import evaluation.patch_applier

        monkeypatch.setattr(tr, "_clone_repo", lambda *a, **k: None)
        monkeypatch.setattr(tr, "_ensure_checked_out", lambda *a, **k: None)
        monkeypatch.setattr(tr, "_install_repo", lambda *a, **k: None)
        monkeypatch.setattr(tr, "_reset_to_base", lambda *a, **k: None)

        def fake_collect(repo_path, names, timeout=30, max_retries=0, **kw):
            status = "failed" if max_retries == 0 else "passed"
            return [TestResult(name=n, status=status, duration=0.1) for n in names]

        monkeypatch.setattr(tr, "collect_test_results", fake_collect)
        monkeypatch.setattr(evaluation.metrics, "compute_f2p", lambda b, h, f, p: (f2p, 1.0, 1, 1))
        if not prep_ok:

            def bad_clone(*a, **k):
                raise RuntimeError("prep boom")

            monkeypatch.setattr(tr, "_clone_repo", bad_clone)

        def fake_apply(repo_path, patch, base_sha, skip_checkout=False):
            return _fake_apply(True)

        monkeypatch.setattr(evaluation.patch_applier, "apply_patch", fake_apply)

    def test_prep_failure_returns_error_per_job(self, monkeypatch):
        self._setup(monkeypatch, prep_ok=False)
        results = tr.run_tests_batch.local(
            "o/r", "sha", "tp", [{"generated_patch": "g1"}, {"generated_patch": "g2"}]
        )
        assert len(results) == 2
        assert all(r["error"] == "prep boom" for r in results)

    def test_shared_test_patch_fast_path(self, monkeypatch):
        self._setup(monkeypatch)
        jobs = [
            {"test_patch": "tp", "generated_patch": "g1", "fail_to_pass": ["t1"]},
            {"test_patch": "tp", "generated_patch": "g2", "fail_to_pass": ["t1"]},
        ]
        results = tr.run_tests_batch.local("o/r", "sha", "tp", jobs)
        assert len(results) == 2
        assert all(r.get("error") is None for r in results)
        assert all(r["ground_truth"] == {"f2p": 1.0, "p2p": 1.0, "warning": False} for r in results)
        assert all(len(r["tests_after"]) == 1 for r in results)

    def test_distinct_test_patches_per_job(self, monkeypatch):
        import evaluation.patch_applier

        self._setup(monkeypatch)
        applied = []
        monkeypatch.setattr(
            evaluation.patch_applier,
            "apply_patch",
            lambda repo_path, patch, base_sha, skip_checkout=False: (
                applied.append(patch) or _fake_apply(True)
            ),
        )
        jobs = [
            {"test_patch": "tpA", "generated_patch": "g1", "fail_to_pass": ["t1"]},
            {"test_patch": "tpB", "generated_patch": "g2", "fail_to_pass": ["t1"]},
        ]
        results = tr.run_tests_batch.local("o/r", "sha", "tp", jobs)
        assert len(results) == 2
        assert "tpA" in applied and "tpB" in applied
        assert all(r["ground_truth"]["f2p"] == 1.0 for r in results)

    def test_per_job_gold_patch_flow(self, monkeypatch):
        import evaluation.patch_applier

        self._setup(monkeypatch)
        applied: list[str] = []
        monkeypatch.setattr(
            evaluation.patch_applier,
            "apply_patch",
            lambda repo_path, patch, base_sha, skip_checkout=False: (
                applied.append(patch) or _fake_apply(True)
            ),
        )
        jobs = [
            {
                "test_patch": "tpA",
                "gold_patch": "gA",
                "generated_patch": "genA",
                "fail_to_pass": ["t1"],
            },
            {
                "test_patch": "tpB",
                "gold_patch": "gB",
                "generated_patch": "genB",
                "fail_to_pass": ["t1"],
            },
        ]
        results = tr.run_tests_batch.local("o/r", "sha", "tp", jobs)
        assert len(results) == 2
        # per job: test_patch → gold_patch (head) → reset → test_patch → generated (after)
        assert applied == ["tpA", "gA", "tpA", "genA", "tpB", "gB", "tpB", "genB"]
        assert all(r["ground_truth"]["f2p"] == 1.0 for r in results)
        assert all(len(r["tests_head"]) == 1 for r in results)
        assert all(len(r["tests_after"]) == 1 for r in results)

    def test_job_ground_truth_f2p_below_threshold(self, monkeypatch):
        self._setup(monkeypatch, f2p=0.0)
        jobs = [{"test_patch": "tp", "generated_patch": "g1", "fail_to_pass": ["t1"]}]
        results = tr.run_tests_batch.local("o/r", "sha", "tp", jobs)
        assert results[0]["error"] == "ground truth F2P<100% (env drift or install incomplete?)"
        assert results[0]["tests_after"] == []

    def test_no_generated_patch(self, monkeypatch):
        self._setup(monkeypatch)
        jobs = [{"test_patch": "tp", "fail_to_pass": ["t1"]}]
        results = tr.run_tests_batch.local("o/r", "sha", "tp", jobs)
        assert results[0]["patch_application"]["error"] == "no generated patch provided"
        assert results[0]["tests_after"] == []

    def test_truncation_on_timeout(self, monkeypatch):
        self._setup(monkeypatch)
        calls = {"n": 0}

        def fake_time():
            calls["n"] += 1
            return 5000.0 if calls["n"] > 1 else 0.0

        monkeypatch.setattr(time, "time", fake_time)
        jobs = [{"generated_patch": f"g{i}", "fail_to_pass": ["t1"]} for i in range(3)]
        results = tr.run_tests_batch.local("o/r", "sha", "tp", jobs)
        assert len(results) == 3
        assert results[0]["error"] == "timeout approaching, truncated"
        assert all(r["error"] == "timeout approaching, truncated" for r in results)
