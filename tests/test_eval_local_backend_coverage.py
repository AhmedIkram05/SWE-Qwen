"""Coverage tests for ``evaluation.local_backend``.

Inference path mocks the Ollama HTTP client; test-runner path mocks
``apply_patch`` / ``collect_test_results``; repo-prep path exercises real git
on tiny local repos. No network, no model servers.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from evaluation import schema as eschema
from evaluation.config import EvalConfig
from evaluation.local_backend import (
    _ensure_local_repo,
    _error_response,
    generate_patches_local,
    run_tests_local,
)
from evaluation.schema import EvalInput, PatchApplicationResult

# ── helpers / fixtures ─────────────────────────────────────────────────────


def _example(**overrides: Any) -> EvalInput:
    base: dict[str, Any] = {
        "instance_id": "django__django-10554",
        "repo": "django/django",
        "issue_body": "BooleanField crashes on None.",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "test_patch": "",
        "fail_to_pass": ["tests/test_models.py::test_x"],
        "pass_to_pass": ["tests/test_models.py::test_y"],
        "repo_domain": "python",
    }
    base.update(overrides)
    return EvalInput(**base)


def _cfg(**overrides: Any) -> EvalConfig:
    return EvalConfig().model_copy(
        update={"test_timeout_seconds": 11, "max_retries": 3, **overrides}
    )


@pytest.fixture
def fake_httpx(monkeypatch) -> dict[str, Any]:
    """Install a fake ``httpx`` module; scripts the responses for ``post``."""
    state: dict[str, Any] = {"posts": [], "responses": [], "fail_mode": None}

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            if state["fail_mode"] == "raise_for_status":
                raise RuntimeError("HTTP 500")

        def json(self):
            return self._payload

    def post(url, json, timeout):
        state["posts"].append((url, json, timeout))
        if state["fail_mode"] == "post":
            raise OSError("connection refused")
        if state["fail_mode"] == "bad_json":
            return FakeResp({})
        payload = state["responses"].pop(0)
        return FakeResp(payload)

    fake = types.ModuleType("httpx")

    # generate_patches_local lazily pulls peft/transformers/huggingface_hub
    # (which import httpx.HTTPError/Response/Client at import time). Delegate
    # anything the fake doesn't define back to the real httpx so those imports
    # still work while `post` below stays scripted.
    def _delegate(name: str):
        return getattr(httpx, name)

    fake.__getattr__ = _delegate
    fake.post = post
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return state


def _chat_payload(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _fake_ensure_repo(state: dict[str, Any]):
    """Repo-prep stand-in that raises scripted errors (or succeeds)."""

    def ensure(repo, repo_dir, base_sha):
        if state["prep_errors"]:
            raise state["prep_errors"].pop(0)

    return ensure


# ── generate_patches_local ─────────────────────────────────────────────────


class TestGeneratePatchesLocal:
    def test_happy_path_orders_patches(self, fake_httpx):
        fake_httpx["responses"] = [
            _chat_payload("```patch\ndiff --git a/one.py b/one.py\n@@\n```"),
            _chat_payload("```diff\ndiff --git a/two.py b/two.py\n@@\n```"),
        ]
        out = generate_patches_local(
            "qwen3-14b",
            "baseline_14b",
            "chat",
            [_example(), _example(instance_id="inst-2")],
            ollama_model="qwen2.5-coder:7b",
            ollama_base_url="http://localhost:11434/",
            max_tokens=512,
            temperature=0.3,
            top_p=0.8,
        )
        assert out == ["diff --git a/one.py b/one.py\n@@", "diff --git a/two.py b/two.py\n@@"]

        assert len(fake_httpx["posts"]) == 2
        url, body, timeout = fake_httpx["posts"][0]
        assert url == "http://localhost:11434/v1/chat/completions"
        assert body["model"] == "qwen2.5-coder:7b"
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert "django__django-10554" in body["messages"][1]["content"]
        assert body["max_tokens"] == 512
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.8
        assert body["stream"] is False
        assert timeout == 600

    def test_post_failure_yields_empty_patch(self, fake_httpx, caplog):
        fake_httpx["fail_mode"] = "post"
        with caplog.at_level("WARNING", logger="evaluation.local_backend"):
            out = generate_patches_local("m", "v", "chat", [_example()])
        assert out == [""]
        assert "Ollama inference failed" in caplog.text

    def test_raise_for_status_failure_yields_empty_patch(self, fake_httpx):
        fake_httpx["fail_mode"] = "raise_for_status"
        assert generate_patches_local("m", "v", "chat", [_example()]) == [""]

    def test_malformed_json_yields_empty_patch(self, fake_httpx, caplog):
        fake_httpx["fail_mode"] = "bad_json"
        with caplog.at_level("WARNING", logger="evaluation.local_backend"):
            out = generate_patches_local("m", "v", "chat", [_example()])
        assert out == [""]

    def test_empty_examples(self, fake_httpx):
        assert generate_patches_local("m", "v", "chat", []) == []
        assert fake_httpx["posts"] == []


# ── run_tests_local ────────────────────────────────────────────────────────


@pytest.fixture
def _patched_runner(monkeypatch) -> dict[str, Any]:
    """No-op repo prep + scripted apply_patch / collect_test_results."""
    import evaluation.local_backend as lb
    import evaluation.patch_applier as pa
    import evaluation.test_runner as tr

    state: dict[str, Any] = {
        "apply_results": [],
        "collect_calls": [],
        "collect_results": [],
        "prep_errors": [],
    }

    monkeypatch.setattr(lb, "_ensure_local_repo", _fake_ensure_repo(state))

    def fake_apply(repo_path, patch, base_sha, *, skip_checkout=False):
        state["apply_calls"] = (str(repo_path), patch, base_sha, skip_checkout)
        return state["apply_results"].pop(0)

    def fake_collect(repo_path, test_names, timeout=30, max_retries=2, python_cmd=None):
        state["collect_calls"].append((str(repo_path), list(test_names), timeout, max_retries))
        return state["collect_results"].pop(0)

    monkeypatch.setattr(pa, "apply_patch", fake_apply)
    monkeypatch.setattr(tr, "collect_test_results", fake_collect)
    monkeypatch.setattr(tr, "_reset_to_base", lambda repo_dir, base_sha: None)
    return state


class TestRunTestsLocal:
    def test_happy_path(self, _patched_runner):
        ex = _example()
        _patched_runner["collect_results"] = [
            [eschema.TestResult(name="t1", status="failed", duration=0.5)],
            [eschema.TestResult(name="t1", status="passed", duration=0.4)],
        ]
        _patched_runner["apply_results"] = [
            PatchApplicationResult(success=True, method_used="git_apply", error=None)
        ]
        out = run_tests_local(ex, "DIFF", _cfg())

        assert out["repo"] == "django/django"
        assert out["base_sha"] == ex.base_sha
        assert out["tests_before"] == [
            {"name": "t1", "status": "failed", "duration": 0.5, "output": "", "retry_count": 0}
        ]
        assert out["tests_after"] == [
            {"name": "t1", "status": "passed", "duration": 0.4, "output": "", "retry_count": 0}
        ]
        assert out["tests_head"] == []
        assert out["ground_truth"] == {}
        assert out["patch_application"]["success"] is True
        assert out["patch_application"]["method_used"] == "git_apply"

        # test names = fail_to_pass + pass_to_pass; timeouts/retries from config
        assert _patched_runner["collect_calls"][0][1] == [
            "tests/test_models.py::test_x",
            "tests/test_models.py::test_y",
        ]
        assert _patched_runner["collect_calls"][0][2] == 11
        assert _patched_runner["collect_calls"][0][3] == 3

    def test_test_patch_and_gold_flow(self, _patched_runner):
        ex = _example(
            test_patch="diff --git a/tests/test_models.py b/tests/test_models.py",
            gold_patch="diff --git a/models.py b/models.py",
        )
        _patched_runner["collect_results"] = [
            [
                eschema.TestResult(
                    name="tests/test_models.py::test_x", status="failed", duration=0.5
                )
            ],
            [
                eschema.TestResult(
                    name="tests/test_models.py::test_x", status="passed", duration=0.4
                )
            ],
            [
                eschema.TestResult(
                    name="tests/test_models.py::test_x", status="passed", duration=0.4
                )
            ],
        ]
        _patched_runner["apply_results"] = [
            PatchApplicationResult(success=True, method_used="git_apply", error=None),
            PatchApplicationResult(success=True, method_used="git_apply", error=None),
            PatchApplicationResult(success=True, method_used="git_apply", error=None),
            PatchApplicationResult(success=True, method_used="git_apply", error=None),
        ]
        out = run_tests_local(ex, "DIFF", _cfg())
        assert out["ground_truth"] == {"f2p": 1.0, "p2p": 1.0, "warning": False}
        assert len(out["tests_head"]) == 1
        assert len(out["tests_after"]) == 1
        # applies: test_patch → gold_patch → generated_patch (all skip_checkout)
        assert _patched_runner["apply_calls"][3] is True

    def test_patch_apply_failure_skips_after(self, _patched_runner, caplog):
        _patched_runner["collect_results"] = [
            [eschema.TestResult(name="t1", status="passed", duration=0.1)]
        ]
        _patched_runner["apply_results"] = [
            PatchApplicationResult(success=False, method_used="failed", error="apply boom")
        ]
        with caplog.at_level("WARNING", logger="evaluation.local_backend"):
            out = run_tests_local(_example(), "DIFF", _cfg())
        assert out["tests_after"] == []
        assert out["patch_application"] == {
            "success": False,
            "method_used": "failed",
            "error": "apply boom",
            "files_modified": [],
        }
        assert "local patch apply failed" in caplog.text
        assert len(_patched_runner["collect_calls"]) == 1  # only tests_before

    def test_no_patch_uses_no_patch_result(self, _patched_runner):
        _patched_runner["collect_results"] = [
            [eschema.TestResult(name="t1", status="passed", duration=0.1)]
        ]
        out = run_tests_local(_example(), "", _cfg())
        assert out["tests_after"] == []
        assert out["patch_application"] == {
            "success": False,
            "method_used": "failed",
            "error": "no patch",
            "files_modified": [],
        }

    def test_repo_prep_runtime_error(self, _patched_runner):
        _patched_runner["prep_errors"].append(RuntimeError("network down"))
        out = run_tests_local(_example(), "DIFF", _cfg())
        assert out["error"] == "repo prep: network down"
        assert out["tests_before"] == []
        assert out["patch_application"]["success"] is False

    def test_repo_prep_timeout(self, _patched_runner):
        _patched_runner["prep_errors"].append(subprocess.TimeoutExpired("git", 300))
        out = run_tests_local(_example(), "DIFF", _cfg())
        assert out["error"].startswith("repo prep:")


# ── _error_response ────────────────────────────────────────────────────────


class TestErrorResponse:
    def test_shape(self):
        out = _error_response(_example(), "boom")
        assert out == {
            "repo": "django/django",
            "base_sha": "a" * 40,
            "error": "boom",
            "tests_before": [],
            "tests_head": [],
            "tests_after": [],
            "patch_application": {"success": False, "method_used": "failed", "error": "boom"},
            "ground_truth": {},
        }


# ── _ensure_local_repo ─────────────────────────────────────────────────────


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _make_repo_with_commit(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    _git(repo_dir, "config", "user.email", "t@t")
    _git(repo_dir, "config", "user.name", "t")
    (repo_dir / "f.txt").write_text("x")
    _git(repo_dir, "add", ".")
    _git(repo_dir, "commit", "-qm", "init")
    return _git(repo_dir, "rev-parse", "HEAD").stdout.strip()


class TestEnsureLocalRepo:
    def test_cached_repo_with_sha_no_network(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "cached"
        sha = _make_repo_with_commit(repo_dir)
        seen: list[list[str]] = []
        real_run = subprocess.run

        def spy(cmd, *a, **kw):
            seen.append(cmd)
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(subprocess, "run", spy)
        _ensure_local_repo("dummy/repo", repo_dir, sha)
        cmds = [" ".join(c) for c in seen]
        assert any("cat-file" in c for c in cmds)
        assert not any("clone" in c for c in cmds)
        assert not any("fetch" in c for c in cmds)
        assert any("checkout" in c for c in cmds)
        assert _git(repo_dir, "rev-parse", "HEAD").stdout.strip() == sha

    def test_clone_branch_uses_local_origin(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        sha = _make_repo_with_commit(src)
        repo_dir = tmp_path / "fresh"

        real_run = subprocess.run

        def clone_local(cmd, *a, **kw):
            if len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "clone":
                cmd = [c if not c.startswith("https://github.com/") else str(src) for c in cmd]
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(subprocess, "run", clone_local)
        _ensure_local_repo("dummy/repo", repo_dir, sha)
        assert (repo_dir / ".git").is_dir()
        assert _git(repo_dir, "rev-parse", "HEAD").stdout.strip() == sha

    def test_clone_failure_raises_called_process_error(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "fresh"

        def fail_clone(cmd, *a, **kw):
            raise subprocess.CalledProcessError(1, cmd, stderr="cannot clone")

        monkeypatch.setattr(subprocess, "run", fail_clone)
        with pytest.raises(subprocess.CalledProcessError):
            _ensure_local_repo("dummy/repo", repo_dir, "deadbeef" * 5)

    def test_missing_sha_triggers_fetch_then_checkout(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "cached2"
        _make_repo_with_commit(repo_dir)
        seen: list[list[str]] = []
        real_run = subprocess.run
        missing_sha = "cafebabe" * 5

        def spy(cmd, *a, **kw):
            seen.append(cmd)
            if len(cmd) >= 3 and cmd[0] == "git" and cmd[1] == "fetch":
                return subprocess.CompletedProcess(cmd, 0)
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(subprocess, "run", spy)
        # fetch is mocked as no-op, so checkout of the missing sha fails → raises.
        with pytest.raises(subprocess.CalledProcessError):
            _ensure_local_repo("dummy/repo", repo_dir, missing_sha)
        cmds = [" ".join(c) for c in seen]
        assert any("cat-file" in c for c in cmds)
        assert any(missing_sha in c for c in cmds if "fetch" in c)
