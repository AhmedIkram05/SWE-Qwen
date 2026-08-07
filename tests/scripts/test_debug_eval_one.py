"""Unit tests for ``scripts/debug_eval_one.py``.

git/pip subprocess calls and every pipeline helper are mocked; the golden
file lives under a patched cwd, not the real repo.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _load():
    from scripts import debug_eval_one

    return debug_eval_one


@pytest.fixture
def mod():
    return _load()


_RECORD = {
    "instance_id": "sympy__sympy-200",
    "repo": "sympy/sympy",
    "problem_statement": "Fix the bug.",
    "base_commit": "0123456789abcdef0123456789abcdef01234567",
    "environment_setup_commit": "fedcba9876543210fedcba9876543210fedcba98",
    "test_patch": "--- a/tests/t.py\n+++ b/tests/t.py\n",
    "FAIL_TO_PASS": "test_fix",
    "PASS_TO_PASS": "test_a test_b",
}

_RECORD_NO_PATCH = {
    "instance_id": "django__django-9",
    "repo": "django/django",
    **{
        k: v
        for k, v in _RECORD.items()
        if k in ("problem_statement", "base_commit", "environment_setup_commit")
    },
}

_DECOY = {
    "instance_id": "django__django-1",
    "repo": "django/django",
    "problem_statement": "Fix it.",
    "base_commit": "0123456789abcdef0123456789abcdef01234567",
    "environment_setup_commit": "fedcba9876543210fedcba9876543210fedcba98",
}


def _setup(mod, mocker, tmp_path, monkeypatch, records, args):
    monkeypatch.chdir(tmp_path)
    golden_dir = tmp_path / "data" / "expanded-repos" / "swebench"
    golden_dir.mkdir(parents=True)
    (golden_dir / "golden.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    mocker.patch("sys.argv", ["debug_eval_one", *args])
    mocker.patch.object(mod, "subprocess")
    mocker.patch.object(mod, "_install_repo")
    mocker.patch.object(
        mod,
        "collect_test_results",
        side_effect=[
            [SimpleNamespace(name="test_fix", status="failed")],
            [SimpleNamespace(name="test_fix", status="passed")],
        ],
    )
    mocker.patch.object(
        mod, "apply_patch", return_value=SimpleNamespace(method_used="git", success=True)
    )


class TestMain:
    def test_default_sympy(self, mod, mocker, tmp_path, monkeypatch):
        # decoy non-sympy line forces the loop to continue before the match
        _setup(mod, mocker, tmp_path, monkeypatch, [_DECOY, _RECORD], [])
        mocker.patch.object(mod, "compute_f2p", return_value=(1.0, 1.0, 1, 2))
        mod.main()
        assert mod.collect_test_results.call_count == 2
        mod.apply_patch.assert_called_once()
        mod.compute_f2p.assert_called_once()
        mod._install_repo.assert_called_once()

    def test_sympy_arg(self, mod, mocker, tmp_path, monkeypatch):
        _setup(mod, mocker, tmp_path, monkeypatch, [_DECOY, _RECORD], ["SYMPY"])
        mocker.patch.object(mod, "compute_f2p", return_value=(1.0, 1.0, 1, 2))
        mod.main()
        assert mod.collect_test_results.call_count == 2

    def test_int_arg(self, mod, mocker, tmp_path, monkeypatch):
        _setup(mod, mocker, tmp_path, monkeypatch, [_RECORD], ["0"])
        mocker.patch.object(mod, "compute_f2p", return_value=(0.0, 1.0, 0, 1))
        mod.main()
        assert mod.collect_test_results.call_count == 2
        mod.compute_f2p.assert_called_once()

    def test_no_test_patch(self, mod, mocker, tmp_path, monkeypatch):
        _setup(mod, mocker, tmp_path, monkeypatch, [_RECORD_NO_PATCH], ["0"])
        mocker.patch.object(mod, "compute_f2p")
        mod.main()
        mod.collect_test_results.assert_not_called()
        mod.apply_patch.assert_not_called()
        mod.compute_f2p.assert_not_called()

    def test_f2p_zero(self, mod, mocker, tmp_path, monkeypatch):
        _setup(mod, mocker, tmp_path, monkeypatch, [_RECORD], ["0"])
        mocker.patch.object(mod, "compute_f2p", return_value=(0.0, 1.0, 0, 1))
        mod.main()
        mod.compute_f2p.assert_called_once()
