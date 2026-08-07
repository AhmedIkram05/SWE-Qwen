"""Unit tests for ``scripts/validate_golden.py`` (no network / mocked inputs)."""

from __future__ import annotations

import json

import pytest


def _load():
    from scripts import validate_golden

    return validate_golden


# A record EvalInput can load *directly* via model_validate.
_DIRECT = {
    "instance_id": "sympy__sympy-1",
    "repo": "sympy/sympy",
    "issue_body": "body",
    "base_sha": "0123456789abcdef0123456789abcdef01234567",
    "head_sha": "0123456789abcdef0123456789abcdef01234567",
    "test_patch": "diff --git a/tests/t.py b/tests/t.py",
    "fail_to_pass": ["test_a"],
    "pass_to_pass": ["test_b"],
    "repo_domain": "data-ml",
}

# An IssueRecord-shaped line: fails model_validate, loads via from_swebench_record.
_RECORD = {
    "issue_id": "sympy__sympy-2",
    "repo": "sympy/sympy",
    "issue_body": "body",
    "patch_diff": "--- a/x.py\n+++ b/x.py\n",
    "test_results": {"failed": ["test_c"], "passed": ["test_d"], "errored": []},
    "repo_domain": "data-ml",
    "files_changed": ["x.py"],
    "test_files_changed": ["tests/t.py"],
    "metadata": {
        "base_sha": "0123456789abcdef0123456789abcdef01234567",
        "head_sha": "0123456789abcdef0123456789abcdef01234567",
        "test_patch": "diff --git a/tests/t.py b/tests/t.py",
        "instance_id": "sympy__sympy-2",
    },
}


class TestValidate:
    def test_valid_direct(self, tmp_path):
        p = tmp_path / "golden.jsonl"
        p.write_text(json.dumps(_DIRECT) + "\n")
        assert _load().validate(str(p)) == 0

    def test_swebench_record_with_blank_line(self, tmp_path):
        p = tmp_path / "golden.jsonl"
        p.write_text(json.dumps(_RECORD) + "\n\n" + json.dumps(_DIRECT) + "\n")
        assert _load().validate(str(p)) == 0

    def test_bad_json(self, tmp_path, caplog):
        p = tmp_path / "golden.jsonl"
        p.write_text("{not valid json}\n")
        assert _load().validate(str(p)) == 1
        assert any("JSON decode error" in r.message for r in caplog.records)

    def test_record_fails_everything(self, tmp_path, caplog):
        # metadata that is not a dict makes EvalInput.from_swebench_record raise.
        p = tmp_path / "golden.jsonl"
        p.write_text(json.dumps({"metadata": "not-a-dict", "repo": "sympy/sympy"}) + "\n")
        assert _load().validate(str(p)) == 1
        assert any("EvalInput construction failed" in r.message for r in caplog.records)

    def test_missing_file(self, tmp_path):
        assert _load().validate(str(tmp_path / "nope.jsonl")) == 1

    def test_missing_fields_warning(self, tmp_path, caplog):
        rec = dict(_RECORD)
        rec["repo"] = ""
        p = tmp_path / "golden.jsonl"
        p.write_text(json.dumps(rec) + "\n")
        assert _load().validate(str(p)) == 0
        assert any("missing repo" in r.message for r in caplog.records)

    def test_missing_every_required_field(self, tmp_path, caplog):
        # every sanity check trips in _check_input; validation still exits 0
        p = tmp_path / "golden.jsonl"
        p.write_text(json.dumps({"repo": "r", "issue_body": "b", "repo_domain": "d"}) + "\n")
        assert _load().validate(str(p)) == 0
        messages = "\n".join(r.message for r in caplog.records)
        assert "missing instance_id" in messages
        assert "missing base_sha" in messages
        assert "missing test_patch" in messages
        assert "both fail_to_pass and pass_to_pass empty" in messages


class TestMain:
    def test_main_exit_zero(self, tmp_path, mocker):
        p = tmp_path / "golden.jsonl"
        p.write_text(json.dumps(_DIRECT) + "\n")
        mocker.patch("sys.argv", ["validate_golden", "--path", str(p)])
        with pytest.raises(SystemExit) as e:
            _load().main()
        assert e.value.code == 0

    def test_main_exit_one(self, tmp_path, mocker):
        mocker.patch("sys.argv", ["validate_golden", "--path", str(tmp_path / "missing.jsonl")])
        with pytest.raises(SystemExit) as e:
            _load().main()
        assert e.value.code == 1
