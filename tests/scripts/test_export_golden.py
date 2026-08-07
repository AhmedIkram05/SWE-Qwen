"""Unit tests for ``scripts/export_golden.py`` (HF datasets / config mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace


def _load():
    from scripts import export_golden

    return export_golden


def _convert(ex, repo_domain, split):  # noqa: ARG001
    return SimpleNamespace(
        test_files_changed=[] if ex.get("no_test_files") else ["tests/test_x.py"],
        model_dump_json=lambda: json.dumps(
            {"instance_id": ex["instance_id"], "domain": repo_domain, "split": split}
        ),
    )


def _example(instance_id, repo, patch, **kw):
    return {"instance_id": instance_id, "repo": repo, "patch": patch, **kw}


class TestMain:
    def _splits(self):
        return {
            "verified": [
                _example("sympy__sympy-1", "sympy/sympy", "--- a/x.py\n+++ b/x.py\n"),
                # no patch → skipped
                _example("sympy__sympy-2", "sympy/sympy", ""),
                # repo not in SWE_BENCH_PYTHON_REPOS → skipped
                _example("other__repo-3", "unknown/repo", "--- a/x.py\n+++ b/x.py\n"),
                # patch present but no test files → not appended
                _example(
                    "sympy__sympy-4", "sympy/sympy", "--- a/x.py\n+++ b/x.py\n", no_test_files=True
                ),
            ],
            "test": [_example("django__django-9", "django/django", "--- a/y.py\n+++ b/y.py\n")],
            "dev": [],
        }

    def test_main_writes_golden(self, tmp_path, mocker):
        mod = _load()
        mod.SWE_BENCH_PYTHON_REPOS = {"sympy/sympy", "django/django"}
        mod.REPO_DOMAIN_MAP = {"sympy/sympy": "data-ml", "django/django": "web-api"}
        mocker.patch.object(mod, "DataPipelineConfig")
        mocker.patch.object(mod, "load_swebench_splits", return_value=self._splits())
        mocker.patch.object(mod, "swebench_to_issue_record", side_effect=_convert)

        out = tmp_path / "golden.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "export_golden",
                "--out",
                str(out),
                "--swe-bench-dir",
                str(tmp_path / "swe_bench"),
            ],
        )
        mod.main()

        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert {l["instance_id"] for l in lines} == {
            "sympy__sympy-1",
            "django__django-9",
        }
        assert next(l for l in lines if l["instance_id"] == "sympy__sympy-1")["domain"] == "data-ml"
        assert next(l for l in lines if l["instance_id"] == "django__django-9")["split"] == "test"

    def test_main_empty_splits(self, tmp_path, mocker):
        mod = _load()
        mocker.patch.object(mod, "DataPipelineConfig")
        mocker.patch.object(mod, "load_swebench_splits", return_value={})
        mocker.patch.object(mod, "swebench_to_issue_record", side_effect=_convert)
        out = tmp_path / "golden.jsonl"
        mocker.patch("sys.argv", ["export_golden", "--out", str(out)])
        mod.main()
        assert out.read_text() == ""

    def test_main_directory_created(self, tmp_path, mocker):
        mod = _load()
        mocker.patch.object(mod, "DataPipelineConfig")
        mocker.patch.object(mod, "load_swebench_splits", return_value={})
        mocker.patch.object(mod, "swebench_to_issue_record", side_effect=_convert)
        out = tmp_path / "nested" / "dir" / "golden.jsonl"
        mocker.patch("sys.argv", ["export_golden", "--out", str(out)])
        mod.main()
        assert out.exists()
