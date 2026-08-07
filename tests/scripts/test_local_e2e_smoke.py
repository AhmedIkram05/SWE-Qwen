"""Unit tests for ``scripts/local_e2e_smoke.py``.

The evaluation harness and local backends are mocked — no Ollama, no pytest
subprocesses.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _load():
    from scripts import local_e2e_smoke

    return local_e2e_smoke


@pytest.fixture
def mod():
    return _load()


def _example(i):
    return SimpleNamespace(
        instance_id=f"sympy__sympy-{i}",
        repo="sympy/sympy",
        fail_to_pass=["test_a"],
        pass_to_pass=["test_b"],
        test_patch="--- a/tests/t.py\n+++ b/tests/t.py\n",
    )


def _result(success=True, error=None):
    return SimpleNamespace(
        instance_id="sympy__sympy-1",
        patch_application=SimpleNamespace(
            success=success, files_modified=["tests/t.py"] if success else []
        ),
        f2p=0.5,
        p2p=1.0,
        latency_seconds=1.2,
        error=error,
        tests_before=[SimpleNamespace(status="failed")],
        tests_after=[SimpleNamespace(status="passed"), SimpleNamespace(status="passed")],
    )


@pytest.fixture
def harness_mocks(mocker):
    """Swap the real harness internals for mocks before main() imports them."""
    import evaluation.harness as harness_mod

    mocker.patch.object(harness_mod, "EvaluationHarness")
    # main() assigns harness_mod._generate_patches / _run_tests → restore after.
    mocker.patch.object(harness_mod, "_generate_patches")
    mocker.patch.object(harness_mod, "_run_tests")
    mocker.patch("evaluation.config.EvalConfig")
    return harness_mod


class TestMain:
    def test_dry_run(self, mod, harness_mocks, mocker):
        mocker.patch(
            "sys.argv", ["local_e2e_smoke", "--dry-run", "--golden-path", "data/golden.jsonl"]
        )
        harness = harness_mocks.EvaluationHarness.return_value
        harness.load_examples.return_value = [_example(1)]
        mod.main()
        assert harness.load_examples.call_count == 1
        assert harness.run_example.call_count == 0

    def test_use_golden_patch(self, mod, harness_mocks, mocker):
        mocker.patch(
            "sys.argv",
            ["local_e2e_smoke", "--use-golden-patch", "--golden-path", "g.jsonl"],
        )
        harness = harness_mocks.EvaluationHarness.return_value
        harness.load_examples.return_value = [_example(1)]
        harness.run_example.return_value = _result(success=True)
        import evaluation.local_backend as lb

        run = mocker.patch.object(lb, "run_tests_local", return_value="ran")
        mod.main()
        harness.run_example.assert_called_once()
        # golden-patch closure returns the ground-truth test patches
        patches = harness_mocks._generate_patches("m", "v", "t", [_example(2)])
        assert patches == ["--- a/tests/t.py\n+++ b/tests/t.py\n"]
        # patched _run_tests delegates to run_tests_local
        assert harness_mocks._run_tests(_example(1), "patch", None) == "ran"
        run.assert_called_once()

    def test_sample_zero_skips_sample(self, mod, harness_mocks, mocker):
        mocker.patch("sys.argv", ["local_e2e_smoke", "--sample", "0", "--golden-path", "g.jsonl"])
        harness = harness_mocks.EvaluationHarness.return_value
        harness.load_examples.return_value = [_example(1)]
        harness.run_example.return_value = _result(success=True)
        mod.main()
        assert harness.run_example.call_count == 1

    def test_inference_branch_failure_result(self, mod, harness_mocks, mocker):
        mocker.patch(
            "sys.argv",
            [
                "local_e2e_smoke",
                "--golden-path",
                "g.jsonl",
                "--sample",
                "5",
                "--model",
                "qwen2.5-coder:7b",
                "--ollama-url",
                "http://localhost:11434",
            ],
        )
        harness = harness_mocks.EvaluationHarness.return_value
        harness.load_examples.return_value = [_example(1), _example(2)]
        harness.run_example.return_value = _result(success=False, error="no adapter")
        import evaluation.local_backend as lb

        gen = mocker.patch.object(lb, "generate_patches_local", return_value=["gen"])
        run = mocker.patch.object(lb, "run_tests_local", return_value="ran")
        mod.main()
        harness.run_example.assert_called_once()
        # local_backend.generate_patches_local was importable (function defined)
        assert callable(harness_mocks._generate_patches)
        patches = harness_mocks._generate_patches("m", "v", "t", [_example(1)])
        assert patches == ["gen"]
        gen.assert_called_once()
        assert harness_mocks._run_tests(_example(1), "patch", None) == "ran"
        run.assert_called_once()

    def test_json_summary(self, mod, harness_mocks, mocker, capsys):
        mocker.patch(
            "sys.argv",
            ["local_e2e_smoke", "--use-golden-patch", "--golden-path", "g.jsonl"],
        )
        harness = harness_mocks.EvaluationHarness.return_value
        harness.load_examples.return_value = [_example(1)]
        harness.run_example.return_value = _result(success=True)
        mod.main()
        out = capsys.readouterr().out
        summary = json.loads(out.split("--- JSON summary ---", 1)[1].strip())
        assert summary["instance_id"] == "sympy__sympy-1"
        assert summary["patch_success"] is True
        assert summary["f2p"] == 0.5
        assert summary["tests_after"] == 2
