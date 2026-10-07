"""Train/eval prompt-parity contract tests (P0-1).

The training prompt builder (``data_engineering.tokenize._format_parts``)
and the eval prompt builder (``inference.prompt_builder.render_patch_prompt``)
must stay in lockstep through the shared ``PROMPT_CONTRACT`` in
``inference.prompt_builder``:

- identical system strings (``SYSTEM_TASK`` / ``SYSTEM_STYLE`` = the strings
  the model was trained with),
- identical snippet caps (``SNIPPET_MAX_FILES`` x ``SNIPPET_MAX_LINES``),
- no few-shot by default (``FEWSHOT_POLICY = "none"``),
- a bare-diff instruction (``FENCED_POLICY = "bare"``) matching the bare-diff
  training target,
- no ``/no_think`` in either builder's output (``NO_THINK_POLICY`` — the
  thinking gate is applied only by the inference-time wrappers, identically
  to train-shaped and eval-shaped prompts).

These tests pin that parity; a regression that diverges one builder from the
other fails here. Network fetches are mocked; no Modal, no real tokenizer.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_engineering.tokenize import _format_parts, format_training_prompt
from evaluation.schema import EvalInput
from inference import prompt_builder
from training.prompt_loader import PromptLoader

pytestmark = pytest.mark.unit

GOLD = "diff --git a/a/b.py b/a/b.py\n@@ -1 +1 @@\n-old\n+new\n"


@pytest.fixture(autouse=True)
def _force_local_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the _eval() bridge so patched state resolves deterministically."""
    monkeypatch.setattr(prompt_builder, "_eval", lambda: prompt_builder)


def _record() -> dict[str, Any]:
    """A training record with equivalent content to ``_example()``."""
    return {
        "issue_id": "django__django-10554",
        "issue_body": "BooleanField crashes on None.",
        "repo": "django/django",
        "repo_domain": "python",
        "patch_diff": GOLD,
        "files_changed": ["a/b.py", "c/d.py"],
        "test_files_changed": ["tests/test_models.py"],
        "metadata": {"base_sha": "base123"},
    }


def _example() -> EvalInput:
    """An eval input equivalent to ``_record()`` (same issue, repo, files)."""
    return EvalInput(
        instance_id="django__django-10554",
        repo="django/django",
        issue_body="BooleanField crashes on None.",
        base_sha="base123",
        head_sha="head123",
        test_patch="diff --git a/tests/test_models.py b/tests/test_models.py\n@@ -1 +1 @@\n",
        gold_patch=GOLD,
        fail_to_pass=["tests/test_models.py::test_x"],
        pass_to_pass=[],
        repo_domain="python",
        metadata={"context_files": ["a/b.py", "c/d.py"]},
    )


@pytest.fixture
def loader() -> PromptLoader:
    return PromptLoader()


def _mock_fetch(monkeypatch: pytest.MonkeyPatch, content: str = "# snippet\n") -> None:
    monkeypatch.setattr(
        prompt_builder,
        "_fetch_raw_file",
        lambda repo, base_sha, path: content,
    )


class TestContractConstants:
    def test_snippet_caps(self):
        assert prompt_builder.SNIPPET_MAX_FILES == 5
        assert prompt_builder.SNIPPET_MAX_LINES == 150

    def test_policies(self):
        assert prompt_builder.FEWSHOT_POLICY == "none"
        assert prompt_builder.FENCED_POLICY == "bare"
        assert prompt_builder.NO_THINK_POLICY == "inference_wrapper_only"

    def test_contract_dict_matches_constants(self):
        c = prompt_builder.PROMPT_CONTRACT
        assert c["system_task"] == prompt_builder.SYSTEM_TASK
        assert c["system_style"] == prompt_builder.SYSTEM_STYLE
        assert c["system_language_default"] == prompt_builder.SYSTEM_LANGUAGE_DEFAULT
        assert c["snippet_max_files"] == prompt_builder.SNIPPET_MAX_FILES
        assert c["snippet_max_lines"] == prompt_builder.SNIPPET_MAX_LINES
        assert c["fewshot_policy"] == prompt_builder.FEWSHOT_POLICY
        assert c["fenced_policy"] == prompt_builder.FENCED_POLICY
        assert c["no_think_policy"] == prompt_builder.NO_THINK_POLICY


class TestTrainEvalParity:
    def test_eval_prompt_identical_to_train_prompt_only(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same inputs in, byte-identical prompts out (same system text,
        same snippet caps, same empty few-shot, same bare-diff instruction)."""
        _mock_fetch(monkeypatch)
        full, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        assert eval_prompt == prompt_only

    def test_train_full_text_is_prompt_plus_bare_gold(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_fetch(monkeypatch)
        full, prompt_only = _format_parts(_record(), loader)
        assert full.startswith(prompt_only)
        # Target is the bare patch (no fences) — FENCED_POLICY.
        assert full[len(prompt_only) :] == GOLD
        assert "```diff" not in full

    def test_contract_strings_present_in_both(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_fetch(monkeypatch)
        _, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        for text in (prompt_only, eval_prompt):
            assert prompt_builder.SYSTEM_TASK in text
            assert prompt_builder.SYSTEM_STYLE in text

    def test_snippet_caps_enforced_in_both(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = [f"f{i}.py" for i in range(8)]
        rec = _record()
        rec["files_changed"] = files
        ex = _example()
        ex.metadata["context_files"] = files
        _mock_fetch(monkeypatch)
        _, prompt_only = _format_parts(rec, loader)
        eval_prompt = prompt_builder.render_patch_prompt(ex, include_file_contents=True)
        assert prompt_only == eval_prompt
        assert prompt_only.count("#### `") == prompt_builder.SNIPPET_MAX_FILES
        assert eval_prompt.count("#### `") == prompt_builder.SNIPPET_MAX_FILES

    def test_long_file_truncated_at_contract_lines(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            prompt_builder,
            "_fetch_raw_file",
            lambda repo, base_sha, path: "\n".join(f"line{i}" for i in range(200)),
        )
        _, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        assert prompt_only == eval_prompt
        assert "# ... (file truncated)" in eval_prompt
        assert "line149" in eval_prompt
        assert "line150" not in eval_prompt

    def test_no_fewshot_by_default_in_either_builder(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: Any, **k: Any) -> None:
            raise AssertionError("golden must not be auto-injected (FEWSHOT_POLICY=none)")

        monkeypatch.setattr(prompt_builder, "_golden_patches", boom)
        _mock_fetch(monkeypatch)
        _, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        assert "Example Patches" not in prompt_only
        assert "Example Patches" not in eval_prompt

    def test_bare_diff_instruction_in_both(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # FENCED_POLICY=bare: user.j2 must request raw patch text (no fence),
        # matching the bare-diff training target.
        _mock_fetch(monkeypatch)
        _, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        for text in (prompt_only, eval_prompt):
            assert "no code fences" in text
            assert "fenced" not in text


class TestNoThinkParity:
    def test_no_think_absent_from_both_builders(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NO_THINK_POLICY=inference_wrapper_only: neither builder inserts
        # /no_think; tokenized training data stays canonical bare text.
        _mock_fetch(monkeypatch)
        full, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        assert "/no_think" not in prompt_only
        assert "/no_think" not in full
        assert "/no_think" not in eval_prompt

    def test_no_think_wrapper_identical_on_both_shapes(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_fetch(monkeypatch)
        monkeypatch.setattr(prompt_builder, "no_think_flag", lambda hf_id: True)
        _, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        train_wrapped = prompt_builder.apply_no_think_raw("m", prompt_only)
        eval_wrapped = prompt_builder.apply_no_think_raw("m", eval_prompt)
        assert train_wrapped == eval_wrapped
        assert train_wrapped.count("/no_think") == 1
        assert "/no_think\n### Response" in train_wrapped

    def test_no_think_wrapper_passthrough_when_unflagged(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_fetch(monkeypatch)
        monkeypatch.setattr(prompt_builder, "no_think_flag", lambda hf_id: False)
        _, prompt_only = _format_parts(_record(), loader)
        eval_prompt = prompt_builder.render_patch_prompt(_example(), include_file_contents=True)
        assert prompt_builder.apply_no_think_raw("m", prompt_only) == prompt_only
        assert prompt_builder.apply_no_think_raw("m", eval_prompt) == eval_prompt


class TestFormatTrainingPromptStillWorks:
    """format_training_prompt keeps its public behavior (full text with
    the gold patch appended) under the shared contract."""

    def test_contains_issue_and_bare_patch(
        self, loader: PromptLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_fetch(monkeypatch)
        text = format_training_prompt(_record(), loader)
        assert "django__django-10554" in text
        assert "BooleanField crashes on None." in text
        assert "### Input" in text
        assert "### Response" in text
        assert text.endswith(GOLD)
        assert "```diff" not in text
