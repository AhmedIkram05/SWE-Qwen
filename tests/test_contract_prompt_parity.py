"""P0-7 parity contract: train format vs eval format prompt builders.

``data_engineering.tokenize._format_parts`` (train) and
``inference.prompt_builder.render_patch_prompt`` (eval) must stay identical
given the same example inputs: same system text, same snippet caps, same
few-shot policy, same no-think handling. These tests fail on any re-skew.

No network, no GPU, no Modal: file-snippet fetches are mocked.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_engineering.tokenize import _format_parts
from evaluation.schema import EvalInput
from inference import prompt_builder
from inference.prompt_builder import (
    apply_no_think_raw,
    log_prompt_hash,
    no_think_flag,
    prompt_hash,
    render_patch_prompt,
)
from training.prompt_loader import PromptLoader

# ── The contract (independent of any implementation constants) ──────────────
EXPECTED_SYSTEM_TASK = "Fix the bug described in the issue by generating a correct patch."
EXPECTED_SYSTEM_STYLE = "Follow PEP 8 and the repository's existing code style."
EXPECTED_SYSTEM_LANGUAGE = "Python"
EXPECTED_SNIPPET_MAX_FILES = 5
EXPECTED_SNIPPET_MAX_LINES = 150
EXPECTED_CONTEXT_FILES_MAX = 20
EXPECTED_TEST_FILES_MAX = 10
FEWSHOT_SECTION = "### Example Patches"
NO_THINK_MARKER = "/no_think"

QWEN_HF_ID = "Qwen/Qwen3-14B"
NON_QWEN_HF_ID = "HuggingFaceTB/SmolLM2-135M"

_UNSET = object()


def _snippet_payload() -> list[dict[str, str]]:
    return [
        {"path": "src/a.py", "content": "def a():\n    return 1\n"},
        {"path": "src/b.py", "content": "def b():\n    return 2\n"},
        {"path": "src/c.py", "content": "def c():\n    return 3\n"},
    ]


def _split_prompt(prompt: str) -> tuple[str, str]:
    """(system, user) parts of a rendered ``chat.j2`` prompt."""
    system, _, rest = prompt.partition("\n### Input\n")
    user = rest.partition("\n\n### Response")[0]
    return system, user


@pytest.fixture
def example_pair() -> tuple[dict[str, Any], EvalInput]:
    """Same example expressed as a train record and an eval input."""
    record: dict[str, Any] = {
        "issue_id": "parity#1",
        "issue_body": "Button does nothing when clicked.",
        "repo": "parity/repo",
        "repo_domain": "web",
        "patch_diff": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-1\n+2\n",
        "files_changed": ["src/a.py", "src/b.py", "src/c.py"],
        "test_files_changed": ["tests/test_x.py"],
        "metadata": {"base_sha": "deadbeef"},
    }
    example = EvalInput(
        instance_id="parity#1",
        repo="parity/repo",
        issue_body="Button does nothing when clicked.",
        base_sha="deadbeef",
        head_sha="cafef00d",
        test_patch=(
            "diff --git a/tests/test_x.py b/tests/test_x.py\n"
            "--- a/tests/test_x.py\n"
            "+++ b/tests/test_x.py\n"
            "@@ -1 +1,2 @@\n"
            "+def test_x():\n"
            "     pass\n"
        ),
        fail_to_pass=["tests/test_x.py::test_x"],
        pass_to_pass=[],
        repo_domain="web",
        metadata={
            "task_description": EXPECTED_SYSTEM_TASK,
            "style_guide": EXPECTED_SYSTEM_STYLE,
            "language": EXPECTED_SYSTEM_LANGUAGE,
            "context_files": ["src/a.py", "src/b.py", "src/c.py"],
        },
    )
    return record, example


@pytest.fixture
def prompt_loader() -> PromptLoader:
    return PromptLoader()


@pytest.fixture
def mock_snippets(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Spy on every ``_file_snippets`` binding site; record requested caps.

    Both builders resolve the fetch through module attributes that differ
    across checkouts (the train side imports it directly; the shared contract
    helpers resolve via the ``_eval()`` bridge), so all sites are spied with
    one shared call list. Each test invokes exactly one builder, so its call
    is the only entry.
    """
    calls: list[dict[str, Any]] = []
    payload = _snippet_payload()

    def spy(
        repo: str, base_sha: str, paths: list[str], max_files: Any = _UNSET, max_lines: Any = _UNSET
    ) -> list[dict[str, str]]:
        calls.append(
            {
                "repo": repo,
                "base_sha": base_sha,
                "paths": list(paths),
                "max_files": max_files,
                "max_lines": max_lines,
            }
        )
        return [dict(s) for s in payload]

    monkeypatch.setattr("inference.prompt_builder._file_snippets", spy)
    monkeypatch.setattr("evaluation.inference._file_snippets", spy, raising=False)
    monkeypatch.setattr("data_engineering.tokenize._file_snippets", spy, raising=False)
    return calls


def _require_p01() -> None:
    """Skip (not fail) when the P0-1 contract surface is not in this checkout.

    Lets this branch merge independently: the guards activate automatically
    once fix/p01-prompt-parity lands on main.
    """
    if not isinstance(getattr(prompt_builder, "PROMPT_CONTRACT", None), dict):
        pytest.skip("P0-1 shared prompt contract not in this checkout")


def _contract() -> dict[str, Any]:
    _require_p01()
    contract = prompt_builder.PROMPT_CONTRACT
    if not isinstance(contract, dict):
        pytest.fail(
            "inference.prompt_builder.PROMPT_CONTRACT is not a dict — "
            "the shared prompt contract surface (P0-1) is malformed"
        )
    return contract


class TestTrainFormatContract:
    """``_format_parts`` must follow the contract."""

    def test_system_text_is_contract_system(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, _ = example_pair
        _, prompt_only = _format_parts(record, prompt_loader)
        system, _ = _split_prompt(prompt_only)
        expected = prompt_loader.render(
            "system",
            language=EXPECTED_SYSTEM_LANGUAGE,
            task_description=EXPECTED_SYSTEM_TASK,
            style_guide=EXPECTED_SYSTEM_STYLE,
        )
        assert system.strip() == expected.strip()

    def test_snippet_caps_are_contract_caps(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, _ = example_pair
        _format_parts(record, prompt_loader)
        call = mock_snippets[0]
        assert call["max_files"] == EXPECTED_SNIPPET_MAX_FILES
        assert call["max_lines"] == EXPECTED_SNIPPET_MAX_LINES

    def test_context_files_capped(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, _ = example_pair
        record["files_changed"] = [f"f{i:02d}.py" for i in range(EXPECTED_CONTEXT_FILES_MAX + 5)]
        _, prompt_only = _format_parts(record, prompt_loader)
        _, user = _split_prompt(prompt_only)
        section = user.split("### Relevant Files", 1)[1].split("###", 1)[0]
        assert section.count("- `") == EXPECTED_CONTEXT_FILES_MAX

    def test_test_files_capped(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, _ = example_pair
        record["test_files_changed"] = [
            f"test_f{i:02d}.py" for i in range(EXPECTED_TEST_FILES_MAX + 2)
        ]
        _, prompt_only = _format_parts(record, prompt_loader)
        _, user = _split_prompt(prompt_only)
        section = user.split("### Test Files", 1)[1].split("###", 1)[0]
        assert section.count("- `") == EXPECTED_TEST_FILES_MAX

    def test_no_fewshot_section(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, _ = example_pair
        _, prompt_only = _format_parts(record, prompt_loader)
        assert FEWSHOT_SECTION not in prompt_only

    def test_no_no_think_in_train_output(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, _ = example_pair
        full, prompt_only = _format_parts(record, prompt_loader)
        assert NO_THINK_MARKER not in full
        assert NO_THINK_MARKER not in prompt_only


class TestEvalFormatContract:
    """``render_patch_prompt`` must follow the contract."""

    def _render(self, example_pair: tuple[dict[str, Any], EvalInput]) -> str:
        _, example = example_pair
        return render_patch_prompt(example, include_file_contents=True, example_patches=[])

    def test_system_text_is_contract_system(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        prompt = self._render(example_pair)
        system, _ = _split_prompt(prompt)
        expected = prompt_loader.render(
            "system",
            language=EXPECTED_SYSTEM_LANGUAGE,
            task_description=EXPECTED_SYSTEM_TASK,
            style_guide=EXPECTED_SYSTEM_STYLE,
        )
        assert system.strip() == expected.strip()

    def test_snippet_caps_are_contract_caps(
        self, example_pair: tuple[dict[str, Any], EvalInput], mock_snippets: list
    ) -> None:
        _require_p01()
        self._render(example_pair)
        call = mock_snippets[0]
        assert call["max_files"] == EXPECTED_SNIPPET_MAX_FILES, (
            "eval format fetches snippets with non-contract caps — "
            "render_patch_prompt must use the shared contract caps"
        )
        assert call["max_lines"] == EXPECTED_SNIPPET_MAX_LINES

    def test_no_fewshot_section(
        self, example_pair: tuple[dict[str, Any], EvalInput], mock_snippets: list
    ) -> None:
        assert FEWSHOT_SECTION not in self._render(example_pair)

    def test_no_no_think_in_eval_output(
        self, example_pair: tuple[dict[str, Any], EvalInput], mock_snippets: list
    ) -> None:
        assert NO_THINK_MARKER not in self._render(example_pair)


class TestSharedContractSurface:
    """The shared ``PROMPT_CONTRACT`` constants must pin the contract values."""

    def test_contract_pins_system_text(self) -> None:
        c = _contract()
        assert c["system_task"] == EXPECTED_SYSTEM_TASK
        assert c["system_style"] == EXPECTED_SYSTEM_STYLE
        assert c["system_language_default"] == EXPECTED_SYSTEM_LANGUAGE

    def test_contract_pins_snippet_and_file_caps(self) -> None:
        c = _contract()
        assert c["snippet_max_files"] == EXPECTED_SNIPPET_MAX_FILES
        assert c["snippet_max_lines"] == EXPECTED_SNIPPET_MAX_LINES
        assert c["context_files_max"] == EXPECTED_CONTEXT_FILES_MAX
        assert c["test_files_max"] == EXPECTED_TEST_FILES_MAX

    def test_contract_pins_fewshot_and_no_think_policy(self) -> None:
        c = _contract()
        assert c["fewshot_policy"] == "none"
        assert c["no_think_policy"] == "inference_wrapper_only"

    def test_default_example_patches_are_none(self) -> None:
        _require_p01()
        helper = getattr(prompt_builder, "contract_example_patches", None)
        if helper is None:
            pytest.fail("inference.prompt_builder.contract_example_patches is missing (P0-1)")
        assert helper(None) == []
        assert helper(["patch-a"]) == ["patch-a"]


class TestSameInputSameOutput:
    """The actual parity: same example inputs → identical rendered prompt."""

    def test_train_prompt_equals_eval_prompt(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, example = example_pair
        _, prompt_only = _format_parts(record, prompt_loader)
        eval_prompt = render_patch_prompt(example, include_file_contents=True, example_patches=[])
        assert eval_prompt == prompt_only

    def test_system_and_user_parts_match(
        self,
        example_pair: tuple[dict[str, Any], EvalInput],
        prompt_loader: PromptLoader,
        mock_snippets: list,
    ) -> None:
        record, example = example_pair
        _, prompt_only = _format_parts(record, prompt_loader)
        eval_prompt = render_patch_prompt(example, include_file_contents=True, example_patches=[])
        train_system, train_user = _split_prompt(prompt_only)
        eval_system, eval_user = _split_prompt(eval_prompt)
        assert train_system == eval_system, "system text diverges between train and eval"
        assert train_user == eval_user, "user content diverges between train and eval"


class TestNoThinkGate:
    """No-think handling is the inference wrappers' job, keyed off the registry."""

    def test_qwen_flag_on_non_qwen_flag_off(self) -> None:
        assert no_think_flag(QWEN_HF_ID) is True
        assert no_think_flag(NON_QWEN_HF_ID) is False

    def test_wrapper_inserts_marker_once(self) -> None:
        prompt = "system text\n\n### Input\nuser text\n\n### Response"
        wrapped = apply_no_think_raw(QWEN_HF_ID, prompt)
        assert wrapped.count(NO_THINK_MARKER) == 1
        assert f"{NO_THINK_MARKER}\n### Response" in wrapped

    def test_wrapper_passthrough_for_non_qwen(self) -> None:
        prompt = "system text\n\n### Input\nuser text\n\n### Response"
        assert apply_no_think_raw(NON_QWEN_HF_ID, prompt) == prompt


class TestPromptHashHelper:
    def test_deterministic(self) -> None:
        assert prompt_hash("abc") == prompt_hash("abc")

    def test_sensitive_to_change(self) -> None:
        assert prompt_hash("abc") != prompt_hash("abd")

    def test_log_prompt_hash_emits_and_returns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="inference.prompt_builder"):
            digest = log_prompt_hash("train", "abc")
        assert digest == prompt_hash("abc")
        assert f"prompt_hash train sha256={digest}" in caplog.text
