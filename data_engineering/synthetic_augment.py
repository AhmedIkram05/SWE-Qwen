"""
Synthetic data augmentation: CodeContests + CodeAlpaca → valid unified diffs.
Converts competitive programming / instruction-following examples to IssueRecord
with proper patch_diff that passes Pydantic validation and git apply.
"""

import hashlib
import re
from difflib import unified_diff
from typing import Any

from datasets import load_dataset
from unidiff import PatchSet

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, ParsedHunk, TestResults


def extract_function_signature(problem_description: str) -> str | None:
    """Extract likely function signature from problem description."""
    # Look for patterns like "def solve()", "def function_name(", etc.
    patterns = [
        r"def\s+(\w+)\s*\(",
        r"function\s+(\w+)\s*\(",
        r"implement\s+(\w+)\s*\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, problem_description, re.IGNORECASE)
        if match:
            return f"def {match.group(1)}("
    return None


def create_stub_file(problem_name: str, description: str) -> str:
    """Create a minimal stub file for the problem."""
    func_sig = extract_function_signature(description)
    if func_sig:
        callable_name = func_sig.split("(")[0].split()[-1]
        stub = (
            f"# TODO: implement {problem_name}\n{func_sig}\n    pass\n"
            f"\nif __name__ == '__main__':\n    {callable_name}()\n"
        )
    else:
        stub = (
            f"# TODO: implement {problem_name}\n"
            f"def solve():\n    pass\n\nif __name__ == '__main__':\n    solve()\n"
        )
    return stub


def solution_to_unified_diff(
    problem_name: str,
    description: str,
    solution: str,
) -> str:
    """
    Convert problem + solution → unified diff.
    Creates stub before, solution after.
    """
    filename = re.sub(r"[^a-zA-Z0-9_]", "_", problem_name.lower()) + ".py"

    before = create_stub_file(problem_name, description)
    after = solution.rstrip() + "\n"

    diff_lines = list(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="\n",
        )
    )
    return "".join(diff_lines)


def parse_hunks_from_diff(patch_diff: str) -> list[dict[str, Any]]:
    """Parse unified diff into structured hunks for IssueRecord.parsed_hunks."""
    hunks = []
    patch_set = PatchSet(patch_diff)
    for patched_file in patch_set:
        filepath = patched_file.path

        for hunk in patched_file:
            hunks.append(
                {
                    "file": filepath,
                    "old_start": hunk.source_start,
                    "old_lines": hunk.source_length,
                    "new_start": hunk.target_start,
                    "new_lines": hunk.target_length,
                    "diff_lines": [str(line) for line in hunk],
                }
            )
    return hunks


PYTHON_LANGUAGE = 3


def load_codecontests(config: DataPipelineConfig) -> list[IssueRecord]:
    """
    Load CodeContests dataset, convert to IssueRecord with valid unified diffs.
    CodeContests uses parallel arrays: solutions = {"solution": [...], "language": [...]}
    where language=3 means Python.
    """
    ds = load_dataset("deepmind/code_contests", split="train")
    records = []

    for ex in ds:
        problem_name = ex.get("name", "unknown")
        description = ex.get("description", "")

        solutions = ex.get("solutions", {})
        if not isinstance(solutions, dict):
            continue
        solution_list = solutions.get("solution", [])
        language_list = solutions.get("language", [])

        for i, sol in enumerate(solution_list):
            lang = language_list[i] if i < len(language_list) else -1
            if lang != PYTHON_LANGUAGE:
                continue

            solution_code = sol.strip()
            if not solution_code:
                continue

            patch_diff = solution_to_unified_diff(problem_name, description, solution_code)
            parsed_hunks_dict = parse_hunks_from_diff(patch_diff)

            parsed_hunks = [ParsedHunk(**h) for h in parsed_hunks_dict]
            test_results = TestResults(passed=[], failed=[], errored=[])

            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", problem_name)
            issue_id = (
                f"codecontests_{safe_name}_{hashlib.md5(solution_code.encode()).hexdigest()[:8]}"
            )

            records.append(
                IssueRecord(
                    issue_id=issue_id,
                    repo="synthetic/codecontests",
                    issue_body=description,
                    patch_diff=patch_diff,
                    parsed_hunks=parsed_hunks,
                    test_results=test_results,
                    metadata={
                        "source": "codecontests",
                        "has_test_patch": False,
                        "problem_name": problem_name,
                        "difficulty": ex.get("difficulty", "unknown"),
                    },
                )
            )

    return records


def load_codealpaca(config: DataPipelineConfig) -> list[IssueRecord]:
    """
    Load CodeAlpaca dataset, convert to IssueRecord with valid unified diffs.
    Filters to Python-relevant examples.
    """
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    records = []

    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        input_text = ex.get("input", "").strip()
        output = ex.get("output", "").strip()

        if not output:
            continue

        # Filter: only keep Python-relevant examples
        combined = f"{instruction}\n{input_text}".lower()
        if not any(
            kw in combined for kw in ["python", "def ", "function", "class ", "script", "code"]
        ):
            continue

        problem_name = (
            f"codealpaca_{hashlib.md5((instruction + input_text).encode()).hexdigest()[:8]}"
        )
        description = f"{instruction}\n\n{input_text}" if input_text else instruction

        patch_diff = solution_to_unified_diff(problem_name, description, output)
        parsed_hunks_dict = parse_hunks_from_diff(patch_diff)

        parsed_hunks = [ParsedHunk(**h) for h in parsed_hunks_dict]
        test_results = TestResults(passed=[], failed=[], errored=[])

        issue_id = f"codealpaca_{hashlib.md5((instruction + input_text).encode()).hexdigest()[:8]}"

        records.append(
            IssueRecord(
                issue_id=issue_id,
                repo="synthetic/codealpaca",
                issue_body=description,
                patch_diff=patch_diff,
                parsed_hunks=parsed_hunks,
                test_results=test_results,
                metadata={"source": "codealpaca", "has_test_patch": False},
            )
        )

    return records


def augment_training_data(
    swe_bench_train: list[IssueRecord],
    config: DataPipelineConfig,
) -> list[IssueRecord]:
    """
    Merge SWE-bench train + synthetic, deduplicate, cap at max_train_examples.
    Synthetic records are ONLY added to training split (no val/test/golden leakage).
    """
    synthetic = []

    if config.augment_codecontests:
        synthetic.extend(load_codecontests(config))

    if config.augment_codealpaca:
        synthetic.extend(load_codealpaca(config))

    # Deduplicate by issue_body hash
    seen = set()
    merged = []

    for r in swe_bench_train + synthetic:
        h = hashlib.md5(r.issue_body.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            merged.append(r)

    # Cap for training budget
    if len(merged) > config.max_train_examples:
        merged = merged[: config.max_train_examples]

    return merged
