"""Local inference and test execution backends for debugging the eval pipeline.

Replaces Modal ``_generate_patches`` and ``_run_tests`` in ``harness.py``
with local implementations that run on the dev machine. Swapped in via
``--backend local`` on the CLI.

Inference uses either Ollama (default) or MLX. Test runner shells out to
the local ``pytest``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Local inference backend ───────────────────────────────────────────────


def generate_patches_local(  # noqa: PLR0913
    model_name: str,
    variant: str,
    prompt_template: str,
    examples: list[Any],
    *,
    ollama_model: str = "qwen2.5-coder:7b",
    ollama_base_url: str = "http://localhost:11434",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> list[str]:
    """Generate patches via Ollama's OpenAI-compatible API (no Modal, no vLLM).

    Args:
        model_name: Only used for logging (model registry key).
        variant: Variant key (adapter not applied locally).
        prompt_template: Template name for ``render_patch_prompt``.
        examples: Eval instances.
        ollama_model: Ollama model tag (e.g. ``qwen3-14b``).
        ollama_base_url: Ollama server URL.
        max_tokens: Max completion length.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.

    Returns:
        Patches in same order as *examples*.
    """
    import httpx

    from evaluation.inference import extract_patch, render_patch_prompt

    patches: list[str] = []
    for i, example in enumerate(examples):
        prompt = render_patch_prompt(example, template_name=prompt_template)
        logger.info(
            "local inference: %s/%s example %d/%d (instance=%s, prompt_len=%d)",
            model_name,
            variant,
            i + 1,
            len(examples),
            example.instance_id,
            len(prompt),
        )
        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{ollama_base_url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a software engineer fixing bugs. Output only the patch.",  # noqa: E501
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                },
                timeout=600,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("Ollama inference failed for %s: %s", example.instance_id, exc)
            text = ""

        patch = extract_patch(text)
        elapsed = time.monotonic() - start
        logger.info(
            "  inference done: %.1fs, patch_len=%d, extracted=%s",
            elapsed,
            len(text),
            "yes" if patch else "no",
        )
        patches.append(patch)

    return patches


# ── Local test runner backend ─────────────────────────────────────────────


def run_tests_local(
    example: Any,
    generated_patch: str,
    config: Any,
    *,
    repos_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run tests locally (no Modal container).

    Clones the repo if not cached, applies patches, runs pytest, returns
    the same dict shape as ``run_tests_in_container``.

    Args:
        example: ``EvalInput`` instance.
        generated_patch: Model-generated fix patch.
        config: ``EvalConfig`` (used for timeouts/retries).
        repos_dir: Where to cache cloned repos (default ``/tmp/swe-eval-repos``).

    Returns:
        Dict with ``tests_before``/``tests_after``/``patch_application``
        matching the Modal backend's return shape.
    """
    from evaluation.patch_applier import apply_patch
    from evaluation.schema import PatchApplicationResult, TestResult
    from evaluation.test_runner import collect_test_results

    repos_dir = Path(repos_dir or "/tmp/swe-eval-repos")
    repo_dir = repos_dir / example.repo

    test_names = [*(example.fail_to_pass or []), *(example.pass_to_pass or [])]

    try:
        _ensure_local_repo(example.repo, repo_dir, example.base_sha)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        return _error_response(example, f"repo prep: {exc}")

    # tests_before
    tests_before = collect_test_results(
        repo_dir, test_names, timeout=config.test_timeout_seconds, max_retries=config.max_retries
    )

    # Apply generated patch → tests_after
    tests_after: list[TestResult] = []
    if generated_patch:
        patch_result = apply_patch(repo_dir, generated_patch, example.base_sha)
        if patch_result.success:
            tests_after = collect_test_results(
                repo_dir,
                test_names,
                timeout=config.test_timeout_seconds,
                max_retries=config.max_retries,
            )
        else:
            logger.warning(
                "local patch apply failed for %s: %s", example.instance_id, patch_result.error
            )
    else:
        patch_result = PatchApplicationResult(success=False, method_used="failed", error="no patch")

    return {
        "repo": example.repo,
        "base_sha": example.base_sha,
        "tests_before": [t.model_dump() for t in tests_before],
        "tests_head": [],
        "tests_after": [t.model_dump() for t in tests_after],
        "patch_application": patch_result.model_dump(),
        "ground_truth": {},
    }


def _ensure_local_repo(repo: str, repo_dir: Path, base_sha: str) -> None:
    """Clone repo (shallow) and checkout base_sha."""
    if (repo_dir / ".git").is_dir():
        logger.info("repo %s already cached at %s", repo, repo_dir)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth=1",
                f"https://github.com/{repo}.git",
                str(repo_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    # Fetch the exact SHA if missing
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-e", base_sha],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # we check returncode below
    )
    if proc.returncode != 0:
        logger.info("fetching base_sha %s for %s", base_sha, repo)
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "--quiet", "--depth=1", "origin", base_sha],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--quiet", base_sha],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    # Install the package in editable mode so its conftest / imports work
    # during test collection.  Silently skip on failure — some old repos
    # are incompatible with Python 3.14 (e.g. matplotlib builds from
    # source), and the user will see the collection error as a pytest
    # failure downstream.
    if not (repo_dir / ".installed").exists():
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            (repo_dir / ".installed").touch()
        except Exception:
            logger.warning(
                "pip install -e failed for %s at %s — tests will fail if "
                "conftest imports the package",
                repo,
                repo_dir,
            )


def _error_response(example: Any, error: str) -> dict[str, Any]:
    return {
        "repo": example.repo,
        "base_sha": example.base_sha,
        "error": error,
        "tests_before": [],
        "tests_head": [],
        "tests_after": [],
        "patch_application": {"success": False, "method_used": "failed", "error": error},
        "ground_truth": {},
    }
