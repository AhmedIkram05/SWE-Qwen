"""Local eval debug: one golden example, full pipeline (no Modal).

Uses actual pipeline functions: collect_test_results, compute_f2p, apply_patch.
Exercises the same code path Modal will run.

Usage:
    python scripts/debug_eval_one.py          # first sympy example (pure Python)
    python scripts/debug_eval_one.py SYMPY    # first sympy example
    python scripts/debug_eval_one.py 5        # 6th golden example (any repo)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluation.metrics import compute_f2p
from evaluation.patch_applier import apply_patch
from evaluation.schema import EvalInput
from evaluation.test_runner import _install_repo, collect_test_results


def main() -> None:
    golden = Path("data/expanded-repos/swebench/golden.jsonl")

    # Default: first sympy (pure Python, fast install)
    use_sympy = len(sys.argv) < 2 or sys.argv[1].upper() == "SYMPY"  # noqa: PLR2004
    if use_sympy:
        with golden.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("repo") == "sympy/sympy":
                    break
    else:
        idx = int(sys.argv[1])
        with golden.open() as f:
            records = [json.loads(line) for line in f]
        rec = records[idx]

    # Use actual pipeline parser
    inp = EvalInput.from_swebench_record(rec)
    all_tests = inp.fail_to_pass + inp.pass_to_pass

    print(f"\n=== Debug: {inp.repo} | {inp.instance_id} | base={inp.base_sha[:7]} ===")
    print(f"F2P: {len(inp.fail_to_pass)}, P2P: {len(inp.pass_to_pass)}")
    print(f"First F2P: {inp.fail_to_pass[0] if inp.fail_to_pass else 'NONE'}")

    if not inp.test_patch:
        print("SKIP: no test_patch")
        return

    with tempfile.TemporaryDirectory(prefix="eval-debug-") as tmpdir:
        repo_dir = Path(tmpdir) / "repo"

        # Clone + checkout (shallow first, then fetch specific commit)
        print("\n[1/4] Cloning + checkout...")
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{inp.repo}.git", str(repo_dir)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        subprocess.run(
            ["git", "fetch", "origin", inp.base_sha, "--depth", "1"],
            cwd=repo_dir,
            capture_output=True,
            timeout=120,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=repo_dir,
            capture_output=True,
            timeout=60,
            check=True,
        )

        # Install — the fix that was missing
        print("[2/4] pip install -e . ...")
        _install_repo(repo_dir)

        # Run tests BEFORE patch — uses ACTUAL pipeline function
        print(f"[3/4] collect_test_results (before, {len(all_tests)} tests)...")
        tests_before = collect_test_results(repo_dir, all_tests[:50], timeout=60, max_retries=0)
        failed_before = {t.name for t in tests_before if t.status in ("failed", "errored")}
        passed_before = {t.name for t in tests_before if t.status == "passed"}
        print(
            f"  Collected: {len(tests_before)}, Failed: {len(failed_before)}, "
            f"Passed: {len(passed_before)}"
        )

        # Apply test patch (ground truth)
        print("[4/4] Apply test patch + collect_test_results (after)...")
        pr = apply_patch(repo_dir, inp.test_patch, inp.base_sha)
        print(f"  Patch: {pr.method_used}, success={pr.success}")

        tests_after = collect_test_results(repo_dir, all_tests[:50], timeout=60, max_retries=0)
        passed_after = {t.name for t in tests_after if t.status == "passed"}
        print(f"  Collected: {len(tests_after)}, Passed: {len(passed_after)}")

        # Compute F2P/P2P — uses ACTUAL pipeline function
        f2p_rate, p2p_rate, f2p_count, p2p_count = compute_f2p(
            tests_before, tests_after, inp.fail_to_pass, inp.pass_to_pass
        )
        print(f"\n  F2P: {f2p_count}/{f2p_rate:.0%}, P2P: {p2p_count}/{p2p_rate:.0%}")

    if f2p_rate > 0:
        print("\n PIPELINE WORKS. F2P > 0, ready for Modal.")
    else:
        print("\n  F2P = 0. Check: are tests being collected? Is pip install working?")


if __name__ == "__main__":
    main()
