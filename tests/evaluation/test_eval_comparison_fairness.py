"""P0-5 comparison fairness: cross-variant paired McNemar + paired bootstrap
over strict instance-resolve outcomes (F2P == 1.0) on identical instances.

``paired_significance`` previously only paired the SAME variant across runs
(prompt A/B); the variant-vs-variant head-to-head (baseline vs fine-tune)
was never computed here even though ``stats.mcnemar_p`` /
``stats.paired_bootstrap_ci`` were already correct and available.
"""

from __future__ import annotations

from datetime import UTC, datetime

from evaluation.comparison import paired_significance
from evaluation.config import EvalConfig
from evaluation.schema import EvalResult, EvalRun, PatchApplicationResult


def _result(
    instance_id: str,
    variant: str = "baseline_14b",
    model: str = "qwen3-14b",
    f2p: float = 1.0,
) -> EvalResult:
    return EvalResult(
        instance_id=instance_id,
        repo="django/django",
        model_name=model,
        variant=variant,
        prompt_template="chat",
        generated_patch="",
        patch_application=PatchApplicationResult(success=True, method_used="git_apply"),
        tests_before=[],
        tests_after=[],
        f2p=f2p,
        p2p=1.0,
        latency_seconds=1.0,
        timestamp=datetime.now(UTC),
        error=None,
    )


def _run(run_id: str, results: list[EvalResult]) -> EvalRun:
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=EvalConfig(),
        models_evaluated=sorted({f"{r.model_name}:{r.variant}" for r in results}),
        results=results,
        aggregate=[],
        status="completed",
    )


def _line(out: str, needle: str) -> str:
    lines = [l for l in out.splitlines() if needle in l]
    assert len(lines) == 1, f"expected exactly one line with {needle!r} in:\n{out}"
    return lines[0]


class TestCrossVariantPairedSignificance:
    def test_cross_variant_significant(self):
        # 40 shared instances; baseline resolves 4, lora resolves a superset
        # of 14 → 10 discordant pairs all one way → p = 2*0.5^10 ≈ 0.002.
        a = _run(
            "run-a",
            [_result(f"i{k}", f2p=1.0 if k < 4 else 0.0) for k in range(40)],
        )
        b = _run(
            "run-b",
            [
                _result(f"i{k}", variant="higher_rank_14b", f2p=1.0 if k < 14 else 0.0)
                for k in range(40)
            ],
        )
        out = paired_significance(a, b)
        line = _line(out, "qwen3-14b:baseline_14b vs qwen3-14b:higher_rank_14b")
        assert "McNemar p=0.0020" in line
        assert "(n.s.)" not in line
        assert "resolve diff -25.00%" in line
        assert "(n=40)" in line

    def test_cross_variant_not_significant(self):
        # Discordants b01=2 (b-only wins) vs b10=1 (a-only win): n=3,
        # p = 2*P(Bin(3,0.5)<=1) = 1.0 → not significant.
        a = _run(
            "run-a",
            [_result(f"i{k}", f2p=1.0 if k in {0, 1, 2, 3, 4} else 0.0) for k in range(20)],
        )
        b = _run(
            "run-b",
            [
                _result(
                    f"i{k}",
                    variant="higher_rank_14b",
                    f2p=1.0 if k in {0, 1, 2, 3, 6, 7} else 0.0,
                )
                for k in range(20)
            ],
        )
        out = paired_significance(a, b)
        line = _line(out, "qwen3-14b:baseline_14b vs qwen3-14b:higher_rank_14b")
        assert "McNemar p=1.0000 (n.s.)" in line
        assert "resolve diff -5.00%" in line
        assert "(n=20)" in line

    def test_same_variant_pairing_preserved(self):
        # Same variant across two runs: repeat-run pairing still works.
        a = _run("run-a", [_result("i1", f2p=1.0), _result("i2", f2p=0.0)])
        b = _run("run-b", [_result("i1", f2p=0.0), _result("i2", f2p=1.0)])
        out = paired_significance(a, b)
        line = _line(out, "qwen3-14b:baseline_14b:")
        assert "McNemar p=1.0000 (n.s.)" in line
        assert "(n=2)" in line
        # No cross-variant section: only one variant evaluated.
        assert "baseline_14b vs" not in out

    def test_partial_credit_counts_as_failure(self):
        # Partial credit (f2p=0.5) is NOT a resolve for McNemar/bootstrap.
        a = _run("run-a", [_result(f"i{k}", f2p=0.5) for k in range(10)])
        b = _run("run-b", [_result(f"i{k}", f2p=1.0) for k in range(10)])
        out = paired_significance(a, b)
        line = _line(out, "qwen3-14b:baseline_14b:")
        assert "resolve diff -100.00%" in line
        assert "McNemar p=0.0020" in line  # 10 discordant pairs one way: 2*0.5^10

    def test_blank_instance_ids_never_paired(self):
        # A blank instance_id must never pair (even with itself); the
        # shared non-blank instance pairs as n=1 only.
        a = _run("run-a", [_result("i1"), _result("")])
        b = _run("run-b", [_result("i1", f2p=0.0), _result("i2", f2p=0.0)])
        out = paired_significance(a, b)
        line = _line(out, "qwen3-14b:baseline_14b:")
        assert "(n=1)" in line

    def test_different_models_not_cross_paired(self):
        a = _run("run-a", [_result("i1", model="m1")])
        b = _run("run-b", [_result("i1", model="m2", variant="lora_x")])
        out = paired_significance(a, b)
        assert "no variant evaluated in both runs" in out
        assert "McNemar" not in out

    def test_cross_variant_disjoint_instances(self):
        a = _run("run-a", [_result("i1")])
        b = _run("run-b", [_result("i2", variant="lora_x")])
        out = paired_significance(a, b)
        assert "share no instances" in out
        assert "McNemar" not in out
