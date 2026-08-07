"""Unit tests for evaluation.stats (Wilson CI, McNemar, paired bootstrap)."""

from evaluation.stats import mcnamar_p, paired_bootstrap_ci, wilson_ci


class TestWilsonCI:
    def test_edges_do_not_collapse(self):
        # 0/N and N/N keep a non-degenerate interval (unlike Wald).
        lo, hi = wilson_ci(0, 20)
        assert lo == 0.0 and hi > 0.0
        lo, hi = wilson_ci(20, 20)
        assert hi == 1.0 and lo < 1.0

    def test_zero_total(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_contains_point_estimate(self):
        lo, hi = wilson_ci(10, 40)
        assert lo <= 0.25 <= hi
        assert lo < 0.25 < hi


class TestMcNemar:
    def test_no_discordant_pairs(self):
        assert mcnamar_p(0, 0) == 1.0

    def test_balanced_discordance_not_significant(self):
        # 30 vs 30 flips: p = 1.0
        assert mcnamar_p(30, 30) == 1.0

    def test_extreme_discordance_significant(self):
        # 0 vs 40 flips: p < 0.05
        assert mcnamar_p(0, 40) < 0.05
        assert mcnamar_p(40, 0) < 0.05

    def test_small_extreme_case(self):
        # 0 vs 3: two-sided p = 0.25 (2 * 0.125)
        assert mcnamar_p(0, 3) == 0.25

    def test_symmetric(self):
        assert mcnamar_p(3, 7) == mcnamar_p(7, 3)


class TestPairedBootstrap:
    def test_identical_sequences_zero_diff(self):
        lo, hi, diff = paired_bootstrap_ci([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
        assert diff == 0.0
        assert lo <= 0.0 <= hi

    def test_known_mean_diff(self):
        lo, hi, diff = paired_bootstrap_ci([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        assert diff == 1.0
        assert lo == 1.0 and hi == 1.0  # constant difference

    def test_deterministic_seed(self):
        a = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        assert paired_bootstrap_ci(a, b, seed=42) == paired_bootstrap_ci(a, b, seed=42)
        # Any seed gives a sane interval around the observed mean diff
        lo, hi, diff = paired_bootstrap_ci(a, b, seed=7)
        assert -1.0 <= lo <= hi <= 1.0
        assert lo <= diff <= hi

    def test_empty(self):
        assert paired_bootstrap_ci([], []) == (0.0, 0.0, 0.0)
