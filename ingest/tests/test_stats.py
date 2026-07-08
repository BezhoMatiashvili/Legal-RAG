"""Bootstrap CIs, paired difference CIs, and the paired permutation test / decision rule."""

import math

from eval.stats import bootstrap_ci, compare, paired_permutation_p


def test_bootstrap_ci_brackets_mean_and_orders():
    vals = [1.0] * 50 + [0.0] * 50
    ci = bootstrap_ci(vals, seed=1)
    assert abs(ci.mean - 0.5) < 1e-9
    assert ci.lo < ci.mean < ci.hi
    # a constant vector has a degenerate (zero-width) CI at the mean
    const = bootstrap_ci([0.7] * 30, seed=1)
    assert math.isclose(const.lo, const.hi) and math.isclose(const.mean, 0.7)


def test_permutation_p_high_when_no_difference():
    a = [0.5, 0.6, 0.4, 0.55, 0.45] * 6
    p = paired_permutation_p(a, a, seed=3)
    assert p == 1.0  # zero observed difference → not significant


def test_compare_detects_a_real_win():
    # B strictly better on every query by a wide, consistent margin
    a = [0.1, 0.2, 0.15, 0.05, 0.25] * 8
    b = [x + 0.5 for x in a]
    c = compare("recall10", a, b, seed=7)
    assert c.diff > 0.4
    assert c.diff_lo > 0.0            # CI excludes zero
    assert c.p_value < 0.05
    assert c.verdict == "ADOPT B"


def test_compare_calls_tie_on_noise():
    # symmetric small differences → not significant
    a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    b = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
    c = compare("ndcg10", a, b, seed=11)
    assert c.verdict == "TIE"
    assert c.diff_lo <= 0.0 <= c.diff_hi


def test_compare_symmetry_direction():
    a = [0.1, 0.2, 0.15, 0.05, 0.25] * 8
    b = [x + 0.5 for x in a]
    # swapping args flips the verdict direction
    assert compare("mrr10", b, a, seed=7).verdict == "ADOPT A"
