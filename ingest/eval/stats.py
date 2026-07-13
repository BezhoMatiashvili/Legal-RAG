"""Statistical rigor for A/B retrieval comparisons — numpy only, no scipy dependency.

Every adopted change must show a statistically-meaningful win, so each metric ships with a
bootstrap 95% CI, and each A/B pair ships with a **paired** difference CI plus a paired
permutation (sign-flip) p-value on the per-query scores. The decision rule: ADOPT only on a
significant win; otherwise TIE (adopt only for a real speed/simplicity gain).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 12345


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float


@dataclass(frozen=True)
class Comparison:
    metric: str
    mean_a: float
    mean_b: float
    diff: float          # mean_b - mean_a
    diff_lo: float
    diff_hi: float
    p_value: float
    verdict: str         # "ADOPT B" | "ADOPT A" | "TIE"


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> CI:
    """Percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(arr.mean()), float(lo), float(hi))


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> CI:
    """Bootstrap CI for the paired mean difference ``mean(b - a)`` (resample query indices)."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("paired arrays must have equal length")
    diff = bb - aa
    if diff.size == 0:
        return CI(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(diff.mean()), float(lo), float(hi))


def paired_permutation_p(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided paired permutation (sign-flip) test on per-query differences.

    Under H0 the sign of each per-query difference is exchangeable; we flip signs at random
    and count how often the permuted mean-|diff| is at least the observed one.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    diff = bb - aa
    n = diff.size
    if n == 0:
        return 1.0
    observed = abs(diff.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(resamples, n))
    permuted = np.abs((signs * diff).mean(axis=1))
    # +1 smoothing so p is never exactly 0.
    return float((np.count_nonzero(permuted >= observed) + 1) / (resamples + 1))


def compare(
    metric: str,
    a: Sequence[float],
    b: Sequence[float],
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """Paired A/B on one metric. B is the candidate, A the baseline."""
    dci = paired_diff_ci(a, b, resamples=resamples, alpha=alpha, seed=seed)
    p = paired_permutation_p(a, b, resamples=resamples, seed=seed)
    significant = p < alpha and not (dci.lo <= 0.0 <= dci.hi)
    if significant:
        verdict = "ADOPT B" if dci.mean > 0 else "ADOPT A"
    else:
        verdict = "TIE"
    return Comparison(
        metric=metric,
        mean_a=float(np.mean(a)) if len(a) else 0.0,
        mean_b=float(np.mean(b)) if len(b) else 0.0,
        diff=dci.mean,
        diff_lo=dci.lo,
        diff_hi=dci.hi,
        p_value=p,
        verdict=verdict,
    )
