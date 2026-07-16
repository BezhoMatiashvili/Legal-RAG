"""Cluster-aware confidence intervals and multiplicity-corrected paired tests.

Queries about the same document or amendment lineage are not independent.  All public
helpers therefore accept an optional cluster label per query; when supplied, bootstrap
resampling and permutation sign flips operate on whole clusters.  A/B reports apply Holm's
step-down correction across the family of reported metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 12345


def _validate_inputs(arrays: Sequence[np.ndarray], *, resamples: int, alpha: float) -> None:
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("statistical inputs must be finite")


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
    p_value_adjusted: float | None = None
    n: int = 0
    n_clusters: int = 0
    higher_is_better: bool = True


def _cluster_arrays(
    n: int, clusters: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inverse labels, cluster totals helper labels, and counts."""
    if clusters is None:
        labels = np.arange(n, dtype=int)
    else:
        if len(clusters) != n:
            raise ValueError("cluster labels must match values length")
        # np.unique sorts labels, which is deterministic and immaterial to resampling.
        _unique, labels = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    n_clusters = int(labels.max()) + 1 if n else 0
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    return labels, np.arange(n_clusters, dtype=int), counts


def _cluster_bootstrap_means(
    arr: np.ndarray,
    clusters: Sequence[str] | None,
    *,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    labels, cluster_ids, counts = _cluster_arrays(arr.size, clusters)
    if not cluster_ids.size:
        return np.zeros(resamples, dtype=float), 0
    sums = np.bincount(labels, weights=arr, minlength=cluster_ids.size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, cluster_ids.size, size=(resamples, cluster_ids.size))
    sampled_sums = sums[draws].sum(axis=1)
    sampled_counts = counts[draws].sum(axis=1)
    return sampled_sums / sampled_counts, int(cluster_ids.size)


def bootstrap_ci(
    values: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> CI:
    """Percentile cluster-bootstrap CI for the query-level mean."""
    arr = np.asarray(values, dtype=float)
    _validate_inputs((arr,), resamples=resamples, alpha=alpha)
    if arr.size == 0:
        return CI(0.0, 0.0, 0.0)
    means, _ = _cluster_bootstrap_means(arr, clusters, resamples=resamples, seed=seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(arr.mean()), float(lo), float(hi))


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> CI:
    """Cluster-bootstrap CI for paired ``mean(b - a)``."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    _validate_inputs((aa, bb), resamples=resamples, alpha=alpha)
    if aa.shape != bb.shape:
        raise ValueError("paired arrays must have equal length")
    diff = bb - aa
    if diff.size == 0:
        return CI(0.0, 0.0, 0.0)
    means, _ = _cluster_bootstrap_means(diff, clusters, resamples=resamples, seed=seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(diff.mean()), float(lo), float(hi))


def paired_permutation_p(
    a: Sequence[float],
    b: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided paired sign-flip test, flipping whole document/version clusters."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    _validate_inputs((aa, bb), resamples=resamples, alpha=0.5)
    if aa.shape != bb.shape:
        raise ValueError("paired arrays must have equal length")
    diff = bb - aa
    n = diff.size
    if n == 0:
        return 1.0
    observed = abs(diff.mean())
    if observed == 0.0:
        return 1.0
    labels, cluster_ids, _counts = _cluster_arrays(n, clusters)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(resamples, cluster_ids.size))
    permuted = np.abs((signs[:, labels] * diff).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (resamples + 1))


def _verdict(
    diff: float,
    diff_lo: float,
    diff_hi: float,
    p_value: float,
    *,
    alpha: float,
    higher_is_better: bool,
) -> str:
    significant = p_value < alpha and not (diff_lo <= 0.0 <= diff_hi)
    if not significant:
        return "TIE"
    b_better = diff > 0 if higher_is_better else diff < 0
    return "ADOPT B" if b_better else "ADOPT A"


def compare(
    metric: str,
    a: Sequence[float],
    b: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """Paired A/B on one metric.  B is the candidate, A the baseline."""
    dci = paired_diff_ci(
        a, b, clusters=clusters, resamples=resamples, alpha=alpha, seed=seed
    )
    p = paired_permutation_p(a, b, clusters=clusters, resamples=resamples, seed=seed)
    cluster_count = len(set(clusters)) if clusters is not None else len(a)
    return Comparison(
        metric=metric,
        mean_a=float(np.mean(a)) if len(a) else 0.0,
        mean_b=float(np.mean(b)) if len(b) else 0.0,
        diff=dci.mean,
        diff_lo=dci.lo,
        diff_hi=dci.hi,
        p_value=p,
        p_value_adjusted=p,
        verdict=_verdict(
            dci.mean, dci.lo, dci.hi, p, alpha=alpha,
            higher_is_better=higher_is_better,
        ),
        n=len(a),
        n_clusters=cluster_count,
        higher_is_better=higher_is_better,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise-error adjusted p-values, in original order."""
    if not p_values:
        return []
    ordered = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(ordered)
    running = 0.0
    m = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[original_index] = running
    return adjusted


def holm_correct(
    comparisons: Mapping[str, Comparison], *, alpha: float = 0.05
) -> dict[str, Comparison]:
    """Apply Holm correction to a metric family and recompute adoption verdicts."""
    names = list(comparisons)
    adjusted = holm_adjust([comparisons[name].p_value for name in names])
    out: dict[str, Comparison] = {}
    for name, p_adjusted in zip(names, adjusted):
        comparison = comparisons[name]
        out[name] = replace(
            comparison,
            p_value_adjusted=p_adjusted,
            verdict=_verdict(
                comparison.diff,
                comparison.diff_lo,
                comparison.diff_hi,
                p_adjusted,
                alpha=alpha,
                higher_is_better=comparison.higher_is_better,
            ),
        )
    return out
