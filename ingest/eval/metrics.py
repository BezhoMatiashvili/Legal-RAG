"""Accuracy-first retrieval metrics with evidence-unit denominators.

The historical evaluator called a binary any-hit indicator ``Recall@k``.  That number is
useful, but it is *success rate*, not recall.  The canonical names are now
``success5``/``success10``; read-only ``recall5``/``recall10`` aliases remain so older
analysis scripts and experiment readers keep working.

True evidence recall is measured over required evidence groups.  One annotated evidence
span maps to a group of all chunks that overlap that span, and retrieving *any* member
satisfies the group.  This avoids penalising a system merely because overlap/rechunking
created several interchangeable chunks for the same legal passage.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Legacy keys are ``(source, document_id[, chunk_index])``.  Canonical v3 keys insert the
# immutable version before the chunk index: ``(source, document_id, version_id[, chunk])``.
# Keeping both shapes lets frozen v1/v2 runs retain their exact historical identities while
# ensuring two versions of the same source document can never satisfy each other's qrels.
Key = tuple
EvidenceGroups = Mapping[str, frozenset[Key] | set[Key]]


@dataclass(frozen=True)
class Hit:
    source: str
    document_id: str
    chunk_index: int
    score: float
    version_id: str | None = None


def hit_key(hit: Hit, level: str) -> Key:
    """Return the legacy or canonical-version identity for one ranked hit."""

    if level not in {"chunk", "doc"}:
        raise ValueError(f"unknown relevance level: {level!r}")
    if hit.version_id is None:
        return (
            (hit.source, hit.document_id, hit.chunk_index)
            if level == "chunk"
            else (hit.source, hit.document_id)
        )
    return (
        (hit.source, hit.document_id, hit.version_id, hit.chunk_index)
        if level == "chunk"
        else (hit.source, hit.document_id, hit.version_id)
    )


def chunk_key_parts(key: Key) -> tuple[str, str, int, str | None]:
    """Decode either supported chunk-key shape for point-id construction."""

    if len(key) == 3:
        source, document_id, chunk_index = key
        return str(source), str(document_id), int(chunk_index), None
    if len(key) == 4:
        source, document_id, version_id, chunk_index = key
        return str(source), str(document_id), int(chunk_index), str(version_id)
    raise ValueError(f"invalid evaluator chunk key: {key!r}")


@dataclass(frozen=True)
class QueryScore:
    """All per-query retrieval signals.

    Candidate recall is optional because the legacy serving path only exposes final hits.
    Accuracy-path branch traces and evaluator backends expose the candidate pool and fill
    these fields.  Missing candidate measurements are never silently converted to zero.
    """

    id: str
    query_type: str
    language: str
    cluster_id: str
    success5: float
    success10: float
    required_evidence_recall10: float
    candidate_recall50: float | None
    candidate_recall80: float | None
    document_identity1: float
    passage_accuracy1: float
    context_duplication10: float
    context_noise10: float
    ndcg10: float
    mrr10: float
    failed: bool = False
    failure_reason: str | None = None

    # Compatibility with historical Python consumers.  New records are emitted under the
    # semantically-correct ``success*`` names, with explicit deprecated aliases in aggregate
    # output (see ``aggregate``).
    @property
    def recall5(self) -> float:
        return self.success5

    @property
    def recall10(self) -> float:
        return self.success10


def reduce_ranking(hits: Sequence[Hit], level: str) -> list[Key]:
    """Collapse chunk hits to a rank-ordered, de-duplicated list of keys at ``level``."""
    if level not in {"chunk", "doc"}:
        raise ValueError(f"unknown relevance level: {level!r}")
    seen: set[Key] = set()
    order: list[Key] = []
    for h in hits:
        key = hit_key(h, level)
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_ranking(
    ranked_keys: Sequence[Key],
    relevant: Mapping[Key, int],
    *,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    success_ks: tuple[int, ...] = (5, 10),
    evidence_groups: EvidenceGroups | None = None,
    # Deprecated keyword retained for callers written against eval-r1.
    recall_ks: tuple[int, ...] | None = None,
) -> tuple[dict[int, float], float, float]:
    """Return ``(success@ks, nDCG@k, MRR@k)`` for one query.

    ``relevant`` maps a relevant key to graded relevance (gain = ``2**grade - 1``).
    ``success@k`` is one iff at least one relevant item occurs in the first ``k`` ranks.
    """
    if recall_ks is not None:
        success_ks = recall_ks
    if evidence_groups:
        group_items = list(evidence_groups.items())
        group_grades = {
            group_id: max((relevant.get(key, 0) for key in alternatives), default=0)
            for group_id, alternatives in group_items
        }
        satisfied: set[str] = set()
        gains = []
        for key in ranked_keys[:ndcg_k]:
            newly_satisfied = [
                group_id
                for group_id, alternatives in group_items
                if group_id not in satisfied and key in alternatives
            ]
            gains.append(sum(2 ** group_grades[group_id] - 1 for group_id in newly_satisfied))
            satisfied.update(newly_satisfied)
        ideal_gains = sorted(
            (2 ** grade - 1 for grade in group_grades.values() if grade > 0), reverse=True
        )[:ndcg_k]
    else:
        gains = [(2 ** relevant[k] - 1) if k in relevant else 0 for k in ranked_keys[:ndcg_k]]
        ideal_gains = sorted((2 ** g - 1 for g in relevant.values()), reverse=True)[:ndcg_k]
    idcg = _dcg(ideal_gains)
    ndcg = min(1.0, (_dcg(gains) / idcg)) if idcg > 0 else 0.0

    first = next((i + 1 for i, key in enumerate(ranked_keys[:mrr_k]) if key in relevant), None)
    mrr = (1.0 / first) if first else 0.0
    successes = {
        kk: (1.0 if any(key in relevant for key in ranked_keys[:kk]) else 0.0)
        for kk in success_ks
    }
    return successes, ndcg, mrr


def evidence_group_recall(
    ranked_keys: Sequence[Key], evidence_groups: EvidenceGroups, k: int
) -> float:
    """Fraction of required evidence groups satisfied in the first ``k`` results.

    Every value in ``evidence_groups`` is an equivalence set: any overlapping chunk in the
    set satisfies that annotated evidence unit.  Empty groups are unsatisfied and therefore
    remain in the denominator, making annotation/chunking failures visible as failures.
    """
    if not evidence_groups:
        return 0.0
    retrieved = set(ranked_keys[:k])
    covered = sum(1 for alternatives in evidence_groups.values() if retrieved & set(alternatives))
    return covered / len(evidence_groups)


def _gold_documents(relevant: Mapping[Key, int], level: str) -> set[Key]:
    if level == "doc":
        return set(relevant)
    documents: set[Key] = set()
    for key in relevant:
        if len(key) == 3:
            documents.add((key[0], key[1]))
        elif len(key) == 4:
            documents.add((key[0], key[1], key[2]))
    return documents


def context_quality(
    hits: Sequence[Hit], evidence_groups: EvidenceGroups, *, k: int = 10
) -> tuple[float, float]:
    """Return ``(duplication, noise)`` for the selected top-k chunk context.

    A result is redundant when its exact chunk key was already seen, or when it overlaps
    evidence but adds no evidence group beyond earlier results.  Noise is the share of
    returned chunks that overlaps no required evidence group.  Empty context has maximal
    noise (it contains no usable evidence) and zero duplication.
    """
    chosen = list(hits[:k])
    if not chosen:
        return 0.0, 1.0
    group_sets = [set(keys) for keys in evidence_groups.values()]
    seen_keys: set[Key] = set()
    covered_groups: set[int] = set()
    duplicates = 0
    noisy = 0
    for hit in chosen:
        key = hit_key(hit, "chunk")
        matching = {i for i, alternatives in enumerate(group_sets) if key in alternatives}
        if key in seen_keys or (matching and matching <= covered_groups):
            duplicates += 1
        if not matching:
            noisy += 1
        seen_keys.add(key)
        covered_groups.update(matching)
    n = len(chosen)
    return duplicates / n, noisy / n


def query_score(
    query_id: str,
    query_type: str,
    language: str,
    hits: Sequence[Hit],
    relevant: Mapping[Key, int],
    level: str,
    *,
    evidence_groups: EvidenceGroups | None = None,
    candidate_recall: Mapping[int, float | None] | None = None,
    cluster_id: str = "",
    failed: bool = False,
    failure_reason: str | None = None,
) -> QueryScore:
    """Score one query, including evidence recall and context diagnostics."""
    ranked = reduce_ranking(hits, level)
    chunk_ranked = reduce_ranking(hits, "chunk")
    groups: EvidenceGroups = evidence_groups or {
        # Compatibility fallback for direct callers without span groups: every qrel is one
        # required unit.  The harness always supplies true span-equivalence groups.
        f"legacy:{i}": frozenset({key}) for i, key in enumerate(relevant)
    }
    successes, ndcg, mrr = score_ranking(
        ranked,
        relevant,
        evidence_groups=groups if level == "chunk" else None,
    )
    duplication, noise = context_quality(hits, groups, k=10)
    docs = _gold_documents(relevant, level)
    first_doc = hit_key(hits[0], "doc") if hits else None
    first_chunk = chunk_ranked[0] if chunk_ranked else None
    relevant_chunks = set().union(*(set(v) for v in groups.values())) if groups else set()
    candidate_recall = candidate_recall or {}
    return QueryScore(
        id=query_id,
        query_type=query_type,
        language=language,
        cluster_id=cluster_id or query_id,
        success5=successes[5],
        success10=successes[10],
        required_evidence_recall10=evidence_group_recall(chunk_ranked, groups, 10),
        candidate_recall50=candidate_recall.get(50),
        candidate_recall80=candidate_recall.get(80),
        document_identity1=1.0 if first_doc in docs else 0.0,
        passage_accuracy1=1.0 if first_chunk in relevant_chunks else 0.0,
        context_duplication10=duplication,
        context_noise10=noise,
        ndcg10=ndcg,
        mrr10=mrr,
        failed=failed,
        failure_reason=failure_reason,
    )


# Canonical metrics emitted by eval-r2.  Lower is better only for the two context-cost
# metrics; comparisons use this direction map rather than treating every increase as a win.
METRIC_NAMES = (
    "success5",
    "success10",
    "required_evidence_recall10",
    "candidate_recall50",
    "candidate_recall80",
    "document_identity1",
    "passage_accuracy1",
    "context_duplication10",
    "context_noise10",
    "ndcg10",
    "mrr10",
)
METRIC_DIRECTIONS = {
    "context_duplication10": "lower",
    "context_noise10": "lower",
}
LEGACY_METRIC_ALIASES = {"recall5": "success5", "recall10": "success10"}


def values(scores: Sequence[QueryScore], metric: str) -> list[float]:
    """Available numeric values for ``metric`` (candidate metrics may be unavailable)."""
    canonical = LEGACY_METRIC_ALIASES.get(metric, metric)
    out: list[float] = []
    for score in scores:
        value = getattr(score, canonical)
        if value is not None:
            out.append(float(value))
    return out


def paired_values(
    a: Sequence[QueryScore], b: Sequence[QueryScore], metric: str
) -> tuple[list[float], list[float], list[str]]:
    """Align two tracks by query id and retain pairs where the metric exists on both."""
    canonical = LEGACY_METRIC_ALIASES.get(metric, metric)
    by_id_a = {score.id: score for score in a}
    by_id_b = {score.id: score for score in b}
    if set(by_id_a) != set(by_id_b):
        missing_a = sorted(set(by_id_b) - set(by_id_a))
        missing_b = sorted(set(by_id_a) - set(by_id_b))
        raise ValueError(
            f"paired query-id mismatch: missing_from_a={missing_a}, missing_from_b={missing_b}"
        )
    available_a = {qid for qid, score in by_id_a.items() if getattr(score, canonical) is not None}
    available_b = {qid for qid, score in by_id_b.items() if getattr(score, canonical) is not None}
    if available_a != available_b:
        raise ValueError(f"paired metric availability mismatch for {canonical}")
    aa: list[float] = []
    bb: list[float] = []
    clusters: list[str] = []
    for left in a:
        right = by_id_b[left.id]
        av = getattr(left, canonical)
        bv = getattr(right, canonical)
        if av is None or bv is None:
            continue
        if left.cluster_id != right.cluster_id:
            raise ValueError(f"cluster mismatch for paired query {left.id!r}")
        aa.append(float(av))
        bb.append(float(bv))
        clusters.append(left.cluster_id)
    return aa, bb, clusters


def aggregate(scores: Sequence[QueryScore]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for metric in METRIC_NAMES:
        vals = values(scores, metric)
        out[metric] = (sum(vals) / len(vals)) if vals else 0.0
        out[f"n_{metric}"] = len(vals)
    # JSON compatibility aliases are deliberately explicit and can be retired only after
    # all historical report builders migrate to ``success*``.
    for alias, canonical in LEGACY_METRIC_ALIASES.items():
        out[alias] = out[canonical]
    out["n_queries"] = len(scores)
    out["n"] = len(scores)  # legacy breakdown/report alias
    out["failed_queries"] = sum(1 for score in scores if score.failed)
    return out


def breakdown(scores: Sequence[QueryScore], attr: str) -> dict[str, dict[str, float | int]]:
    """Aggregate by a grouping attribute (``query_type`` or ``language``)."""
    groups: dict[str, list[QueryScore]] = {}
    for score in scores:
        groups.setdefault(getattr(score, attr), []).append(score)
    return {group: aggregate(group_scores) for group, group_scores in sorted(groups.items())}


def percentiles(samples: Sequence[float], ps: Sequence[float] = (50, 95)) -> dict[str, float]:
    """Latency percentiles (linear interpolation); empty input -> zeros."""
    if not samples:
        return {f"p{int(p)}": 0.0 for p in ps}
    ordered = sorted(samples)
    out: dict[str, float] = {}
    for p in ps:
        if len(ordered) == 1:
            out[f"p{int(p)}"] = ordered[0]
            continue
        rank = (p / 100.0) * (len(ordered) - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        frac = rank - lo
        out[f"p{int(p)}"] = ordered[lo] + (ordered[hi] - ordered[lo]) * frac
    return out
