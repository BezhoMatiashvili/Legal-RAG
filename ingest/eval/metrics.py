"""Retrieval metrics: Recall@k (hit-rate), nDCG@k (graded), MRR@k, with breakdowns.

Relevance is evaluated at two granularities:
  * **chunk** — a hit is relevant iff its ``(source, document_id, chunk_index)`` is in the
    gold span→chunk set (the strict, span-anchored view; the primary metric).
  * **doc**   — a hit is relevant iff its ``(source, document_id)`` is a gold document (the
    robust, coarse fallback; also what the pre-Part-2 harness measured).

"Recall@k" here is hit-rate — 1 if *any* relevant item is in the top-k — matching the
legacy harness and the usual RAG success criterion (one chunk with the evidence suffices).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

Key = tuple  # (source, document_id[, chunk_index])


@dataclass(frozen=True)
class Hit:
    source: str
    document_id: str
    chunk_index: int
    score: float


@dataclass(frozen=True)
class QueryScore:
    id: str
    query_type: str
    language: str
    recall5: float
    recall10: float
    ndcg10: float
    mrr10: float


def reduce_ranking(hits: Sequence[Hit], level: str) -> list[Key]:
    """Collapse chunk hits to a rank-ordered, de-duplicated list of keys at ``level``."""
    seen: set[Key] = set()
    order: list[Key] = []
    for h in hits:
        key: Key = (
            (h.source, h.document_id, h.chunk_index)
            if level == "chunk"
            else (h.source, h.document_id)
        )
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_ranking(
    ranked_keys: Sequence[Key],
    relevant: dict[Key, int],
    *,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_ks: tuple[int, ...] = (5, 10),
) -> tuple[dict[int, float], float, float]:
    """Return ``(recall@ks, ndcg@ndcg_k, mrr@mrr_k)`` for one query.

    ``relevant`` maps a relevant key to its graded relevance (gain = ``2**grade - 1``).
    """
    gains = [(2 ** relevant[k] - 1) if k in relevant else 0 for k in ranked_keys[:ndcg_k]]
    ideal_gains = sorted((2 ** g - 1 for g in relevant.values()), reverse=True)[:ndcg_k]
    idcg = _dcg(ideal_gains)
    ndcg = (_dcg(gains) / idcg) if idcg > 0 else 0.0

    first = next((i + 1 for i, k in enumerate(ranked_keys[:mrr_k]) if k in relevant), None)
    mrr = (1.0 / first) if first else 0.0

    recalls = {
        kk: (1.0 if any(k in relevant for k in ranked_keys[:kk]) else 0.0)
        for kk in recall_ks
    }
    return recalls, ndcg, mrr


def query_score(
    query_id: str,
    query_type: str,
    language: str,
    hits: Sequence[Hit],
    relevant: dict[Key, int],
    level: str,
) -> QueryScore:
    ranked = reduce_ranking(hits, level)
    recalls, ndcg, mrr = score_ranking(ranked, relevant)
    return QueryScore(
        id=query_id,
        query_type=query_type,
        language=language,
        recall5=recalls[5],
        recall10=recalls[10],
        ndcg10=ndcg,
        mrr10=mrr,
    )


METRIC_NAMES = ("recall5", "recall10", "ndcg10", "mrr10")


def values(scores: Sequence[QueryScore], metric: str) -> list[float]:
    return [getattr(s, metric) for s in scores]


def aggregate(scores: Sequence[QueryScore]) -> dict[str, float]:
    n = len(scores) or 1
    return {m: sum(getattr(s, m) for s in scores) / n for m in METRIC_NAMES}


def breakdown(scores: Sequence[QueryScore], attr: str) -> dict[str, dict[str, float]]:
    """Aggregate by a grouping attribute (``query_type`` or ``language``)."""
    groups: dict[str, list[QueryScore]] = {}
    for s in scores:
        groups.setdefault(getattr(s, attr), []).append(s)
    return {g: {"n": len(gs), **aggregate(gs)} for g, gs in sorted(groups.items())}


def percentiles(samples: Sequence[float], ps: Sequence[float] = (50, 95)) -> dict[str, float]:
    """Latency percentiles (linear interpolation); empty input → zeros."""
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
