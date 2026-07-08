"""Hybrid (dense + sparse) retrieval via Qdrant's Query API with RRF fusion.

Used to verify ingestion quality and as a reusable retrieval helper.
"""

import math
import re

from qdrant_client import models

from .config import Config

_MKHEDRULI_RE = re.compile(r"[ა-ჿ]")


def detect_language(text: str) -> str:
    """``"ka"`` if the query carries any Mkhedruli (Georgian) letters, else ``"en"``.

    The corpus is ~99% Georgian, so the only routing question is whether a query is a
    cross-lingual English one (no Georgian script). Mixed KA/EN counts as ``"ka"``.
    """
    return "ka" if _MKHEDRULI_RE.search(text or "") else "en"


def _date_bound(value: str | None) -> str | None:
    if value and len(value) == 10:
        return f"{value}T00:00:00Z"
    return value


def build_filter(
    *,
    source: str | None = None,
    language: str | None = None,
    document_type: str | None = None,
    court: str | None = None,
    document_id: str | None = None,
    document_number: str | None = None,
    registration_code: str | None = None,
    parties: str | None = None,
    contains: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> models.Filter | None:
    """Build an AND filter over the indexed payload fields.

    Exact keyword fields use ``MatchValue``; ``parties`` and ``contains`` use full-text
    ``MatchText`` (needs the TEXT payload index). ``date_*`` ranges over the sortable
    ISO ``date`` field. Returns ``None`` when no filter is requested.
    """
    must = []
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if language:
        must.append(models.FieldCondition(key="language", match=models.MatchValue(value=language)))
    if document_type:
        must.append(
            models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type))
        )
    if court:
        must.append(models.FieldCondition(key="court", match=models.MatchValue(value=court)))
    if document_id:
        must.append(
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
        )
    if document_number:
        must.append(
            models.FieldCondition(
                key="document_number", match=models.MatchValue(value=document_number)
            )
        )
    if registration_code:
        must.append(
            models.FieldCondition(
                key="registration_code", match=models.MatchValue(value=registration_code)
            )
        )
    if parties:
        must.append(models.FieldCondition(key="parties", match=models.MatchText(text=parties)))
    if contains:
        must.append(models.FieldCondition(key="text", match=models.MatchText(text=contains)))
    if status:
        must.append(models.FieldCondition(key="status", match=models.MatchValue(value=status)))
    if date_from or date_to:
        must.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(gte=_date_bound(date_from), lte=_date_bound(date_to)),
            )
        )
    return models.Filter(must=must) if must else None


def hybrid_search(
    cfg: Config,
    client,
    embedder,
    query: str,
    *,
    top_k: int = 10,
    reranker=None,
    rerank_candidates: int | None = None,
    rerank_min_score: float | None = None,
    route: bool = True,
    max_per_doc: int | None = None,
    mmr_lambda: float | None = None,
    **filters,
):
    """Two-stage retrieval: hybrid (dense+sparse, RRF-fused) recall → optional rerank.

    Stage 1 over-fetches a candidate pool from Qdrant by RRF-fusing the dense and
    (when present) sparse matches. Stage 2, when a ``reranker`` is given, re-scores each
    candidate against the query with a cross-encoder, drops anything below
    ``rerank_min_score``, and returns the best ``top_k``. Without a reranker the fused
    RRF order is returned directly (CLI / offline use). Returned points carry the
    final ordering score in ``.score`` (rerank score when reranked, else RRF score).

    **Language routing** (``route=True``): a cross-lingual English query drops the sparse
    branch — BGE-M3's learned-sparse is lexical, so English query tokens can't match
    Georgian sub-words and only add noise. Georgian queries use the full hybrid. Pass
    ``route=False`` to force hybrid regardless (e.g. for an A/B in the eval harness).
    """
    emb = embedder.encode_query(query)
    flt = build_filter(**filters)

    # How many candidates to fuse before reranking. With a reranker we want a deep pool
    # so the cross-encoder has real recall to work with; without one, the old top_k*5/50.
    diversity_on = max_per_doc is not None or mmr_lambda is not None
    want_vec = mmr_lambda is not None  # MMR needs candidate dense vectors
    fetch_n = (rerank_candidates or max(top_k * 5, 50)) if reranker else top_k
    if diversity_on:
        fetch_n = max(fetch_n, top_k * 5, 50)  # diversity needs a pool, not just top_k
    candidate = max(fetch_n, top_k * 5, 50)

    use_sparse = bool(emb.sparse.indices) and not (route and detect_language(query) == "en")
    prefetch = [
        models.Prefetch(query=emb.dense, using="dense", limit=candidate, filter=flt)
    ]
    if use_sparse:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
                using="sparse",
                limit=candidate,
                filter=flt,
            )
        )

    result = client.query_points(
        collection_name=cfg.collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_n,
        with_payload=True,
        with_vectors=want_vec,
    )
    points = result.points
    if reranker is not None and points:
        # Rerank the whole pool when diversifying so MMR/cap choose the final top_k.
        points = rerank_points(
            reranker, query, points,
            top_k=len(points) if diversity_on else top_k, min_score=rerank_min_score,
        )
    if diversity_on:
        return diversify(points, top_k=top_k, max_per_doc=max_per_doc,
                         mmr_lambda=mmr_lambda, query_vec=emb.dense)
    if reranker is None:
        return points[:top_k]
    return points


def _point_dense(pt):
    """Dense vector of a returned point (named-vector dict or bare list), or None."""
    v = getattr(pt, "vector", None)
    if isinstance(v, dict):
        return v.get("dense")
    return v


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def diversify(points, *, top_k: int, max_per_doc: int | None = None,
              mmr_lambda: float | None = None, query_vec=None):
    """Re-rank a candidate pool for result diversity; returns at most ``top_k`` points.

    ``max_per_doc`` greedily caps how many chunks may share a ``document_id`` (order
    preserved) so one long document can't monopolise the results. ``mmr_lambda`` applies
    Maximal Marginal Relevance — each pick maximises ``λ·rel − (1-λ)·max sim(picked)`` over
    dense vectors, trading relevance against redundancy — and needs candidate vectors on the
    points (query them ``with_vectors=True``); it degrades to the cap-only path if vectors
    are absent. When both are set the per-doc cap constrains the MMR selection. ``rel`` is
    the query↔chunk cosine when ``query_vec`` is given, else the pool-normalised score.
    """
    pts = list(points)
    if not pts:
        return []

    def doc_of(pt):
        return (pt.payload or {}).get("document_id")

    counts: dict = {}

    def cap_ok(pt) -> bool:
        return max_per_doc is None or counts.get(doc_of(pt), 0) < max_per_doc

    def take(pt) -> None:
        counts[doc_of(pt)] = counts.get(doc_of(pt), 0) + 1

    if mmr_lambda is None:
        out = []
        for pt in pts:  # already in relevance order
            if cap_ok(pt):
                out.append(pt)
                take(pt)
                if len(out) >= top_k:
                    break
        return out

    vecs = {id(pt): _point_dense(pt) for pt in pts}
    if not any(vecs.values()):  # no candidate vectors → cap-only fallback
        return diversify(pts, top_k=top_k, max_per_doc=max_per_doc, mmr_lambda=None)

    if query_vec and any(vecs.values()):
        rel = {id(pt): _cosine(query_vec, vecs[id(pt)]) for pt in pts}
    else:  # normalise the ranking scores into [0,1] so λ is meaningful across score scales
        ss = [float(pt.score) for pt in pts]
        lo, hi = min(ss), max(ss)
        rng = (hi - lo) or 1.0
        rel = {id(pt): (float(pt.score) - lo) / rng for pt in pts}

    lam = mmr_lambda
    remaining = list(pts)
    selected: list = []
    while remaining and len(selected) < top_k:
        best = None
        best_mmr = None
        for pt in remaining:
            if not cap_ok(pt):
                continue
            sim = max((_cosine(vecs[id(pt)], vecs[id(s)]) for s in selected), default=0.0)
            mmr = lam * rel[id(pt)] - (1 - lam) * sim
            if best_mmr is None or mmr > best_mmr:
                best_mmr, best = mmr, pt
        if best is None:  # everything left is blocked by the per-doc cap
            break
        selected.append(best)
        take(best)
        remaining.remove(best)
    return selected


def rerank_points(reranker, query: str, points, *, top_k: int, min_score: float | None = None):
    """Re-score fused candidates with a cross-encoder, gate, and return the best ``top_k``.

    Each point's ``.score`` is overwritten with its rerank score so downstream
    formatting reports the calibrated relevance, not the RRF rank score.
    """
    texts = [(pt.payload or {}).get("text") or "" for pt in points]
    scores = reranker.score(query, texts)
    for pt, score in zip(points, scores):
        pt.score = float(score)
    ranked = sorted(points, key=lambda pt: pt.score, reverse=True)
    if min_score is not None:
        ranked = [pt for pt in ranked if pt.score >= min_score]
    return ranked[:top_k]
