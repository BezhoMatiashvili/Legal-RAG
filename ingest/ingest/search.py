"""Hybrid (dense + sparse) retrieval via Qdrant's Query API with RRF fusion.

Used to verify ingestion quality and as a reusable retrieval helper.
"""

from qdrant_client import models

from .config import Config


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
    **filters,
):
    """Two-stage retrieval: hybrid (dense+sparse, RRF-fused) recall → optional rerank.

    Stage 1 over-fetches a candidate pool from Qdrant by RRF-fusing the dense and
    (when present) sparse matches. Stage 2, when a ``reranker`` is given, re-scores each
    candidate against the query with a cross-encoder, drops anything below
    ``rerank_min_score``, and returns the best ``top_k``. Without a reranker the fused
    RRF order is returned directly (CLI / offline use). Returned points carry the
    final ordering score in ``.score`` (rerank score when reranked, else RRF score).
    """
    emb = embedder.encode_query(query)
    flt = build_filter(**filters)

    # How many candidates to fuse before reranking. With a reranker we want a deep pool
    # so the cross-encoder has real recall to work with; without one, the old top_k*5/50.
    fetch_n = (rerank_candidates or max(top_k * 5, 50)) if reranker else top_k
    candidate = max(fetch_n, top_k * 5, 50)

    prefetch = [
        models.Prefetch(query=emb.dense, using="dense", limit=candidate, filter=flt)
    ]
    if emb.sparse.indices:
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
    )
    points = result.points
    if reranker is None or not points:
        return points[:top_k]

    return rerank_points(reranker, query, points, top_k=top_k, min_score=rerank_min_score)


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
