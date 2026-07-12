"""Answer-quality metrics — the deterministic layer of the answer-correctness eval.

The retrieval harness (``eval/metrics.py``) scores whether the right *chunks* are retrieved.
These metrics score properties of the *answer a user would actually receive*, computed
deterministically from the retrieved+reranked results and the golden set — **no answer
generation and no LLM judge** — so they log and gate exactly like the retrieval metrics
(``experiments.jsonl`` / ``config_hash`` / gates G1–G5). They are the deterministic floor
beneath the (approved, Claude-in-session) faithfulness layer, per the quality plan §2a.

Three families, all pure functions over already-computed inputs (ranked ``Hit``s and the
``{key: grade}`` relevance dict the retrieval harness already builds), so they unit-test
offline with synthetic data — no Qdrant, no torch:

* **citation-identity correctness** — for *known-item* queries (a specific named law/order/
  number), is the **top** result actually the named/gold document, not merely *a* related
  one? This is the deterministic guard against the legal "real citation → wrong law" failure
  (Stanford HAI): when the client composes an answer it cites the top hit, so a wrong rank-1
  document is a wrong answer even if the right document sits at rank 3. Stricter and more
  answer-predictive than doc-``Recall@10``; its per-query failure list is the actionable
  output ("these citation queries surface the wrong law at rank 1").

* **context-sufficiency / answerability@k** — are *all* the gold evidence spans present in
  the top-k retrieved context (can the answer be fully grounded)? The coverage complement to
  any-hit ``Recall``. NOTE: at ``golden_set_v2`` (≈ one gold span/doc per query) this
  collapses toward chunk-``Recall``; it becomes a distinct multi-doc metric only once v3 adds
  multi-span / multi-doc queries. Implemented generally so it is correct the moment that
  data lands.

* **abstention correctness** — given top-1 reranker scores for a set of KNOWN-UNANSWERABLE
  queries (law absent from the corpus) and for the answerable golden queries, does the ~0.92
  abstention threshold correctly refuse the unanswerable ones while retaining the answerable
  ones? Turns the advisory 0.92 (``scripts/calibrate_min_score.py``) into a measured
  refusal/retention pair. The metric machinery is here; the held-out unanswerable-query
  artifact is authored separately (plan §2a).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .goldset import GoldQuery
from .metrics import Hit, Key, reduce_ranking

# Query types whose answer cites one specific document — where identity@1 is the meaningful
# answer-correctness signal. Mirrors the golden_set_v2 query_type vocabulary.
KNOWN_ITEM_TYPES: tuple[str, ...] = ("legal_citation", "keyword")

DEFAULT_ABSTENTION_THRESHOLD = 0.92  # from scripts/calibrate_min_score.py (top-1, v1, rc=50)


def gold_doc_key(q: GoldQuery) -> tuple[str, str]:
    """The ``(source, document_id)`` the query's answer should cite."""
    return (q.gold_source, q.gold_document_id)


def top1_doc(hits: Sequence[Hit]) -> tuple[str, str] | None:
    """The ``(source, document_id)`` of the highest-ranked retrieved document, or None."""
    order = reduce_ranking(hits, "doc")
    return order[0] if order else None  # reduce_ranking yields 2-tuples at doc level


def identity_at_1(hits: Sequence[Hit], q: GoldQuery) -> bool:
    """Is the rank-1 retrieved document the exact gold document? (answer would cite it)."""
    return top1_doc(hits) == gold_doc_key(q)


def identity_at_k(hits: Sequence[Hit], q: GoldQuery, k: int = 10) -> bool:
    """Is the gold document anywhere in the top-k? (== doc-level Recall@k; for context)."""
    order = reduce_ranking(hits, "doc")[:k]
    return gold_doc_key(q) in order


def span_coverage_at_k(hits: Sequence[Hit], gold_chunks: set[Key], k: int = 10) -> float:
    """Fraction of the gold chunk set present in the retrieved top-k (mean coverage).

    ``gold_chunks`` is the set of ``(source, document_id, chunk_index)`` the gold spans map
    to under the current chunk config (the same relevance keys the retrieval harness builds).
    Empty gold set → 0.0 (nothing to ground).
    """
    if not gold_chunks:
        return 0.0
    retrieved = set(reduce_ranking(hits, "chunk")[:k])
    covered = sum(1 for gk in gold_chunks if gk in retrieved)
    return covered / len(gold_chunks)


def fully_grounded_at_k(hits: Sequence[Hit], gold_chunks: set[Key], k: int = 10) -> bool:
    """Are ALL gold spans present in the top-k context (answer can be fully grounded)?"""
    return bool(gold_chunks) and span_coverage_at_k(hits, gold_chunks, k) >= 1.0


@dataclass(frozen=True)
class AnswerScore:
    """Per-query deterministic answer-correctness signals."""

    id: str
    query_type: str
    language: str
    is_known_item: bool
    identity_at_1: bool
    identity_at_10: bool
    span_coverage_at_10: float
    fully_grounded_at_10: bool
    top1_doc: tuple[str, str] | None
    gold_doc: tuple[str, str]


def score_answer(q: GoldQuery, hits: Sequence[Hit], gold_chunks: set[Key], *, k: int = 10) -> AnswerScore:
    """Compute all per-query deterministic answer signals for one query."""
    return AnswerScore(
        id=q.id,
        query_type=q.query_type,
        language=q.query_language,
        is_known_item=q.query_type in KNOWN_ITEM_TYPES,
        identity_at_1=identity_at_1(hits, q),
        identity_at_10=identity_at_k(hits, q, k),
        span_coverage_at_10=span_coverage_at_k(hits, gold_chunks, k),
        fully_grounded_at_10=fully_grounded_at_k(hits, gold_chunks, k),
        top1_doc=top1_doc(hits),
        gold_doc=gold_doc_key(q),
    )


def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def aggregate_answer_scores(scores: Sequence[AnswerScore]) -> dict[str, float | int]:
    """Overall means plus a known-item-only slice (where identity@1 is meaningful)."""
    known = [s for s in scores if s.is_known_item]
    return {
        "n": len(scores),
        "identity_at_1": _mean([1.0 if s.identity_at_1 else 0.0 for s in scores]),
        "identity_at_10": _mean([1.0 if s.identity_at_10 else 0.0 for s in scores]),
        "span_coverage_at_10": _mean([s.span_coverage_at_10 for s in scores]),
        "fully_grounded_at_10": _mean([1.0 if s.fully_grounded_at_10 else 0.0 for s in scores]),
        "n_known_item": len(known),
        "known_item_identity_at_1": _mean([1.0 if s.identity_at_1 else 0.0 for s in known]),
        "known_item_identity_at_10": _mean([1.0 if s.identity_at_10 else 0.0 for s in known]),
    }


def breakdown_by(scores: Sequence[AnswerScore], attr: str) -> dict[str, dict[str, float | int]]:
    """Aggregate answer signals grouped by ``query_type`` or ``language``."""
    groups: dict[str, list[AnswerScore]] = {}
    for s in scores:
        groups.setdefault(getattr(s, attr), []).append(s)
    return {g: aggregate_answer_scores(gs) for g, gs in sorted(groups.items())}


def identity_failures(scores: Sequence[AnswerScore]) -> list[dict[str, object]]:
    """Known-item queries whose rank-1 document is NOT the gold document — the actionable list.

    Each entry surfaces the wrong top document so a human can inspect the "wrong law" cases.
    """
    return [
        {"id": s.id, "query_type": s.query_type, "gold_doc": s.gold_doc, "got_top1": s.top1_doc}
        for s in scores
        if s.is_known_item and not s.identity_at_1
    ]


@dataclass(frozen=True)
class AbstentionResult:
    """Measured abstention behaviour at a threshold (retrieval-score based)."""

    threshold: float
    n_answerable: int
    n_unanswerable: int
    retention: float        # frac of answerable queries with top-1 >= threshold (kept — good)
    correct_refusal: float  # frac of unanswerable queries with top-1 < threshold (abstained — good)
    false_refusal: float    # frac of answerable wrongly below threshold (would over-abstain)
    false_accept: float     # frac of unanswerable wrongly at/above threshold (would answer anyway)


def abstention_correctness(
    answerable_top1: Sequence[float],
    unanswerable_top1: Sequence[float],
    threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
) -> AbstentionResult:
    """Measure how well ``threshold`` separates answerable from known-unanswerable queries.

    Inputs are the top-1 reranker scores (0..1 sigmoid) for each group. A query is "answered"
    iff its top-1 score is ``>= threshold``. Good outcomes: answerable answered (retention),
    unanswerable refused (correct_refusal).
    """
    n_ans = len(answerable_top1)
    n_un = len(unanswerable_top1)
    retention = _mean([1.0 if s >= threshold else 0.0 for s in answerable_top1])
    correct_refusal = _mean([1.0 if s < threshold else 0.0 for s in unanswerable_top1])
    return AbstentionResult(
        threshold=threshold,
        n_answerable=n_ans,
        n_unanswerable=n_un,
        retention=retention,
        correct_refusal=correct_refusal,
        false_refusal=(1.0 - retention) if n_ans else 0.0,
        false_accept=(1.0 - correct_refusal) if n_un else 0.0,
    )
