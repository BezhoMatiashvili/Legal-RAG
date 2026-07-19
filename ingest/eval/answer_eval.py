"""Answer-quality metrics — the deterministic layer of the answer-correctness eval.

The retrieval harness (``eval/metrics.py``) scores whether the right *chunks* are retrieved.
These metrics score properties of the *answer a user would actually receive*, computed
deterministically from the retrieved+reranked results and the golden set — **no answer
generation and no LLM judge** — so they log and gate exactly like the retrieval metrics
(``experiments.jsonl`` / ``config_hash`` / gates G1–G5). They are the deterministic floor
beneath the approved manual faithfulness-review layer, per the quality plan §2a.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .goldset import GoldQuery
from .metrics import Hit, Key, hit_key, reduce_ranking

# Query types whose answer cites one specific document — where identity@1 is the meaningful
# answer-correctness signal. Mirrors the golden_set_v2 query_type vocabulary.
KNOWN_ITEM_TYPES: tuple[str, ...] = ("legal_citation", "keyword")

DEFAULT_ABSTENTION_THRESHOLD = 0.92  # from scripts/calibrate_min_score.py (top-1, v1, rc=50)


def gold_doc_key(q: GoldQuery) -> Key:
    """The legacy document or exact canonical version the answer should cite."""

    if q.gold_version_id is not None:
        return (q.gold_source, q.gold_document_id, q.gold_version_id)
    return (q.gold_source, q.gold_document_id)


def top1_doc(hits: Sequence[Hit]) -> Key | None:
    """The highest-ranked legacy document or canonical version identity, or ``None``."""
    order = reduce_ranking(hits, "doc")
    return order[0] if order else None


def top1_score(hits: Sequence[Hit]) -> float:
    """Reranker score (calibrated 0..1 in rerank mode) of the rank-1 hit; 0.0 if none."""
    return float(hits[0].score) if hits else 0.0


def _doc_ranked_with_scores(hits: Sequence[Hit]) -> list[tuple[Key, float]]:
    """Doc-reduced ranking WITH each doc's (best/first-seen-chunk) score, in rank order.

    ``metrics.reduce_ranking`` drops scores; rebuilt here so a margin can be computed
    between distinct DOCUMENTS, not between two chunks of the same top document (which
    would read as a false "tight margin" — hits are pre-sorted, so a doc's first
    occurrence is already its highest-scored chunk).
    """
    seen: set[Key] = set()
    out: list[tuple[Key, float]] = []
    for h in hits:
        key = hit_key(h, "doc")
        if key not in seen:
            seen.add(key)
            out.append((key, float(h.score)))
    return out


def top2_score(hits: Sequence[Hit]) -> float:
    """Reranker score of the SECOND distinct document in the ranking; 0.0 if none exists."""
    ranked = _doc_ranked_with_scores(hits)
    return ranked[1][1] if len(ranked) > 1 else 0.0


def top1_top2_margin(hits: Sequence[Hit]) -> float:
    """``top1_doc_score - top2_doc_score`` (doc-reduced, not raw chunk hits).

    When fewer than 2 distinct documents are retrieved there is no competing wrong
    document at all — returns ``top1_score`` itself as the margin (an unambiguous top-1
    reads as a maximal, not zero, margin under this convention).
    """
    ranked = _doc_ranked_with_scores(hits)
    if len(ranked) < 2:
        return ranked[0][1] if ranked else 0.0
    return ranked[0][1] - ranked[1][1]


def confident_wrong(hits: Sequence[Hit], q: GoldQuery, threshold: float = DEFAULT_ABSTENTION_THRESHOLD) -> bool:
    """Rank-1 hit scores ABOVE the abstention threshold yet is NOT the gold document.

    This is the demonstrated legal hallucination mode ("high score, wrong law/case"): a
    score-only abstention gate at ``threshold`` would confidently serve this wrong document.
    Only meaningful when the score is calibrated (rerank mode). Restricted to KNOWN-ITEM
    queries, where "the wrong document at rank 1" is unambiguously an error to catch."""
    return (
        q.query_type in KNOWN_ITEM_TYPES
        and top1_score(hits) >= threshold
        and not identity_at_1(hits, q)
    )


def identity_at_1(hits: Sequence[Hit], q: GoldQuery) -> bool:
    """Is the rank-1 retrieved document the exact gold document? (answer would cite it)."""
    return top1_doc(hits) == gold_doc_key(q)


def identity_at_k(hits: Sequence[Hit], q: GoldQuery, k: int = 10) -> bool:
    """Is the gold document anywhere in the top-k? (document Success@k; for context)."""
    order = reduce_ranking(hits, "doc")[:k]
    return gold_doc_key(q) in order


def _evidence_groups(gold_evidence) -> list[set[Key]]:
    """Normalise new evidence-group mappings and the legacy flattened chunk-set API."""
    if isinstance(gold_evidence, Mapping):
        return [set(alternatives) for alternatives in gold_evidence.values()]
    # A legacy set had no way to distinguish separate spans from overlapping alternatives;
    # retain its historical each-key-required semantics for API compatibility.  The harness
    # always passes the new mapping and therefore gets correct evidence-unit semantics.
    if isinstance(gold_evidence, (set, frozenset)):
        return [{key} for key in gold_evidence]
    return [set(alternatives) for alternatives in gold_evidence]


def span_coverage_at_k(hits: Sequence[Hit], gold_evidence, k: int = 10) -> float:
    """Fraction of required evidence groups satisfied by any alternative chunk at k."""
    groups = _evidence_groups(gold_evidence)
    if not groups:
        return 0.0
    retrieved = set(reduce_ranking(hits, "chunk")[:k])
    covered = sum(1 for alternatives in groups if alternatives & retrieved)
    return covered / len(groups)


def fully_grounded_at_k(hits: Sequence[Hit], gold_evidence, k: int = 10) -> bool:
    """Are ALL gold spans present in the top-k context (answer can be fully grounded)?"""
    groups = _evidence_groups(gold_evidence)
    return bool(groups) and span_coverage_at_k(hits, groups, k) >= 1.0


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
    top1_doc: Key | None
    gold_doc: Key
    top1_score: float
    top2_score: float
    margin: float
    confident_wrong: bool
    confidence_threshold: float


def score_answer(
    q: GoldQuery, hits: Sequence[Hit], gold_evidence, *,
    k: int = 10, threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
) -> AnswerScore:
    """Compute all per-query deterministic answer signals for one query."""
    return AnswerScore(
        id=q.id,
        query_type=q.query_type,
        language=q.query_language,
        is_known_item=q.query_type in KNOWN_ITEM_TYPES,
        identity_at_1=identity_at_1(hits, q),
        identity_at_10=identity_at_k(hits, q, k),
        span_coverage_at_10=span_coverage_at_k(hits, gold_evidence, k),
        fully_grounded_at_10=fully_grounded_at_k(hits, gold_evidence, k),
        top1_doc=top1_doc(hits),
        gold_doc=gold_doc_key(q),
        top1_score=top1_score(hits),
        top2_score=top2_score(hits),
        margin=top1_top2_margin(hits),
        confident_wrong=confident_wrong(hits, q, threshold),
        confidence_threshold=threshold,
    )


def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def aggregate_answer_scores(scores: Sequence[AnswerScore]) -> dict[str, float | int]:
    """Overall means plus a known-item-only slice (where identity@1 is meaningful)."""
    known = [s for s in scores if s.is_known_item]
    # Margin-separation diagnostic: does a thin top1-top2 margin distinguish confident-wrong
    # hits from confident-CORRECT ones? Only meaningful over the same known-item, "confident"
    # (score >= threshold) population on both sides — comparing wrong-but-confident vs
    # right-but-confident, not vs the whole set (which would mix in low-score misses).
    confident_known = [s for s in known if s.top1_score >= s.confidence_threshold]
    confident_wrong_group = [s for s in confident_known if not s.identity_at_1]
    confident_correct_group = [s for s in confident_known if s.identity_at_1]
    return {
        "n": len(scores),
        "identity_at_1": _mean([1.0 if s.identity_at_1 else 0.0 for s in scores]),
        "identity_at_10": _mean([1.0 if s.identity_at_10 else 0.0 for s in scores]),
        "span_coverage_at_10": _mean([s.span_coverage_at_10 for s in scores]),
        "fully_grounded_at_10": _mean([1.0 if s.fully_grounded_at_10 else 0.0 for s in scores]),
        "n_known_item": len(known),
        "known_item_identity_at_1": _mean([1.0 if s.identity_at_1 else 0.0 for s in known]),
        "known_item_identity_at_10": _mean([1.0 if s.identity_at_10 else 0.0 for s in known]),
        # confident-wrong = rank-1 scores >= threshold but is NOT the gold doc (the
        # "confident hallucination" rate a score-only abstention gate would let through).
        # Only set True for known-item queries → rate is over the known-item denominator.
        "known_item_confident_wrong": _mean([1.0 if s.confident_wrong else 0.0 for s in known]),
        "mean_top1_score": _mean([s.top1_score for s in scores]),
        "n_confident_wrong": len(confident_wrong_group),
        "n_confident_correct": len(confident_correct_group),
        "mean_margin_confident_wrong": _mean([s.margin for s in confident_wrong_group]),
        "mean_margin_confident_correct": _mean([s.margin for s in confident_correct_group]),
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
