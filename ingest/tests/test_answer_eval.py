"""Offline unit tests for the deterministic answer-quality metrics (eval/answer_eval.py).

Pure synthetic data — no Qdrant, no torch, no golden-set file. Exercises every metric and
its edge cases so the answer-eval logs/gates on numbers we've pinned by construction.
"""

from eval.answer_eval import (
    KNOWN_ITEM_TYPES,
    AbstentionResult,
    abstention_correctness,
    aggregate_answer_scores,
    breakdown_by,
    confident_wrong,
    fully_grounded_at_k,
    identity_at_1,
    identity_at_k,
    identity_failures,
    score_answer,
    span_coverage_at_k,
    top1_doc,
    top1_score,
    top1_top2_margin,
    top2_score,
)
from eval.goldset import GoldQuery
from eval.metrics import Hit


def _gold(
    qid="q1",
    qtype="legal_citation",
    lang="ka",
    gsrc="matsne",
    gdoc="D1",
    gversion=None,
) -> GoldQuery:
    return GoldQuery(
        id=qid,
        query="…",
        query_type=qtype,
        query_language=lang,
        source=gsrc,
        document_id=gdoc,
        gold_source=gsrc,
        gold_document_id=gdoc,
        gold_version_id=gversion,
    )


def _hit(doc="D1", chunk=0, score=0.9, source="matsne", version=None) -> Hit:
    return Hit(
        source=source,
        document_id=doc,
        chunk_index=chunk,
        score=score,
        version_id=version,
    )


# ---- citation-identity ------------------------------------------------------------------


def test_top1_doc_and_identity_at_1_hit():
    hits = [_hit("D1", 0), _hit("D2", 0)]
    assert top1_doc(hits) == ("matsne", "D1")
    assert identity_at_1(hits, _gold(gdoc="D1")) is True


def test_identity_at_1_wrong_top_but_gold_lower():
    # gold D1 sits at rank 2 → identity@1 fails (answer would cite the wrong D9) but @10 holds.
    hits = [_hit("D9", 0), _hit("D1", 0)]
    g = _gold(gdoc="D1")
    assert identity_at_1(hits, g) is False
    assert identity_at_k(hits, g, k=10) is True


def test_identity_at_k_outside_topk():
    hits = [_hit(f"X{i}", 0) for i in range(10)] + [_hit("D1", 0)]
    g = _gold(gdoc="D1")
    assert identity_at_k(hits, g, k=10) is False
    assert identity_at_k(hits, g, k=11) is True


def test_identity_ignores_chunk_order_within_same_doc():
    # multiple chunks of the same top doc collapse to one doc key
    hits = [_hit("D1", 3), _hit("D1", 0), _hit("D2", 0)]
    assert top1_doc(hits) == ("matsne", "D1")


def test_empty_hits():
    assert top1_doc([]) is None
    assert identity_at_1([], _gold()) is False
    assert identity_at_k([], _gold(), k=10) is False


def test_source_is_part_of_identity():
    # same document_id under a different source is NOT the gold doc
    hits = [_hit("D1", 0, source="napr")]
    assert identity_at_1(hits, _gold(gsrc="matsne", gdoc="D1")) is False


def test_canonical_version_is_part_of_answer_identity_and_span_coverage():
    wrong = [_hit("D1", 0, version="v-repealed")]
    gold = _gold(gdoc="D1", gversion="v-current")
    groups = {"rule": {("matsne", "D1", "v-current", 0)}}
    assert top1_doc(wrong) == ("matsne", "D1", "v-repealed")
    assert identity_at_1(wrong, gold) is False
    assert identity_at_k(wrong, gold) is False
    assert span_coverage_at_k(wrong, groups) == 0.0

    correct = [_hit("D1", 0, version="v-current")]
    assert identity_at_1(correct, gold) is True
    assert span_coverage_at_k(correct, groups) == 1.0


# ---- context-sufficiency / answerability@k ----------------------------------------------


def test_span_coverage_full_partial_empty():
    hits = [_hit("D1", 0), _hit("D1", 1), _hit("D2", 0)]
    gold_chunks = {("matsne", "D1", 0), ("matsne", "D1", 1)}
    assert span_coverage_at_k(hits, gold_chunks, k=10) == 1.0
    assert fully_grounded_at_k(hits, gold_chunks, k=10) is True

    partial = {("matsne", "D1", 0), ("matsne", "D1", 9)}  # chunk 9 not retrieved
    assert span_coverage_at_k(hits, partial, k=10) == 0.5
    assert fully_grounded_at_k(hits, partial, k=10) is False

    assert span_coverage_at_k(hits, set(), k=10) == 0.0
    assert fully_grounded_at_k(hits, set(), k=10) is False


def test_span_coverage_respects_k():
    hits = [_hit("A", 0), _hit("B", 0), _hit("D1", 0)]  # gold chunk at rank 3
    gold_chunks = {("matsne", "D1", 0)}
    assert span_coverage_at_k(hits, gold_chunks, k=2) == 0.0
    assert span_coverage_at_k(hits, gold_chunks, k=3) == 1.0


def test_span_alternative_chunks_form_one_required_evidence_group():
    hits = [_hit("D1", 1)]
    groups = {
        "operative-rule": {
            ("matsne", "D1", 0),
            ("matsne", "D1", 1),
            ("matsne", "D1", 2),
        }
    }
    assert span_coverage_at_k(hits, groups, k=10) == 1.0
    assert fully_grounded_at_k(hits, groups, k=10) is True


# ---- per-query + aggregation ------------------------------------------------------------


def test_score_answer_known_item_flag():
    s = score_answer(_gold(qtype="legal_citation", gdoc="D1"), [_hit("D1", 0)], {("matsne", "D1", 0)})
    assert s.is_known_item is True
    assert s.identity_at_1 is True
    assert s.fully_grounded_at_10 is True

    s2 = score_answer(_gold(qid="q2", qtype="paraphrase", gdoc="D1"), [_hit("D2", 0)], set())
    assert s2.is_known_item is False  # paraphrase is not a known-item type
    assert s2.identity_at_1 is False


def test_known_item_types_are_citation_and_keyword():
    assert set(KNOWN_ITEM_TYPES) == {"legal_citation", "keyword"}


def test_aggregate_overall_and_known_item_slice():
    scores = [
        score_answer(_gold("q1", "legal_citation", gdoc="D1"), [_hit("D1", 0)], {("matsne", "D1", 0)}),
        score_answer(_gold("q2", "legal_citation", gdoc="D2"), [_hit("DX", 0)], {("matsne", "D2", 0)}),
        score_answer(_gold("q3", "natural_question", gdoc="D3"), [_hit("D3", 0)], {("matsne", "D3", 0)}),
    ]
    agg = aggregate_answer_scores(scores)
    assert agg["n"] == 3
    assert agg["n_known_item"] == 2
    # 1 of 2 known-item queries has correct rank-1 identity
    assert agg["known_item_identity_at_1"] == 0.5
    # overall identity@1: q1 (D1) and q3 (D3) correct, q2 wrong → 2/3
    assert abs(agg["identity_at_1"] - 2 / 3) < 1e-9


def test_identity_failures_lists_only_known_item_misses():
    scores = [
        score_answer(_gold("q1", "legal_citation", gdoc="D1"), [_hit("WRONG", 0)], set()),
        score_answer(_gold("q2", "keyword", gdoc="D2"), [_hit("D2", 0)], set()),
        score_answer(_gold("q3", "paraphrase", gdoc="D3"), [_hit("NOPE", 0)], set()),  # not known-item
    ]
    fails = identity_failures(scores)
    assert len(fails) == 1
    assert fails[0]["id"] == "q1"
    assert fails[0]["got_top1"] == ("matsne", "WRONG")
    assert fails[0]["gold_doc"] == ("matsne", "D1")


def test_breakdown_by_query_type():
    scores = [
        score_answer(_gold("q1", "legal_citation", gdoc="D1"), [_hit("D1", 0)], set()),
        score_answer(_gold("q2", "keyword", gdoc="D2"), [_hit("DX", 0)], set()),
    ]
    bd = breakdown_by(scores, "query_type")
    assert set(bd) == {"legal_citation", "keyword"}
    assert bd["legal_citation"]["identity_at_1"] == 1.0
    assert bd["keyword"]["identity_at_1"] == 0.0


# ---- abstention -------------------------------------------------------------------------


def test_abstention_perfect_separation():
    r = abstention_correctness([0.99, 0.95, 0.93], [0.10, 0.50, 0.80], threshold=0.92)
    assert isinstance(r, AbstentionResult)
    assert r.retention == 1.0          # all answerable kept
    assert r.correct_refusal == 1.0    # all unanswerable refused
    assert r.false_refusal == 0.0
    assert r.false_accept == 0.0
    assert r.n_answerable == 3 and r.n_unanswerable == 3


def test_abstention_threshold_boundary_is_inclusive_for_answered():
    # score == threshold counts as "answered" (>= threshold)
    r = abstention_correctness([0.92], [0.92], threshold=0.92)
    assert r.retention == 1.0        # answerable at exactly threshold is retained
    assert r.correct_refusal == 0.0  # unanswerable at exactly threshold is (wrongly) answered
    assert r.false_accept == 1.0


def test_abstention_mixed_rates():
    # 1 of 2 answerable below → retention 0.5; 1 of 2 unanswerable below → correct_refusal 0.5
    r = abstention_correctness([0.95, 0.40], [0.30, 0.99], threshold=0.92)
    assert r.retention == 0.5
    assert r.correct_refusal == 0.5
    assert r.false_refusal == 0.5
    assert r.false_accept == 0.5


def test_abstention_empty_groups_do_not_crash():
    r = abstention_correctness([], [], threshold=0.92)
    assert r.n_answerable == 0 and r.n_unanswerable == 0
    assert r.retention == 0.0 and r.correct_refusal == 0.0
    assert r.false_refusal == 0.0 and r.false_accept == 0.0


# ---- confident-wrong (the demonstrated "high score, wrong doc" hallucination) ----------


def test_top1_score():
    assert top1_score([_hit("D1", 0, score=0.88)]) == 0.88
    assert top1_score([]) == 0.0


def test_confident_wrong_known_item_high_score_wrong_doc():
    hits = [_hit("WRONG", 0, score=0.99)]  # rank-1 confident but the wrong law
    assert confident_wrong(hits, _gold(qtype="legal_citation", gdoc="D1"), threshold=0.92) is True


def test_confident_wrong_negatives():
    g = _gold(qtype="legal_citation", gdoc="D1")
    assert confident_wrong([_hit("D1", 0, score=0.99)], g, 0.92) is False       # right doc
    assert confident_wrong([_hit("WRONG", 0, score=0.50)], g, 0.92) is False     # low score → gate abstains
    g2 = _gold(qtype="paraphrase", gdoc="D1")                                    # not a known-item type
    assert confident_wrong([_hit("WRONG", 0, score=0.99)], g2, 0.92) is False
    assert confident_wrong([], g, 0.92) is False                                 # no hits


def test_aggregate_includes_confident_wrong_and_mean_score():
    scores = [
        score_answer(_gold("q1", "legal_citation", gdoc="D1"), [_hit("WRONG", 0, score=0.99)], set(), threshold=0.92),
        score_answer(_gold("q2", "legal_citation", gdoc="D2"), [_hit("D2", 0, score=0.95)], set(), threshold=0.92),
    ]
    agg = aggregate_answer_scores(scores)
    assert agg["known_item_confident_wrong"] == 0.5           # q1 confident-wrong, q2 correct
    assert abs(agg["mean_top1_score"] - 0.97) < 1e-9


# ---- top1/top2 margin (doc-reduced, not raw chunk hits) ---------------------------------


def test_top2_score_is_doc_reduced_not_chunk_reduced():
    # Two chunks of the SAME top document must not read as a "second candidate" — the real
    # second candidate is D2, scored lower.
    hits = [_hit("D1", 0, score=0.95), _hit("D1", 1, score=0.93), _hit("D2", 0, score=0.40)]
    assert top2_score(hits) == 0.40
    assert abs(top1_top2_margin(hits) - (0.95 - 0.40)) < 1e-9


def test_top2_score_and_margin_with_fewer_than_two_docs():
    hits = [_hit("D1", 0, score=0.95), _hit("D1", 1, score=0.90)]  # only one distinct doc
    assert top2_score(hits) == 0.0
    assert top1_top2_margin(hits) == 0.95  # no competing doc → margin reads as maximal
    assert top2_score([]) == 0.0
    assert top1_top2_margin([]) == 0.0


def test_margin_separates_tight_wrong_from_wide_correct():
    hits_wrong_tight = [_hit("WRONG", 0, score=0.93), _hit("D1", 0, score=0.92)]  # gold D1 barely 2nd
    hits_correct_wide = [_hit("D1", 0, score=0.99), _hit("OTHER", 0, score=0.30)]
    assert top1_top2_margin(hits_wrong_tight) < top1_top2_margin(hits_correct_wide)


def test_aggregate_reports_margin_by_confidence_and_correctness():
    # q1: confident + wrong, tight margin. q2: confident + correct, wide margin.
    scores = [
        score_answer(
            _gold("q1", "legal_citation", gdoc="D1"),
            [_hit("WRONG", 0, score=0.95), _hit("D1", 0, score=0.94)],
            set(), threshold=0.92,
        ),
        score_answer(
            _gold("q2", "legal_citation", gdoc="D2"),
            [_hit("D2", 0, score=0.98), _hit("OTHER", 0, score=0.20)],
            set(), threshold=0.92,
        ),
    ]
    agg = aggregate_answer_scores(scores)
    assert agg["n_confident_wrong"] == 1
    assert agg["n_confident_correct"] == 1
    assert abs(agg["mean_margin_confident_wrong"] - 0.01) < 1e-9
    assert abs(agg["mean_margin_confident_correct"] - 0.78) < 1e-9
    assert agg["mean_margin_confident_wrong"] < agg["mean_margin_confident_correct"]


def test_margin_aggregation_uses_the_score_specific_threshold():
    scores = [
        score_answer(
            _gold("q1", "legal_citation", gdoc="D1"),
            [_hit("WRONG", 0, score=0.60), _hit("D1", 0, score=0.55)],
            set(), threshold=0.50,
        ),
        score_answer(
            _gold("q2", "legal_citation", gdoc="D2"),
            [_hit("D2", 0, score=0.70), _hit("OTHER", 0, score=0.20)],
            set(), threshold=0.50,
        ),
    ]
    agg = aggregate_answer_scores(scores)
    assert agg["n_confident_wrong"] == 1
    assert agg["n_confident_correct"] == 1
