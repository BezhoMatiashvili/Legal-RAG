"""Retrieval metrics: known-value checks, graded nDCG, breakdowns, latency percentiles."""

import math

from eval.metrics import (
    Hit,
    aggregate,
    breakdown,
    evidence_group_recall,
    percentiles,
    query_score,
    reduce_ranking,
    score_ranking,
)


def _hit(doc, ci, score=1.0, source="s", version=None):
    return Hit(source, doc, ci, score, version_id=version)


def test_reduce_ranking_dedupes_preserving_order():
    hits = [_hit("d1", 0), _hit("d1", 1), _hit("d2", 0), _hit("d1", 0)]
    assert reduce_ranking(hits, "doc") == [("s", "d1"), ("s", "d2")]
    assert reduce_ranking(hits, "chunk") == [
        ("s", "d1", 0), ("s", "d1", 1), ("s", "d2", 0),
    ]


def test_versioned_ranking_identity_preserves_legacy_shape_and_separates_versions():
    hits = [
        _hit("law", 0, version="v-current"),
        _hit("law", 0, version="v-repealed"),
        _hit("law", 0),
    ]
    assert reduce_ranking(hits, "chunk") == [
        ("s", "law", "v-current", 0),
        ("s", "law", "v-repealed", 0),
        ("s", "law", 0),
    ]
    assert reduce_ranking(hits, "doc") == [
        ("s", "law", "v-current"),
        ("s", "law", "v-repealed"),
        ("s", "law"),
    ]


def test_wrong_canonical_version_cannot_satisfy_document_or_passage_qrels():
    required = ("s", "law", "v-current", 0)
    groups = {"operative-rule": frozenset({required})}
    wrong = _hit("law", 0, version="v-repealed")

    chunk_score = query_score(
        "q",
        "historical",
        "ka",
        [wrong],
        {required: 2},
        "chunk",
        evidence_groups=groups,
    )
    assert chunk_score.success10 == 0.0
    assert chunk_score.required_evidence_recall10 == 0.0
    assert chunk_score.document_identity1 == 0.0
    assert chunk_score.passage_accuracy1 == 0.0
    assert chunk_score.context_noise10 == 1.0

    doc_score = query_score(
        "q",
        "historical",
        "ka",
        [wrong],
        {("s", "law", "v-current"): 2},
        "doc",
        evidence_groups=groups,
    )
    assert doc_score.success10 == 0.0
    assert doc_score.document_identity1 == 0.0

    correct = _hit("law", 0, version="v-current")
    correct_score = query_score(
        "q",
        "historical",
        "ka",
        [correct],
        {required: 2},
        "chunk",
        evidence_groups=groups,
    )
    assert correct_score.success10 == 1.0
    assert correct_score.document_identity1 == 1.0
    assert correct_score.passage_accuracy1 == 1.0


def test_recall_mrr_when_relevant_at_rank_3():
    ranked = [("s", "x"), ("s", "y"), ("s", "d")]
    relevant = {("s", "d"): 2}
    recalls, ndcg, mrr = score_ranking(ranked, relevant)
    assert recalls[5] == 1.0 and recalls[10] == 1.0
    assert mrr == 1.0 / 3


def test_recall_zero_when_not_in_topk():
    ranked = [("s", f"d{i}") for i in range(6)]
    relevant = {("s", "gold"): 2}
    recalls, ndcg, mrr = score_ranking(ranked, relevant, recall_ks=(5, 10))
    assert recalls[5] == 0.0 and mrr == 0.0 and ndcg == 0.0


def test_graded_ndcg_perfect_and_partial():
    relevant = {("s", "a"): 2, ("s", "b"): 1}
    # perfect ordering (higher grade first) → nDCG 1.0
    _, ndcg_perfect, _ = score_ranking([("s", "a"), ("s", "b")], relevant)
    assert math.isclose(ndcg_perfect, 1.0)
    # swapped ordering → strictly less than 1
    _, ndcg_swapped, _ = score_ranking([("s", "b"), ("s", "a")], relevant)
    assert ndcg_swapped < 1.0


def test_query_score_chunk_vs_doc_level():
    hits = [_hit("gold", 3), _hit("other", 0)]
    # chunk-level: relevant chunk is (s, gold, 3) → hit at rank 1
    rel_chunk = {("s", "gold", 3): 2}
    s = query_score("q1", "keyword", "ka", hits, rel_chunk, "chunk")
    assert s.success1 == 1.0 and s.recall5 == 1.0 and s.mrr10 == 1.0
    # chunk-level with the wrong chunk index → miss even though the doc matches
    rel_wrong = {("s", "gold", 9): 2}
    s2 = query_score("q1", "keyword", "ka", hits, rel_wrong, "chunk")
    assert s2.recall5 == 0.0
    # doc-level: any chunk of the gold doc counts
    s3 = query_score("q1", "keyword", "ka", hits, {("s", "gold"): 2}, "doc")
    assert s3.recall5 == 1.0


def test_aggregate_and_breakdown():
    scores = [
        query_score("a", "keyword", "ka", [_hit("g", 0)], {("s", "g", 0): 2}, "chunk"),
        query_score("b", "keyword", "en", [_hit("x", 0)], {("s", "g", 0): 2}, "chunk"),
    ]
    agg = aggregate(scores)
    assert agg["recall5"] == 0.5
    by_lang = breakdown(scores, "language")
    assert by_lang["ka"]["recall5"] == 1.0 and by_lang["en"]["recall5"] == 0.0
    assert by_lang["ka"]["n"] == 1


def test_percentiles():
    p = percentiles([0.0, 0.010, 0.020, 0.030, 0.040], ps=(50, 95))
    assert math.isclose(p["p50"], 0.020)
    assert p["p95"] > p["p50"]
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0}


def test_required_evidence_recall_uses_span_alternatives_not_chunk_denominator():
    hits = [_hit("d1", 7), _hit("noise", 0)]
    groups = {
        "rule": frozenset({("s", "d1", 6), ("s", "d1", 7)}),
        "exception": frozenset({("s", "d2", 2)}),
    }
    ranked = reduce_ranking(hits, "chunk")
    assert evidence_group_recall(ranked, groups, 10) == 0.5
    score = query_score(
        "q", "paraphrase", "ka", hits,
        {("s", "d1", 6): 2, ("s", "d1", 7): 2, ("s", "d2", 2): 2},
        "chunk", evidence_groups=groups,
    )
    assert score.success10 == 1.0
    assert score.required_evidence_recall10 == 0.5

    one_group = {"rule": groups["rule"]}
    one_span_score = query_score(
        "q2", "paraphrase", "ka", hits,
        {("s", "d1", 6): 2, ("s", "d1", 7): 2},
        "chunk", evidence_groups=one_group,
    )
    assert one_span_score.ndcg10 == 1.0  # alternatives count once in the ideal denominator


def test_document_passage_candidate_and_context_quality_metrics():
    hits = [_hit("gold", 1), _hit("gold", 2), _hit("noise", 0)]
    groups = {"span": frozenset({("s", "gold", 1), ("s", "gold", 2)})}
    score = query_score(
        "q", "keyword", "ka", hits,
        {("s", "gold", 1): 2, ("s", "gold", 2): 2},
        "chunk", evidence_groups=groups,
        candidate_recall={50: 1.0, 80: 1.0}, cluster_id="family-1",
    )
    assert score.document_identity1 == 1.0
    assert score.passage_accuracy1 == 1.0
    assert score.candidate_recall50 == score.candidate_recall80 == 1.0
    assert math.isclose(score.context_duplication10, 1 / 3)
    assert math.isclose(score.context_noise10, 1 / 3)
    assert score.cluster_id == "family-1"
