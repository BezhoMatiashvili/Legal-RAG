"""Retrieval metrics: known-value checks, graded nDCG, breakdowns, latency percentiles."""

import math

from eval.metrics import (
    Hit,
    aggregate,
    breakdown,
    percentiles,
    query_score,
    reduce_ranking,
    score_ranking,
)


def _hit(doc, ci, score=1.0, source="s"):
    return Hit(source, doc, ci, score)


def test_reduce_ranking_dedupes_preserving_order():
    hits = [_hit("d1", 0), _hit("d1", 1), _hit("d2", 0), _hit("d1", 0)]
    assert reduce_ranking(hits, "doc") == [("s", "d1"), ("s", "d2")]
    assert reduce_ranking(hits, "chunk") == [
        ("s", "d1", 0), ("s", "d1", 1), ("s", "d2", 0),
    ]


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
    assert s.recall5 == 1.0 and s.mrr10 == 1.0
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
