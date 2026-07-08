"""Hand-rolled Okapi BM25 baseline + Georgian-aware tokenizer."""

from eval.bm25 import BM25Index, tokenize


def test_tokenize_georgian_and_english():
    assert tokenize("სიტყვა, მეორე. სამი!") == ["სიტყვა", "მეორე", "სამი"]
    assert tokenize("Article TWO and two") == ["article", "two", "and", "two"]  # casefold


def test_ranks_documents_by_term_frequency():
    idx = BM25Index.from_pairs([
        ("a", "მუხლი მუხლი მუხლი კანონი"),
        ("b", "მუხლი კანონი"),
        ("c", "სხვა ტექსტი სულ"),
    ])
    ranked = idx.search("მუხლი", k=10)
    ids = [i for i, _ in ranked]
    assert ids[0] == "a"          # more occurrences of the query term
    assert "c" not in ids         # no query term → score 0, excluded


def test_rare_terms_outweigh_common_terms():
    idx = BM25Index.from_pairs([
        ("d1", "საერთო საერთო იშვიათი"),
        ("d2", "საერთო საერთო საერთო"),
        ("d3", "საერთო"),
        ("d4", "საერთო"),
    ])
    # "იშვიათი" appears in only one doc → high idf; querying it should surface d1 first
    ranked = idx.search("იშვიათი", k=10)
    assert ranked[0][0] == "d1"


def test_topk_and_empty_query():
    idx = BM25Index.from_pairs([(i, f"ტოკ{i} საერთო") for i in range(10)])
    assert len(idx.search("საერთო", k=3)) == 3
    assert idx.search("არარსებული", k=5) == []   # unseen term
    assert idx.search("", k=5) == []              # empty query


def test_empty_index_is_safe():
    idx = BM25Index([], [])
    assert idx.search("რამე", k=5) == []
