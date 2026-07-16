"""The streaming full-corpus BM25 index must score identically to the reference BM25Index.

FullCorpusBM25 exists only to make the mandatory BM25 floor tractable at 2.45M chunks; its
scoring must stay byte-for-byte faithful to eval.bm25.BM25Index (same tokenizer, IDF, k1/b).
These tests build both over the same tiny corpus (via a fake scroll client — no Qdrant) and
assert parity, plus the save/load round-trip and query-edge behaviour.
"""

import math

from eval.bm25 import BM25Index
from eval.bm25_full import FullCorpusBM25


class _Pt:
    def __init__(self, payload):
        self.payload = payload


class _Count:
    def __init__(self, n):
        self.count = n


class FakeClient:
    """Minimal Qdrant-scroll stand-in: deterministic order, int offset paging."""

    def __init__(self, docs):
        # docs: list[(source, document_id, chunk_index, text)]
        self._pts = [
            _Pt({"source": s, "document_id": d, "chunk_index": c, "text": t})
            for (s, d, c, t) in docs
        ]

    def scroll(self, collection_name, with_payload=True, with_vectors=False,
               limit=1000, offset=None):
        start = offset or 0
        end = min(start + limit, len(self._pts))
        nxt = end if end < len(self._pts) else None
        return self._pts[start:end], nxt

    def count(self, collection_name, exact=False):
        return _Count(len(self._pts))


# A corpus that exercises: term frequency, doc length, rare-vs-common IDF, repeated terms,
# multiple sources, and multi-chunk documents.
DOCS = [
    ("matsne", "1", 0, "labor code annual paid leave twenty four working days per year"),
    ("matsne", "1", 1, "labor code overtime pay compensation weekend rules"),
    ("ecd", "2", 0, "criminal code theft punishment imprisonment sentence"),
    ("napr", "3", 0, "civil code contract obligations between the parties"),
    ("matsne", "4", 0, "labor leave leave leave sabbatical"),
    ("tas", "5", 7, "tax code value added tax rate exemption"),
    ("constcourt", "6", 0, "constitution fundamental rights freedom of expression"),
]
KEYS = [(s, d, c) for (s, d, c, _t) in DOCS]
REF_PAIRS = [((s, d, c), t) for (s, d, c, t) in DOCS]

QUERIES = [
    "labor code leave",       # multi-term, hits several docs of varying length
    "leave",                  # concentrated in doc 4 (3x) and doc 0 (1x)
    "leave leave",            # repeated query term must be summed (matches reference)
    "criminal punishment",    # single doc
    "tax rate exemption",     # single doc, different source
    "code",                   # common term across many docs (low IDF)
    "labor zzzunknownzzz code",  # OOV term must be ignored, == "labor code"
]


def _build(tmp_path, batch=3):
    client = FakeClient(DOCS)
    # small batch forces multi-page scrolls in both passes → exercises paging/offset logic
    return FullCorpusBM25.build(client, "test_coll", index_dir=tmp_path / "idx", batch=batch)


def _as_scores(ranked):
    return {key: score for key, score in ranked}


def test_matches_reference_scores_and_ranking(tmp_path):
    ref = BM25Index.from_pairs(REF_PAIRS)
    full = _build(tmp_path)
    for q in QUERIES:
        r = ref.search(q, k=10)
        f = full.search(q, k=10)
        r_scores, f_scores = _as_scores(r), _as_scores(f)
        # same set of positive-scoring docs
        assert set(r_scores) == set(f_scores), q
        # scores agree (float32 storage tolerance)
        for key, rs in r_scores.items():
            assert math.isclose(rs, f_scores[key], rel_tol=1e-4, abs_tol=1e-4), (q, key)
        # ranking agrees where scores are not tied
        r_keys = [k for k, _ in r]
        f_keys = [k for k, _ in f]
        strict = all(
            not math.isclose(r[i][1], r[i + 1][1], rel_tol=1e-6) for i in range(len(r) - 1)
        )
        if strict:
            assert r_keys == f_keys, q


def test_repeated_query_term_is_summed(tmp_path):
    full = _build(tmp_path)
    once = _as_scores(full.search("leave", k=10))
    twice = _as_scores(full.search("leave leave", k=10))
    # doc 4 contains "leave" → its score must roughly double when the query term repeats
    key = ("matsne", "4", 0)
    assert math.isclose(twice[key], 2 * once[key], rel_tol=1e-4)


def test_oov_and_empty_queries(tmp_path):
    full = _build(tmp_path)
    assert full.search("zzzznotarealtoken", k=10) == []
    assert full.search("", k=10) == []
    # an OOV term alongside a known term behaves like the known term alone
    assert _as_scores(full.search("labor zzz code", k=10)) == _as_scores(
        full.search("labor code", k=10)
    )


def test_top_k_is_respected_and_sorted(tmp_path):
    full = _build(tmp_path)
    res = full.search("code", k=2)
    assert len(res) <= 2
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_keys_roundtrip_as_tuples(tmp_path):
    full = _build(tmp_path)
    res = full.search("labor code leave", k=10)
    for (source, document_id, chunk_index), _score in res:
        assert (source, document_id, chunk_index) in KEYS
        assert isinstance(chunk_index, int)


def test_save_load_roundtrip(tmp_path):
    built = _build(tmp_path)
    assert FullCorpusBM25.cache_exists(tmp_path / "idx")
    loaded = FullCorpusBM25.load(tmp_path / "idx")
    assert loaded.meta["N"] == len(DOCS)
    assert loaded.meta["V"] == built.meta["V"]
    for q in QUERIES:
        assert _as_scores(built.search(q, k=10)) == _as_scores(loaded.search(q, k=10))


def test_version_scoped_keys_survive_build_and_load(tmp_path):
    class VersionedClient(FakeClient):
        def __init__(self):
            self._pts = [
                _Pt(
                    {
                        "source": "matsne",
                        "document_id": "law",
                        "version_id": version,
                        "chunk_index": 0,
                        "text": text,
                    }
                )
                for version, text in (
                    ("v-current", "current operative labor rule"),
                    ("v-repealed", "repealed historical labor rule"),
                )
            ]

    built = FullCorpusBM25.build(
        VersionedClient(), "versioned", index_dir=tmp_path / "idx", batch=1
    )
    expected = {
        ("matsne", "law", "v-current", 0),
        ("matsne", "law", "v-repealed", 0),
    }
    assert {key for key, _score in built.search("labor rule", 10)} == expected

    loaded = FullCorpusBM25.load(tmp_path / "idx")
    assert {key for key, _score in loaded.search("labor rule", 10)} == expected


def test_build_or_load_uses_cache(tmp_path):
    _build(tmp_path)
    # build_or_load with a client that would explode if scrolled proves the cache is used
    class Boom:
        def scroll(self, *a, **k):
            raise AssertionError("should have loaded from cache, not rebuilt")

        def count(self, *a, **k):
            raise AssertionError("should have loaded from cache, not rebuilt")

    idx = FullCorpusBM25.build_or_load(Boom(), "test_coll", index_dir=tmp_path / "idx")
    assert idx.meta["N"] == len(DOCS)


# Tied scores that straddle the top-k cutoff: A/B/C are identical-length, identical-tf docs
# (bit-identical BM25 score for "code"); D scores lower. The retained subset AND order must
# match the reference's stable, lowest-index-first tie-breaking — not just have length <= k.
TIE_DOCS = [
    ("s", "A", 0, "code alpha"),
    ("s", "B", 0, "code beta"),
    ("s", "C", 0, "code gamma"),
    ("s", "D", 0, "code delta epsilon zeta theta"),  # longer → lower BM25 score
]


def test_tie_break_matches_reference_at_k_cutoff(tmp_path):
    ref = BM25Index.from_pairs([((s, d, c), t) for (s, d, c, t) in TIE_DOCS])
    full = FullCorpusBM25.build(FakeClient(TIE_DOCS), "c", index_dir=tmp_path / "idx", batch=2)
    for q, k in [("code", 1), ("code", 2), ("code", 3), ("code", 4)]:
        r = [key for key, _ in ref.search(q, k)]
        f = [key for key, _ in full.search(q, k)]
        assert r == f, (q, k, r, f)  # exact set AND order parity, including ties


def test_empty_corpus_degrades_gracefully(tmp_path):
    docs = [("s", "A", 0, ""), ("s", "B", 0, "   "), ("s", "C", 0, "!!! ???")]
    full = FullCorpusBM25.build(FakeClient(docs), "c", index_dir=tmp_path / "idx", batch=2)
    assert full.search("anything", 5) == []  # no crash, empty result — like the reference


def test_non_positive_k_returns_empty(tmp_path):
    full = _build(tmp_path)
    assert full.search("code", 0) == []
    assert full.search("code", -1) == []


def test_load_rejects_wrong_collection(tmp_path):
    FullCorpusBM25.build(FakeClient(DOCS), "coll_A", index_dir=tmp_path / "idx", batch=3)
    FullCorpusBM25.load(tmp_path / "idx", expect_collection="coll_A")  # ok
    import pytest
    with pytest.raises(RuntimeError, match="coll_A"):
        FullCorpusBM25.load(tmp_path / "idx", expect_collection="coll_B")
