"""Part-3 harness backend tests: routed mode + rerank-depth/fusion/search-params/diversity knobs.

Uses a fake Qdrant client that records ``query_points`` kwargs so we can assert exactly what
the backend asked the server for, with no torch/Qdrant.
"""

from types import SimpleNamespace

from qdrant_client import models

from eval.backend import MODES, ChunkRecord, FakeBackend, QdrantBackend
from ingest.config import load_config


class FakeEmb:
    def encode_query(self, q):
        return SimpleNamespace(
            dense=[0.1, 0.2, 0.3],
            sparse=SimpleNamespace(indices=[1, 2], values=[0.5, 0.5]),
        )


def _pt(doc, ci, score, *, version=None):
    payload = {
        "document_id": doc,
        "source": "matsne",
        "chunk_index": ci,
        "text": f"{doc}-{ci}",
    }
    if version is not None:
        payload["version_id"] = version
    return SimpleNamespace(
        payload=payload,
        score=score, vector={"dense": [0.1, 0.2, 0.3]},
    )


class FakeQdrant:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def query_points(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(points=list(self.points))


def _cfg():
    return load_config()


def test_routed_in_modes():
    assert "routed" in MODES


def test_fake_backend_handles_every_mode_including_routed():
    recs = [ChunkRecord("matsne", "d1", 0, "the law of georgia article one"),
            ChunkRecord("matsne", "d2", 0, "something entirely unrelated")]
    fb = FakeBackend(recs)
    for m in MODES:
        hits, lat = fb.search("law georgia", m, 2)
        assert isinstance(hits, list)


def test_backend_hits_retain_optional_canonical_version_identity():
    records = [ChunkRecord("matsne", "law", 0, "operative rule", version_id="v1")]
    fake_hits, _ = FakeBackend(records).search("operative", "bm25", 1)
    assert fake_hits[0].version_id == "v1"

    client = FakeQdrant([_pt("law", 0, 0.9, version="v1")])
    qdrant_hits, _ = QdrantBackend(_cfg(), client, FakeEmb()).search("operative", "dense", 1)
    assert qdrant_hits[0].version_id == "v1"


def test_qdrant_routed_drops_sparse_for_english():
    client = FakeQdrant([_pt("A", 0, 0.9)])
    QdrantBackend(_cfg(), client, FakeEmb()).search("hello world", "routed", 5)
    prefetch = client.calls[-1]["prefetch"]
    assert [p.using for p in prefetch] == ["dense"]


def test_qdrant_routed_keeps_sparse_for_georgian():
    client = FakeQdrant([_pt("A", 0, 0.9)])
    QdrantBackend(_cfg(), client, FakeEmb()).search("გამარჯობა მსოფლიო", "routed", 5)
    prefetch = client.calls[-1]["prefetch"]
    assert {p.using for p in prefetch} == {"dense", "sparse"}


def test_rerank_pool_depth_is_faithful():
    client = FakeQdrant([_pt("A", i, 0.9 - i * 0.001) for i in range(80)])

    class FakeRr:
        def score(self, q, texts):
            return [1.0 / (i + 1) for i in range(len(texts))]

    QdrantBackend(_cfg(), client, FakeEmb(), reranker=FakeRr(), rerank_candidates=30).search(
        "hello", "rerank", 10)
    kw = client.calls[-1]
    assert kw["limit"] == 30                 # not masked up to 50 by the old floor
    assert kw["prefetch"][0].limit == 30


def test_fusion_flag_selects_dbsf():
    client = FakeQdrant([_pt("A", 0, 0.9)])
    QdrantBackend(_cfg(), client, FakeEmb(), fusion="dbsf").search("hello", "hybrid", 5)
    assert client.calls[-1]["query"].fusion == models.Fusion.DBSF


def test_prefetch_limit_override():
    client = FakeQdrant([_pt("A", 0, 0.9)])
    QdrantBackend(_cfg(), client, FakeEmb(), prefetch_limit=200).search("hello", "hybrid", 5)
    assert client.calls[-1]["prefetch"][0].limit == 200


def test_search_params_threaded():
    client = FakeQdrant([_pt("A", 0, 0.9)])
    QdrantBackend(_cfg(), client, FakeEmb(), hnsw_ef=128, rescore=False).search("hello", "dense", 5)
    sp = client.calls[-1]["search_params"]
    assert sp is not None and sp.hnsw_ef == 128


def test_diversity_caps_per_doc_in_backend():
    client = FakeQdrant([_pt("A", 0, 0.9), _pt("A", 1, 0.8), _pt("B", 0, 0.7)])
    hits, _ = QdrantBackend(_cfg(), client, FakeEmb(), max_per_doc=1).search("hello", "hybrid", 5)
    docs = [h.document_id for h in hits]
    assert docs.count("A") == 1 and "B" in docs
