"""Knob-behavior tests for citation routing (I1): hybrid_search + QdrantBackend + hashes.

A fake Qdrant client records every ``query_points`` call so we can assert (a) the exact
extra lookup the knob issues, and (b) byte-identical behavior when the knob is off or the
query carries no citation — the runbook's G2-by-construction requirement.
"""

import dataclasses
from types import SimpleNamespace

from eval import explog
from eval.backend import QdrantBackend
from ingest.config import load_config, retrieval_fingerprint
from ingest.search import hybrid_search


class FakeEmb:
    def encode_query(self, q):
        return SimpleNamespace(
            dense=[0.1, 0.2, 0.3],
            sparse=SimpleNamespace(indices=[1, 2], values=[0.5, 0.5]),
        )


def _pt(pid, doc, ci, score):
    return SimpleNamespace(
        id=pid, score=score, vector={"dense": [0.1, 0.2, 0.3]},
        payload={"source": "matsne", "document_id": doc, "chunk_index": ci, "text": f"{doc}-{ci}"},
    )


class FakeQdrant:
    """Returns ``filtered_points`` for citation lookups (query_filter set), else ``points``."""

    def __init__(self, points, filtered_points=None):
        self.points = points
        self.filtered_points = filtered_points if filtered_points is not None else []
        self.calls = []

    def query_points(self, **kw):
        self.calls.append(kw)
        if kw.get("query_filter") is not None:
            return SimpleNamespace(points=list(self.filtered_points))
        return SimpleNamespace(points=list(self.points))


_CITE_Q = "შსს მინისტრის №71 ბრძანებაში ცვლილება"
_PLAIN_Q = "ყოველწლიური ანაზღაურებადი შვებულების ხანგრძლივობა"


def _cfg(route=None):
    return dataclasses.replace(load_config(), citation_route=route)


# --- hybrid_search (serving path) ----------------------------------------------------


def test_hybrid_search_no_citation_is_byte_identical():
    sem = [_pt("s1", "A", 0, 0.9), _pt("s2", "B", 0, 0.8)]
    on, off = FakeQdrant(sem), FakeQdrant(sem)
    r_on = hybrid_search(_cfg("ids"), on, FakeEmb(), _PLAIN_Q, top_k=2)
    r_off = hybrid_search(_cfg(None), off, FakeEmb(), _PLAIN_Q, top_k=2)
    assert [p.id for p in r_on] == [p.id for p in r_off]
    assert len(on.calls) == len(off.calls) == 1  # no extra lookup issued


def test_hybrid_search_prepends_pinned_hit():
    sem = [_pt("s1", "A", 0, 0.9), _pt("s2", "B", 0, 0.8)]
    exact = [_pt("x1", "GOLD", 0, 0.001)]
    client = FakeQdrant(sem, filtered_points=exact)
    out = hybrid_search(_cfg("ids"), client, FakeEmb(), _CITE_Q, top_k=2)
    assert [p.id for p in out] == ["x1", "s1"]
    assert out[0].score == 1.0
    assert client.calls[0].get("query_filter") is not None  # lookup ran first


def test_hybrid_search_pins_survive_rerank_and_min_score():
    sem = [_pt("s1", "A", 0, 0.9), _pt("s2", "B", 0, 0.8)]
    exact = [_pt("x1", "GOLD", 0, 0.001)]
    client = FakeQdrant(sem, filtered_points=exact)

    class HostileReranker:  # scores everything low — would drop/demote the pin if it could
        def score(self, q, texts):
            return [0.01] * len(texts)

    out = hybrid_search(_cfg("ids"), client, FakeEmb(), _CITE_Q, top_k=2,
                        reranker=HostileReranker(), rerank_min_score=0.3)
    assert out and out[0].id == "x1" and out[0].score == 1.0


def test_hybrid_search_empty_lookup_falls_through():
    sem = [_pt("s1", "A", 0, 0.9)]
    client = FakeQdrant(sem, filtered_points=[])  # tbappeal-style NULL document_number
    out = hybrid_search(_cfg("ids"), client, FakeEmb(), _CITE_Q, top_k=1)
    assert [p.id for p in out] == ["s1"]


def test_hybrid_search_skips_route_when_caller_filters():
    sem = [_pt("s1", "A", 0, 0.9)]
    client = FakeQdrant(sem, filtered_points=[_pt("x1", "GOLD", 0, 0.5)])
    out = hybrid_search(_cfg("ids"), client, FakeEmb(), _CITE_Q, top_k=1, source="matsne")
    assert [p.id for p in out] == ["s1"]
    assert all(kw.get("query_filter") is None for kw in client.calls)  # no citation lookup


# --- QdrantBackend (eval path) --------------------------------------------------------


def test_backend_citation_route_pins_and_issues_filtered_call():
    sem = [_pt("s1", "A", 0, 0.9), _pt("s2", "B", 0, 0.8)]
    exact = [_pt("x1", "GOLD", 5, 0.001)]
    client = FakeQdrant(sem, filtered_points=exact)
    hits, _ = QdrantBackend(_cfg(), client, FakeEmb(), citation_route="ids").search(
        _CITE_Q, "hybrid", 2)
    assert (hits[0].document_id, hits[0].chunk_index) == ("GOLD", 5)
    lookup = client.calls[0]
    assert lookup["using"] == "dense" and lookup["limit"] == 3


def test_backend_citation_route_off_no_extra_calls():
    sem = [_pt("s1", "A", 0, 0.9)]
    client = FakeQdrant(sem)
    QdrantBackend(_cfg(), client, FakeEmb()).search(_CITE_Q, "hybrid", 2)
    assert len(client.calls) == 1 and client.calls[0].get("query_filter") is None


def test_backend_plain_query_identical_with_knob_on():
    sem = [_pt("s1", "A", 0, 0.9), _pt("s2", "B", 0, 0.8)]
    on, off = FakeQdrant(sem), FakeQdrant(sem)
    h_on, _ = QdrantBackend(_cfg(), on, FakeEmb(), citation_route="ids").search(_PLAIN_Q, "hybrid", 2)
    h_off, _ = QdrantBackend(_cfg(), off, FakeEmb()).search(_PLAIN_Q, "hybrid", 2)
    assert h_on == h_off
    assert len(on.calls) == len(off.calls) == 1


# --- hashes / fingerprint --------------------------------------------------------------


def test_config_hash_changes_with_citation_route():
    base = {"mode": "hybrid", "relevance": "chunk", "top_k": 10}
    assert explog.config_hash(base) != explog.config_hash({**base, "citation_route": "ids"})


def test_fingerprint_unchanged_when_route_off_changes_when_on():
    off, on = _cfg(None), _cfg("ids")
    assert retrieval_fingerprint(off) == retrieval_fingerprint(dataclasses.replace(off))
    assert retrieval_fingerprint(on) != retrieval_fingerprint(off)
