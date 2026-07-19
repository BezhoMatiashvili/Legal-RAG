"""Tests for the EN→KA query-translation knob (improvement I2)."""

import json
from types import SimpleNamespace

import pytest

from eval.backend import QdrantBackend
from eval.translations import DEFAULT_TRANSLATIONS, load_query_translations
from ingest.config import load_config

EN_Q = "annual paid leave duration for employees"
KA_Q = "დასაქმებულის ყოველწლიური ანაზღაურებადი შვებულების ხანგრძლივობა"


def _write(tmp_path, entries):
    p = tmp_path / "tr.json"
    p.write_text(json.dumps({"version": "t", "translations": entries}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def _gold(*pairs):
    return [SimpleNamespace(id=i, query=q) for i, q in pairs]


# --- loader validation ----------------------------------------------------------------


def test_loader_returns_mapping_and_content_hash(tmp_path):
    p = _write(tmp_path, {"q1": {"query": EN_Q, "ka": KA_Q}})
    mapping, h1 = load_query_translations(p, _gold(("q1", EN_Q)))
    assert mapping == {EN_Q: KA_Q}
    assert len(h1) == 64
    p.write_text(p.read_text().replace("შვებულების", "შვებულებisა"), encoding="utf-8")
    _, h2 = load_query_translations(p, _gold(("q1", EN_Q)))
    assert h1 != h2  # content-addressed: any edit is a new eval config


def test_loader_fails_on_unknown_id(tmp_path):
    p = _write(tmp_path, {"qX": {"query": EN_Q, "ka": KA_Q}})
    with pytest.raises(ValueError, match="unknown golden id"):
        load_query_translations(p, _gold(("q1", EN_Q)))


def test_loader_fails_on_query_drift(tmp_path):
    p = _write(tmp_path, {"q1": {"query": EN_Q + " CHANGED", "ka": KA_Q}})
    with pytest.raises(ValueError, match="drifted"):
        load_query_translations(p, _gold(("q1", EN_Q)))


def test_loader_rejects_non_english_source(tmp_path):
    p = _write(tmp_path, {"q1": {"query": KA_Q, "ka": KA_Q}})
    with pytest.raises(ValueError, match="not English"):
        load_query_translations(p, _gold(("q1", KA_Q)))


def test_loader_rejects_non_georgian_target(tmp_path):
    p = _write(tmp_path, {"q1": {"query": EN_Q, "ka": "still english"}})
    with pytest.raises(ValueError, match="no Georgian script"):
        load_query_translations(p, _gold(("q1", EN_Q)))


def test_checked_in_artifact_matches_golden_set():
    """Drift guard for the real file: every entry byte-matches golden_set_v1."""
    from eval import goldset

    gold = goldset.load_golden_set()
    mapping, _ = load_query_translations(DEFAULT_TRANSLATIONS, gold)
    en_queries = {q.query for q in gold if q.query_language == "en"}
    assert set(mapping) == en_queries  # covers all 22, nothing else


# --- backend substitution ----------------------------------------------------------------


class RecordingEmb:
    def __init__(self):
        self.seen = []

    def encode_query(self, q):
        self.seen.append(q)
        return SimpleNamespace(
            dense=[0.1, 0.2, 0.3],
            sparse=SimpleNamespace(indices=[1, 2], values=[0.5, 0.5]),
        )


class RecordingReranker:
    def __init__(self):
        self.seen = []

    def score(self, q, texts):
        self.seen.append(q)
        return [0.9] * len(texts)


def _pt(doc, ci, score):
    return SimpleNamespace(
        payload={"document_id": doc, "source": "matsne", "chunk_index": ci, "text": f"{doc}-{ci}"},
        score=score, vector=None,
    )


class FakeQdrant:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def query_points(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(points=list(self.points))


def _backend(**kw):
    return QdrantBackend(load_config(), FakeQdrant([_pt("A", 0, 0.9)]), RecordingEmb(), **kw)


def test_backend_substitutes_translated_query():
    b = _backend(translations={EN_Q: KA_Q})
    b.search(EN_Q, "hybrid", 5)
    assert b.embedder.seen == [KA_Q]


def test_translated_query_keeps_sparse_in_routed_mode():
    b = _backend(translations={EN_Q: KA_Q})
    b.search(EN_Q, "routed", 5)
    prefetch = b.client.calls[-1]["prefetch"]
    assert {p.using for p in prefetch} == {"dense", "sparse"}  # KA text → sparse stays


def test_untranslated_english_query_still_drops_sparse_in_routed_mode():
    b = _backend(translations={"some other query": KA_Q})
    b.search(EN_Q, "routed", 5)
    prefetch = b.client.calls[-1]["prefetch"]
    assert [p.using for p in prefetch] == ["dense"]


def test_georgian_queries_mechanically_unchanged():
    on = _backend(translations={EN_Q: KA_Q})
    off = _backend()
    h_on, _ = on.search(KA_Q, "hybrid", 5)
    h_off, _ = off.search(KA_Q, "hybrid", 5)
    assert on.embedder.seen == off.embedder.seen == [KA_Q]
    assert h_on == h_off


def test_reranker_receives_translated_query():
    rr = RecordingReranker()
    b = QdrantBackend(load_config(), FakeQdrant([_pt("A", 0, 0.9)]), RecordingEmb(),
                      reranker=rr, translations={EN_Q: KA_Q})
    b.search(EN_Q, "rerank", 5)
    assert rr.seen == [KA_Q]


def test_legal_search_docstring_has_translation_guidance():
    """Serving-layer half of I2: the MCP client is told to query in Georgian."""
    import inspect

    from ingest import mcp_server

    doc = " ".join((inspect.getdoc(mcp_server.legal_search) or "").split())
    assert "translate the query into Georgian legal terminology" in doc


def test_config_hash_distinguishes_translation_runs():
    from eval import explog

    base = {"mode": "hybrid", "relevance": "chunk", "top_k": 10}
    assert explog.config_hash(base) != explog.config_hash(
        {**base, "translate_queries": "abcd1234abcd1234"})
