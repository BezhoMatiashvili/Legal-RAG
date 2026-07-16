"""Characterization tests for the shared production retrieval contract."""

import dataclasses
from types import SimpleNamespace

import pytest

from eval import backend as eval_backend
from eval.backend import ProductionBackend
from eval.evaluate import run_mode, selected_modes
from eval.metrics import Hit
from ingest.config import load_config, retrieval_fingerprint
from ingest import retrieval
from ingest.mcp_server import SearchInput, _retrieval_request
from ingest.retrieval import (
    EvaluationProvenance,
    RetrievalPolicy,
    RetrievalRequest,
    TemporalContext,
    execute_retrieval,
    RetrievalOutcome,
)


def _cfg(**overrides):
    return dataclasses.replace(
        load_config(),
        rerank_enabled=True,
        rerank_candidates=80,
        rerank_min_score=0.3,
        **overrides,
    )


def test_executor_forwards_current_production_policy_and_request(monkeypatch):
    cfg = _cfg()
    calls = []
    point = object()

    def fake_search(*args, **kwargs):
        calls.append((args, kwargs))
        return [point]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    request = RetrievalRequest(
        query="Civil Code article 829",
        requested_limit=7,
        filters={"source": "matsne", "status": "in_force"},
        language="ka",
        temporal_context=TemporalContext(date_from="2020-01-01", as_of="2026-01-01"),
    )
    reranker = object()
    outcome = execute_retrieval(cfg, "client", "embedder", reranker, request)

    args, kwargs = calls[0]
    assert args == (cfg, "client", "embedder", request.query)
    assert kwargs == {
        "top_k": 7,
        "reranker": reranker,
        "rerank_candidates": 80,
        "rerank_min_score": 0.3,
        "route": True,
        "max_per_doc": None,
        "mmr_lambda": None,
        "timings_ms": {"embed": 0.0, "search": 0.0, "rerank": 0.0},
        "source": "matsne",
        "status": "in_force",
        "language": "ka",
        "date_from": "2020-01-01",
        "as_of": "2026-01-01",
    }
    assert outcome.hits == (point,)
    assert outcome.effective_route == "dense_only"
    assert outcome.retrieval_fingerprint == retrieval_fingerprint(cfg)
    assert set(outcome.timings_ms) == {"embed", "search", "rerank", "total"}
    assert not outcome.degraded and not outcome.service_abstention


def test_mcp_constructs_the_provider_neutral_request_without_losing_filters():
    request = _retrieval_request(
        SearchInput(
            query="შრომითი დავა",
            top_k=17,
            source="tas",
            court="supreme",
            status="in_force",
            is_consolidated=True,
            language="ka",
            document_type="decision",
            document_number="N-1",
            registration_code="REG",
            parties="A v B",
            contains="ზიანი",
            date_from="2020-01-01",
            date_to="2026-01-01",
        )
    )

    assert request.query == "შრომითი დავა"
    assert request.requested_limit == 17
    assert request.route is True and request.track == "production_direct"
    assert request.search_kwargs() == {
        "source": "tas",
        "court": "supreme",
        "status": "in_force",
        "is_consolidated": True,
        "document_type": "decision",
        "document_number": "N-1",
        "registration_code": "REG",
        "parties": "A v B",
        "contains": "ზიანი",
        "language": "ka",
        "date_from": "2020-01-01",
        "date_to": "2026-01-01",
    }


def test_executor_labels_direct_georgian_as_hybrid(monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [object()])
    outcome = execute_retrieval(
        _cfg(), None, None, object(), RetrievalRequest(query="სამოქალაქო კოდექსი")
    )
    assert outcome.effective_route == "hybrid"


def test_remote_reranker_failure_retries_same_request_as_degraded(monkeypatch):
    calls = []

    class Remote:
        device = "remote"

    def fake_search(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["reranker"] is not None:
            raise ConnectionError("pod stopped")
        return ["rrf-hit"]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_retrieval(
        _cfg(), None, None, Remote(), RetrievalRequest(query="შრომითი დავა", filters={"court": "tas"})
    )

    assert [call["reranker"] for call in calls] == [calls[0]["reranker"], None]
    assert calls[0]["court"] == calls[1]["court"] == "tas"
    assert outcome.hits == ("rrf-hit",)
    assert outcome.degraded is True
    assert outcome.degraded_reason == "ConnectionError: pod stopped"


def test_local_retrieval_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(
        retrieval, "hybrid_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad index"))
    )
    with pytest.raises(RuntimeError, match="bad index"):
        execute_retrieval(
            _cfg(), None, None, object(), RetrievalRequest(query="შრომითი დავა")
        )


def test_empty_result_is_structured_service_abstention(monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [])
    outcome = execute_retrieval(
        _cfg(), None, None, object(), RetrievalRequest(query="შრომითი დავა")
    )
    assert outcome.service_abstention is True
    assert outcome.abstention_reason == "no_results_after_policy"


def test_request_rejects_conflicting_filter_context():
    request = RetrievalRequest(
        query="q", language="ka", filters={"language": "en"}
    )
    with pytest.raises(ValueError, match="language conflicts"):
        request.search_kwargs()

    request = RetrievalRequest(
        query="q",
        filters={"as_of": "2025-01-01"},
        temporal_context=TemporalContext(as_of="2026-01-01"),
    )
    with pytest.raises(ValueError, match="temporal context conflicts"):
        request.search_kwargs()


def test_publication_date_and_effective_as_of_are_independent_filters():
    request = RetrievalRequest(
        query="q",
        temporal_context=TemporalContext(date_to="2025-01-01", as_of="2026-01-01"),
    )
    assert request.search_kwargs() == {
        "date_to": "2025-01-01",
        "as_of": "2026-01-01",
    }


def test_policy_records_the_existing_candidate_floor():
    policy = RetrievalPolicy(rerank_candidates=30, rerank_min_score=0.3)
    assert policy.candidate_depth(10, reranker_present=True) == 50
    assert policy.candidate_depth(20, reranker_present=True) == 100
    assert policy.candidate_depth(2, reranker_present=False) == 50


def _provenance(**overrides):
    values = {
        "collection_alias": "georgian_legal",
        "serving_alias": "georgian_legal",
        "physical_collection": "georgian_legal__gen_20260713",
        "queried_collection": "georgian_legal",
        "access_kind": "serving_alias",
        "generation_id": "20260713",
        "points_count": 12,
        "corpus_hash": "1" * 64,
        "snapshot_hash": "2" * 64,
        "embedding_model": "embed",
        "embedding_revision": "a" * 40,
        "tokenizer_model": "tokenizer",
        "tokenizer_revision": "b" * 40,
        "reranker_model": "rerank",
        "reranker_revision": "c" * 40,
        "vector_space_id": "3" * 64,
        "chunk_config_id": "4" * 64,
        "header_config_id": "5" * 64,
        "retrieval_fingerprint_revision": 2,
        "retrieval_fingerprint": "6" * 64,
        "frozen_set_hashes": {"v2": "7" * 64},
        "dependency_identity": "8" * 64,
        "image_identity": "sha256:" + "9" * 64,
        "git_sha": "d" * 40,
        "dirty_patch_hash": "e" * 64,
        "execution_mode": "production_direct_en",
    }
    values.update(overrides)
    return EvaluationProvenance(**values)


def test_evaluation_provenance_fails_closed_on_missing_identity():
    _provenance().validate_complete()
    assert _provenance().to_dict()["tokenizer_model"] == "tokenizer"
    with pytest.raises(ValueError, match="embedding_revision"):
        _provenance(embedding_revision=None).validate_complete()
    with pytest.raises(ValueError, match="frozen_set_hashes"):
        _provenance(frozen_set_hashes={}).validate_complete()
    with pytest.raises(ValueError, match="snapshot_hash"):
        _provenance(snapshot_hash="not-a-hash").validate_complete()
    with pytest.raises(ValueError, match="physical_collection"):
        _provenance(physical_collection="georgian_legal").validate_complete()
    with pytest.raises(ValueError, match="retrieval_fingerprint_revision"):
        _provenance(retrieval_fingerprint_revision=1).validate_complete()
    with pytest.raises(ValueError, match="queried_collection"):
        _provenance(
            access_kind="direct_physical",
            queried_collection="georgian_legal",
        ).validate_complete()


@pytest.mark.parametrize(
    ("mode", "expected_query"),
    [("production", "English query"), ("client_translated", "ქართული თარგმანი")],
)
def test_production_eval_backend_uses_shared_executor_and_separate_translation_track(
    monkeypatch, mode, expected_query
):
    captured = []
    point = SimpleNamespace(
        payload={"source": "matsne", "document_id": "1", "chunk_index": 2},
        score=0.91,
    )

    def fake_execute(cfg, client, embedder, reranker, request):
        captured.append((cfg, client, embedder, reranker, request))
        return RetrievalOutcome(
            hits=(point,),
            timings_ms={"embed": 1.0, "search": 2.0, "rerank": 3.0, "total": 6.0},
            degraded=False,
            degraded_reason=None,
            service_abstention=False,
            abstention_reason=None,
            retrieval_fingerprint="fp",
            effective_route="dense_only",
            generation_id="gen",
            track=mode,
        )

    monkeypatch.setattr(eval_backend, "execute_retrieval", fake_execute)
    backend = ProductionBackend(
        _cfg(), "client", "embedder", "reranker",
        translations={"English query": "ქართული თარგმანი"},
    )
    hits, latency = backend.search("English query", mode, 10)

    request = captured[0][-1]
    assert request.query == expected_query
    assert request.track == mode and request.route is True
    assert hits[0].document_id == "1" and hits[0].chunk_index == 2
    assert latency == {"embed": 0.001, "search": 0.002, "rerank": 0.003}
    assert backend.last_outcome.retrieval_fingerprint == "fp"


def test_client_translated_track_never_falls_back_for_missing_english_translation(
    monkeypatch,
):
    monkeypatch.setattr(
        eval_backend,
        "execute_retrieval",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("degraded translation must not retrieve")
        ),
    )
    backend = ProductionBackend(_cfg(), "client", "embedder", "reranker", translations={})

    hits, latency = backend.search("What does the current law require?", "client_translated", 10)

    assert hits == []
    assert latency == {"embed": 0.0, "search": 0.0, "rerank": 0.0}
    assert backend.last_outcome.degraded is True
    assert backend.last_outcome.degraded_reason == "missing_authored_translation"


def test_accuracy_strict_eval_backend_uses_branch_traced_executor_and_static_translator(
    monkeypatch,
):
    captured = []
    point = SimpleNamespace(
        payload={"source": "matsne", "document_id": "1", "chunk_index": 2},
        score=0.91,
    )

    def fake_execute(cfg, client, embedder, reranker, request, *, translator, candidate_depth):
        captured.append((cfg, client, embedder, reranker, request, translator, candidate_depth))
        return SimpleNamespace(
            hits=(point,),
            timings_ms={"candidate_branches": 4.0, "rerank": 5.0, "total": 10.0},
            degraded=False,
            degraded_reason=None,
            service_abstention=False,
            abstention_reason=None,
            branches=(),
            result_hash="strict-hash",
        )

    monkeypatch.setattr(eval_backend, "execute_accuracy_retrieval", fake_execute)
    backend = ProductionBackend(
        _cfg(), "client", "embedder", "reranker",
        translations={"What is the law": "რა არის კანონი"},
    )
    hits, latency = backend.search("What is the law", "accuracy_strict", 10)

    request = captured[0][4]
    translator = captured[0][5]
    assert request.query == "What is the law"
    assert request.query_language == "en" and request.track == "accuracy_strict"
    assert captured[0][6] == 80
    assert translator.version.startswith("static-map:")
    assert hits[0].document_id == "1"
    assert latency == {"embed": 0.0, "search": 0.004, "rerank": 0.005}
    assert backend.last_outcome.result_hash == "strict-hash"


def test_production_is_primary_and_translation_is_a_separate_track():
    assert selected_modes("qdrant", None, has_translations=False) == ["production"]
    assert selected_modes("fake", None, has_translations=False) == list(eval_backend.MODES)
    all_tracks = selected_modes("qdrant", "all", has_translations=True)
    assert all_tracks[:3] == ["production", "accuracy_strict", "client_translated"]
    with pytest.raises(ValueError, match="requires --translate-queries"):
        selected_modes("qdrant", "client_translated", has_translations=False)
    with pytest.raises(ValueError, match="requires --translate-queries"):
        selected_modes("qdrant", "accuracy_strict", has_translations=False)


def test_degraded_execution_is_counted_as_failure_and_reported_separately():
    gold = [
        SimpleNamespace(id="q1", query="one", query_type="keyword", query_language="en"),
        SimpleNamespace(id="q2", query="two", query_type="keyword", query_language="en"),
    ]
    rel = {
        "q1": {"chunk": {("s", "d", 0): 1}},
        "q2": {"chunk": {("s", "d", 0): 1}},
    }

    class Backend:
        last_outcome = None

        def search(self, query, mode, k):
            degraded = query == "one"
            self.last_outcome = RetrievalOutcome(
                hits=(),
                timings_ms={"embed": 1, "search": 2, "rerank": 3, "total": 6},
                degraded=degraded,
                degraded_reason="remote down" if degraded else None,
                service_abstention=not degraded,
                abstention_reason="no_results_after_policy" if not degraded else None,
                retrieval_fingerprint="fp",
                effective_route="dense_only",
                generation_id="gen",
                track=mode,
            )
            return [Hit("s", "d", 0, 1.0)], {
                "embed": 0.001, "search": 0.002, "rerank": 0.003,
            }

    scores, latency = run_mode(Backend(), gold, rel, "production", "chunk", 10)
    assert [score.id for score in scores] == ["q1", "q2"]
    assert scores[0].failed is True
    assert scores[0].success10 == 0.0
    assert scores[1].failed is False
    assert latency["degraded"] == [{
        "query_id": "q1",
        "reason": "remote down",
        "timings_ms": {"embed": 1, "search": 2, "rerank": 3, "total": 6},
    }]


def test_production_backend_hit_adapter_retains_version_identity():
    point = SimpleNamespace(
        payload={
            "source": "matsne",
            "document_id": "law",
            "version_id": "law@2026",
            "chunk_index": 4,
        },
        score=0.9,
    )
    (hit,) = ProductionBackend._hits((point,))
    assert hit.version_id == "law@2026"
