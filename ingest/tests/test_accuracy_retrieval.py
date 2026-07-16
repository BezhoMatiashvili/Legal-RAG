import dataclasses
from types import SimpleNamespace

from ingest.config import load_config
from ingest import retrieval
from ingest.retrieval import RetrievalRequest, TemporalContext, execute_accuracy_retrieval


def _cfg():
    return dataclasses.replace(load_config(), generation_id="generation-20260715")


def _point(pid, text, *, source="matsne", document_id="d1", **payload):
    return SimpleNamespace(
        id=pid,
        score=0.2,
        payload={
            "source": source,
            "document_id": document_id,
            "document_type": "legislation",
            "status": "in_force",
            "version_id": "current-version",
            "effective_from": "2000-01-01T00:00:00Z",
            "effective_to": None,
            "repeal_date": None,
            "version_lineage_complete": True,
            "version_ambiguous": False,
            "chunk_index": 0,
            "text": text,
            "content_complete": True,
            "extraction_status": "full_text",
            "freshness_sla_met": True,
            "char_start": 0,
            "char_end": len(text),
            **payload,
        },
    )


class _Translator:
    version = "mt-rev"

    def translate(self, text, **kwargs):
        return "ქართული თარგმანი " + text


class _Embedder:
    def encode_query(self, query):
        return SimpleNamespace(dense=[0.1], sparse=SimpleNamespace(indices=[], values=[]))


class _Reranker:
    def score(self, query, texts):
        return [0.8 + i / 100 for i, _ in enumerate(texts)]


def test_accuracy_path_unions_original_and_translation_before_one_rerank(monkeypatch):
    calls = []
    original = _point("a", "original candidate")
    translated = _point("b", "translated candidate", document_id="d2")

    def fake_search(cfg, client, embedder, query, **kwargs):
        calls.append((query, kwargs))
        return [translated] if "ქართული" in query else [original]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="What does the law say?", query_language="en"),
        translator=_Translator(),
    )

    assert [branch.name for branch in outcome.branches] == [
        "global_original", "global_translated_ka"
    ]
    assert all(call[1]["top_k"] == 80 and call[1]["reranker"] is None for call in calls)
    assert [point.id for point in outcome.hits] == ["b", "a"]
    assert outcome.candidate_ids == ("a", "b")
    assert not outcome.degraded and not outcome.service_abstention
    assert outcome.translator_version == "mt-rev"
    assert len(outcome.result_hash) == 64


def test_accuracy_path_is_deterministic_for_identical_rankings(monkeypatch):
    def fake_search(*args, **kwargs):
        return [_point("a", "same")]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    request = RetrievalRequest(query="რა ამბობს კანონი?")
    first = execute_accuracy_retrieval(_cfg(), None, _Embedder(), _Reranker(), request)
    second = execute_accuracy_retrieval(_cfg(), None, _Embedder(), _Reranker(), request)
    assert first.result_hash == second.result_hash


def test_accuracy_hash_binds_the_exact_pre_rerank_pool():
    plan = retrieval.plan_query("რა ამბობს კანონი?")
    first = retrieval._accuracy_outcome(
        _cfg(), plan, candidate_ids=("first", "second"),
        resolved_current_date="2026-07-15",
    )
    reordered = retrieval._accuracy_outcome(
        _cfg(), plan, candidate_ids=("second", "first"),
        resolved_current_date="2026-07-15",
    )

    assert first.candidate_ids == ("first", "second")
    assert first.result_hash != reordered.result_hash


def test_resolved_current_date_is_bound_into_retrieval_hash(monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [_point("a", "same")])
    request = RetrievalRequest(query="რა ამბობს კანონი?")

    first = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(), request, current_date="2026-07-15"
    )
    second = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(), request, current_date="2026-07-16"
    )

    assert first.resolved_current_date == "2026-07-15"
    assert second.resolved_current_date == "2026-07-16"
    assert first.result_hash != second.result_hash


def test_filtered_request_also_runs_truly_unrestricted_semantic_branch(monkeypatch):
    filtered = _point("filtered", "filtered", document_id="filtered")
    unrestricted = _point("unrestricted", "unrestricted", document_id="unrestricted")

    def fake_search(*args, **kwargs):
        return [filtered] if kwargs.get("source") == "matsne" else [unrestricted]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რა ამბობს კანონი?", filters={"source": "matsne"}),
    )

    assert [branch.name for branch in outcome.branches] == [
        "global_original", "global_unrestricted_original"
    ]
    assert dict(outcome.branches[0].filters) == {"source": "matsne"}
    assert dict(outcome.branches[1].filters) == {}
    assert [point.id for point in outcome.hits] == ["filtered"]
    assert outcome.candidate_ids == ("filtered",)
    assert "unrestricted" in outcome.branches[1].hit_ids


def test_identical_text_in_distinct_authorities_and_versions_is_not_deduplicated(monkeypatch):
    digest = "a" * 64
    current = _point(
        "current", "იდენტური ნორმა", source="matsne", document_id="law-1",
        version_id="current", passage_hash=digest, passage_id="passage:current",
    )
    repealed = _point(
        "repealed", "იდენტური ნორმა", source="matsne", document_id="law-1",
        version_id="repealed", passage_hash=digest, passage_id="passage:repealed",
    )
    other_authority = _point(
        "other", "იდენტური ნორმა", source="constcourt", document_id="case-2",
        version_id="decision", passage_hash=digest, passage_id="passage:other",
    )
    monkeypatch.setattr(
        retrieval, "hybrid_search", lambda *a, **k: [current, repealed, other_authority]
    )

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="იმ დროისთვის რომელი ვერსიები არსებობდა?"),
    )

    assert {point.id for point in outcome.hits} == {"current", "repealed", "other"}


def test_missing_or_non_full_extraction_never_enters_answer_reranker(monkeypatch):
    incomplete = _point("bad", "summary", extraction_status="complete")
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [incomplete])

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რა ამბობს კანონი?"),
    )

    assert outcome.hits == ()
    assert outcome.candidate_ids == ()
    assert outcome.abstention_reason == "insufficient_evidence"


def test_unsupported_language_and_missing_translator_fail_closed(monkeypatch):
    monkeypatch.setattr(
        retrieval, "hybrid_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    unsupported = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="cual es la ley"),
    )
    assert unsupported.service_abstention
    assert unsupported.abstention_reason == "unsupported_language"

    no_mt = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="What does the law say?", query_language="en"),
    )
    assert no_mt.service_abstention and no_mt.degraded
    assert no_mt.abstention_reason == "translator_degraded"


def test_as_of_uses_effective_version_filter_and_rejects_missing_lineage(monkeypatch):
    calls = []

    def fake_search(*args, **kwargs):
        calls.append(kwargs)
        return [_point("a", "legacy passage", version_lineage_complete=False)]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(
            query="რა იყო წესი?",
            temporal_context=TemporalContext(as_of="2020-01-01"),
        ),
    )
    assert calls[0]["as_of"] == "2020-01-01"
    assert "date_to" not in calls[0]
    assert outcome.service_abstention
    assert outcome.abstention_reason == "temporal_lineage_missing_or_ambiguous"


def test_as_of_checks_the_returned_effective_interval(monkeypatch):
    valid = _point(
        "valid", "historical passage",
        version_id="v1", effective_from="2019-01-01", effective_to="2021-01-01",
        version_lineage_complete=True,
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [valid])
    request = RetrievalRequest(
        query="რა იყო წესი?",
        temporal_context=TemporalContext(as_of="2020-01-01"),
    )
    outcome = execute_accuracy_retrieval(_cfg(), None, _Embedder(), _Reranker(), request)
    assert not outcome.service_abstention

    expired = _point(
        "expired", "expired passage",
        version_id="v0", effective_from="2010-01-01", effective_to="2019-01-01",
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [expired])
    outcome = execute_accuracy_retrieval(_cfg(), None, _Embedder(), _Reranker(), request)
    assert outcome.abstention_reason == "temporal_lineage_missing_or_ambiguous"


def test_ambiguous_exact_identity_clarifies_without_rewriting_scores(monkeypatch):
    one = _point("one", "one", document_id="d1")
    two = _point("two", "two", document_id="d2")
    one.payload["identity_ambiguous"] = True
    two.payload["identity_ambiguous"] = True
    one.payload["identity_confidence"] = two.payload["identity_confidence"] = 0.5

    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "citation_lookup", lambda *a, **k: [one, two])
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="კანონი №71", query_language="ka"),
    )
    assert outcome.identity_ambiguous and outcome.service_abstention
    assert outcome.abstention_reason == "ambiguous_entity"
    assert one.score != 1.0 and two.score != 1.0


def test_explicit_identifier_that_does_not_resolve_returns_typed_clarification(monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "citation_lookup", lambda *a, **k: [])

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="კანონი №71", query_language="ka"),
    )

    assert outcome.service_abstention
    assert outcome.abstention_reason == "unresolved_entity"


def test_unique_exact_identity_cannot_be_overridden_by_unrestricted_semantic_hit(monkeypatch):
    exact = _point(
        "exact", "ზუსტი დოკუმენტი", document_id="law-71", version_id="law-71-current",
        match_type="document_number", identity_priority=True,
        identity_ambiguous=False, identity_confidence=1.0,
    )
    unrelated = _point(
        "unrelated", "ლექსიკურად მსგავსი სხვა კანონი", document_id="other-law",
        version_id="other-current",
    )

    def fake_search(*args, **kwargs):
        return [exact] if kwargs.get("document_id") == "law-71" else [unrelated]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    monkeypatch.setattr(retrieval, "citation_lookup", lambda *a, **k: [exact])

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="კანონი №71", query_language="ka"),
        current_date="2026-07-15",
    )

    assert not outcome.service_abstention
    assert [point.id for point in outcome.hits] == ["exact"]
    assert outcome.candidate_ids == ("exact",)
    assert "unrelated" in outcome.branches[0].hit_ids


def test_explicit_cross_reference_gets_scoped_branch_and_second_rerank(monkeypatch):
    main = _point(
        "main",
        "ამ კანონის 2-ე მუხლით განსაზღვრული პირი",
        article_id="5",
    )
    referenced = _point("ref", "მუხლი 2. ტერმინის განსაზღვრება", article_id="2")
    calls = []

    def fake_search(*args, **kwargs):
        calls.append(kwargs)
        return [referenced] if kwargs.get("article_id") == "2" else [main]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რას ნიშნავს ეს პირი?", query_language="ka"),
    )
    assert any(branch.name.startswith("cross_reference:matsne:d1:2") for branch in outcome.branches)
    structural = next(
        branch for branch in outcome.branches
        if branch.name == "cross_reference:matsne:d1:2"
    )
    assert dict(structural.filters)["article_id"] == "2"
    assert "contains" not in structural.filters
    assert outcome.timings_ms["cross_reference_rerank"] >= 0
    assert {point.id for point in outcome.hits} == {"main", "ref"}
    assert outcome.candidate_ids == ("main",)


def test_structured_cross_reference_does_not_fall_back_to_wrong_article_text(monkeypatch):
    main = _point(
        "main", "ამ კანონის 2-ე მუხლით განსაზღვრული პირი", article_id="5"
    )
    wrong_article = _point(
        "wrong", "ეს ტექსტი ახსენებს მუხლი 2-ს, მაგრამ სხვა მუხლია", article_id="5"
    )

    def fake_search(*args, **kwargs):
        if kwargs.get("article_id") == "2":
            return []
        if kwargs.get("contains") == "მუხლი 2":
            return [wrong_article]
        return [main]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რას ნიშნავს ეს პირი?", query_language="ka"),
    )

    assert "wrong" not in {point.id for point in outcome.hits}
    assert not any("lexical_fallback" in branch.name for branch in outcome.branches)


def test_strict_cross_reference_never_uses_lexical_fallback(monkeypatch):
    unstructured = _point("main", "ამ კანონის 2-ე მუხლით განსაზღვრული პირი")
    referenced = _point("ref", "მუხლი 2. ტერმინის განსაზღვრება")

    def fake_search(*args, **kwargs):
        if kwargs.get("article_id") == "2":
            return []
        if kwargs.get("contains") == "მუხლი 2":
            return [referenced]
        return [unstructured]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რას ნიშნავს ეს პირი?", query_language="ka"),
    )

    assert not any("lexical_fallback" in branch.name for branch in outcome.branches)
    assert [point.id for point in outcome.hits] == ["main"]


def test_standalone_retrieval_does_not_treat_point_flag_as_generation_audit(monkeypatch):
    unrecorded = _point("unrecorded", "მოქმედი წესი", freshness_sla_met=None)
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [unrecorded])
    # The immutable payload cannot be updated by a post-build freshness audit. The
    # LegalAnswerService owns that gate; standalone retrieval only proves legal version.
    request = RetrievalRequest(query="რას ამბობს კანონი?")
    outcome = execute_accuracy_retrieval(_cfg(), None, _Embedder(), _Reranker(), request)
    assert not outcome.service_abstention


def test_present_law_selects_active_version_before_rerank(monkeypatch):
    active = _point(
        "active", "მოქმედი ვერსია", document_id="law-1", version_id="v2",
        status="in_force", effective_from="2025-01-01", effective_to=None,
    )
    repealed = _point(
        "repealed", "ძველი ვერსია", document_id="law-1", version_id="v1",
        status="repealed", effective_from="2020-01-01", effective_to="2025-01-01",
        repeal_date="2025-01-01",
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [repealed, active])

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რას ამბობს კანონი?"), current_date="2026-07-15",
    )

    assert not outcome.service_abstention
    assert [point.id for point in outcome.hits] == ["active"]
    assert outcome.candidate_ids == ("active",)


def test_overlapping_present_versions_fail_closed_before_rerank(monkeypatch):
    first = _point(
        "first", "პირველი ვერსია", document_id="law-1", version_id="v1",
        status="in_force", effective_from="2025-01-01", effective_to=None,
    )
    second = _point(
        "second", "მეორე ვერსია", document_id="law-1", version_id="v2",
        status="in_force", effective_from="2026-01-01", effective_to=None,
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [first, second])

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="რას ამბობს კანონი?"), current_date="2026-07-15",
    )

    assert outcome.abstention_reason == "operative_status_unverified"
    assert outcome.hits == ()


def test_present_law_normative_candidate_must_prove_operative_interval(monkeypatch):
    request = RetrievalRequest(query="რას ამბობს კანონი?")
    expired = _point(
        "expired", "ძველი წესი", effective_from="2020-01-01", effective_to="2026-01-01"
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [expired])

    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(), request, current_date="2026-07-15"
    )

    assert outcome.abstention_reason == "operative_status_unverified"

    operative = _point(
        "operative", "მოქმედი წესი", effective_from="2020-01-01", effective_to=None
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [operative])
    outcome = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(), request, current_date="2026-07-15"
    )
    assert not outcome.service_abstention


def test_exact_article_proves_version_while_case_and_historical_lookup_skip_current_status(monkeypatch):
    unrecorded = _point(
        "unrecorded", "მოქმედი, მაგრამ აუდიტის ნიშნულის გარეშე", freshness_sla_met=None
    )
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [unrecorded])
    monkeypatch.setattr(retrieval, "citation_lookup", lambda *a, **k: [unrecorded])

    exact_article = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="სამოქალაქო კოდექსის მუხლი 829"),
    )
    assert not exact_article.service_abstention

    historical = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="იმ დროისთვის რა წესი მოქმედებდა?"),
    )
    assert not historical.service_abstention

    case = execute_accuracy_retrieval(
        _cfg(), None, _Embedder(), _Reranker(),
        RetrievalRequest(query="უზენაესი სასამართლოს საქმე № ბს-729-721(კ-16) რას ადგენს?"),
    )
    assert not case.service_abstention
