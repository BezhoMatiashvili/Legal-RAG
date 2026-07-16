import dataclasses
import hashlib
from types import SimpleNamespace

import pytest

from ingest.chunking import chunk_document
from ingest.config import load_config
from ingest.generation import CANONICAL_PAYLOAD_REVISION, GENERATION_SCHEMA_VERSION
from ingest import legal_answer
from ingest.legal_answer import (
    AnswerOutcome,
    CalibrationDecision,
    CanonicalEvidenceRepository,
    ClaimKind,
    DraftAnswer,
    DraftClaim,
    DraftQuotation,
    EvidenceContractError,
    LegalAnswerService,
    canonical_evidence_from_point,
    parse_evidence_id,
    render_validated_answer,
    validate_draft,
)
from ingest.retrieval import AccuracyRetrievalOutcome, RetrievalBranch
from ingest.qdrant_store import build_payload, point_id
from ingest.sources import finalize_canonical_text, normalize


GENERATION = "generation-20260715"


def _cfg():
    return dataclasses.replace(
        load_config(),
        collection_name=f"georgian_legal__gen_{GENERATION}",
        generation_id=GENERATION,
        embedding_revision="a" * 40,
        tokenizer_revision="b" * 40,
        reranker_revision="c" * 40,
    )


TEXT = "კანონის ზუსტი ტექსტი და გაგრძელება"


def _point(pid="p1", chunk_index=0, text=TEXT, **overrides):
    import hashlib

    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        "generation_id": GENERATION,
        "source": "matsne",
        "source_authority": "official",
        "source_fingerprint": "a" * 64,
        "normalizer_revision": "canonical-source-normalizer-v2",
        "chunker_revision": "structural-article-clause-v2",
        "model_revision": "a" * 40,
        "document_id": "doc-1",
        "title": "კანონი",
        "document_number": "71",
        "document_type": "legislation",
        "article_id": "5",
        "clause_id": "5.1",
        "subarticle": "5.1",
        "chapter": "I",
        "heading_path": ["თავი I", "მუხლი 5"],
        "parent_id": "matsne:doc-1:v2:article:5:0",
        "article_start_chunk_index": 0,
        "parent_chunk_index": 0,
        "status": "in_force",
        "version_id": "v2",
        "supersedes": ["v1"],
        "effective_from": "2020-01-01T00:00:00Z",
        "effective_to": None,
        "repeal_date": None,
        "consolidation_status": "official_consolidated",
        "version_lineage_status": "complete",
        "version_lineage_complete": True,
        "content_complete": True,
        "extraction_status": "full_text",
        "official_url": "https://official.example/doc-1",
        "official_binary_url": None,
        "page_start": None,
        "page_end": None,
        "char_start": 100 + chunk_index * 100,
        "char_end": 100 + chunk_index * 100 + len(text),
        "offset_unit": "unicode_codepoint",
        "canonical_text_exact": True,
        "content_hash": "d" * 64,
        "canonical_content_hash": "d" * 64,
        "passage_hash": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "token_count": len(text.split()),
        "chunk_index": chunk_index,
        "freshness_sla_met": True,
    }
    payload.update(overrides)
    if "canonical_content_hash" not in overrides:
        payload["canonical_content_hash"] = payload["content_hash"]
    if "passage_id" not in overrides:
        material = (
            f"{payload['source']}\0{payload['document_id']}\0{payload['version_id']}\0"
            f"{payload['char_start']}\0{payload['char_end']}\0{payload['passage_hash']}"
        )
        payload["passage_id"] = "passage:" + hashlib.sha256(material.encode()).hexdigest()
    return SimpleNamespace(id=pid, score=0.95, payload=payload)


def _outcome(point):
    plan = SimpleNamespace(
        question="რა ამბობს კანონი?",
        language=SimpleNamespace(value="ka"),
        intent=SimpleNamespace(value="general_research"),
        as_of=None,
    )
    # The service only consumes the public outcome fields; a real run carries QueryPlan.
    return AccuracyRetrievalOutcome(
        hits=(point,),
        plan=plan,
        branches=(RetrievalBranch(
            "global_original", "რა ამბობს კანონი?", {}, (point.id,), 1.0, route="hybrid"
        ),),
        timings_ms={"total": 2.0},
        service_abstention=False,
        abstention_reason=None,
        degraded=False,
        degraded_reason=None,
        identity_ambiguous=False,
        retrieval_fingerprint="fp",
        generation_id=GENERATION,
        translator_version=None,
        result_hash="r" * 64,
        resolved_current_date="2026-07-15",
        candidate_ids=(point.id,),
    )


def test_canonical_evidence_id_binds_generation_point_and_passage_hash():
    evidence = canonical_evidence_from_point(_cfg(), _point())
    locator = parse_evidence_id(evidence.evidence_id)
    assert locator == {
        "generation_id": GENERATION,
        "point_id": "p1",
        "passage_hash": evidence.passage_hash,
    }
    with pytest.raises(EvidenceContractError, match="invalid evidence_id"):
        parse_evidence_id(evidence.evidence_id[:-1] + "0")


def test_schema_v2_producer_payload_round_trips_through_answer_contract():
    body = "მუხლი 5. კანონის ზუსტი ტექსტი და გაგრძელება"
    doc = normalize(
        "matsne",
        {
            "document_id": "doc-produced",
            "title": "კანონი",
            "document_url": "https://matsne.gov.ge/ka/document/view/doc-produced",
            "body_markdown": body,
            "entry_into_force_date": "2020-01-01",
            "freshness_sla_met": True,
        },
    )
    doc = finalize_canonical_text(doc, body)
    chunk = chunk_document(
        body,
        max_tokens=64,
        overlap=8,
        min_tokens=1,
        count_tokens=lambda value: len(value.split()),
    )[0]
    payload = build_payload(
        doc,
        chunk,
        document_chunk_count=1,
        document_state_hash="e" * 64,
        cfg=_cfg(),
    )
    point = SimpleNamespace(
        id=point_id(
            doc.source,
            doc.document_id,
            chunk.chunk_index,
            version_id=payload["version_id"],
        ),
        score=0.9,
        payload=payload,
    )

    evidence = canonical_evidence_from_point(_cfg(), point)

    assert evidence.schema_version == GENERATION_SCHEMA_VERSION
    assert evidence.canonical_payload_revision == CANONICAL_PAYLOAD_REVISION
    assert evidence.text == body
    assert evidence.offset_unit == "unicode_codepoint"
    assert evidence.passage_id == payload["passage_id"]
    assert evidence.subarticle_id == payload["subarticle"]


def test_composition_prompt_version_is_derived_from_exact_security_instructions():
    expected = "accuracy-first-atomic-claims-sha256:" + hashlib.sha256(
        legal_answer.STRICT_COMPOSER_INSTRUCTIONS.encode("utf-8")
    ).hexdigest()

    assert legal_answer.STRICT_PROMPT_VERSION == expected
    assert legal_answer.CompositionRequest.__dataclass_fields__["instructions"].default == (
        legal_answer.STRICT_COMPOSER_INSTRUCTIONS
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema_version": 1}, "canonical payload schema v2"),
        ({"canonical_payload_revision": "legacy"}, "canonical payload schema v2"),
        ({"generation_id": "old-generation"}, "active generation"),
        ({"passage_hash": "0" * 64}, "passage hash"),
        ({"passage_id": "passage:wrong"}, "passage identity"),
        ({"canonical_text_exact": False}, "exact canonical source slice"),
        ({"canonical_content_hash": "e" * 64}, "document content hash"),
        ({"offset_unit": "utf8_byte"}, "offset unit"),
        ({"token_count": "3"}, "token count"),
        ({"content_complete": False}, "incomplete"),
        ({"source_authority": "summary"}, "not official"),
        ({"source_fingerprint": "not-a-hash"}, "source fingerprint"),
        ({"normalizer_revision": None}, "provenance is incomplete"),
        ({"model_revision": "f" * 40}, "model revision"),
        ({"official_url": None}, "official evidence URL is missing"),
    ],
)
def test_noncanonical_or_incomplete_points_fail_closed(override, message):
    with pytest.raises(EvidenceContractError, match=message):
        canonical_evidence_from_point(_cfg(), _point(**override))


def test_validator_checks_exact_quote_offsets_hash_links_and_version():
    point = _point()
    pack = CanonicalEvidenceRepository(_cfg(), None).build_pack(
        (point,), retrieval_result_hash="r" * 64
    )
    evidence = pack.items[0]
    quote = "ზუსტი ტექსტი"
    rel = TEXT.index(quote)
    draft = DraftAnswer(
        "ციტირებული კანონი და სისტემის ინტერპრეტაცია.",
        (
            DraftClaim(
                "c1",
                "კანონი შეიცავს ამ ტექსტს.",
                ClaimKind.QUOTED_LAW,
                (evidence.evidence_id,),
                (DraftQuotation(
                    evidence.evidence_id,
                    quote,
                    evidence.char_start + rel,
                    evidence.char_start + rel + len(quote),
                ),),
            ),
            DraftClaim(
                "c2",
                "ეს არის სისტემის ინტერპრეტაცია.",
                ClaimKind.INTERPRETATION,
                (evidence.evidence_id,),
            ),
        ),
    )
    result = validate_draft(draft, pack, as_of="2024-01-01")
    assert result.valid and result.material_claim_coverage == 1.0
    assert len(result.claims[0].quotations[0].quote_hash) == 64

    bad = dataclasses.replace(
        draft,
        claims=(dataclasses.replace(
            draft.claims[0],
            quotations=(DraftQuotation(
                evidence.evidence_id,
                "შეცვლილი",
                evidence.char_start + rel,
                evidence.char_start + rel + len(quote),
            ),),
        ),),
    )
    invalid = validate_draft(bad, pack, as_of="2019-01-01")
    assert not invalid.valid
    assert {issue.code for issue in invalid.issues} == {
        "quotation_mismatch", "wrong_or_unknown_version"
    }


def test_validator_rejects_zero_claims_and_unlinked_nonmaterial_claims():
    pack = CanonicalEvidenceRepository(_cfg(), None).build_pack(
        (_point(),), retrieval_result_hash="r" * 64
    )

    empty = validate_draft(DraftAnswer("looks like an answer", ()), pack, as_of=None)
    assert not empty.valid and empty.material_claim_coverage == 0.0
    assert [issue.code for issue in empty.issues] == ["no_claims"]

    unlinked = validate_draft(
        DraftAnswer(
            "uncited interpretation",
            (DraftClaim(
                "c1", "uncited interpretation", ClaimKind.INTERPRETATION, (),
                material=False,
            ),),
        ),
        pack,
        as_of=None,
    )
    assert not unlinked.valid
    assert "unsupported_material_claim" in {issue.code for issue in unlinked.issues}


def test_rendered_interpretation_visibly_includes_evidence_ids():
    pack = CanonicalEvidenceRepository(_cfg(), None).build_pack(
        (_point(),), retrieval_result_hash="r" * 64
    )
    evidence_id = pack.items[0].evidence_id
    validated = validate_draft(
        DraftAnswer(
            "supported interpretation",
            (DraftClaim(
                "c1", "supported interpretation", ClaimKind.INTERPRETATION,
                (evidence_id,), material=False,
            ),),
        ),
        pack,
        as_of=None,
    )

    rendered = render_validated_answer(validated.claims)

    assert validated.valid
    assert f"evidence: {evidence_id}" in rendered


class _Composer:
    version = "generator-rev"

    def __init__(self):
        self.repairs = 0

    def compose(self, request):
        return DraftAnswer(
            "unsupported first draft",
            (DraftClaim("bad", "unsupported", ClaimKind.INTERPRETATION, ()),),
        )

    def repair(self, request, draft, issues):
        self.repairs += 1
        ev = request.evidence.items[0]
        quote = "ზუსტი ტექსტი"
        rel = ev.text.index(quote)
        return DraftAnswer(
            "კანონის ციტატა ცალკეა; ინტერპრეტაცია ცალკეა.",
            (DraftClaim(
                "fixed",
                "კანონი შეიცავს მითითებულ ტექსტს.",
                ClaimKind.QUOTED_LAW,
                (ev.evidence_id,),
                (DraftQuotation(
                    ev.evidence_id, quote,
                    ev.char_start + rel, ev.char_start + rel + len(quote),
                ),),
            ),),
        )


class _Calibrator:
    version = "heldout-calibrator-v1"

    def assess(self, features):
        assert features.validator_passed and features.evidence_coverage == 1.0
        return CalibrationDecision(True)


class _MalformedComposer:
    version = "malformed-generator"

    def compose(self, request):
        return {"answer_text": "not the typed schema", "claims": []}

    def repair(self, request, draft, issues):
        raise AssertionError("malformed output must not enter repair")


class _RaisingCalibrator:
    version = "broken-calibrator"

    def assess(self, features):
        raise RuntimeError("calibration runtime failed")


class _AllowFreshnessGuard:
    def __init__(self):
        self.sources = None

    def decision(self, relevant_sources, *, now=None):
        self.sources = relevant_sources
        return SimpleNamespace(
            allowed=True,
            generation_id=GENERATION,
            audit_id="audit-1",
            relevant_sources=relevant_sources,
            reasons=(),
        )


def test_service_allows_one_repair_then_returns_typed_answer(monkeypatch):
    point = _point()
    monkeypatch.setattr(legal_answer, "execute_accuracy_retrieval", lambda *a, **k: _outcome(point))
    composer = _Composer()
    result = LegalAnswerService(
        _cfg(), None, object(), object(), composer=composer, calibrator=_Calibrator(),
        freshness_guard=_AllowFreshnessGuard(),
    ).ask("რა ამბობს კანონი?")
    assert result.outcome is AnswerOutcome.ANSWER
    assert composer.repairs == 1
    assert result.claims[0].kind is ClaimKind.QUOTED_LAW
    assert result.answer_text.startswith("## Quoted law")
    assert "## System interpretation" in result.answer_text
    assert result.trace.repair_attempted
    assert len(result.trace.trace_id) == 64
    assert result.versions.generator == "generator-rev"
    assert result.trace.retrieval_result_hash == "r" * 64
    assert result.trace.resolved_current_date == "2026-07-15"
    assert result.trace.ranked_hits == (
        legal_answer.RankedHitTrace("p1", "0.94999999999999996", None),
    )
    assert result.trace.candidate_ids == ("p1",)
    assert result.trace.service_abstention is False
    assert result.trace.abstention_reason is None
    assert result.trace.degraded is False
    assert result.trace.degraded_reason is None
    assert result.trace.identity_ambiguous is False
    assert result.trace.translator_version is None
    assert result.trace.retrieval_branches[0].route == "hybrid"
    assert result.trace.evidence_pack_id == result.evidence.pack_id
    assert result.trace.freshness_audit_id == "audit-1"
    assert result.trace.freshness_decision == "eligible"
    assert result.trace.calibration_allowed is True
    assert result.trace.calibration_reason is None


def test_verified_generation_audit_allows_immutable_payload_without_freshness_flag(monkeypatch):
    point = _point(freshness_sla_met=None)
    monkeypatch.setattr(
        legal_answer, "execute_accuracy_retrieval", lambda *a, **k: _outcome(point)
    )
    guard = _AllowFreshnessGuard()

    result = LegalAnswerService(
        _cfg(), None, object(), object(), composer=_Composer(), calibrator=_Calibrator(),
        freshness_guard=guard,
    ).ask("რა ამბობს კანონი?")

    assert result.outcome is AnswerOutcome.ANSWER
    assert guard.sources == ("matsne",)
    assert result.trace.freshness_audit_id == "audit-1"


def test_malformed_composer_output_returns_typed_abstention(monkeypatch):
    point = _point()
    monkeypatch.setattr(
        legal_answer, "execute_accuracy_retrieval", lambda *a, **k: _outcome(point)
    )

    result = LegalAnswerService(
        _cfg(), None, object(), object(),
        composer=_MalformedComposer(), calibrator=_Calibrator(),
        freshness_guard=_AllowFreshnessGuard(),
    ).ask("რა ამბობს კანონი?")

    assert result.outcome is AnswerOutcome.ABSTAIN
    assert result.abstention_reason == "malformed_composer_output"
    assert [issue.code for issue in result.validation_issues] == [
        "malformed_composer_output"
    ]


def test_calibrator_exception_returns_typed_abstention(monkeypatch):
    point = _point()
    monkeypatch.setattr(
        legal_answer, "execute_accuracy_retrieval", lambda *a, **k: _outcome(point)
    )

    result = LegalAnswerService(
        _cfg(), None, object(), object(),
        composer=_Composer(), calibrator=_RaisingCalibrator(),
        freshness_guard=_AllowFreshnessGuard(),
    ).ask("რა ამბობს კანონი?")

    assert result.outcome is AnswerOutcome.ABSTAIN
    assert result.abstention_reason == "selective_risk_calibrator_degraded"
    assert [issue.code for issue in result.validation_issues] == [
        "selective_risk_calibrator_degraded"
    ]
    assert result.trace.repair_attempted
    assert result.trace.calibration_allowed is None
    assert result.trace.calibration_reason == "degraded"


def test_service_without_generator_or_calibrator_abstains_before_retrieval(monkeypatch):
    monkeypatch.setattr(
        legal_answer,
        "execute_accuracy_retrieval",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve")),
    )
    no_generator = LegalAnswerService(_cfg(), None, None, None).ask("რა ამბობს კანონი?")
    assert no_generator.outcome is AnswerOutcome.ABSTAIN
    assert no_generator.abstention_reason == "generator_unavailable"

    no_calibrator = LegalAnswerService(
        _cfg(), None, None, None, composer=_Composer()
    ).ask("რა ამბობს კანონი?")
    assert no_calibrator.abstention_reason == "selective_risk_calibrator_unavailable"


def test_present_law_service_requires_verified_freshness_audit_before_retrieval(monkeypatch):
    monkeypatch.setattr(
        legal_answer,
        "execute_accuracy_retrieval",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve")),
    )

    result = LegalAnswerService(
        _cfg(), None, object(), object(), composer=_Composer(), calibrator=_Calibrator()
    ).ask("რა ამბობს კანონი?")

    assert result.outcome is AnswerOutcome.ABSTAIN
    assert result.abstention_reason == "freshness_audit_unavailable"


def test_freshness_audit_denial_is_typed_and_binds_retrieved_sources(monkeypatch):
    point = _point()
    monkeypatch.setattr(
        legal_answer, "execute_accuracy_retrieval", lambda *a, **k: _outcome(point)
    )

    class DenyGuard:
        def decision(self, relevant_sources, *, now=None):
            assert relevant_sources == ("matsne",)
            reason = SimpleNamespace(
                code="source_snapshot_expired",
                source="matsne",
                detail="freshness deadline passed",
            )
            return SimpleNamespace(
                allowed=False,
                generation_id=GENERATION,
                audit_id="audit-expired",
                relevant_sources=relevant_sources,
                reasons=(reason,),
            )

    result = LegalAnswerService(
        _cfg(), None, object(), object(), composer=_Composer(), calibrator=_Calibrator(),
        freshness_guard=DenyGuard(),
    ).ask("რა ამბობს კანონი?")

    assert result.outcome is AnswerOutcome.ABSTAIN
    assert result.abstention_reason == "current_law_freshness_unverified"
    assert [issue.code for issue in result.validation_issues] == [
        "source_snapshot_expired"
    ]
    assert result.trace.freshness_audit_id == "audit-expired"
    assert result.trace.freshness_decision == "abstain"
    assert result.trace.evidence_pack_id is None


def test_unsupported_language_clarifies_even_without_model_runtime():
    result = LegalAnswerService(_cfg(), None, None, None).ask("cual es la ley")
    assert result.outcome is AnswerOutcome.CLARIFY
    assert result.abstention_reason == "unsupported_language"
    assert "Georgian or English" in result.clarification_question
