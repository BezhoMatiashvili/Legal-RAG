"""Focused chain-of-custody tests for blind answer release review artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from eval.answer_release_review import (
    CandidateOutput,
    CandidateReview,
    RankedHit,
    ReleaseEvidenceError,
    aggregate_release_candidate,
)
from eval.release_gate import (
    ReleasePolicy,
    ReleasePolicySlice,
    evaluate_release_candidate,
)
from eval.v3_dataset import (
    BlindProtocol,
    CanonicalDocument,
    ErrorCode,
    ErrorFinding,
    ErrorSeverity,
    EvidenceEquivalenceGroup,
    ExpectedOutcome,
    Language,
    QuestionOrigin,
    QuestionProvenance,
    QuestionTag,
    ReviewerJudgment,
    ReviewerRole,
    RiskLevel,
    Split,
    V3Dataset,
    V3Manifest,
    V3Question,
    anchor_span,
    blind_ids_hash,
    dataset_hash,
    sha256_text,
)
from ingest.legal_answer import make_evidence_id
from ingest.query_planner import plan_query

GENERATION = "generation-sealed-001"
CONFIGURATION = sha256_text("private-pipeline-configuration-v1")
FRESHNESS_AUDIT = sha256_text("verified-freshness-audit")
LAW_PASSAGE = "მოქმედი ნორმის ზუსტი ტექსტი."
LAW_TEXT = "x" * 100 + LAW_PASSAGE
LAW_DOCUMENT = CanonicalDocument(
    source="matsne",
    document_id="law-1",
    version_id="version-2",
    lineage_family_id="lineage-law-1",
    text=LAW_TEXT,
    content_sha256=sha256_text(LAW_TEXT),
    content_complete=True,
    authoritative=True,
    version_lineage_complete=True,
    version_ambiguous=False,
    effective_from="2024-01-01",
    effective_to=None,
    status="in_force",
    repeal_date=None,
    consolidation_status="official",
)
TRAIN_DOCUMENT_A = CanonicalDocument(
    source="matsne",
    document_id="training-law",
    version_id="training-law@2019",
    lineage_family_id="lineage-training-law",
    text="A [historical training rule] end",
    content_sha256=sha256_text("A [historical training rule] end"),
    content_complete=True,
    authoritative=True,
    version_lineage_complete=True,
    version_ambiguous=False,
    effective_from="2018-01-01",
    effective_to="2020-01-01",
    status="repealed",
    repeal_date="2020-01-01",
    consolidation_status="official",
)
TRAIN_DOCUMENT_B = CanonicalDocument(
    source="tas",
    document_id="training-incomplete",
    version_id="training-incomplete@1",
    lineage_family_id="lineage-training-incomplete",
    text="B [incomplete adversarial summary] end",
    content_sha256=sha256_text("B [incomplete adversarial summary] end"),
    content_complete=False,
    authoritative=False,
    version_lineage_complete=True,
    version_ambiguous=False,
    effective_from=None,
    effective_to=None,
)
CANONICAL_DOCUMENTS = (LAW_DOCUMENT, TRAIN_DOCUMENT_A, TRAIN_DOCUMENT_B)
CURRENT_GROUP = EvidenceEquivalenceGroup(
    group_id="current-rule",
    proposition="The currently operative rule.",
    required=True,
    alternatives=(
        anchor_span(
            LAW_DOCUMENT,
            evidence_id="gold-current-rule",
            char_start=100,
            char_end=len(LAW_TEXT),
            article_id="7",
        ),
    ),
)


def _versions() -> dict:
    return {
        "corpus_generation": GENERATION,
        "retriever": "bge-m3@revision-1",
        "reranker": "qwen-reranker@revision-1",
        "translator": None,
        "generator": "qwen-private@revision-1",
        "prompt": "strict-atomic-claims@revision-1",
        "calibrator": "held-out-risk@revision-1",
    }


def _judgment(
    reviewer_id: str,
    outcome: ExpectedOutcome,
    evidence_group_ids: tuple[str, ...] = (),
) -> ReviewerJudgment:
    return ReviewerJudgment(
        reviewer_id=reviewer_id,
        role=ReviewerRole.REVIEWER,
        blinded=True,
        answerable=outcome is ExpectedOutcome.ANSWER,
        expected_outcome=outcome,
        evidence_group_ids=evidence_group_ids,
        overall_severity=ErrorSeverity.NONE,
        errors=(),
        rationale="The expected disposition was independently checked.",
    )


def _question(
    question_id: str,
    text: str,
    *,
    outcome: ExpectedOutcome,
    risk: RiskLevel,
    tags: tuple[QuestionTag, ...],
    language: Language = Language.KA,
    language_code: str = "ka",
    split: Split = Split.BLIND,
    groups: tuple[EvidenceEquivalenceGroup, ...] = (),
    as_of: str | None = None,
    identifiers: tuple[str, ...] = (),
    origin: QuestionOrigin = QuestionOrigin.EXPERT_WRITTEN,
) -> V3Question:
    selected_groups = (
        tuple(group.group_id for group in groups if group.required)
        if outcome is ExpectedOutcome.ANSWER
        else ()
    )
    families = tuple(
        dict.fromkeys(
            span.lineage_family_id
            for group in groups
            for span in group.alternatives
        )
    )
    real = origin is QuestionOrigin.REAL_ANONYMIZED_LAWYER
    return V3Question(
        question_id=question_id,
        question=text,
        language=language,
        language_code=language_code,
        split=split,
        tags=tags,
        risk_level=risk,
        as_of=as_of,
        identifiers=identifiers,
        partition_family_ids=families,
        provenance=QuestionProvenance(
            origin=origin,
            source_record_id=f"sealed-{question_id}",
            anonymized=real,
            pii_reviewed=real,
            original_text_retained=False,
            question_sha256=sha256_text(text),
            note="Hermetic release-review fixture.",
        ),
        evidence_groups=groups,
        reference_response="Sealed reference disposition.",
        judgments=(
            _judgment(f"gold-a-{question_id}", outcome, selected_groups),
            _judgment(f"gold-b-{question_id}", outcome, selected_groups),
        ),
    )


@pytest.fixture(scope="module")
def release_fixture():
    blind_questions = tuple(
        _question(
            f"q-answer-{index:04d}",
            f"რა არის მოქმედი წესი? კითხვა {index}",
            outcome=ExpectedOutcome.ANSWER,
            risk=RiskLevel.HIGH,
            tags=(QuestionTag.CURRENT_LAW, QuestionTag.RISK_HIGH),
            groups=(CURRENT_GROUP,),
        )
        for index in range(1000)
    )
    train_group_a = EvidenceEquivalenceGroup(
        "training-rule",
        "Historical training authority.",
        True,
        (
            anchor_span(
                TRAIN_DOCUMENT_A,
                evidence_id="gold-training-rule",
                char_start=3,
                char_end=27,
            ),
        ),
    )
    train_group_b = EvidenceEquivalenceGroup(
        "training-incomplete",
        "Incomplete adversarial authority.",
        True,
        (
            anchor_span(
                TRAIN_DOCUMENT_B,
                evidence_id="gold-training-incomplete",
                char_start=3,
                char_end=33,
            ),
        ),
    )
    training_question = _question(
        "q-training-coverage",
        "Which exact authorities conflict for identifier 12-34 in 2019?",
        outcome=ExpectedOutcome.ABSTAIN,
        risk=RiskLevel.MEDIUM,
        tags=(
            QuestionTag.PARAPHRASE,
            QuestionTag.TYPO,
            QuestionTag.IDENTIFIER_HEAVY,
            QuestionTag.EXACT_AUTHORITY_LOOKUP,
            QuestionTag.HISTORICAL_AS_OF,
            QuestionTag.MULTI_EVIDENCE,
            QuestionTag.CONFLICTING_AUTHORITIES,
            QuestionTag.INCOMPLETE_SOURCE,
            QuestionTag.AMBIGUOUS_IDENTIFIER,
            QuestionTag.UNANSWERABLE,
            QuestionTag.PROMPT_INJECTION,
            QuestionTag.RISK_MEDIUM,
        ),
        language=Language.EN,
        language_code="en",
        split=Split.TRAIN,
        groups=(train_group_a, train_group_b),
        as_of="2019-07-15",
        identifiers=("12-34",),
        origin=QuestionOrigin.REAL_ANONYMIZED_LAWYER,
    )
    unsupported_question = _question(
        "q-dev-unsupported",
        "Quelle affaire applique cette loi?",
        outcome=ExpectedOutcome.CLARIFY,
        risk=RiskLevel.LOW,
        tags=(
            QuestionTag.CASES_APPLYING_AUTHORITY,
            QuestionTag.UNSUPPORTED_LANGUAGE,
            QuestionTag.RISK_LOW,
        ),
        language=Language.UNSUPPORTED,
        language_code="fr",
        split=Split.DEV,
    )
    questions = (*blind_questions, training_question, unsupported_question)
    blind_hash = blind_ids_hash(
        question.question_id for question in blind_questions
    )
    dataset = V3Dataset(
        manifest=V3Manifest(
            dataset_id="sealed-v3-release-fixture",
            corpus_generation=GENERATION,
            corpus_as_of="2026-07-15",
            blind_protocol=BlindProtocol(
                planned_question_count=1000,
                target_answered_count=1000,
                predeclared_at="2026-07-01T00:00:00+04:00",
                sampling_plan_sha256=sha256_text("predeclared sampling plan"),
                sealed=True,
                blind_ids_sha256=blind_hash,
            ),
        ),
        questions=questions,
    )
    policy = ReleasePolicy(
        policy_id="sealed-policy-v1",
        blind_question_set_sha256=blind_hash,
        slices=(
            ReleasePolicySlice(
                name="risk_high",
                baseline=1.0,
                category="high_risk",
            ),
        ),
    )
    return dataset, policy


def _pack_id(retrieval_hash: str, evidence_ids: list[str]) -> str:
    material = {
        "generation_id": GENERATION,
        "retrieval_hash": retrieval_hash,
        "evidence_ids": evidence_ids,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _answer_result(
    *,
    question: str = "რა არის მოქმედი წესი? კითხვა 0",
    resolved_current_date: str = "2026-07-15",
    elapsed_ms: float = 1.0,
    fabricate_evidence: bool = False,
    tamper_quote: bool = False,
) -> dict:
    text = (
        "გამოგონილი, მაგრამ შიდა ჰეშებთან შეთანხმებული ტექსტი."
        if fabricate_evidence
        else LAW_PASSAGE
    )
    point_id = "matsne:law-1:version-2:0"
    passage_hash = sha256_text(text)
    evidence_id = make_evidence_id(
        generation_id=GENERATION,
        point_id=point_id,
        passage_hash=passage_hash,
    )
    branch = {
        "name": "global_original",
        "query": question,
        "filters": {},
        "hit_ids": [point_id],
        "elapsed_ms": elapsed_ms,
        "route": "global",
    }
    branch_hash = _json_hash(
        {
            "name": branch["name"],
            "query": branch["query"],
            "filters": branch["filters"],
            "hits": branch["hit_ids"],
            "route": branch["route"],
        }
    )
    ranked_hits = [
        {
            "point_id": point_id,
            "score": "0.94999999999999996",
            "match_type": None,
        }
    ]
    retrieval_fingerprint = "retrieval-fingerprint-v1"
    query_plan = plan_query(question, language="ka")
    retrieval_hash = _json_hash(
        {
            "question": query_plan.question,
            "language": query_plan.language.value,
            "intent": query_plan.intent.value,
            "as_of": query_plan.as_of,
            "resolved_current_date": resolved_current_date,
            "fingerprint": retrieval_fingerprint,
            "generation_id": GENERATION,
            "service_abstention": False,
            "abstention_reason": None,
            "degraded": False,
            "degraded_reason": None,
            "identity_ambiguous": False,
            "translator_version": None,
            "branches": [
                {
                    "name": branch["name"],
                    "query": branch["query"],
                    "filters": branch["filters"],
                    "hit_ids": branch["hit_ids"],
                    "route": branch["route"],
                }
            ],
            "candidate_ids": [point_id],
            "final_ranking": [
                {"point_id": hit["point_id"], "score": hit["score"]}
                for hit in ranked_hits
            ],
        }
    )
    pack_id = _pack_id(retrieval_hash, [evidence_id])
    answer_text = "## Quoted law\n\n- მოქმედი ნორმა."
    quoted_text = "შეცვლილი ციტატა" if tamper_quote else text
    trace_id = _json_hash(
        {
            "question": question,
            "versions": _versions(),
            "retrieval_hash": retrieval_hash,
            "branches": [branch_hash],
            "validation_codes": [],
            "repair_attempted": False,
            "evidence_pack_id": pack_id,
            "freshness_audit_id": FRESHNESS_AUDIT,
            "freshness_decision": "eligible",
            "calibration_allowed": True,
            "calibration_reason": None,
        }
    )
    return {
        "outcome": "answer",
        "answer_text": answer_text,
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "მოქმედი ნორმა.",
                "kind": "quoted_law",
                "evidence_ids": [evidence_id],
                "quotations": [
                    {
                        "evidence_id": evidence_id,
                        "quote": quoted_text,
                        "char_start": 100,
                        "char_end": 100 + len(text),
                        "quote_hash": sha256_text(quoted_text),
                    }
                ],
                "material": True,
                "qualified_identity": False,
            }
        ],
        "evidence": {
            "pack_id": pack_id,
            "generation_id": GENERATION,
            "retrieval_result_hash": retrieval_hash,
            "items": [
                {
                    "evidence_id": evidence_id,
                    "point_id": point_id,
                    "schema_version": 2,
                    "canonical_payload_revision": "canonical-evidence-v2",
                    "generation_id": GENERATION,
                    "source": "matsne",
                    "source_authority": "official",
                    "source_fingerprint": sha256_text("official-source-record"),
                    "normalizer_revision": "normalizer-v1",
                    "chunker_revision": "structural-v1",
                    "model_revision": "bge-m3@revision-1",
                    "document_id": "law-1",
                    "document_title": "Fixture law",
                    "document_number": "1",
                    "registration_code": None,
                    "document_type": "law",
                    "version_id": "version-2",
                    "article_id": "7",
                    "clause_id": "7.1",
                    "subarticle_id": "7.1",
                    "chapter": "I",
                    "heading_path": ["Chapter I", "Article 7"],
                    "parent_id": "matsne:law-1:version-2:article:7",
                    "court": None,
                    "case_number": None,
                    "status": "in_force",
                    "supersedes": ["version-1"],
                    "effective_from": "2024-01-01",
                    "effective_to": None,
                    "repeal_date": None,
                    "consolidation_status": "official",
                    "version_lineage_status": "complete",
                    "version_lineage_complete": True,
                    "content_complete": True,
                    "extraction_status": "full_text",
                    "official_url": "https://matsne.gov.ge/law-1",
                    "official_binary_url": None,
                    "page_start": None,
                    "page_end": None,
                    "passage_hash": passage_hash,
                    "passage_id": f"passage:{sha256_text('law-1:version-2:100')}",
                    "content_hash": sha256_text(
                        "fabricated-document" if fabricate_evidence else LAW_TEXT
                    ),
                    "text": text,
                    "token_count": 8,
                    "char_start": 100,
                    "char_end": 100 + len(text),
                    "offset_unit": "unicode_codepoint",
                    "identity_ambiguous": False,
                }
            ],
            "token_count": 8,
            "max_tokens": 4096,
        },
        "clarification_question": None,
        "abstention_reason": None,
        "validation_issues": [],
        "versions": _versions(),
        "trace": {
            "trace_id": trace_id,
            "retrieval_result_hash": retrieval_hash,
            "retrieval_fingerprint": retrieval_fingerprint,
            "generation_id": GENERATION,
            "service_abstention": False,
            "abstention_reason": None,
            "degraded": False,
            "degraded_reason": None,
            "identity_ambiguous": False,
            "translator_version": None,
            "branch_hashes": [branch_hash],
            "retrieval_branches": [branch],
            "candidate_ids": [point_id],
            "ranked_hits": ranked_hits,
            "retrieval_timings_ms": {"total": elapsed_ms},
            "resolved_current_date": resolved_current_date,
            "evidence_pack_id": pack_id,
            "freshness_audit_id": FRESHNESS_AUDIT,
            "freshness_decision": "eligible",
            "validation_codes": [],
            "repair_attempted": False,
            "calibration_features": {},
            "calibration_allowed": True,
            "calibration_reason": None,
            "answer_hash": sha256_text(answer_text),
        },
        "trace_id": trace_id,
    }


def _abstention_result(question_id: str) -> dict:
    trace_id = sha256_text(f"stable-abstention-trace:{question_id}")
    return {
        "outcome": "abstain",
        "answer_text": None,
        "claims": [],
        "evidence": None,
        "clarification_question": None,
        "abstention_reason": "insufficient_evidence",
        "validation_issues": [],
        "versions": _versions(),
        "trace": {
            "trace_id": trace_id,
            "retrieval_result_hash": None,
            "ranked_hits": [],
            "retrieval_branches": [],
            "candidate_ids": [],
            "generation_id": GENERATION,
        },
        "trace_id": trace_id,
    }


def _canonical_evidence() -> dict[str, dict]:
    item = _answer_result()["evidence"]["items"][0]
    return {item["evidence_id"]: dict(item)}


def _runs(
    dataset: V3Dataset,
    *,
    resolved_current_date: str = "2026-07-15",
    fabricate_primary_evidence: bool = False,
    tamper_primary_quote: bool = False,
) -> tuple[tuple[CandidateOutput, ...], tuple[CandidateOutput, ...]]:
    dataset_sha = dataset_hash(dataset)
    blind_sha = dataset.manifest.blind_protocol.blind_ids_sha256
    assert blind_sha is not None
    runs = []
    for repeat_index, elapsed in enumerate((1.0, 9.0)):
        outputs = []
        for question in dataset.questions:
            if question.split is not Split.BLIND:
                continue
            is_answer = True
            outputs.append(
                CandidateOutput.from_result(
                    dataset_id=dataset.manifest.dataset_id,
                    dataset_sha256=dataset_sha,
                    blind_question_set_sha256=blind_sha,
                    candidate_id="candidate-private-v1",
                    generation_id=GENERATION,
                    configuration_sha256=CONFIGURATION,
                    run_id=f"repeat-{repeat_index}",
                    repeat_index=repeat_index,
                    question_id=question.question_id,
                    question_sha256=sha256_text(question.question),
                    ranking=(
                        (
                            RankedHit(
                                "matsne:law-1:version-2:0",
                                0.95,
                                ("global",),
                            ),
                        )
                        if is_answer
                        else ()
                    ),
                    answer_result=(
                        _answer_result(
                            question=question.question,
                            resolved_current_date=resolved_current_date,
                            elapsed_ms=elapsed,
                            fabricate_evidence=(
                                fabricate_primary_evidence
                                and repeat_index == 0
                                and question.question_id == "q-answer-0000"
                            ),
                            tamper_quote=(
                                tamper_primary_quote
                                and repeat_index == 0
                                and question.question_id == "q-answer-0000"
                            ),
                        )
                        if is_answer
                        else _abstention_result(question.question_id)
                    ),
                    latency_ms=(10.0 if is_answer else 5.0) + elapsed,
                )
            )
        runs.append(tuple(outputs))
    return runs[0], runs[1]


def _review(
    output: CandidateOutput,
    reviewer_id: str,
    *,
    severity: ErrorSeverity = ErrorSeverity.NONE,
    errors: tuple[ErrorFinding, ...] = (),
    role: ReviewerRole = ReviewerRole.REVIEWER,
) -> CandidateReview:
    return CandidateReview(
        dataset_id=output.dataset_id,
        candidate_id=output.candidate_id,
        question_id=output.question_id,
        candidate_output_sha256=output.sha256,
        reviewer_id=reviewer_id,
        role=role,
        blinded=True,
        overall_severity=severity,
        errors=errors,
        rationale="The exact sealed candidate output was independently reviewed.",
    )


def _agreeing_reviews(
    primary: tuple[CandidateOutput, ...],
) -> tuple[CandidateReview, ...]:
    reviews = []
    for output in primary:
        reviews.extend((_review(output, "lawyer-a"), _review(output, "lawyer-b")))
    return tuple(reviews)


def test_sealed_dataset_and_output_bindings_are_enforced(release_fixture):
    dataset, policy = release_fixture
    runs = _runs(dataset)
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=runs,
        reviews=_agreeing_reviews(runs[0]),
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    assert bundle.candidate.metadata["dataset_sha256"] == dataset_hash(dataset)

    bad_output = replace(runs[0][0], dataset_sha256="0" * 64)
    with pytest.raises(ReleaseEvidenceError, match="another dataset/generation"):
        aggregate_release_candidate(
            dataset,
            policy=policy,
            runs=((bad_output, runs[0][1]), runs[1]),
            reviews=_agreeing_reviews((bad_output, runs[0][1])),
            canonical_evidence=_canonical_evidence(),
            canonical_documents=CANONICAL_DOCUMENTS,
        )

    bad_protocol = replace(
        dataset.manifest.blind_protocol,
        blind_ids_sha256="f" * 64,
    )
    bad_dataset = replace(
        dataset,
        manifest=replace(dataset.manifest, blind_protocol=bad_protocol),
    )
    with pytest.raises(ReleaseEvidenceError, match="canonically validated"):
        aggregate_release_candidate(
            bad_dataset,
            policy=policy,
            runs=runs,
            reviews=_agreeing_reviews(runs[0]),
            canonical_evidence=_canonical_evidence(),
            canonical_documents=CANONICAL_DOCUMENTS,
        )


def test_repeat_hashes_ignore_wall_clock_noise_but_seal_full_outputs(release_fixture):
    dataset, policy = release_fixture
    runs = _runs(dataset)
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=runs,
        reviews=_agreeing_reviews(runs[0]),
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    first, second = bundle.repeat_manifests
    assert first.ranking_sha256 == second.ranking_sha256
    assert first.answer_sha256 == second.answer_sha256
    assert first.output_manifest_sha256 != second.output_manifest_sha256
    assert runs[0][0].sha256 != runs[1][0].sha256


@pytest.mark.parametrize(
    "ranking",
    [
        (RankedHit("different-point", 0.95, ("global",)),),
        (RankedHit("matsne:law-1:version-2:0", 0.90, ("global",)),),
        (RankedHit("matsne:law-1:version-2:0", 0.95, ("invented-route",)),),
    ],
)
def test_caller_supplied_ranking_must_match_traced_ids_scores_and_routes(
    release_fixture,
    ranking,
):
    dataset, _policy = release_fixture
    primary, _repeat = _runs(dataset)
    forged = replace(primary[0], ranking=ranking)
    with pytest.raises(ReleaseEvidenceError, match="ranking"):
        _ = forged.sha256


def test_review_disagreement_requires_and_uses_third_adjudicator(release_fixture):
    dataset, policy = release_fixture
    runs = _runs(dataset)
    severe = ErrorFinding(ErrorCode.WRONG_AUTHORITY, ErrorSeverity.SEVERE, "claim-1")
    primary = runs[0]
    reviews = list(_agreeing_reviews(primary))
    reviews[1] = _review(
        primary[0],
        "lawyer-b",
        severity=ErrorSeverity.SEVERE,
        errors=(severe,),
    )
    with pytest.raises(ReleaseEvidenceError, match="third adjudicator"):
        aggregate_release_candidate(
            dataset,
            policy=policy,
            runs=runs,
            reviews=reviews,
            canonical_evidence=_canonical_evidence(),
            canonical_documents=CANONICAL_DOCUMENTS,
        )

    reviews.append(
        _review(
            primary[0],
            "lawyer-c",
            severity=ErrorSeverity.SEVERE,
            errors=(severe,),
            role=ReviewerRole.ADJUDICATOR,
        )
    )
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=runs,
        reviews=reviews,
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    assert bundle.candidate.severe_errors == 1
    assert bundle.candidate.material_errors == 1


def test_mechanical_audit_detects_tampered_exact_quotation(release_fixture):
    dataset, policy = release_fixture
    runs = _runs(dataset, tamper_primary_quote=True)
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=runs,
        reviews=_agreeing_reviews(runs[0]),
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    assert bundle.candidate.answered_outcomes == 1000
    assert bundle.candidate.mechanically_valid_answers == 999
    assert (
        bundle.candidate.mechanical_checks_valid
        < bundle.candidate.mechanical_checks_total
    )
    assert bundle.candidate.validation_failed_answers == 1


@pytest.mark.parametrize(
    "field", ["branch_hashes", "candidate_ids", "retrieval_result_hash", "trace_id"]
)
def test_mechanical_audit_recomputes_branch_retrieval_and_trace_hashes(
    release_fixture,
    field,
):
    dataset, policy = release_fixture
    runs = _runs(dataset)
    result = runs[0][0].answer_result
    replacement_hash = sha256_text(f"tampered-{field}")
    if field == "branch_hashes":
        result["trace"]["branch_hashes"] = [replacement_hash]
    elif field == "candidate_ids":
        result["trace"]["candidate_ids"].append("unranked-extra-candidate")
    elif field == "retrieval_result_hash":
        result["trace"]["retrieval_result_hash"] = replacement_hash
    else:
        result["trace"]["trace_id"] = replacement_hash
        result["trace_id"] = replacement_hash
    canonical_json = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    forged = replace(runs[0][0], answer_result_json=canonical_json)
    forged_runs = ((forged, *runs[0][1:]), runs[1])
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=forged_runs,
        reviews=_agreeing_reviews(forged_runs[0]),
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    assert bundle.candidate.mechanically_valid_answers == 999
    assert bundle.candidate.validation_failed_answers == 1


def test_answered_output_cannot_claim_degraded_or_abstaining_retrieval(
    release_fixture,
):
    dataset, _policy = release_fixture
    output = _runs(dataset)[0][0]
    result = output.answer_result
    result["trace"]["degraded"] = True
    result["trace"]["degraded_reason"] = "reranker unavailable"
    forged = replace(
        output,
        answer_result_json=json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    with pytest.raises(ReleaseEvidenceError, match="degraded, abstaining"):
        forged.validate()


def test_fabricated_self_consistent_evidence_cannot_replace_canonical_record(
    release_fixture,
):
    dataset, policy = release_fixture
    runs = _runs(dataset, fabricate_primary_evidence=True)
    with pytest.raises(ReleaseEvidenceError, match="does not resolve"):
        aggregate_release_candidate(
            dataset,
            policy=policy,
            runs=runs,
            reviews=_agreeing_reviews(runs[0]),
            canonical_evidence=_canonical_evidence(),
            canonical_documents=CANONICAL_DOCUMENTS,
        )


def test_answer_status_cannot_diverge_from_canonical_version(release_fixture):
    dataset, policy = release_fixture
    runs = _runs(dataset)
    result = runs[0][0].answer_result
    result["evidence"]["items"][0]["status"] = "repealed"
    forged = replace(
        runs[0][0],
        answer_result_json=json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    forged_runs = ((forged, *runs[0][1:]), runs[1])
    with pytest.raises(ReleaseEvidenceError, match="active canonical record"):
        aggregate_release_candidate(
            dataset,
            policy=policy,
            runs=forged_runs,
            reviews=_agreeing_reviews(forged_runs[0]),
            canonical_evidence=_canonical_evidence(),
            canonical_documents=CANONICAL_DOCUMENTS,
        )


def test_current_answer_date_must_match_release_date_and_effective_interval(
    release_fixture,
):
    dataset, policy = release_fixture
    runs = _runs(dataset, resolved_current_date="2023-07-15")
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=runs,
        reviews=_agreeing_reviews(runs[0]),
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    assert bundle.candidate.mechanically_valid_answers == 0
    assert bundle.candidate.validation_failed_answers == 1000


def test_aggregate_counts_coverage_latency_slices_and_manifests(release_fixture):
    dataset, policy = release_fixture
    runs = _runs(dataset)
    bundle = aggregate_release_candidate(
        dataset,
        policy=policy,
        runs=runs,
        reviews=_agreeing_reviews(runs[0]),
        canonical_evidence=_canonical_evidence(),
        canonical_documents=CANONICAL_DOCUMENTS,
    )
    candidate = bundle.candidate
    assert (
        candidate.answered_outcomes,
        candidate.severe_errors,
        candidate.material_errors,
    ) == (
        1000,
        0,
        0,
    )
    assert (candidate.in_scope_answerable, candidate.answered_in_scope) == (1000, 1000)
    assert candidate.mechanically_valid_answers == 1000
    assert candidate.mechanical_checks_total == candidate.mechanical_checks_valid
    assert candidate.p95_latency_ms == 19.0
    assert candidate.degraded_answers == 0
    assert candidate.stale_current_law_answers == 0
    assert candidate.ambiguous_identity_answers == 0
    assert candidate.validation_failed_answers == 0
    assert candidate.slice_comparisons[0].candidate == 1.0
    assert len(bundle.repeat_manifests) == 2
    assert len(bundle.review_manifest_sha256) == 64
    assert len(bundle.mechanical_manifest_sha256) == 64
    assert len(bundle.canonical_evidence_manifest_sha256) == 64
    assert len(bundle.release_evidence_manifest_sha256) == 64
    assert (
        candidate.metadata["release_evidence_manifest_sha256"]
        == bundle.release_evidence_manifest_sha256
    )
    assert len(bundle.sha256) == 64

    gate = evaluate_release_candidate(bundle, policy=policy)
    assert gate.answered_clusters == 1
    assert gate.passed is False
    assert gate.severe_error_upper_95 > 0.01

    forged_bundle = replace(
        bundle,
        candidate=replace(candidate, answered_outcomes=999),
    )
    with pytest.raises(ReleaseEvidenceError, match="differs from recomputed"):
        evaluate_release_candidate(forged_bundle, policy=policy)
