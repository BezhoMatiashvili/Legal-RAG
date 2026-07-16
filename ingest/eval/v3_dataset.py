"""Strict schema and validation for the lawyer-adjudicated v3 legal eval set.

The v3 format is deliberately independent of chunk ids.  Gold evidence is stored as
equivalence groups of exact character spans into immutable, versioned canonical documents.
Each question is assigned to a split by both document identity and amendment/version lineage,
and is independently reviewed by two blinded Georgian legal reviewers.  A third, distinct
reviewer adjudicates only when the first two decisions disagree.

JSONL files written by this module contain one manifest record followed by question records.
Canonical document text is not duplicated in the eval file; callers provide the frozen
canonical records to :func:`validate_dataset`, which rechecks offsets, quotes, hashes,
version identity, completeness, authority, and effective dates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "legal-eval-v3.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DatasetFormatError(ValueError):
    """A JSONL row cannot be decoded as the v3 schema."""


class DatasetValidationError(ValueError):
    """One or more cross-record or canonical-evidence invariants failed."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("v3 dataset validation failed:\n- " + "\n- ".join(self.errors))


class Language(StrEnum):
    KA = "ka"
    EN = "en"
    UNSUPPORTED = "unsupported"


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    BLIND = "blind"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QuestionTag(StrEnum):
    PARAPHRASE = "paraphrase"
    TYPO = "typo"
    IDENTIFIER_HEAVY = "identifier_heavy"
    EXACT_AUTHORITY_LOOKUP = "exact_authority_lookup"
    CASES_APPLYING_AUTHORITY = "cases_applying_authority"
    CURRENT_LAW = "current_law"
    HISTORICAL_AS_OF = "historical_as_of"
    MULTI_EVIDENCE = "multi_evidence"
    CONFLICTING_AUTHORITIES = "conflicting_authorities"
    INCOMPLETE_SOURCE = "incomplete_source"
    AMBIGUOUS_IDENTIFIER = "ambiguous_identifier"
    UNANSWERABLE = "unanswerable"
    PROMPT_INJECTION = "prompt_injection"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    RISK_LOW = "risk_low"
    RISK_MEDIUM = "risk_medium"
    RISK_HIGH = "risk_high"


REQUIRED_V3_TAGS = frozenset(QuestionTag)
_RISK_TAG = {
    RiskLevel.LOW: QuestionTag.RISK_LOW,
    RiskLevel.MEDIUM: QuestionTag.RISK_MEDIUM,
    RiskLevel.HIGH: QuestionTag.RISK_HIGH,
}


class QuestionOrigin(StrEnum):
    REAL_ANONYMIZED_LAWYER = "real_anonymized_lawyer_question"
    EXPERT_WRITTEN = "expert_written"
    SYNTHETIC_ATTACK = "synthetic_attack"
    LEGACY = "legacy"


class ExpectedOutcome(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class ReviewerRole(StrEnum):
    REVIEWER = "reviewer"
    ADJUDICATOR = "adjudicator"


class ErrorSeverity(StrEnum):
    NONE = "none"
    NON_MATERIAL = "non_material"
    MATERIAL = "material"
    SEVERE = "severe"


class ErrorCode(StrEnum):
    # An error in any of these may materially change user action and is always severe.
    WRONG_AUTHORITY = "wrong_authority"
    WRONG_OPERATIVE_RULE = "wrong_operative_rule"
    WRONG_STATUS_OR_VERSION = "wrong_status_or_version"
    WRONG_EFFECTIVE_DATE = "wrong_effective_date"
    WRONG_OBLIGATION = "wrong_obligation"
    WRONG_RIGHT = "wrong_right"
    WRONG_EXCEPTION = "wrong_exception"
    WRONG_CASE_OUTCOME = "wrong_case_outcome"

    # Material but not automatically severe under the predeclared taxonomy.
    WRONG_PASSAGE = "wrong_passage"
    INCOMPLETE_ANSWER = "incomplete_answer"
    INVALID_CITATION = "invalid_citation"
    UNSUPPORTED_MATERIAL_CLAIM = "unsupported_material_claim"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"
    SHOULD_HAVE_CLARIFIED = "should_have_clarified"
    WRONG_INTERPRETATION = "wrong_interpretation"

    # Tracked quality defects outside the material-error numerator.
    MINOR_WORDING = "minor_wording"
    STYLE_ONLY = "style_only"


ERROR_TAXONOMY: Mapping[ErrorCode, ErrorSeverity] = {
    ErrorCode.WRONG_AUTHORITY: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_OPERATIVE_RULE: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_STATUS_OR_VERSION: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_EFFECTIVE_DATE: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_OBLIGATION: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_RIGHT: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_EXCEPTION: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_CASE_OUTCOME: ErrorSeverity.SEVERE,
    ErrorCode.WRONG_PASSAGE: ErrorSeverity.MATERIAL,
    ErrorCode.INCOMPLETE_ANSWER: ErrorSeverity.MATERIAL,
    ErrorCode.INVALID_CITATION: ErrorSeverity.MATERIAL,
    ErrorCode.UNSUPPORTED_MATERIAL_CLAIM: ErrorSeverity.MATERIAL,
    ErrorCode.SHOULD_HAVE_ABSTAINED: ErrorSeverity.MATERIAL,
    ErrorCode.SHOULD_HAVE_CLARIFIED: ErrorSeverity.MATERIAL,
    ErrorCode.WRONG_INTERPRETATION: ErrorSeverity.MATERIAL,
    ErrorCode.MINOR_WORDING: ErrorSeverity.NON_MATERIAL,
    ErrorCode.STYLE_ONLY: ErrorSeverity.NON_MATERIAL,
}
_SEVERITY_RANK = {
    ErrorSeverity.NONE: 0,
    ErrorSeverity.NON_MATERIAL: 1,
    ErrorSeverity.MATERIAL: 2,
    ErrorSeverity.SEVERE: 3,
}


def sha256_text(text: str) -> str:
    """Return the full lowercase SHA-256 of exact UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the full SHA-256 of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blind_ids_hash(question_ids: Iterable[str]) -> str:
    """Hash a sorted set of blind ids with an unambiguous JSON encoding."""

    encoded = json.dumps(
        sorted(set(question_ids)), ensure_ascii=False, separators=(",", ":")
    )
    return sha256_text(encoded)


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    where: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing {sorted(missing)}")
        if extra:
            pieces.append(f"unknown {sorted(extra)}")
        raise DatasetFormatError(f"{where}: {', '.join(pieces)}")


def _enum(enum_type: type[StrEnum], value: Any, where: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise DatasetFormatError(
            f"{where}: expected one of {allowed}; got {value!r}"
        ) from exc


def _tuple_of_strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetFormatError(f"{where}: expected a JSON array of strings")
    return tuple(value)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise DatasetFormatError(f"{where}: expected a string")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetFormatError(f"{where}: expected a boolean")
    return value


def _integer(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DatasetFormatError(f"{where}: expected an integer")
    return value


@dataclass(frozen=True)
class BlindProtocol:
    """Release gate fixed before blind evaluation begins.

    ``planned_question_count`` and ``target_answered_count`` are predeclared even while a
    set is being assembled.  Once ``sealed`` is true, the exact count and sorted id hash are
    also enforced.  The target is fixed at no fewer than 1,000 answered outcomes because the
    severe-error confidence gate is otherwise underpowered.
    """

    planned_question_count: int
    target_answered_count: int
    predeclared_at: str
    sampling_plan_sha256: str
    sealed: bool = False
    blind_ids_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned_question_count": self.planned_question_count,
            "target_answered_count": self.target_answered_count,
            "predeclared_at": self.predeclared_at,
            "sampling_plan_sha256": self.sampling_plan_sha256,
            "sealed": self.sealed,
            "blind_ids_sha256": self.blind_ids_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BlindProtocol:
        _strict_keys(
            value,
            required={
                "planned_question_count",
                "target_answered_count",
                "predeclared_at",
                "sampling_plan_sha256",
                "sealed",
                "blind_ids_sha256",
            },
            where="blind_protocol",
        )
        return cls(
            planned_question_count=_integer(
                value["planned_question_count"], "blind_protocol.planned_question_count"
            ),
            target_answered_count=_integer(
                value["target_answered_count"], "blind_protocol.target_answered_count"
            ),
            predeclared_at=_string(
                value["predeclared_at"], "blind_protocol.predeclared_at"
            ),
            sampling_plan_sha256=_string(
                value["sampling_plan_sha256"], "blind_protocol.sampling_plan_sha256"
            ),
            sealed=_bool(value["sealed"], "blind_protocol.sealed"),
            blind_ids_sha256=(
                None
                if value["blind_ids_sha256"] is None
                else _string(
                    value["blind_ids_sha256"], "blind_protocol.blind_ids_sha256"
                )
            ),
        )


@dataclass(frozen=True)
class V3Manifest:
    dataset_id: str
    corpus_generation: str
    corpus_as_of: str
    blind_protocol: BlindProtocol
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "corpus_generation": self.corpus_generation,
            "corpus_as_of": self.corpus_as_of,
            "blind_protocol": self.blind_protocol.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> V3Manifest:
        _strict_keys(
            value,
            required={
                "schema_version",
                "dataset_id",
                "corpus_generation",
                "corpus_as_of",
                "blind_protocol",
            },
            where="manifest",
        )
        protocol = value["blind_protocol"]
        if not isinstance(protocol, Mapping):
            raise DatasetFormatError("manifest.blind_protocol: expected an object")
        return cls(
            schema_version=_string(value["schema_version"], "manifest.schema_version"),
            dataset_id=_string(value["dataset_id"], "manifest.dataset_id"),
            corpus_generation=_string(
                value["corpus_generation"], "manifest.corpus_generation"
            ),
            corpus_as_of=_string(value["corpus_as_of"], "manifest.corpus_as_of"),
            blind_protocol=BlindProtocol.from_dict(protocol),
        )


@dataclass(frozen=True)
class QuestionProvenance:
    origin: QuestionOrigin
    source_record_id: str
    anonymized: bool
    pii_reviewed: bool
    original_text_retained: bool
    question_sha256: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin.value,
            "source_record_id": self.source_record_id,
            "anonymized": self.anonymized,
            "pii_reviewed": self.pii_reviewed,
            "original_text_retained": self.original_text_retained,
            "question_sha256": self.question_sha256,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuestionProvenance:
        _strict_keys(
            value,
            required={
                "origin",
                "source_record_id",
                "anonymized",
                "pii_reviewed",
                "original_text_retained",
                "question_sha256",
                "note",
            },
            where="provenance",
        )
        return cls(
            origin=_enum(QuestionOrigin, value["origin"], "provenance.origin"),
            source_record_id=_string(
                value["source_record_id"], "provenance.source_record_id"
            ),
            anonymized=_bool(value["anonymized"], "provenance.anonymized"),
            pii_reviewed=_bool(value["pii_reviewed"], "provenance.pii_reviewed"),
            original_text_retained=_bool(
                value["original_text_retained"], "provenance.original_text_retained"
            ),
            question_sha256=_string(
                value["question_sha256"], "provenance.question_sha256"
            ),
            note=_string(value["note"], "provenance.note"),
        )


@dataclass(frozen=True)
class CanonicalDocument:
    """Frozen canonical text supplied by the corpus/version resolver."""

    source: str
    document_id: str
    version_id: str
    lineage_family_id: str
    text: str
    content_sha256: str
    content_complete: bool
    authoritative: bool
    version_lineage_complete: bool
    version_ambiguous: bool
    effective_from: str | None = None
    effective_to: str | None = None  # exclusive
    status: str | None = None
    repeal_date: str | None = None
    consolidation_status: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.document_id, self.version_id)

    def effective_on(self, when: date) -> bool:
        if self.effective_from is None:
            return False
        start = date.fromisoformat(self.effective_from)
        end = date.fromisoformat(self.effective_to) if self.effective_to else None
        return start <= when and (end is None or when < end)


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    source: str
    document_id: str
    version_id: str
    lineage_family_id: str
    char_start: int
    char_end: int
    quote: str
    quote_sha256: str
    document_sha256: str
    content_complete: bool
    authoritative: bool
    status: str | None
    repeal_date: str | None
    consolidation_status: str | None
    article_id: str | None = None

    @property
    def document_key(self) -> tuple[str, str, str]:
        return (self.source, self.document_id, self.version_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "lineage_family_id": self.lineage_family_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "document_sha256": self.document_sha256,
            "content_complete": self.content_complete,
            "authoritative": self.authoritative,
            "status": self.status,
            "repeal_date": self.repeal_date,
            "consolidation_status": self.consolidation_status,
            "article_id": self.article_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceSpan:
        _strict_keys(
            value,
            required={
                "evidence_id",
                "source",
                "document_id",
                "version_id",
                "lineage_family_id",
                "char_start",
                "char_end",
                "quote",
                "quote_sha256",
                "document_sha256",
                "content_complete",
                "authoritative",
                "status",
                "repeal_date",
                "consolidation_status",
                "article_id",
            },
            where="evidence_span",
        )
        return cls(
            evidence_id=_string(value["evidence_id"], "evidence_span.evidence_id"),
            source=_string(value["source"], "evidence_span.source"),
            document_id=_string(value["document_id"], "evidence_span.document_id"),
            version_id=_string(value["version_id"], "evidence_span.version_id"),
            lineage_family_id=_string(
                value["lineage_family_id"], "evidence_span.lineage_family_id"
            ),
            char_start=_integer(value["char_start"], "evidence_span.char_start"),
            char_end=_integer(value["char_end"], "evidence_span.char_end"),
            quote=_string(value["quote"], "evidence_span.quote"),
            quote_sha256=_string(value["quote_sha256"], "evidence_span.quote_sha256"),
            document_sha256=_string(
                value["document_sha256"], "evidence_span.document_sha256"
            ),
            content_complete=_bool(
                value["content_complete"], "evidence_span.content_complete"
            ),
            authoritative=_bool(value["authoritative"], "evidence_span.authoritative"),
            status=(
                None
                if value["status"] is None
                else _string(value["status"], "evidence_span.status")
            ),
            repeal_date=(
                None
                if value["repeal_date"] is None
                else _string(value["repeal_date"], "evidence_span.repeal_date")
            ),
            consolidation_status=(
                None
                if value["consolidation_status"] is None
                else _string(
                    value["consolidation_status"],
                    "evidence_span.consolidation_status",
                )
            ),
            article_id=(
                None
                if value["article_id"] is None
                else _string(value["article_id"], "evidence_span.article_id")
            ),
        )


def anchor_span(
    document: CanonicalDocument,
    *,
    evidence_id: str,
    char_start: int,
    char_end: int,
    article_id: str | None = None,
) -> EvidenceSpan:
    """Create a correctly hashed exact span from a canonical document."""

    quote = document.text[char_start:char_end]
    return EvidenceSpan(
        evidence_id=evidence_id,
        source=document.source,
        document_id=document.document_id,
        version_id=document.version_id,
        lineage_family_id=document.lineage_family_id,
        char_start=char_start,
        char_end=char_end,
        quote=quote,
        quote_sha256=sha256_text(quote),
        document_sha256=document.content_sha256,
        content_complete=document.content_complete,
        authoritative=document.authoritative,
        status=document.status,
        repeal_date=document.repeal_date,
        consolidation_status=document.consolidation_status,
        article_id=article_id,
    )


@dataclass(frozen=True)
class EvidenceEquivalenceGroup:
    """One required proposition, satisfied by any one of its alternative exact spans."""

    group_id: str
    proposition: str
    required: bool
    alternatives: tuple[EvidenceSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "proposition": self.proposition,
            "required": self.required,
            "alternatives": [span.to_dict() for span in self.alternatives],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceEquivalenceGroup:
        _strict_keys(
            value,
            required={"group_id", "proposition", "required", "alternatives"},
            where="evidence_group",
        )
        alternatives = value["alternatives"]
        if not isinstance(alternatives, list):
            raise DatasetFormatError("evidence_group.alternatives: expected an array")
        if any(not isinstance(item, Mapping) for item in alternatives):
            raise DatasetFormatError("evidence_group.alternatives: expected objects")
        return cls(
            group_id=_string(value["group_id"], "evidence_group.group_id"),
            proposition=_string(value["proposition"], "evidence_group.proposition"),
            required=_bool(value["required"], "evidence_group.required"),
            alternatives=tuple(EvidenceSpan.from_dict(item) for item in alternatives),
        )


@dataclass(frozen=True)
class ErrorFinding:
    code: ErrorCode
    severity: ErrorSeverity
    claim_ref: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "claim_ref": self.claim_ref,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ErrorFinding:
        _strict_keys(
            value,
            required={"code", "severity", "claim_ref"},
            where="error_finding",
        )
        return cls(
            code=_enum(ErrorCode, value["code"], "error_finding.code"),
            severity=_enum(ErrorSeverity, value["severity"], "error_finding.severity"),
            claim_ref=_string(value["claim_ref"], "error_finding.claim_ref"),
        )


@dataclass(frozen=True)
class ReviewerJudgment:
    reviewer_id: str
    role: ReviewerRole
    blinded: bool
    answerable: bool
    expected_outcome: ExpectedOutcome
    evidence_group_ids: tuple[str, ...]
    overall_severity: ErrorSeverity
    errors: tuple[ErrorFinding, ...]
    rationale: str

    def decision_key(self) -> tuple[Any, ...]:
        """Fields on which two independent gold judgments can disagree."""

        error_key = tuple(
            sorted(
                (item.code.value, item.severity.value, item.claim_ref)
                for item in self.errors
            )
        )
        return (
            self.answerable,
            self.expected_outcome.value,
            tuple(sorted(set(self.evidence_group_ids))),
            self.overall_severity.value,
            error_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "role": self.role.value,
            "blinded": self.blinded,
            "answerable": self.answerable,
            "expected_outcome": self.expected_outcome.value,
            "evidence_group_ids": list(self.evidence_group_ids),
            "overall_severity": self.overall_severity.value,
            "errors": [error.to_dict() for error in self.errors],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReviewerJudgment:
        _strict_keys(
            value,
            required={
                "reviewer_id",
                "role",
                "blinded",
                "answerable",
                "expected_outcome",
                "evidence_group_ids",
                "overall_severity",
                "errors",
                "rationale",
            },
            where="reviewer_judgment",
        )
        errors = value["errors"]
        if not isinstance(errors, list) or any(
            not isinstance(item, Mapping) for item in errors
        ):
            raise DatasetFormatError(
                "reviewer_judgment.errors: expected an array of objects"
            )
        return cls(
            reviewer_id=_string(value["reviewer_id"], "reviewer_judgment.reviewer_id"),
            role=_enum(ReviewerRole, value["role"], "reviewer_judgment.role"),
            blinded=_bool(value["blinded"], "reviewer_judgment.blinded"),
            answerable=_bool(value["answerable"], "reviewer_judgment.answerable"),
            expected_outcome=_enum(
                ExpectedOutcome,
                value["expected_outcome"],
                "reviewer_judgment.expected_outcome",
            ),
            evidence_group_ids=_tuple_of_strings(
                value["evidence_group_ids"], "reviewer_judgment.evidence_group_ids"
            ),
            overall_severity=_enum(
                ErrorSeverity,
                value["overall_severity"],
                "reviewer_judgment.overall_severity",
            ),
            errors=tuple(ErrorFinding.from_dict(item) for item in errors),
            rationale=_string(value["rationale"], "reviewer_judgment.rationale"),
        )


@dataclass(frozen=True)
class V3Question:
    question_id: str
    question: str
    language: Language
    language_code: str
    split: Split
    tags: tuple[QuestionTag, ...]
    risk_level: RiskLevel
    as_of: str | None
    identifiers: tuple[str, ...]
    partition_family_ids: tuple[str, ...]
    provenance: QuestionProvenance
    evidence_groups: tuple[EvidenceEquivalenceGroup, ...]
    reference_response: str
    judgments: tuple[ReviewerJudgment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "language": self.language.value,
            "language_code": self.language_code,
            "split": self.split.value,
            "tags": [tag.value for tag in self.tags],
            "risk_level": self.risk_level.value,
            "as_of": self.as_of,
            "identifiers": list(self.identifiers),
            "partition_family_ids": list(self.partition_family_ids),
            "provenance": self.provenance.to_dict(),
            "evidence_groups": [group.to_dict() for group in self.evidence_groups],
            "reference_response": self.reference_response,
            "judgments": [judgment.to_dict() for judgment in self.judgments],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> V3Question:
        _strict_keys(
            value,
            required={
                "question_id",
                "question",
                "language",
                "language_code",
                "split",
                "tags",
                "risk_level",
                "as_of",
                "identifiers",
                "partition_family_ids",
                "provenance",
                "evidence_groups",
                "reference_response",
                "judgments",
            },
            where="question",
        )
        tags = value["tags"]
        groups = value["evidence_groups"]
        judgments = value["judgments"]
        provenance = value["provenance"]
        if not isinstance(tags, list):
            raise DatasetFormatError("question.tags: expected an array")
        if not isinstance(groups, list) or any(
            not isinstance(item, Mapping) for item in groups
        ):
            raise DatasetFormatError(
                "question.evidence_groups: expected an array of objects"
            )
        if not isinstance(judgments, list) or any(
            not isinstance(item, Mapping) for item in judgments
        ):
            raise DatasetFormatError("question.judgments: expected an array of objects")
        if not isinstance(provenance, Mapping):
            raise DatasetFormatError("question.provenance: expected an object")
        return cls(
            question_id=_string(value["question_id"], "question.question_id"),
            question=_string(value["question"], "question.question"),
            language=_enum(Language, value["language"], "question.language"),
            language_code=_string(value["language_code"], "question.language_code"),
            split=_enum(Split, value["split"], "question.split"),
            tags=tuple(_enum(QuestionTag, item, "question.tags") for item in tags),
            risk_level=_enum(RiskLevel, value["risk_level"], "question.risk_level"),
            as_of=(
                None
                if value["as_of"] is None
                else _string(value["as_of"], "question.as_of")
            ),
            identifiers=_tuple_of_strings(value["identifiers"], "question.identifiers"),
            partition_family_ids=_tuple_of_strings(
                value["partition_family_ids"], "question.partition_family_ids"
            ),
            provenance=QuestionProvenance.from_dict(provenance),
            evidence_groups=tuple(
                EvidenceEquivalenceGroup.from_dict(item) for item in groups
            ),
            reference_response=_string(
                value["reference_response"], "question.reference_response"
            ),
            judgments=tuple(ReviewerJudgment.from_dict(item) for item in judgments),
        )


@dataclass(frozen=True)
class V3Dataset:
    manifest: V3Manifest
    questions: tuple[V3Question, ...]


def _parse_date(value: str, where: str, errors: list[str]) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        errors.append(f"{where}: expected an ISO date, got {value!r}")
        return None


def _valid_sha256(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _validate_manifest(manifest: V3Manifest, errors: list[str]) -> date | None:
    if manifest.schema_version != SCHEMA_VERSION:
        errors.append(
            f"manifest.schema_version: expected {SCHEMA_VERSION!r}, "
            f"got {manifest.schema_version!r}"
        )
    if not manifest.dataset_id.strip():
        errors.append("manifest.dataset_id: must be non-empty")
    if not manifest.corpus_generation.strip():
        errors.append("manifest.corpus_generation: must be non-empty")
    corpus_as_of = _parse_date(manifest.corpus_as_of, "manifest.corpus_as_of", errors)

    protocol = manifest.blind_protocol
    if protocol.planned_question_count < 1000:
        errors.append("blind protocol: planned_question_count must be at least 1000")
    if protocol.target_answered_count < 1000:
        errors.append("blind protocol: target_answered_count must be at least 1000")
    if protocol.target_answered_count > protocol.planned_question_count:
        errors.append(
            "blind protocol: target_answered_count cannot exceed planned count"
        )
    if not _valid_sha256(protocol.sampling_plan_sha256):
        errors.append(
            "blind protocol: sampling_plan_sha256 must be a full lowercase SHA-256"
        )
    try:
        declared = datetime.fromisoformat(
            protocol.predeclared_at.replace("Z", "+00:00")
        )
        if declared.tzinfo is None:
            errors.append("blind protocol: predeclared_at must include a timezone")
    except (TypeError, ValueError):
        errors.append("blind protocol: predeclared_at must be an ISO-8601 timestamp")
    if protocol.sealed and not protocol.blind_ids_sha256:
        errors.append("blind protocol: sealed sets require blind_ids_sha256")
    if protocol.blind_ids_sha256 is not None and not _valid_sha256(
        protocol.blind_ids_sha256
    ):
        errors.append(
            "blind protocol: blind_ids_sha256 must be a full lowercase SHA-256"
        )
    return corpus_as_of


def _document_index(
    documents: Iterable[CanonicalDocument]
    | Mapping[tuple[str, str, str], CanonicalDocument],
    errors: list[str],
) -> dict[tuple[str, str, str], CanonicalDocument]:
    values = documents.values() if isinstance(documents, Mapping) else documents
    index: dict[tuple[str, str, str], CanonicalDocument] = {}
    for document in values:
        if document.key in index:
            errors.append(f"canonical documents: duplicate key {document.key!r}")
            continue
        index[document.key] = document
        if any(not part.strip() for part in document.key):
            errors.append(
                f"canonical document {document.key!r}: identity fields must be non-empty"
            )
        if not document.lineage_family_id.strip():
            errors.append(
                f"canonical document {document.key!r}: missing lineage_family_id"
            )
        if not isinstance(document.version_lineage_complete, bool):
            errors.append(
                f"canonical document {document.key!r}: version_lineage_complete must be boolean"
            )
        if not isinstance(document.version_ambiguous, bool):
            errors.append(
                f"canonical document {document.key!r}: version_ambiguous must be boolean"
            )
        if document.source == "matsne":
            if not isinstance(document.status, str) or not document.status.strip():
                errors.append(
                    f"canonical document {document.key!r}: normative status is required"
                )
            if (
                not isinstance(document.consolidation_status, str)
                or not document.consolidation_status.strip()
            ):
                errors.append(
                    f"canonical document {document.key!r}: normative consolidation_status "
                    "is required"
                )
        actual_hash = sha256_text(document.text)
        if document.content_sha256 != actual_hash:
            errors.append(
                f"canonical document {document.key!r}: content hash mismatch "
                f"({document.content_sha256} != {actual_hash})"
            )
        if document.effective_from is not None:
            start = _parse_date(
                document.effective_from,
                f"canonical document {document.key!r}.effective_from",
                errors,
            )
            end = None
            if document.effective_to is not None:
                end = _parse_date(
                    document.effective_to,
                    f"canonical document {document.key!r}.effective_to",
                    errors,
                )
            if start is not None and end is not None and end <= start:
                errors.append(
                    f"canonical document {document.key!r}: effective_to must be after "
                    "effective_from"
                )
        elif document.effective_to is not None:
            errors.append(
                f"canonical document {document.key!r}: effective_to requires effective_from"
            )
        if document.repeal_date is not None:
            repeal = _parse_date(
                document.repeal_date,
                f"canonical document {document.key!r}.repeal_date",
                errors,
            )
            effective_start = None
            if document.effective_from is not None:
                try:
                    effective_start = date.fromisoformat(document.effective_from)
                except ValueError:
                    pass
            if (
                repeal is not None
                and effective_start is not None
                and repeal <= effective_start
            ):
                errors.append(
                    f"canonical document {document.key!r}: repeal_date must be after "
                    "effective_from"
                )
    return index


def _current_normative_operative(document: CanonicalDocument, when: date | None) -> bool:
    if document.source != "matsne":
        return True
    if when is None or document.status != "in_force" or not document.consolidation_status:
        return False
    if document.repeal_date is None:
        return True
    try:
        return when < date.fromisoformat(document.repeal_date)
    except ValueError:
        return False


def _max_severity(findings: Sequence[ErrorFinding]) -> ErrorSeverity:
    if not findings:
        return ErrorSeverity.NONE
    return max(
        (finding.severity for finding in findings), key=_SEVERITY_RANK.__getitem__
    )


def adjudicated_judgment(question: V3Question) -> ReviewerJudgment:
    """Return the consensus primary judgment or the disagreement adjudication.

    Call :func:`validate_dataset` first.  This helper raises ``ValueError`` when the review
    panel shape is incomplete instead of guessing a decision.
    """

    reviewers = [
        item for item in question.judgments if item.role is ReviewerRole.REVIEWER
    ]
    adjudicators = [
        item for item in question.judgments if item.role is ReviewerRole.ADJUDICATOR
    ]
    if len(reviewers) != 2:
        raise ValueError(f"{question.question_id}: expected exactly two reviewers")
    if reviewers[0].decision_key() == reviewers[1].decision_key():
        if adjudicators:
            raise ValueError(
                f"{question.question_id}: agreement must not be adjudicated"
            )
        return reviewers[0]
    if len(adjudicators) != 1:
        raise ValueError(
            f"{question.question_id}: disagreement requires one adjudicator"
        )
    return adjudicators[0]


def _validate_judgments(
    question: V3Question,
    groups: Mapping[str, EvidenceEquivalenceGroup],
    errors: list[str],
) -> ReviewerJudgment | None:
    where = question.question_id
    reviewers = [
        item for item in question.judgments if item.role is ReviewerRole.REVIEWER
    ]
    adjudicators = [
        item for item in question.judgments if item.role is ReviewerRole.ADJUDICATOR
    ]
    if len(reviewers) != 2:
        errors.append(f"{where}: requires exactly two independent primary reviewers")
    reviewer_ids = [item.reviewer_id for item in question.judgments]
    if any(not reviewer_id.strip() for reviewer_id in reviewer_ids):
        errors.append(f"{where}: reviewer ids must be non-empty")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        errors.append(f"{where}: reviewer and adjudicator ids must be distinct")
    if any(not item.blinded for item in question.judgments):
        errors.append(
            f"{where}: every legal reviewer must be blinded to system identity"
        )

    for judgment in question.judgments:
        prefix = f"{where}/{judgment.reviewer_id or '<missing-reviewer>'}"
        if not judgment.rationale.strip():
            errors.append(f"{prefix}: rationale must be non-empty")
        if len(judgment.evidence_group_ids) != len(set(judgment.evidence_group_ids)):
            errors.append(f"{prefix}: duplicate evidence group ids")
        missing_groups = set(judgment.evidence_group_ids) - set(groups)
        if missing_groups:
            errors.append(f"{prefix}: unknown evidence groups {sorted(missing_groups)}")
        if (
            judgment.expected_outcome is ExpectedOutcome.ANSWER
            and not judgment.answerable
        ):
            errors.append(f"{prefix}: answer outcome requires answerable=true")
        if (
            judgment.expected_outcome is not ExpectedOutcome.ANSWER
            and judgment.answerable
        ):
            errors.append(f"{prefix}: non-answer outcome requires answerable=false")
        required_groups = {
            group_id for group_id, group in groups.items() if group.required
        }
        if (
            judgment.expected_outcome is ExpectedOutcome.ANSWER
            and set(judgment.evidence_group_ids) != required_groups
        ):
            errors.append(
                f"{prefix}: answer judgment must select every and only required evidence group"
            )
        for finding in judgment.errors:
            expected = ERROR_TAXONOMY[finding.code]
            if finding.severity is not expected:
                errors.append(
                    f"{prefix}: {finding.code.value} must be {expected.value}, "
                    f"not {finding.severity.value}"
                )
        maximum = _max_severity(judgment.errors)
        if judgment.overall_severity is not maximum:
            errors.append(
                f"{prefix}: overall_severity must equal maximum finding severity "
                f"{maximum.value}"
            )

    final: ReviewerJudgment | None = None
    if len(reviewers) == 2:
        disagree = reviewers[0].decision_key() != reviewers[1].decision_key()
        if disagree and len(adjudicators) != 1:
            errors.append(
                f"{where}: reviewer disagreement requires exactly one adjudicator"
            )
        if not disagree and adjudicators:
            errors.append(
                f"{where}: adjudicator is only permitted on reviewer disagreement"
            )
        if disagree and len(adjudicators) == 1:
            final = adjudicators[0]
        elif not disagree:
            final = reviewers[0]
    elif adjudicators:
        errors.append(f"{where}: adjudication cannot replace missing primary reviews")
    return final


def _validate_span(
    question_id: str,
    span: EvidenceSpan,
    documents: Mapping[tuple[str, str, str], CanonicalDocument],
    errors: list[str],
) -> CanonicalDocument | None:
    prefix = f"{question_id}/{span.evidence_id or '<missing-evidence-id>'}"
    if any(
        not value.strip()
        for value in (
            span.evidence_id,
            span.source,
            span.document_id,
            span.version_id,
            span.lineage_family_id,
        )
    ):
        errors.append(f"{prefix}: identity fields must be non-empty")
    if span.char_start < 0 or span.char_end <= span.char_start:
        errors.append(f"{prefix}: invalid half-open character offsets")
    if not span.quote:
        errors.append(f"{prefix}: quote must be non-empty")
    if span.quote_sha256 != sha256_text(span.quote):
        errors.append(f"{prefix}: quote hash mismatch")
    if not _valid_sha256(span.document_sha256):
        errors.append(f"{prefix}: document_sha256 must be a full lowercase SHA-256")

    document = documents.get(span.document_key)
    if document is None:
        errors.append(
            f"{prefix}: canonical version {span.document_key!r} does not resolve"
        )
        return None
    if span.lineage_family_id != document.lineage_family_id:
        errors.append(f"{prefix}: lineage family does not match canonical version")
    if span.document_sha256 != document.content_sha256:
        errors.append(f"{prefix}: document hash does not match canonical version")
    if span.content_complete is not document.content_complete:
        errors.append(f"{prefix}: completeness flag does not match canonical version")
    if span.authoritative is not document.authoritative:
        errors.append(f"{prefix}: authority flag does not match canonical version")
    if span.status != document.status:
        errors.append(f"{prefix}: legal status does not match canonical version")
    if span.repeal_date != document.repeal_date:
        errors.append(f"{prefix}: repeal date does not match canonical version")
    if span.consolidation_status != document.consolidation_status:
        errors.append(f"{prefix}: consolidation status does not match canonical version")
    if span.char_end > len(document.text):
        errors.append(f"{prefix}: offsets exceed canonical text length")
    elif document.text[span.char_start : span.char_end] != span.quote:
        errors.append(f"{prefix}: quote does not exactly match canonical offsets")
    return document


def _validate_question(
    question: V3Question,
    manifest_as_of: date | None,
    documents: Mapping[tuple[str, str, str], CanonicalDocument],
    errors: list[str],
) -> tuple[set[tuple[str, str]], set[str]]:
    where = question.question_id or "<missing-question-id>"
    if not question.question_id.strip() or not question.question.strip():
        errors.append(f"{where}: question_id and question must be non-empty")
    if len(question.tags) != len(set(question.tags)):
        errors.append(f"{where}: duplicate question tags")
    tags = set(question.tags)
    risk_tags = tags & set(_RISK_TAG.values())
    if risk_tags != {_RISK_TAG[question.risk_level]}:
        errors.append(f"{where}: exactly one risk tag must match risk_level")

    if question.language in (Language.KA, Language.EN):
        if question.language_code != question.language.value:
            errors.append(f"{where}: supported language_code must equal language")
        if QuestionTag.UNSUPPORTED_LANGUAGE in tags:
            errors.append(
                f"{where}: supported question cannot carry unsupported_language tag"
            )
    else:
        if question.language_code in {Language.KA.value, Language.EN.value, ""}:
            errors.append(
                f"{where}: unsupported input requires its actual non-ka/en language_code"
            )
        if QuestionTag.UNSUPPORTED_LANGUAGE not in tags:
            errors.append(
                f"{where}: unsupported input requires unsupported_language tag"
            )

    if (
        QuestionTag.EXACT_AUTHORITY_LOOKUP in tags
        and QuestionTag.CASES_APPLYING_AUTHORITY in tags
    ):
        errors.append(
            f"{where}: exact lookup and cases-applying tags are mutually exclusive"
        )
    if QuestionTag.CURRENT_LAW in tags and QuestionTag.HISTORICAL_AS_OF in tags:
        errors.append(f"{where}: current and historical tags are mutually exclusive")
    query_as_of = None
    if QuestionTag.HISTORICAL_AS_OF in tags:
        if question.as_of is None:
            errors.append(f"{where}: historical_as_of requires an as_of date")
        else:
            query_as_of = _parse_date(question.as_of, f"{where}.as_of", errors)
    elif question.as_of is not None:
        errors.append(f"{where}: as_of is only valid for historical_as_of questions")
    elif QuestionTag.CURRENT_LAW in tags:
        query_as_of = manifest_as_of
    if QuestionTag.IDENTIFIER_HEAVY in tags and not question.identifiers:
        errors.append(f"{where}: identifier_heavy requires extracted identifiers")
    if any(not identifier.strip() for identifier in question.identifiers):
        errors.append(f"{where}: identifiers must be non-empty strings")
    if len(question.partition_family_ids) != len(set(question.partition_family_ids)):
        errors.append(f"{where}: duplicate partition family ids")
    if any(not family.strip() for family in question.partition_family_ids):
        errors.append(f"{where}: partition family ids must be non-empty")

    provenance = question.provenance
    if provenance.question_sha256 != sha256_text(question.question):
        errors.append(
            f"{where}: provenance question hash does not match anonymized question"
        )
    if not provenance.source_record_id.strip():
        errors.append(
            f"{where}: provenance source_record_id must be a non-empty opaque id"
        )
    if not _valid_sha256(provenance.question_sha256):
        errors.append(f"{where}: provenance question_sha256 must be a full SHA-256")
    if provenance.origin is QuestionOrigin.REAL_ANONYMIZED_LAWYER:
        if not provenance.anonymized or not provenance.pii_reviewed:
            errors.append(
                f"{where}: real lawyer traffic must be anonymized and PII-reviewed"
            )
        if provenance.original_text_retained:
            errors.append(
                f"{where}: raw lawyer question text must not be retained in v3"
            )

    group_map: dict[str, EvidenceEquivalenceGroup] = {}
    evidence_ids: set[str] = set()
    evidence_documents: set[tuple[str, str]] = set()
    evidence_families: set[str] = set()
    resolved: dict[str, list[CanonicalDocument]] = {}
    for group in question.evidence_groups:
        prefix = f"{where}/{group.group_id or '<missing-group-id>'}"
        if not group.group_id.strip() or not group.proposition.strip():
            errors.append(f"{prefix}: group id and proposition must be non-empty")
        if group.group_id in group_map:
            errors.append(f"{where}: duplicate evidence group id {group.group_id!r}")
        group_map[group.group_id] = group
        if not group.alternatives:
            errors.append(f"{prefix}: equivalence group must contain at least one span")
        resolved[group.group_id] = []
        for span in group.alternatives:
            if span.evidence_id in evidence_ids:
                errors.append(f"{where}: duplicate evidence id {span.evidence_id!r}")
            evidence_ids.add(span.evidence_id)
            evidence_documents.add((span.source, span.document_id))
            evidence_families.add(span.lineage_family_id)
            document = _validate_span(where, span, documents, errors)
            if document is not None:
                resolved[group.group_id].append(document)

    missing_partition_families = evidence_families - set(question.partition_family_ids)
    if missing_partition_families:
        errors.append(
            f"{where}: evidence lineages missing from partition_family_ids "
            f"{sorted(missing_partition_families)}"
        )
    required_groups = [group for group in question.evidence_groups if group.required]
    if QuestionTag.MULTI_EVIDENCE in tags and len(required_groups) < 2:
        errors.append(f"{where}: multi_evidence requires at least two required groups")
    if QuestionTag.CONFLICTING_AUTHORITIES in tags:
        required_docs = {
            (span.source, span.document_id)
            for group in required_groups
            for span in group.alternatives
        }
        if len(required_groups) < 2 or len(required_docs) < 2:
            errors.append(
                f"{where}: conflicting_authorities requires required evidence from "
                "at least two documents"
            )
    if QuestionTag.INCOMPLETE_SOURCE in tags and not any(
        not span.content_complete
        for group in question.evidence_groups
        for span in group.alternatives
    ):
        errors.append(
            f"{where}: incomplete_source requires an incomplete canonical span"
        )
    if not question.reference_response.strip():
        errors.append(f"{where}: reference response must be non-empty")

    final = _validate_judgments(question, group_map, errors)
    if final is not None:
        if final.expected_outcome is ExpectedOutcome.ANSWER:
            if not required_groups:
                errors.append(
                    f"{where}: answer outcome requires at least one evidence group"
                )
            for group in required_groups:
                usable = []
                for document in resolved.get(group.group_id, []):
                    temporal_ok = query_as_of is None
                    if query_as_of is not None:
                        try:
                            temporal_ok = document.effective_on(query_as_of)
                        except ValueError:
                            temporal_ok = False
                    if (
                        document.content_complete
                        and document.authoritative
                        and document.version_lineage_complete
                        and not document.version_ambiguous
                        and temporal_ok
                        and (
                            QuestionTag.CURRENT_LAW not in tags
                            or _current_normative_operative(document, query_as_of)
                        )
                    ):
                        usable.append(document)
                if not usable:
                    errors.append(
                        f"{where}/{group.group_id}: answer has no complete, authoritative, "
                        "unambiguous lineage-proven, temporally valid evidence alternative"
                    )
        if question.language is Language.UNSUPPORTED and final.expected_outcome is not (
            ExpectedOutcome.CLARIFY
        ):
            errors.append(f"{where}: unsupported language must produce clarification")
        if QuestionTag.AMBIGUOUS_IDENTIFIER in tags and final.expected_outcome is (
            ExpectedOutcome.ANSWER
        ):
            errors.append(
                f"{where}: unresolved ambiguous identifier cannot be answered"
            )
        if QuestionTag.INCOMPLETE_SOURCE in tags and final.expected_outcome is (
            ExpectedOutcome.ANSWER
        ):
            errors.append(f"{where}: incomplete source cannot produce an answer")
        if QuestionTag.UNANSWERABLE in tags and final.expected_outcome is not (
            ExpectedOutcome.ABSTAIN
        ):
            errors.append(f"{where}: unanswerable question must abstain")

    return evidence_documents, set(question.partition_family_ids)


def validate_dataset(
    dataset: V3Dataset,
    canonical_documents: Iterable[CanonicalDocument]
    | Mapping[tuple[str, str, str], CanonicalDocument],
    *,
    require_coverage: bool = True,
) -> None:
    """Validate the complete v3 dataset or raise one aggregated validation error.

    ``require_coverage=False`` is useful while annotators are constructing a batch; all
    record-level, evidence, adjudication, and leakage checks remain active.  Release
    validation should use the default, which additionally requires every declared language,
    split, risk tier, adversarial/query tag, and at least one real anonymized lawyer question.
    """

    errors: list[str] = []
    manifest_as_of = _validate_manifest(dataset.manifest, errors)
    documents = _document_index(canonical_documents, errors)
    question_ids: set[str] = set()
    document_splits: dict[tuple[str, str], Split] = {}
    family_splits: dict[str, Split] = {}
    evidence_by_id: dict[str, EvidenceSpan] = {}

    for question in dataset.questions:
        if question.question_id in question_ids:
            errors.append(f"duplicate question id {question.question_id!r}")
        question_ids.add(question.question_id)
        evidence_documents, partition_families = _validate_question(
            question, manifest_as_of, documents, errors
        )
        for group in question.evidence_groups:
            for span in group.alternatives:
                prior_span = evidence_by_id.setdefault(span.evidence_id, span)
                if prior_span != span:
                    errors.append(
                        f"evidence id collision: {span.evidence_id!r} resolves to "
                        "different canonical spans"
                    )
        for document_id in evidence_documents:
            prior = document_splits.setdefault(document_id, question.split)
            if prior is not question.split:
                errors.append(
                    f"split leakage: document {document_id!r} occurs in "
                    f"{prior.value} and {question.split.value}"
                )
        for family_id in partition_families:
            prior = family_splits.setdefault(family_id, question.split)
            if prior is not question.split:
                errors.append(
                    f"split leakage: lineage family {family_id!r} occurs in "
                    f"{prior.value} and {question.split.value}"
                )

    blind_ids = [
        question.question_id
        for question in dataset.questions
        if question.split is Split.BLIND
    ]
    protocol = dataset.manifest.blind_protocol
    if len(blind_ids) > protocol.planned_question_count:
        errors.append(
            "blind protocol: collected blind questions exceed predeclared size"
        )
    if protocol.sealed:
        if len(blind_ids) != protocol.planned_question_count:
            errors.append(
                "blind protocol: sealed blind question count does not equal predeclared size"
            )
        actual_blind_hash = blind_ids_hash(blind_ids)
        if protocol.blind_ids_sha256 != actual_blind_hash:
            errors.append("blind protocol: sealed blind id hash mismatch")
        adjudicated_answerable = 0
        for question in dataset.questions:
            if question.split is not Split.BLIND:
                continue
            try:
                judgment = adjudicated_judgment(question)
            except ValueError:
                continue
            adjudicated_answerable += int(judgment.answerable)
        if protocol.target_answered_count > adjudicated_answerable:
            errors.append(
                "blind protocol: target_answered_count exceeds adjudicated in-scope "
                f"answerable questions ({adjudicated_answerable})"
            )

    if require_coverage:
        languages = {question.language for question in dataset.questions}
        missing_languages = set(Language) - languages
        if missing_languages:
            errors.append(
                "coverage: missing languages "
                + repr(sorted(item.value for item in missing_languages))
            )
        splits = {question.split for question in dataset.questions}
        missing_splits = set(Split) - splits
        if missing_splits:
            errors.append(
                "coverage: missing splits "
                + repr(sorted(item.value for item in missing_splits))
            )
        tags = {tag for question in dataset.questions for tag in question.tags}
        missing_tags = REQUIRED_V3_TAGS - tags
        if missing_tags:
            errors.append(
                "coverage: missing query/risk tags "
                + repr(sorted(item.value for item in missing_tags))
            )
        if not any(
            question.provenance.origin is QuestionOrigin.REAL_ANONYMIZED_LAWYER
            for question in dataset.questions
        ):
            errors.append(
                "coverage: requires at least one real anonymized lawyer question"
            )

    if errors:
        raise DatasetValidationError(errors)


def render_dataset_jsonl(dataset: V3Dataset) -> str:
    """Return the canonical JSONL representation used for reproducibility hashes."""

    rows: list[dict[str, Any]] = [
        {"record_type": "manifest", "manifest": dataset.manifest.to_dict()}
    ]
    rows.extend(
        {"record_type": "question", "question": question.to_dict()}
        for question in dataset.questions
    )
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def dataset_hash(dataset: V3Dataset) -> str:
    """Hash canonical JSONL, independent of an input file's whitespace or key order."""

    return sha256_text(render_dataset_jsonl(dataset))


def write_dataset_jsonl(path: str | Path, dataset: V3Dataset) -> str:
    """Write canonical UTF-8 JSONL and return its reproducibility hash."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_dataset_jsonl(dataset), encoding="utf-8")
    return dataset_hash(dataset)


def load_dataset_jsonl(path: str | Path) -> V3Dataset:
    """Strictly load one manifest followed by zero or more v3 question records."""

    manifest: V3Manifest | None = None
    questions: list[V3Question] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise DatasetFormatError(f"line {line_number}: expected a JSON object")
            record_type = row.get("record_type")
            if record_type == "manifest":
                _strict_keys(
                    row,
                    required={"record_type", "manifest"},
                    where=f"line {line_number}",
                )
                if manifest is not None or questions:
                    raise DatasetFormatError(
                        f"line {line_number}: manifest must be the first and only manifest"
                    )
                value = row["manifest"]
                if not isinstance(value, Mapping):
                    raise DatasetFormatError(
                        f"line {line_number}: manifest must be an object"
                    )
                manifest = V3Manifest.from_dict(value)
            elif record_type == "question":
                _strict_keys(
                    row,
                    required={"record_type", "question"},
                    where=f"line {line_number}",
                )
                if manifest is None:
                    raise DatasetFormatError(
                        f"line {line_number}: question precedes manifest"
                    )
                value = row["question"]
                if not isinstance(value, Mapping):
                    raise DatasetFormatError(
                        f"line {line_number}: question must be an object"
                    )
                questions.append(V3Question.from_dict(value))
            else:
                raise DatasetFormatError(
                    f"line {line_number}: unknown record_type {record_type!r}"
                )
    if manifest is None:
        raise DatasetFormatError("dataset contains no manifest")
    return V3Dataset(manifest=manifest, questions=tuple(questions))
