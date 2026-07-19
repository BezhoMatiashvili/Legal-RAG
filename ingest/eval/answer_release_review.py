"""Immutable blind-review artifacts and release-candidate aggregation.

The release gate intentionally operates on small aggregate counts.  Those counts must not
be typed into a JSON file by hand.  This module is the evidence-producing layer: it binds
two complete deterministic runs to the sealed v3 blind set, verifies the public
``AnswerResult`` evidence contract, requires two configuration-blind legal reviews (and a
third adjudicator on disagreement), and derives a :class:`ReleaseCandidate` from those
rows.

No model judgment occurs here.  Human error findings use the taxonomy frozen in
``v3_dataset``; evidence IDs, quotations, hashes, outcomes, coverage, operational failures,
latency, repeat hashes, and slice scores are calculated deterministically.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from eval.release_gate import (
    ReleaseCandidate,
    ReleasePolicy,
    ReleaseQuestionOutcome,
    SliceComparison,
)
from eval.v3_dataset import (
    CanonicalDocument,
    ERROR_TAXONOMY,
    ErrorFinding,
    ErrorSeverity,
    QuestionTag,
    ReviewerRole,
    Split,
    V3Dataset,
    V3Question,
    adjudicated_judgment,
    blind_ids_hash,
    dataset_hash,
    validate_dataset,
)
from ingest.legal_answer import EvidenceContractError, parse_evidence_id
from ingest.query_planner import plan_query

SCHEMA_VERSION = "legal-answer-release-review/v1"
_SHA256_LENGTH = 64
_SEVERITY_RANK = {
    ErrorSeverity.NONE: 0,
    ErrorSeverity.NON_MATERIAL: 1,
    ErrorSeverity.MATERIAL: 2,
    ErrorSeverity.SEVERE: 3,
}
_CANONICAL_EVIDENCE_FIELDS = (
    "evidence_id",
    "point_id",
    "schema_version",
    "canonical_payload_revision",
    "generation_id",
    "source",
    "source_authority",
    "source_fingerprint",
    "normalizer_revision",
    "chunker_revision",
    "model_revision",
    "document_id",
    "document_title",
    "document_number",
    "registration_code",
    "document_type",
    "version_id",
    "article_id",
    "clause_id",
    "subarticle_id",
    "chapter",
    "heading_path",
    "parent_id",
    "court",
    "case_number",
    "status",
    "supersedes",
    "effective_from",
    "effective_to",
    "repeal_date",
    "consolidation_status",
    "version_lineage_status",
    "version_lineage_complete",
    "content_complete",
    "extraction_status",
    "official_url",
    "official_binary_url",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "offset_unit",
    "content_hash",
    "passage_hash",
    "passage_id",
    "text",
    "token_count",
    "identity_ambiguous",
)


class ReleaseEvidenceError(ValueError):
    """The candidate-output/review artifacts are incomplete or inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object(value: str, *, where: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceError(f"{where}: invalid canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise ReleaseEvidenceError(f"{where}: expected a JSON object")
    if value != _canonical_json(decoded):
        raise ReleaseEvidenceError(f"{where}: JSON is not in canonical form")
    return decoded


@dataclass(frozen=True)
class RankedHit:
    """One exact ranked candidate, including its score and contributing routes."""

    hit_id: str
    score: float
    routes: tuple[str, ...]

    def validate(self) -> None:
        if (
            not isinstance(self.hit_id, str)
            or not self.hit_id
            or isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ReleaseEvidenceError("ranked hit requires an ID and finite score")
        if not self.routes or any(
            not isinstance(route, str) or not route for route in self.routes
        ):
            raise ReleaseEvidenceError("ranked hit must retain at least one route")
        if len(self.routes) != len(set(self.routes)):
            raise ReleaseEvidenceError("ranked hit routes must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"hit_id": self.hit_id, "score": self.score, "routes": list(self.routes)}


@dataclass(frozen=True)
class CandidateOutput:
    """One immutable system output for one question in one repeated run.

    ``answer_result_json`` is canonical JSON rather than a mutable ``dict``.  The row hash
    therefore remains stable after construction and is safe for reviewers to sign by hash.
    """

    dataset_id: str
    dataset_sha256: str
    blind_question_set_sha256: str
    candidate_id: str
    generation_id: str
    configuration_sha256: str
    run_id: str
    repeat_index: int
    question_id: str
    question_sha256: str
    ranking: tuple[RankedHit, ...]
    answer_result_json: str
    latency_ms: float

    @classmethod
    def from_result(
        cls,
        *,
        answer_result: Mapping[str, Any],
        ranking: Sequence[RankedHit],
        **identity: Any,
    ) -> CandidateOutput:
        return cls(
            ranking=tuple(ranking),
            answer_result_json=_canonical_json(dict(answer_result)),
            **identity,
        )

    @property
    def answer_result(self) -> dict[str, Any]:
        return _json_object(
            self.answer_result_json,
            where=f"candidate output {self.run_id}/{self.question_id}",
        )

    def validate(self) -> None:
        required = {
            "dataset_id": self.dataset_id,
            "candidate_id": self.candidate_id,
            "generation_id": self.generation_id,
            "run_id": self.run_id,
            "question_id": self.question_id,
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise ReleaseEvidenceError("candidate output identity fields are required")
        for name, value in (
            ("dataset_sha256", self.dataset_sha256),
            ("blind_question_set_sha256", self.blind_question_set_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("question_sha256", self.question_sha256),
        ):
            if not _is_sha256(value):
                raise ReleaseEvidenceError(f"candidate output {name} is not SHA-256")
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 0
        ):
            raise ReleaseEvidenceError("candidate repeat_index must be non-negative")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ReleaseEvidenceError(
                "candidate latency must be finite and non-negative"
            )
        if not isinstance(self.ranking, tuple) or any(
            not isinstance(hit, RankedHit) for hit in self.ranking
        ):
            raise ReleaseEvidenceError(
                "candidate ranking must contain RankedHit values"
            )
        for hit in self.ranking:
            hit.validate()
        hit_ids = [hit.hit_id for hit in self.ranking]
        if len(hit_ids) != len(set(hit_ids)):
            raise ReleaseEvidenceError("candidate ranking contains duplicate hit IDs")
        result = self.answer_result
        if result.get("outcome") not in {"answer", "clarify", "abstain"}:
            raise ReleaseEvidenceError("candidate output has an invalid outcome")
        trace = result.get("trace")
        if not isinstance(trace, dict) or not _is_sha256(trace.get("trace_id")):
            raise ReleaseEvidenceError("candidate output must retain a full trace ID")
        if result.get("trace_id") != trace["trace_id"]:
            raise ReleaseEvidenceError("top-level and nested trace IDs differ")
        versions = result.get("versions")
        if not isinstance(versions, dict):
            raise ReleaseEvidenceError("candidate output lacks pipeline versions")
        required_versions = {
            "corpus_generation",
            "retriever",
            "reranker",
            "translator",
            "generator",
            "prompt",
            "calibrator",
        }
        if not required_versions.issubset(versions):
            raise ReleaseEvidenceError(
                "candidate output has incomplete pipeline versions"
            )
        if versions.get("corpus_generation") != self.generation_id:
            raise ReleaseEvidenceError(
                "answer corpus generation differs from run identity"
            )
        if result.get("outcome") == "answer" and any(
            not isinstance(versions.get(name), str) or not versions[name]
            for name in ("retriever", "reranker", "generator", "prompt", "calibrator")
        ):
            raise ReleaseEvidenceError(
                "answered output has an unpinned pipeline component"
            )
        translator = versions.get("translator")
        if translator is not None and (
            not isinstance(translator, str) or not translator
        ):
            raise ReleaseEvidenceError("candidate translator version is invalid")
        if result.get("outcome") == "answer" and (
            trace.get("service_abstention") is not False
            or trace.get("abstention_reason") is not None
            or trace.get("degraded") is not False
            or trace.get("degraded_reason") is not None
            or trace.get("identity_ambiguous") is not False
            or trace.get("translator_version") != translator
        ):
            raise ReleaseEvidenceError(
                "answered output has degraded, abstaining, ambiguous, or mismatched "
                "retrieval state"
            )
        ranked_trace = trace.get("ranked_hits")
        if not isinstance(ranked_trace, list):
            raise ReleaseEvidenceError(
                "candidate output lacks the exact ranked-hit trace"
            )
        expected_ranking = [
            {
                "point_id": hit.hit_id,
                "score": format(float(hit.score), ".17g"),
            }
            for hit in self.ranking
        ]
        observed_ranking = [
            {
                "point_id": item.get("point_id"),
                "score": item.get("score"),
            }
            for item in ranked_trace
            if isinstance(item, dict)
        ]
        if (
            len(observed_ranking) != len(ranked_trace)
            or observed_ranking != expected_ranking
        ):
            raise ReleaseEvidenceError(
                "candidate ranking differs from the AnswerResult ranked-hit trace"
            )
        branches = trace.get("retrieval_branches")
        if not isinstance(branches, list) or any(
            not isinstance(branch, dict)
            or not isinstance(branch.get("name"), str)
            or not isinstance(branch.get("route"), str)
            or not isinstance(branch.get("hit_ids"), list)
            or any(not isinstance(hit_id, str) for hit_id in branch["hit_ids"])
            for branch in branches
        ):
            raise ReleaseEvidenceError(
                "candidate output has an invalid retrieval-branch trace"
            )
        candidate_ids = trace.get("candidate_ids")
        if (
            not isinstance(candidate_ids, list)
            or any(not isinstance(point_id, str) or not point_id for point_id in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
        ):
            raise ReleaseEvidenceError(
                "candidate output has an invalid exact candidate-ID trace"
            )
        final_candidate_ids = set(candidate_ids)
        for branch in branches:
            if branch["name"].startswith("cross_reference:"):
                final_candidate_ids.update(branch["hit_ids"])
        if any(hit.hit_id not in final_candidate_ids for hit in self.ranking):
            raise ReleaseEvidenceError(
                "candidate output final ranking contains an untraced candidate ID"
            )
        for hit in self.ranking:
            traced_routes = tuple(
                dict.fromkeys(
                    branch["route"]
                    for branch in branches
                    if hit.hit_id in branch["hit_ids"]
                )
            )
            if hit.routes != traced_routes:
                raise ReleaseEvidenceError(
                    "candidate ranking routes differ from the retrieval-branch trace"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "blind_question_set_sha256": self.blind_question_set_sha256,
            "candidate_id": self.candidate_id,
            "generation_id": self.generation_id,
            "configuration_sha256": self.configuration_sha256,
            "run_id": self.run_id,
            "repeat_index": self.repeat_index,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "ranking": [hit.to_dict() for hit in self.ranking],
            "answer_result": self.answer_result,
            "latency_ms": self.latency_ms,
        }

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class RepeatManifest:
    """Cryptographic summary derived from all ordered outputs in one repeated run."""

    run_id: str
    repeat_index: int
    output_count: int
    output_manifest_sha256: str
    ranking_sha256: str
    answer_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateReview:
    """Configuration-blind legal review bound to the exact primary-run output row."""

    dataset_id: str
    candidate_id: str
    question_id: str
    candidate_output_sha256: str
    reviewer_id: str
    role: ReviewerRole
    blinded: bool
    overall_severity: ErrorSeverity
    errors: tuple[ErrorFinding, ...]
    rationale: str

    def validate(self) -> None:
        identity = (
            self.dataset_id,
            self.candidate_id,
            self.question_id,
            self.reviewer_id,
            self.rationale,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identity):
            raise ReleaseEvidenceError(
                "candidate review identity and rationale are required"
            )
        if not _is_sha256(self.candidate_output_sha256):
            raise ReleaseEvidenceError("candidate review output hash is invalid")
        if self.blinded is not True:
            raise ReleaseEvidenceError(
                "candidate reviewers must be blind to configuration"
            )
        if not isinstance(self.role, ReviewerRole):
            raise ReleaseEvidenceError("candidate review role is invalid")
        if not isinstance(self.overall_severity, ErrorSeverity):
            raise ReleaseEvidenceError("candidate review severity is invalid")
        if not isinstance(self.errors, tuple):
            raise ReleaseEvidenceError(
                "candidate review errors must be an immutable tuple"
            )
        for finding in self.errors:
            if (
                not isinstance(finding, ErrorFinding)
                or finding.code not in ERROR_TAXONOMY
            ):
                raise ReleaseEvidenceError("candidate review finding is invalid")
            if ERROR_TAXONOMY[finding.code] is not finding.severity:
                raise ReleaseEvidenceError(
                    f"{finding.code.value} has non-taxonomy severity {finding.severity.value}"
                )
        maximum = max(
            (finding.severity for finding in self.errors),
            key=_SEVERITY_RANK.__getitem__,
            default=ErrorSeverity.NONE,
        )
        if self.overall_severity is not maximum:
            raise ReleaseEvidenceError(
                "review overall severity differs from its findings"
            )

    def decision_key(self) -> tuple[Any, ...]:
        return (
            self.overall_severity.value,
            tuple(
                sorted(
                    (item.code.value, item.severity.value, item.claim_ref)
                    for item in self.errors
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "candidate_id": self.candidate_id,
            "question_id": self.question_id,
            "candidate_output_sha256": self.candidate_output_sha256,
            "reviewer_id": self.reviewer_id,
            "role": self.role.value,
            "blinded": self.blinded,
            "overall_severity": self.overall_severity.value,
            "errors": [finding.to_dict() for finding in self.errors],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MechanicalAudit:
    question_id: str
    answered: bool
    valid: bool
    checks_total: int
    checks_valid: int
    issue_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseEvidenceBundle:
    """The derived aggregate plus hashes of every source artifact family."""

    candidate: ReleaseCandidate
    repeat_manifests: tuple[RepeatManifest, ...]
    review_manifest_sha256: str
    mechanical_manifest_sha256: str
    outcome_manifest_sha256: str
    canonical_evidence_manifest_sha256: str
    release_evidence_manifest_sha256: str
    primary_output_manifest_sha256: str
    outcomes: tuple[ReleaseQuestionOutcome, ...]
    dataset: V3Dataset
    runs: tuple[tuple[CandidateOutput, ...], ...]
    reviews: tuple[CandidateReview, ...]
    canonical_evidence: Mapping[str, Mapping[str, Any]]
    canonical_documents: tuple[CanonicalDocument, ...]

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate": self.candidate.to_dict(),
            "repeat_manifests": [item.to_dict() for item in self.repeat_manifests],
            "review_manifest_sha256": self.review_manifest_sha256,
            "mechanical_manifest_sha256": self.mechanical_manifest_sha256,
            "outcome_manifest_sha256": self.outcome_manifest_sha256,
            "canonical_evidence_manifest_sha256": (
                self.canonical_evidence_manifest_sha256
            ),
            "release_evidence_manifest_sha256": self.release_evidence_manifest_sha256,
            "primary_output_manifest_sha256": self.primary_output_manifest_sha256,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    def verified_material(
        self, *, policy: ReleasePolicy
    ) -> tuple[ReleaseCandidate, tuple[ReleaseQuestionOutcome, ...]]:
        """Recompute every aggregate and manifest from the retained source artifacts."""

        rebuilt = aggregate_release_candidate(
            self.dataset,
            policy=policy,
            runs=self.runs,
            reviews=self.reviews,
            canonical_evidence=self.canonical_evidence,
            canonical_documents=self.canonical_documents,
        )
        if _canonical_json(rebuilt.to_dict()) != _canonical_json(self.to_dict()):
            raise ReleaseEvidenceError(
                "release evidence bundle differs from recomputed source artifacts"
            )
        return rebuilt.candidate, rebuilt.outcomes


def _make_repeat_manifest(outputs: Sequence[CandidateOutput]) -> RepeatManifest:
    ordered = sorted(outputs, key=lambda item: item.question_id)
    first = ordered[0]
    ranking_material = [
        {
            "question_id": item.question_id,
            "ranking": [hit.to_dict() for hit in item.ranking],
        }
        for item in ordered
    ]
    answer_material = [
        {
            "question_id": item.question_id,
            "answer_result": _deterministic_answer_material(item.answer_result),
        }
        for item in ordered
    ]
    output_material = [
        {"question_id": item.question_id, "candidate_output_sha256": item.sha256}
        for item in ordered
    ]
    return RepeatManifest(
        run_id=first.run_id,
        repeat_index=first.repeat_index,
        output_count=len(ordered),
        output_manifest_sha256=_sha256(_canonical_json(output_material)),
        ranking_sha256=_sha256(_canonical_json(ranking_material)),
        answer_sha256=_sha256(_canonical_json(answer_material)),
    )


def _deterministic_answer_material(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove wall-clock telemetry while retaining every answer/provenance decision.

    Retrieval timings are expected to vary between otherwise identical executions.  They
    remain sealed in :class:`CandidateOutput` and its full output hash, but cannot be part
    of the repeat *answer* hash.  Ranked IDs, exact scores, and routes are independently
    retained in ``ranking_sha256``.
    """

    material = json.loads(_canonical_json(dict(result)))
    trace = material.get("trace")
    if isinstance(trace, dict):
        trace.pop("retrieval_timings_ms", None)
        branches = trace.get("retrieval_branches")
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    branch.pop("elapsed_ms", None)
    return material


def _question_sha256(question: V3Question) -> str:
    return _sha256(question.question)


def _canonical_evidence_subset_manifest(
    outputs: Sequence[CandidateOutput],
    canonical_evidence: Mapping[str, Mapping[str, Any]],
    canonical_documents: Sequence[CanonicalDocument],
) -> str:
    """Resolve every answered evidence item against the active-generation ledger.

    Evidence IDs and hashes are not signatures: a fabricated item can be internally
    self-consistent.  Release review therefore requires a separately supplied canonical
    mapping produced from the immutable active generation and compares all legally
    material identity, text, offset, authority, completeness, and URL fields exactly.
    """

    if not isinstance(canonical_evidence, Mapping):
        raise ReleaseEvidenceError("canonical evidence resolver must be a mapping")
    document_index = {document.key: document for document in canonical_documents}
    if len(document_index) != len(canonical_documents):
        raise ReleaseEvidenceError("canonical document resolver contains duplicate versions")
    referenced: dict[str, dict[str, Any]] = {}
    for output in outputs:
        result = output.answer_result
        if result.get("outcome") != "answer":
            continue
        pack = result.get("evidence")
        items = pack.get("items") if isinstance(pack, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not isinstance(
                item.get("evidence_id"), str
            ):
                continue
            evidence_id = item["evidence_id"]
            record = canonical_evidence.get(evidence_id)
            if not isinstance(record, Mapping):
                raise ReleaseEvidenceError(
                    f"{output.question_id}: evidence ID does not resolve in the "
                    f"active generation: {evidence_id!r}"
                )
            missing_item = set(_CANONICAL_EVIDENCE_FIELDS) - set(item)
            missing_record = set(_CANONICAL_EVIDENCE_FIELDS) - set(record)
            if missing_item or missing_record:
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence contract is incomplete; "
                    f"item missing {sorted(missing_item)}, record missing "
                    f"{sorted(missing_record)}"
                )
            mismatches = [
                field
                for field in _CANONICAL_EVIDENCE_FIELDS
                if _canonical_json(item[field]) != _canonical_json(record[field])
            ]
            if mismatches:
                raise ReleaseEvidenceError(
                    f"{output.question_id}: evidence differs from the active canonical "
                    f"record in {sorted(mismatches)}"
                )
            if record["generation_id"] != output.generation_id:
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence generation mismatch"
                )
            document_key = (
                record["source"],
                record["document_id"],
                record["version_id"],
            )
            document = document_index.get(document_key)
            if document is None:
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence document version does not "
                    "resolve in the validated v3 corpus"
                )
            start, end = record["char_start"], record["char_end"]
            if (
                record["content_hash"] != document.content_sha256
                or record["effective_from"] != document.effective_from
                or record["effective_to"] != document.effective_to
                or record["status"] != document.status
                or record["repeal_date"] != document.repeal_date
                or record["consolidation_status"] != document.consolidation_status
                or record["content_complete"] is not document.content_complete
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or document.text[start:end] != record["text"]
            ):
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence is not anchored to the "
                    "validated document text, hash, offsets, and effective interval"
                )
            if (
                record["content_complete"] is not True
                or record["extraction_status"] != "full_text"
                or record["source_authority"] not in {"official", "primary_official"}
                or record["version_lineage_complete"] is not True
                or record["identity_ambiguous"] is not False
            ):
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence is not authoritative, "
                    "complete, lineage-proven, and unambiguous"
                )
            if not any(
                isinstance(record[field], str) and bool(record[field])
                for field in ("official_url", "official_binary_url")
            ):
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence lacks an official URL"
                )
            try:
                locator = parse_evidence_id(evidence_id)
            except EvidenceContractError as exc:
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence locator is invalid"
                ) from exc
            if (
                locator["generation_id"] != record["generation_id"]
                or locator["point_id"] != record["point_id"]
                or locator["passage_hash"] != record["passage_hash"]
            ):
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical evidence locator mismatch"
                )
            normalized = json.loads(_canonical_json(dict(record)))
            existing = referenced.get(evidence_id)
            if existing is not None and existing != normalized:
                raise ReleaseEvidenceError(
                    f"{output.question_id}: canonical resolver changed within the run"
                )
            referenced[evidence_id] = normalized
    material = [
        {"evidence_id": evidence_id, "record": referenced[evidence_id]}
        for evidence_id in sorted(referenced)
    ]
    return _sha256(_canonical_json(material))


def _parse_iso_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _effective_on(item: Mapping[str, Any], when: date) -> bool:
    start = _parse_iso_date(item.get("effective_from"))
    raw_end = item.get("effective_to")
    end = _parse_iso_date(raw_end) if raw_end is not None else None
    if start is None or (raw_end is not None and end is None):
        return False
    return start <= when and (end is None or when < end)


def _normative_evidence(item: Mapping[str, Any]) -> bool:
    return item.get("source") == "matsne" or item.get("document_type") in {
        "law",
        "legislation",
        "normative_act",
    }


def _check_result(
    output: CandidateOutput,
    question: V3Question,
    *,
    corpus_as_of: str,
) -> MechanicalAudit:
    """Recompute the mechanically decidable subset of the AnswerResult contract."""

    result = output.answer_result
    if result.get("outcome") != "answer":
        return MechanicalAudit(output.question_id, False, True, 0, 0, ())

    issues: list[str] = []
    checks_total = checks_valid = 0

    def check(code: str, condition: bool) -> None:
        nonlocal checks_total, checks_valid
        checks_total += 1
        if condition:
            checks_valid += 1
        else:
            issues.append(code)

    answer_text = result.get("answer_text")
    claims = result.get("claims")
    evidence_pack = result.get("evidence")
    validation_issues = result.get("validation_issues")
    trace = result.get("trace")
    check(
        "answer_text_missing",
        isinstance(answer_text, str) and bool(answer_text.strip()),
    )
    check("claims_missing", isinstance(claims, list) and bool(claims))
    check("evidence_pack_missing", isinstance(evidence_pack, dict))
    check("validation_issues_present", validation_issues == [])
    check("trace_missing", isinstance(trace, dict))
    if not isinstance(claims, list) or not isinstance(evidence_pack, dict):
        return MechanicalAudit(
            output.question_id, True, False, checks_total, checks_valid, tuple(issues)
        )

    items = evidence_pack.get("items")
    items = items if isinstance(items, list) else []
    check("evidence_items_missing", bool(items))
    generation_id = evidence_pack.get("generation_id")
    retrieval_hash = evidence_pack.get("retrieval_result_hash")
    check("pack_generation_mismatch", generation_id == output.generation_id)
    check("retrieval_hash_invalid", _is_sha256(retrieval_hash))
    token_count = evidence_pack.get("token_count")
    max_tokens = evidence_pack.get("max_tokens")
    check(
        "pack_token_count_invalid",
        isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count >= 0,
    )
    check(
        "pack_token_bound_invalid",
        isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens > 0
        and isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count <= max_tokens,
    )
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_item in enumerate(items):
        prefix = f"evidence_{index}"
        check(f"{prefix}_not_object", isinstance(raw_item, dict))
        if not isinstance(raw_item, dict):
            continue
        evidence_id = raw_item.get("evidence_id")
        passage_hash = raw_item.get("passage_hash")
        text = raw_item.get("text")
        check(
            f"{prefix}_id_invalid", isinstance(evidence_id, str) and bool(evidence_id)
        )
        check(
            f"{prefix}_passage_hash_mismatch",
            isinstance(text, str) and passage_hash == _sha256(text),
        )
        check(
            f"{prefix}_content_hash_invalid", _is_sha256(raw_item.get("content_hash"))
        )
        item_start, item_end = raw_item.get("char_start"), raw_item.get("char_end")
        check(
            f"{prefix}_offsets_invalid",
            isinstance(item_start, int)
            and not isinstance(item_start, bool)
            and isinstance(item_end, int)
            and not isinstance(item_end, bool)
            and isinstance(text, str)
            and item_start >= 0
            and item_end == item_start + len(text),
        )
        check(
            f"{prefix}_offset_unit_invalid",
            raw_item.get("offset_unit") == "unicode_codepoint",
        )
        locator: dict[str, str] | None = None
        try:
            locator = parse_evidence_id(evidence_id)
        except (EvidenceContractError, TypeError):
            pass
        check(
            f"{prefix}_locator_invalid",
            locator is not None
            and locator.get("generation_id") == output.generation_id
            and locator.get("point_id") == raw_item.get("point_id")
            and locator.get("passage_hash") == passage_hash,
        )
        check(
            f"{prefix}_generation_mismatch",
            raw_item.get("generation_id") == generation_id,
        )
        check(f"{prefix}_incomplete", raw_item.get("content_complete") is True)
        check(
            f"{prefix}_extraction_unverified",
            raw_item.get("extraction_status") == "full_text",
        )
        check(
            f"{prefix}_authority_invalid",
            raw_item.get("source_authority") in {"official", "primary_official"},
        )
        check(f"{prefix}_version_missing", bool(raw_item.get("version_id")))
        check(
            f"{prefix}_lineage_incomplete",
            raw_item.get("version_lineage_complete") is True,
        )
        check(f"{prefix}_ambiguous", raw_item.get("identity_ambiguous") is False)
        if _normative_evidence(raw_item):
            trace_date = (
                trace.get("resolved_current_date") if isinstance(trace, dict) else None
            )
            effective_date = _parse_iso_date(question.as_of or trace_date)
            check(
                f"{prefix}_effective_date_invalid",
                effective_date is not None
                and _effective_on(raw_item, effective_date),
            )
            if question.as_of is None:
                repeal = raw_item.get("repeal_date")
                repeal_date = _parse_iso_date(repeal) if repeal is not None else None
                check(
                    f"{prefix}_status_not_in_force",
                    raw_item.get("status") == "in_force",
                )
                check(
                    f"{prefix}_repealed_for_current_date",
                    repeal is None
                    or (
                        repeal_date is not None
                        and effective_date is not None
                        and effective_date < repeal_date
                    ),
                )
        if isinstance(evidence_id, str) and evidence_id not in evidence_by_id:
            evidence_by_id[evidence_id] = raw_item
        else:
            check(f"{prefix}_duplicate_id", False)

    expected_pack_id = _sha256(
        _canonical_json(
            {
                "generation_id": generation_id,
                "retrieval_hash": retrieval_hash,
                "evidence_ids": [
                    item.get("evidence_id") for item in items if isinstance(item, dict)
                ],
            }
        )
    )
    check("pack_hash_mismatch", evidence_pack.get("pack_id") == expected_pack_id)

    seen_claims: set[str] = set()
    for index, raw_claim in enumerate(claims):
        prefix = f"claim_{index}"
        check(f"{prefix}_not_object", isinstance(raw_claim, dict))
        if not isinstance(raw_claim, dict):
            continue
        claim_id = raw_claim.get("claim_id")
        evidence_ids = raw_claim.get("evidence_ids")
        check(
            f"{prefix}_id_invalid",
            isinstance(claim_id, str)
            and bool(claim_id)
            and claim_id not in seen_claims,
        )
        if isinstance(claim_id, str):
            seen_claims.add(claim_id)
        check(f"{prefix}_text_missing", bool(str(raw_claim.get("text") or "").strip()))
        check(
            f"{prefix}_unsupported",
            isinstance(evidence_ids, list)
            and bool(evidence_ids)
            and all(evidence_id in evidence_by_id for evidence_id in evidence_ids),
        )
        evidence_ids = evidence_ids if isinstance(evidence_ids, list) else []
        quotations = raw_claim.get("quotations")
        check(f"{prefix}_quotations_invalid", isinstance(quotations, list))
        for quote_index, quotation in enumerate(
            quotations if isinstance(quotations, list) else []
        ):
            qprefix = f"{prefix}_quote_{quote_index}"
            check(f"{qprefix}_not_object", isinstance(quotation, dict))
            if not isinstance(quotation, dict):
                continue
            evidence_id = quotation.get("evidence_id")
            item = evidence_by_id.get(evidence_id)
            check(
                f"{qprefix}_unlinked", item is not None and evidence_id in evidence_ids
            )
            quote = quotation.get("quote")
            start, end = quotation.get("char_start"), quotation.get("char_end")
            exact = False
            if (
                item is not None
                and isinstance(quote, str)
                and isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and isinstance(item.get("char_start"), int)
                and isinstance(item.get("text"), str)
            ):
                relative_start = start - item["char_start"]
                relative_end = end - item["char_start"]
                exact = (
                    0 <= relative_start < relative_end <= len(item["text"])
                    and item["text"][relative_start:relative_end] == quote
                )
            check(f"{qprefix}_offset_mismatch", exact)
            check(
                f"{qprefix}_hash_mismatch",
                isinstance(quote, str)
                and quotation.get("quote_hash") == _sha256(quote),
            )

    if isinstance(trace, dict):
        versions = result.get("versions")
        if question.as_of is None:
            check(
                "trace_current_date_mismatch",
                trace.get("resolved_current_date") == corpus_as_of,
            )
        check(
            "trace_generation_mismatch",
            trace.get("generation_id") == output.generation_id,
        )
        check("trace_service_abstention", trace.get("service_abstention") is False)
        check("trace_abstention_reason", trace.get("abstention_reason") is None)
        check("trace_degraded", trace.get("degraded") is False)
        check("trace_degraded_reason", trace.get("degraded_reason") is None)
        check("trace_identity_ambiguous", trace.get("identity_ambiguous") is False)
        check(
            "trace_translator_version_mismatch",
            isinstance(versions, dict)
            and trace.get("translator_version") == versions.get("translator"),
        )
        check(
            "trace_retrieval_hash_mismatch",
            trace.get("retrieval_result_hash") == retrieval_hash,
        )
        check(
            "trace_pack_hash_mismatch",
            trace.get("evidence_pack_id") == evidence_pack.get("pack_id"),
        )
        check("trace_validation_codes_present", trace.get("validation_codes") == [])
        check(
            "trace_answer_hash_mismatch",
            isinstance(answer_text, str)
            and trace.get("answer_hash") == _sha256(answer_text),
        )
        check("calibration_decision_missing", trace.get("calibration_allowed") is True)

        branches = trace.get("retrieval_branches")
        branches_valid = isinstance(branches, list) and all(
            isinstance(branch, dict)
            and isinstance(branch.get("name"), str)
            and isinstance(branch.get("query"), str)
            and isinstance(branch.get("filters"), dict)
            and isinstance(branch.get("hit_ids"), list)
            and all(isinstance(hit_id, str) for hit_id in branch["hit_ids"])
            and isinstance(branch.get("route"), str)
            for branch in branches
        )
        check("trace_branches_invalid", branches_valid)
        branch_material: list[dict[str, Any]] = []
        branch_hashes: list[str] = []
        if branches_valid:
            for branch in branches:
                branch_material.append(
                    {
                        "name": branch["name"],
                        "query": branch["query"],
                        "filters": branch["filters"],
                        "hit_ids": branch["hit_ids"],
                        "route": branch["route"],
                    }
                )
                branch_hashes.append(
                    _sha256(
                        _canonical_json(
                            {
                                "name": branch["name"],
                                "query": branch["query"],
                                "filters": branch["filters"],
                                "hits": branch["hit_ids"],
                                "route": branch["route"],
                            }
                        )
                    )
                )
        check(
            "trace_branch_hashes_mismatch", trace.get("branch_hashes") == branch_hashes
        )

        ranked_hits = trace.get("ranked_hits")
        ranking_valid = isinstance(ranked_hits, list) and all(
            isinstance(hit, dict)
            and isinstance(hit.get("point_id"), str)
            and isinstance(hit.get("score"), str)
            for hit in ranked_hits
        )
        check("trace_ranking_invalid", ranking_valid)
        candidate_ids = trace.get("candidate_ids")
        candidates_valid = (
            isinstance(candidate_ids, list)
            and all(isinstance(point_id, str) and bool(point_id) for point_id in candidate_ids)
            and len(candidate_ids) == len(set(candidate_ids))
        )
        check("trace_candidate_ids_invalid", candidates_valid)
        try:
            plan = plan_query(
                question.question,
                language=question.language.value,
                as_of=question.as_of,
            )
        except (TypeError, ValueError):
            plan = None
        resolved_current_date = trace.get("resolved_current_date")
        fingerprint = trace.get("retrieval_fingerprint")
        retrieval_material_valid = (
            plan is not None
            and branches_valid
            and ranking_valid
            and candidates_valid
            and isinstance(resolved_current_date, str)
            and bool(resolved_current_date)
            and isinstance(fingerprint, str)
            and bool(fingerprint)
        )
        check("trace_retrieval_material_invalid", retrieval_material_valid)
        expected_retrieval_hash = None
        if retrieval_material_valid and plan is not None:
            expected_retrieval_hash = _sha256(
                _canonical_json(
                    {
                        "question": plan.question,
                        "language": plan.language.value,
                        "intent": plan.intent.value,
                        "as_of": plan.as_of,
                        "resolved_current_date": resolved_current_date,
                        "fingerprint": fingerprint,
                        "generation_id": trace.get("generation_id"),
                        "service_abstention": trace.get("service_abstention"),
                        "abstention_reason": trace.get("abstention_reason"),
                        "degraded": trace.get("degraded"),
                        "degraded_reason": trace.get("degraded_reason"),
                        "identity_ambiguous": trace.get("identity_ambiguous"),
                        "translator_version": trace.get("translator_version"),
                        "branches": branch_material,
                        "candidate_ids": candidate_ids,
                        "final_ranking": [
                            {
                                "point_id": hit["point_id"],
                                "score": hit["score"],
                            }
                            for hit in ranked_hits
                        ],
                    }
                )
            )
        check(
            "trace_retrieval_result_hash_not_reproducible",
            expected_retrieval_hash is not None
            and trace.get("retrieval_result_hash") == expected_retrieval_hash,
        )

        validation_codes = trace.get("validation_codes")
        repair_attempted = trace.get("repair_attempted")
        trace_id_material_valid = (
            isinstance(validation_codes, list)
            and isinstance(repair_attempted, bool)
            and branches_valid
        )
        check("trace_id_material_invalid", trace_id_material_valid)
        expected_trace_id = None
        if trace_id_material_valid:
            expected_trace_id = _sha256(
                _canonical_json(
                    {
                        "question": question.question,
                        "versions": result.get("versions"),
                        "retrieval_hash": trace.get("retrieval_result_hash"),
                        "branches": branch_hashes,
                        "validation_codes": validation_codes,
                        "repair_attempted": repair_attempted,
                        "evidence_pack_id": trace.get("evidence_pack_id"),
                        "freshness_audit_id": trace.get("freshness_audit_id"),
                        "freshness_decision": trace.get("freshness_decision"),
                        "calibration_allowed": trace.get("calibration_allowed"),
                        "calibration_reason": trace.get("calibration_reason"),
                    }
                )
            )
        check(
            "trace_id_mismatch",
            expected_trace_id is not None
            and trace.get("trace_id") == expected_trace_id,
        )
    return MechanicalAudit(
        output.question_id,
        True,
        not issues,
        checks_total,
        checks_valid,
        tuple(issues),
    )


def _final_reviews(
    *,
    questions: Mapping[str, V3Question],
    primary_outputs: Mapping[str, CandidateOutput],
    reviews: Sequence[CandidateReview],
) -> dict[str, CandidateReview]:
    grouped: dict[str, list[CandidateReview]] = {
        question_id: [] for question_id in questions
    }
    for review in reviews:
        review.validate()
        if review.question_id not in grouped:
            raise ReleaseEvidenceError(
                f"review references unknown question {review.question_id!r}"
            )
        output = primary_outputs[review.question_id]
        if (
            review.dataset_id != output.dataset_id
            or review.candidate_id != output.candidate_id
            or review.candidate_output_sha256 != output.sha256
        ):
            raise ReleaseEvidenceError(
                "candidate review is not bound to the primary output"
            )
        grouped[review.question_id].append(review)

    final: dict[str, CandidateReview] = {}
    for question_id, panel in grouped.items():
        primaries = [item for item in panel if item.role is ReviewerRole.REVIEWER]
        adjudicators = [item for item in panel if item.role is ReviewerRole.ADJUDICATOR]
        reviewer_ids = [item.reviewer_id for item in panel]
        if len(primaries) != 2 or len(reviewer_ids) != len(set(reviewer_ids)):
            raise ReleaseEvidenceError(
                f"{question_id}: requires two distinct primary candidate reviewers"
            )
        disagree = primaries[0].decision_key() != primaries[1].decision_key()
        if disagree and len(adjudicators) != 1:
            raise ReleaseEvidenceError(
                f"{question_id}: reviewer disagreement requires one third adjudicator"
            )
        if not disagree and adjudicators:
            raise ReleaseEvidenceError(
                f"{question_id}: agreement must not receive an adjudicator"
            )
        final[question_id] = adjudicators[0] if disagree else primaries[0]
    return final


def _slice_members(question: V3Question, *, name: str, category: str) -> bool:
    normalized = name.casefold()
    if category == "source":
        return any(
            span.source.casefold() == normalized
            for group in question.evidence_groups
            for span in group.alternatives
        )
    aliases = {
        "historical": QuestionTag.HISTORICAL_AS_OF.value,
        "current": QuestionTag.CURRENT_LAW.value,
    }
    normalized = aliases.get(normalized, normalized)
    return (
        normalized in {tag.value for tag in question.tags}
        or normalized == question.risk_level.value
        or normalized == question.language.value
        or normalized == f"risk_{question.risk_level.value}"
    )


def _release_cluster_ids(questions: Sequence[V3Question]) -> dict[str, str]:
    """Build connected document/version-family clusters for release inference."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            parent[larger] = smaller

    for question in questions:
        families = tuple(dict.fromkeys(question.partition_family_ids))
        for family in families:
            find(family)
        for family in families[1:]:
            union(families[0], family)
    components: dict[str, list[str]] = {}
    for family in sorted(parent):
        components.setdefault(find(family), []).append(family)
    component_ids = {
        root: "family:" + _sha256(_canonical_json(families))
        for root, families in components.items()
    }
    return {
        question.question_id: (
            component_ids[find(question.partition_family_ids[0])]
            if question.partition_family_ids
            else "unpartitioned"
        )
        for question in questions
    }


def _freshness_allowed(result: Mapping[str, Any]) -> bool:
    trace = result.get("trace")
    if not isinstance(trace, dict):
        return False
    decision = trace.get("freshness_decision")
    audit_id = trace.get("freshness_audit_id")
    if isinstance(decision, dict):
        allowed = decision.get("outcome") == "eligible"
        audit_id = audit_id or decision.get("audit_id")
    else:
        allowed = decision == "eligible"
    return allowed and _is_sha256(audit_id)


def _answered_operational_flags(
    question: V3Question,
    output: CandidateOutput,
    audit: MechanicalAudit,
) -> tuple[bool, bool, bool, bool]:
    result = output.answer_result
    if result.get("outcome") != "answer":
        return False, False, False, False
    trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
    validation_issues = result.get("validation_issues")
    codes = (
        {
            str(item.get("code", ""))
            for item in validation_issues
            if isinstance(item, dict)
        }
        if isinstance(validation_issues, list)
        else {"validation_contract_missing"}
    )
    codes.update(
        str(code) for code in trace.get("validation_codes", []) if isinstance(code, str)
    )
    degraded = any("degrad" in code for code in codes) or trace.get("degraded") is True
    evidence = result.get("evidence")
    items = evidence.get("items", []) if isinstance(evidence, dict) else []
    ambiguous = any(
        isinstance(item, dict) and item.get("identity_ambiguous") is not False
        for item in items
    )
    has_normative_evidence = any(
        isinstance(item, dict)
        and (
            item.get("source") == "matsne"
            or item.get("document_type") in {"legislation", "normative_act"}
        )
        for item in items
    )
    current_law = question.as_of is None and (
        QuestionTag.CURRENT_LAW in question.tags or has_normative_evidence
    )
    stale = current_law and not _freshness_allowed(result)
    validation_failed = not audit.valid or bool(codes)
    return degraded, stale, ambiguous, validation_failed


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def aggregate_release_candidate(
    dataset: V3Dataset,
    *,
    policy: ReleasePolicy,
    runs: Sequence[Sequence[CandidateOutput]],
    reviews: Sequence[CandidateReview],
    canonical_evidence: Mapping[str, Mapping[str, Any]],
    canonical_documents: Sequence[CanonicalDocument],
) -> ReleaseEvidenceBundle:
    """Derive the only release-gate aggregate accepted by the strict review workflow."""

    policy.validate()
    try:
        validate_dataset(dataset, canonical_documents)
    except ValueError as exc:
        raise ReleaseEvidenceError(
            "release aggregation requires a canonically validated v3 dataset"
        ) from exc
    protocol = dataset.manifest.blind_protocol
    blind_rows = tuple(
        question for question in dataset.questions if question.split is Split.BLIND
    )
    blind_questions = {question.question_id: question for question in blind_rows}
    if not protocol.sealed or not protocol.blind_ids_sha256:
        raise ReleaseEvidenceError(
            "release aggregation requires a sealed blind protocol"
        )
    if len(blind_questions) != len(blind_rows):
        raise ReleaseEvidenceError("sealed blind question IDs must be unique")
    if protocol.planned_question_count != len(blind_questions):
        raise ReleaseEvidenceError(
            "sealed blind question count differs from the predeclared count"
        )
    if (
        protocol.target_answered_count < 1000
        or protocol.target_answered_count > protocol.planned_question_count
    ):
        raise ReleaseEvidenceError(
            "sealed blind protocol requires a feasible target of at least 1,000 answers"
        )
    if not _is_sha256(protocol.sampling_plan_sha256):
        raise ReleaseEvidenceError("sealed blind sampling-plan hash is invalid")
    if protocol.blind_ids_sha256 != blind_ids_hash(blind_questions):
        raise ReleaseEvidenceError("sealed blind question-set hash is invalid")
    if policy.blind_question_set_sha256 != protocol.blind_ids_sha256:
        raise ReleaseEvidenceError(
            "release policy targets a different blind question set"
        )
    if len(runs) < 2:
        raise ReleaseEvidenceError("at least two complete repeated runs are required")

    expected_dataset_sha = dataset_hash(dataset)
    run_maps: list[dict[str, CandidateOutput]] = []
    manifests: list[RepeatManifest] = []
    common_identity: tuple[str, str, str] | None = None
    seen_run_ids: set[str] = set()
    seen_repeat_indexes: set[int] = set()
    for run in runs:
        if not run:
            raise ReleaseEvidenceError("repeated run cannot be empty")
        by_question: dict[str, CandidateOutput] = {}
        for output in run:
            output.validate()
            if output.question_id in by_question:
                raise ReleaseEvidenceError(
                    "repeated run contains duplicate question output"
                )
            question = blind_questions.get(output.question_id)
            if question is None:
                raise ReleaseEvidenceError("repeated run contains a non-blind question")
            if output.question_sha256 != _question_sha256(question):
                raise ReleaseEvidenceError("candidate output question hash mismatch")
            if (
                output.dataset_id != dataset.manifest.dataset_id
                or output.dataset_sha256 != expected_dataset_sha
                or output.blind_question_set_sha256 != protocol.blind_ids_sha256
                or output.generation_id != dataset.manifest.corpus_generation
            ):
                raise ReleaseEvidenceError(
                    "candidate output is bound to another dataset/generation"
                )
            identity = (
                output.candidate_id,
                output.generation_id,
                output.configuration_sha256,
            )
            common_identity = common_identity or identity
            if identity != common_identity:
                raise ReleaseEvidenceError(
                    "candidate identity differs across repeated outputs"
                )
            by_question[output.question_id] = output
        if set(by_question) != set(blind_questions):
            raise ReleaseEvidenceError(
                "each run must contain every and only blind question"
            )
        first = next(iter(by_question.values()))
        if any(
            item.run_id != first.run_id or item.repeat_index != first.repeat_index
            for item in by_question.values()
        ):
            raise ReleaseEvidenceError("run identity differs within a repeated run")
        if first.run_id in seen_run_ids or first.repeat_index in seen_repeat_indexes:
            raise ReleaseEvidenceError("run IDs and repeat indexes must be unique")
        seen_run_ids.add(first.run_id)
        seen_repeat_indexes.add(first.repeat_index)
        run_maps.append(by_question)
        manifests.append(_make_repeat_manifest(tuple(by_question.values())))
    if seen_repeat_indexes != set(range(len(runs))):
        raise ReleaseEvidenceError("repeat indexes must be contiguous starting at zero")

    canonical_evidence_manifest_sha = _canonical_evidence_subset_manifest(
        tuple(output for run in run_maps for output in run.values()),
        canonical_evidence,
        canonical_documents,
    )

    primary_outputs = run_maps[
        min(range(len(manifests)), key=lambda i: manifests[i].repeat_index)
    ]
    final_reviews = _final_reviews(
        questions=blind_questions,
        primary_outputs=primary_outputs,
        reviews=reviews,
    )
    audits = {
        question_id: _check_result(
            output,
            blind_questions[question_id],
            corpus_as_of=dataset.manifest.corpus_as_of,
        )
        for question_id, output in primary_outputs.items()
    }

    cluster_ids = _release_cluster_ids(blind_rows)
    answered = severe = material = 0
    in_scope = answered_in_scope = 0
    degraded = stale = ambiguous = validation_failed = 0
    successful: dict[str, bool] = {}
    release_outcomes: list[ReleaseQuestionOutcome] = []
    for question_id, question in blind_questions.items():
        output = primary_outputs[question_id]
        result = output.answer_result
        actual_outcome = result["outcome"]
        gold = adjudicated_judgment(question)
        review = final_reviews[question_id]
        is_answer = actual_outcome == "answer"
        if gold.answerable:
            in_scope += 1
            answered_in_scope += int(is_answer)
        severe_error = is_answer and review.overall_severity is ErrorSeverity.SEVERE
        material_error = is_answer and (
            _SEVERITY_RANK[review.overall_severity]
            >= _SEVERITY_RANK[ErrorSeverity.MATERIAL]
            or actual_outcome != gold.expected_outcome.value
        )
        if is_answer:
            answered += 1
            severe += int(severe_error)
            # Answering a gold clarification/refusal case is mechanically a material
            # should-have-clarified/abstained error even if a reviewer accidentally omits
            # that finding.  Human review may increase severity, never erase the mismatch.
            material += int(material_error)
            flags = _answered_operational_flags(question, output, audits[question_id])
            degraded += int(flags[0])
            stale += int(flags[1])
            ambiguous += int(flags[2])
            validation_failed += int(flags[3])
        release_outcomes.append(
            ReleaseQuestionOutcome(
                question_id=question_id,
                cluster_id=cluster_ids[question_id],
                answered=is_answer,
                severe_error=severe_error,
                material_error=material_error,
                in_scope_answerable=gold.answerable,
                answered_in_scope=gold.answerable and is_answer,
            )
        )
        successful[question_id] = (
            actual_outcome == gold.expected_outcome.value
            and _SEVERITY_RANK[review.overall_severity]
            < _SEVERITY_RANK[ErrorSeverity.MATERIAL]
        )

    comparisons: list[SliceComparison] = []
    for declared in policy.slices:
        members = [
            question_id
            for question_id, question in blind_questions.items()
            if _slice_members(question, name=declared.name, category=declared.category)
        ]
        if not members:
            raise ReleaseEvidenceError(
                f"release policy slice {declared.name!r} has no blind questions"
            )
        comparisons.append(
            SliceComparison(
                name=declared.name,
                baseline=declared.baseline,
                candidate=sum(successful[item] for item in members) / len(members),
                higher_is_better=declared.higher_is_better,
                category=declared.category,
            )
        )

    ordered_reviews = sorted(
        (review.to_dict() for review in reviews),
        key=lambda item: (item["question_id"], item["role"], item["reviewer_id"]),
    )
    ordered_audits = [audits[key].to_dict() for key in sorted(audits)]
    review_manifest_sha = _sha256(_canonical_json(ordered_reviews))
    mechanical_manifest_sha = _sha256(_canonical_json(ordered_audits))
    ordered_outcomes = tuple(
        sorted(release_outcomes, key=lambda item: item.question_id)
    )
    outcome_manifest_sha = _sha256(
        _canonical_json([item.to_dict() for item in ordered_outcomes])
    )
    valid_answer_audits = [
        audit for audit in audits.values() if audit.answered and audit.valid
    ]
    mechanical_checks_total = sum(audit.checks_total for audit in audits.values())
    mechanical_checks_valid = sum(audit.checks_valid for audit in audits.values())
    assert common_identity is not None
    candidate_id, generation_id, configuration_sha = common_identity
    ordered_manifests = tuple(sorted(manifests, key=lambda item: item.repeat_index))
    primary_manifest = ordered_manifests[0]
    repeat_manifest_hashes = [
        _sha256(_canonical_json(item.to_dict())) for item in ordered_manifests
    ]
    output_manifest_hashes = [item.output_manifest_sha256 for item in ordered_manifests]
    release_evidence_manifest_sha = _sha256(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_sha256": expected_dataset_sha,
                "blind_question_set_sha256": protocol.blind_ids_sha256,
                "candidate_id": candidate_id,
                "generation_id": generation_id,
                "configuration_sha256": configuration_sha,
                "release_policy_sha256": policy.sha256,
                "primary_output_manifest_sha256": (
                    primary_manifest.output_manifest_sha256
                ),
                "output_manifest_sha256": output_manifest_hashes,
                "repeat_manifest_sha256": repeat_manifest_hashes,
                "review_manifest_sha256": review_manifest_sha,
                "mechanical_manifest_sha256": mechanical_manifest_sha,
                "outcome_manifest_sha256": outcome_manifest_sha,
                "canonical_evidence_manifest_sha256": (canonical_evidence_manifest_sha),
            }
        )
    )
    candidate = ReleaseCandidate(
        candidate_id=candidate_id,
        generation_id=generation_id,
        configuration_hash=configuration_sha,
        answered_outcomes=answered,
        severe_errors=severe,
        material_errors=material,
        in_scope_answerable=in_scope,
        answered_in_scope=answered_in_scope,
        mechanically_valid_answers=len(valid_answer_audits),
        mechanical_checks_total=mechanical_checks_total,
        mechanical_checks_valid=mechanical_checks_valid,
        degraded_answers=degraded,
        stale_current_law_answers=stale,
        ambiguous_identity_answers=ambiguous,
        validation_failed_answers=validation_failed,
        p95_latency_ms=_percentile_95(
            [
                output.latency_ms
                for run_outputs in run_maps
                for output in run_outputs.values()
            ]
        ),
        ranking_hashes=tuple(
            item.ranking_sha256
            for item in sorted(manifests, key=lambda item: item.repeat_index)
        ),
        answer_hashes=tuple(
            item.answer_sha256
            for item in sorted(manifests, key=lambda item: item.repeat_index)
        ),
        slice_comparisons=tuple(comparisons),
        required_slice_names=policy.required_slice_names,
        release_policy_sha256=policy.sha256,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_sha256": expected_dataset_sha,
            "blind_question_set_sha256": protocol.blind_ids_sha256,
            "review_manifest_sha256": review_manifest_sha,
            "mechanical_manifest_sha256": mechanical_manifest_sha,
            "outcome_manifest_sha256": outcome_manifest_sha,
            "canonical_evidence_manifest_sha256": canonical_evidence_manifest_sha,
            "primary_output_manifest_sha256": (primary_manifest.output_manifest_sha256),
            "output_manifest_sha256": output_manifest_hashes,
            "repeat_manifest_sha256": repeat_manifest_hashes,
            "release_evidence_manifest_sha256": release_evidence_manifest_sha,
        },
    )
    candidate.validate()
    return ReleaseEvidenceBundle(
        candidate=candidate,
        repeat_manifests=ordered_manifests,
        review_manifest_sha256=review_manifest_sha,
        mechanical_manifest_sha256=mechanical_manifest_sha,
        outcome_manifest_sha256=outcome_manifest_sha,
        canonical_evidence_manifest_sha256=canonical_evidence_manifest_sha,
        release_evidence_manifest_sha256=release_evidence_manifest_sha,
        primary_output_manifest_sha256=primary_manifest.output_manifest_sha256,
        outcomes=ordered_outcomes,
        dataset=dataset,
        runs=tuple(tuple(run) for run in runs),
        reviews=tuple(reviews),
        canonical_evidence={
            key: dict(value) for key, value in canonical_evidence.items()
        },
        canonical_documents=tuple(canonical_documents),
    )


def current_law_as_of(question: V3Question) -> date | None:
    """Return a historical date, leaving present-law questions to freshness audit checks."""

    return date.fromisoformat(question.as_of) if question.as_of else None
