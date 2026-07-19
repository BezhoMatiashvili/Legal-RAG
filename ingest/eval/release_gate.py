"""Mechanical release gate for selective legal-answer accuracy.

This module deliberately contains no model judgment.  It consumes the adjudicated counts
and deterministic validator/trace outputs of a release-candidate evaluation and answers one
question: is the candidate eligible to progress to shadow/canary deployment?
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from math import isfinite, sqrt
from statistics import NormalDist

SCHEMA_VERSION = "legal-rag-release-gate/v1"
RELEASE_EVIDENCE_SCHEMA_VERSION = "legal-answer-release-review/v1"
ONE_SIDED_CONFIDENCE = 0.95
SEVERE_ERROR_UPPER_LIMIT = 0.01
MATERIAL_ERROR_UPPER_LIMIT = 0.03
MIN_ANSWERED_OUTCOMES = 1000
MIN_ANSWERABLE_COVERAGE = 0.50
MAX_P95_LATENCY_MS = 30_000.0
MAX_SLICE_REGRESSION = 0.02

RELEASE_CANDIDATE_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id", "generation_id", "configuration_hash", "answered_outcomes",
        "severe_errors", "material_errors", "in_scope_answerable", "answered_in_scope",
        "mechanically_valid_answers", "mechanical_checks_total", "mechanical_checks_valid",
        "degraded_answers", "stale_current_law_answers", "ambiguous_identity_answers",
        "validation_failed_answers",
        "p95_latency_ms", "ranking_hashes", "answer_hashes", "slice_comparisons",
        "required_slice_names",
        "release_policy_sha256",
        "metadata",
    ],
    "properties": {
        "candidate_id": {"type": "string", "minLength": 1},
        "generation_id": {"type": "string", "minLength": 1},
        "configuration_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        **{
            name: {"type": "integer", "minimum": 0}
            for name in (
                "answered_outcomes", "severe_errors", "material_errors",
                "in_scope_answerable", "answered_in_scope", "mechanically_valid_answers",
                "mechanical_checks_total", "mechanical_checks_valid", "degraded_answers",
                "stale_current_law_answers", "ambiguous_identity_answers",
                "validation_failed_answers",
            )
        },
        "p95_latency_ms": {"type": "number", "minimum": 0},
        "ranking_hashes": {
            "type": "array", "minItems": 2,
            "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "answer_hashes": {
            "type": "array", "minItems": 2,
            "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "required_slice_names": {
            "type": "array", "minItems": 1, "items": {"type": "string"},
        },
        "release_policy_sha256": {
            "type": "string", "pattern": "^[0-9a-f]{64}$",
        },
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "dataset_id",
                "dataset_sha256",
                "blind_question_set_sha256",
                "review_manifest_sha256",
                "mechanical_manifest_sha256",
                "outcome_manifest_sha256",
                "canonical_evidence_manifest_sha256",
                "primary_output_manifest_sha256",
                "output_manifest_sha256",
                "repeat_manifest_sha256",
                "release_evidence_manifest_sha256",
            ],
            "properties": {
                "schema_version": {"const": RELEASE_EVIDENCE_SCHEMA_VERSION},
                "dataset_id": {"type": "string", "minLength": 1},
                **{
                    name: {"type": "string", "pattern": "^[0-9a-f]{64}$"}
                    for name in (
                        "dataset_sha256",
                        "blind_question_set_sha256",
                        "review_manifest_sha256",
                        "mechanical_manifest_sha256",
                        "outcome_manifest_sha256",
                        "canonical_evidence_manifest_sha256",
                        "primary_output_manifest_sha256",
                        "release_evidence_manifest_sha256",
                    )
                },
                "output_manifest_sha256": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "repeat_manifest_sha256": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
        },
        "slice_comparisons": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "baseline", "candidate"],
                "properties": {
                    "name": {"type": "string"},
                    "baseline": {"type": "number"},
                    "candidate": {"type": "number"},
                    "higher_is_better": {"type": "boolean"},
                    "category": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
}


@dataclass(frozen=True)
class SliceComparison:
    """One source or high-risk slice compared with the accepted baseline."""

    name: str
    baseline: float
    candidate: float
    higher_is_better: bool = True
    category: str = "high_risk"

    @property
    def regression(self) -> float:
        return (
            self.baseline - self.candidate
            if self.higher_is_better
            else self.candidate - self.baseline
        )


@dataclass(frozen=True)
class ReleasePolicySlice:
    """One externally predeclared source/high-risk slice and accepted baseline."""

    name: str
    baseline: float
    higher_is_better: bool = True
    category: str = "high_risk"

    def validate(self) -> None:
        if not self.name or not self.category:
            raise ValueError("release policy slice identity is required")
        if not isfinite(self.baseline) or not 0.0 <= self.baseline <= 1.0:
            raise ValueError("release policy slice baseline must be within [0,1]")


@dataclass(frozen=True)
class ReleasePolicy:
    """Immutable policy input controlled outside the candidate evaluation output."""

    policy_id: str
    blind_question_set_sha256: str
    slices: tuple[ReleasePolicySlice, ...]

    def validate(self) -> None:
        if not self.policy_id or not _is_sha256(self.blind_question_set_sha256):
            raise ValueError("release policy identity and blind question-set hash are required")
        if not self.slices:
            raise ValueError("release policy requires source/high-risk slices")
        names = [item.name for item in self.slices]
        if len(names) != len(set(names)):
            raise ValueError("release policy slice names must be unique")
        for item in self.slices:
            item.validate()

    @property
    def sha256(self) -> str:
        self.validate()
        material = {
            "policy_id": self.policy_id,
            "blind_question_set_sha256": self.blind_question_set_sha256,
            "slices": [asdict(item) for item in self.slices],
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def required_slice_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.slices)


@dataclass(frozen=True)
class ReleaseCandidate:
    candidate_id: str
    generation_id: str
    configuration_hash: str
    answered_outcomes: int
    severe_errors: int
    material_errors: int
    in_scope_answerable: int
    answered_in_scope: int
    mechanically_valid_answers: int
    mechanical_checks_total: int
    mechanical_checks_valid: int
    degraded_answers: int
    stale_current_law_answers: int
    ambiguous_identity_answers: int
    validation_failed_answers: int
    p95_latency_ms: float = 0.0
    ranking_hashes: tuple[str, ...] = ()
    answer_hashes: tuple[str, ...] = ()
    slice_comparisons: tuple[SliceComparison, ...] = ()
    required_slice_names: tuple[str, ...] = ()
    release_policy_sha256: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not self.candidate_id or not self.generation_id or not self.configuration_hash:
            raise ValueError("candidate_id, generation_id, and configuration_hash are required")
        if not _is_sha256(self.configuration_hash):
            raise ValueError("configuration_hash must be a full lowercase SHA-256")
        if not _is_sha256(self.release_policy_sha256):
            raise ValueError("release_policy_sha256 must be a full lowercase SHA-256")
        integer_fields = (
            "answered_outcomes", "severe_errors", "material_errors",
            "in_scope_answerable", "answered_in_scope", "mechanically_valid_answers",
            "mechanical_checks_total", "mechanical_checks_valid",
            "degraded_answers", "stale_current_law_answers", "ambiguous_identity_answers",
            "validation_failed_answers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.severe_errors > self.material_errors:
            raise ValueError("severe_errors must be included in material_errors")
        if self.material_errors > self.answered_outcomes:
            raise ValueError("material_errors cannot exceed answered_outcomes")
        if self.answered_in_scope > self.in_scope_answerable:
            raise ValueError("answered_in_scope cannot exceed in_scope_answerable")
        if self.answered_in_scope > self.answered_outcomes:
            raise ValueError("answered_in_scope cannot exceed answered_outcomes")
        if self.mechanically_valid_answers > self.answered_outcomes:
            raise ValueError("mechanically_valid_answers cannot exceed answered_outcomes")
        if self.mechanical_checks_valid > self.mechanical_checks_total:
            raise ValueError("mechanical_checks_valid cannot exceed mechanical_checks_total")
        if self.mechanical_checks_total < self.answered_outcomes:
            raise ValueError("mechanical_checks_total must cover every answered outcome")
        for name, hashes in (
            ("ranking_hashes", self.ranking_hashes),
            ("answer_hashes", self.answer_hashes),
        ):
            if len(hashes) < 2 or any(not _is_sha256(value) for value in hashes):
                raise ValueError(f"{name} must contain at least two full SHA-256 values")
        if not isfinite(self.p95_latency_ms) or self.p95_latency_ms < 0:
            raise ValueError("p95_latency_ms must be finite and non-negative")
        slice_names = [comparison.name for comparison in self.slice_comparisons]
        if len(slice_names) != len(set(slice_names)):
            raise ValueError("slice comparison names must be unique")
        for comparison in self.slice_comparisons:
            if not comparison.name or not all(
                isfinite(value) and 0.0 <= value <= 1.0
                for value in (comparison.baseline, comparison.candidate)
            ):
                raise ValueError("slice scores must be finite proportions in [0, 1]")
        _validate_release_evidence_metadata(
            self.metadata,
            repeat_count=len(self.ranking_hashes),
        )


@dataclass(frozen=True)
class ReleaseQuestionOutcome:
    """One adjudicated blind output retained for cluster-aware release inference."""

    question_id: str
    cluster_id: str
    answered: bool
    severe_error: bool
    material_error: bool
    in_scope_answerable: bool
    answered_in_scope: bool

    def validate(self) -> None:
        if not self.question_id or not self.cluster_id:
            raise ValueError("release outcome question_id and cluster_id are required")
        for name in (
            "answered",
            "severe_error",
            "material_error",
            "in_scope_answerable",
            "answered_in_scope",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"release outcome {name} must be boolean")
        if self.severe_error and not self.material_error:
            raise ValueError("a severe release outcome must also be material")
        if (self.severe_error or self.material_error) and not self.answered:
            raise ValueError("release errors are defined only among answered outcomes")
        if self.answered_in_scope and not (
            self.answered and self.in_scope_answerable
        ):
            raise ValueError(
                "answered_in_scope requires an answered, in-scope answerable question"
            )

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: object
    requirement: str


@dataclass(frozen=True)
class ReleaseGateResult:
    schema_version: str
    candidate_id: str
    generation_id: str
    configuration_hash: str
    passed: bool
    severe_error_rate: float
    severe_error_upper_95: float
    material_error_rate: float
    material_error_upper_95: float
    answered_clusters: int
    severe_error_clusters: int
    material_error_clusters: int
    answerable_coverage: float
    answerable_coverage_lower_95: float
    checks: tuple[GateCheck, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_one_sided(
    successes: int, total: int, *, confidence: float = ONE_SIDED_CONFIDENCE
) -> tuple[float, float]:
    """One-sided Wilson lower/upper bounds for a binomial proportion."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if total <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half = z * sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _identical_repeats(hashes: tuple[str, ...]) -> bool:
    return len(hashes) >= 2 and bool(hashes[0]) and len(set(hashes)) == 1


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_release_evidence_metadata(
    metadata: object,
    *,
    repeat_count: int,
) -> None:
    """Require hashes derived from immutable blind-output/review source artifacts.

    The aggregate gate cannot independently recreate human review decisions.  It can and
    does refuse aggregate-only records that omit the evidence-producing layer's sealed
    dataset, per-run output, review, mechanical, and canonical-ledger manifests.
    """

    required = {
        "schema_version",
        "dataset_id",
        "dataset_sha256",
        "blind_question_set_sha256",
        "review_manifest_sha256",
        "mechanical_manifest_sha256",
        "outcome_manifest_sha256",
        "canonical_evidence_manifest_sha256",
        "primary_output_manifest_sha256",
        "output_manifest_sha256",
        "repeat_manifest_sha256",
        "release_evidence_manifest_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        missing = sorted(required - set(metadata)) if isinstance(metadata, dict) else sorted(required)
        unknown = sorted(set(metadata) - required) if isinstance(metadata, dict) else []
        raise ValueError(
            "release evidence metadata fields mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if metadata["schema_version"] != RELEASE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("release evidence metadata schema_version is unsupported")
    if not isinstance(metadata["dataset_id"], str) or not metadata["dataset_id"].strip():
        raise ValueError("release evidence metadata dataset_id is required")
    scalar_hashes = (
        "dataset_sha256",
        "blind_question_set_sha256",
        "review_manifest_sha256",
        "mechanical_manifest_sha256",
        "outcome_manifest_sha256",
        "canonical_evidence_manifest_sha256",
        "primary_output_manifest_sha256",
        "release_evidence_manifest_sha256",
    )
    if any(not _is_sha256(metadata[name]) for name in scalar_hashes):
        raise ValueError("release evidence metadata contains an invalid SHA-256")
    output_hashes = metadata["output_manifest_sha256"]
    repeat_hashes = metadata["repeat_manifest_sha256"]
    for name, hashes in (
        ("output_manifest_sha256", output_hashes),
        ("repeat_manifest_sha256", repeat_hashes),
    ):
        if (
            not isinstance(hashes, list)
            or len(hashes) != repeat_count
            or repeat_count < 2
            or any(not _is_sha256(value) for value in hashes)
        ):
            raise ValueError(
                f"release evidence metadata {name} must bind every repeated run"
            )
    if metadata["primary_output_manifest_sha256"] != output_hashes[0]:
        raise ValueError("primary output manifest must be the first repeated-run manifest")


def _validate_outcomes(
    candidate: ReleaseCandidate,
    outcomes: tuple[ReleaseQuestionOutcome, ...],
) -> tuple[int, int, int]:
    """Bind aggregate counters to immutable rows and return cluster error counts.

    A family is conservatively counted as erroneous when any answered question in that
    document/version family has the corresponding error.  The release bound is the worse of
    the ordinary query-level Wilson bound and this cluster-level any-error Wilson bound.
    This prevents hundreds of correlated questions about one authority from masquerading as
    hundreds of independent observations.
    """

    if not outcomes:
        raise ValueError("strict release evaluation requires per-question outcomes")
    question_ids: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, ReleaseQuestionOutcome):
            raise TypeError("release outcomes must be ReleaseQuestionOutcome values")
        outcome.validate()
        if outcome.question_id in question_ids:
            raise ValueError("release outcome question IDs must be unique")
        question_ids.add(outcome.question_id)
    counters = {
        "answered_outcomes": sum(item.answered for item in outcomes),
        "severe_errors": sum(item.severe_error for item in outcomes),
        "material_errors": sum(item.material_error for item in outcomes),
        "in_scope_answerable": sum(item.in_scope_answerable for item in outcomes),
        "answered_in_scope": sum(item.answered_in_scope for item in outcomes),
    }
    mismatches = {
        name: {"candidate": getattr(candidate, name), "derived": derived}
        for name, derived in counters.items()
        if getattr(candidate, name) != derived
    }
    if mismatches:
        raise ValueError(f"release aggregate/outcome counter mismatch: {mismatches}")

    answered_clusters: dict[str, list[ReleaseQuestionOutcome]] = {}
    for outcome in outcomes:
        if outcome.answered:
            answered_clusters.setdefault(outcome.cluster_id, []).append(outcome)
    severe_clusters = sum(
        any(item.severe_error for item in rows)
        for rows in answered_clusters.values()
    )
    material_clusters = sum(
        any(item.material_error for item in rows)
        for rows in answered_clusters.values()
    )
    return len(answered_clusters), severe_clusters, material_clusters


def _evaluate_verified_candidate(
    candidate: ReleaseCandidate,
    *,
    policy: ReleasePolicy,
    outcomes: tuple[ReleaseQuestionOutcome, ...],
) -> ReleaseGateResult:
    candidate.validate()
    policy.validate()
    answered_clusters, severe_clusters, material_clusters = _validate_outcomes(
        candidate, outcomes
    )
    answered = candidate.answered_outcomes
    severe_rate = candidate.severe_errors / answered if answered else 1.0
    material_rate = candidate.material_errors / answered if answered else 1.0
    _severe_lower, query_severe_upper = wilson_one_sided(
        candidate.severe_errors, answered
    )
    _material_lower, query_material_upper = wilson_one_sided(
        candidate.material_errors, answered
    )
    _cluster_severe_lower, cluster_severe_upper = wilson_one_sided(
        severe_clusters, answered_clusters
    )
    _cluster_material_lower, cluster_material_upper = wilson_one_sided(
        material_clusters, answered_clusters
    )
    severe_upper = max(query_severe_upper, cluster_severe_upper)
    material_upper = max(query_material_upper, cluster_material_upper)
    coverage = (
        candidate.answered_in_scope / candidate.in_scope_answerable
        if candidate.in_scope_answerable else 0.0
    )
    coverage_lower, _coverage_upper = wilson_one_sided(
        candidate.answered_in_scope, candidate.in_scope_answerable
    )

    compared = {comparison.name: comparison for comparison in candidate.slice_comparisons}
    compared_names = set(compared)
    missing_slices = sorted(set(policy.required_slice_names) - compared_names)
    unexpected_slices = sorted(compared_names - set(policy.required_slice_names))
    policy_mismatches = []
    for expected in policy.slices:
        actual = compared.get(expected.name)
        if actual is not None and (
            actual.baseline != expected.baseline
            or actual.higher_is_better is not expected.higher_is_better
            or actual.category != expected.category
        ):
            policy_mismatches.append(expected.name)
    policy_bound = (
        candidate.release_policy_sha256 == policy.sha256
        and candidate.metadata["blind_question_set_sha256"]
        == policy.blind_question_set_sha256
        and candidate.required_slice_names == policy.required_slice_names
        and not missing_slices
        and not unexpected_slices
        and not policy_mismatches
    )
    regressions = {
        comparison.name: comparison.regression
        for comparison in candidate.slice_comparisons
        if comparison.regression > MAX_SLICE_REGRESSION
    }
    zero_risk_counts = {
        "degraded": candidate.degraded_answers,
        "stale_current_law": candidate.stale_current_law_answers,
        "ambiguous_identity": candidate.ambiguous_identity_answers,
        "validation_failed": candidate.validation_failed_answers,
    }
    checks = (
        GateCheck(
            "release_policy_binding",
            policy_bound,
            {
                "candidate_policy_sha256": candidate.release_policy_sha256,
                "expected_policy_sha256": policy.sha256,
                "candidate_blind_question_set_sha256": candidate.metadata[
                    "blind_question_set_sha256"
                ],
                "expected_blind_question_set_sha256": policy.blind_question_set_sha256,
                "missing_required": missing_slices,
                "unexpected": unexpected_slices,
                "mismatched_baselines": policy_mismatches,
            },
            "candidate is bound to the external blind-set and slice policy",
        ),
        GateCheck(
            "minimum_answered", answered >= MIN_ANSWERED_OUTCOMES, answered,
            f">= {MIN_ANSWERED_OUTCOMES} answered outcomes",
        ),
        GateCheck(
            "severe_error_bound", severe_upper < SEVERE_ERROR_UPPER_LIMIT,
            {
                "conservative_upper": severe_upper,
                "query_upper": query_severe_upper,
                "cluster_any_error_upper": cluster_severe_upper,
                "answered_clusters": answered_clusters,
                "error_clusters": severe_clusters,
            },
            f"cluster-conservative one-sided 95% upper bound < {SEVERE_ERROR_UPPER_LIMIT:.1%}",
        ),
        GateCheck(
            "material_error_bound", material_upper < MATERIAL_ERROR_UPPER_LIMIT,
            {
                "conservative_upper": material_upper,
                "query_upper": query_material_upper,
                "cluster_any_error_upper": cluster_material_upper,
                "answered_clusters": answered_clusters,
                "error_clusters": material_clusters,
            },
            f"cluster-conservative one-sided 95% upper bound < {MATERIAL_ERROR_UPPER_LIMIT:.1%}",
        ),
        GateCheck(
            "answerable_coverage", coverage >= MIN_ANSWERABLE_COVERAGE,
            {"point": coverage, "lower_95": coverage_lower},
            f"point coverage >= {MIN_ANSWERABLE_COVERAGE:.0%}; rank passers by lower_95",
        ),
        GateCheck(
            "mechanical_validity",
            answered > 0
            and candidate.mechanically_valid_answers == answered
            and candidate.mechanical_checks_total > 0
            and candidate.mechanical_checks_valid == candidate.mechanical_checks_total,
            {
                "valid_answers": candidate.mechanically_valid_answers,
                "answered": answered,
                "valid_evidence_checks": candidate.mechanical_checks_valid,
                "evidence_checks": candidate.mechanical_checks_total,
            },
            "100% of answered outcomes and evidence ID/quote/hash checks mechanically valid",
        ),
        GateCheck(
            "fail_closed_answers", all(value == 0 for value in zero_risk_counts.values()),
            zero_risk_counts,
            "zero degraded, stale, ambiguous-identity, or validation-failed answers",
        ),
        GateCheck(
            "deterministic_rankings", _identical_repeats(candidate.ranking_hashes),
            list(candidate.ranking_hashes), "at least two identical ranking hashes",
        ),
        GateCheck(
            "deterministic_answers", _identical_repeats(candidate.answer_hashes),
            list(candidate.answer_hashes), "at least two identical answer hashes",
        ),
        GateCheck(
            "latency", candidate.p95_latency_ms <= MAX_P95_LATENCY_MS,
            candidate.p95_latency_ms, f"private-profile p95 <= {MAX_P95_LATENCY_MS:.0f} ms",
        ),
        GateCheck(
            "slice_regressions",
            policy_bound and not regressions,
            {
                "regressions": regressions,
                "missing_required": missing_slices,
                "unexpected": unexpected_slices,
            },
            f"no source/high-risk slice regression > {MAX_SLICE_REGRESSION:.0%}",
        ),
    )
    return ReleaseGateResult(
        schema_version=SCHEMA_VERSION,
        candidate_id=candidate.candidate_id,
        generation_id=candidate.generation_id,
        configuration_hash=candidate.configuration_hash,
        passed=all(check.passed for check in checks),
        severe_error_rate=severe_rate,
        severe_error_upper_95=severe_upper,
        material_error_rate=material_rate,
        material_error_upper_95=material_upper,
        answered_clusters=answered_clusters,
        severe_error_clusters=severe_clusters,
        material_error_clusters=material_clusters,
        answerable_coverage=coverage,
        answerable_coverage_lower_95=coverage_lower,
        checks=checks,
    )


def evaluate_release_candidate(evidence_bundle: object, *, policy: ReleasePolicy) -> ReleaseGateResult:
    """Strict release entry point accepting only a recomputable evidence bundle.

    Hash-shaped strings on a hand-authored aggregate are not evidence.  The local import
    avoids the module cycle while ensuring callers cannot substitute an object that merely
    implements a similarly named method.
    """

    from .answer_release_review import ReleaseEvidenceBundle

    if not isinstance(evidence_bundle, ReleaseEvidenceBundle):
        raise TypeError(
            "strict release evaluation requires a ReleaseEvidenceBundle, not an aggregate"
        )
    candidate, outcomes = evidence_bundle.verified_material(policy=policy)
    return _evaluate_verified_candidate(candidate, policy=policy, outcomes=outcomes)


def evaluate_release_aggregate_for_diagnostics(
    candidate: ReleaseCandidate,
    *,
    policy: ReleasePolicy,
    outcomes: tuple[ReleaseQuestionOutcome, ...],
) -> ReleaseGateResult:
    """Evaluate already-derived math without conferring strict release eligibility.

    This helper exists for report migration and unit diagnostics.  Promotion code must call
    :func:`evaluate_release_candidate` with the source-bearing evidence bundle.
    """

    return _evaluate_verified_candidate(candidate, policy=policy, outcomes=outcomes)


def release_candidate_from_dict(data: dict) -> ReleaseCandidate:
    """Parse a JSON-compatible release-candidate record, rejecting unknown fields."""
    field_names = {item.name for item in fields(ReleaseCandidate)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ValueError(f"unknown release candidate fields: {unknown}")
    material = dict(data)
    material["ranking_hashes"] = tuple(material.get("ranking_hashes", ()))
    material["answer_hashes"] = tuple(material.get("answer_hashes", ()))
    material["required_slice_names"] = tuple(material.get("required_slice_names", ()))
    material["slice_comparisons"] = tuple(
        item if isinstance(item, SliceComparison) else SliceComparison(**item)
        for item in material.get("slice_comparisons", ())
    )
    candidate = ReleaseCandidate(**material)
    candidate.validate()
    return candidate


def select_release_candidate(
    candidates: tuple[object, ...] | list[object],
    *,
    policy: ReleasePolicy,
) -> tuple[object, ReleaseGateResult] | None:
    """Choose the passing candidate with the highest coverage lower confidence bound."""
    evaluated = [
        (candidate, evaluate_release_candidate(candidate, policy=policy))
        for candidate in candidates
    ]
    passing = [(candidate, result) for candidate, result in evaluated if result.passed]
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: (
            item[1].answerable_coverage_lower_95,
            item[1].answerable_coverage,
            item[1].candidate_id,
        ),
    )
