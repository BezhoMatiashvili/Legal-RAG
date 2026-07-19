"""Pure shadow/canary policy for atomic legal-RAG serving tuples.

This module performs no deployment or alias mutation.  It makes the allowed transition
and rollback decision explicit so an orchestrator cannot promote one model component at a
time.  A serving tuple always binds the generation, retrieval stack, translation,
generation, and prompt revisions that were evaluated together.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Literal


RELEASE_AUTHORIZATION_SCHEMA_VERSION = "legal-rag-release-authorization/v1"


class RolloutPolicyError(ValueError):
    """A requested rollout transition violates the guarded release policy."""


class RolloutStage(StrEnum):
    OFFLINE = "offline_replay"
    SHADOW = "shadow"
    CANARY_5 = "canary_5_percent"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


class SLOBreachReason(StrEnum):
    """Machine-readable reasons that require atomic rollback of a candidate tuple."""

    SEVERE_ERROR_BOUND = "severe_error_bound"
    MATERIAL_ERROR_BOUND = "material_error_bound"
    ANSWERABLE_COVERAGE_FLOOR = "answerable_coverage_floor"
    EVIDENCE_ID_INTEGRITY = "evidence_id_integrity"
    QUOTATION_OR_HASH_INTEGRITY = "quotation_or_hash_integrity"
    DEGRADED_PIPELINE_ANSWER = "degraded_pipeline_answer"
    STALE_CURRENT_LAW_ANSWER = "stale_current_law_answer"
    AMBIGUOUS_IDENTITY_ANSWER = "ambiguous_identity_answer"
    FAILED_VALIDATION_ANSWER = "failed_validation_answer"
    SOURCE_OR_RISK_SLICE_REGRESSION = "source_or_risk_slice_regression"
    NONDETERMINISTIC_OUTPUT = "nondeterministic_output"
    P95_LATENCY = "p95_latency"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"


@dataclass(frozen=True)
class ServingTuple:
    """Indivisible deployment identity evaluated and rolled back as one value."""

    generation_id: str
    retriever_version: str
    reranker_version: str
    translator_version: str
    generator_version: str
    prompt_version: str
    calibrator_version: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise RolloutPolicyError(
                    f"serving tuple {field} must be pinned and non-empty"
                )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        """Canonical checksum used by release-candidate ``configuration_hash``."""

        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class ReleaseAuthorization:
    """Checksum-bound proof that a tuple passed evidence-backed release policy.

    The authorization is stage-specific because shadow observations can become evidence
    for canary entry, and canary observations can become evidence for production entry.
    It contains no mutable gate Boolean: construction requires the structured passing gate
    result whose configuration hash is exactly ``serving_tuple.sha256``.
    """

    schema_version: str
    target_stage: RolloutStage
    serving_tuple: ServingTuple
    serving_tuple_sha256: str
    release_evidence_bundle_sha256: str
    release_policy_sha256: str
    release_gate_result_sha256: str
    authorization_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_AUTHORIZATION_SCHEMA_VERSION:
            raise RolloutPolicyError("unsupported release authorization schema")
        if self.target_stage not in {
            RolloutStage.SHADOW,
            RolloutStage.CANARY_5,
            RolloutStage.PRODUCTION,
        }:
            raise RolloutPolicyError(
                "release authorization target must be shadow, canary, or production"
            )
        for field in (
            "serving_tuple_sha256",
            "release_evidence_bundle_sha256",
            "release_policy_sha256",
            "release_gate_result_sha256",
            "authorization_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.serving_tuple_sha256 != self.serving_tuple.sha256:
            raise RolloutPolicyError(
                "release authorization serving tuple checksum is invalid"
            )
        expected = _authorization_sha256(
            target_stage=self.target_stage,
            serving_tuple=self.serving_tuple,
            release_evidence_bundle_sha256=self.release_evidence_bundle_sha256,
            release_policy_sha256=self.release_policy_sha256,
            release_gate_result_sha256=self.release_gate_result_sha256,
        )
        if self.authorization_sha256 != expected:
            raise RolloutPolicyError("release authorization checksum is invalid")

    @classmethod
    def from_passing_gate(
        cls,
        serving_tuple: ServingTuple,
        *,
        target_stage: RolloutStage,
        release_gate_result: Mapping[str, Any] | Any,
        release_evidence_bundle_sha256: str,
        release_policy_sha256: str,
    ) -> ReleaseAuthorization:
        """Issue a stage authorization from one structured passing gate result."""

        _require_sha256(
            release_evidence_bundle_sha256,
            field="release_evidence_bundle_sha256",
        )
        _require_sha256(release_policy_sha256, field="release_policy_sha256")
        gate = _gate_result_dict(release_gate_result)
        if gate.get("passed") is not True:
            raise RolloutPolicyError("release gate result did not pass")
        if gate.get("generation_id") != serving_tuple.generation_id:
            raise RolloutPolicyError(
                "release gate generation does not match serving tuple"
            )
        if gate.get("configuration_hash") != serving_tuple.sha256:
            raise RolloutPolicyError(
                "release gate configuration hash does not match serving tuple"
            )
        checks = gate.get("checks")
        if not isinstance(checks, (list, tuple)) or not checks:
            raise RolloutPolicyError("release gate result has no mechanical checks")
        if any(_check_passed(check) is not True for check in checks):
            raise RolloutPolicyError(
                "release gate result contains a failed or malformed check"
            )
        gate_sha256 = hashlib.sha256(_canonical_json(gate)).hexdigest()
        authorization_sha256 = _authorization_sha256(
            target_stage=target_stage,
            serving_tuple=serving_tuple,
            release_evidence_bundle_sha256=release_evidence_bundle_sha256,
            release_policy_sha256=release_policy_sha256,
            release_gate_result_sha256=gate_sha256,
        )
        return cls(
            schema_version=RELEASE_AUTHORIZATION_SCHEMA_VERSION,
            target_stage=target_stage,
            serving_tuple=serving_tuple,
            serving_tuple_sha256=serving_tuple.sha256,
            release_evidence_bundle_sha256=release_evidence_bundle_sha256,
            release_policy_sha256=release_policy_sha256,
            release_gate_result_sha256=gate_sha256,
            authorization_sha256=authorization_sha256,
        )

    def validate_for(
        self,
        serving_tuple: ServingTuple,
        target_stage: RolloutStage,
        *,
        release_policy_sha256: str | None = None,
    ) -> None:
        """Require exact tuple/stage binding and optionally policy continuity."""

        self.__post_init__()
        if self.serving_tuple != serving_tuple:
            raise RolloutPolicyError(
                "release authorization belongs to a different serving tuple"
            )
        if self.target_stage is not target_stage:
            raise RolloutPolicyError(
                f"release authorization does not permit {target_stage.value}"
            )
        if (
            release_policy_sha256 is not None
            and self.release_policy_sha256 != release_policy_sha256
        ):
            raise RolloutPolicyError(
                "release authorization changed policy during rollout"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_stage": self.target_stage.value,
            "serving_tuple": self.serving_tuple.to_dict(),
            "serving_tuple_sha256": self.serving_tuple_sha256,
            "release_evidence_bundle_sha256": self.release_evidence_bundle_sha256,
            "release_policy_sha256": self.release_policy_sha256,
            "release_gate_result_sha256": self.release_gate_result_sha256,
            "authorization_sha256": self.authorization_sha256,
        }


@dataclass(frozen=True)
class SLOBreach:
    reason: SLOBreachReason
    detail: str

    def __post_init__(self) -> None:
        if not self.detail or not self.detail.strip():
            raise RolloutPolicyError("SLO breach detail must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason.value, "detail": self.detail}


@dataclass(frozen=True)
class RolloutState:
    """Stable and candidate identities plus the only permitted traffic stage."""

    stable: ServingTuple
    stable_authorization: ReleaseAuthorization
    candidate: ServingTuple | None
    candidate_authorization: ReleaseAuthorization | None
    stage: RolloutStage
    canary_percent: int
    rollback_breaches: tuple[SLOBreach, ...] = ()

    def __post_init__(self) -> None:
        self.stable_authorization.validate_for(
            self.stable,
            RolloutStage.PRODUCTION,
        )
        expected_percent = 5 if self.stage is RolloutStage.CANARY_5 else 0
        if self.canary_percent != expected_percent:
            raise RolloutPolicyError(
                f"{self.stage.value} requires canary_percent={expected_percent}"
            )
        if self.stage in {RolloutStage.SHADOW, RolloutStage.CANARY_5}:
            if self.candidate is None:
                raise RolloutPolicyError(
                    f"{self.stage.value} requires a candidate tuple"
                )
            if self.candidate == self.stable:
                raise RolloutPolicyError(
                    "candidate tuple must differ from stable tuple"
                )
            if self.candidate_authorization is None:
                raise RolloutPolicyError(
                    f"{self.stage.value} requires candidate release authorization"
                )
            self.candidate_authorization.validate_for(
                self.candidate,
                self.stage,
            )
        elif self.candidate is not None:
            raise RolloutPolicyError(
                f"{self.stage.value} cannot retain a candidate tuple"
            )
        elif self.candidate_authorization is not None:
            raise RolloutPolicyError(
                f"{self.stage.value} cannot retain candidate release authorization"
            )
        if self.stage is RolloutStage.ROLLED_BACK and not self.rollback_breaches:
            raise RolloutPolicyError(
                "rolled-back state requires at least one SLO breach"
            )
        if self.stage is not RolloutStage.ROLLED_BACK and self.rollback_breaches:
            raise RolloutPolicyError(
                "rollback breaches are valid only in rolled-back state"
            )

    @classmethod
    def production(
        cls,
        serving_tuple: ServingTuple,
        authorization: ReleaseAuthorization,
    ) -> RolloutState:
        authorization.validate_for(serving_tuple, RolloutStage.PRODUCTION)
        return cls(
            stable=serving_tuple,
            stable_authorization=authorization,
            candidate=None,
            candidate_authorization=None,
            stage=RolloutStage.PRODUCTION,
            canary_percent=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stable": self.stable.to_dict(),
            "stable_authorization": self.stable_authorization.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "candidate_authorization": (
                None
                if self.candidate_authorization is None
                else self.candidate_authorization.to_dict()
            ),
            "stage": self.stage.value,
            "canary_percent": self.canary_percent,
            "rollback_breaches": [
                breach.to_dict() for breach in self.rollback_breaches
            ],
        }


@dataclass(frozen=True)
class RollbackDecision:
    action: Literal["continue", "rollback"]
    state: RolloutState

    @property
    def serving_tuple(self) -> ServingTuple:
        """The complete stable tuple to route after applying this decision."""

        return self.state.stable


def begin_shadow(
    state: RolloutState,
    candidate: ServingTuple,
    *,
    authorization: ReleaseAuthorization,
) -> RolloutState:
    """Attach a candidate for zero-user-traffic shadow evaluation."""

    if state.stage not in {RolloutStage.PRODUCTION, RolloutStage.ROLLED_BACK}:
        raise RolloutPolicyError("shadow can begin only from production or rolled_back")
    authorization.validate_for(candidate, RolloutStage.SHADOW)
    return RolloutState(
        stable=state.stable,
        stable_authorization=state.stable_authorization,
        candidate=candidate,
        candidate_authorization=authorization,
        stage=RolloutStage.SHADOW,
        canary_percent=0,
    )


def promote_shadow_to_five_percent(
    state: RolloutState,
    *,
    authorization: ReleaseAuthorization | None = None,
    breaches: tuple[SLOBreach, ...] = (),
) -> RolloutState:
    """Promote only a clean shadow tuple, and only to the fixed five-percent canary."""

    if state.stage is not RolloutStage.SHADOW:
        raise RolloutPolicyError("five-percent canary requires a shadow candidate")
    if breaches:
        return _rolled_back(state, breaches)
    if state.candidate is None or state.candidate_authorization is None:
        raise RolloutPolicyError("shadow state lost candidate release authorization")
    if authorization is None:
        raise RolloutPolicyError(
            "five-percent canary requires checksum-bound release authorization"
        )
    authorization.validate_for(
        state.candidate,
        RolloutStage.CANARY_5,
        release_policy_sha256=state.candidate_authorization.release_policy_sha256,
    )
    return RolloutState(
        stable=state.stable,
        stable_authorization=state.stable_authorization,
        candidate=state.candidate,
        candidate_authorization=authorization,
        stage=RolloutStage.CANARY_5,
        canary_percent=5,
    )


def evaluate_canary(
    state: RolloutState,
    *,
    breaches: tuple[SLOBreach, ...],
) -> RollbackDecision:
    """Rollback the entire candidate tuple on any accuracy, integrity, or SLO breach."""

    if state.stage is not RolloutStage.CANARY_5:
        raise RolloutPolicyError("canary evaluation requires a five-percent canary")
    if breaches:
        return RollbackDecision(action="rollback", state=_rolled_back(state, breaches))
    return RollbackDecision(action="continue", state=state)


def promote_canary_to_production(
    state: RolloutState,
    *,
    authorization: ReleaseAuthorization,
) -> RolloutState:
    """Atomically make the already-evaluated candidate the new stable tuple."""

    if state.stage is not RolloutStage.CANARY_5 or state.candidate is None:
        raise RolloutPolicyError("production promotion requires a five-percent canary")
    if state.candidate_authorization is None:
        raise RolloutPolicyError("canary state lost candidate release authorization")
    authorization.validate_for(
        state.candidate,
        RolloutStage.PRODUCTION,
        release_policy_sha256=state.candidate_authorization.release_policy_sha256,
    )
    return RolloutState.production(state.candidate, authorization)


def serving_tuple_for_bucket(state: RolloutState, bucket: int) -> ServingTuple:
    """Route a stable 0..99 bucket without ever mixing tuple components."""

    if isinstance(bucket, bool) or not isinstance(bucket, int) or not 0 <= bucket <= 99:
        raise RolloutPolicyError("traffic bucket must be an integer from 0 through 99")
    if state.stage is RolloutStage.CANARY_5 and bucket < 5:
        if state.candidate is None:  # defensive; RolloutState already enforces this
            raise RolloutPolicyError("canary state lost its candidate")
        return state.candidate
    return state.stable


def _rolled_back(
    state: RolloutState,
    breaches: tuple[SLOBreach, ...],
) -> RolloutState:
    if not breaches:
        raise RolloutPolicyError("rollback requires at least one SLO breach")
    # Clearing candidate and retaining the exact stable object makes component-wise
    # rollback unrepresentable through this policy API.
    return replace(
        state,
        candidate=None,
        candidate_authorization=None,
        stage=RolloutStage.ROLLED_BACK,
        canary_percent=0,
        rollback_breaches=breaches,
    )


def _gate_result_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise RolloutPolicyError(
                "release gate result must be a mapping or expose to_dict()"
            )
        raw = to_dict()
        if not isinstance(raw, Mapping):
            raise RolloutPolicyError("release gate to_dict() did not return a mapping")
        result = dict(raw)
    if (
        not isinstance(result.get("schema_version"), str)
        or not result["schema_version"]
    ):
        raise RolloutPolicyError("release gate result schema_version is missing")
    return result


def _check_passed(value: Any) -> bool | None:
    if isinstance(value, Mapping):
        passed = value.get("passed")
    else:
        passed = getattr(value, "passed", None)
    return passed if isinstance(passed, bool) else None


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RolloutPolicyError(f"{field} must be a full lowercase SHA-256")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RolloutPolicyError(f"release evidence is not strict JSON: {exc}") from exc
    return encoded.encode("utf-8")


def _authorization_sha256(
    *,
    target_stage: RolloutStage,
    serving_tuple: ServingTuple,
    release_evidence_bundle_sha256: str,
    release_policy_sha256: str,
    release_gate_result_sha256: str,
) -> str:
    if target_stage not in {
        RolloutStage.SHADOW,
        RolloutStage.CANARY_5,
        RolloutStage.PRODUCTION,
    }:
        raise RolloutPolicyError("invalid release authorization target stage")
    material = {
        "schema_version": RELEASE_AUTHORIZATION_SCHEMA_VERSION,
        "target_stage": target_stage.value,
        "serving_tuple": serving_tuple.to_dict(),
        "serving_tuple_sha256": serving_tuple.sha256,
        "release_evidence_bundle_sha256": release_evidence_bundle_sha256,
        "release_policy_sha256": release_policy_sha256,
        "release_gate_result_sha256": release_gate_result_sha256,
    }
    return hashlib.sha256(_canonical_json(material)).hexdigest()


__all__ = [
    "RELEASE_AUTHORIZATION_SCHEMA_VERSION",
    "ReleaseAuthorization",
    "RollbackDecision",
    "RolloutPolicyError",
    "RolloutStage",
    "RolloutState",
    "SLOBreach",
    "SLOBreachReason",
    "ServingTuple",
    "begin_shadow",
    "evaluate_canary",
    "promote_canary_to_production",
    "promote_shadow_to_five_percent",
    "serving_tuple_for_bucket",
]
