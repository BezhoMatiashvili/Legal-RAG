"""Private model bakeoff and production-eligibility contracts.

This module does not download or bless any model.  It records the shortlisted candidates,
requires immutable revisions plus an explicit commercial/data-license attestation, and
implements the accuracy-first lexicographic selection rule.  Published benchmark scores
are intentionally not inputs to production selection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class ModelRole(str, Enum):
    TRANSLATOR = "translator"
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    GENERATOR = "generator"
    VERIFIER = "verifier"


@dataclass(frozen=True)
class ShortlistedModel:
    model_id: str
    role: ModelRole
    purpose: str
    reported_license: str | None = None


SHORTLIST: tuple[ShortlistedModel, ...] = (
    ShortlistedModel(
        "google/madlad400-3b-mt", ModelRole.TRANSLATOR, "primary Georgian MT candidate",
        "Apache-2.0 (requires independent attestation)",
    ),
    ShortlistedModel(
        "facebook/m2m100_1.2B", ModelRole.TRANSLATOR, "smaller MT baseline",
        "MIT (requires independent attestation)",
    ),
    ShortlistedModel(
        "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        ModelRole.TRANSLATOR,
        "strict translation-only legal-context challenger",
    ),
    ShortlistedModel("BAAI/bge-m3", ModelRole.EMBEDDER, "dense and learned-sparse retrieval"),
    ShortlistedModel(
        "BAAI/bge-m3", ModelRole.RERANKER, "native multi-vector scoring challenger"
    ),
    ShortlistedModel(
        "BAAI/bge-reranker-v2-m3", ModelRole.RERANKER, "current cross-encoder baseline"
    ),
    ShortlistedModel("mGTE-multilingual-reranker", ModelRole.RERANKER, "independent reranker"),
    ShortlistedModel("Qwen/Qwen3-Reranker-0.6B", ModelRole.RERANKER, "small private reranker"),
    ShortlistedModel("Qwen/Qwen3-Reranker-4B", ModelRole.RERANKER, "offline accuracy ceiling"),
    ShortlistedModel("Qwen/Qwen3-Embedding-0.6B", ModelRole.EMBEDDER, "small shadow dense retriever"),
    ShortlistedModel("Qwen/Qwen3-Embedding-4B", ModelRole.EMBEDDER, "shadow dense ceiling"),
    ShortlistedModel(
        "Qwen/Qwen3-30B-A3B-GPTQ-Int4", ModelRole.GENERATOR, "primary private pilot"
    ),
    ShortlistedModel("Qwen/Qwen3.6-35B-A3B", ModelRole.GENERATOR, "accuracy challenger"),
    ShortlistedModel(
        "google/gemma-4-26B-A4B-it", ModelRole.GENERATOR, "independent-family drafting challenger"
    ),
    ShortlistedModel(
        "google/gemma-4-26B-A4B-it", ModelRole.VERIFIER, "independent-family verifier"
    ),
)


_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")


@dataclass(frozen=True)
class LicenseAttestation:
    model_id: str
    revision: str
    version: str
    role: ModelRole
    license_id: str
    commercial_use_allowed: bool
    weights_private_deployment_allowed: bool
    training_data_use_allowed: bool
    reviewed_by: str
    reviewed_at: str
    authoritative_source_url: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LicenseAttestation:
        expected = {
            "model_id",
            "revision",
            "version",
            "role",
            "license_id",
            "commercial_use_allowed",
            "weights_private_deployment_allowed",
            "training_data_use_allowed",
            "reviewed_by",
            "reviewed_at",
            "authoritative_source_url",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            unknown = sorted(set(value) - expected)
            raise ValueError(
                "model-license attestation fields mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        try:
            role = ModelRole(value["role"])
        except (TypeError, ValueError) as exc:
            raise ValueError("model-license attestation role is invalid") from exc
        attestation = cls(
            model_id=value["model_id"],
            revision=value["revision"],
            version=value["version"],
            role=role,
            license_id=value["license_id"],
            commercial_use_allowed=value["commercial_use_allowed"],
            weights_private_deployment_allowed=value[
                "weights_private_deployment_allowed"
            ],
            training_data_use_allowed=value["training_data_use_allowed"],
            reviewed_by=value["reviewed_by"],
            reviewed_at=value["reviewed_at"],
            authoritative_source_url=value["authoritative_source_url"],
        )
        attestation.validate()
        return attestation

    @classmethod
    def from_path(cls, path: str | Path) -> LicenseAttestation:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid model-license attestation JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("model-license attestation must be a JSON object")
        return cls.from_dict(value)

    def validate(self) -> None:
        required_strings = (
            "model_id",
            "version",
            "license_id",
            "reviewed_by",
            "reviewed_at",
            "authoritative_source_url",
        )
        missing = [
            name
            for name in required_strings
            if not isinstance(getattr(self, name), str)
            or not getattr(self, name).strip()
        ]
        if not isinstance(self.revision, str) or not _REVISION_RE.fullmatch(self.revision):
            missing.append("revision")
        if not isinstance(self.role, ModelRole):
            missing.append("role")
        boolean_fields = (
            "commercial_use_allowed",
            "weights_private_deployment_allowed",
            "training_data_use_allowed",
        )
        invalid_booleans = [
            name for name in boolean_fields if type(getattr(self, name)) is not bool
        ]
        if invalid_booleans:
            raise ValueError(
                "model-license attestation booleans must be literal bool: "
                + ", ".join(invalid_booleans)
            )
        if missing:
            raise ValueError("incomplete model-license attestation: " + ", ".join(missing))

    def validate_provider(self, provider: object, *, expected_role: ModelRole) -> None:
        """Prove a runtime provider is exactly the model this attestation reviewed."""

        self.validate()
        if (
            self.role is not expected_role
            or getattr(provider, "role", None) is not expected_role
            or getattr(provider, "model_id", None) != self.model_id
            or getattr(provider, "revision", None) != self.revision
            or getattr(provider, "version", None) != self.version
        ):
            raise ValueError(
                "provider role/model/revision/version does not match license attestation"
            )

    @property
    def production_eligible(self) -> bool:
        try:
            self.validate()
        except ValueError:
            return False
        return (
            self.commercial_use_allowed is True
            and self.weights_private_deployment_allowed is True
            and self.training_data_use_allowed is True
        )


@dataclass(frozen=True)
class BakeoffResult:
    configuration_id: str
    role: ModelRole
    model_id: str
    revision: str
    blind_set_hash: str
    generation_id: str
    answered: int
    severe_errors: int
    material_errors: int
    citation_support: float
    temporal_correctness: float
    completeness: float
    correct_refusal: float
    p95_latency_seconds: float
    cost_per_1000_questions: float
    attestation: LicenseAttestation

    def validate(self) -> None:
        if self.answered <= 0:
            raise ValueError("bakeoff result must contain answered outcomes")
        if not 0 <= self.severe_errors <= self.material_errors <= self.answered:
            raise ValueError("invalid severe/material error counts")
        for name in (
            "citation_support", "temporal_correctness", "completeness", "correct_refusal"
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")
        if (
            self.model_id != self.attestation.model_id
            or self.revision != self.attestation.revision
            or self.role is not self.attestation.role
        ):
            raise ValueError("result model identity does not match license attestation")
        if not self.blind_set_hash or not self.generation_id:
            raise ValueError("blind set and generation identities are required")
        self.attestation.validate()

    @property
    def severe_error_rate(self) -> float:
        return self.severe_errors / self.answered

    @property
    def material_error_rate(self) -> float:
        return self.material_errors / self.answered


def select_bakeoff_winner(results: Iterable[BakeoffResult]) -> BakeoffResult:
    """Choose an eligible winner by the plan's accuracy-first lexicographic rule."""

    candidates = list(results)
    if not candidates:
        raise ValueError("no bakeoff results supplied")
    for result in candidates:
        result.validate()
    set_hashes = {result.blind_set_hash for result in candidates}
    generations = {result.generation_id for result in candidates}
    roles = {result.role for result in candidates}
    if len(set_hashes) != 1 or len(generations) != 1 or len(roles) != 1:
        raise ValueError("bakeoff candidates must share blind set, generation, and role")
    eligible = [result for result in candidates if result.attestation.production_eligible]
    if not eligible:
        raise ValueError("no commercially attested private-deployment candidate")
    return min(
        eligible,
        key=lambda result: (
            result.severe_error_rate,
            result.material_error_rate,
            -result.citation_support,
            -result.temporal_correctness,
            -result.completeness,
            -result.correct_refusal,
            result.p95_latency_seconds,
            result.cost_per_1000_questions,
            result.configuration_id,
        ),
    )


HARD_NEGATIVE_KINDS = frozenset(
    {
        "wrong_passage_same_document",
        "similar_law_or_case",
        "neighboring_article",
        "current_or_repealed_version",
    }
)


@dataclass(frozen=True)
class RerankerTrainingExample:
    query_id: str
    document_version_family: str
    positive_evidence_id: str
    positive_is_annotated_evidence: bool
    hard_negative_kinds: frozenset[str]


def validate_reranker_training_examples(
    examples: Iterable[RerankerTrainingExample],
    *,
    blind_document_version_families: set[str] | frozenset[str],
) -> None:
    """Reject chunk-zero positives and blind-family leakage before fine-tuning."""

    rows = list(examples)
    if not rows:
        raise ValueError("reranker training set is empty")
    for row in rows:
        if not row.positive_is_annotated_evidence or not row.positive_evidence_id:
            raise ValueError(f"{row.query_id}: positive must be annotated evidence")
        if row.document_version_family in blind_document_version_families:
            raise ValueError(f"{row.query_id}: blind document/version family leaked into training")
        missing = HARD_NEGATIVE_KINDS - set(row.hard_negative_kinds)
        if missing:
            raise ValueError(
                f"{row.query_id}: missing hard-negative classes {sorted(missing)}"
            )
