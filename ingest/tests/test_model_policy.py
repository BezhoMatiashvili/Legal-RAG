import dataclasses
from types import SimpleNamespace

import pytest

from ingest.model_policy import (
    HARD_NEGATIVE_KINDS,
    SHORTLIST,
    BakeoffResult,
    LicenseAttestation,
    ModelRole,
    RerankerTrainingExample,
    select_bakeoff_winner,
    validate_reranker_training_examples,
)


def test_shortlist_covers_every_planned_private_bakeoff_family():
    identities = {(candidate.model_id, candidate.role) for candidate in SHORTLIST}
    assert ("google/madlad400-3b-mt", ModelRole.TRANSLATOR) in identities
    assert ("facebook/m2m100_1.2B", ModelRole.TRANSLATOR) in identities
    assert ("BAAI/bge-m3", ModelRole.EMBEDDER) in identities
    assert ("BAAI/bge-m3", ModelRole.RERANKER) in identities
    assert ("BAAI/bge-reranker-v2-m3", ModelRole.RERANKER) in identities
    assert ("Qwen/Qwen3-Reranker-0.6B", ModelRole.RERANKER) in identities
    assert ("Qwen/Qwen3-Embedding-0.6B", ModelRole.EMBEDDER) in identities
    assert ("Qwen/Qwen3-30B-A3B-GPTQ-Int4", ModelRole.GENERATOR) in identities
    assert ("Qwen/Qwen3.6-35B-A3B", ModelRole.GENERATOR) in identities
    assert ("google/gemma-4-26B-A4B-it", ModelRole.VERIFIER) in identities


def _attestation(model="model/a", **overrides):
    values = {
        "model_id": model,
        "revision": "a" * 40,
        "version": "provider-v1",
        "role": ModelRole.GENERATOR,
        "license_id": "Apache-2.0",
        "commercial_use_allowed": True,
        "weights_private_deployment_allowed": True,
        "training_data_use_allowed": True,
        "reviewed_by": "legal-reviewer",
        "reviewed_at": "2026-07-15",
        "authoritative_source_url": "https://example.test/license",
    }
    values.update(overrides)
    return LicenseAttestation(**values)


def _result(config, model="model/a", **overrides):
    values = {
        "configuration_id": config,
        "role": ModelRole.GENERATOR,
        "model_id": model,
        "revision": "a" * 40,
        "blind_set_hash": "b" * 64,
        "generation_id": "generation-20260715",
        "answered": 1000,
        "severe_errors": 1,
        "material_errors": 5,
        "citation_support": 0.99,
        "temporal_correctness": 0.99,
        "completeness": 0.9,
        "correct_refusal": 0.95,
        "p95_latency_seconds": 20.0,
        "cost_per_1000_questions": 10.0,
        "attestation": _attestation(model),
    }
    values.update(overrides)
    return BakeoffResult(**values)


def test_selection_prioritizes_legal_error_before_latency_or_cost():
    fast_but_wrong = _result(
        "fast", severe_errors=2, p95_latency_seconds=2, cost_per_1000_questions=1
    )
    slower_accurate = _result(
        "accurate", severe_errors=1, p95_latency_seconds=29, cost_per_1000_questions=50
    )
    assert select_bakeoff_winner([fast_but_wrong, slower_accurate]) == slower_accurate


def test_selection_uses_support_then_latency_after_equal_error():
    lower_support = _result("a", citation_support=0.95, p95_latency_seconds=10)
    higher_support = _result("b", citation_support=0.99, p95_latency_seconds=30)
    assert select_bakeoff_winner([lower_support, higher_support]) == higher_support


def test_noncommercial_or_unreviewed_models_are_ineligible():
    blocked = _result(
        "blocked",
        severe_errors=0,
        attestation=_attestation(commercial_use_allowed=False),
    )
    eligible = _result("eligible", severe_errors=1)
    assert select_bakeoff_winner([blocked, eligible]) == eligible
    with pytest.raises(ValueError, match="no commercially attested"):
        select_bakeoff_winner([blocked])


@pytest.mark.parametrize("value", ["true", 1, 0, None])
def test_license_attestation_requires_literal_booleans(value):
    attestation = _attestation(commercial_use_allowed=value)
    with pytest.raises(ValueError, match="literal bool"):
        attestation.validate()
    assert attestation.production_eligible is False


def test_license_attestation_matches_exact_provider_identity():
    attestation = _attestation()
    provider = SimpleNamespace(
        role=ModelRole.GENERATOR,
        model_id=attestation.model_id,
        revision=attestation.revision,
        version=attestation.version,
    )
    attestation.validate_provider(provider, expected_role=ModelRole.GENERATOR)
    for field, value in (
        ("role", "generator"),
        ("model_id", "model/b"),
        ("revision", "b" * 40),
        ("version", "provider-v2"),
    ):
        mismatched = SimpleNamespace(**vars(provider))
        setattr(mismatched, field, value)
        with pytest.raises(ValueError, match="does not match"):
            attestation.validate_provider(
                mismatched, expected_role=ModelRole.GENERATOR
            )


def test_bakeoff_requires_same_blind_set_and_generation():
    with pytest.raises(ValueError, match="share blind set"):
        select_bakeoff_winner([
            _result("a"),
            _result("b", blind_set_hash="c" * 64),
        ])


def test_reranker_training_requires_annotated_positive_hard_negatives_and_no_leakage():
    row = RerankerTrainingExample(
        "q1", "family-train", "ev1", True, HARD_NEGATIVE_KINDS
    )
    validate_reranker_training_examples(
        [row], blind_document_version_families={"family-blind"}
    )
    with pytest.raises(ValueError, match="annotated evidence"):
        validate_reranker_training_examples(
            [dataclasses.replace(row, positive_is_annotated_evidence=False)],
            blind_document_version_families=set(),
        )
    with pytest.raises(ValueError, match="leaked"):
        validate_reranker_training_examples(
            [row], blind_document_version_families={"family-train"}
        )
    with pytest.raises(ValueError, match="missing hard-negative"):
        validate_reranker_training_examples(
            [dataclasses.replace(row, hard_negative_kinds=frozenset())],
            blind_document_version_families=set(),
        )
