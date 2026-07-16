import json

import pytest

from ingest.answer_feedback import (
    AnswerFeedback,
    FeedbackCategory,
    FeedbackSeverity,
    ServingTuple,
    append_feedback,
)
from ingest.serving_rollout import (
    ReleaseAuthorization,
    RolloutPolicyError,
    RolloutStage,
)


POLICY_SHA256 = "f" * 64


def _serving():
    return ServingTuple(
        "generation-20260715",
        "embed@rev",
        "rerank@rev",
        "translate@rev",
        "generator@rev",
        "prompt-v1",
        "calibrator-v1",
    )


def _authorization(serving, stage=RolloutStage.PRODUCTION):
    gate = {
        "schema_version": "legal-rag-release-gate/v1",
        "generation_id": serving.generation_id,
        "configuration_hash": serving.sha256,
        "passed": True,
        "checks": [{"name": "release", "passed": True}],
    }
    return ReleaseAuthorization.from_passing_gate(
        serving,
        target_stage=stage,
        release_gate_result=gate,
        release_evidence_bundle_sha256="e" * 64,
        release_policy_sha256=POLICY_SHA256,
    )


def _feedback(**overrides):
    serving = overrides.pop("serving", _serving())
    values = {
        "trace_id": "a" * 64,
        "category": FeedbackCategory.WRONG_VERSION,
        "severity": FeedbackSeverity.SEVERE,
        "serving": serving,
        "release_authorization": _authorization(serving),
        "traffic_stage": RolloutStage.PRODUCTION,
        "question_language": "ka",
        "reviewer_id": "reviewer-1",
    }
    values.update(overrides)
    return AnswerFeedback(**values)


def test_feedback_records_required_category_and_atomic_serving_tuple(tmp_path):
    path = tmp_path / "feedback.jsonl"
    append_feedback(_feedback(), path)
    row = json.loads(path.read_text())
    assert row["category"] == "wrong_version"
    assert row["serving"]["generation_id"] == "generation-20260715"
    assert row["serving"]["translator_version"] == "translate@rev"
    assert row["release_authorization"]["target_stage"] == "production"
    assert len(row["release_authorization"]["authorization_sha256"]) == 64
    assert row["traffic_stage"] == "production"
    assert row["question_language"] == "ka"
    assert row["created_at"].endswith("+00:00")
    assert path.stat().st_mode & 0o777 == 0o600


def test_feedback_rejects_incomplete_identity_and_public_log(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    record = _feedback(
        category=FeedbackCategory.INVALID_CITATION,
        severity=FeedbackSeverity.MATERIAL,
        reviewer_id="reviewer",
    )
    with pytest.raises(PermissionError, match="owner-only"):
        append_feedback(record, path)

    bad = _feedback(
        trace_id="not-a-trace",
        category=FeedbackCategory.WRONG_PASSAGE,
        severity=FeedbackSeverity.MATERIAL,
        reviewer_id="reviewer",
    )
    with pytest.raises(ValueError, match="trace_id"):
        bad.normalized()


def test_feedback_accepts_georgian_and_english_canary_or_production():
    serving = _serving()
    english_canary = _feedback(
        serving=serving,
        release_authorization=_authorization(serving, RolloutStage.CANARY_5),
        traffic_stage=RolloutStage.CANARY_5,
        question_language="EN",
    ).normalized()

    assert english_canary.question_language == "en"
    assert english_canary.serving == serving


def test_feedback_rejects_wrong_tuple_stage_or_unsupported_language():
    serving = _serving()
    other = ServingTuple(
        "other-generation",
        "embed@other",
        "rerank@other",
        "translate@other",
        "generator@other",
        "prompt-other",
        "calibrator-other",
    )
    with pytest.raises(ValueError, match="does not match"):
        _feedback(
            serving=serving,
            release_authorization=_authorization(other),
        ).normalized()
    with pytest.raises(ValueError, match="canary or production"):
        _feedback(
            serving=serving,
            release_authorization=_authorization(serving, RolloutStage.SHADOW),
            traffic_stage=RolloutStage.SHADOW,
        ).normalized()
    with pytest.raises(ValueError, match="Georgian or English"):
        _feedback(question_language="fr").normalized()


def test_feedback_serving_tuple_requires_non_null_translation():
    with pytest.raises(RolloutPolicyError, match="translator_version"):
        ServingTuple(  # type: ignore[arg-type]
            "generation",
            "retriever",
            "reranker",
            None,
            "generator",
            "prompt",
            "calibrator",
        )
