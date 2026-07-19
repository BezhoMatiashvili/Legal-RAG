from dataclasses import replace

import pytest

from ingest.serving_rollout import (
    ReleaseAuthorization,
    RolloutPolicyError,
    RolloutStage,
    RolloutState,
    SLOBreach,
    SLOBreachReason,
    ServingTuple,
    begin_shadow,
    evaluate_canary,
    promote_canary_to_production,
    promote_shadow_to_five_percent,
    serving_tuple_for_bucket,
)


POLICY_SHA256 = "f" * 64


def _tuple(suffix: str) -> ServingTuple:
    return ServingTuple(
        generation_id=f"generation-{suffix}",
        retriever_version=f"retriever-{suffix}",
        reranker_version=f"reranker-{suffix}",
        translator_version=f"translator-{suffix}",
        generator_version=f"generator-{suffix}",
        prompt_version=f"prompt-{suffix}",
        calibrator_version=f"calibrator-{suffix}",
    )


def _gate(serving: ServingTuple, *, passed=True, check_passed=True):
    return {
        "schema_version": "legal-rag-release-gate/v1",
        "generation_id": serving.generation_id,
        "configuration_hash": serving.sha256,
        "passed": passed,
        "checks": [{"name": "all_release_gates", "passed": check_passed}],
    }


def _authorization(
    serving: ServingTuple,
    stage: RolloutStage,
    *,
    evidence_digit: str,
    policy_sha256: str = POLICY_SHA256,
) -> ReleaseAuthorization:
    return ReleaseAuthorization.from_passing_gate(
        serving,
        target_stage=stage,
        release_gate_result=_gate(serving),
        release_evidence_bundle_sha256=evidence_digit * 64,
        release_policy_sha256=policy_sha256,
    )


def _production(serving: ServingTuple) -> RolloutState:
    return RolloutState.production(
        serving,
        _authorization(
            serving,
            RolloutStage.PRODUCTION,
            evidence_digit="0",
        ),
    )


def _canary():
    stable = _tuple("stable")
    candidate = _tuple("candidate")
    production = _production(stable)
    shadow = begin_shadow(
        production,
        candidate,
        authorization=_authorization(
            candidate,
            RolloutStage.SHADOW,
            evidence_digit="1",
        ),
    )
    canary = promote_shadow_to_five_percent(
        shadow,
        authorization=_authorization(
            candidate,
            RolloutStage.CANARY_5,
            evidence_digit="2",
        ),
    )
    return stable, candidate, canary


def test_shadow_can_promote_only_to_fixed_five_percent():
    stable, candidate, canary = _canary()

    assert canary.stage is RolloutStage.CANARY_5
    assert canary.canary_percent == 5
    assert serving_tuple_for_bucket(canary, 0) is candidate
    assert serving_tuple_for_bucket(canary, 4) is candidate
    assert serving_tuple_for_bucket(canary, 5) is stable
    assert serving_tuple_for_bucket(canary, 99) is stable

    with pytest.raises(RolloutPolicyError, match="canary_percent=5"):
        RolloutState(
            stable=stable,
            stable_authorization=canary.stable_authorization,
            candidate=candidate,
            candidate_authorization=canary.candidate_authorization,
            stage=RolloutStage.CANARY_5,
            canary_percent=10,
        )


def test_any_canary_slo_breach_rolls_back_the_complete_stable_tuple():
    stable, _, canary = _canary()
    breaches = (
        SLOBreach(
            SLOBreachReason.STALE_CURRENT_LAW_ANSWER,
            "current-law answer used a source after its freshness deadline",
        ),
    )

    decision = evaluate_canary(canary, breaches=breaches)

    assert decision.action == "rollback"
    assert decision.serving_tuple is stable
    assert decision.serving_tuple.calibrator_version == "calibrator-stable"
    assert decision.state.stage is RolloutStage.ROLLED_BACK
    assert decision.state.candidate is None
    assert decision.state.rollback_breaches == breaches
    assert serving_tuple_for_bucket(decision.state, 0) is stable


def test_clean_canary_continues_then_promotes_atomically():
    _, candidate, canary = _canary()

    decision = evaluate_canary(canary, breaches=())
    promoted = promote_canary_to_production(
        decision.state,
        authorization=_authorization(
            candidate,
            RolloutStage.PRODUCTION,
            evidence_digit="3",
        ),
    )

    assert decision.action == "continue"
    assert promoted.stable == candidate
    assert promoted.stable_authorization.target_stage is RolloutStage.PRODUCTION
    assert promoted.candidate is None


def test_failed_offline_or_shadow_gate_never_receives_user_traffic():
    production = _production(_tuple("stable"))
    candidate = _tuple("candidate")
    with pytest.raises(RolloutPolicyError, match="did not pass"):
        ReleaseAuthorization.from_passing_gate(
            candidate,
            target_stage=RolloutStage.SHADOW,
            release_gate_result=_gate(candidate, passed=False),
            release_evidence_bundle_sha256="1" * 64,
            release_policy_sha256=POLICY_SHA256,
        )

    shadow = begin_shadow(
        production,
        candidate,
        authorization=_authorization(
            candidate,
            RolloutStage.SHADOW,
            evidence_digit="1",
        ),
    )
    breach = SLOBreach(
        SLOBreachReason.NONDETERMINISTIC_OUTPUT,
        "repeated shadow replay produced different answer hashes",
    )
    rolled_back = promote_shadow_to_five_percent(shadow, breaches=(breach,))

    assert shadow.canary_percent == 0
    assert serving_tuple_for_bucket(shadow, 0) is production.stable
    assert rolled_back.stage is RolloutStage.ROLLED_BACK
    assert rolled_back.stable is production.stable
    assert rolled_back.stable_authorization is production.stable_authorization


def test_authorization_binds_tuple_evidence_policy_and_gate_checksum():
    serving = _tuple("candidate")
    authorization = _authorization(
        serving,
        RolloutStage.SHADOW,
        evidence_digit="1",
    )

    assert authorization.serving_tuple is serving
    assert authorization.serving_tuple_sha256 == serving.sha256
    assert authorization.release_evidence_bundle_sha256 == "1" * 64
    assert authorization.release_policy_sha256 == POLICY_SHA256
    assert len(authorization.release_gate_result_sha256) == 64
    assert len(authorization.authorization_sha256) == 64

    with pytest.raises(RolloutPolicyError, match="checksum"):
        replace(authorization, release_evidence_bundle_sha256="2" * 64)
    with pytest.raises(RolloutPolicyError, match="different serving tuple"):
        authorization.validate_for(_tuple("other"), RolloutStage.SHADOW)


def test_canary_and_production_require_authorization_and_policy_continuity():
    stable = _tuple("stable")
    candidate = _tuple("candidate")
    production = _production(stable)
    shadow = begin_shadow(
        production,
        candidate,
        authorization=_authorization(
            candidate,
            RolloutStage.SHADOW,
            evidence_digit="1",
        ),
    )
    with pytest.raises(RolloutPolicyError, match="requires checksum-bound"):
        promote_shadow_to_five_percent(shadow)
    with pytest.raises(RolloutPolicyError, match="changed policy"):
        promote_shadow_to_five_percent(
            shadow,
            authorization=_authorization(
                candidate,
                RolloutStage.CANARY_5,
                evidence_digit="2",
                policy_sha256="e" * 64,
            ),
        )

    canary = promote_shadow_to_five_percent(
        shadow,
        authorization=_authorization(
            candidate,
            RolloutStage.CANARY_5,
            evidence_digit="2",
        ),
    )
    with pytest.raises(RolloutPolicyError, match="changed policy"):
        promote_canary_to_production(
            canary,
            authorization=_authorization(
                candidate,
                RolloutStage.PRODUCTION,
                evidence_digit="3",
                policy_sha256="e" * 64,
            ),
        )


def test_boolean_only_transition_api_is_not_accepted():
    production = _production(_tuple("stable"))
    candidate = _tuple("candidate")
    with pytest.raises(TypeError):
        begin_shadow(  # type: ignore[call-arg]
            production,
            candidate,
            offline_release_gates_passed=True,
        )


def test_serving_tuple_cannot_have_unpinned_component():
    with pytest.raises(RolloutPolicyError, match="prompt_version"):
        ServingTuple(
            generation_id="generation",
            retriever_version="retriever",
            reranker_version="reranker",
            translator_version="translator",
            generator_version="generator",
            prompt_version=" ",
            calibrator_version="calibrator",
        )

    with pytest.raises(RolloutPolicyError, match="calibrator_version"):
        ServingTuple(
            generation_id="generation",
            retriever_version="retriever",
            reranker_version="reranker",
            translator_version="translator",
            generator_version="generator",
            prompt_version="prompt",
            calibrator_version="",
        )
