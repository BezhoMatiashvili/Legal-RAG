import pytest

from eval.release_gate import (
    RELEASE_CANDIDATE_JSON_SCHEMA,
    ReleaseCandidate,
    ReleasePolicy,
    ReleasePolicySlice,
    ReleaseQuestionOutcome,
    SliceComparison,
    evaluate_release_aggregate_for_diagnostics,
    evaluate_release_candidate,
    release_candidate_from_dict,
    wilson_one_sided,
)


POLICY = ReleasePolicy(
    policy_id="strict-v3-policy",
    blind_question_set_sha256="f" * 64,
    slices=(
        ReleasePolicySlice("napr", baseline=0.70, category="source"),
        ReleasePolicySlice("historical", baseline=0.80),
    ),
)


def _candidate(**overrides):
    output_manifests = ["1" * 64, "2" * 64]
    values = dict(
        candidate_id="rc-a",
        generation_id="gen-20260715",
        configuration_hash="c" * 64,
        answered_outcomes=1000,
        severe_errors=0,
        material_errors=0,
        in_scope_answerable=1800,
        answered_in_scope=1000,
        mechanically_valid_answers=1000,
        mechanical_checks_total=4000,
        mechanical_checks_valid=4000,
        degraded_answers=0,
        stale_current_law_answers=0,
        ambiguous_identity_answers=0,
        validation_failed_answers=0,
        p95_latency_ms=29_000,
        ranking_hashes=("a" * 64, "a" * 64),
        answer_hashes=("b" * 64, "b" * 64),
        slice_comparisons=(
            SliceComparison("napr", baseline=0.70, candidate=0.69, category="source"),
            SliceComparison("historical", baseline=0.80, candidate=0.80),
        ),
        required_slice_names=("napr", "historical"),
        release_policy_sha256=POLICY.sha256,
        metadata={
            "schema_version": "legal-answer-release-review/v1",
            "dataset_id": "sealed-v3",
            "dataset_sha256": "3" * 64,
            "blind_question_set_sha256": POLICY.blind_question_set_sha256,
            "review_manifest_sha256": "4" * 64,
            "mechanical_manifest_sha256": "5" * 64,
            "outcome_manifest_sha256": "0" * 64,
            "canonical_evidence_manifest_sha256": "6" * 64,
            "primary_output_manifest_sha256": output_manifests[0],
            "output_manifest_sha256": output_manifests,
            "repeat_manifest_sha256": ["7" * 64, "8" * 64],
            "release_evidence_manifest_sha256": "9" * 64,
        },
    )
    values.update(overrides)
    return ReleaseCandidate(**values)


def _outcomes(
    candidate: ReleaseCandidate, *, single_cluster: bool = False
) -> tuple[ReleaseQuestionOutcome, ...]:
    rows = []
    for index in range(candidate.answered_outcomes):
        in_scope = index < candidate.answered_in_scope
        rows.append(
            ReleaseQuestionOutcome(
                question_id=f"q-answer-{index}",
                cluster_id=("family-one" if single_cluster else f"family-{index}"),
                answered=True,
                severe_error=index < candidate.severe_errors,
                material_error=index < candidate.material_errors,
                in_scope_answerable=in_scope,
                answered_in_scope=in_scope,
            )
        )
    for index in range(
        candidate.in_scope_answerable - candidate.answered_in_scope
    ):
        rows.append(
            ReleaseQuestionOutcome(
                question_id=f"q-unanswered-{index}",
                cluster_id=(
                    "family-one" if single_cluster else f"unanswered-family-{index}"
                ),
                answered=False,
                severe_error=False,
                material_error=False,
                in_scope_answerable=True,
                answered_in_scope=False,
            )
        )
    return tuple(rows)


def _evaluate(candidate: ReleaseCandidate):
    return evaluate_release_aggregate_for_diagnostics(
        candidate,
        policy=POLICY,
        outcomes=_outcomes(candidate),
    )


def test_passing_release_candidate_and_one_sided_bounds():
    result = _evaluate(_candidate())
    assert result.passed is True
    assert result.severe_error_upper_95 < 0.01
    assert result.material_error_upper_95 < 0.03
    assert result.answerable_coverage > 0.50
    assert result.answerable_coverage_lower_95 < result.answerable_coverage
    assert result.to_dict()["schema_version"].startswith("legal-rag-release-gate/")


def test_error_bounds_and_minimum_answered_fail_closed():
    result = _evaluate(
        _candidate(
            answered_outcomes=999,
            answered_in_scope=999,
            severe_errors=10,
            material_errors=35,
            mechanically_valid_answers=999,
        )
    )
    failed = {check.name for check in result.checks if not check.passed}
    assert {"minimum_answered", "severe_error_bound", "material_error_bound"} <= failed


def test_operational_determinism_latency_and_slice_gates():
    result = _evaluate(
        _candidate(
            degraded_answers=1,
            mechanically_valid_answers=999,
            ranking_hashes=("a" * 64, "d" * 64),
            answer_hashes=("b" * 64, "e" * 64),
            p95_latency_ms=30_001,
            slice_comparisons=(
                SliceComparison("napr", baseline=0.70, candidate=0.679),
            ),
        )
    )
    failed = {check.name for check in result.checks if not check.passed}
    assert {
        "mechanical_validity", "fail_closed_answers", "deterministic_rankings",
        "deterministic_answers", "latency", "slice_regressions",
    } <= failed
    no_slices = _evaluate(
        _candidate(slice_comparisons=(), required_slice_names=())
    )
    assert not next(check for check in no_slices.checks
                    if check.name == "slice_regressions").passed


def test_coverage_gate_and_lower_bound_selection():
    assert not _evaluate(
        _candidate(in_scope_answerable=2100, answered_in_scope=1000)
    ).passed
    low = _candidate(candidate_id="low", in_scope_answerable=1900, answered_in_scope=1000)
    high = _candidate(candidate_id="high", in_scope_answerable=1700, answered_in_scope=1000)
    low_result, high_result = _evaluate(low), _evaluate(high)
    assert high_result.answerable_coverage_lower_95 > low_result.answerable_coverage_lower_95


def test_schema_count_validation_and_wilson_empty_denominator():
    assert wilson_one_sided(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError, match="included"):
        evaluate_release_aggregate_for_diagnostics(
            _candidate(severe_errors=2, material_errors=1),
            policy=POLICY,
            outcomes=_outcomes(_candidate()),
        )
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_release_aggregate_for_diagnostics(
            _candidate(answered_outcomes=-1),
            policy=POLICY,
            outcomes=_outcomes(_candidate()),
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate_release_aggregate_for_diagnostics(
            _candidate(p95_latency_ms=float("nan")),
            policy=POLICY,
            outcomes=_outcomes(_candidate()),
        )


def test_json_schema_and_parser_round_trip():
    candidate = _candidate()
    parsed = release_candidate_from_dict(candidate.to_dict())
    assert parsed == candidate
    assert "ranking_hashes" in RELEASE_CANDIDATE_JSON_SCHEMA["required"]
    with pytest.raises(ValueError, match="unknown"):
        release_candidate_from_dict({**candidate.to_dict(), "surprise": True})
    missing_counter = candidate.to_dict()
    missing_counter.pop("degraded_answers")
    with pytest.raises(TypeError, match="degraded_answers"):
        release_candidate_from_dict(missing_counter)


def test_strict_release_gate_rejects_even_hash_shaped_aggregate():
    with pytest.raises(TypeError, match="ReleaseEvidenceBundle"):
        evaluate_release_candidate(_candidate(), policy=POLICY)


def test_diagnostic_gate_still_validates_aggregate_metadata():
    with pytest.raises(ValueError, match="release evidence metadata"):
        _evaluate(_candidate(metadata={}))
    tampered = dict(_candidate().metadata)
    tampered["primary_output_manifest_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="primary output manifest"):
        _evaluate(_candidate(metadata=tampered))

    wrong_blind_set = dict(_candidate().metadata)
    wrong_blind_set["blind_question_set_sha256"] = "0" * 64
    result = _evaluate(_candidate(metadata=wrong_blind_set))
    assert not next(
        check for check in result.checks if check.name == "release_policy_binding"
    ).passed


def test_cluster_conservative_bound_rejects_insufficient_family_support():
    candidate = _candidate()
    result = evaluate_release_aggregate_for_diagnostics(
        candidate,
        policy=POLICY,
        outcomes=_outcomes(candidate, single_cluster=True),
    )
    assert result.answered_clusters == 1
    assert result.severe_error_upper_95 > 0.01
    assert result.material_error_upper_95 > 0.03
    failed = {check.name for check in result.checks if not check.passed}
    assert {"severe_error_bound", "material_error_bound"} <= failed
