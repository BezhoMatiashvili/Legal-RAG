"""Hermetic reversible-promotion tests; the backend is in-memory and never uses Qdrant."""

from __future__ import annotations

import json
import stat
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest.promotion import (
    CollectionInspection,
    PromotionError,
    PromotionLockedError,
    PromotionPlan,
    PromotionPlanExists,
    PromotionPreconditionError,
    execute_promotion,
    load_promotion_state,
    physical_collection_name,
    promotion_lock,
    write_promotion_plan,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import promote_generation  # noqa: E402

GENERATION_ID = "gen-20260713"
PREVIOUS = "georgian_legal__gen_previous"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _plan() -> PromotionPlan:
    return PromotionPlan.from_dict(
        {
            "schema_version": 1,
            "promotion_id": "promotion-20260713",
            "generation_id": GENERATION_ID,
            "manifest_sha256": SHA_A,
            "verification_report_sha256": "f" * 64,
            "snapshot_ref": "immutable://snapshots/gen-20260713.snapshot",
            "snapshot_sha256": SHA_B,
            "serving_alias": "georgian_legal",
            "physical_collection": physical_collection_name(GENERATION_ID),
            "expected_collection": {
                "payload_schema_version": 1,
                "points_count": 2,
                "dense_name": "dense",
                "dense_dimension": 1024,
                "distance": "cosine",
                "sparse_name": "sparse",
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": "a" * 40,
                "tokenizer_model": "BAAI/bge-m3",
                "tokenizer_revision": "b" * 40,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_revision": "c" * 40,
                "vector_space_id": SHA_C,
                "chunking_fingerprint": SHA_D,
                "document_header": True,
                "retrieval_fingerprint": SHA_E,
            },
            "created_at": "2026-07-13T12:00:00Z",
            "created_by": "test-operator",
        }
    )


def _inspection(plan: PromotionPlan) -> CollectionInspection:
    expected = plan.expected_collection
    return CollectionInspection(
        name=plan.physical_collection,
        payload_schema_version=expected.payload_schema_version,
        points_count=expected.points_count,
        dense_name=expected.dense_name,
        dense_dimension=expected.dense_dimension,
        distance=expected.distance,
        sparse_name=expected.sparse_name,
        generation_id=plan.generation_id,
        manifest_sha256=plan.manifest_sha256,
        embedding_model=expected.embedding_model,
        embedding_revision=expected.embedding_revision,
        tokenizer_model=expected.tokenizer_model,
        tokenizer_revision=expected.tokenizer_revision,
        reranker_model=expected.reranker_model,
        reranker_revision=expected.reranker_revision,
        vector_space_id=expected.vector_space_id,
        chunking_fingerprint=expected.chunking_fingerprint,
        document_header=expected.document_header,
        retrieval_fingerprint=expected.retrieval_fingerprint,
        optimizer_status="green",
        integrity_ok=True,
        verification_provenance_sha256="f" * 64,
    )


class FakeBackend:
    def __init__(self, plan: PromotionPlan, *, alias=PREVIOUS):
        self.plan = plan
        self.alias = alias
        self.inspection = _inspection(plan)
        self.calls: list[tuple] = []
        self.smoke_ok = True
        self.ready = {PREVIOUS: True, plan.physical_collection: True}
        self.fail_switch_once: str | None = None

    def restore_candidate(self, plan):
        assert plan == self.plan
        self.calls.append(("restore", plan.physical_collection))

    def wait_for_green(self, collection, timeout_seconds):
        self.calls.append(("green", collection, timeout_seconds))
        return self.inspection

    def smoke(self, collection):
        self.calls.append(("smoke", collection))
        return self.smoke_ok

    def readiness(self, collection):
        self.calls.append(("readiness", collection))
        return self.ready.get(collection, False)

    def alias_target(self, alias):
        assert alias == "georgian_legal"
        self.calls.append(("alias", alias))
        return self.alias

    def switch_alias(self, alias, collection):
        assert alias == "georgian_legal"
        self.calls.append(("switch", collection))
        self.alias = collection
        if self.fail_switch_once == collection:
            self.fail_switch_once = None
            raise RuntimeError("injected post-switch crash")


@pytest.fixture()
def operation(tmp_path):
    plan = _plan()
    directory = tmp_path / "operations"
    plan_path = directory / "plan.json"
    state_path = directory / "state.json"
    write_promotion_plan(plan_path, plan)
    return plan, plan_path, state_path


def _execute(operation, backend, *, forward=False):
    _, plan_path, state_path = operation
    return execute_promotion(
        plan_path,
        state_path,
        backend,
        forward_after_rollback=forward,
        now=lambda: datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )


def test_default_promotion_proves_rollback_and_leaves_previous_serving(operation):
    plan, plan_path, state_path = operation
    backend = FakeBackend(plan)

    state = _execute(operation, backend)

    assert state.rollback_proven
    assert not state.final_forward_switched
    assert state.phase == "rollback_proven"
    assert state.candidate_verification_sha256 == "f" * 64
    assert backend.alias == PREVIOUS
    assert [call for call in backend.calls if call[0] == "switch"] == [
        ("switch", plan.physical_collection),
        ("switch", PREVIOUS),
    ]
    assert _mode(plan_path) == 0o600
    assert _mode(state_path) == 0o600
    assert _mode(state_path.parent) == 0o700
    assert load_promotion_state(state_path) == state


def test_explicit_final_forward_occurs_only_after_rollback_proof(operation):
    plan, _, _ = operation
    backend = FakeBackend(plan)

    state = _execute(operation, backend, forward=True)

    assert state.rollback_proven and state.final_forward_switched
    assert state.phase == "completed_forward"
    assert backend.alias == plan.physical_collection
    assert [call for call in backend.calls if call[0] == "switch"] == [
        ("switch", plan.physical_collection),
        ("switch", PREVIOUS),
        ("switch", plan.physical_collection),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_schema_version", 2),
        ("points_count", 3),
        ("dense_dimension", 768),
        ("embedding_revision", "f" * 40),
        ("tokenizer_model", "other-tokenizer"),
        ("reranker_model", "other-reranker"),
        ("reranker_revision", "f" * 40),
        ("vector_space_id", "f" * 64),
        ("document_header", False),
        ("retrieval_fingerprint", "f" * 64),
        ("optimizer_status", "yellow"),
        ("integrity_ok", False),
    ],
)
def test_exact_candidate_mismatch_fails_before_alias_switch(operation, field, value):
    plan, _, state_path = operation
    backend = FakeBackend(plan)
    backend.inspection = replace(backend.inspection, **{field: value})

    with pytest.raises(PromotionPreconditionError, match="compatibility failed"):
        _execute(operation, backend)

    assert backend.alias == PREVIOUS
    assert not any(call[0] == "switch" for call in backend.calls)
    state = load_promotion_state(state_path)
    assert state is not None and state.phase == "failed"


@pytest.mark.parametrize(
    "failure", ["smoke", "candidate_readiness", "rollback_readiness"]
)
def test_smoke_and_readiness_fail_closed_before_switch(operation, failure):
    plan, _, _ = operation
    backend = FakeBackend(plan)
    if failure == "smoke":
        backend.smoke_ok = False
    elif failure == "candidate_readiness":
        backend.ready[plan.physical_collection] = False
    else:
        backend.ready[PREVIOUS] = False

    with pytest.raises(PromotionPreconditionError):
        _execute(operation, backend)

    assert backend.alias == PREVIOUS
    assert not any(call[0] == "switch" for call in backend.calls)


def test_missing_alias_target_blocks_legacy_first_migration(operation):
    plan, _, state_path = operation
    backend = FakeBackend(plan, alias=None)
    with pytest.raises(PromotionPreconditionError, match="maintenance migration"):
        _execute(operation, backend)
    assert not state_path.exists()
    assert not any(call[0] == "restore" for call in backend.calls)


def test_post_switch_error_is_immediately_contained_and_reconciled(operation):
    plan, _, state_path = operation
    backend = FakeBackend(plan)
    backend.fail_switch_once = plan.physical_collection

    with pytest.raises(RuntimeError, match="post-switch crash"):
        _execute(operation, backend)
    assert backend.alias == PREVIOUS
    failed = load_promotion_state(state_path)
    assert failed is not None and failed.rollback_proven
    assert "emergency_rollback_alias_switched" in failed.events

    recovered = _execute(operation, backend)
    assert recovered.rollback_proven
    assert backend.alias == PREVIOUS
    assert "rollback_proof_reconfirmed" in recovered.events


def test_final_forward_readiness_failure_switches_back_to_proven_target(operation):
    plan, _, state_path = operation
    backend = FakeBackend(plan)
    readiness_calls = 0

    def readiness(collection):
        nonlocal readiness_calls
        if collection == plan.physical_collection:
            readiness_calls += 1
            return readiness_calls == 1
        return True

    backend.readiness = readiness
    with pytest.raises(PromotionPreconditionError, match="after final forward"):
        _execute(operation, backend, forward=True)
    assert backend.alias == PREVIOUS
    state = load_promotion_state(state_path)
    assert state is not None and state.rollback_proven
    assert "final_forward_readiness_rollback" in state.events


def test_error_after_final_forward_switch_is_immediately_rolled_back(
    operation,
):
    plan, _, state_path = operation
    backend = FakeBackend(plan)
    original_switch = backend.switch_alias
    candidate_switches = 0

    def crash_after_second_candidate_switch(alias, collection):
        nonlocal candidate_switches
        original_switch(alias, collection)
        if collection == plan.physical_collection:
            candidate_switches += 1
            if candidate_switches == 2:
                raise RuntimeError("injected final-forward crash")

    backend.switch_alias = crash_after_second_candidate_switch
    with pytest.raises(RuntimeError, match="final-forward crash"):
        _execute(operation, backend, forward=True)
    assert backend.alias == PREVIOUS
    failed = load_promotion_state(state_path)
    assert failed is not None and failed.rollback_proven
    assert not failed.final_forward_switched
    assert "emergency_rollback_alias_switched" in failed.events

    recovered = _execute(operation, backend)
    assert backend.alias == PREVIOUS
    assert recovered.phase == "rollback_proven"
    assert "rollback_proof_reconfirmed" in recovered.events


def test_local_lock_rejects_concurrent_publisher(operation):
    plan, _, state_path = operation
    backend = FakeBackend(plan)
    lock_path = state_path.parent / ".promotion.lock"
    with promotion_lock(lock_path):
        with pytest.raises(PromotionLockedError):
            _execute(operation, backend)
    assert not any(call[0] == "restore" for call in backend.calls)


def test_plan_is_owner_only_and_never_overwritten(operation):
    plan, plan_path, _ = operation
    before = plan_path.read_bytes()
    with pytest.raises(PromotionPlanExists):
        write_promotion_plan(plan_path, plan)
    assert plan_path.read_bytes() == before


def test_plan_rejects_mutable_model_revisions():
    payload = _plan().to_dict()
    payload["expected_collection"]["reranker_revision"] = "latest"

    with pytest.raises(PromotionError, match="immutable"):
        PromotionPlan.from_dict(payload)


def test_plan_rejects_blank_model_identity():
    payload = _plan().to_dict()
    payload["expected_collection"]["tokenizer_model"] = "   "

    with pytest.raises(PromotionError, match="non-empty model identity"):
        PromotionPlan.from_dict(payload)


def test_non_private_plan_is_rejected_before_backend_use(operation):
    plan, plan_path, _ = operation
    backend = FakeBackend(plan)
    plan_path.chmod(0o644)
    with pytest.raises(PromotionError, match="not owner-only"):
        _execute(operation, backend)
    assert not any(call[0] == "restore" for call in backend.calls)


def test_impossible_persisted_state_flags_are_rejected(operation):
    plan, _, state_path = operation
    backend = FakeBackend(plan)
    _execute(operation, backend)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["candidate_verified"] = False
    payload["candidate_verification_sha256"] = None
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path.chmod(0o600)
    with pytest.raises(PromotionError, match="impossible flag sequence"):
        load_promotion_state(state_path)


def test_cli_apply_requires_flag_and_environment_before_loading_backend(
    operation,
):
    _, plan_path, state_path = operation
    common = [
        "apply",
        "--plan",
        str(plan_path),
        "--state",
        str(state_path),
        "--backend-factory",
        "does_not_exist:factory",
    ]
    with pytest.raises(SystemExit):
        promote_generation.main(common, environ={})
    with pytest.raises(SystemExit):
        promote_generation.main(
            [*common, "--apply", "--forward-after-rollback"],
            environ={"PROMOTION_APPROVED": "1"},
        )
    assert not state_path.exists()
