import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qdrant_client import models

import ingest.qdrant_promotion as promotion_backend
from ingest.promotion import (
    PromotionPlan,
    PromotionPreconditionError,
    physical_collection_name,
)
from ingest.qdrant_promotion import (
    CHECKS_FACTORY_ENV,
    CandidateVerificationProof,
    QdrantPromotionBackend,
    make_qdrant_promotion_backend,
)
from ingest.release_inputs import (
    GENERATION_ID as FROZEN_CANDIDATE_GENERATION_ID,
    PHYSICAL_COLLECTION as FROZEN_CANDIDATE_PHYSICAL_COLLECTION,
)


def _proof(tmp_path, *, ok=True):
    report = tmp_path / "candidate-report.json"
    provenance = tmp_path / "candidate-provenance.json"
    report.write_bytes(b"report\n")
    provenance.write_bytes(b"provenance\n")
    report.chmod(0o600)
    provenance.chmod(0o600)
    return CandidateVerificationProof(
        ok=ok,
        report_path=report,
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        provenance_path=provenance,
        provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),
    )


def _plan(snapshot) -> PromotionPlan:
    blob = snapshot.read_bytes()
    generation_id = "gen-20260713"
    return PromotionPlan.from_dict(
        {
            "schema_version": 1,
            "promotion_id": "promotion-20260713",
            "generation_id": generation_id,
            "manifest_sha256": "a" * 64,
            "verification_report_sha256": "f" * 64,
            "snapshot_ref": snapshot.as_uri(),
            "snapshot_sha256": hashlib.sha256(blob).hexdigest(),
            "serving_alias": "georgian_legal",
            "physical_collection": physical_collection_name(generation_id),
            "expected_collection": {
                "payload_schema_version": 1,
                "points_count": 2,
                "dense_name": "dense",
                "dense_dimension": 1024,
                "distance": "cosine",
                "sparse_name": "sparse",
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": "1" * 40,
                "tokenizer_model": "BAAI/bge-m3",
                "tokenizer_revision": "2" * 40,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_revision": "3" * 40,
                "vector_space_id": "b" * 64,
                "chunking_fingerprint": "c" * 64,
                "document_header": True,
                "retrieval_fingerprint_revision": 2,
                "retrieval_fingerprint": "d" * 64,
            },
            "created_at": "2026-07-13T12:00:00Z",
            "created_by": "test-operator",
        }
    )


def _info(plan, *, points=None, status="green", optimizer="ok"):
    expected = plan.expected_collection
    return {
        "points_count": expected.points_count if points is None else points,
        "status": status,
        "optimizer_status": optimizer,
        "config": {
            "params": {
                "vectors": {
                    expected.dense_name: {
                        "size": expected.dense_dimension,
                        "distance": expected.distance,
                    }
                },
                "sparse_vectors": {expected.sparse_name: {}},
            }
        },
    }


class _FakeClient:
    def __init__(self, plan, *, present=False, points=None):
        self.plan = plan
        self.collections = {}
        if present:
            self.collections[plan.physical_collection] = _info(plan, points=points)
        self.aliases = {"georgian_legal": "georgian_legal__gen_previous"}
        self.recover_calls = []
        self.alias_updates = []

    def collection_exists(self, collection):
        return collection in self.collections

    def recover_snapshot(self, **kwargs):
        self.recover_calls.append(kwargs)
        self.collections[kwargs["collection_name"]] = _info(self.plan)
        return True

    def get_collection(self, collection):
        return self.collections[collection]

    def count(self, *, collection_name, count_filter, exact):
        assert collection_name == self.plan.physical_collection
        assert exact is True
        keys = {condition.key for condition in count_filter.must}
        assert {
            "schema_version",
            "generation_id",
            "embedding_model",
            "embedding_revision",
            "tokenizer_model",
            "tokenizer_revision",
            "reranker_model",
            "reranker_revision",
            "vector_space_id",
            "chunking_fingerprint",
            "document_header",
            "retrieval_fingerprint_revision",
            "retrieval_fingerprint",
        } == keys
        return SimpleNamespace(count=self.plan.expected_collection.points_count)

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name=alias, collection_name=collection)
                for alias, collection in self.aliases.items()
            ]
        )

    def update_collection_aliases(self, *, change_aliases_operations):
        self.alias_updates.append(change_aliases_operations)
        assert len(change_aliases_operations) == 2
        delete, create = change_aliases_operations
        assert isinstance(delete, models.DeleteAliasOperation)
        assert isinstance(create, models.CreateAliasOperation)
        alias = create.create_alias.alias_name
        assert delete.delete_alias.alias_name == alias
        self.aliases[alias] = create.create_alias.collection_name
        return True


def test_restores_absent_candidate_with_bound_checksum_and_exact_inspection(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"immutable qdrant snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan)
    backend = QdrantPromotionBackend(
        client,
        integrity_check=lambda _plan, _client: _proof(tmp_path),
        smoke_check=lambda _collection, _client: True,
        readiness_check=lambda _collection, _client: True,
        poll_interval_seconds=0,
    )

    backend.restore_candidate(plan)
    inspection = backend.wait_for_green(plan.physical_collection, 1)

    assert len(client.recover_calls) == 1
    call = client.recover_calls[0]
    assert call["collection_name"] == plan.physical_collection
    assert call["checksum"] == plan.snapshot_sha256
    assert call["priority"] is models.SnapshotPriority.SNAPSHOT
    assert inspection.integrity_ok is True
    assert inspection.optimizer_status == "green"
    assert backend.smoke(plan.physical_collection) is True
    assert backend.readiness(plan.physical_collection) is True


def test_frozen_candidate_restore_refuses_before_client_access(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"immutable qdrant snapshot")
    plan = replace(
        _plan(snapshot),
        generation_id=FROZEN_CANDIDATE_GENERATION_ID,
        physical_collection=FROZEN_CANDIDATE_PHYSICAL_COLLECTION,
    )

    class NoClientAccess:
        def __getattr__(self, name):
            pytest.fail(f"frozen-candidate refusal must precede client access: {name}")

    with pytest.raises(PromotionPreconditionError, match="cannot enter a promotion plan"):
        QdrantPromotionBackend(NoClientAccess()).restore_candidate(plan)


def test_snapshot_checksum_mismatch_fails_before_recovery(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = replace(_plan(snapshot), snapshot_sha256="0" * 64)
    client = _FakeClient(plan)

    with pytest.raises(PromotionPreconditionError, match="SHA-256 mismatch"):
        QdrantPromotionBackend(client).restore_candidate(plan)

    assert client.recover_calls == []


def test_existing_incompatible_candidate_is_never_overwritten(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True, points=1)

    with pytest.raises(PromotionPreconditionError, match="will not be overwritten"):
        QdrantPromotionBackend(client).restore_candidate(plan)

    assert client.recover_calls == []


def test_existing_in_progress_recovery_is_waited_not_resubmitted(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    client.collections[plan.physical_collection] = _info(
        plan, status="yellow", optimizer="ok"
    )
    backend = QdrantPromotionBackend(
        client,
        integrity_check=lambda _plan, _client: _proof(tmp_path),
        poll_interval_seconds=0,
    )

    backend.restore_candidate(plan)
    client.collections[plan.physical_collection] = _info(plan)
    inspection = backend.wait_for_green(plan.physical_collection, 1)

    assert client.recover_calls == []
    assert inspection.integrity_ok is True


def test_alias_switch_is_one_atomic_batch_and_never_creates_missing_alias(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    backend = QdrantPromotionBackend(client)

    backend.switch_alias("georgian_legal", plan.physical_collection)

    assert len(client.alias_updates) == 1
    assert backend.alias_target("georgian_legal") == plan.physical_collection
    client.aliases.clear()
    with pytest.raises(PromotionPreconditionError, match="maintenance migration"):
        backend.switch_alias("georgian_legal", plan.physical_collection)


def test_direct_alias_switch_refuses_frozen_candidate_before_client_access():
    class ExplodingClient:
        def get_aliases(self):
            pytest.fail("frozen alias refusal must happen before client access")

    backend = QdrantPromotionBackend(ExplodingClient())

    with pytest.raises(PromotionPreconditionError, match="cannot be targeted by an alias"):
        backend.switch_alias(
            "georgian_legal",
            FROZEN_CANDIDATE_PHYSICAL_COLLECTION,
        )


def test_semantic_checks_are_mandatory_even_for_exact_storage(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    backend = QdrantPromotionBackend(client)
    backend._plans[plan.physical_collection] = plan

    assert backend.smoke(plan.physical_collection) is False
    assert backend.readiness(plan.physical_collection) is False


def test_semantic_checks_require_literal_true(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    backend = QdrantPromotionBackend(
        client,
        smoke_check=lambda _collection, _client: "false",
        readiness_check=lambda _collection, _client: "false",
    )

    assert backend.smoke(plan.physical_collection) is False
    assert backend.readiness("georgian_legal__gen_previous") is False


def test_streamed_integrity_is_mandatory_after_storage_checks_pass(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    backend = QdrantPromotionBackend(client, poll_interval_seconds=0)

    backend.restore_candidate(plan)
    with pytest.raises(PromotionPreconditionError, match="integrity check is not configured"):
        backend.wait_for_green(plan.physical_collection, 1)


def test_streamed_integrity_failure_cannot_be_reported_as_compatible(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    calls = []
    backend = QdrantPromotionBackend(
        client,
        integrity_check=lambda checked_plan, checked_client: calls.append(
            (checked_plan, checked_client)
        )
        and False,
        poll_interval_seconds=0,
    )

    backend.restore_candidate(plan)
    with pytest.raises(PromotionPreconditionError, match="streamed integrity check failed"):
        backend.wait_for_green(plan.physical_collection, 1)

    assert calls == [(plan, client)]
    assert plan.physical_collection not in backend._integrity_verified


def test_streamed_integrity_proof_must_remain_owner_only_and_intact(tmp_path):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    client = _FakeClient(plan, present=True)
    proof = _proof(tmp_path)
    proof.report_path.chmod(0o644)
    backend = QdrantPromotionBackend(
        client,
        integrity_check=lambda _plan, _client: proof,
        poll_interval_seconds=0,
    )

    backend.restore_candidate(plan)
    with pytest.raises(PromotionPreconditionError, match="intact owner-only"):
        backend.wait_for_green(plan.physical_collection, 1)

    assert plan.physical_collection not in backend._integrity_verified


def test_generation_integrity_check_streams_and_persists_post_restore_report(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "candidate.snapshot"
    snapshot.write_bytes(b"snapshot")
    plan = _plan(snapshot)
    generation_root = tmp_path / plan.generation_id
    generation_root.mkdir()
    artifacts = SimpleNamespace(
        root=generation_root,
        manifest=SimpleNamespace(generation_id=plan.generation_id),
        checksums=SimpleNamespace(files={"manifest.json": plan.manifest_sha256}),
    )
    client = SimpleNamespace(
        scroll=lambda **_kwargs: ([{"id": "point-1"}, {"id": "point-2"}], None),
        get_collection=lambda _name: SimpleNamespace(
            config={"params": {}},
            payload_schema={},
        ),
    )
    observed = {}
    report = SimpleNamespace(
        ok=True,
        generation_id=plan.generation_id,
        manifest_sha256=plan.manifest_sha256,
        physical_collection=plan.physical_collection,
    )

    def verify(
        checked_artifacts,
        points,
        *,
        physical_collection,
        observed_collection_configuration_sha256,
        verification_id,
    ):
        observed["artifacts"] = checked_artifacts
        observed["points"] = list(points)
        observed["physical_collection"] = physical_collection
        observed["collection_configuration_sha256"] = (
            observed_collection_configuration_sha256
        )
        observed["verification_id"] = verification_id
        return report

    def persist(path, checked_report):
        observed["report_path"] = path
        observed["report"] = checked_report
        path.write_bytes(b"candidate verification report\n")
        path.chmod(0o600)

    monkeypatch.setattr(promotion_backend, "verify_generation_artifacts", verify)
    monkeypatch.setattr(promotion_backend, "write_verification_report", persist)

    proof = promotion_backend._generation_integrity_check(artifacts)(plan, client)
    assert proof.ok is True
    assert observed["artifacts"] is artifacts
    assert observed["points"] == [{"id": "point-1"}, {"id": "point-2"}]
    assert observed["physical_collection"] == plan.physical_collection
    assert len(observed["collection_configuration_sha256"]) == 64
    assert observed["verification_id"] == plan.promotion_id
    assert observed["report"] is report
    assert observed["report_path"] == tmp_path / (
        f"{plan.generation_id}.{plan.promotion_id}.candidate-verification.json"
    )
    assert proof.report_sha256 == hashlib.sha256(proof.report_path.read_bytes()).hexdigest()
    assert proof.provenance_path.stat().st_mode & 0o777 == 0o600
    provenance = json.loads(proof.provenance_path.read_text(encoding="utf-8"))
    assert provenance["plan_sha256"]
    assert provenance["physical_collection"] == plan.physical_collection
    assert provenance["snapshot_sha256"] == plan.snapshot_sha256
    assert provenance["verification_report_sha256"] == proof.report_sha256
    assert proof.provenance_sha256 == hashlib.sha256(
        proof.provenance_path.read_bytes()
    ).hexdigest()


def test_cli_factory_fails_before_config_without_semantic_checks(monkeypatch):
    monkeypatch.delenv(CHECKS_FACTORY_ENV, raising=False)

    with pytest.raises(PromotionPreconditionError, match=CHECKS_FACTORY_ENV):
        make_qdrant_promotion_backend()
