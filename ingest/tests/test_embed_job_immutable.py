from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from types import SimpleNamespace

import pytest

from ingest import __main__ as primary_cli
from ingest import chunk_inventory
from ingest import embed_job
from ingest import qdrant_store
from ingest.config import ConfigurationError, load_config


_COLLECTION_CONFIGURATION = {
    "revision": qdrant_store.COLLECTION_CONFIGURATION_REVISION,
    "profile_revision": qdrant_store.EMBED_COLLECTION_PROFILE_REVISION,
    "config": {
        "params": {
            "vectors": {"dense": {"distance": "Cosine", "size": 1024}},
            "sparse_vectors": {"sparse": {}},
        }
    },
    "payload_schema": {},
}


class _ProbeEmbedder:
    def encode_query(self, text):
        offset = len(text.encode("utf-8")) / 1000.0
        return SimpleNamespace(
            dense=[1.0 + offset, 0.5 - offset],
            sparse=SimpleNamespace(indices=[9, 2], values=[0.25, 0.75]),
        )


def _checksum_reference(tmp_path):
    path = tmp_path / "vector-checksum.json"
    if not path.exists():
        embed_job.save_checksum_reference(_ProbeEmbedder(), path)
    return embed_job.load_checksum_reference(path)


def _prepare_binding(cfg, sealed, *, resume, worker_count=1, storage_sha=None):
    configuration_sha = qdrant_store.collection_configuration_sha256(
        _COLLECTION_CONFIGURATION
    )
    return embed_job.prepare_binding(
        cfg,
        sealed,
        checksum=_checksum_reference(cfg.state_dir.parent),
        worker_count=worker_count,
        collection_configuration=_COLLECTION_CONFIGURATION,
        collection_configuration_sha256=configuration_sha,
        storage_identity_sha256=storage_sha,
        resume=resume,
    )


def _cfg(tmp_path, **over):
    generation_id = "gen_20260715_embed"
    cfg = dataclasses.replace(
        load_config(),
        generation_id=generation_id,
        collection_name=f"georgian_legal__gen_{generation_id}",
        embedding_revision="a" * 40,
        tokenizer_revision="b" * 40,
        reranker_revision="c" * 40,
        state_dir=tmp_path / "state",
    )
    return dataclasses.replace(cfg, **over)


def _frozen_cfg(tmp_path):
    from ingest.release_inputs import GENERATION_ID, PHYSICAL_COLLECTION

    return dataclasses.replace(
        _cfg(tmp_path),
        generation_id=GENERATION_ID,
        collection_name=PHYSICAL_COLLECTION,
        embed_device="cuda",
        embed_use_fp16=True,
        embed_batch_size=256,
    )


def _reviewed_capability(cfg, *, operations=("create", "recover", "upsert")):
    configuration = qdrant_store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )
    return qdrant_store._issue_reviewed_mutation_capability(
        generation_id=cfg.generation_id,
        collection_name=cfg.collection_name,
        reviewed_plan_sha256="1" * 64,
        launch_evidence_sha256="2" * 64,
        collection_configuration_digest=(
            qdrant_store.collection_configuration_sha256(configuration)
        ),
        storage_identity_sha256="3" * 64,
        allowed_operations=operations,
    )


def _capability_scope():
    return {
        "reviewed_plan_sha256": "1" * 64,
        "launch_evidence_sha256": "2" * 64,
        "storage_identity_sha256": "3" * 64,
    }


def _sealed(tmp_path, cfg=None, *, records=None):
    cfg = cfg or _cfg(tmp_path)
    root = tmp_path / "snapshot"
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for source in embed_job.SOURCES:
        source_rows = (records or {}).get(source, [])
        (docs / f"{source}.jsonl").write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in source_rows
            ),
            encoding="utf-8",
        )
    inventory_path = root / chunk_inventory.CHUNK_INVENTORY_FILENAME
    inventory_path.unlink(missing_ok=True)
    inventory = chunk_inventory.compute_structural_chunk_inventory(
        docs,
        sources=embed_job.SOURCES,
        tokenizer_model=cfg.tokenizer_model,
        tokenizer_revision=cfg.tokenizer_revision,
        max_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        document_header=cfg.embed_header_v2,
        count_tokens=lambda text: len(text.split()),
        output=inventory_path,
    )
    return embed_job.SealedSnapshot(
        root=root.resolve(),
        docs=docs.resolve(),
        snapshot_id="v3_512_candidate_20260715",
        snapshot_sha256="d" * 64,
        corpus_sha256="e" * 64,
        sources=embed_job.SOURCES,
        manifest={
            "build": {
                "embed_model": cfg.embed_model,
                "embedding_revision": cfg.embedding_revision,
                "tokenizer": {
                    "model": cfg.tokenizer_model,
                    "revision": cfg.tokenizer_revision,
                },
                "chunk": {
                    "tokens": cfg.chunk_tokens,
                    "overlap": cfg.chunk_overlap,
                    "min_tokens": cfg.chunk_min_tokens,
                },
                "document_header": cfg.embed_header_v2,
            },
            "structural_chunk_inventory": inventory.manifest_entry(),
        },
    )


def _doc(document_id: str, version_id: str):
    return SimpleNamespace(
        source="matsne",
        document_id=document_id,
        version_id=version_id,
    )


def _canonical_snapshot_record(
    document_id="1",
    version_id="derived:v1",
    *,
    body="complete official body for deterministic resume proof",
):
    value = {
            "source": "matsne",
            "document_id": document_id,
            "body_markdown": body,
            "language": "ka",
            "source_fingerprint": "f" * 64,
            "normalizer_revision": "canonical-v4",
            "version_id": version_id,
            "version_id_kind": "derived",
            "content_kind": "full_text",
            "content_complete": True,
            "extraction_status": "full_text",
            "version_lineage_status": "derived",
            "version_lineage_complete": False,
            "source_authority": "primary_official",
            "official_url": "https://example.test/law/1",
            "source_url": "https://example.test/law/1",
            "freshness_sla_met": True,
            "admissible": True,
            "page_boundaries": [],
            "page_coordinate_reason": "source_not_paginated",
            "supersedes": [],
            "consolidated_dates": [],
        }
    value.update(
        {
            "doc_id": f"matsne:{document_id}:{version_id}",
            "content_hash": qdrant_store.content_hash(body),
            "body_char_len": len(body),
        }
    )
    return value


def _canonical_doc(document_id="1", version_id="derived:v1"):
    return embed_job.snapshot_doc_to_canonical(
        _canonical_snapshot_record(document_id, version_id),
        strict=True,
    )


class _Client:
    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def upsert(self, collection_name, points, wait=False):
        self.calls.append((collection_name, [point.id for point in points], wait))
        if self.fail_on_call == len(self.calls):
            raise OSError("acknowledgement unavailable")


def test_binding_is_create_only_and_resume_requires_exact_full_tuple(tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = _prepare_binding(cfg, sealed, resume=False)

    assert binding.path == tmp_path / "state" / "embed" / cfg.generation_id / "binding.json"
    assert binding.value["retrieval"] == {
        "fingerprint_revision": 2,
        "fingerprint_sha256": binding.value["retrieval"]["fingerprint_sha256"],
    }
    assert binding.value["payload_schema"]["generation_schema_version"] == 2
    assert _prepare_binding(cfg, sealed, resume=True).value == binding.value

    with pytest.raises(embed_job.EmbedStateError, match="already exists"):
        _prepare_binding(cfg, sealed, resume=False)
    with pytest.raises(embed_job.EmbedStateError, match="configuration mismatch"):
        _prepare_binding(
            dataclasses.replace(cfg, chunk_tokens=cfg.chunk_tokens + 1),
            sealed,
            resume=True,
        )


def test_snapshot_build_config_is_exact_and_enforces_512_candidate(tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    embed_job.validate_snapshot_build_config(sealed, cfg)

    bad_manifest = json.loads(json.dumps(sealed.manifest))
    bad_manifest["build"]["tokenizer"]["revision"] = "9" * 40
    with pytest.raises(embed_job.EmbedStateError, match="tokenizer.revision"):
        embed_job.validate_snapshot_build_config(
            dataclasses.replace(sealed, manifest=bad_manifest), cfg
        )

    non_candidate = dataclasses.replace(cfg, chunk_tokens=513)
    with pytest.raises(embed_job.EmbedStateError, match="candidate chunk contract"):
        embed_job.validate_snapshot_build_config(
            _sealed(tmp_path, non_candidate), non_candidate
        )

    unavailable_manifest = json.loads(json.dumps(sealed.manifest))
    identity = unavailable_manifest["structural_chunk_inventory"]["identity"]
    unavailable_manifest["structural_chunk_inventory"] = (
        chunk_inventory.unavailable_manifest_entry(identity)
    )
    with pytest.raises(embed_job.EmbedStateError, match="requires an available"):
        embed_job.validate_snapshot_build_config(
            dataclasses.replace(sealed, manifest=unavailable_manifest), cfg
        )


def test_binding_refuses_symlinked_state_parent(tmp_path):
    cfg = _cfg(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    cfg.state_dir.mkdir()
    (cfg.state_dir / "embed").symlink_to(outside, target_is_directory=True)
    with pytest.raises(embed_job.EmbedStateError, match="symlink"):
        _prepare_binding(cfg, _sealed(tmp_path, cfg), resume=False)
    assert list(outside.iterdir()) == []


def test_resume_rejects_absent_corrupt_and_mismatched_checkpoint(tmp_path):
    binding = _prepare_binding(_cfg(tmp_path), _sealed(tmp_path), resume=False)
    with pytest.raises(embed_job.EmbedStateError, match="checkpoint is absent"):
        embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=True)

    path = embed_job.checkpoint_path(binding, "matsne")
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(embed_job.EmbedStateError, match="invalid JSON"):
        embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=True)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": binding.value["generation_id"],
                "variant_id": binding.variant_id,
                "snapshot_sha256": "0" * 64,
                "physical_collection": binding.value["physical_collection"],
                "source": "matsne",
                "shard": {"index": 0, "count": 1},
                "cursor": None,
                "documents_completed": 0,
                "chunks_completed": 0,
                "complete": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(embed_job.EmbedStateError, match="snapshot_sha256 mismatch"):
        embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=True)


def test_coordinator_creates_complete_zero_topology_before_workers(tmp_path):
    cfg = _cfg(tmp_path)
    binding = _prepare_binding(
        cfg,
        _sealed(tmp_path, cfg),
        resume=False,
        worker_count=2,
        storage_sha="a" * 64,
    )

    coordinator = embed_job.initialize_coordinator(binding)
    assert coordinator.value["binding_sha256"] == embed_job.binding_sha256(binding)
    assert coordinator.value["checkpoint_count"] == 2 * len(embed_job.SOURCES)
    assert embed_job.load_coordinator(binding).value == coordinator.value
    for shard_index in range(2):
        checkpoints = embed_job.preflight_checkpoints(
            binding,
            embed_job.SOURCES,
            (shard_index, 2),
            resume=True,
        )
        assert all(
            checkpoint["cursor"] is None
            and checkpoint["documents_completed"] == 0
            and checkpoint["chunks_completed"] == 0
            for checkpoint in checkpoints.values()
        )
    with pytest.raises(embed_job.EmbedStateError, match="already exists"):
        embed_job.initialize_coordinator(binding)


def test_marker_absent_coordinator_recovery_completes_only_zero_topology(tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    binding = _prepare_binding(cfg, sealed, resume=False, worker_count=2)
    # Simulate a crash after one checkpoint but before the coordinator marker.
    embed_job.preflight_checkpoints(
        binding, ["matsne"], (0, 2), resume=False
    )
    coordinator = embed_job.initialize_coordinator(binding, recover=True)
    assert coordinator.value["checkpoint_count"] == 2 * len(embed_job.SOURCES)
    assert embed_job.load_coordinator(binding).value == coordinator.value

    other_cfg = _cfg(tmp_path / "nonzero")
    other_sealed = _sealed(tmp_path / "nonzero", other_cfg)
    other = _prepare_binding(other_cfg, other_sealed, resume=False)
    initial = embed_job.preflight_checkpoints(
        other, ["matsne"], None, resume=False
    )["matsne"]
    changed = dict(initial)
    changed.update(
        {
            "cursor": {
                "global_index": 0,
                "source": "matsne",
                "document_id": "doc-1",
                "version_id": "v1",
            },
            "documents_completed": 1,
            "chunks_completed": 1,
        }
    )
    embed_job._replace_checkpoint(
        embed_job.checkpoint_path(other, "matsne"),
        changed,
        expected_previous=initial,
    )
    with pytest.raises(embed_job.EmbedStateError, match="non-zero checkpoint"):
        embed_job.initialize_coordinator(other, recover=True)


def test_initialization_and_binding_are_exactly_recoverable_after_crash(tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    checksum = _checksum_reference(tmp_path)
    configuration = qdrant_store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )
    configuration_sha = qdrant_store.collection_configuration_sha256(configuration)
    first = embed_job.prepare_initialization(
        cfg,
        sealed,
        checksum=checksum,
        worker_count=1,
        collection_configuration=configuration,
        collection_configuration_sha256=configuration_sha,
        storage_identity_sha256="a" * 64,
    )
    assert first.recovered is False
    assert first.value["collection_configuration"] == {
        "value": configuration,
        "sha256": configuration_sha,
    }
    assert first.value["embedding_runtime"] == {
        "device": cfg.embed_device,
        "use_fp16": cfg.embed_use_fp16,
        "batch_size": cfg.embed_batch_size,
    }
    recovered = embed_job.prepare_initialization(
        cfg,
        sealed,
        checksum=checksum,
        worker_count=1,
        collection_configuration=configuration,
        collection_configuration_sha256=configuration_sha,
        storage_identity_sha256="a" * 64,
    )
    assert recovered.recovered is True
    changed_configuration = copy.deepcopy(configuration)
    changed_configuration["config"]["wal_config"]["wal_capacity_mb"] += 1
    with pytest.raises(embed_job.EmbedStateError, match="complete profile"):
        embed_job.prepare_initialization(
            cfg,
            sealed,
            checksum=checksum,
            worker_count=1,
            collection_configuration=changed_configuration,
            collection_configuration_sha256=(
                qdrant_store.collection_configuration_sha256(changed_configuration)
            ),
            storage_identity_sha256="a" * 64,
        )
    with pytest.raises(embed_job.EmbedStateError, match="does not match"):
        embed_job.prepare_initialization(
            cfg,
            sealed,
            checksum=checksum,
            worker_count=2,
            collection_configuration=configuration,
            collection_configuration_sha256=configuration_sha,
            storage_identity_sha256="a" * 64,
        )

    binding = _prepare_binding(
        cfg, sealed, resume=False, storage_sha="a" * 64
    )
    configuration_sha = qdrant_store.collection_configuration_sha256(
        _COLLECTION_CONFIGURATION
    )
    recovered_binding = embed_job.prepare_binding(
        cfg,
        sealed,
        checksum=checksum,
        worker_count=1,
        collection_configuration=_COLLECTION_CONFIGURATION,
        collection_configuration_sha256=configuration_sha,
        storage_identity_sha256="a" * 64,
        resume=False,
        recover_initialization=True,
    )
    assert recovered_binding.value == binding.value


def test_storage_identity_binds_inode_and_shared_volume_evidence(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "state" / "embed" / "candidate"
    qdrant_root = tmp_path / "qdrant"
    qdrant_root.mkdir()
    identity = tmp_path / "volume.identity"
    identity.write_bytes(b"immutable-volume-one\n")
    baseline = embed_job.storage_identity_sha256(
        identity,
        checkpoint_root=checkpoint_root,
        qdrant_storage_root=qdrant_root,
    )
    copied = tmp_path / "copied.identity"
    shutil.copyfile(identity, copied)
    copied_hash = embed_job.storage_identity_sha256(
        copied,
        checkpoint_root=checkpoint_root,
        qdrant_storage_root=qdrant_root,
    )
    assert baseline != copied_hash

    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    _prepare_binding(cfg, sealed, resume=False, storage_sha=baseline)
    with pytest.raises(embed_job.EmbedStateError, match="binding does not match"):
        _prepare_binding(cfg, sealed, resume=True, storage_sha=copied_hash)

    original = embed_job._directory_volume_evidence

    def different_qdrant_volume(path, *, label, allow_missing_leaf):
        evidence = original(path, label=label, allow_missing_leaf=allow_missing_leaf)
        if label == "qdrant storage root":
            evidence["st_dev"] += 1
        return evidence

    monkeypatch.setattr(
        embed_job, "_directory_volume_evidence", different_qdrant_volume
    )
    with pytest.raises(embed_job.EmbedStateError, match="same device/filesystem"):
        embed_job.storage_identity_sha256(
            identity,
            checkpoint_root=checkpoint_root,
            qdrant_storage_root=qdrant_root,
        )


def test_empty_collection_initialization_recovery_fills_only_expected_indexes(tmp_path):
    cfg = _cfg(tmp_path)
    expected = qdrant_store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )

    class Client:
        def __init__(self, *, unknown=False, configuration=None, payload_schema=None):
            selected = copy.deepcopy(configuration or expected)
            self.config = qdrant_store.models.CollectionConfig.model_validate(
                selected["config"]
            )
            self.payload_schema = copy.deepcopy(payload_schema or {})
            if unknown:
                self.payload_schema["intruder"] = {}

        def collection_exists(self, name):
            assert name == cfg.collection_name
            return True

        def get_collection(self, name):
            assert name == cfg.collection_name
            return SimpleNamespace(
                points_count=0,
                payload_schema=self.payload_schema,
                config=self.config,
            )

        def create_payload_index(self, name, *, field_name, field_schema):
            assert name == cfg.collection_name
            del field_schema
            self.payload_schema[field_name] = copy.deepcopy(
                expected["payload_schema"][field_name]
            )

    client = Client()
    assert qdrant_store.recover_embed_collection_initialization(
        client,
        cfg,
        expected_configuration=expected,
        apply=True,
        environ={"QDRANT_WRITE_APPROVED": "1"},
    ) == 0
    assert set(client.payload_schema) == set(qdrant_store._expected_payload_indexes())

    with pytest.raises(RuntimeError, match="unknown payload indexes"):
        qdrant_store.recover_embed_collection_initialization(
            Client(unknown=True),
            cfg,
            expected_configuration=expected,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )

    drifted = copy.deepcopy(expected)
    drifted["config"]["optimizer_config"]["indexing_threshold"] += 1
    with pytest.raises(RuntimeError, match="foreign collection configuration"):
        qdrant_store.recover_embed_collection_initialization(
            Client(configuration=drifted),
            cfg,
            expected_configuration=expected,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )

    wrong_source_index = copy.deepcopy(expected["payload_schema"]["source"])
    wrong_source_index["data_type"] = "text"
    with pytest.raises(RuntimeError, match="mismatched payload index"):
        qdrant_store.recover_embed_collection_initialization(
            Client(payload_schema={"source": wrong_source_index}),
            cfg,
            expected_configuration=expected,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("config", "params", "on_disk_payload"), False),
        (("config", "hnsw_config", "on_disk"), True),
        (("config", "optimizer_config", "indexing_threshold"), 9_999),
        (("config", "wal_config", "wal_capacity_mb"), 33),
        (("config", "quantization_config", "scalar", "always_ram"), False),
        (("config", "strict_mode_config"), {"enabled": False}),
    ],
)
def test_complete_collection_profile_hash_covers_server_sensitive_fields(
    tmp_path, path, replacement
):
    cfg = _cfg(tmp_path)
    expected = qdrant_store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )
    changed = copy.deepcopy(expected)
    target = changed
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = replacement
    assert changed != expected
    assert (
        qdrant_store.collection_configuration_sha256(changed)
        != qdrant_store.collection_configuration_sha256(expected)
    )

    class ForeignEmptyClient:
        def collection_exists(self, name):
            assert name == cfg.collection_name
            return True

        def get_collection(self, name):
            assert name == cfg.collection_name
            return SimpleNamespace(
                points_count=0,
                config=qdrant_store.models.CollectionConfig.model_validate(
                    changed["config"]
                ),
                payload_schema=copy.deepcopy(expected["payload_schema"]),
            )

    with pytest.raises(RuntimeError, match="foreign collection configuration"):
        qdrant_store.recover_embed_collection_initialization(
            ForeignEmptyClient(),
            cfg,
            expected_configuration=expected,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


@pytest.mark.parametrize(
    "leaf", ["prepare", "ensure", "recover", "upsert", "delete"]
)
def test_frozen_qdrant_mutation_leaves_refuse_generic_apply_without_capability(
    tmp_path, leaf
):
    cfg = _frozen_cfg(tmp_path)
    expected = qdrant_store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )

    class UntouchedClient:
        def __getattr__(self, name):
            pytest.fail(f"client must remain untouched before authority validation: {name}")

    client = UntouchedClient()
    with pytest.raises(ConfigurationError, match="reviewed-workflow authority"):
        if leaf == "prepare":
            qdrant_store.prepare_embed_collection(
                client,
                cfg,
                resume=False,
                recreate=False,
                apply=True,
                environ={"QDRANT_WRITE_APPROVED": "1"},
            )
        elif leaf == "ensure":
            qdrant_store.ensure_collection(
                client,
                cfg,
                apply=True,
                environ={"QDRANT_WRITE_APPROVED": "1"},
            )
        elif leaf == "recover":
            qdrant_store.recover_embed_collection_initialization(
                client,
                cfg,
                expected_configuration=expected,
                apply=True,
                environ={"QDRANT_WRITE_APPROVED": "1"},
            )
        elif leaf == "upsert":
            qdrant_store.upsert_points(client, cfg.collection_name, [object()], wait=True)
        else:
            qdrant_store.delete_doc_chunks_from(
                client,
                cfg.collection_name,
                "matsne",
                "doc-1",
                1,
                version_id="v1",
            )


def test_frozen_upsert_capability_is_exactly_launch_and_storage_scoped(tmp_path):
    cfg = _frozen_cfg(tmp_path)
    capability = _reviewed_capability(cfg, operations=("upsert",))

    class Client:
        calls = []

        def upsert(self, *, collection_name, points, wait):
            self.calls.append((collection_name, points, wait))

    client = Client()
    for changed_scope in (
        {**_capability_scope(), "reviewed_plan_sha256": "4" * 64},
        {**_capability_scope(), "launch_evidence_sha256": "5" * 64},
        {**_capability_scope(), "storage_identity_sha256": "6" * 64},
    ):
        with pytest.raises(ConfigurationError, match="reviewed-workflow authority"):
            qdrant_store.upsert_points(
                client,
                cfg.collection_name,
                [object()],
                wait=True,
                mutation_capability=capability,
                **changed_scope,
            )
    assert client.calls == []
    qdrant_store.upsert_points(
        client,
        cfg.collection_name,
        ["authorized"],
        wait=True,
        mutation_capability=capability,
        **_capability_scope(),
    )
    assert client.calls == [(cfg.collection_name, ["authorized"], True)]


def test_frozen_worker_capability_cannot_create_and_forged_token_is_rejected(tmp_path):
    cfg = _frozen_cfg(tmp_path)
    worker = _reviewed_capability(cfg, operations=("upsert",))
    with pytest.raises(ConfigurationError, match="reviewed-workflow authority"):
        qdrant_store.require_reviewed_mutation_capability(
            cfg,
            worker,
            operation="create",
            **_capability_scope(),
        )
    forged = dataclasses.replace(worker, _authority=object())
    with pytest.raises(ConfigurationError, match="reviewed-workflow authority"):
        qdrant_store.require_reviewed_mutation_capability(
            cfg,
            forged,
            operation="upsert",
            **_capability_scope(),
        )


def test_direct_embed_job_import_cannot_reach_frozen_upsert_without_capability(tmp_path):
    cfg = _frozen_cfg(tmp_path)
    with pytest.raises(ConfigurationError, match="reviewed-workflow authority"):
        embed_job.embed_docs(
            cfg,
            SimpleNamespace(),
            SimpleNamespace(),
            lambda text: len(text.split()),
            [],
        )


def test_reviewed_capability_threads_through_frozen_embed_upsert(monkeypatch, tmp_path):
    cfg = _frozen_cfg(tmp_path)
    capability = _reviewed_capability(cfg, operations=("upsert",))
    point = SimpleNamespace(id="candidate-point")
    monkeypatch.setattr(
        embed_job,
        "_build_doc_points",
        lambda *_args, **_kwargs: ([point], 1),
    )
    client = _Client()

    result = embed_job.embed_docs(
        cfg,
        client,
        SimpleNamespace(),
        lambda text: len(text.split()),
        [SimpleNamespace(source="matsne", document_id="1", version_id="v1")],
        mutation_capability=capability,
        **_capability_scope(),
    )

    assert result == (1, 1, 0)
    assert client.calls == [(cfg.collection_name, [point.id], True)]


def test_checkpoint_uses_global_source_document_version_cursor_and_variant_namespace(
    monkeypatch, tmp_path
):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = _prepare_binding(cfg, sealed, resume=False)
    prepared = embed_job.preflight_checkpoints(
        binding, ["matsne"], (1, 2), resume=False
    )["matsne"]
    docs = [_doc(str(index), f"v{index}") for index in range(5)]
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter(docs),
    )
    monkeypatch.setattr(
        embed_job,
        "_build_doc_points",
        lambda cfg, embedder, count_tokens, doc: ([SimpleNamespace(id=doc.document_id)], 1),
    )

    client = _Client()
    result = embed_job.embed_source_resumable(
        cfg,
        client,
        None,
        None,
        "matsne",
        binding=binding,
        snapshot_docs=sealed.docs,
        resume=False,
        prepared_checkpoint=prepared,
        batch_size=1,
        shard=(1, 2),
    )

    assert result == (2, 2, 0)
    assert client.calls == [
        (cfg.collection_name, ["1"], True),
        (cfg.collection_name, ["3"], True),
    ]
    path = embed_job.checkpoint_path(binding, "matsne", (1, 2))
    assert binding.variant_id in path.parts
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert checkpoint["cursor"] == {
        "global_index": 3,
        "source": "matsne",
        "document_id": "3",
        "version_id": "v3",
    }
    assert checkpoint["complete"] is True


def test_checkpoint_never_advances_past_failed_acknowledgement(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = _prepare_binding(cfg, sealed, resume=False)
    prepared = embed_job.preflight_checkpoints(
        binding, ["matsne"], None, resume=False
    )["matsne"]
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter([_doc("0", "v0"), _doc("1", "v1")]),
    )
    monkeypatch.setattr(
        embed_job,
        "_build_doc_points",
        lambda cfg, embedder, count_tokens, doc: ([SimpleNamespace(id=doc.document_id)], 1),
    )

    with pytest.raises(OSError, match="acknowledgement unavailable"):
        embed_job.embed_source_resumable(
            cfg,
            _Client(fail_on_call=2),
            None,
            None,
            "matsne",
            binding=binding,
            snapshot_docs=sealed.docs,
            resume=False,
            prepared_checkpoint=prepared,
            batch_size=1,
        )

    checkpoint = json.loads(
        embed_job.checkpoint_path(binding, "matsne").read_text(encoding="utf-8")
    )
    assert checkpoint["cursor"]["document_id"] == "0"
    assert checkpoint["documents_completed"] == 1
    assert checkpoint["complete"] is False


def test_embedding_failure_is_fatal_and_does_not_advance_checkpoint(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = _prepare_binding(cfg, sealed, resume=False)
    prepared = embed_job.preflight_checkpoints(
        binding, ["matsne"], None, resume=False
    )["matsne"]
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter([_doc("0", "v0")]),
    )

    def fail(*args, **kwargs):
        raise ValueError("conversion failed")

    monkeypatch.setattr(embed_job, "_build_doc_points", fail)
    with pytest.raises(embed_job.EmbedStateError, match="embedding failed"):
        embed_job.embed_source_resumable(
            cfg,
            _Client(),
            None,
            None,
            "matsne",
            binding=binding,
            snapshot_docs=sealed.docs,
            resume=False,
            prepared_checkpoint=prepared,
        )
    checkpoint = json.loads(
        embed_job.checkpoint_path(binding, "matsne").read_text(encoding="utf-8")
    )
    assert checkpoint["cursor"] is None
    assert checkpoint["documents_completed"] == 0


def test_strict_snapshot_conversion_never_defaults_authority_completeness_or_version(
    tmp_path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    record = {
        "source": "matsne",
        "document_id": "1",
        "body_markdown": "complete official text",
        "language": "ka",
        "source_fingerprint": "a" * 64,
        "normalizer_revision": "canonical-v4",
        "version_id": "derived:abc",
        "version_id_kind": "derived",
        "content_kind": "full_text",
        "content_complete": True,
        "extraction_status": "full_text",
        "version_lineage_status": "derived",
        "version_lineage_complete": False,
        "source_authority": "primary_official",
        "official_url": "https://example.test/1",
        "freshness_sla_met": True,
        "admissible": True,
        "page_boundaries": [],
        "page_coordinate_reason": "source_not_paginated",
        "supersedes": [],
        "consolidated_dates": [],
    }
    for missing in (
        "source_authority",
        "content_complete",
        "version_id",
        "admissible",
        "page_boundaries",
        "page_coordinate_reason",
    ):
        candidate = dict(record)
        candidate.pop(missing)
        (docs / "matsne.jsonl").write_text(
            json.dumps(candidate, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(embed_job.EmbedStateError, match=missing):
            list(embed_job.iter_snapshot_docs("matsne", root=docs, strict=True))


def test_checksum_output_is_explicit_create_only_and_outside_snapshot(tmp_path):
    class Embedder:
        def encode_query(self, _sentence):
            return SimpleNamespace(
                dense=[1.0, 0.5],
                sparse=SimpleNamespace(indices=[3, 1], values=[0.25, 0.75]),
            )

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "evidence" / "checksum.json"
    assert embed_job.save_checksum_reference(
        Embedder(), output, snapshot_root=snapshot
    )
    with pytest.raises(embed_job.EmbedStateError, match="already exists"):
        embed_job.save_checksum_reference(Embedder(), output, snapshot_root=snapshot)
    with pytest.raises(embed_job.EmbedStateError, match="outside"):
        embed_job.save_checksum_reference(
            Embedder(), snapshot / "checksum.json", snapshot_root=snapshot
        )


def test_checksum_hashes_exact_dense_and_sorted_sparse_float32_bytes(tmp_path):
    reference = _checksum_reference(tmp_path)
    assert len(reference.probe_sha256) == 64
    assert all(probe["sparse"]["indices"] == [2, 9] for probe in reference.probes)

    class ChangedSparse(_ProbeEmbedder):
        def encode_query(self, text):
            embedded = super().encode_query(text)
            return SimpleNamespace(
                dense=embedded.dense,
                sparse=SimpleNamespace(indices=[9, 2], values=[0.2500001, 0.75]),
            )

    changed_path = tmp_path / "changed-sparse.json"
    embed_job.save_checksum_reference(ChangedSparse(), changed_path)
    changed = embed_job.load_checksum_reference(changed_path)
    assert changed.probe_sha256 != reference.probe_sha256


def test_checksum_comparison_is_create_only_and_enforces_full_suite_gate(tmp_path):
    cpu = _checksum_reference(tmp_path)
    runtime_path = tmp_path / "runtime-checksum.json"
    embed_job.save_checksum_reference(_ProbeEmbedder(), runtime_path)
    runtime = embed_job.load_checksum_reference(runtime_path)
    comparison_path = tmp_path / "comparison.json"

    comparison = embed_job.save_checksum_comparison(
        cpu,
        runtime,
        comparison_path,
        minimum_cosine=0.999,
    )
    assert comparison.dense_cosines == pytest.approx((1.0, 1.0, 1.0))
    assert embed_job.load_checksum_comparison(
        comparison_path,
        cpu=cpu,
        runtime=runtime,
    ).file_sha256 == comparison.file_sha256
    with pytest.raises(embed_job.EmbedStateError, match="already exists"):
        embed_job.save_checksum_comparison(cpu, runtime, comparison_path)

    class Orthogonal(_ProbeEmbedder):
        def encode_query(self, text):
            return SimpleNamespace(
                dense=[0.0, 1.0],
                sparse=SimpleNamespace(indices=[2, 9], values=[0.75, 0.25]),
            )

    failed_runtime_path = tmp_path / "failed-runtime.json"
    embed_job.save_checksum_reference(Orthogonal(), failed_runtime_path)
    with pytest.raises(embed_job.EmbedStateError, match="cosine gate failed"):
        embed_job.save_checksum_comparison(
            cpu,
            embed_job.load_checksum_reference(failed_runtime_path),
            tmp_path / "failed-comparison.json",
        )
    assert not (tmp_path / "failed-comparison.json").exists()


def test_snapshot_verification_rejects_preflight_even_if_delegate_returns_it(
    monkeypatch, tmp_path
):
    root = tmp_path / "snapshot"
    docs = root / "docs"
    docs.mkdir(parents=True)
    manifest = {
        "snapshot_id": "v3_test",
        "snapshot_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "preflight": True,
        "build": {"sources": list(embed_job.SOURCES)},
    }
    from ingest import snapshot

    monkeypatch.setattr(snapshot, "verify_sealed_snapshot", lambda *args, **kwargs: manifest)
    with pytest.raises(embed_job.EmbedStateError, match="preflight"):
        embed_job.verify_snapshot_docs(docs)


def _resume_inventory_case(tmp_path):
    from ingest.pipeline import _document_state_hash

    cfg = _cfg(tmp_path)
    body = " ".join(f"სიტყვა-{index}" for index in range(900))
    snapshot_record = _canonical_snapshot_record(body=body)
    sealed = _sealed(
        tmp_path,
        cfg,
        records={"matsne": [snapshot_record]},
    )
    inventory_rows = list(
        chunk_inventory.iter_validated_inventory(
            sealed.root / chunk_inventory.CHUNK_INVENTORY_FILENAME,
            manifest_entry=sealed.manifest["structural_chunk_inventory"],
            expected_identity=sealed.manifest["structural_chunk_inventory"]["identity"],
        )
    )
    assert len(inventory_rows) == 1
    inventory_row = inventory_rows[0]
    assert len(inventory_row["chunks"]) > 1
    doc = embed_job._prepare_doc_for_index(
        next(embed_job.iter_snapshot_docs("matsne", root=sealed.docs, strict=True))
    )
    binding = _prepare_binding(cfg, sealed, resume=False)
    initial = embed_job.preflight_checkpoints(
        binding, ["matsne"], None, resume=False
    )["matsne"]
    checkpoint = dict(initial)
    checkpoint.update(
        {
            "cursor": {
                "global_index": 0,
                "source": doc.source,
                "document_id": doc.document_id,
                "version_id": doc.version_id,
            },
            "documents_completed": 1,
            "chunks_completed": len(inventory_row["chunks"]),
        }
    )
    path = embed_job.checkpoint_path(binding, "matsne")
    embed_job._replace_checkpoint(path, checkpoint, expected_previous=initial)
    state_hash = _document_state_hash(cfg, doc=doc)
    points = {}
    for chunk_value in inventory_row["chunks"]:
        chunk = embed_job._inventory_chunk(
            doc,
            chunk_value,
            count_tokens=lambda text: len(text.split()),
        )
        point_id = qdrant_store.point_id(
            doc.source,
            doc.document_id,
            chunk.chunk_index,
            version_id=doc.version_id,
        )
        points[point_id] = SimpleNamespace(
            id=point_id,
            payload=qdrant_store.build_payload(
                doc,
                chunk,
                document_chunk_count=len(inventory_row["chunks"]),
                document_state_hash=state_hash,
                cfg=cfg,
            ),
        )
    return cfg, sealed, checkpoint, points


def test_resume_proof_rejects_missing_acknowledged_chunk_even_when_other_point_masks_count(
    tmp_path,
):
    cfg, sealed, checkpoint, points = _resume_inventory_case(tmp_path)
    missing_id = list(points)[-1]
    points.pop(missing_id)
    masking_id = qdrant_store.point_id("matsne", "unrelated", 0, version_id="v-other")
    points[masking_id] = SimpleNamespace(id=masking_id, payload={})
    point_map = points

    class Client:
        # Collection-wide counts can still be two because this unrelated same-identity
        # point masks the deleted acknowledged chunk. Exact ID retrieval must catch it.
        def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
            assert collection_name == cfg.collection_name
            assert with_payload is True
            assert with_vectors is False
            return [point_map[point_id] for point_id in ids if point_id in point_map]

    with pytest.raises(embed_job.EmbedStateError, match="deterministic points are missing"):
        embed_job.verify_resume_checkpoint_points(
            cfg,
            Client(),
            sealed,
            {"matsne": checkpoint},
            count_tokens=lambda text: len(text.split()),
        )
    assert missing_id not in point_map


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("text", "corrupted acknowledged passage"),
        ("passage_hash", "0" * 64),
        ("char_start", 1),
        ("heading_path", ["corrupted structure"]),
        ("page_start", 1),
        ("title", "corrupted embed header"),
    ],
)
def test_resume_proof_rejects_exact_acknowledged_payload_corruption(
    tmp_path, field, changed
):
    cfg, sealed, checkpoint, points = _resume_inventory_case(tmp_path)
    first = points[next(iter(points))]
    first.payload = dict(first.payload)
    first.payload[field] = changed

    class Client:
        def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
            assert collection_name == cfg.collection_name
            assert with_payload is True
            assert with_vectors is False
            return [points[point_id] for point_id in ids if point_id in points]

    with pytest.raises(embed_job.EmbedStateError, match="exact payload mismatch"):
        embed_job.verify_resume_checkpoint_points(
            cfg,
            Client(),
            sealed,
            {"matsne": checkpoint},
            count_tokens=lambda text: len(text.split()),
        )


def test_resume_proof_recomputes_context_enriched_embed_input_identity(tmp_path):
    from ingest.pipeline import _document_state_hash

    cfg, sealed, _checkpoint, points = _resume_inventory_case(tmp_path)
    rows = list(
        chunk_inventory.iter_validated_inventory(
            sealed.root / chunk_inventory.CHUNK_INVENTORY_FILENAME,
            manifest_entry=sealed.manifest["structural_chunk_inventory"],
            expected_identity=sealed.manifest["structural_chunk_inventory"]["identity"],
        )
    )
    row = rows[0]
    doc = embed_job._prepare_doc_for_index(
        next(embed_job.iter_snapshot_docs("matsne", root=sealed.docs, strict=True))
    )
    chunk_value = copy.deepcopy(row["chunks"][0])
    chunk_value["embed_input"]["sha256"] = "0" * 64
    record = points[next(iter(points))]

    with pytest.raises(embed_job.EmbedStateError, match="embed-input identity"):
        embed_job._validate_acknowledged_inventory_point(
            cfg,
            doc,
            chunk_value,
            record,
            document_chunk_count=len(row["chunks"]),
            document_state_hash=_document_state_hash(cfg, doc=doc),
            count_tokens=lambda text: len(text.split()),
        )


def _cli_args(tmp_path, **over):
    value = {
        "collection": None,
        "snapshot_docs": tmp_path / "snapshot" / "docs",
        "source": "all",
        "checksum": False,
        "checksum_output": None,
        "vector_checksum": _checksum_reference(tmp_path).path,
        "storage_identity": None,
        "initialize_workers": None,
        "resume": False,
        "recreate": False,
        "apply": True,
        "shard": None,
        "batch_size": 10,
    }
    value.update(over)
    return SimpleNamespace(**value)


def test_cli_rejects_mutable_revision_before_snapshot_client_or_model(
    monkeypatch, tmp_path
):
    cfg = _cfg(tmp_path, embedding_revision="main")
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        embed_job,
        "verify_snapshot_docs",
        lambda path: pytest.fail("snapshot must not be accessed"),
    )
    from ingest import qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    with pytest.raises(ConfigurationError, match="immutable revisions"):
        primary_cli._cmd_embed(_cli_args(tmp_path))


def test_cli_snapshot_failure_happens_before_client_or_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        embed_job,
        "verify_snapshot_docs",
        lambda path: (_ for _ in ()).throw(embed_job.EmbedStateError("tampered snapshot")),
    )
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(embed_job.EmbedStateError, match="tampered snapshot"):
        primary_cli._cmd_embed(_cli_args(tmp_path))


def test_frozen_candidate_requires_volume_identity_even_for_one_worker(
    monkeypatch, tmp_path
):
    from ingest.release_inputs import GENERATION_ID, PHYSICAL_COLLECTION, SNAPSHOT_ID

    cfg = dataclasses.replace(
        _cfg(tmp_path),
        generation_id=GENERATION_ID,
        collection_name=PHYSICAL_COLLECTION,
    )
    sealed = dataclasses.replace(_sealed(tmp_path, cfg), snapshot_id=SNAPSHOT_ID)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda _cfg: pytest.fail("missing volume proof must abort before Qdrant"),
    )
    with pytest.raises(SystemExit, match="requires --storage-identity"):
        primary_cli._cmd_embed(_cli_args(tmp_path))


@pytest.mark.parametrize(
    ("command", "runner"),
    [("ingest", primary_cli._cmd_ingest), ("watch", primary_cli._cmd_watch)],
)
def test_legacy_mutation_commands_refuse_frozen_target_before_client_or_model(
    monkeypatch, tmp_path, command, runner
):
    from ingest.release_inputs import GENERATION_ID, PHYSICAL_COLLECTION

    cfg = dataclasses.replace(
        _cfg(tmp_path),
        generation_id=GENERATION_ID,
        collection_name=PHYSICAL_COLLECTION,
    )
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda _cfg: pytest.fail("legacy refusal must precede Qdrant access"),
    )
    from ingest import embedding

    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda _cfg: pytest.fail("legacy refusal must precede model access"),
    )
    args = SimpleNamespace(recreate=False, resume=False, apply=True)
    with pytest.raises(SystemExit, match=f"legacy `{command}` is forbidden"):
        runner(args)


def test_reviewed_launch_validation_precedes_write_authorization_and_qdrant(
    monkeypatch, tmp_path
):
    from ingest import embedding, gpu_workflow
    from ingest.release_inputs import GENERATION_ID, PHYSICAL_COLLECTION, SNAPSHOT_ID

    cfg = dataclasses.replace(
        _cfg(tmp_path),
        generation_id=GENERATION_ID,
        collection_name=PHYSICAL_COLLECTION,
    )
    sealed = dataclasses.replace(_sealed(tmp_path, cfg), snapshot_id=SNAPSHOT_ID)
    identity = tmp_path / "volume.identity"
    identity.write_text("mounted-volume\n", encoding="utf-8")
    qdrant_root = tmp_path / "qdrant"
    qdrant_root.mkdir()
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setattr(
        embedding,
        "make_token_counter",
        lambda *_args, **_kwargs: lambda text: len(text.split()),
    )
    monkeypatch.setattr(
        gpu_workflow,
        "validate_reviewed_launch",
        lambda **_kwargs: (_ for _ in ()).throw(
            gpu_workflow.GpuWorkflowError("review rejected")
        ),
    )
    monkeypatch.setattr(
        qdrant_store,
        "validate_generation_write_target",
        lambda *_args, **_kwargs: pytest.fail(
            "write authorization must follow reviewed launch validation"
        ),
    )
    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda _cfg: pytest.fail("Qdrant must not be contacted"),
    )
    with pytest.raises(gpu_workflow.GpuWorkflowError, match="review rejected"):
        primary_cli._cmd_embed(
            _cli_args(
                tmp_path,
                storage_identity=identity,
                qdrant_storage_root=qdrant_root,
                initialize_workers=1,
                plan=tmp_path / "plan.json",
                review=tmp_path / "review.json",
                bundle_root=tmp_path / "bundle",
            )
        )


def test_cli_resume_requires_binding_before_client_or_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(embed_job.EmbedStateError, match="binding is absent"):
        primary_cli._cmd_embed(_cli_args(tmp_path, resume=True))


def test_cli_rejects_zero_batch_before_snapshot_state_client_or_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        embed_job,
        "verify_snapshot_docs",
        lambda path: pytest.fail("snapshot must not be accessed"),
    )
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(SystemExit, match="batch-size must be a positive"):
        primary_cli._cmd_embed(_cli_args(tmp_path, batch_size=0))
    assert not cfg.state_dir.exists()


def test_cli_runs_exact_resume_point_proof_before_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    binding = _prepare_binding(cfg, sealed, resume=False)
    embed_job.initialize_coordinator(binding)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        embedding, "make_token_counter", lambda *_args, **_kwargs: lambda text: len(text.split())
    )

    class Client:
        def get_aliases(self):
            return SimpleNamespace(aliases=[])

        def get_collection(self, _name):
            return SimpleNamespace(
                config=_COLLECTION_CONFIGURATION["config"],
                payload_schema={},
            )

    client = Client()
    monkeypatch.setattr(qdrant_store, "make_client", lambda value: client)
    monkeypatch.setattr(qdrant_store, "prepare_embed_collection", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        embed_job,
        "verify_resume_checkpoint_points",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            embed_job.EmbedStateError("missing acknowledged point")
        ),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(embed_job.EmbedStateError, match="missing acknowledged point"):
        primary_cli._cmd_embed(_cli_args(tmp_path, resume=True, source="matsne"))


def test_cli_chunk_inventory_mismatch_aborts_before_make_client(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        embedding, "make_token_counter", lambda *_args, **_kwargs: lambda text: len(text.split())
    )
    manifest_inventory = sealed.manifest["structural_chunk_inventory"]
    manifest_inventory["sha256"] = "0" * 64
    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("chunk mismatch must abort before make_client"),
    )

    with pytest.raises(embed_job.EmbedStateError, match="chunk inventory verification failed"):
        primary_cli._cmd_embed(_cli_args(tmp_path))


def test_fresh_cli_refuses_preexisting_empty_target_without_recovery_intent(
    monkeypatch, tmp_path
):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    from ingest import embedding

    monkeypatch.setattr(
        embedding,
        "make_token_counter",
        lambda *_args, **_kwargs: lambda text: len(text.split()),
    )

    class Client:
        def get_aliases(self):
            return SimpleNamespace(aliases=[])

        def collection_exists(self, name):
            assert name == cfg.collection_name
            return True

    monkeypatch.setattr(qdrant_store, "make_client", lambda _cfg: Client())
    with pytest.raises(RuntimeError, match="pre-existing physical collection"):
        primary_cli._cmd_embed(_cli_args(tmp_path))
    assert not embed_job.initialization_path(cfg).exists()
