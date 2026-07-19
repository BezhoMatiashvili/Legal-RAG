from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest import gpu_workflow
from ingest.config import Config, load_config
from ingest.embed_job import EmbedBinding
from ingest.integrity import point_content_sha256, whole_collection_sha256
from ingest.release_inputs import GENERATION_ID, PHYSICAL_COLLECTION, SNAPSHOT_ID


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_cfg(tmp_path: Path) -> Config:
    return dataclasses.replace(
        load_config(),
        generation_id=GENERATION_ID,
        collection_name=PHYSICAL_COLLECTION,
        production_mode=True,
        state_dir=tmp_path / "state",
        qdrant_url="http://127.0.0.1:6333",
        embed_device="cuda",
        embed_use_fp16=True,
        embed_batch_size=gpu_workflow.REVIEWED_EMBED_BATCH_SIZE,
    )


def _release_manifest(cfg: Config) -> dict[str, object]:
    return {
        "runtime": {
            "oci_reference": "registry.invalid/legal@sha256:" + "3" * 64,
            "oci_digest": "sha256:" + "3" * 64,
            "identity_sha256": "4" * 64,
            "runtime_revision": "5" * 40,
            "qdrant_revision": "v1.15.0",
            "environment": {
                "EMBED_DEVICE": "cuda",
                "EMBED_USE_FP16": "true",
                "EMBED_BATCH_SIZE": str(gpu_workflow.REVIEWED_EMBED_BATCH_SIZE),
            },
        },
        "models": {
            "embedding": {"name": cfg.embed_model, "revision": cfg.embedding_revision},
            "tokenizer": {
                "name": cfg.tokenizer_model,
                "revision": cfg.tokenizer_revision,
            },
            "reranker": {"name": cfg.rerank_model, "revision": cfg.reranker_revision},
        },
        "retrieval": {
            "configuration_hash": "2" * 64,
            "knobs": gpu_workflow._runtime_retrieval_knobs(cfg),
        },
    }


def _plan(
    tmp_path: Path,
    *,
    workers: int = 1,
    cfg: Config | None = None,
    release_manifest_sha256: str = "1" * 64,
) -> tuple[Path, dict]:
    cfg = cfg or _candidate_cfg(tmp_path)
    path = (tmp_path / "workflow.json").absolute()
    identity = (tmp_path / "volume.identity").absolute()
    identity.write_bytes(b"immutable volume identity\n")
    qdrant_root = (tmp_path / "qdrant-storage").absolute()
    qdrant_root.mkdir()
    launch_root = gpu_workflow.binding_path(cfg).parent / "reviewed-launches"
    paths = {
        "plan": path.as_posix(),
        "review": (tmp_path / "workflow.review.json").absolute().as_posix(),
        "snapshot_docs": (tmp_path / "snapshot" / "docs").absolute().as_posix(),
        "cpu_checksum": (tmp_path / "cpu.json").absolute().as_posix(),
        "gpu_checksum": (tmp_path / "gpu.json").absolute().as_posix(),
        "checksum_comparison": (tmp_path / "comparison.json").absolute().as_posix(),
        "storage_identity": identity.as_posix(),
        "snapshot_export": (tmp_path / "candidate.snapshot").absolute().as_posix(),
        "export_manifest": (tmp_path / "candidate.snapshot.json").absolute().as_posix(),
        "release_bundle": gpu_workflow.RELEASE_BUNDLE_PATH,
        "launch_evidence_root": launch_root.absolute().as_posix(),
        "qdrant_storage_root": qdrant_root.as_posix(),
    }
    commands = gpu_workflow._commands(paths, worker_count=workers)
    compute = {
        "gpu_sku": "NVIDIA-L40S",
        "gpu_count": workers,
        "worker_count": workers,
        "total_hourly_usd": 1.0,
        "storage_gib_month_usd": 0.1,
        "max_runtime_hours": 4.0,
        "max_exposure_usd": 5.0,
        "derived_bound_usd": 4.0 + (0.1 * 120 * 4.0 / (24.0 * 30.0)),
        "auto_teardown": True,
    }
    value = {
        "schema_version": gpu_workflow.WORKFLOW_SCHEMA_VERSION,
        "kind": gpu_workflow.WORKFLOW_KIND,
        "workflow_id": "gpu-run-20260715",
        "release": {
            "manifest_sha256": release_manifest_sha256,
            "snapshot_id": SNAPSHOT_ID,
            "generation_id": GENERATION_ID,
            "physical_collection": PHYSICAL_COLLECTION,
            "configuration_hash": "2" * 64,
        },
        "runtime": {
            "oci_reference": "registry.invalid/legal@sha256:" + "3" * 64,
            "oci_digest": "sha256:" + "3" * 64,
            "runtime_identity_sha256": "4" * 64,
            "runtime_revision": "5" * 40,
            "qdrant_revision": "v1.15.0",
            "embed_device": "cuda",
            "embed_use_fp16": True,
            "embed_batch_size": gpu_workflow.REVIEWED_EMBED_BATCH_SIZE,
        },
        "snapshot": {
            "snapshot_id": SNAPSHOT_ID,
            "snapshot_sha256": "6" * 64,
            "corpus_sha256": "7" * 64,
        },
        "vector_gate": {
            "cpu_artifact_sha256": "8" * 64,
            "cpu_probe_sha256": "9" * 64,
            "minimum_dense_cosine": 0.999,
        },
        "storage": {
            "identity_content_sha256": hashlib.sha256(
                identity.read_bytes()
            ).hexdigest(),
            "volume_size_gib": 120,
            "isolation": "dedicated-empty-persistent-qdrant-volume",
        },
        "compute": compute,
        "collection_configuration": {
            "value": gpu_workflow.store.expected_embed_collection_configuration(
                dense_dim=gpu_workflow.FROZEN_DENSE_DIM
            ),
            "sha256": gpu_workflow.store.collection_configuration_sha256(
                gpu_workflow.store.expected_embed_collection_configuration(
                    dense_dim=gpu_workflow.FROZEN_DENSE_DIM
                )
            ),
        },
        "paths": paths,
        "commands": commands,
        "commands_sha256": gpu_workflow._command_hash(commands),
    }
    _write_json(path, value)
    return path, value


def _review(tmp_path: Path, plan_path: Path) -> Path:
    return gpu_workflow.create_workflow_review(
        plan_path,
        tmp_path / "workflow.review.json",
        reviewer="release-operator",
        reviewed_at="2026-07-15T12:00:00Z",
        paid_approval_id="approval-123",
    )


def _install_launch_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    workers: int = 1,
) -> tuple[Config, Path, dict, Path]:
    cfg = _candidate_cfg(tmp_path)
    bundle = (tmp_path / "release-bundle").absolute()
    manifest = _release_manifest(cfg)
    _write_json(bundle / "manifest.json", manifest)
    manifest_sha = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(gpu_workflow, "RELEASE_BUNDLE_PATH", bundle.as_posix())
    plan_path, plan = _plan(
        tmp_path,
        workers=workers,
        cfg=cfg,
        release_manifest_sha256=manifest_sha,
    )
    review_path = _review(tmp_path, plan_path)
    monkeypatch.setattr(
        gpu_workflow,
        "validate_release_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            root=bundle,
            manifest_sha256=manifest_sha,
            configuration_hash="2" * 64,
        ),
    )
    sealed = SimpleNamespace(
        snapshot_id=SNAPSHOT_ID,
        snapshot_sha256="6" * 64,
        corpus_sha256="7" * 64,
    )
    monkeypatch.setattr(gpu_workflow, "verify_snapshot_docs", lambda _path: sealed)
    cpu = SimpleNamespace(file_sha256="8" * 64, probe_sha256="9" * 64)
    runtime = SimpleNamespace(file_sha256="b" * 64, probe_sha256="c" * 64)

    def load_checksum(path):
        return cpu if Path(path) == Path(plan["paths"]["cpu_checksum"]) else runtime

    monkeypatch.setattr(gpu_workflow, "load_checksum_reference", load_checksum)
    monkeypatch.setattr(
        gpu_workflow,
        "load_checksum_comparison",
        lambda *_args, **_kwargs: SimpleNamespace(
            file_sha256="d" * 64,
            minimum_cosine=gpu_workflow.MINIMUM_CHECKSUM_COSINE,
        ),
    )
    return cfg, plan_path, plan, review_path


def _launch(
    cfg: Config,
    plan: dict,
    plan_path: Path,
    review_path: Path,
    *,
    initialize_workers: int | None,
    shard: tuple[int, int] | None,
    source: str = "all",
    batch_size: int = gpu_workflow.REVIEWED_EMBED_BATCH_SIZE,
    apply: bool = True,
) -> gpu_workflow.ReviewedLaunchAuthorization:
    return gpu_workflow.validate_reviewed_launch(
        plan_path=plan_path,
        review_path=review_path,
        bundle_root=plan["paths"]["release_bundle"],
        cfg=cfg,
        snapshot_docs=plan["paths"]["snapshot_docs"],
        vector_checksum=plan["paths"]["gpu_checksum"],
        storage_identity=plan["paths"]["storage_identity"],
        qdrant_storage_root=plan["paths"]["qdrant_storage_root"],
        initialize_workers=initialize_workers,
        resume=initialize_workers is None,
        shard=shard,
        source=source,
        batch_size=batch_size,
        apply=apply,
        environ={},
    )


def test_plan_commands_and_paid_review_are_exact_and_create_only(tmp_path):
    plan_path, value = _plan(tmp_path, workers=2)
    loaded_path, loaded, plan_sha = gpu_workflow.load_workflow_plan(plan_path)

    assert loaded_path == plan_path
    assert len(plan_sha) == 64
    commands = [
        loaded["commands"]["initialize_collection_and_workers"],
        *loaded["commands"]["embed_workers"],
    ]
    for command in commands:
        assert command[command.index("--plan") + 1] == value["paths"]["plan"]
        assert command[command.index("--review") + 1] == value["paths"]["review"]
        assert (
            command[command.index("--bundle-root") + 1]
            == value["paths"]["release_bundle"]
        )
        assert (
            command[command.index("--qdrant-storage-root") + 1]
            == value["paths"]["qdrant_storage_root"]
        )
        assert "--recreate" not in command
    assert [command[command.index("--shard") + 1] for command in commands[1:]] == [
        "0/2",
        "1/2",
    ]

    review_path = _review(tmp_path, plan_path)
    _, review, _ = gpu_workflow.load_workflow_review(
        review_path,
        plan=value,
        plan_sha256=plan_sha,
    )
    assert review["approved_compute"] == value["compute"]
    with pytest.raises(gpu_workflow.GpuWorkflowError, match="already exists"):
        _review(tmp_path, plan_path)

    tampered = copy.deepcopy(value)
    tampered_path = (tmp_path / "tampered.json").absolute()
    tampered["paths"]["plan"] = tampered_path.as_posix()
    tampered["commands"]["embed_workers"][0].remove("--review")
    _write_json(tampered_path, tampered)
    with pytest.raises(
        gpu_workflow.GpuWorkflowError, match="commands do not reproduce"
    ):
        gpu_workflow.load_workflow_plan(tampered_path)

    config_tampered = copy.deepcopy(value)
    config_tampered_path = (tmp_path / "config-tampered.json").absolute()
    config_tampered["paths"]["plan"] = config_tampered_path.as_posix()
    config_tampered["commands"] = gpu_workflow._commands(
        config_tampered["paths"], worker_count=2
    )
    config_tampered["commands_sha256"] = gpu_workflow._command_hash(
        config_tampered["commands"]
    )
    config_tampered["collection_configuration"]["value"]["profile_revision"] = 99
    config_tampered["collection_configuration"]["sha256"] = (
        gpu_workflow.store.collection_configuration_sha256(
            config_tampered["collection_configuration"]["value"]
        )
    )
    _write_json(config_tampered_path, config_tampered)
    with pytest.raises(
        gpu_workflow.GpuWorkflowError, match="configuration is not exact"
    ):
        gpu_workflow.load_workflow_plan(config_tampered_path)


def test_plan_creation_does_not_require_future_remote_volume(tmp_path, monkeypatch):
    cfg = _candidate_cfg(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = _release_manifest(cfg)
    _write_json(bundle / "manifest.json", manifest)
    manifest_sha = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(
        gpu_workflow, "RELEASE_BUNDLE_PATH", bundle.absolute().as_posix()
    )
    monkeypatch.setattr(
        gpu_workflow,
        "validate_release_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            root=bundle.absolute(),
            manifest_sha256=manifest_sha,
            configuration_hash="2" * 64,
        ),
    )
    monkeypatch.setattr(
        gpu_workflow,
        "verify_snapshot_docs",
        lambda _path: SimpleNamespace(
            snapshot_id=SNAPSHOT_ID,
            snapshot_sha256="6" * 64,
            corpus_sha256="7" * 64,
        ),
    )
    monkeypatch.setattr(
        gpu_workflow,
        "load_checksum_reference",
        lambda _path: SimpleNamespace(file_sha256="8" * 64, probe_sha256="9" * 64),
    )
    local_identity = tmp_path / "plan.identity"
    local_identity.write_bytes(b"identity copied to the future reviewed volume\n")
    output = (tmp_path / "planned.json").absolute()
    future_root = "/future-persistent-volume"
    paths = {
        "plan": "/workflow/planned.json",
        "review": "/workflow/planned.review.json",
        "snapshot_docs": "/workflow/snapshot/docs",
        "cpu_checksum": "/workflow/cpu.json",
        "gpu_checksum": "/workflow/gpu.json",
        "checksum_comparison": "/workflow/comparison.json",
        "storage_identity": future_root + "/volume.identity",
        "snapshot_export": "/workflow/candidate.snapshot",
        "export_manifest": "/workflow/candidate.snapshot.json",
        "release_bundle": bundle.absolute().as_posix(),
        "launch_evidence_root": future_root + "/state/reviewed-launches",
        "qdrant_storage_root": future_root + "/qdrant",
    }

    created = gpu_workflow.create_workflow_plan(
        cfg=cfg,
        bundle_root=bundle,
        snapshot_docs=tmp_path / "local-snapshot-docs",
        cpu_checksum=tmp_path / "local-cpu.json",
        storage_identity=local_identity,
        output=output,
        workflow_id="pre-spend-plan",
        container_paths=paths,
        volume_size_gib=120,
        worker_count=1,
        gpu_sku="NVIDIA-L40S",
        gpu_count=1,
        total_hourly_usd=1.0,
        storage_gib_month_usd=0.1,
        max_runtime_hours=4.0,
        max_exposure_usd=5.0,
        auto_teardown=True,
        environ={},
    )
    planned = json.loads(created.read_text(encoding="utf-8"))
    assert (
        planned["storage"]["identity_content_sha256"]
        == hashlib.sha256(local_identity.read_bytes()).hexdigest()
    )
    assert not Path(paths["qdrant_storage_root"]).exists()


def test_launch_evidence_exact_reuse_and_conflict_or_copied_identity_fail(
    tmp_path, monkeypatch
):
    cfg, plan_path, plan, review_path = _install_launch_fakes(
        monkeypatch, tmp_path, workers=2
    )

    for command_drift in (
        {"source": "matsne"},
        {"batch_size": 128},
        {"apply": False},
    ):
        with pytest.raises(gpu_workflow.GpuWorkflowError, match="reviewed command"):
            _launch(
                cfg,
                plan,
                plan_path,
                review_path,
                initialize_workers=2,
                shard=None,
                **command_drift,
            )

    for runtime_drift in (
        {"embed_device": "cpu"},
        {"embed_use_fp16": False},
        {"embed_batch_size": 128},
    ):
        with pytest.raises(
            gpu_workflow.GpuWorkflowError, match="configuration differs"
        ):
            _launch(
                dataclasses.replace(cfg, **runtime_drift),
                plan,
                plan_path,
                review_path,
                initialize_workers=2,
                shard=None,
            )

    initialize_authorization = _launch(
        cfg,
        plan,
        plan_path,
        review_path,
        initialize_workers=2,
        shard=None,
    )
    initialize = initialize_authorization.evidence_path
    assert (
        initialize_authorization.evidence_sha256
        == hashlib.sha256(initialize.read_bytes()).hexdigest()
    )
    init_capability = initialize_authorization.mutation_capability
    assert (
        init_capability.reviewed_plan_sha256
        == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    )
    assert init_capability.launch_evidence_sha256 == (
        initialize_authorization.evidence_sha256
    )
    assert (
        init_capability.collection_configuration_sha256
        == plan["collection_configuration"]["sha256"]
    )
    assert init_capability.allowed_operations == ("create", "recover")
    initialize_inode = initialize.stat().st_ino
    initialize_bytes = initialize.read_bytes()
    assert (
        _launch(
            cfg,
            plan,
            plan_path,
            review_path,
            initialize_workers=2,
            shard=None,
        ).evidence_path
        == initialize
    )
    assert initialize.stat().st_ino == initialize_inode
    assert initialize.read_bytes() == initialize_bytes

    worker_authorization = _launch(
        cfg,
        plan,
        plan_path,
        review_path,
        initialize_workers=None,
        shard=(0, 2),
    )
    worker = worker_authorization.evidence_path
    assert worker_authorization.mutation_capability.allowed_operations == ("upsert",)
    worker_inode = worker.stat().st_ino
    assert (
        _launch(
            cfg,
            plan,
            plan_path,
            review_path,
            initialize_workers=None,
            shard=(0, 2),
        ).evidence_path
        == worker
    )
    assert worker.stat().st_ino == worker_inode

    worker_value = json.loads(worker.read_text(encoding="utf-8"))
    worker_value["command_sha256"] = "f" * 64
    _write_json(worker, worker_value)
    with pytest.raises(gpu_workflow.GpuWorkflowError, match="existing reviewed launch"):
        _launch(
            cfg,
            plan,
            plan_path,
            review_path,
            initialize_workers=None,
            shard=(0, 2),
        )

    # Replacing the reviewed identity path with the same bytes changes its inode.
    # The plan's content hash still matches, but immutable init storage evidence does not.
    worker.unlink()
    identity = Path(plan["paths"]["storage_identity"])
    identity_bytes = identity.read_bytes()
    identity.unlink()
    identity.write_bytes(identity_bytes)
    with pytest.raises(gpu_workflow.GpuWorkflowError, match="immutable initialization"):
        _launch(
            cfg,
            plan,
            plan_path,
            review_path,
            initialize_workers=None,
            shard=(0, 2),
        )


def _points() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=point_id,
            payload={"source": "matsne", "document_id": f"doc-{index}", "n": index},
            vector={
                "dense": [0.1 + index, 0.2],
                "sparse": {"indices": [7, 2], "values": [0.25, 0.5]},
            },
        )
        for index, point_id in enumerate(("point-b", "point-a"))
    ]


def _info(points_count: int, *, indexed_points: int) -> SimpleNamespace:
    expected = gpu_workflow.store.expected_embed_collection_configuration(
        dense_dim=gpu_workflow.FROZEN_DENSE_DIM
    )
    payload_schema = {
        field: {**value, "points": indexed_points}
        for field, value in expected["payload_schema"].items()
    }
    return SimpleNamespace(
        points_count=points_count,
        config=expected["config"],
        payload_schema=payload_schema,
    )


class _ReadClient:
    def __init__(self, points: list[SimpleNamespace]):
        self.points = points
        self.info_calls = 0

    def get_collection(self, _name):
        self.info_calls += 1
        return _info(len(self.points), indexed_points=self.info_calls)

    def scroll(self, **kwargs):
        start = kwargs["offset"] or 0
        end = min(start + kwargs["limit"], len(self.points))
        return self.points[start:end], (end if end < len(self.points) else None)


def test_collection_seal_is_sorted_exact_and_ignores_only_live_index_counts():
    points = _points()
    seal = gpu_workflow.scan_collection_seal(
        _ReadClient(points), "candidate", page_size=1
    )
    rows = sorted(
        (
            point.id,
            point_content_sha256(point.id, point.payload, point.vector),
        )
        for point in points
    )
    expected, count = whole_collection_sha256(rows)

    assert seal.collection_sha256 == expected
    assert seal.point_count == count == 2
    assert seal.collection_configuration["payload_schema"]["source"] == {
        "data_type": "keyword",
        "params": None,
    }


def _reviewed_vector_gate(plan: dict) -> dict[str, object]:
    return {
        "cpu_artifact_sha256": plan["vector_gate"]["cpu_artifact_sha256"],
        "cpu_probe_sha256": plan["vector_gate"]["cpu_probe_sha256"],
        "runtime_artifact_sha256": "b" * 64,
        "runtime_probe_sha256": "c" * 64,
        "comparison_sha256": "d" * 64,
        "minimum_dense_cosine": gpu_workflow.MINIMUM_CHECKSUM_COSINE,
    }


def _launch_summary(plan: dict) -> dict[str, object]:
    count = plan["compute"]["worker_count"]
    rows = []
    for index in range(-1, count):
        initializing = index == -1
        command = (
            plan["commands"]["initialize_collection_and_workers"]
            if initializing
            else plan["commands"]["embed_workers"][index]
        )
        path = gpu_workflow._launch_evidence_path(
            plan,
            initialize_workers=count if initializing else None,
            shard=None if initializing else (index, count),
        )
        rows.append(
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(f"launch-{index}".encode()).hexdigest(),
                "launch_type": "initialize" if initializing else "worker",
                "worker": {
                    "worker_id": None if initializing else index,
                    "worker_count": count,
                },
                "command_sha256": hashlib.sha256(
                    gpu_workflow._canonical_bytes(command)
                ).hexdigest(),
            }
        )
    return {
        "initialization": rows[0],
        "workers": rows[1:],
        "aggregate_sha256": hashlib.sha256(
            gpu_workflow._canonical_bytes(rows)
        ).hexdigest(),
    }


def _export_value(
    *,
    plan_path: Path,
    plan: dict,
    review_path: Path,
    seal: gpu_workflow.CollectionSeal,
    snapshot: Path,
    cfg: Config,
) -> dict[str, object]:
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    storage = gpu_workflow._storage_runtime_descriptor(
        cfg,
        storage_identity=plan["paths"]["storage_identity"],
        qdrant_storage_root=plan["paths"]["qdrant_storage_root"],
        launch_evidence_root=plan["paths"]["launch_evidence_root"],
    )
    vector_gate = _reviewed_vector_gate(plan)
    return {
        "schema_version": gpu_workflow.WORKFLOW_SCHEMA_VERSION,
        "kind": gpu_workflow.EXPORT_KIND,
        "workflow_id": plan["workflow_id"],
        "workflow_plan_sha256": plan_sha,
        "workflow_review_sha256": review_sha,
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "snapshot_id": SNAPSHOT_ID,
        "run_evidence": {
            "release": plan["release"],
            "runtime": plan["runtime"],
            "snapshot": plan["snapshot"],
            "vector_gate": vector_gate,
            "storage": storage,
            "collection_configuration": plan["collection_configuration"],
            "reviewed_launch": _launch_summary(plan),
        },
        "qdrant_snapshot": {
            "name": "candidate.snapshot",
            "reported_size_bytes": snapshot.stat().st_size,
            "reported_checksum": None,
            "export_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "export_size_bytes": snapshot.stat().st_size,
        },
        "remote_collection": {
            "point_count": seal.point_count,
            "collection_sha256": seal.collection_sha256,
            "collection_configuration_sha256": seal.collection_configuration_sha256,
            "collection_configuration": seal.collection_configuration,
        },
        "embed_binding_sha256": "e" * 64,
        "runtime_vector_checksum_artifact_sha256": vector_gate[
            "runtime_artifact_sha256"
        ],
        "runtime_vector_probe_sha256": vector_gate["runtime_probe_sha256"],
        "checksum_comparison_sha256": vector_gate["comparison_sha256"],
        "storage_identity_content_sha256": plan["storage"]["identity_content_sha256"],
        "storage_identity_sha256": storage["binding_descriptor_sha256"],
        "aliases_sha256": "0" * 64,
    }


def _export_manifest(
    *,
    plan_path: Path,
    plan: dict,
    review_path: Path,
    seal: gpu_workflow.CollectionSeal,
    snapshot: Path,
    cfg: Config,
) -> tuple[Path, dict[str, object]]:
    value = _export_value(
        plan_path=plan_path,
        plan=plan,
        review_path=review_path,
        seal=seal,
        snapshot=snapshot,
        cfg=cfg,
    )
    path = Path(plan["paths"]["export_manifest"])
    _write_json(path, value)
    return path, value


class _RestoreClient(_ReadClient):
    def __init__(self, points: list[SimpleNamespace], *, exists: bool = False):
        super().__init__(points)
        self.exists = exists
        self.uploads = 0

    def collection_exists(self, _name):
        return self.exists

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name="georgian_legal", collection_name="legacy")
            ]
        )


class _NoTouchClient:
    def __getattr__(self, name):
        pytest.fail(f"Qdrant client must not be touched before validation: {name}")


def _mutate_remote_configuration(value: dict[str, object]) -> None:
    remote = value["remote_collection"]
    remote["collection_configuration"]["profile_revision"] = 99
    remote["collection_configuration_sha256"] = (
        gpu_workflow.store.collection_configuration_sha256(
            remote["collection_configuration"]
        )
    )


def test_local_restore_refuses_existing_target_and_proves_remote_local_digest(tmp_path):
    cfg = _candidate_cfg(tmp_path)
    plan_path, plan = _plan(tmp_path, cfg=cfg)
    review_path = _review(tmp_path, plan_path)
    points = _points()
    seal = gpu_workflow.scan_collection_seal(_ReadClient(points), "candidate")
    snapshot = Path(plan["paths"]["snapshot_export"])
    snapshot.write_bytes(b"qdrant snapshot bytes")
    export, _value = _export_manifest(
        plan_path=plan_path,
        plan=plan,
        review_path=review_path,
        seal=seal,
        snapshot=snapshot,
        cfg=cfg,
    )
    existing = _RestoreClient(points, exists=True)
    with pytest.raises(gpu_workflow.GpuWorkflowError, match="pre-existing target"):
        gpu_workflow.restore_local_export(
            existing,
            cfg,
            plan_path=plan_path,
            review_path=review_path,
            export_manifest_path=export,
            snapshot_path=snapshot,
            proof_output=tmp_path / "existing-proof.json",
            apply=True,
            environ={gpu_workflow.LOCAL_RESTORE_APPROVAL_ENV: "1"},
            uploader=lambda *_args: pytest.fail("existing target must not be mutated"),
        )

    client = _RestoreClient(points)

    def upload(_cfg, collection_name, restored_snapshot):
        assert collection_name == PHYSICAL_COLLECTION
        assert restored_snapshot == snapshot
        client.uploads += 1
        client.exists = True

    proof_path = gpu_workflow.restore_local_export(
        client,
        cfg,
        plan_path=plan_path,
        review_path=review_path,
        export_manifest_path=export,
        snapshot_path=snapshot,
        proof_output=tmp_path / "restore-proof.json",
        apply=True,
        environ={gpu_workflow.LOCAL_RESTORE_APPROVAL_ENV: "1"},
        uploader=upload,
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert client.uploads == 1
    assert proof["digest_match"] is True
    assert proof["local_collection_sha256"] == proof["remote_collection_sha256"]
    assert proof["aliases_unchanged"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.__setitem__("workflow_plan_sha256", "f" * 64),
            "identity mismatch",
        ),
        (
            lambda value: value["run_evidence"]["runtime"].__setitem__(
                "runtime_revision", "f" * 40
            ),
            "reviewed identities",
        ),
        (
            lambda value: value["run_evidence"]["snapshot"].__setitem__(
                "corpus_sha256", "f" * 64
            ),
            "reviewed identities",
        ),
        (
            lambda value: value["run_evidence"]["vector_gate"].__setitem__(
                "comparison_sha256", "f" * 64
            ),
            "vector evidence",
        ),
        (
            lambda value: value.__setitem__("storage_identity_sha256", "f" * 64),
            "storage descriptor",
        ),
        (
            lambda value: value["run_evidence"]["reviewed_launch"]["workers"][0][
                "worker"
            ].__setitem__("worker_id", 9),
            "launch summary",
        ),
        (_mutate_remote_configuration, "configuration hash mismatch"),
    ],
)
def test_restore_rejects_cross_binding_mismatch_before_qdrant(
    tmp_path, mutation, match
):
    cfg = _candidate_cfg(tmp_path)
    plan_path, plan = _plan(tmp_path, cfg=cfg)
    review_path = _review(tmp_path, plan_path)
    seal = gpu_workflow.scan_collection_seal(_ReadClient(_points()), "candidate")
    snapshot = Path(plan["paths"]["snapshot_export"])
    snapshot.write_bytes(b"qdrant snapshot bytes")
    export, value = _export_manifest(
        plan_path=plan_path,
        plan=plan,
        review_path=review_path,
        seal=seal,
        snapshot=snapshot,
        cfg=cfg,
    )
    mutation(value)
    _write_json(export, value)

    with pytest.raises(gpu_workflow.GpuWorkflowError, match=match):
        gpu_workflow.restore_local_export(
            _NoTouchClient(),
            cfg,
            plan_path=plan_path,
            review_path=review_path,
            export_manifest_path=export,
            snapshot_path=snapshot,
            proof_output=tmp_path / "must-not-exist.json",
            apply=True,
            environ={gpu_workflow.LOCAL_RESTORE_APPROVAL_ENV: "1"},
            uploader=lambda *_args: pytest.fail("invalid export must not be uploaded"),
        )


def test_restore_rejects_legacy_plan_a_export_f_storage_fixture_before_qdrant(tmp_path):
    cfg = _candidate_cfg(tmp_path)
    plan_path, plan = _plan(tmp_path, cfg=cfg)
    plan["storage"]["identity_content_sha256"] = "a" * 64
    _write_json(plan_path, plan)
    review_path = _review(tmp_path, plan_path)
    seal = gpu_workflow.scan_collection_seal(_ReadClient(_points()), "candidate")
    snapshot = Path(plan["paths"]["snapshot_export"])
    snapshot.write_bytes(b"qdrant snapshot bytes")
    export, value = _export_manifest(
        plan_path=plan_path,
        plan=plan,
        review_path=review_path,
        seal=seal,
        snapshot=snapshot,
        cfg=cfg,
    )
    value["storage_identity_sha256"] = "f" * 64
    _write_json(export, value)

    with pytest.raises(gpu_workflow.GpuWorkflowError, match="storage descriptor"):
        gpu_workflow.restore_local_export(
            _NoTouchClient(),
            cfg,
            plan_path=plan_path,
            review_path=review_path,
            export_manifest_path=export,
            snapshot_path=snapshot,
            proof_output=tmp_path / "must-not-exist.json",
            apply=True,
            environ={gpu_workflow.LOCAL_RESTORE_APPROVAL_ENV: "1"},
            uploader=lambda *_args: pytest.fail("invalid export must not be uploaded"),
        )


def test_export_requires_complete_untampered_launch_evidence_before_qdrant(
    tmp_path, monkeypatch
):
    cfg, plan_path, plan, review_path = _install_launch_fakes(
        monkeypatch, tmp_path, workers=1
    )
    with pytest.raises(
        gpu_workflow.GpuWorkflowError, match="required workflow path is absent"
    ):
        gpu_workflow.seal_remote_export(
            _NoTouchClient(),
            cfg,
            plan_path=plan_path,
            review_path=review_path,
            qdrant_storage_root=plan["paths"]["qdrant_storage_root"],
            snapshot_output=plan["paths"]["snapshot_export"],
            manifest_output=plan["paths"]["export_manifest"],
        )

    _launch(
        cfg,
        plan,
        plan_path,
        review_path,
        initialize_workers=1,
        shard=None,
    )
    worker = _launch(
        cfg,
        plan,
        plan_path,
        review_path,
        initialize_workers=None,
        shard=(0, 1),
    ).evidence_path
    value = json.loads(worker.read_text(encoding="utf-8"))
    value["runtime"]["runtime_revision"] = "f" * 40
    _write_json(worker, value)
    with pytest.raises(gpu_workflow.GpuWorkflowError, match="reviewed launch evidence"):
        gpu_workflow.seal_remote_export(
            _NoTouchClient(),
            cfg,
            plan_path=plan_path,
            review_path=review_path,
            qdrant_storage_root=plan["paths"]["qdrant_storage_root"],
            snapshot_output=plan["paths"]["snapshot_export"],
            manifest_output=plan["paths"]["export_manifest"],
        )


class _ExportClient(_ReadClient):
    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name="georgian_legal", collection_name="legacy")
            ]
        )

    def create_snapshot(self, **_kwargs):
        return SimpleNamespace(
            name="remote.snapshot", size=20, checksum="remote-checksum"
        )


def test_successful_export_seals_reviewed_run_evidence(tmp_path, monkeypatch):
    cfg, plan_path, plan, review_path = _install_launch_fakes(
        monkeypatch, tmp_path, workers=1
    )
    _launch(
        cfg,
        plan,
        plan_path,
        review_path,
        initialize_workers=1,
        shard=None,
    )
    _launch(
        cfg,
        plan,
        plan_path,
        review_path,
        initialize_workers=None,
        shard=(0, 1),
    )
    points = _points()
    monkeypatch.setattr(
        gpu_workflow.store, "refuse_aliased_write_target", lambda *_a: None
    )
    monkeypatch.setattr(
        gpu_workflow.store, "prepare_embed_collection", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        gpu_workflow,
        "prepare_binding",
        lambda *_a, **kwargs: EmbedBinding(
            path=tmp_path / "binding.json",
            value={
                "variant_id": "f" * 64,
                "storage_identity_sha256": kwargs["storage_identity_sha256"],
                "reviewed_plan_sha256": kwargs["reviewed_plan_sha256"],
            },
        ),
    )
    monkeypatch.setattr(gpu_workflow, "load_coordinator", lambda _binding: None)
    source_counts = {source: 0 for source in gpu_workflow.SOURCES}
    source_counts[gpu_workflow.SOURCES[0]] = len(points)
    monkeypatch.setattr(
        gpu_workflow,
        "preflight_checkpoints",
        lambda _binding, sources, _shard, resume: {
            source: {"complete": True, "chunks_completed": source_counts[source]}
            for source in sources
        },
    )

    def download(_cfg, _collection, _name, output):
        payload = b"qdrant snapshot data"
        assert len(payload) == 20
        output.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    manifest_path = gpu_workflow.seal_remote_export(
        _ExportClient(points),
        cfg,
        plan_path=plan_path,
        review_path=review_path,
        qdrant_storage_root=plan["paths"]["qdrant_storage_root"],
        snapshot_output=plan["paths"]["snapshot_export"],
        manifest_output=plan["paths"]["export_manifest"],
        downloader=download,
    )
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    _, review, _ = gpu_workflow.load_workflow_review(
        review_path, plan=plan, plan_sha256=plan_sha
    )
    _, export, _ = gpu_workflow.load_export_manifest(
        manifest_path,
        plan=plan,
        plan_sha256=plan_sha,
        review=review,
        review_sha256=review_sha,
    )
    assert export["run_evidence"]["release"] == plan["release"]
    assert export["run_evidence"]["runtime"] == plan["runtime"]
    assert export["run_evidence"]["snapshot"] == plan["snapshot"]
    assert len(export["run_evidence"]["reviewed_launch"]["workers"]) == 1
    assert (
        export["storage_identity_sha256"]
        == export["run_evidence"]["storage"]["binding_descriptor_sha256"]
    )
