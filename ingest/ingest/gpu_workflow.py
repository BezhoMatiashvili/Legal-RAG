"""Immutable GPU embedding, Qdrant snapshot export, and local restore workflow.

This module deliberately does not provision GPUs or approve spend.  It turns an already
validated release bundle and an operator-reviewed cost envelope into a create-only plan,
then enforces that exact plan while sealing a remote Qdrant snapshot and restoring it to
the generation-named local collection.  No alias mutation exists in this workflow.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from . import qdrant_store as store
from .config import Config
from .embed_job import (
    SOURCES,
    binding_path,
    binding_sha256,
    load_checksum_comparison,
    load_checksum_reference,
    load_coordinator,
    preflight_checkpoints,
    prepare_binding,
    storage_identity_descriptor,
    verify_snapshot_docs,
)
from .integrity import point_content_sha256, whole_collection_sha256
from .release_inputs import (
    GENERATION_ID,
    PHYSICAL_COLLECTION,
    RELEASE_BUNDLE_PATH,
    SNAPSHOT_ID,
    validate_release_inputs,
)

WORKFLOW_SCHEMA_VERSION = 1
WORKFLOW_KIND = "immutable-gpu-qdrant-snapshot-workflow"
REVIEW_KIND = "immutable-gpu-workflow-review"
EXPORT_KIND = "immutable-qdrant-snapshot-export"
RESTORE_KIND = "immutable-qdrant-snapshot-restore-proof"
LAUNCH_KIND = "immutable-reviewed-embed-launch"
LOCAL_RESTORE_APPROVAL_ENV = "QDRANT_LOCAL_RESTORE_APPROVED"
MINIMUM_VOLUME_GIB = 120
MINIMUM_CHECKSUM_COSINE = 0.999
REVIEWED_EMBED_BATCH_SIZE = 256
FROZEN_DENSE_DIM = 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OCI_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class GpuWorkflowError(RuntimeError):
    """The reviewed GPU/export/restore workflow cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class CollectionSeal:
    point_count: int
    collection_sha256: str
    collection_configuration: Mapping[str, Any]
    collection_configuration_sha256: str


@dataclass(frozen=True, slots=True)
class ReviewedLaunchAuthorization:
    evidence_path: Path
    evidence_sha256: str
    mutation_capability: store.ReviewedMutationCapability


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GpuWorkflowError(f"workflow value is not canonical JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GpuWorkflowError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _path_components(path: Path):
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        yield cursor


def _require_nonsymlink_path(path: Path, *, allow_missing: bool) -> None:
    for component in _path_components(path):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            if allow_missing:
                return
            raise GpuWorkflowError(f"required workflow path is absent: {component}")
        except OSError as exc:
            raise GpuWorkflowError(
                f"cannot inspect workflow path {component}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise GpuWorkflowError(f"workflow path contains a symlink: {component}")


def _create_private_directories(path: Path) -> None:
    for component in _path_components(path):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            try:
                component.mkdir(mode=0o700)
            except FileExistsError:
                pass
            mode = component.lstat().st_mode
        except OSError as exc:
            raise GpuWorkflowError(
                f"cannot inspect workflow directory {component}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise GpuWorkflowError(
                f"workflow directory is a symlink or non-directory: {component}"
            )


def _write_create_only(path: Path, value: object) -> Path:
    destination = path.expanduser().absolute()
    _create_private_directories(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise GpuWorkflowError(
            f"immutable artifact already exists: {destination}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    return destination


def _load_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any], str]:
    supplied = Path(path).expanduser().absolute()
    _require_nonsymlink_path(supplied, allow_missing=False)
    artifact = supplied.resolve(strict=True)
    if not artifact.is_file():
        raise GpuWorkflowError(f"{label} must be a regular non-symlink file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise GpuWorkflowError(f"{label} contains duplicate key {key!r}")
            output[key] = item
        return output

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(artifact, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GpuWorkflowError(f"{label} must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                GpuWorkflowError(f"{label} contains non-finite number {item!r}")
            ),
        )
    except GpuWorkflowError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuWorkflowError(f"cannot parse {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise GpuWorkflowError(f"{label} must be a JSON object")
    return artifact, value, hashlib.sha256(raw).hexdigest()


def _exact_keys(value: object, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise GpuWorkflowError(f"{label} must be an object")
    if set(value) != expected:
        raise GpuWorkflowError(
            f"{label} keys mismatch: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise GpuWorkflowError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise GpuWorkflowError(f"{label} must be a finite positive number")
    return float(value)


def _absolute_container_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise GpuWorkflowError(f"{label} is invalid")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise GpuWorkflowError(f"{label} must be an absolute normalized container path")
    return path.as_posix()


def _command_hash(commands: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(commands))).hexdigest()


def _bound_repository_root() -> Path:
    """Return the checkout root containing this workflow, independent of cwd.

    Release validation compares its declared revision with the checkout that supplies
    this code.  Derive that checkout from the resolved module location and verify the
    expected repository layout so neither a launch directory nor a caller-controlled
    path can select a different Git identity.
    """

    try:
        module_path = Path(__file__).resolve(strict=True)
        repository_root = module_path.parents[2]
        expected_module = (
            repository_root / "ingest" / "ingest" / "gpu_workflow.py"
        ).resolve(strict=True)
        git_directory = repository_root / ".git"
        git_directory_mode = git_directory.lstat().st_mode
        git_head = git_directory / "HEAD"
        git_head_mode = git_head.lstat().st_mode
    except (IndexError, OSError) as exc:
        raise GpuWorkflowError(
            "cannot bind the immutable GPU workflow to its repository checkout"
        ) from exc
    if expected_module != module_path:
        raise GpuWorkflowError(
            "immutable GPU workflow module is outside the expected repository layout"
        )
    if stat.S_ISLNK(git_directory_mode) or not stat.S_ISDIR(git_directory_mode):
        raise GpuWorkflowError("repository .git must be a real directory")
    if stat.S_ISLNK(git_head_mode) or not stat.S_ISREG(git_head_mode):
        raise GpuWorkflowError("repository HEAD must be a regular non-symlink file")
    return repository_root


def _commands(paths: Mapping[str, str], *, worker_count: int) -> dict[str, Any]:
    common = [
        "python",
        "-m",
        "ingest",
        "embed",
        "--snapshot-docs",
        paths["snapshot_docs"],
        "--source",
        "all",
    ]
    checksum = [
        *common,
        "--checksum",
        "--checksum-output",
        paths["gpu_checksum"],
    ]
    initialize = [
        *common,
        "--vector-checksum",
        paths["gpu_checksum"],
        "--storage-identity",
        paths["storage_identity"],
        "--qdrant-storage-root",
        paths["qdrant_storage_root"],
        "--batch-size",
        str(REVIEWED_EMBED_BATCH_SIZE),
        "--initialize-workers",
        str(worker_count),
        "--plan",
        paths["plan"],
        "--review",
        paths["review"],
        "--bundle-root",
        paths["release_bundle"],
        "--apply",
    ]
    workers = [
        [
            *common,
            "--vector-checksum",
            paths["gpu_checksum"],
            "--storage-identity",
            paths["storage_identity"],
            "--qdrant-storage-root",
            paths["qdrant_storage_root"],
            "--batch-size",
            str(REVIEWED_EMBED_BATCH_SIZE),
            "--resume",
            "--shard",
            f"{index}/{worker_count}",
            "--plan",
            paths["plan"],
            "--review",
            paths["review"],
            "--bundle-root",
            paths["release_bundle"],
            "--apply",
        ]
        for index in range(worker_count)
    ]
    return {
        "generate_gpu_checksum": checksum,
        "compare_cpu_gpu_checksum": [
            "python",
            "scripts/immutable_gpu_workflow.py",
            "compare-checksums",
            "--cpu",
            paths["cpu_checksum"],
            "--runtime",
            paths["gpu_checksum"],
            "--output",
            paths["checksum_comparison"],
        ],
        "initialize_collection_and_workers": initialize,
        "embed_workers": workers,
        "seal_export": [
            "python",
            "scripts/immutable_gpu_workflow.py",
            "seal-export",
            "--plan",
            paths["plan"],
            "--review",
            paths["review"],
            "--qdrant-storage-root",
            paths["qdrant_storage_root"],
            "--snapshot-output",
            paths["snapshot_export"],
            "--manifest-output",
            paths["export_manifest"],
            "--apply",
        ],
    }


def create_workflow_plan(
    *,
    cfg: Config | None = None,
    bundle_root: str | Path,
    snapshot_docs: str | Path,
    cpu_checksum: str | Path,
    storage_identity: str | Path,
    output: str | Path,
    workflow_id: str,
    container_paths: Mapping[str, str],
    volume_size_gib: int,
    worker_count: int,
    gpu_sku: str,
    gpu_count: int,
    total_hourly_usd: float,
    storage_gib_month_usd: float,
    max_runtime_hours: float,
    max_exposure_usd: float,
    auto_teardown: bool,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Validate release inputs and create the exact, non-executable spend/workflow plan."""

    if not _SAFE_ID_RE.fullmatch(workflow_id):
        raise GpuWorkflowError("workflow_id is unsafe")
    if (
        isinstance(volume_size_gib, bool)
        or not isinstance(volume_size_gib, int)
        or volume_size_gib < MINIMUM_VOLUME_GIB
    ):
        raise GpuWorkflowError(
            f"persistent volume must be at least {MINIMUM_VOLUME_GIB} GiB"
        )
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
        or isinstance(gpu_count, bool)
        or not isinstance(gpu_count, int)
        or gpu_count < 1
        or worker_count != gpu_count
    ):
        raise GpuWorkflowError("worker_count must equal the positive GPU count")
    if not isinstance(gpu_sku, str) or not gpu_sku.strip():
        raise GpuWorkflowError("GPU SKU is required")
    hourly = _positive_number(total_hourly_usd, label="total_hourly_usd")
    storage_rate = _positive_number(
        storage_gib_month_usd, label="storage_gib_month_usd"
    )
    runtime_hours = _positive_number(max_runtime_hours, label="max_runtime_hours")
    exposure = _positive_number(max_exposure_usd, label="max_exposure_usd")
    derived_exposure = hourly * runtime_hours + (
        storage_rate * volume_size_gib * runtime_hours / (24.0 * 30.0)
    )
    if exposure + 1e-9 < derived_exposure:
        raise GpuWorkflowError(
            "maximum dollar exposure is below the bound compute+storage envelope"
        )
    if auto_teardown is not True:
        raise GpuWorkflowError("immutable GPU workflow requires auto_teardown=true")

    validated = validate_release_inputs(
        bundle_root,
        repo_root=_bound_repository_root(),
        environ=environ,
    )
    bundle_manifest_path = validated.root / "manifest.json"
    _, bundle_manifest, observed_manifest_sha = _load_json(
        bundle_manifest_path, label="release manifest"
    )
    if observed_manifest_sha != validated.manifest_sha256:
        raise GpuWorkflowError("validated release manifest changed")
    runtime_identity = _manifest_runtime_identity(
        bundle_manifest, label="release runtime"
    )

    sealed = verify_snapshot_docs(snapshot_docs)
    if sealed.snapshot_id != SNAPSHOT_ID:
        raise GpuWorkflowError("workflow snapshot is not the frozen release snapshot")
    checksum = load_checksum_reference(cpu_checksum)
    if cfg is None:
        raise GpuWorkflowError("workflow plan requires the exact runtime configuration")
    _validate_runtime_configuration(
        cfg,
        manifest=bundle_manifest,
        plan={
            "release": {"configuration_hash": validated.configuration_hash},
            "runtime": runtime_identity,
        },
        label="plan",
    )
    expected_collection_configuration = store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )
    expected_collection_configuration_sha = store.collection_configuration_sha256(
        expected_collection_configuration
    )

    expected_path_keys = {
        "plan",
        "review",
        "snapshot_docs",
        "cpu_checksum",
        "gpu_checksum",
        "checksum_comparison",
        "storage_identity",
        "snapshot_export",
        "export_manifest",
        "release_bundle",
        "launch_evidence_root",
        "qdrant_storage_root",
    }
    _exact_keys(container_paths, expected_path_keys, label="container_paths")
    paths = {
        key: _absolute_container_path(value, label=f"container_paths.{key}")
        for key, value in container_paths.items()
    }
    if paths["release_bundle"] != RELEASE_BUNDLE_PATH:
        raise GpuWorkflowError(
            "container release bundle path is not the frozen candidate bundle"
        )
    launch_root = PurePosixPath(paths["launch_evidence_root"])
    if launch_root.name != "reviewed-launches":
        raise GpuWorkflowError(
            "launch evidence root must be reviewed-launches under the binding root"
        )
    local_storage_identity = Path(storage_identity).expanduser().absolute()
    try:
        local_storage_descriptor = storage_identity_descriptor(local_storage_identity)
    except Exception as exc:  # noqa: BLE001 - identity uncertainty is fatal
        raise GpuWorkflowError(
            f"cannot bind plan storage identity content: {exc}"
        ) from exc
    if local_storage_descriptor is None:  # pragma: no cover - path is required
        raise GpuWorkflowError("plan storage identity content is unavailable")
    storage_content_sha = local_storage_descriptor["content_sha256"]
    commands = _commands(paths, worker_count=worker_count)
    value = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "kind": WORKFLOW_KIND,
        "workflow_id": workflow_id,
        "release": {
            "manifest_sha256": validated.manifest_sha256,
            "snapshot_id": SNAPSHOT_ID,
            "generation_id": GENERATION_ID,
            "physical_collection": PHYSICAL_COLLECTION,
            "configuration_hash": validated.configuration_hash,
        },
        "runtime": runtime_identity,
        "snapshot": {
            "snapshot_id": sealed.snapshot_id,
            "snapshot_sha256": sealed.snapshot_sha256,
            "corpus_sha256": sealed.corpus_sha256,
        },
        "vector_gate": {
            "cpu_artifact_sha256": checksum.file_sha256,
            "cpu_probe_sha256": checksum.probe_sha256,
            "minimum_dense_cosine": MINIMUM_CHECKSUM_COSINE,
        },
        "storage": {
            "identity_content_sha256": storage_content_sha,
            "volume_size_gib": volume_size_gib,
            "isolation": "dedicated-empty-persistent-qdrant-volume",
        },
        "compute": {
            "gpu_sku": gpu_sku.strip(),
            "gpu_count": gpu_count,
            "worker_count": worker_count,
            "total_hourly_usd": hourly,
            "storage_gib_month_usd": storage_rate,
            "max_runtime_hours": runtime_hours,
            "max_exposure_usd": exposure,
            "derived_bound_usd": derived_exposure,
            "auto_teardown": True,
        },
        "collection_configuration": {
            "value": expected_collection_configuration,
            "sha256": expected_collection_configuration_sha,
        },
        "paths": paths,
        "commands": commands,
        "commands_sha256": _command_hash(commands),
    }
    destination = Path(output).expanduser().absolute()
    return _write_create_only(destination, value)


def load_workflow_plan(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    artifact, value, artifact_sha = _load_json(path, label="GPU workflow plan")
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "workflow_id",
            "release",
            "runtime",
            "snapshot",
            "vector_gate",
            "storage",
            "collection_configuration",
            "compute",
            "paths",
            "commands",
            "commands_sha256",
        },
        label="GPU workflow plan",
    )
    if (
        value["schema_version"] != WORKFLOW_SCHEMA_VERSION
        or value["kind"] != WORKFLOW_KIND
        or not isinstance(value["workflow_id"], str)
        or not _SAFE_ID_RE.fullmatch(value["workflow_id"])
    ):
        raise GpuWorkflowError("GPU workflow identity is invalid")
    release = value.get("release")
    _exact_keys(
        release,
        {
            "manifest_sha256",
            "snapshot_id",
            "generation_id",
            "physical_collection",
            "configuration_hash",
        },
        label="workflow.release",
    )
    if (
        release["snapshot_id"] != SNAPSHOT_ID
        or release["generation_id"] != GENERATION_ID
        or release["physical_collection"] != PHYSICAL_COLLECTION
    ):
        raise GpuWorkflowError(
            "workflow release tuple is not frozen candidate identity"
        )
    _sha(release["manifest_sha256"], label="release.manifest_sha256")
    _sha(release["configuration_hash"], label="release.configuration_hash")
    runtime = value.get("runtime")
    _exact_keys(
        runtime,
        {
            "oci_reference",
            "oci_digest",
            "runtime_identity_sha256",
            "runtime_revision",
            "qdrant_revision",
            "embed_device",
            "embed_use_fp16",
            "embed_batch_size",
        },
        label="workflow.runtime",
    )
    if (
        not isinstance(runtime["oci_digest"], str)
        or not _OCI_RE.fullmatch(runtime["oci_digest"])
        or not isinstance(runtime["oci_reference"], str)
        or not runtime["oci_reference"].endswith(runtime["oci_digest"])
    ):
        raise GpuWorkflowError("workflow OCI identity is invalid")
    _sha(runtime["runtime_identity_sha256"], label="runtime.identity_sha256")
    if (
        not isinstance(runtime["runtime_revision"], str)
        or not re.fullmatch(r"[0-9a-f]{7,64}", runtime["runtime_revision"])
        or not isinstance(runtime["qdrant_revision"], str)
        or not runtime["qdrant_revision"]
        or runtime["embed_device"] != "cuda"
        or not isinstance(runtime["embed_use_fp16"], bool)
        or isinstance(runtime["embed_batch_size"], bool)
        or not isinstance(runtime["embed_batch_size"], int)
        or runtime["embed_batch_size"] != REVIEWED_EMBED_BATCH_SIZE
    ):
        raise GpuWorkflowError("workflow runtime/Qdrant revisions are invalid")
    snapshot = value.get("snapshot")
    _exact_keys(
        snapshot,
        {"snapshot_id", "snapshot_sha256", "corpus_sha256"},
        label="workflow.snapshot",
    )
    if snapshot["snapshot_id"] != SNAPSHOT_ID:
        raise GpuWorkflowError("workflow snapshot identity mismatch")
    _sha(snapshot["snapshot_sha256"], label="snapshot.snapshot_sha256")
    _sha(snapshot["corpus_sha256"], label="snapshot.corpus_sha256")
    gate = value.get("vector_gate")
    _exact_keys(
        gate,
        {"cpu_artifact_sha256", "cpu_probe_sha256", "minimum_dense_cosine"},
        label="workflow.vector_gate",
    )
    _sha(gate["cpu_artifact_sha256"], label="vector_gate.cpu_artifact_sha256")
    _sha(gate["cpu_probe_sha256"], label="vector_gate.cpu_probe_sha256")
    if gate["minimum_dense_cosine"] != MINIMUM_CHECKSUM_COSINE:
        raise GpuWorkflowError("workflow weakens the fixed CPU/GPU cosine gate")
    storage = value.get("storage")
    _exact_keys(
        storage,
        {"identity_content_sha256", "volume_size_gib", "isolation"},
        label="workflow.storage",
    )
    _sha(
        storage["identity_content_sha256"],
        label="storage.identity_content_sha256",
    )
    if (
        isinstance(storage["volume_size_gib"], bool)
        or not isinstance(storage["volume_size_gib"], int)
        or storage["volume_size_gib"] < MINIMUM_VOLUME_GIB
        or storage["isolation"] != "dedicated-empty-persistent-qdrant-volume"
    ):
        raise GpuWorkflowError("workflow persistent storage contract is invalid")
    compute = value.get("compute")
    _exact_keys(
        compute,
        {
            "gpu_sku",
            "gpu_count",
            "worker_count",
            "total_hourly_usd",
            "storage_gib_month_usd",
            "max_runtime_hours",
            "max_exposure_usd",
            "derived_bound_usd",
            "auto_teardown",
        },
        label="workflow.compute",
    )
    for field in (
        "total_hourly_usd",
        "storage_gib_month_usd",
        "max_runtime_hours",
        "max_exposure_usd",
        "derived_bound_usd",
    ):
        _positive_number(compute[field], label=f"compute.{field}")
    expected_bound = compute["total_hourly_usd"] * compute["max_runtime_hours"] + (
        compute["storage_gib_month_usd"]
        * storage["volume_size_gib"]
        * compute["max_runtime_hours"]
        / (24.0 * 30.0)
    )
    if (
        not isinstance(compute["gpu_sku"], str)
        or not compute["gpu_sku"].strip()
        or compute["auto_teardown"] is not True
        or isinstance(compute["gpu_count"], bool)
        or not isinstance(compute["gpu_count"], int)
        or isinstance(compute["worker_count"], bool)
        or not isinstance(compute["worker_count"], int)
        or compute["gpu_count"] < 1
        or compute["worker_count"] != compute["gpu_count"]
        or compute["max_exposure_usd"] < compute["derived_bound_usd"]
        or not math.isclose(
            compute["derived_bound_usd"], expected_bound, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise GpuWorkflowError("workflow compute envelope is invalid")
    reviewed_configuration = value.get("collection_configuration")
    _exact_keys(
        reviewed_configuration,
        {"value", "sha256"},
        label="workflow.collection_configuration",
    )
    expected_configuration = store.expected_embed_collection_configuration(
        dense_dim=FROZEN_DENSE_DIM
    )
    if reviewed_configuration["value"] != expected_configuration or _sha(
        reviewed_configuration["sha256"],
        label="workflow.collection_configuration.sha256",
    ) != store.collection_configuration_sha256(expected_configuration):
        raise GpuWorkflowError("workflow collection configuration is not exact")
    paths = value.get("paths")
    if not isinstance(paths, dict):
        raise GpuWorkflowError("workflow.paths must be an object")
    expected_paths = {
        "plan",
        "review",
        "snapshot_docs",
        "cpu_checksum",
        "gpu_checksum",
        "checksum_comparison",
        "storage_identity",
        "snapshot_export",
        "export_manifest",
        "release_bundle",
        "launch_evidence_root",
        "qdrant_storage_root",
    }
    _exact_keys(paths, expected_paths, label="workflow.paths")
    normalized_paths = {
        key: _absolute_container_path(item, label=f"workflow.paths.{key}")
        for key, item in paths.items()
    }
    if normalized_paths["release_bundle"] != RELEASE_BUNDLE_PATH:
        raise GpuWorkflowError(
            "workflow release bundle path is not the frozen candidate bundle"
        )
    if (
        PurePosixPath(normalized_paths["launch_evidence_root"]).name
        != "reviewed-launches"
    ):
        raise GpuWorkflowError(
            "workflow launch evidence root is not immutable/reviewed"
        )
    expected_commands = _commands(
        normalized_paths, worker_count=compute["worker_count"]
    )
    if value.get("commands") != expected_commands:
        raise GpuWorkflowError("workflow commands do not reproduce from bound inputs")
    if value.get("commands_sha256") != _command_hash(expected_commands):
        raise GpuWorkflowError("workflow command hash mismatch")
    return artifact, value, artifact_sha


def create_workflow_review(
    plan: str | Path,
    output: str | Path,
    *,
    reviewer: str,
    reviewed_at: str,
    paid_approval_id: str,
) -> Path:
    """Create the explicit human review/paid-compute authorization sidecar."""

    plan_path, value, plan_sha = load_workflow_plan(plan)
    if not _SAFE_ID_RE.fullmatch(reviewer) or not _SAFE_ID_RE.fullmatch(
        paid_approval_id
    ):
        raise GpuWorkflowError(
            "reviewer and paid_approval_id must be safe non-placeholder IDs"
        )
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise GpuWorkflowError("reviewed_at must be RFC3339") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise GpuWorkflowError("reviewed_at must include a timezone")
    review = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "kind": REVIEW_KIND,
        "workflow_id": value["workflow_id"],
        "workflow_plan": value["paths"]["plan"],
        "workflow_sha256": plan_sha,
        "commands_sha256": value["commands_sha256"],
        "approved_compute": value["compute"],
        "reviewer": reviewer,
        "reviewed_at": timestamp.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "paid_approval_id": paid_approval_id,
        "approved": True,
    }
    destination = Path(output).expanduser().absolute()
    return _write_create_only(destination, review)


def load_workflow_review(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    artifact, review, review_sha = _load_json(path, label="GPU workflow review")
    _exact_keys(
        review,
        {
            "schema_version",
            "kind",
            "workflow_id",
            "workflow_plan",
            "workflow_sha256",
            "commands_sha256",
            "approved_compute",
            "reviewer",
            "reviewed_at",
            "paid_approval_id",
            "approved",
        },
        label="GPU workflow review",
    )
    if (
        review["schema_version"] != WORKFLOW_SCHEMA_VERSION
        or review["kind"] != REVIEW_KIND
        or review["workflow_id"] != plan["workflow_id"]
        or review["workflow_plan"] != plan["paths"]["plan"]
        or review["workflow_sha256"] != plan_sha256
        or review["commands_sha256"] != plan["commands_sha256"]
        or review["approved_compute"] != plan["compute"]
        or review["approved"] is not True
        or not isinstance(review["reviewer"], str)
        or not _SAFE_ID_RE.fullmatch(review["reviewer"])
        or not isinstance(review["paid_approval_id"], str)
        or not _SAFE_ID_RE.fullmatch(review["paid_approval_id"])
    ):
        raise GpuWorkflowError("GPU workflow review does not authorize this exact plan")
    return artifact, review, review_sha


def _runtime_retrieval_knobs(cfg: Config) -> dict[str, object]:
    return {
        "dense_dimension": cfg.dense_dim,
        "chunk_tokens": cfg.chunk_tokens,
        "chunk_overlap": cfg.chunk_overlap,
        "chunk_min_tokens": cfg.chunk_min_tokens,
        "rerank_enabled": cfg.rerank_enabled,
        "rerank_candidates": cfg.rerank_candidates,
        "rerank_min_score": cfg.rerank_min_score,
        "rerank_backend": cfg.rerank_backend,
        "rerank_context_enriched": cfg.rerank_context_enriched,
        "rerank_max_length": cfg.rerank_max_length,
        "citation_route": cfg.citation_route,
        "document_header": cfg.embed_header_v2,
    }


def _manifest_runtime_identity(
    manifest: Mapping[str, Any], *, label: str
) -> dict[str, object]:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise GpuWorkflowError(f"{label} identity is unavailable")
    environment = runtime.get("environment")
    if not isinstance(environment, Mapping):
        raise GpuWorkflowError(f"{label} environment is unavailable")
    oci_digest = runtime.get("oci_digest")
    oci_reference = runtime.get("oci_reference")
    fp16_raw = environment.get("EMBED_USE_FP16")
    batch_raw = environment.get("EMBED_BATCH_SIZE")
    if (
        not isinstance(oci_digest, str)
        or not _OCI_RE.fullmatch(oci_digest)
        or not isinstance(oci_reference, str)
        or not oci_reference.endswith(oci_digest)
        or environment.get("EMBED_DEVICE") != "cuda"
        or fp16_raw not in {"true", "false"}
        or not isinstance(batch_raw, str)
        or not re.fullmatch(r"[1-9][0-9]*", batch_raw)
        or int(batch_raw) != REVIEWED_EMBED_BATCH_SIZE
    ):
        raise GpuWorkflowError(f"{label} OCI/embedding identity is invalid")
    return {
        "oci_reference": oci_reference,
        "oci_digest": oci_digest,
        "runtime_identity_sha256": runtime.get("identity_sha256"),
        "runtime_revision": runtime.get("runtime_revision"),
        "qdrant_revision": runtime.get("qdrant_revision"),
        "embed_device": "cuda",
        "embed_use_fp16": fp16_raw == "true",
        "embed_batch_size": int(batch_raw),
    }


def _validate_runtime_configuration(
    cfg: Config,
    *,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    label: str,
) -> None:
    models = manifest.get("models")
    retrieval = manifest.get("retrieval")
    if not isinstance(models, Mapping) or not isinstance(retrieval, Mapping):
        raise GpuWorkflowError(f"{label} model/configuration provenance is unavailable")
    try:
        expected_models = {
            "embed_model": models["embedding"]["name"],
            "embedding_revision": models["embedding"]["revision"],
            "tokenizer_model": models["tokenizer"]["name"],
            "tokenizer_revision": models["tokenizer"]["revision"],
            "rerank_model": models["reranker"]["name"],
            "reranker_revision": models["reranker"]["revision"],
        }
    except (KeyError, TypeError) as exc:
        raise GpuWorkflowError(f"{label} model provenance is malformed") from exc
    if (
        cfg.generation_id != GENERATION_ID
        or cfg.collection_name != PHYSICAL_COLLECTION
        or cfg.production_mode is not True
        or any(
            getattr(cfg, field) != expected
            for field, expected in expected_models.items()
        )
        or _runtime_retrieval_knobs(cfg) != retrieval.get("knobs")
        or retrieval.get("configuration_hash") != plan["release"]["configuration_hash"]
        or cfg.embed_device != plan["runtime"]["embed_device"]
        or cfg.embed_use_fp16 is not plan["runtime"]["embed_use_fp16"]
        or cfg.embed_batch_size != plan["runtime"]["embed_batch_size"]
    ):
        raise GpuWorkflowError(
            f"{label} model or retrieval configuration differs from reviewed plan"
        )


def _filesystem_identity(
    path: Path, *, label: str, directory: bool
) -> dict[str, object]:
    _require_nonsymlink_path(path, allow_missing=False)
    try:
        info = path.lstat()
        filesystem = os.statvfs(path)
    except OSError as exc:
        raise GpuWorkflowError(f"cannot inspect {label}: {exc}") from exc
    expected_mode = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_mode(info.st_mode) or stat.S_ISLNK(info.st_mode):
        kind = "directory" if directory else "regular file"
        raise GpuWorkflowError(f"{label} must be a real {kind}")
    fsid = getattr(filesystem, "f_fsid", None)
    if isinstance(fsid, bool) or not isinstance(fsid, int):
        raise GpuWorkflowError(f"{label} filesystem identity is unavailable")
    return {
        "path": path.absolute().as_posix(),
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "f_fsid": fsid,
    }


def _storage_runtime_descriptor(
    cfg: Config,
    *,
    storage_identity: str | Path,
    qdrant_storage_root: str | Path,
    launch_evidence_root: str | Path,
) -> dict[str, object]:
    identity_path = Path(storage_identity).expanduser().absolute()
    qdrant_root = Path(qdrant_storage_root).expanduser().absolute()
    checkpoint_root = binding_path(cfg).parent.expanduser().absolute()
    evidence_root = Path(launch_evidence_root).expanduser().absolute()
    expected_evidence_root = checkpoint_root / "reviewed-launches"
    if evidence_root != expected_evidence_root:
        raise GpuWorkflowError(
            "launch evidence root must be reviewed-launches under the embed binding root"
        )
    _create_private_directories(checkpoint_root)
    _create_private_directories(evidence_root)
    try:
        binding_descriptor = storage_identity_descriptor(
            identity_path,
            checkpoint_root=checkpoint_root,
            qdrant_storage_root=qdrant_root,
        )
    except Exception as exc:  # noqa: BLE001 - storage uncertainty is fatal
        raise GpuWorkflowError(f"cannot bind persistent storage: {exc}") from exc
    if binding_descriptor is None:  # pragma: no cover - identity path is required
        raise GpuWorkflowError("persistent storage descriptor is unavailable")
    binding_descriptor_sha = hashlib.sha256(
        _canonical_bytes(binding_descriptor)
    ).hexdigest()
    launch_descriptor = _filesystem_identity(
        evidence_root, label="reviewed launch evidence root", directory=True
    )
    identity_descriptor = binding_descriptor["identity_file"]
    if (
        launch_descriptor["st_dev"] != identity_descriptor["st_dev"]
        or launch_descriptor["f_fsid"] != identity_descriptor["f_fsid"]
    ):
        raise GpuWorkflowError(
            "storage identity, checkpoints, launch evidence, and Qdrant data must share "
            "one persistent filesystem"
        )
    descriptor = {
        "binding_descriptor": binding_descriptor,
        "binding_descriptor_sha256": binding_descriptor_sha,
        "launch_evidence_root": launch_descriptor,
    }
    descriptor["descriptor_sha256"] = hashlib.sha256(
        _canonical_bytes(descriptor)
    ).hexdigest()
    return descriptor


def _launch_evidence_path(
    plan: Mapping[str, Any],
    *,
    initialize_workers: int | None,
    shard: tuple[int, int] | None,
) -> Path:
    root = Path(plan["paths"]["launch_evidence_root"])
    if initialize_workers is not None:
        return root / "initialize.json"
    if shard is None:  # pragma: no cover - caller validates the selector first
        raise GpuWorkflowError("worker launch lacks a shard")
    return root / f"worker-{shard[0]:05d}-of-{shard[1]:05d}.json"


def _write_or_validate_launch_evidence(path: Path, value: Mapping[str, Any]) -> Path:
    if not os.path.lexists(path):
        try:
            return _write_create_only(path, value)
        except GpuWorkflowError:
            # A simultaneous exact launch may have won O_EXCL.  Re-read below; only
            # byte-equivalent canonical evidence is reusable and nothing is replaced.
            if not os.path.lexists(path):
                raise
    _artifact, observed, _sha256 = _load_json(path, label="reviewed launch evidence")
    if observed != value:
        raise GpuWorkflowError(
            "existing reviewed launch evidence differs from this launch"
        )
    return path.absolute()


def _require_matching_initialization_evidence(
    plan: Mapping[str, Any],
    *,
    plan_sha256: str,
    review_sha256: str,
    vector_gate: Mapping[str, object],
    storage_descriptor: Mapping[str, object],
) -> None:
    worker_count = plan["compute"]["worker_count"]
    path = _launch_evidence_path(plan, initialize_workers=worker_count, shard=None)
    _artifact, value, _artifact_sha = _load_json(
        path, label="reviewed initialization evidence"
    )
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "workflow_id",
            "workflow_plan_sha256",
            "workflow_review_sha256",
            "commands_sha256",
            "launch_type",
            "worker",
            "command",
            "command_sha256",
            "release",
            "runtime",
            "snapshot",
            "vector_gate",
            "storage",
            "collection_configuration",
        },
        label="reviewed initialization evidence",
    )
    command = plan["commands"]["initialize_collection_and_workers"]
    if (
        value["schema_version"] != WORKFLOW_SCHEMA_VERSION
        or value["kind"] != LAUNCH_KIND
        or value["workflow_id"] != plan["workflow_id"]
        or value["workflow_plan_sha256"] != plan_sha256
        or value["workflow_review_sha256"] != review_sha256
        or value["commands_sha256"] != plan["commands_sha256"]
        or value["launch_type"] != "initialize"
        or value["worker"] != {"worker_id": None, "worker_count": worker_count}
        or value["command"] != command
        or value["command_sha256"]
        != hashlib.sha256(_canonical_bytes(command)).hexdigest()
        or value["release"] != plan["release"]
        or value["runtime"] != plan["runtime"]
        or value["snapshot"] != plan["snapshot"]
        or value["vector_gate"] != vector_gate
        or value["storage"] != storage_descriptor
        or value["collection_configuration"] != plan["collection_configuration"]
    ):
        raise GpuWorkflowError(
            "worker persistent/runtime evidence differs from immutable initialization"
        )


def validate_reviewed_launch(
    *,
    plan_path: str | Path,
    review_path: str | Path,
    bundle_root: str | Path,
    cfg: Config,
    snapshot_docs: str | Path,
    vector_checksum: str | Path,
    storage_identity: str | Path,
    qdrant_storage_root: str | Path,
    initialize_workers: int | None,
    resume: bool,
    shard: tuple[int, int] | None,
    source: str,
    batch_size: int,
    apply: bool,
    environ: Mapping[str, str] | None = None,
) -> ReviewedLaunchAuthorization:
    """Validate and attest one exact reviewed init/worker launch without Qdrant access."""

    plan_file, plan, plan_sha = load_workflow_plan(plan_path)
    review_file, _review, review_sha = load_workflow_review(
        review_path, plan=plan, plan_sha256=plan_sha
    )
    if (
        source != "all"
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size != REVIEWED_EMBED_BATCH_SIZE
        or apply is not True
    ):
        raise GpuWorkflowError(
            "launch source, batch size, or apply mode differs from reviewed command"
        )
    if (
        plan_file.as_posix() != plan["paths"]["plan"]
        or review_file.as_posix() != plan["paths"]["review"]
    ):
        raise GpuWorkflowError("launch plan/review paths differ from reviewed paths")
    bundle = Path(bundle_root).expanduser().absolute()
    if bundle.as_posix() != plan["paths"]["release_bundle"]:
        raise GpuWorkflowError("launch release bundle path differs from reviewed path")
    validated = validate_release_inputs(
        bundle,
        repo_root=_bound_repository_root(),
        environ=environ,
    )
    manifest_path, manifest, manifest_sha = _load_json(
        validated.root / "manifest.json", label="launch release manifest"
    )
    if manifest_path.parent != bundle or manifest_sha != validated.manifest_sha256:
        raise GpuWorkflowError("launch release manifest changed during validation")
    expected_release = {
        "manifest_sha256": validated.manifest_sha256,
        "snapshot_id": SNAPSHOT_ID,
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "configuration_hash": validated.configuration_hash,
    }
    if plan["release"] != expected_release:
        raise GpuWorkflowError("launch release identity differs from reviewed plan")
    expected_runtime = _manifest_runtime_identity(
        manifest, label="launch release runtime"
    )
    if plan["runtime"] != expected_runtime:
        raise GpuWorkflowError("launch OCI/runtime identity differs from reviewed plan")
    _validate_runtime_configuration(cfg, manifest=manifest, plan=plan, label="launch")
    actual_snapshot_path = Path(snapshot_docs).expanduser().absolute().as_posix()
    if actual_snapshot_path != plan["paths"]["snapshot_docs"]:
        raise GpuWorkflowError("launch snapshot path differs from reviewed path")
    sealed = verify_snapshot_docs(snapshot_docs)
    if {
        "snapshot_id": sealed.snapshot_id,
        "snapshot_sha256": sealed.snapshot_sha256,
        "corpus_sha256": sealed.corpus_sha256,
    } != plan["snapshot"]:
        raise GpuWorkflowError("launch snapshot identity differs from reviewed plan")
    checksum_path = Path(vector_checksum).expanduser().absolute().as_posix()
    if checksum_path != plan["paths"]["gpu_checksum"]:
        raise GpuWorkflowError("launch vector checksum path differs from reviewed path")
    cpu = load_checksum_reference(plan["paths"]["cpu_checksum"])
    runtime_checksum = load_checksum_reference(vector_checksum)
    comparison = load_checksum_comparison(
        plan["paths"]["checksum_comparison"], cpu=cpu, runtime=runtime_checksum
    )
    if (
        cpu.file_sha256 != plan["vector_gate"]["cpu_artifact_sha256"]
        or cpu.probe_sha256 != plan["vector_gate"]["cpu_probe_sha256"]
        or comparison.minimum_cosine != MINIMUM_CHECKSUM_COSINE
    ):
        raise GpuWorkflowError("launch vector checksum gate differs from reviewed plan")
    identity_path = Path(storage_identity).expanduser().absolute()
    qdrant_root = Path(qdrant_storage_root).expanduser().absolute()
    if (
        identity_path.as_posix() != plan["paths"]["storage_identity"]
        or qdrant_root.as_posix() != plan["paths"]["qdrant_storage_root"]
    ):
        raise GpuWorkflowError("launch persistent storage differs from reviewed plan")
    worker_count = plan["compute"]["worker_count"]
    if initialize_workers is not None:
        if initialize_workers != worker_count or resume or shard is not None:
            raise GpuWorkflowError("reviewed initialization selector is invalid")
        launch_type = "initialize"
        worker = {"worker_id": None, "worker_count": worker_count}
        command = plan["commands"]["initialize_collection_and_workers"]
    else:
        if (
            not resume
            or shard is None
            or shard[1] != worker_count
            or not 0 <= shard[0] < worker_count
        ):
            raise GpuWorkflowError("reviewed worker shard selector is invalid")
        launch_type = "worker"
        worker = {"worker_id": shard[0], "worker_count": shard[1]}
        command = plan["commands"]["embed_workers"][shard[0]]
    storage_descriptor = _storage_runtime_descriptor(
        cfg,
        storage_identity=identity_path,
        qdrant_storage_root=qdrant_root,
        launch_evidence_root=plan["paths"]["launch_evidence_root"],
    )
    if (
        storage_descriptor["binding_descriptor"]["content_sha256"]
        != plan["storage"]["identity_content_sha256"]
    ):
        raise GpuWorkflowError("launch persistent storage differs from reviewed plan")
    reviewed_vector_gate = {
        "cpu_artifact_sha256": cpu.file_sha256,
        "cpu_probe_sha256": cpu.probe_sha256,
        "runtime_artifact_sha256": runtime_checksum.file_sha256,
        "runtime_probe_sha256": runtime_checksum.probe_sha256,
        "comparison_sha256": comparison.file_sha256,
        "minimum_dense_cosine": comparison.minimum_cosine,
    }
    if launch_type == "worker":
        _require_matching_initialization_evidence(
            plan,
            plan_sha256=plan_sha,
            review_sha256=review_sha,
            vector_gate=reviewed_vector_gate,
            storage_descriptor=storage_descriptor,
        )
    value = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "kind": LAUNCH_KIND,
        "workflow_id": plan["workflow_id"],
        "workflow_plan_sha256": plan_sha,
        "workflow_review_sha256": review_sha,
        "commands_sha256": plan["commands_sha256"],
        "launch_type": launch_type,
        "worker": worker,
        "command": command,
        "command_sha256": hashlib.sha256(_canonical_bytes(command)).hexdigest(),
        "release": dict(plan["release"]),
        "runtime": dict(plan["runtime"]),
        "snapshot": dict(plan["snapshot"]),
        "vector_gate": reviewed_vector_gate,
        "storage": storage_descriptor,
        "collection_configuration": dict(plan["collection_configuration"]),
    }
    evidence_path = _launch_evidence_path(
        plan, initialize_workers=initialize_workers, shard=shard
    )
    evidence_path = _write_or_validate_launch_evidence(evidence_path, value)
    _artifact, observed_evidence, evidence_sha = _load_json(
        evidence_path, label="reviewed launch evidence"
    )
    if observed_evidence != value:
        raise GpuWorkflowError("reviewed launch evidence changed before authorization")
    operations = (
        frozenset({"create", "recover"})
        if launch_type == "initialize"
        else frozenset({"upsert"})
    )
    capability = store._issue_reviewed_mutation_capability(
        generation_id=GENERATION_ID,
        collection_name=PHYSICAL_COLLECTION,
        reviewed_plan_sha256=plan_sha,
        launch_evidence_sha256=evidence_sha,
        collection_configuration_digest=plan["collection_configuration"]["sha256"],
        storage_identity_sha256=storage_descriptor["binding_descriptor_sha256"],
        allowed_operations=operations,
    )
    return ReviewedLaunchAuthorization(
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha,
        mutation_capability=capability,
    )


def _load_reviewed_launch_set(
    plan: Mapping[str, Any],
    *,
    plan_sha256: str,
    review_sha256: str,
    vector_gate: Mapping[str, object],
    storage_descriptor: Mapping[str, object],
) -> dict[str, object]:
    worker_count = plan["compute"]["worker_count"]
    selectors: list[tuple[int | None, tuple[int, int] | None]] = [(worker_count, None)]
    selectors.extend((None, (index, worker_count)) for index in range(worker_count))
    rows: list[dict[str, object]] = []
    for initialize_workers, shard in selectors:
        evidence_path = _launch_evidence_path(
            plan, initialize_workers=initialize_workers, shard=shard
        )
        artifact, value, artifact_sha = _load_json(
            evidence_path, label="reviewed launch evidence"
        )
        _exact_keys(
            value,
            {
                "schema_version",
                "kind",
                "workflow_id",
                "workflow_plan_sha256",
                "workflow_review_sha256",
                "commands_sha256",
                "launch_type",
                "worker",
                "command",
                "command_sha256",
                "release",
                "runtime",
                "snapshot",
                "vector_gate",
                "storage",
                "collection_configuration",
            },
            label="reviewed launch evidence",
        )
        if initialize_workers is not None:
            expected_type = "initialize"
            expected_worker = {"worker_id": None, "worker_count": worker_count}
            expected_command = plan["commands"]["initialize_collection_and_workers"]
        else:
            if shard is None:  # pragma: no cover - constructed above
                raise GpuWorkflowError("reviewed launch selector is unavailable")
            expected_type = "worker"
            expected_worker = {"worker_id": shard[0], "worker_count": shard[1]}
            expected_command = plan["commands"]["embed_workers"][shard[0]]
        if (
            value["schema_version"] != WORKFLOW_SCHEMA_VERSION
            or value["kind"] != LAUNCH_KIND
            or value["workflow_id"] != plan["workflow_id"]
            or value["workflow_plan_sha256"] != plan_sha256
            or value["workflow_review_sha256"] != review_sha256
            or value["commands_sha256"] != plan["commands_sha256"]
            or value["launch_type"] != expected_type
            or value["worker"] != expected_worker
            or value["command"] != expected_command
            or value["command_sha256"]
            != hashlib.sha256(_canonical_bytes(expected_command)).hexdigest()
            or value["release"] != plan["release"]
            or value["runtime"] != plan["runtime"]
            or value["snapshot"] != plan["snapshot"]
            or value["vector_gate"] != vector_gate
            or value["storage"] != storage_descriptor
            or value["collection_configuration"] != plan["collection_configuration"]
        ):
            raise GpuWorkflowError(
                "reviewed launch evidence does not match the exact plan/review/runtime"
            )
        rows.append(
            {
                "path": artifact.as_posix(),
                "sha256": artifact_sha,
                "launch_type": expected_type,
                "worker": expected_worker,
                "command_sha256": value["command_sha256"],
            }
        )
    return {
        "initialization": rows[0],
        "workers": rows[1:],
        "aggregate_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _point_parts(point: Any) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    point_id = getattr(point, "id", None)
    payload = getattr(point, "payload", None)
    vectors = getattr(point, "vector", None)
    if isinstance(point, Mapping):
        point_id = point.get("id")
        payload = point.get("payload")
        vectors = point.get("vector", point.get("vectors"))
    if (
        not isinstance(point_id, (str, int))
        or not isinstance(payload, Mapping)
        or not isinstance(vectors, Mapping)
    ):
        raise GpuWorkflowError("Qdrant point lacks id/payload/named vectors")
    return str(point_id), payload, vectors


def scan_collection_seal(
    client: Any,
    collection_name: str,
    *,
    page_size: int = 256,
) -> CollectionSeal:
    """Read the exact physical collection twice around a sorted, disk-backed digest scan."""

    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= 10_000
    ):
        raise GpuWorkflowError("page_size must be in 1..10000")
    try:
        initial_info = client.get_collection(collection_name)
        initial_configuration = store.collection_configuration(initial_info)
        initial_configuration_sha = store.collection_configuration_sha256(
            initial_configuration
        )
        initial_count = getattr(initial_info, "points_count")
    except Exception as exc:  # noqa: BLE001 - unavailable identity is fatal
        raise GpuWorkflowError(f"cannot inspect remote collection: {exc}") from exc
    if (
        isinstance(initial_count, bool)
        or not isinstance(initial_count, int)
        or initial_count < 0
    ):
        raise GpuWorkflowError("collection points_count is invalid")

    with tempfile.TemporaryDirectory(prefix="collection-seal-") as temp:
        connection = sqlite3.connect(Path(temp) / "points.sqlite")
        connection.execute(
            "CREATE TABLE points (point_id TEXT PRIMARY KEY, point_sha256 TEXT NOT NULL)"
        )
        offset: Any = None
        seen_offsets: set[str] = set()
        scanned = 0
        while True:
            try:
                page = client.scroll(
                    collection_name=collection_name,
                    offset=offset,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise GpuWorkflowError(
                    f"collection digest scroll failed: {exc}"
                ) from exc
            if not isinstance(page, tuple) or len(page) != 2:
                raise GpuWorkflowError(
                    "collection digest scroll returned an invalid page"
                )
            points, next_offset = page
            if not points and next_offset is not None:
                raise GpuWorkflowError(
                    "collection digest scroll returned an empty continuation"
                )
            for point in points:
                point_id, payload, vectors = _point_parts(point)
                try:
                    point_sha = point_content_sha256(point_id, payload, vectors)
                    connection.execute(
                        "INSERT INTO points VALUES (?, ?)", (point_id, point_sha)
                    )
                except (ValueError, sqlite3.IntegrityError) as exc:
                    raise GpuWorkflowError(
                        f"collection point {point_id!r} cannot be sealed: {exc}"
                    ) from exc
                scanned += 1
            connection.commit()
            if next_offset is None:
                break
            marker = repr(next_offset)
            if marker in seen_offsets or next_offset == offset:
                raise GpuWorkflowError("collection digest scroll continuation cycled")
            seen_offsets.add(marker)
            offset = next_offset
        try:
            collection_sha, digest_count = whole_collection_sha256(
                (str(point_id), str(point_sha))
                for point_id, point_sha in connection.execute(
                    "SELECT point_id, point_sha256 FROM points ORDER BY point_id"
                )
            )
        except ValueError as exc:
            raise GpuWorkflowError(f"cannot compose collection digest: {exc}") from exc
        finally:
            connection.close()
    if scanned != initial_count or digest_count != initial_count:
        raise GpuWorkflowError("collection point count differs from full digest scan")
    try:
        final_info = client.get_collection(collection_name)
        final_configuration = store.collection_configuration(final_info)
        final_configuration_sha = store.collection_configuration_sha256(
            final_configuration
        )
        final_count = getattr(final_info, "points_count")
    except Exception as exc:  # noqa: BLE001
        raise GpuWorkflowError(
            f"cannot re-inspect collection after digest: {exc}"
        ) from exc
    if (
        final_count != initial_count
        or final_configuration != initial_configuration
        or final_configuration_sha != initial_configuration_sha
    ):
        raise GpuWorkflowError("collection changed during whole-collection digest scan")
    return CollectionSeal(
        point_count=initial_count,
        collection_sha256=collection_sha,
        collection_configuration=initial_configuration,
        collection_configuration_sha256=initial_configuration_sha,
    )


def _alias_inventory(client: Any) -> tuple[tuple[str, str], ...]:
    try:
        response = client.get_aliases()
        aliases = getattr(response, "aliases", None)
        if aliases is None and isinstance(response, Mapping):
            aliases = response.get("aliases")
    except Exception as exc:  # noqa: BLE001
        raise GpuWorkflowError(f"cannot inspect Qdrant aliases: {exc}") from exc
    if aliases is None:
        raise GpuWorkflowError("Qdrant alias inventory is unavailable")
    output: list[tuple[str, str]] = []
    for alias in aliases:
        name = getattr(alias, "alias_name", None)
        target = getattr(alias, "collection_name", None)
        if isinstance(alias, Mapping):
            name = alias.get("alias_name")
            target = alias.get("collection_name")
        if not isinstance(name, str) or not isinstance(target, str):
            raise GpuWorkflowError("Qdrant alias inventory contains an invalid row")
        output.append((name, target))
    return tuple(sorted(output))


def _download_snapshot(
    cfg: Config,
    collection_name: str,
    snapshot_name: str,
    output: Path,
) -> tuple[str, int]:
    base = cfg.qdrant_url.rstrip("/")
    url = (
        f"{base}/collections/{urllib.parse.quote(collection_name, safe='')}/snapshots/"
        f"{urllib.parse.quote(snapshot_name, safe='')}"
    )
    headers = {"api-key": cfg.qdrant_api_key} if cfg.qdrant_api_key else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError as exc:
        raise GpuWorkflowError(f"snapshot export already exists: {output}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:  # noqa: S310
            if getattr(response, "status", 200) != 200:
                raise GpuWorkflowError(
                    "Qdrant snapshot download did not return HTTP 200"
                )
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    if size < 1:
        output.unlink(missing_ok=True)
        raise GpuWorkflowError("Qdrant snapshot export is empty")
    return digest.hexdigest(), size


def seal_remote_export(
    client: Any,
    cfg: Config,
    *,
    plan_path: str | Path,
    review_path: str | Path,
    qdrant_storage_root: str | Path,
    snapshot_output: str | Path,
    manifest_output: str | Path,
    downloader: Callable[[Config, str, str, Path], tuple[str, int]] | None = None,
) -> Path:
    """Prove completed remote state, create/download one snapshot, and seal its manifest."""

    plan_file, plan, plan_sha = load_workflow_plan(plan_path)
    review_file, _review, review_sha = load_workflow_review(
        review_path, plan=plan, plan_sha256=plan_sha
    )
    if (
        plan_file.as_posix() != plan["paths"]["plan"]
        or review_file.as_posix() != plan["paths"]["review"]
    ):
        raise GpuWorkflowError(
            "remote plan/review paths differ from reviewed container paths"
        )
    if cfg.generation_id != GENERATION_ID or cfg.collection_name != PHYSICAL_COLLECTION:
        raise GpuWorkflowError(
            "runtime configuration is not the frozen generation target"
        )
    validated = validate_release_inputs(
        plan["paths"]["release_bundle"],
        repo_root=_bound_repository_root(),
    )
    _, release_manifest, release_manifest_sha = _load_json(
        validated.root / "manifest.json", label="export release manifest"
    )
    expected_release = {
        "manifest_sha256": validated.manifest_sha256,
        "snapshot_id": SNAPSHOT_ID,
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "configuration_hash": validated.configuration_hash,
    }
    expected_runtime = _manifest_runtime_identity(
        release_manifest, label="export release runtime"
    )
    if (
        release_manifest_sha != validated.manifest_sha256
        or plan["release"] != expected_release
        or plan["runtime"] != expected_runtime
    ):
        raise GpuWorkflowError(
            "export release/runtime/configuration differs from reviewed plan"
        )
    _validate_runtime_configuration(
        cfg, manifest=release_manifest, plan=plan, label="export"
    )
    snapshot_destination = Path(snapshot_output).expanduser().absolute()
    manifest_destination = Path(manifest_output).expanduser().absolute()
    if (
        snapshot_destination.as_posix() != plan["paths"]["snapshot_export"]
        or manifest_destination.as_posix() != plan["paths"]["export_manifest"]
    ):
        raise GpuWorkflowError("export paths differ from the reviewed workflow")
    if os.path.lexists(snapshot_destination) or os.path.lexists(manifest_destination):
        raise GpuWorkflowError("snapshot export destinations must both be absent")
    qdrant_root = Path(qdrant_storage_root).expanduser().absolute()
    if qdrant_root.as_posix() != plan["paths"]["qdrant_storage_root"]:
        raise GpuWorkflowError("Qdrant storage root differs from reviewed workflow")
    sealed_snapshot = verify_snapshot_docs(plan["paths"]["snapshot_docs"])
    if {
        "snapshot_id": sealed_snapshot.snapshot_id,
        "snapshot_sha256": sealed_snapshot.snapshot_sha256,
        "corpus_sha256": sealed_snapshot.corpus_sha256,
    } != plan["snapshot"]:
        raise GpuWorkflowError("sealed snapshot differs from reviewed workflow")
    cpu = load_checksum_reference(plan["paths"]["cpu_checksum"])
    runtime = load_checksum_reference(plan["paths"]["gpu_checksum"])
    if (
        cpu.file_sha256 != plan["vector_gate"]["cpu_artifact_sha256"]
        or cpu.probe_sha256 != plan["vector_gate"]["cpu_probe_sha256"]
    ):
        raise GpuWorkflowError("CPU vector checksum differs from reviewed workflow")
    comparison = load_checksum_comparison(
        plan["paths"]["checksum_comparison"], cpu=cpu, runtime=runtime
    )
    if comparison.minimum_cosine != MINIMUM_CHECKSUM_COSINE:
        raise GpuWorkflowError("CPU/GPU checksum comparison threshold is not exact")
    reviewed_vector_gate = {
        "cpu_artifact_sha256": cpu.file_sha256,
        "cpu_probe_sha256": cpu.probe_sha256,
        "runtime_artifact_sha256": runtime.file_sha256,
        "runtime_probe_sha256": runtime.probe_sha256,
        "comparison_sha256": comparison.file_sha256,
        "minimum_dense_cosine": comparison.minimum_cosine,
    }
    storage_descriptor = _storage_runtime_descriptor(
        cfg,
        storage_identity=plan["paths"]["storage_identity"],
        qdrant_storage_root=qdrant_root,
        launch_evidence_root=plan["paths"]["launch_evidence_root"],
    )
    storage_sha = storage_descriptor["binding_descriptor_sha256"]
    if (
        storage_descriptor["binding_descriptor"]["content_sha256"]
        != plan["storage"]["identity_content_sha256"]
    ):
        raise GpuWorkflowError(
            "persistent storage identity differs from reviewed workflow"
        )
    reviewed_launch = _load_reviewed_launch_set(
        plan,
        plan_sha256=plan_sha,
        review_sha256=review_sha,
        vector_gate=reviewed_vector_gate,
        storage_descriptor=storage_descriptor,
    )
    aliases_before = _alias_inventory(client)
    store.refuse_aliased_write_target(client, cfg.collection_name)
    info = client.get_collection(cfg.collection_name)
    configuration = store.collection_configuration(info)
    configuration_sha = store.collection_configuration_sha256(configuration)
    if (
        configuration != plan["collection_configuration"]["value"]
        or configuration_sha != plan["collection_configuration"]["sha256"]
    ):
        raise GpuWorkflowError(
            "remote collection configuration differs from reviewed creation profile"
        )
    worker_count = plan["compute"]["worker_count"]
    binding = prepare_binding(
        cfg,
        sealed_snapshot,
        checksum=runtime,
        worker_count=worker_count,
        collection_configuration=configuration,
        collection_configuration_sha256=configuration_sha,
        storage_identity_sha256=storage_sha,
        reviewed_plan_sha256=plan_sha,
        resume=True,
    )
    load_coordinator(binding)
    checkpoints: list[Mapping[str, Any]] = []
    for shard_index in range(worker_count):
        shard_checkpoints = preflight_checkpoints(
            binding,
            SOURCES,
            (shard_index, worker_count),
            resume=True,
        )
        if any(
            checkpoint["complete"] is not True
            for checkpoint in shard_checkpoints.values()
        ):
            raise GpuWorkflowError("not every reviewed embed checkpoint is complete")
        checkpoints.extend(shard_checkpoints.values())
    acknowledged = sum(int(item["chunks_completed"]) for item in checkpoints)
    store.prepare_embed_collection(
        client,
        cfg,
        resume=True,
        recreate=False,
        apply=True,
        minimum_points=acknowledged,
    )
    remote_seal = scan_collection_seal(client, cfg.collection_name)
    if remote_seal.point_count != acknowledged:
        raise GpuWorkflowError("checkpoint chunk total differs from sealed collection")
    try:
        snapshot = client.create_snapshot(
            collection_name=cfg.collection_name, wait=True
        )
    except Exception as exc:  # noqa: BLE001 - export must not continue ambiguously
        raise GpuWorkflowError(f"Qdrant snapshot creation failed: {exc}") from exc
    snapshot_name = getattr(snapshot, "name", None)
    snapshot_size = getattr(snapshot, "size", None)
    snapshot_checksum = getattr(snapshot, "checksum", None)
    if (
        not isinstance(snapshot_name, str)
        or not snapshot_name
        or isinstance(snapshot_size, bool)
        or not isinstance(snapshot_size, int)
        or snapshot_size < 1
    ):
        raise GpuWorkflowError("Qdrant returned an invalid snapshot description")
    _create_private_directories(snapshot_destination.parent)
    snapshot_sha, downloaded_size = (downloader or _download_snapshot)(
        cfg, cfg.collection_name, snapshot_name, snapshot_destination
    )
    if downloaded_size != snapshot_size:
        raise GpuWorkflowError(
            "downloaded snapshot size differs from Qdrant description"
        )
    aliases_after = _alias_inventory(client)
    if aliases_after != aliases_before:
        raise GpuWorkflowError("Qdrant aliases changed during snapshot export")
    export = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "kind": EXPORT_KIND,
        "workflow_id": plan["workflow_id"],
        "workflow_plan_sha256": plan_sha,
        "workflow_review_sha256": review_sha,
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "snapshot_id": SNAPSHOT_ID,
        "run_evidence": {
            "release": dict(plan["release"]),
            "runtime": dict(plan["runtime"]),
            "snapshot": dict(plan["snapshot"]),
            "vector_gate": reviewed_vector_gate,
            "storage": storage_descriptor,
            "collection_configuration": dict(plan["collection_configuration"]),
            "reviewed_launch": reviewed_launch,
        },
        "qdrant_snapshot": {
            "name": snapshot_name,
            "reported_size_bytes": snapshot_size,
            "reported_checksum": snapshot_checksum,
            "export_sha256": snapshot_sha,
            "export_size_bytes": downloaded_size,
        },
        "remote_collection": {
            "point_count": remote_seal.point_count,
            "collection_sha256": remote_seal.collection_sha256,
            "collection_configuration_sha256": (
                remote_seal.collection_configuration_sha256
            ),
            "collection_configuration": dict(remote_seal.collection_configuration),
        },
        "embed_binding_sha256": binding_sha256(binding),
        "runtime_vector_checksum_artifact_sha256": runtime.file_sha256,
        "runtime_vector_probe_sha256": runtime.probe_sha256,
        "checksum_comparison_sha256": comparison.file_sha256,
        "storage_identity_content_sha256": plan["storage"]["identity_content_sha256"],
        "storage_identity_sha256": storage_sha,
        "aliases_sha256": hashlib.sha256(
            _canonical_bytes(list(aliases_before))
        ).hexdigest(),
    }
    return _write_create_only(manifest_destination, export)


def load_export_manifest(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    review: Mapping[str, Any],
    review_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    artifact, value, artifact_sha = _load_json(path, label="Qdrant export manifest")
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "workflow_id",
            "workflow_plan_sha256",
            "workflow_review_sha256",
            "generation_id",
            "physical_collection",
            "snapshot_id",
            "run_evidence",
            "qdrant_snapshot",
            "remote_collection",
            "embed_binding_sha256",
            "runtime_vector_checksum_artifact_sha256",
            "runtime_vector_probe_sha256",
            "checksum_comparison_sha256",
            "storage_identity_content_sha256",
            "storage_identity_sha256",
            "aliases_sha256",
        },
        label="Qdrant export manifest",
    )
    if (
        value["schema_version"] != WORKFLOW_SCHEMA_VERSION
        or value["kind"] != EXPORT_KIND
        or value["workflow_id"] != plan["workflow_id"]
        or review["workflow_id"] != plan["workflow_id"]
        or review["workflow_sha256"] != plan_sha256
        or review["commands_sha256"] != plan["commands_sha256"]
        or review["approved_compute"] != plan["compute"]
        or review["approved"] is not True
        or value["workflow_plan_sha256"] != plan_sha256
        or value["workflow_review_sha256"] != review_sha256
        or value["generation_id"] != GENERATION_ID
        or value["physical_collection"] != PHYSICAL_COLLECTION
        or value["snapshot_id"] != SNAPSHOT_ID
    ):
        raise GpuWorkflowError("Qdrant export manifest identity mismatch")
    run_evidence = value.get("run_evidence")
    _exact_keys(
        run_evidence,
        {
            "release",
            "runtime",
            "snapshot",
            "vector_gate",
            "storage",
            "collection_configuration",
            "reviewed_launch",
        },
        label="export.run_evidence",
    )
    if (
        run_evidence["release"] != plan["release"]
        or run_evidence["runtime"] != plan["runtime"]
        or run_evidence["snapshot"] != plan["snapshot"]
        or run_evidence["collection_configuration"] != plan["collection_configuration"]
    ):
        raise GpuWorkflowError("export run evidence differs from reviewed identities")
    vector_gate = run_evidence["vector_gate"]
    _exact_keys(
        vector_gate,
        {
            "cpu_artifact_sha256",
            "cpu_probe_sha256",
            "runtime_artifact_sha256",
            "runtime_probe_sha256",
            "comparison_sha256",
            "minimum_dense_cosine",
        },
        label="export.run_evidence.vector_gate",
    )
    for field in (
        "cpu_artifact_sha256",
        "cpu_probe_sha256",
        "runtime_artifact_sha256",
        "runtime_probe_sha256",
        "comparison_sha256",
    ):
        _sha(vector_gate[field], label=f"export.run_evidence.vector_gate.{field}")
    if (
        vector_gate["cpu_artifact_sha256"] != plan["vector_gate"]["cpu_artifact_sha256"]
        or vector_gate["cpu_probe_sha256"] != plan["vector_gate"]["cpu_probe_sha256"]
        or vector_gate["minimum_dense_cosine"] != MINIMUM_CHECKSUM_COSINE
        or vector_gate["runtime_artifact_sha256"]
        != value["runtime_vector_checksum_artifact_sha256"]
        or vector_gate["runtime_probe_sha256"] != value["runtime_vector_probe_sha256"]
        or vector_gate["comparison_sha256"] != value["checksum_comparison_sha256"]
    ):
        raise GpuWorkflowError("export vector evidence is not plan-bound")
    storage_evidence = run_evidence["storage"]
    _exact_keys(
        storage_evidence,
        {
            "binding_descriptor",
            "binding_descriptor_sha256",
            "launch_evidence_root",
            "descriptor_sha256",
        },
        label="export.run_evidence.storage",
    )
    descriptor_material = dict(storage_evidence)
    descriptor_sha = descriptor_material.pop("descriptor_sha256")
    binding_descriptor = storage_evidence["binding_descriptor"]
    _exact_keys(
        binding_descriptor,
        {
            "schema_version",
            "content_sha256",
            "identity_file",
            "checkpoint_root",
            "qdrant_storage_root",
        },
        label="export.run_evidence.storage.binding_descriptor",
    )
    binding_descriptor_sha = hashlib.sha256(
        _canonical_bytes(binding_descriptor)
    ).hexdigest()
    if (
        _sha(descriptor_sha, label="export.run_evidence.storage.descriptor_sha256")
        != hashlib.sha256(_canonical_bytes(descriptor_material)).hexdigest()
        or _sha(
            storage_evidence["binding_descriptor_sha256"],
            label="export.run_evidence.storage.binding_descriptor_sha256",
        )
        != binding_descriptor_sha
        or binding_descriptor_sha != value["storage_identity_sha256"]
        or binding_descriptor["content_sha256"]
        != plan["storage"]["identity_content_sha256"]
        or binding_descriptor["content_sha256"]
        != value["storage_identity_content_sha256"]
    ):
        raise GpuWorkflowError("export storage descriptor hash/identity mismatch")
    if binding_descriptor["schema_version"] != 2:
        raise GpuWorkflowError("export storage binding descriptor version mismatch")
    _sha(
        binding_descriptor["content_sha256"],
        label="export.run_evidence.storage.binding_descriptor.content_sha256",
    )
    identity_row = binding_descriptor["identity_file"]
    _exact_keys(
        identity_row,
        {"path", "st_dev", "st_ino", "f_fsid"},
        label="export.run_evidence.storage.binding_descriptor.identity_file",
    )
    if identity_row["path"] != plan["paths"]["storage_identity"]:
        raise GpuWorkflowError(
            "export storage identity path differs from reviewed path"
        )
    for name in ("st_dev", "st_ino", "f_fsid"):
        if isinstance(identity_row[name], bool) or not isinstance(
            identity_row[name], int
        ):
            raise GpuWorkflowError("export storage identity evidence is malformed")
    expected_checkpoint_root = PurePosixPath(
        plan["paths"]["launch_evidence_root"]
    ).parent.as_posix()
    for field, expected_path in (
        ("checkpoint_root", expected_checkpoint_root),
        ("qdrant_storage_root", plan["paths"]["qdrant_storage_root"]),
    ):
        row = binding_descriptor[field]
        _exact_keys(
            row,
            {"requested_path", "st_dev", "f_fsid"},
            label=f"export.run_evidence.storage.binding_descriptor.{field}",
        )
        if (
            row["requested_path"] != expected_path
            or any(
                isinstance(row[name], bool) or not isinstance(row[name], int)
                for name in ("st_dev", "f_fsid")
            )
            or row["st_dev"] != identity_row["st_dev"]
            or row["f_fsid"] != identity_row["f_fsid"]
        ):
            raise GpuWorkflowError("export storage volume evidence is inconsistent")
    launch_row = storage_evidence["launch_evidence_root"]
    _exact_keys(
        launch_row,
        {"path", "st_dev", "st_ino", "f_fsid"},
        label="export.run_evidence.storage.launch_evidence_root",
    )
    if (
        launch_row["path"] != plan["paths"]["launch_evidence_root"]
        or any(
            isinstance(launch_row[name], bool) or not isinstance(launch_row[name], int)
            for name in ("st_dev", "st_ino", "f_fsid")
        )
        or launch_row["st_dev"] != identity_row["st_dev"]
        or launch_row["f_fsid"] != identity_row["f_fsid"]
    ):
        raise GpuWorkflowError("export launch evidence volume is inconsistent")
    launch_summary = run_evidence["reviewed_launch"]
    _exact_keys(
        launch_summary,
        {"initialization", "workers", "aggregate_sha256"},
        label="export.run_evidence.reviewed_launch",
    )
    workers = launch_summary["workers"]
    if not isinstance(workers, list) or len(workers) != plan["compute"]["worker_count"]:
        raise GpuWorkflowError("export reviewed worker evidence count mismatch")
    launch_rows = [launch_summary["initialization"], *workers]
    for index, row in enumerate(launch_rows):
        _exact_keys(
            row,
            {"path", "sha256", "launch_type", "worker", "command_sha256"},
            label=f"export.run_evidence.reviewed_launch[{index}]",
        )
        expected_command = (
            plan["commands"]["initialize_collection_and_workers"]
            if index == 0
            else plan["commands"]["embed_workers"][index - 1]
        )
        expected_path = _launch_evidence_path(
            plan,
            initialize_workers=(
                plan["compute"]["worker_count"] if index == 0 else None
            ),
            shard=(
                None if index == 0 else (index - 1, plan["compute"]["worker_count"])
            ),
        ).as_posix()
        expected_worker = {
            "worker_id": (None if index == 0 else index - 1),
            "worker_count": plan["compute"]["worker_count"],
        }
        _sha(row["sha256"], label="export reviewed launch artifact SHA-256")
        if (
            row["path"] != expected_path
            or row["launch_type"] != ("initialize" if index == 0 else "worker")
            or row["worker"] != expected_worker
            or row["command_sha256"]
            != hashlib.sha256(_canonical_bytes(expected_command)).hexdigest()
        ):
            raise GpuWorkflowError("export reviewed launch summary is inconsistent")
    if (
        launch_summary["aggregate_sha256"]
        != hashlib.sha256(_canonical_bytes(launch_rows)).hexdigest()
    ):
        raise GpuWorkflowError("export reviewed launch aggregate hash mismatch")
    snapshot = value.get("qdrant_snapshot")
    remote = value.get("remote_collection")
    _exact_keys(
        snapshot,
        {
            "name",
            "reported_size_bytes",
            "reported_checksum",
            "export_sha256",
            "export_size_bytes",
        },
        label="export.qdrant_snapshot",
    )
    _exact_keys(
        remote,
        {
            "point_count",
            "collection_sha256",
            "collection_configuration_sha256",
            "collection_configuration",
        },
        label="export.remote_collection",
    )
    if (
        not isinstance(snapshot["name"], str)
        or not snapshot["name"]
        or isinstance(snapshot["reported_size_bytes"], bool)
        or not isinstance(snapshot["reported_size_bytes"], int)
        or snapshot["reported_size_bytes"] < 1
        or isinstance(snapshot["export_size_bytes"], bool)
        or not isinstance(snapshot["export_size_bytes"], int)
        or snapshot["export_size_bytes"] < 1
        or snapshot["reported_size_bytes"] != snapshot["export_size_bytes"]
        or snapshot["reported_checksum"] is not None
        and not isinstance(snapshot["reported_checksum"], str)
        or isinstance(remote["point_count"], bool)
        or not isinstance(remote["point_count"], int)
        or remote["point_count"] < 1
    ):
        raise GpuWorkflowError("Qdrant export snapshot/count values are invalid")
    _sha(snapshot["export_sha256"], label="snapshot.export_sha256")
    _sha(remote["collection_sha256"], label="remote.collection_sha256")
    _sha(
        remote["collection_configuration_sha256"],
        label="remote.collection_configuration_sha256",
    )
    configuration = remote["collection_configuration"]
    if (
        not isinstance(configuration, Mapping)
        or store.collection_configuration_sha256(configuration)
        != remote["collection_configuration_sha256"]
        or configuration != plan["collection_configuration"]["value"]
        or remote["collection_configuration_sha256"]
        != plan["collection_configuration"]["sha256"]
    ):
        raise GpuWorkflowError("export collection configuration hash mismatch")
    for field in (
        "embed_binding_sha256",
        "runtime_vector_checksum_artifact_sha256",
        "runtime_vector_probe_sha256",
        "checksum_comparison_sha256",
        "storage_identity_content_sha256",
        "storage_identity_sha256",
        "aliases_sha256",
    ):
        _sha(value[field], label=f"export.{field}")
    return artifact, value, artifact_sha


def _upload_snapshot_http(cfg: Config, collection_name: str, snapshot: Path) -> None:
    parsed = urllib.parse.urlparse(cfg.qdrant_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1", None}:
        raise GpuWorkflowError("local snapshot restore requires a loopback Qdrant URL")
    if parsed.scheme not in {"http", "https"}:
        raise GpuWorkflowError("local Qdrant URL must use HTTP(S)")
    boundary = (
        "----georgian-legal-snapshot-"
        + hashlib.sha256(snapshot.name.encode("utf-8")).hexdigest()[:24]
    )
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="snapshot"; '
        f'filename="{snapshot.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    content_length = len(prefix) + snapshot.stat().st_size + len(suffix)
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(parsed.hostname, parsed.port, timeout=4 * 3600)
    endpoint = (
        f"/collections/{urllib.parse.quote(collection_name, safe='')}/snapshots/upload"
        "?priority=snapshot&wait=true"
    )
    try:
        connection.putrequest("POST", endpoint)
        connection.putheader(
            "Content-Type", f"multipart/form-data; boundary={boundary}"
        )
        connection.putheader("Content-Length", str(content_length))
        if cfg.qdrant_api_key:
            connection.putheader("api-key", cfg.qdrant_api_key)
        connection.endheaders()
        connection.send(prefix)
        with snapshot.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                connection.send(block)
        connection.send(suffix)
        response = connection.getresponse()
        body = response.read(1024 * 1024)
        if not 200 <= response.status < 300:
            raise GpuWorkflowError(
                f"Qdrant snapshot upload failed with HTTP {response.status}: "
                + body.decode("utf-8", "replace")[:1000]
            )
    finally:
        connection.close()


def restore_local_export(
    client: Any,
    cfg: Config,
    *,
    plan_path: str | Path,
    review_path: str | Path,
    export_manifest_path: str | Path,
    snapshot_path: str | Path,
    proof_output: str | Path,
    apply: bool,
    environ: Mapping[str, str] | None = None,
    uploader: Callable[[Config, str, Path], None] | None = None,
) -> Path:
    """Restore only the absent frozen target and prove remote/local digests are identical."""

    environment = os.environ if environ is None else environ
    if not apply or environment.get(LOCAL_RESTORE_APPROVAL_ENV) != "1":
        raise GpuWorkflowError(
            f"local restore requires --apply and {LOCAL_RESTORE_APPROVAL_ENV}=1"
        )
    if cfg.generation_id != GENERATION_ID or cfg.collection_name != PHYSICAL_COLLECTION:
        raise GpuWorkflowError(
            "local restore target is not the frozen physical generation"
        )
    parsed = urllib.parse.urlparse(cfg.qdrant_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1", None}:
        raise GpuWorkflowError("local restore requires loopback Qdrant")
    plan_file, plan, plan_sha = load_workflow_plan(plan_path)
    review_file, review, review_sha = load_workflow_review(
        review_path, plan=plan, plan_sha256=plan_sha
    )
    export_file, export, export_sha = load_export_manifest(
        export_manifest_path,
        plan=plan,
        plan_sha256=plan_sha,
        review=review,
        review_sha256=review_sha,
    )
    if (
        plan_file.as_posix() != plan["paths"]["plan"]
        or review_file.as_posix() != plan["paths"]["review"]
        or export_file.as_posix() != plan["paths"]["export_manifest"]
    ):
        raise GpuWorkflowError(
            "local restore artifact paths differ from reviewed paths"
        )
    supplied_snapshot = Path(snapshot_path).expanduser().absolute()
    _require_nonsymlink_path(supplied_snapshot, allow_missing=False)
    snapshot = supplied_snapshot.resolve(strict=True)
    if not snapshot.is_file():
        raise GpuWorkflowError("snapshot export must be a regular non-symlink file")
    if snapshot.as_posix() != plan["paths"]["snapshot_export"]:
        raise GpuWorkflowError("local restore snapshot path differs from reviewed path")
    expected_snapshot = export["qdrant_snapshot"]
    if (
        snapshot.stat().st_size != expected_snapshot["export_size_bytes"]
        or _sha256_file(snapshot) != expected_snapshot["export_sha256"]
    ):
        raise GpuWorkflowError("transferred Qdrant snapshot hash/size mismatch")
    proof = Path(proof_output).expanduser().absolute()
    if os.path.lexists(proof):
        raise GpuWorkflowError("local restore proof already exists")
    aliases_before = _alias_inventory(client)
    store.refuse_aliased_write_target(client, cfg.collection_name)
    try:
        exists = client.collection_exists(cfg.collection_name)
    except Exception as exc:  # noqa: BLE001
        raise GpuWorkflowError(f"cannot confirm local target absence: {exc}") from exc
    if exists:
        raise GpuWorkflowError(
            "local restore refuses any pre-existing target, including an empty collection"
        )
    # Reconfirm immediately before the sole local Qdrant mutation.
    if client.collection_exists(cfg.collection_name):
        raise GpuWorkflowError("local target appeared before snapshot upload")
    (uploader or _upload_snapshot_http)(cfg, cfg.collection_name, snapshot)
    if not client.collection_exists(cfg.collection_name):
        raise GpuWorkflowError(
            "Qdrant snapshot upload returned without creating the target"
        )
    local_seal = scan_collection_seal(client, cfg.collection_name)
    remote = export["remote_collection"]
    if (
        local_seal.point_count != remote["point_count"]
        or local_seal.collection_sha256 != remote["collection_sha256"]
        or local_seal.collection_configuration_sha256
        != remote["collection_configuration_sha256"]
        or local_seal.collection_configuration != remote["collection_configuration"]
    ):
        raise GpuWorkflowError(
            "restored local collection differs from remote sealed digest"
        )
    aliases_after = _alias_inventory(client)
    if aliases_after != aliases_before:
        raise GpuWorkflowError("Qdrant aliases changed during local snapshot restore")
    value = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "kind": RESTORE_KIND,
        "workflow_id": plan["workflow_id"],
        "workflow_plan_sha256": plan_sha,
        "workflow_review_sha256": review_sha,
        "export_manifest_sha256": export_sha,
        "snapshot_export_sha256": expected_snapshot["export_sha256"],
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "point_count": local_seal.point_count,
        "remote_collection_sha256": remote["collection_sha256"],
        "local_collection_sha256": local_seal.collection_sha256,
        "remote_configuration_sha256": remote["collection_configuration_sha256"],
        "local_configuration_sha256": local_seal.collection_configuration_sha256,
        "aliases_sha256": hashlib.sha256(
            _canonical_bytes(list(aliases_after))
        ).hexdigest(),
        "aliases_unchanged": True,
        "digest_match": True,
    }
    return _write_create_only(proof, value)
