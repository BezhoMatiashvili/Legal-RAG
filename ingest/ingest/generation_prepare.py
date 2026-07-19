"""Fail-closed preparation of an immutable generation from a sealed snapshot.

Preparation is deliberately read-only with respect to Qdrant.  It verifies the sealed
snapshot and release inputs, scans the exact physical collection, joins every point to
the clean snapshot ledger, and atomically emits the only input accepted by the generation
publisher.  A collection-derived source-state placeholder is never accepted or emitted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import zip_longest
from numbers import Real
from pathlib import Path
from typing import Any

from . import chunk_inventory, qdrant_store as store
from .config import (
    REPO_ROOT,
    RETRIEVAL_FINGERPRINT_REVISION,
    Config,
    retrieval_fingerprint_sha256,
)
from .chunking import STRUCTURAL_CHUNKER_REVISION, build_embed_text
from .dedup import content_hash
from .embed_job import (
    BINDING_SCHEMA_VERSION,
    EmbedBinding,
    SealedSnapshot,
    binding_sha256,
    load_checksum_reference,
    snapshot_doc_to_canonical,
    validate_snapshot_build_config,
)
from .generation import (
    CANONICAL_PAYLOAD_REVISION,
    CHECKSUM_ALGORITHM,
    CHECKSUM_FILENAME,
    COLLECTION_DIGEST_FILENAME,
    DOCUMENTS_FILENAME,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    MAX_METADATA_LINE_BYTES,
    SAMPLE_CHECKS_FILENAME,
    ChecksumInventory,
    CollectionDigest,
    DocumentRecord,
    GenerationManifest,
    SampleCheck,
    parse_rfc3339_utc,
)
from .integrity import (
    COLLECTION_DENSE_ENCODING,
    COLLECTION_PAYLOAD_PROJECTION,
    COLLECTION_SPARSE_ENCODING,
    point_content_sha256,
    whole_collection_sha256,
)
from .generation_snapshot import (
    PRIVATE_DIRECTORY_MODE,
    PRIVATE_FILE_MODE,
    _canonical_json_bytes,
    _fsync_directory,
    _private_binary_writer,
    _rename_noreplace,
    _write_bytes,
    source_state_sha256,
)
from .generation_rematerialize import (
    RematerializationError,
    validate_rematerialization_report,
)
from .pipeline import _document_state_hash
from .promotion import SERVING_ALIAS, physical_collection_name
from .release_inputs import GENERATION_ID as FROZEN_GENERATION_ID
from .snapshot import SNAPSHOT_PIPELINE_VERSION, SOURCES_PRESENT, verify_sealed_snapshot
from .sources import COURT_CANONICAL_FIELDS

PREPARATION_PROVENANCE_FILENAME = "preparation_provenance.json"
SOURCE_STATE_FILENAME = "source_state.json"
PREPARATION_KIND = "immutable-generation-preparation"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_PREPARED_FILES = frozenset(
    {
        MANIFEST_FILENAME,
        DOCUMENTS_FILENAME,
        SAMPLE_CHECKS_FILENAME,
        SOURCE_STATE_FILENAME,
        PREPARATION_PROVENANCE_FILENAME,
        COLLECTION_DIGEST_FILENAME,
    }
)

# Snapshot records are a lossless canonical input to embedding.  Requiring the complete
# current record shape here prevents the generation ledger from blessing values supplied
# by ``dict.get`` defaults in an older converter.
_SNAPSHOT_RECORD_FIELDS = frozenset(
    {
        "snapshot_version",
        "snapshot_id",
        "doc_id",
        "source",
        "document_id",
        "content_hash",
        "title",
        "date",
        "date_raw",
        "language",
        "document_type",
        "court",
        "source_url",
        "source_binary_url",
        "document_number",
        "registration_code",
        "parties",
        "status",
        "status_raw",
        "in_force_date",
        "expiry_date",
        "is_consolidated",
        "consolidated_count",
        "content_kind",
        "content_complete",
        "extraction_status",
        "article_summary",
        "source_fingerprint",
        "normalizer_revision",
        "version_id",
        "version_id_kind",
        "supersedes",
        "effective_from",
        "effective_to",
        "repeal_date",
        "consolidation_status",
        "version_lineage_status",
        "version_lineage_complete",
        "consolidated_dates",
        "official_url",
        "official_binary_url",
        "official_html_url",
        "official_pdf_url",
        "source_authority",
        "freshness_sla_met",
        "admissible",
        "page_boundaries",
        "page_coordinate_reason",
        "promoted",
        "structure",
        "body_char_len",
        "source_run",
        "body_markdown",
    }
)


class GenerationPreparationError(RuntimeError):
    """A candidate cannot be proven to match its immutable inputs."""

    def __init__(self, message: str, *, staging_path: Path | None = None) -> None:
        super().__init__(message)
        self.staging_path = staging_path


@dataclass(frozen=True, slots=True)
class ValidatedPreparationInputs:
    generation_id: str
    snapshot_root: Path
    snapshot_manifest: Mapping[str, Any]
    physical_collection: str
    dependency_lock_sha256: str
    runtime_identity: Mapping[str, Any]
    runtime_identity_sha256: str
    image_digest: str
    code_identity: Mapping[str, str | None]
    embed_binding: Mapping[str, Any]
    embed_binding_sha256: str
    vector_checksum_artifact_sha256: str
    vector_probe_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedGeneration:
    root: Path
    manifest: GenerationManifest
    checksums: ChecksumInventory
    checksums_sha256: str
    source_state: Mapping[str, Any]
    provenance: Mapping[str, Any]
    collection_digest: CollectionDigest

    def iter_documents(self) -> Iterator[DocumentRecord]:
        for line_number, value in _iter_bound_jsonl(
            self.root / DOCUMENTS_FILENAME,
            self.checksums.files[DOCUMENTS_FILENAME],
        ):
            try:
                yield DocumentRecord.from_dict(
                    value,
                    expected_generation_id=self.manifest.generation_id,
                )
            except ValueError as exc:
                raise GenerationPreparationError(
                    f"{DOCUMENTS_FILENAME}:{line_number}: {exc}"
                ) from exc

    def iter_samples(self) -> Iterator[SampleCheck]:
        for line_number, value in _iter_bound_jsonl(
            self.root / SAMPLE_CHECKS_FILENAME,
            self.checksums.files[SAMPLE_CHECKS_FILENAME],
        ):
            try:
                yield SampleCheck.from_dict(
                    value,
                    expected_generation_id=self.manifest.generation_id,
                )
            except ValueError as exc:
                raise GenerationPreparationError(
                    f"{SAMPLE_CHECKS_FILENAME}:{line_number}: {exc}"
                ) from exc


def _strict_json_line(raw: bytes, *, origin: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GenerationPreparationError(
                    f"{origin}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GenerationPreparationError(
            f"{origin}: non-finite JSON number {value!r}"
        )

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except GenerationPreparationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationPreparationError(f"{origin}: invalid JSON") from exc


def _open_regular_nofollow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
    except OSError as exc:
        raise GenerationPreparationError(f"cannot securely open {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise GenerationPreparationError(f"prepared artifact is not regular: {path}")
    return descriptor


def _read_bound_file(path: Path, expected_sha256: str | None) -> tuple[bytes, str]:
    descriptor = _open_regular_nofollow(path)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise GenerationPreparationError(
            f"prepared checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {observed}"
        )
    return b"".join(chunks), observed


def _iter_bound_jsonl(path: Path, expected_sha256: str) -> Iterator[tuple[int, Any]]:
    """Parse and hash one opened inode; replacement/mutation cannot escape the digest."""
    descriptor = _open_regular_nofollow(path)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            initial_digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                initial_digest.update(block)
            initial_observed = initial_digest.hexdigest()
            if initial_observed != expected_sha256:
                raise GenerationPreparationError(
                    f"prepared checksum mismatch for {path.name}: "
                    f"expected {expected_sha256}, got {initial_observed}"
                )
            handle.seek(0)
            streamed_digest = hashlib.sha256()
            for line_number, raw in enumerate(handle, start=1):
                streamed_digest.update(raw)
                if len(raw) > MAX_METADATA_LINE_BYTES:
                    raise GenerationPreparationError(
                        f"{path}:{line_number}: metadata line is too large"
                    )
                if not raw.strip():
                    raise GenerationPreparationError(
                        f"{path}:{line_number}: blank lines are forbidden"
                    )
                yield line_number, _strict_json_line(
                    raw, origin=f"{path}:{line_number}"
                )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = streamed_digest.hexdigest()
    if observed != expected_sha256:
        raise GenerationPreparationError(
            f"prepared checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {observed}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GenerationPreparationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_symlink_components(path: Path) -> None:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise GenerationPreparationError(
                f"refusing symlink path component: {cursor}"
            )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise GenerationPreparationError("created_at must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _collect_code_identity(repo_root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Bind tracked and untracked local code without modifying the repository."""

    def git(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GenerationPreparationError(
                f"cannot collect Git identity with {' '.join(arguments)}"
            ) from exc

    git_sha = git("rev-parse", "HEAD").decode("ascii").strip()
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise GenerationPreparationError("Git HEAD is not a canonical object ID")
    material = bytearray(git("diff", "--binary", "--no-ext-diff", "HEAD", "--", "."))
    untracked = [
        item
        for item in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if item
    ]
    for raw_name in sorted(untracked):
        try:
            name = raw_name.decode("utf-8")
        except UnicodeError as exc:
            raise GenerationPreparationError("untracked path is not UTF-8") from exc
        path = repo_root / name
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise GenerationPreparationError(
                f"untracked code input is not a regular file: {name}"
            )
        material.extend(b"\0untracked\0")
        material.extend(raw_name)
        material.extend(b"\0")
        material.extend(bytes.fromhex(_sha256_file(path)))
    return {
        "git_sha": git_sha,
        "dirty_patch_sha256": hashlib.sha256(material).hexdigest() if material else None,
    }


def _validate_code_identity(value: Mapping[str, Any]) -> dict[str, str | None]:
    if set(value) != {"git_sha", "dirty_patch_sha256"}:
        raise GenerationPreparationError("code identity has invalid keys")
    git_sha = value["git_sha"]
    patch = value["dirty_patch_sha256"]
    if not isinstance(git_sha, str) or not _GIT_SHA_RE.fullmatch(git_sha):
        raise GenerationPreparationError("code.git_sha is invalid")
    if patch is not None and (
        not isinstance(patch, str) or not _SHA256_RE.fullmatch(patch)
    ):
        raise GenerationPreparationError("code.dirty_patch_sha256 is invalid")
    return {"git_sha": git_sha, "dirty_patch_sha256": patch}


def _validate_supply_chain(
    dependency_lock: Path, runtime_identity_path: Path
) -> tuple[str, dict[str, Any], str]:
    # Import the established pure/offline release validators rather than maintaining a
    # second looser interpretation of a production lock or runtime identity.
    try:
        from scripts import supply_chain

        for path, field in (
            (dependency_lock, "dependency lock"),
            (runtime_identity_path, "runtime identity"),
        ):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise GenerationPreparationError(
                    f"{field} must be a regular non-symlink file"
                )
        lock_bytes = dependency_lock.read_bytes()
        runtime_bytes = runtime_identity_path.read_bytes()
        lock = supply_chain.parse_requirements_lock(dependency_lock)
        identity = _strict_json_line(runtime_bytes, origin="runtime identity")
        if not isinstance(identity, dict):
            raise GenerationPreparationError("runtime identity must be an object")
        supply_chain.validate_runtime_identity(
            identity,
            lock,
            base_image=str(identity.get("base_image", "")),
            qdrant_version=str(identity.get("qdrant_version", "")),
            qdrant_archive_url_value=str(identity.get("qdrant_archive_url", "")),
            qdrant_archive_sha256=str(identity.get("qdrant_archive_sha256", "")),
            qdrant_checksum_source_url=str(
                identity.get("qdrant_checksum_source_url", "")
            ),
        )
    except GenerationPreparationError:
        raise
    except (OSError, ValueError) as exc:
        raise GenerationPreparationError(f"invalid release identity: {exc}") from exc
    raw_digest = hashlib.sha256(lock_bytes).hexdigest()
    if raw_digest != lock.sha256:
        raise GenerationPreparationError(
            "requirements-lock validator digest does not match exact file bytes"
        )
    if identity.get("requirements_lock_sha256") != raw_digest:
        raise GenerationPreparationError(
            "runtime identity does not match exact dependency-lock bytes"
        )
    runtime_digest = hashlib.sha256(runtime_bytes).hexdigest()
    if _sha256_file(runtime_identity_path) != runtime_digest:
        raise GenerationPreparationError(
            "runtime identity changed while it was being validated"
        )
    if _sha256_file(dependency_lock) != raw_digest:
        raise GenerationPreparationError(
            "dependency lock changed while it was being validated"
        )
    return raw_digest, identity, runtime_digest


def _structural_inventory_projection(
    snapshot_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw_chunk_inventory = snapshot_manifest.get("structural_chunk_inventory")
    if not isinstance(raw_chunk_inventory, Mapping) or raw_chunk_inventory.get(
        "status"
    ) != "available":
        raise GenerationPreparationError(
            "sealed snapshot lacks an available structural chunk inventory"
        )
    return {
        field: raw_chunk_inventory[field]
        for field in (
            "sha256",
            "size_bytes",
            "identity_sha256",
            "record_count",
            "document_count",
            "chunk_count",
        )
    }


def _validate_bound_structural_inventory(
    snapshot_manifest: Mapping[str, Any], binding_value: Mapping[str, Any]
) -> None:
    expected_snapshot = {
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "snapshot_sha256": snapshot_manifest["snapshot_sha256"],
        "corpus_sha256": snapshot_manifest["corpus_sha256"],
    }
    chunk_inventory_projection = _structural_inventory_projection(snapshot_manifest)
    bound_snapshot = binding_value.get("snapshot")
    if not isinstance(bound_snapshot, Mapping) or any(
        bound_snapshot.get(field) != value for field, value in expected_snapshot.items()
    ):
        raise GenerationPreparationError("embed binding snapshot identity mismatch")
    if bound_snapshot.get("structural_chunk_inventory") != chunk_inventory_projection:
        raise GenerationPreparationError(
            "embed binding structural chunk inventory mismatch"
        )


def _validate_embed_evidence(
    *,
    embed_binding_path: Path,
    vector_checksum_path: Path,
    cfg: Config,
    snapshot_manifest: Mapping[str, Any],
    physical_collection: str,
) -> tuple[dict[str, Any], str, str, str]:
    binding_bytes, binding_file_sha = _read_bound_file(embed_binding_path, None)
    binding_value = _strict_json_line(binding_bytes, origin="embed binding")
    if not isinstance(binding_value, dict):
        raise GenerationPreparationError("embed binding must be an object")
    try:
        checksum = load_checksum_reference(vector_checksum_path)
    except Exception as exc:
        raise GenerationPreparationError(f"invalid vector checksum artifact: {exc}") from exc
    _validate_bound_structural_inventory(snapshot_manifest, binding_value)
    if (
        binding_value.get("generation_id") != cfg.generation_id
        or binding_value.get("physical_collection") != physical_collection
        or not isinstance(binding_value.get("variant_id"), str)
        or not _SHA256_RE.fullmatch(str(binding_value.get("variant_id", "")))
    ):
        raise GenerationPreparationError("embed binding generation identity mismatch")
    if cfg.generation_id == FROZEN_GENERATION_ID and (
        binding_value.get("schema_version") != BINDING_SCHEMA_VERSION
        or not isinstance(binding_value.get("storage_identity_sha256"), str)
        or not _SHA256_RE.fullmatch(binding_value["storage_identity_sha256"])
        or not isinstance(binding_value.get("reviewed_plan_sha256"), str)
        or not _SHA256_RE.fullmatch(binding_value["reviewed_plan_sha256"])
        or binding_value.get("embedding_runtime")
        != {
            "device": cfg.embed_device,
            "use_fp16": cfg.embed_use_fp16,
            "batch_size": cfg.embed_batch_size,
        }
    ):
        raise GenerationPreparationError(
            "frozen embed binding lacks exact reviewed plan, mounted-volume, or "
            "embedding-runtime identity"
        )
    checksum_binding = binding_value.get("vector_checksum")
    if not isinstance(checksum_binding, Mapping) or (
        checksum_binding.get("artifact_sha256") != checksum.file_sha256
        or checksum_binding.get("probe_sha256") != checksum.probe_sha256
    ):
        raise GenerationPreparationError("embed binding vector checksum mismatch")
    collection_binding = binding_value.get("collection_configuration")
    if (
        not isinstance(collection_binding, Mapping)
        or not isinstance(collection_binding.get("value"), Mapping)
        or not isinstance(collection_binding.get("sha256"), str)
        or not _SHA256_RE.fullmatch(collection_binding["sha256"])
        or store.collection_configuration_sha256(collection_binding["value"])
        != collection_binding["sha256"]
    ):
        raise GenerationPreparationError("embed binding collection configuration is invalid")
    if cfg.generation_id == FROZEN_GENERATION_ID and collection_binding != {
        "value": store.expected_embed_collection_configuration(dense_dim=cfg.dense_dim),
        "sha256": store.collection_configuration_sha256(
            store.expected_embed_collection_configuration(dense_dim=cfg.dense_dim)
        ),
    }:
        raise GenerationPreparationError(
            "frozen embed binding collection configuration is not the reviewed profile"
        )
    calculated_binding_sha = binding_sha256(
        EmbedBinding(path=embed_binding_path, value=binding_value)
    )
    if calculated_binding_sha != binding_file_sha:
        # Binding files use the same canonical newline encoding as binding_sha256.
        raise GenerationPreparationError("embed binding file is not canonical or changed")
    return (
        dict(binding_value),
        binding_file_sha,
        checksum.file_sha256,
        checksum.probe_sha256,
    )


def _validate_config(cfg: Config, generation_id: str, physical_collection: str) -> None:
    expected_collection = physical_collection_name(generation_id)
    if physical_collection != expected_collection:
        raise GenerationPreparationError(
            f"physical collection must be exactly {expected_collection!r}"
        )
    if cfg.generation_id != generation_id:
        raise GenerationPreparationError(
            "configuration generation_id does not match requested generation"
        )
    if cfg.collection_name != physical_collection:
        raise GenerationPreparationError(
            "configuration collection_name does not match exact physical collection"
        )
    try:
        identity = store.generation_point_identity(cfg)
    except ValueError as exc:
        raise GenerationPreparationError(f"invalid model identity: {exc}") from exc
    if identity is None:
        raise GenerationPreparationError("generation model identity is not configured")
    if (
        getattr(identity, "retrieval_fingerprint_revision", None)
        != RETRIEVAL_FINGERPRINT_REVISION
    ):
        raise GenerationPreparationError(
            "point identity lacks retrieval fingerprint revision 2"
        )


def _validate_snapshot_config(
    snapshot_root: Path,
    snapshot_manifest: Mapping[str, Any],
    cfg: Config,
) -> None:
    build = snapshot_manifest.get("build")
    raw_sources = build.get("sources") if isinstance(build, Mapping) else None
    sealed = SealedSnapshot(
        root=snapshot_root.expanduser().absolute(),
        docs=snapshot_root.expanduser().absolute() / "docs",
        snapshot_id=str(snapshot_manifest.get("snapshot_id", "")),
        snapshot_sha256=str(snapshot_manifest.get("snapshot_sha256", "")),
        corpus_sha256=str(snapshot_manifest.get("corpus_sha256", "")),
        sources=tuple(raw_sources) if isinstance(raw_sources, list) else (),
        manifest=snapshot_manifest,
    )
    try:
        validate_snapshot_build_config(sealed, cfg)
    except Exception as exc:
        raise GenerationPreparationError(
            f"sealed snapshot build configuration mismatch: {exc}"
        ) from exc


def validate_preparation_inputs(
    cfg: Config,
    *,
    generation_id: str,
    snapshot_root: Path,
    physical_collection: str,
    dependency_lock: Path,
    runtime_identity: Path,
    image_digest: str,
    embed_binding: Path,
    vector_checksum: Path,
    output_dir: Path,
    code_identity: Mapping[str, Any] | None = None,
) -> ValidatedPreparationInputs:
    """Validate immutable local inputs before any Qdrant client access."""
    _reject_symlink_components(output_dir)
    if os.path.lexists(output_dir):
        raise GenerationPreparationError(
            f"prepared generation destination already exists: {output_dir}"
        )
    _validate_config(cfg, generation_id, physical_collection)
    if not isinstance(image_digest, str) or not _IMAGE_DIGEST_RE.fullmatch(image_digest):
        raise GenerationPreparationError(
            "final image digest must be sha256:<64 lowercase hex>"
        )
    try:
        snapshot_manifest = verify_sealed_snapshot(
            snapshot_root,
            allow_preflight=False,
            require_all_sources=True,
        )
    except Exception as exc:  # snapshot exposes a bounded domain exception
        raise GenerationPreparationError(f"sealed snapshot validation failed: {exc}") from exc
    _validate_snapshot_config(snapshot_root, snapshot_manifest, cfg)
    lock_sha256, runtime_value, runtime_sha256 = _validate_supply_chain(
        dependency_lock, runtime_identity
    )
    validated_code = _validate_code_identity(
        code_identity if code_identity is not None else _collect_code_identity()
    )
    (
        embed_binding_value,
        embed_binding_sha,
        vector_checksum_sha,
        vector_probe_sha,
    ) = _validate_embed_evidence(
        embed_binding_path=embed_binding,
        vector_checksum_path=vector_checksum,
        cfg=cfg,
        snapshot_manifest=snapshot_manifest,
        physical_collection=physical_collection,
    )
    return ValidatedPreparationInputs(
        generation_id=generation_id,
        snapshot_root=snapshot_root.expanduser().absolute(),
        snapshot_manifest=snapshot_manifest,
        physical_collection=physical_collection,
        dependency_lock_sha256=lock_sha256,
        runtime_identity=runtime_value,
        runtime_identity_sha256=runtime_sha256,
        image_digest=image_digest,
        code_identity=validated_code,
        embed_binding=embed_binding_value,
        embed_binding_sha256=embed_binding_sha,
        vector_checksum_artifact_sha256=vector_checksum_sha,
        vector_probe_sha256=vector_probe_sha,
    )


def _create_ledger(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    os.chmod(path, PRIVATE_FILE_MODE)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE expected (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            source_run TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            document_state_hash TEXT NOT NULL,
            content_kind TEXT NOT NULL,
            extraction_status TEXT NOT NULL,
            article_summary TEXT,
            source_binary_url TEXT,
            record_json TEXT NOT NULL,
            PRIMARY KEY (source, document_id, version_id)
        );
        CREATE TABLE observed (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            point_id TEXT NOT NULL UNIQUE,
            text_sha256 TEXT NOT NULL,
            declared_chunks INTEGER NOT NULL,
            point_sha256 TEXT NOT NULL,
            PRIMARY KEY (source, document_id, version_id, chunk_index)
        );
        CREATE TABLE expected_chunks (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            document_chunk_count INTEGER NOT NULL,
            canonical_passage_sha256 TEXT NOT NULL,
            canonical_passage_char_length INTEGER NOT NULL,
            canonical_passage_utf8_length INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            structure_json TEXT NOT NULL,
            page_json TEXT NOT NULL,
            embed_input_json TEXT NOT NULL,
            PRIMARY KEY (source, document_id, version_id, chunk_index)
        );
        """
    )
    return connection


def _require_snapshot_record(
    value: Any,
    *,
    source: str,
    snapshot_id: str,
    run_inventory: set[tuple[str, str]],
    cfg: Config,
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, dict):
        raise GenerationPreparationError(f"{source} snapshot record must be an object")
    observed_fields = set(value)
    complete_fields = _SNAPSHOT_RECORD_FIELDS | set(COURT_CANONICAL_FIELDS)
    if observed_fields not in (_SNAPSHOT_RECORD_FIELDS, complete_fields):
        expected = (
            complete_fields
            if observed_fields.intersection(COURT_CANONICAL_FIELDS)
            else _SNAPSHOT_RECORD_FIELDS
        )
        missing = sorted(expected - observed_fields)
        unknown = sorted(observed_fields - complete_fields)
        raise GenerationPreparationError(
            f"{source} snapshot record shape is invalid: "
            f"missing={missing}, unknown={unknown}"
        )
    if value["snapshot_version"] != SNAPSHOT_PIPELINE_VERSION:
        raise GenerationPreparationError("snapshot record pipeline version mismatch")
    if value["source"] != source:
        raise GenerationPreparationError("snapshot record source/file mismatch")
    if value["snapshot_id"] != snapshot_id:
        raise GenerationPreparationError("snapshot record snapshot_id mismatch")
    for field in ("document_id", "version_id", "source_run", "body_markdown"):
        if not isinstance(value[field], str) or not value[field]:
            raise GenerationPreparationError(f"snapshot record {field} is invalid")
    if (source, value["source_run"]) not in run_inventory:
        raise GenerationPreparationError(
            "snapshot record references a run outside the attested inventory"
        )
    if value["doc_id"] != f"{source}:{value['document_id']}:{value['version_id']}":
        raise GenerationPreparationError("snapshot record doc_id is not version-scoped")
    if value["content_complete"] is not True or value["extraction_status"] != "full_text":
        raise GenerationPreparationError(
            "incomplete snapshot content cannot enter generation preparation"
        )
    if value["content_kind"] == "article_summary":
        raise GenerationPreparationError("article summaries cannot enter the clean ledger")
    if value["source_authority"] not in {"official", "primary_official"}:
        raise GenerationPreparationError("snapshot source authority is missing or invalid")
    if value["admissible"] is not True:
        raise GenerationPreparationError("snapshot record is not admissible canonical evidence")
    for field in (
        "source_fingerprint",
        "content_hash",
    ):
        if not isinstance(value[field], str) or not _SHA256_RE.fullmatch(value[field]):
            raise GenerationPreparationError(f"snapshot record {field} is invalid")
    if content_hash(value["body_markdown"]) != value["content_hash"]:
        raise GenerationPreparationError("snapshot body does not match content_hash")
    if not isinstance(value["normalizer_revision"], str) or not value[
        "normalizer_revision"
    ]:
        raise GenerationPreparationError("snapshot normalizer revision is missing")
    if not isinstance(value["version_id_kind"], str) or not value["version_id_kind"]:
        raise GenerationPreparationError("snapshot version identity kind is missing")
    if not isinstance(value["version_lineage_status"], str) or not value[
        "version_lineage_status"
    ]:
        raise GenerationPreparationError("snapshot version lineage status is missing")
    if not isinstance(value["version_lineage_complete"], bool):
        raise GenerationPreparationError("snapshot version lineage completeness is invalid")
    if value["body_char_len"] != len(value["body_markdown"]):
        raise GenerationPreparationError("snapshot body_char_len mismatch")
    try:
        canonical = snapshot_doc_to_canonical(value, strict=True)
        state_hash = _document_state_hash(cfg, doc=canonical)
    except Exception as exc:
        raise GenerationPreparationError(
            f"strict snapshot conversion failed for {source}:{value['document_id']}: {exc}"
        ) from exc
    return value, state_hash


def _load_snapshot_ledger(
    connection: sqlite3.Connection,
    inputs: ValidatedPreparationInputs,
    cfg: Config,
) -> int:
    runs = inputs.snapshot_manifest["runs"]
    run_inventory = {(str(run["source"]), str(run["run_id"])) for run in runs}
    raw_files = inputs.snapshot_manifest.get("files")
    if not isinstance(raw_files, list):
        raise GenerationPreparationError("sealed snapshot lacks an exact file inventory")
    file_inventory = {
        entry.get("path"): entry
        for entry in raw_files
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    count = 0
    for source in SOURCES_PRESENT:
        path = inputs.snapshot_root / "docs" / f"{source}.jsonl"
        relative = f"docs/{source}.jsonl"
        inventory_entry = file_inventory.get(relative)
        if not isinstance(inventory_entry, dict) or not isinstance(
            inventory_entry.get("sha256"), str
        ):
            raise GenerationPreparationError(
                f"sealed snapshot inventory lacks {relative}"
            )
        descriptor = _open_regular_nofollow(path)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                digest.update(raw)
                if not raw.strip():
                    raise GenerationPreparationError(
                        f"{path}:{line_number}: blank lines are forbidden"
                    )
                value = _strict_json_line(raw, origin=f"{path}:{line_number}")
                record, state_hash = _require_snapshot_record(
                    value,
                    source=source,
                    snapshot_id=str(inputs.snapshot_manifest["snapshot_id"]),
                    run_inventory=run_inventory,
                    cfg=cfg,
                )
                try:
                    connection.execute(
                        "INSERT INTO expected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            source,
                            record["document_id"],
                            record["version_id"],
                            record["source_run"],
                            record["source_fingerprint"],
                            record["content_hash"],
                            state_hash,
                            record["content_kind"],
                            record["extraction_status"],
                            record["article_summary"],
                            record["source_binary_url"],
                            json.dumps(
                                record,
                                sort_keys=True,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise GenerationPreparationError(
                        "duplicate clean snapshot document version: "
                        f"{source}:{record['document_id']}:{record['version_id']}"
                    ) from exc
                count += 1
                if count % 1000 == 0:
                    connection.commit()
        if digest.hexdigest() != inventory_entry["sha256"]:
            raise GenerationPreparationError(
                f"snapshot ledger changed or does not match inventory: {relative}"
            )
    connection.commit()
    if count != inputs.snapshot_manifest["totals"]["clean"]:
        raise GenerationPreparationError(
            "snapshot clean total does not match the exact clean ledger"
        )
    return count


def _load_structural_chunk_ledger(
    connection: sqlite3.Connection,
    inputs: ValidatedPreparationInputs,
) -> tuple[int, int]:
    manifest_entry = inputs.snapshot_manifest.get("structural_chunk_inventory")
    if not isinstance(manifest_entry, Mapping):
        raise GenerationPreparationError("sealed snapshot lacks structural chunk inventory")
    identity = manifest_entry.get("identity")
    if not isinstance(identity, Mapping):
        raise GenerationPreparationError("structural chunk inventory identity is invalid")
    relative = manifest_entry.get("path")
    if relative != chunk_inventory.CHUNK_INVENTORY_FILENAME:
        raise GenerationPreparationError("structural chunk inventory path is invalid")
    raw_files = inputs.snapshot_manifest.get("files")
    inventory_file = next(
        (
            entry
            for entry in raw_files
            if isinstance(entry, Mapping) and entry.get("path") == relative
        ),
        None,
    ) if isinstance(raw_files, list) else None
    if (
        not isinstance(inventory_file, Mapping)
        or inventory_file.get("sha256") != manifest_entry.get("sha256")
        or inventory_file.get("size_bytes") != manifest_entry.get("size_bytes")
    ):
        raise GenerationPreparationError(
            "snapshot file inventory does not bind structural chunk inventory"
        )
    documents = chunks = 0
    try:
        rows = chunk_inventory.iter_validated_inventory(
            inputs.snapshot_root / relative,
            manifest_entry=manifest_entry,
            expected_identity=identity,
        )
        for document in rows:
            source = document["source"]
            document_id = document["document_id"]
            version_id = document["version_id"]
            expected = connection.execute(
                "SELECT content_hash, record_json FROM expected "
                "WHERE source=? AND document_id=? AND version_id=?",
                (source, document_id, version_id),
            ).fetchone()
            if expected is None:
                raise GenerationPreparationError(
                    "structural inventory document is outside the clean snapshot ledger: "
                    f"{source}:{document_id}:{version_id}"
                )
            content_sha, record_json = expected
            snapshot_record = json.loads(record_json)
            canonical = snapshot_doc_to_canonical(snapshot_record, strict=True)
            page_boundaries = [boundary.to_dict() for boundary in canonical.page_boundaries]
            page_mapping_sha = (
                hashlib.sha256(
                    json.dumps(
                        page_boundaries,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if page_boundaries
                else None
            )
            body = snapshot_record["body_markdown"]
            if {
                "canonical_content_sha256": content_sha,
                "canonical_body_char_length": len(body),
                "canonical_body_utf8_length": len(body.encode("utf-8")),
                "page_boundaries": page_boundaries,
                "page_boundary_mapping_sha256": page_mapping_sha,
                "page_coordinate_reason": canonical.page_coordinate_reason,
            } != {
                field: document[field]
                for field in (
                    "canonical_content_sha256",
                    "canonical_body_char_length",
                    "canonical_body_utf8_length",
                    "page_boundaries",
                    "page_boundary_mapping_sha256",
                    "page_coordinate_reason",
                )
            }:
                raise GenerationPreparationError(
                    "structural inventory document projection differs from sealed snapshot: "
                    f"{source}:{document_id}:{version_id}"
                )
            for chunk in document["chunks"]:
                try:
                    connection.execute(
                        "INSERT INTO expected_chunks VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            source,
                            document_id,
                            version_id,
                            chunk["chunk_index"],
                            chunk["document_chunk_count"],
                            chunk["canonical_passage_sha256"],
                            chunk["canonical_passage_char_length"],
                            chunk["canonical_passage_utf8_length"],
                            chunk["token_count"],
                            chunk["char_start"],
                            chunk["char_end"],
                            json.dumps(
                                chunk["structure"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                chunk["page"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                chunk["embed_input"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise GenerationPreparationError(
                        "duplicate structural inventory chunk identity"
                    ) from exc
                chunks += 1
            documents += 1
    except chunk_inventory.ChunkInventoryError as exc:
        raise GenerationPreparationError(
            f"invalid sealed structural chunk inventory: {exc}"
        ) from exc
    connection.commit()
    missing = connection.execute(
        "SELECT COUNT(*) FROM expected AS e LEFT JOIN expected_chunks AS c "
        "ON c.source=e.source AND c.document_id=e.document_id "
        "AND c.version_id=e.version_id WHERE c.source IS NULL"
    ).fetchone()[0]
    if missing:
        raise GenerationPreparationError(
            "clean snapshot documents are missing from structural chunk inventory"
        )
    if (
        documents != manifest_entry["document_count"]
        or chunks != manifest_entry["chunk_count"]
    ):
        raise GenerationPreparationError("structural chunk ledger aggregate mismatch")
    return documents, chunks


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalise_enum(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).split(".")[-1].lower()


def _inspect_collection(
    client: Any, collection: str, cfg: Config
) -> tuple[Any, int, Mapping[str, Any], str]:
    try:
        info = client.get_collection(collection_name=collection)
    except TypeError:
        info = client.get_collection(collection)
    except Exception as exc:
        raise GenerationPreparationError(
            f"cannot inspect physical collection {collection!r}: {exc}"
        ) from exc
    status = _normalise_enum(_attr(info, "status", ""))
    if status != "green":
        raise GenerationPreparationError(
            f"physical collection is not green: status={status!r}"
        )
    config = _attr(info, "config")
    params = _attr(config, "params")
    vectors = _attr(params, "vectors")
    sparse = _attr(params, "sparse_vectors")
    dense = vectors.get("dense") if isinstance(vectors, Mapping) else None
    sparse_value = sparse.get("sparse") if isinstance(sparse, Mapping) else None
    if dense is None or sparse_value is None:
        raise GenerationPreparationError(
            "physical collection lacks exact dense/sparse named vector configuration"
        )
    if _attr(dense, "size") != cfg.dense_dim:
        raise GenerationPreparationError("physical collection dense dimension mismatch")
    if _normalise_enum(_attr(dense, "distance")) != "cosine":
        raise GenerationPreparationError("physical collection distance mismatch")
    points_count = _attr(info, "points_count")
    if (
        isinstance(points_count, bool)
        or not isinstance(points_count, int)
        or points_count < 0
    ):
        raise GenerationPreparationError("physical collection points_count is invalid")
    try:
        configuration = store.collection_configuration(info)
        configuration_sha = store.collection_configuration_sha256(configuration)
    except Exception as exc:
        raise GenerationPreparationError(
            f"physical collection configuration is invalid: {exc}"
        ) from exc
    return info, points_count, configuration, configuration_sha


def _point_parts(point: Any) -> tuple[str, Mapping[str, Any], Any]:
    raw_id = _attr(point, "id")
    payload = _attr(point, "payload")
    point_id = str(raw_id)
    if not isinstance(payload, Mapping):
        raise GenerationPreparationError(f"point {point_id!r} has no payload object")
    return point_id, payload, _attr(point, "vector")


def _validate_point_vectors(point_id: str, vectors: Any, cfg: Config) -> None:
    if not isinstance(vectors, Mapping) or set(vectors) != {"dense", "sparse"}:
        raise GenerationPreparationError(
            f"point {point_id} lacks exact dense/sparse named vectors"
        )
    dense = vectors["dense"]
    if (
        not isinstance(dense, Sequence)
        or isinstance(dense, (str, bytes, bytearray))
        or len(dense) != cfg.dense_dim
    ):
        raise GenerationPreparationError(f"point {point_id} dense vector is invalid")
    for value in dense:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(
            float(value)
        ):
            raise GenerationPreparationError(
                f"point {point_id} dense vector contains a non-finite/non-numeric value"
            )
    sparse = vectors["sparse"]
    indices = _attr(sparse, "indices")
    values = _attr(sparse, "values")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes, bytearray))
        or not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(indices) != len(values)
    ):
        raise GenerationPreparationError(f"point {point_id} sparse vector is invalid")
    previous_index: int | None = None
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise GenerationPreparationError(
                f"point {point_id} sparse vector index is invalid"
            )
        if previous_index is not None and index <= previous_index:
            raise GenerationPreparationError(
                f"point {point_id} sparse vector indices are not sorted/unique"
            )
        previous_index = index
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise GenerationPreparationError(
                f"point {point_id} sparse vector value is invalid"
            )


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GenerationPreparationError(f"point payload {field} must be integer >= {minimum}")
    return value


def _expect_payload(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field, value in expected.items():
        if field not in payload:
            raise GenerationPreparationError(f"point payload is missing {field}")
        if payload[field] != value:
            raise GenerationPreparationError(f"point payload identity mismatch for {field}")


def _rfc3339_date(value: Any) -> str | None:
    return f"{value}T00:00:00Z" if isinstance(value, str) and value else None


def _validate_canonical_payload(
    point_id: str,
    payload: Mapping[str, Any],
    snapshot_record: Mapping[str, Any],
    cfg: Config,
) -> str:
    from .generation import CANONICAL_PAYLOAD_REQUIRED_FIELDS, CANONICAL_PAYLOAD_REVISION

    missing = sorted(CANONICAL_PAYLOAD_REQUIRED_FIELDS - set(payload))
    if missing:
        raise GenerationPreparationError(
            f"point {point_id} lacks canonical payload fields: {missing}"
        )
    if payload["canonical_payload_revision"] != CANONICAL_PAYLOAD_REVISION:
        raise GenerationPreparationError("canonical payload revision mismatch")
    if payload["canonical_text_exact"] is not True:
        raise GenerationPreparationError("canonical text is not exact")
    if payload["model_revision"] != cfg.embedding_revision:
        raise GenerationPreparationError("canonical model revision mismatch")
    text = payload.get("text")
    if not isinstance(text, str) or not text:
        raise GenerationPreparationError("canonical point text is missing")
    passage_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if payload.get("passage_hash") != passage_hash:
        raise GenerationPreparationError("canonical passage hash mismatch")
    if payload.get("canonical_content_hash") != snapshot_record["content_hash"]:
        raise GenerationPreparationError("canonical content hash mismatch")
    if payload.get("source_fingerprint") != snapshot_record["source_fingerprint"]:
        raise GenerationPreparationError("canonical source fingerprint mismatch")
    if payload.get("normalizer_revision") != snapshot_record["normalizer_revision"]:
        raise GenerationPreparationError("canonical normalizer revision mismatch")
    if payload.get("chunker_revision") != STRUCTURAL_CHUNKER_REVISION:
        raise GenerationPreparationError("canonical chunker revision mismatch")
    start = _require_int(payload.get("char_start"), field="char_start")
    end = _require_int(payload.get("char_end"), field="char_end", minimum=1)
    if end <= start or end - start != len(text):
        raise GenerationPreparationError("canonical character offsets are invalid")
    if snapshot_record["body_markdown"][start:end] != text:
        raise GenerationPreparationError("point text does not equal its snapshot body slice")
    expected_passage = "passage:" + hashlib.sha256(
        (
            f"{snapshot_record['source']}\0{snapshot_record['document_id']}\0"
            f"{snapshot_record['version_id']}\0{start}\0{end}\0{passage_hash}"
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("passage_id") != expected_passage:
        raise GenerationPreparationError("canonical passage ID mismatch")
    canonical = snapshot_doc_to_canonical(snapshot_record, strict=True)
    boundary_mapping = [boundary.to_dict() for boundary in canonical.page_boundaries]
    mapping_sha = (
        hashlib.sha256(
            json.dumps(
                boundary_mapping,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if boundary_mapping
        else None
    )
    expected_pages = [
        boundary.page
        for boundary in canonical.page_boundaries
        if start < boundary.char_end and end > boundary.char_start
    ]
    expected_page_start = expected_pages[0] if expected_pages else None
    expected_page_end = expected_pages[-1] if expected_pages else None
    if canonical.page_boundaries and not expected_pages:
        raise GenerationPreparationError("canonical chunk does not intersect a physical page")
    _expect_payload(
        payload,
        {
            "admissible": True,
            "page_coordinate_reason": canonical.page_coordinate_reason,
            "page_start": expected_page_start,
            "page_end": expected_page_end,
            "page_boundary_mapping_sha256": mapping_sha,
            "page_boundaries": boundary_mapping if payload.get("chunk_index") == 0 else None,
        },
    )
    lineage = {
        "version_id": snapshot_record["version_id"],
        "supersedes": snapshot_record["supersedes"],
        "effective_from": _rfc3339_date(snapshot_record["effective_from"]),
        "effective_to": _rfc3339_date(snapshot_record["effective_to"]),
        "repeal_date": _rfc3339_date(snapshot_record["repeal_date"]),
        "consolidation_status": snapshot_record["consolidation_status"],
        "version_lineage_status": snapshot_record["version_lineage_status"],
        "version_lineage_complete": snapshot_record["version_lineage_complete"],
        "official_url": snapshot_record["official_url"] or snapshot_record["source_url"],
        "official_binary_url": snapshot_record["official_binary_url"]
        or snapshot_record["source_binary_url"],
        "source_authority": snapshot_record["source_authority"],
        "freshness_sla_met": snapshot_record["freshness_sla_met"],
    }
    _expect_payload(payload, lineage)
    return passage_hash


def _validate_inventory_chunk_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    document_id: str,
    version_id: str,
    snapshot_record: Mapping[str, Any],
    cfg: Config,
    row: Sequence[Any],
) -> None:
    (
        document_chunk_count,
        passage_sha256,
        passage_char_length,
        passage_utf8_length,
        token_count,
        char_start,
        char_end,
        structure_json,
        page_json,
        embed_input_json,
    ) = row
    text = payload.get("text")
    if not isinstance(text, str):
        raise GenerationPreparationError("point text is not a string")
    structure = json.loads(structure_json)
    page = json.loads(page_json)
    embed_input = json.loads(embed_input_json)
    local_parent_id = structure["parent_id"]
    parent_id = (
        f"{source}:{document_id}:{version_id}:{local_parent_id}:"
        f"{structure['parent_chunk_index']}"
        if local_parent_id
        else None
    )
    expected = {
        "document_chunk_count": document_chunk_count,
        "passage_hash": passage_sha256,
        "passage_content_hash": passage_sha256,
        "token_count": token_count,
        "char_start": char_start,
        "char_end": char_end,
        "offset_unit": "unicode_codepoint",
        "heading_path": structure["heading_path"],
        "heading": " > ".join(structure["heading_path"]) or None,
        "article_id": structure["article_id"],
        "article_label": structure["article_label"],
        "article_start": structure["article_start"],
        "clause": structure["clause"],
        "clause_id": structure["clause_id"] or structure["clause"],
        "clause_ids": structure["clause_ids"],
        "subarticle": structure["subarticle"],
        "subarticle_ids": structure["subarticle_ids"],
        "chapter": structure["chapter"],
        "structural_parent_id": local_parent_id,
        "parent_id": parent_id,
        "article_start_chunk_index": structure["article_start_chunk_index"],
        "parent_chunk_index": structure["parent_chunk_index"],
        "chunker_revision": structure["chunker_revision"],
        **page,
    }
    _expect_payload(payload, expected)
    if (
        hashlib.sha256(text.encode("utf-8")).hexdigest() != passage_sha256
        or len(text) != passage_char_length
        or len(text.encode("utf-8")) != passage_utf8_length
    ):
        raise GenerationPreparationError(
            "point passage bytes differ from exact structural inventory chunk"
        )
    _expect_payload(
        payload,
        {
            "title": snapshot_record["title"],
            "document_type": snapshot_record["document_type"],
            "document_number": snapshot_record["document_number"],
            "date": _rfc3339_date(snapshot_record["date"]),
            "date_raw": snapshot_record["date_raw"],
            "status": snapshot_record["status"],
            "is_consolidated": snapshot_record["is_consolidated"],
        },
    )
    header_v2: dict[str, Any] = {}
    if cfg.embed_header_v2:
        header_v2 = {
            "document_number": payload["document_number"],
            "date": payload["date"] or payload["date_raw"],
            "status": payload["status"],
            "is_consolidated": payload["is_consolidated"],
        }
    encoded_text = build_embed_text(
        text,
        title=payload["title"],
        document_type=payload["document_type"],
        heading_path=payload["heading_path"],
        **header_v2,
    )
    encoded_bytes = encoded_text.encode("utf-8")
    if {
        "sha256": hashlib.sha256(encoded_bytes).hexdigest(),
        "char_length": len(encoded_text),
        "utf8_length": len(encoded_bytes),
    } != {
        field: embed_input[field]
        for field in ("sha256", "char_length", "utf8_length")
    }:
        raise GenerationPreparationError(
            "point context-enriched embed input differs from sealed structural inventory"
        )


def _scan_collection(
    client: Any,
    connection: sqlite3.Connection,
    inputs: ValidatedPreparationInputs,
    cfg: Config,
    *,
    batch_size: int,
) -> int:
    identity = store.generation_point_identity(cfg)
    assert identity is not None
    expected_identity = identity.as_payload()
    expected_identity["retrieval_fingerprint_revision"] = RETRIEVAL_FINGERPRINT_REVISION
    offset: Any = None
    seen_offsets: set[str] = set()
    total = 0
    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=inputs.physical_collection,
                with_payload=True,
                with_vectors=True,
                limit=batch_size,
                offset=offset,
            )
        except Exception as exc:
            raise GenerationPreparationError(f"read-only Qdrant scan failed: {exc}") from exc
        if not isinstance(points, list):
            points = list(points)
        if not points and next_offset is not None:
            raise GenerationPreparationError(
                "Qdrant scroll returned an empty page with a continuation offset"
            )
        for point in points:
            point_id, payload, vectors = _point_parts(point)
            _validate_point_vectors(point_id, vectors, cfg)
            try:
                point_sha = point_content_sha256(
                    point_id,
                    payload,
                    vectors,
                    dense_name="dense",
                    sparse_name="sparse",
                )
            except ValueError as exc:
                raise GenerationPreparationError(
                    f"point {point_id} cannot enter collection digest: {exc}"
                ) from exc
            source = payload.get("source")
            document_id = payload.get("document_id")
            version_id = payload.get("version_id")
            if not all(isinstance(value, str) and value for value in (source, document_id, version_id)):
                raise GenerationPreparationError("point lacks source/document/version identity")
            chunk_index = _require_int(payload.get("chunk_index"), field="chunk_index")
            declared_chunks = _require_int(
                payload.get("document_chunk_count"),
                field="document_chunk_count",
                minimum=1,
            )
            expected_point_id = store.point_id(
                source,
                document_id,
                chunk_index,
                version_id=version_id,
            )
            if point_id != expected_point_id:
                raise GenerationPreparationError("deterministic point ID mismatch")
            _expect_payload(payload, expected_identity)
            expected = connection.execute(
                "SELECT source_identity, content_hash, document_state_hash, "
                "content_kind, extraction_status, article_summary, source_binary_url, "
                "record_json FROM expected WHERE source=? AND document_id=? AND version_id=?",
                (source, document_id, version_id),
            ).fetchone()
            if expected is None:
                raise GenerationPreparationError(
                    f"extra point outside clean snapshot ledger: {source}:{document_id}:{version_id}"
                )
            expected_chunk = connection.execute(
                "SELECT document_chunk_count, canonical_passage_sha256, "
                "canonical_passage_char_length, canonical_passage_utf8_length, "
                "token_count, char_start, char_end, structure_json, page_json, "
                "embed_input_json FROM expected_chunks WHERE source=? AND "
                "document_id=? AND version_id=? AND chunk_index=?",
                (source, document_id, version_id, chunk_index),
            ).fetchone()
            if expected_chunk is None:
                raise GenerationPreparationError(
                    "point has no exact sealed structural inventory chunk: "
                    f"{source}:{document_id}:{version_id}:{chunk_index}"
                )
            (
                _source_identity,
                expected_content_hash,
                expected_state_hash,
                expected_kind,
                expected_extraction,
                expected_summary,
                expected_binary,
                record_json,
            ) = expected
            _expect_payload(
                payload,
                {
                    "content_hash": expected_content_hash,
                    "document_state_hash": expected_state_hash,
                    "content_kind": expected_kind,
                    "content_complete": True,
                    "extraction_status": expected_extraction,
                    "article_summary": expected_summary,
                    "source_binary_url": expected_binary,
                },
            )
            snapshot_record = json.loads(record_json)
            _validate_canonical_payload(point_id, payload, snapshot_record, cfg)
            _validate_inventory_chunk_payload(
                payload,
                source=source,
                document_id=document_id,
                version_id=version_id,
                snapshot_record=snapshot_record,
                cfg=cfg,
                row=expected_chunk,
            )
            text_hash = hashlib.sha256(payload["text"].encode("utf-8")).hexdigest()
            try:
                connection.execute(
                    "INSERT INTO observed VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        source,
                        document_id,
                        version_id,
                        chunk_index,
                        point_id,
                        text_hash,
                        declared_chunks,
                        point_sha,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise GenerationPreparationError(
                    "duplicate logical chunk or point ID during scan"
                ) from exc
            total += 1
            if total % 1000 == 0:
                connection.commit()
        connection.commit()
        if next_offset is None:
            break
        marker = repr(next_offset)
        if marker in seen_offsets:
            raise GenerationPreparationError("Qdrant scroll offset cycled or did not advance")
        seen_offsets.add(marker)
        offset = next_offset
    return total


def _validate_join(connection: sqlite3.Connection) -> tuple[int, int]:
    documents = chunks = 0
    rows = connection.execute(
        """
        SELECT e.source, e.document_id, e.version_id,
               COUNT(c.chunk_index), MIN(c.chunk_index), MAX(c.chunk_index),
               MIN(c.document_chunk_count), MAX(c.document_chunk_count),
               COUNT(o.chunk_index), MIN(o.declared_chunks), MAX(o.declared_chunks)
        FROM expected AS e
        LEFT JOIN expected_chunks AS c
          ON c.source=e.source AND c.document_id=e.document_id AND c.version_id=e.version_id
        LEFT JOIN observed AS o
          ON o.source=c.source AND o.document_id=c.document_id
         AND o.version_id=c.version_id AND o.chunk_index=c.chunk_index
        GROUP BY e.source, e.document_id, e.version_id
        ORDER BY e.source, e.document_id, e.version_id
        """
    )
    for (
        source,
        document_id,
        version_id,
        expected_count,
        low,
        high,
        inventory_declared_low,
        inventory_declared_high,
        observed_count,
        observed_declared_low,
        observed_declared_high,
    ) in rows:
        if (
            expected_count < 1
            or low != 0
            or high != expected_count - 1
            or inventory_declared_low != expected_count
            or inventory_declared_high != expected_count
            or observed_count != expected_count
            or observed_declared_low != expected_count
            or observed_declared_high != expected_count
        ):
            raise GenerationPreparationError(
                "structural-inventory/Qdrant chunk accounting mismatch for "
                f"{source}:{document_id}:{version_id}"
            )
        documents += 1
        chunks += expected_count
    unexpected = connection.execute(
        """
        SELECT COUNT(*) FROM observed AS o
        LEFT JOIN expected_chunks AS c
          ON c.source=o.source AND c.document_id=o.document_id
         AND c.version_id=o.version_id AND c.chunk_index=o.chunk_index
        WHERE c.source IS NULL
        """
    ).fetchone()[0]
    if unexpected:
        raise GenerationPreparationError(
            "scan contains points outside exact structural chunk inventory"
        )
    return documents, chunks


def _whole_collection_digest(connection: sqlite3.Connection) -> tuple[str, int]:
    try:
        return whole_collection_sha256(
            (str(point_id), str(point_sha))
            for point_id, point_sha in connection.execute(
                "SELECT point_id, point_sha256 FROM observed ORDER BY point_id"
            )
        )
    except ValueError as exc:
        raise GenerationPreparationError(
            f"cannot seal whole-collection digest: {exc}"
        ) from exc


def _source_state(inputs: ValidatedPreparationInputs) -> dict[str, Any]:
    manifest = inputs.snapshot_manifest
    runs = [dict(run) for run in manifest["runs"]]
    if not runs or any(run.get("success_verified") is not True for run in runs):
        raise GenerationPreparationError("source state is not success-attested")
    covered_sources = {run["source"] for run in runs}
    if covered_sources != set(SOURCES_PRESENT):
        raise GenerationPreparationError("source state does not cover all seven sources")
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_sha256": manifest["snapshot_sha256"],
        "corpus_sha256": manifest["corpus_sha256"],
        "structural_chunk_inventory": {
            field: manifest["structural_chunk_inventory"][field]
            for field in (
                "sha256",
                "size_bytes",
                "identity_sha256",
                "record_count",
                "document_count",
                "chunk_count",
            )
        },
        "source_state_evidence": manifest["source_state_evidence"],
        "runs": runs,
    }


def _build_manifest(
    inputs: ValidatedPreparationInputs,
    cfg: Config,
    source_state: Mapping[str, Any],
    *,
    document_count: int,
    chunk_count: int,
    actor: str,
    run_id: str,
    created_at: str,
) -> GenerationManifest:
    covered_runs = sorted(
        {("%s" % run["source"], "%s" % run["run_id"]) for run in source_state["runs"]}
    )
    return GenerationManifest.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": inputs.generation_id,
            "document_count": document_count,
            "indexed_document_count": document_count,
            "excluded_document_count": 0,
            "chunk_count": chunk_count,
            "sample_count": document_count,
            "corpus": {
                "name": inputs.snapshot_manifest["snapshot_id"],
                "snapshot_sha256": inputs.snapshot_manifest["snapshot_sha256"],
            },
            "source": {
                "name": "attested_snapshot_source_state",
                "state_sha256": source_state_sha256(source_state),
            },
            "model": {
                "embedding_model": cfg.embed_model,
                "embedding_revision": cfg.embedding_revision,
                "tokenizer_model": cfg.tokenizer_model,
                "tokenizer_revision": cfg.tokenizer_revision,
                "reranker_model": cfg.rerank_model,
                "reranker_revision": cfg.reranker_revision,
            },
            "vector_space": {
                "id": store.vector_space_id(cfg),
                "dense_name": "dense",
                "dense_dimension": cfg.dense_dim,
                "distance": "cosine",
                "sparse_name": "sparse",
            },
            "chunking": {
                "fingerprint": store.chunking_fingerprint(cfg),
                "max_tokens": cfg.chunk_tokens,
                "overlap_tokens": cfg.chunk_overlap,
                "document_header": cfg.embed_header_v2,
            },
            "covered_runs": [
                {"source": source, "run_id": covered_run_id}
                for source, covered_run_id in covered_runs
            ],
            "retrieval_fingerprint_revision": RETRIEVAL_FINGERPRINT_REVISION,
            "retrieval_fingerprint": retrieval_fingerprint_sha256(cfg),
            "code": dict(inputs.code_identity),
            "dependency": {
                "lock_sha256": inputs.dependency_lock_sha256,
                "image_digest": inputs.image_digest,
            },
            "creation": {
                "created_at": created_at,
                "run_id": run_id,
                "actor": actor,
            },
        }
    )


def _document_rows(
    connection: sqlite3.Connection, generation_id: str
) -> Iterator[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT e.source, e.document_id, e.version_id, e.source_identity,
               e.content_hash, e.document_state_hash, e.content_kind,
               e.extraction_status, e.article_summary, e.source_binary_url,
               COUNT(o.chunk_index)
        FROM expected AS e JOIN observed AS o
          ON o.source=e.source AND o.document_id=e.document_id AND o.version_id=e.version_id
        GROUP BY e.source, e.document_id, e.version_id
        ORDER BY e.source, e.document_id, e.version_id
        """
    )
    for row in rows:
        yield {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "source": row[0],
            "document_id": row[1],
            "version_id": row[2],
            "source_identity": row[3],
            "content_hash": row[4],
            "document_state_hash": row[5],
            "expected_chunk_count": row[10],
            "content_kind": row[6],
            "content_complete": True,
            "extraction_status": row[7],
            "article_summary": row[8],
            "exclusion_reason": None,
            "refresh_deadline": None,
            "source_binary_url": row[9],
        }


def _sample_rows(
    connection: sqlite3.Connection, generation_id: str
) -> Iterator[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT source, document_id, version_id, point_id, text_sha256
        FROM observed WHERE chunk_index=0
        ORDER BY source, document_id, version_id
        """
    )
    for source, document_id, version_id, point_id, text_sha256 in rows:
        yield {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "source": source,
            "document_id": document_id,
            "version_id": version_id,
            "chunk_index": 0,
            "point_id": point_id,
            "text_sha256": text_sha256,
        }


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    with _private_binary_writer(path) as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row, pretty=False))


def _preparation_provenance(
    inputs: ValidatedPreparationInputs,
    cfg: Config,
    manifest: GenerationManifest,
    *,
    scanned_points: int,
    collection_digest: CollectionDigest,
    materialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    point_identity = store.generation_point_identity(cfg)
    assert point_identity is not None
    identity_payload = point_identity.as_payload()
    identity_payload["retrieval_fingerprint_revision"] = RETRIEVAL_FINGERPRINT_REVISION
    provenance = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "kind": PREPARATION_KIND,
        "generation_id": inputs.generation_id,
        "snapshot": {
            "snapshot_id": inputs.snapshot_manifest["snapshot_id"],
            "snapshot_sha256": inputs.snapshot_manifest["snapshot_sha256"],
            "corpus_sha256": inputs.snapshot_manifest["corpus_sha256"],
            "structural_chunk_inventory": {
                field: inputs.snapshot_manifest["structural_chunk_inventory"][field]
                for field in (
                    "sha256",
                    "size_bytes",
                    "identity_sha256",
                    "record_count",
                    "document_count",
                    "chunk_count",
                )
            },
        },
        "serving_collection": SERVING_ALIAS,
        "physical_collection": inputs.physical_collection,
        "queried_collection": inputs.physical_collection,
        "collection_access_kind": "direct_physical",
        "point_identity": identity_payload,
        "embed_binding_sha256": inputs.embed_binding_sha256,
        "vector_checksum_artifact_sha256": inputs.vector_checksum_artifact_sha256,
        "vector_probe_sha256": inputs.vector_probe_sha256,
        "collection_configuration_sha256": (
            collection_digest.collection_configuration_sha256
        ),
        "whole_collection_sha256": collection_digest.collection_sha256,
        "dependency_lock_sha256": inputs.dependency_lock_sha256,
        "runtime_identity_sha256": inputs.runtime_identity_sha256,
        "runtime_identity_canonical_sha256": _canonical_sha256(
            inputs.runtime_identity
        ),
        "runtime_identity": dict(inputs.runtime_identity),
        "final_image_digest": inputs.image_digest,
        "code": dict(inputs.code_identity),
        "creation": manifest.to_dict()["creation"],
        "scan": {
            "document_count": manifest.document_count,
            "chunk_count": scanned_points,
            "sample_count": manifest.sample_count,
        },
    }
    if materialization is not None:
        provenance["materialization"] = dict(materialization)
    return provenance


def _collection_digest(
    inputs: ValidatedPreparationInputs,
    *,
    point_count: int,
    whole_sha256: str,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
) -> CollectionDigest:
    binding_configuration = inputs.embed_binding.get("collection_configuration")
    if (
        not isinstance(binding_configuration, Mapping)
        or binding_configuration.get("sha256") != configuration_sha256
        or binding_configuration.get("value") != configuration
    ):
        raise GenerationPreparationError(
            "physical collection configuration differs from immutable embed binding"
        )
    return CollectionDigest.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "digest_revision": 1,
            "algorithm": "sha256",
            "point_count": point_count,
            "collection_sha256": whole_sha256,
            "physical_collection": inputs.physical_collection,
            "payload_projection": COLLECTION_PAYLOAD_PROJECTION,
            "dense_encoding": COLLECTION_DENSE_ENCODING,
            "sparse_encoding": COLLECTION_SPARSE_ENCODING,
            "collection_configuration_sha256": configuration_sha256,
            "collection_configuration": dict(configuration),
            "vector_checksum_artifact_sha256": (
                inputs.vector_checksum_artifact_sha256
            ),
            "vector_probe_sha256": inputs.vector_probe_sha256,
            "embed_binding_sha256": inputs.embed_binding_sha256,
        }
    )


def _inventory(stage: Path) -> ChecksumInventory:
    return ChecksumInventory.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "algorithm": CHECKSUM_ALGORITHM,
            "files": {
                name: _sha256_file(stage / name) for name in sorted(_PREPARED_FILES)
            },
        }
    )


def _create_parent(path: Path) -> None:
    _reject_symlink_components(path)
    path.mkdir(mode=PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise GenerationPreparationError(f"output parent is not a real directory: {path}")


def prepare_generation(
    client: Any,
    cfg: Config,
    *,
    generation_id: str,
    snapshot_root: Path,
    physical_collection: str,
    dependency_lock: Path,
    runtime_identity: Path,
    image_digest: str,
    embed_binding: Path | None = None,
    vector_checksum: Path | None = None,
    rematerialization_report: Path | None = None,
    actor: str,
    run_id: str,
    output_dir: Path,
    batch_size: int = 1000,
    created_at: datetime | None = None,
    code_identity: Mapping[str, Any] | None = None,
    validated_inputs: ValidatedPreparationInputs | None = None,
) -> Path:
    """Create one sealed prepared directory without mutating Qdrant."""
    if not isinstance(actor, str) or not actor:
        raise GenerationPreparationError("actor must be a non-empty string")
    if not isinstance(run_id, str) or not run_id:
        raise GenerationPreparationError("run_id must be a non-empty string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise GenerationPreparationError("batch_size must be an integer >= 1")
    inputs = validated_inputs or validate_preparation_inputs(
        cfg,
        generation_id=generation_id,
        snapshot_root=snapshot_root,
        physical_collection=physical_collection,
        dependency_lock=dependency_lock,
        runtime_identity=runtime_identity,
        image_digest=image_digest,
        embed_binding=(
            embed_binding
            if embed_binding is not None
            else cfg.state_dir / "embed" / generation_id / "binding.json"
        ),
        vector_checksum=(
            vector_checksum
            if vector_checksum is not None
            else cfg.state_dir / "embed" / generation_id / "vector-space-checksum.json"
        ),
        output_dir=output_dir,
        code_identity=code_identity,
    )
    if inputs.generation_id != generation_id or inputs.physical_collection != physical_collection:
        raise GenerationPreparationError("prevalidated preparation inputs do not match request")
    materialization_report_value: Mapping[str, Any] | None = None
    materialization_report_sha: str | None = None
    if rematerialization_report is not None:
        try:
            materialization_report_value, materialization_report_sha = (
                validate_rematerialization_report(
                    rematerialization_report,
                    generation_id=generation_id,
                    physical_collection=physical_collection,
                )
            )
        except RematerializationError as exc:
            raise GenerationPreparationError(
                f"invalid rematerialization report: {exc}"
            ) from exc
    _validate_snapshot_config(inputs.snapshot_root, inputs.snapshot_manifest, cfg)
    _validate_bound_structural_inventory(
        inputs.snapshot_manifest,
        inputs.embed_binding,
    )
    if os.path.lexists(output_dir):
        raise GenerationPreparationError(
            f"prepared generation destination already exists: {output_dir}"
        )
    _create_parent(output_dir.parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    os.chmod(stage, PRIVATE_DIRECTORY_MODE)
    ledger_path = stage / ".preparation.sqlite"
    connection: sqlite3.Connection | None = None
    try:
        connection = _create_ledger(ledger_path)
        snapshot_documents = _load_snapshot_ledger(connection, inputs, cfg)
        inventory_documents, inventory_chunks = _load_structural_chunk_ledger(
            connection, inputs
        )
        if inventory_documents != snapshot_documents:
            raise GenerationPreparationError(
                "snapshot and structural inventory document counts differ"
            )
        (
            _info,
            advertised_points,
            initial_configuration,
            initial_configuration_sha,
        ) = _inspect_collection(
            client, inputs.physical_collection, cfg
        )
        scanned_points = _scan_collection(
            client,
            connection,
            inputs,
            cfg,
            batch_size=batch_size,
        )
        if advertised_points is not None and advertised_points != scanned_points:
            raise GenerationPreparationError(
                "collection points_count changed or does not match read-only scan"
            )
        (
            _final_info,
            final_points,
            final_configuration,
            final_configuration_sha,
        ) = _inspect_collection(
            client, inputs.physical_collection, cfg
        )
        if final_points != advertised_points or (
            final_points is not None and final_points != scanned_points
        ):
            raise GenerationPreparationError(
                "physical collection changed during read-only preparation scan"
            )
        if (
            final_configuration_sha != initial_configuration_sha
            or final_configuration != initial_configuration
        ):
            raise GenerationPreparationError(
                "physical collection configuration changed during preparation scan"
            )
        document_count, chunk_count = _validate_join(connection)
        if document_count != snapshot_documents or chunk_count != scanned_points:
            raise GenerationPreparationError("snapshot/Qdrant one-to-one join mismatch")
        expected_snapshot_chunks = inputs.snapshot_manifest[
            "structural_chunk_inventory"
        ]["chunk_count"]
        if chunk_count != expected_snapshot_chunks or chunk_count != inventory_chunks:
            raise GenerationPreparationError(
                "Qdrant chunk count differs from the sealed structural chunk inventory"
            )
        whole_sha256, digested_points = _whole_collection_digest(connection)
        if digested_points != scanned_points:
            raise GenerationPreparationError(
                "whole-collection digest point count differs from scan"
            )
        collection_digest = _collection_digest(
            inputs,
            point_count=digested_points,
            whole_sha256=whole_sha256,
            configuration=initial_configuration,
            configuration_sha256=initial_configuration_sha,
        )
        source_state = _source_state(inputs)
        manifest = _build_manifest(
            inputs,
            cfg,
            source_state,
            document_count=document_count,
            chunk_count=chunk_count,
            actor=actor,
            run_id=run_id,
            created_at=_format_utc(created_at or datetime.now(UTC)),
        )
        materialization: Mapping[str, Any] | None = None
        if materialization_report_value is not None:
            assert materialization_report_sha is not None
            if (
                materialization_report_value["point_count"] != scanned_points
                or materialization_report_value["document_count"] != document_count
                or materialization_report_value["target_configuration_sha256"]
                != initial_configuration_sha
                or materialization_report_value["retrieval_fingerprint"]
                != retrieval_fingerprint_sha256(cfg)
            ):
                raise GenerationPreparationError(
                    "rematerialization report differs from the prepared collection"
                )
            materialization = {
                "report_sha256": materialization_report_sha,
                "binding_sha256": materialization_report_value["binding_sha256"],
                "source_collection": materialization_report_value["source_collection"],
                "target_collection": materialization_report_value["target_collection"],
                "document_count": materialization_report_value["document_count"],
                "point_count": materialization_report_value["point_count"],
                "source_logical_vector_sha256": materialization_report_value[
                    "source_logical_vector_sha256"
                ],
                "target_logical_vector_sha256": materialization_report_value[
                    "target_logical_vector_sha256"
                ],
                "old_id_set_sha256": materialization_report_value["old_id_set_sha256"],
                "new_id_set_sha256": materialization_report_value["new_id_set_sha256"],
                "rekey_map_sha256": materialization_report_value["rekey_map_sha256"],
                "court_extractor_revision": materialization_report_value[
                    "court_extractor_revision"
                ],
                "provenance_strength": materialization_report_value[
                    "provenance_strength"
                ],
            }
        _write_jsonl(
            stage / DOCUMENTS_FILENAME,
            _document_rows(connection, generation_id),
        )
        _write_jsonl(
            stage / SAMPLE_CHECKS_FILENAME,
            _sample_rows(connection, generation_id),
        )
        _write_bytes(
            stage / SOURCE_STATE_FILENAME,
            _canonical_json_bytes(source_state, pretty=True),
        )
        provenance = _preparation_provenance(
            inputs,
            cfg,
            manifest,
            scanned_points=scanned_points,
            collection_digest=collection_digest,
            materialization=materialization,
        )
        _write_bytes(
            stage / PREPARATION_PROVENANCE_FILENAME,
            _canonical_json_bytes(provenance, pretty=True),
        )
        _write_bytes(
            stage / MANIFEST_FILENAME,
            _canonical_json_bytes(manifest.to_dict(), pretty=True),
        )
        _write_bytes(
            stage / COLLECTION_DIGEST_FILENAME,
            _canonical_json_bytes(collection_digest.to_dict(), pretty=True),
        )
        connection.close()
        connection = None
        ledger_path.unlink()
        checksums = _inventory(stage)
        _write_bytes(
            stage / CHECKSUM_FILENAME,
            _canonical_json_bytes(checksums.to_dict(), pretty=True),
        )
        _fsync_directory(stage)
        loaded = load_prepared_generation(stage)
        if loaded.manifest != manifest:
            raise GenerationPreparationError("prepared manifest changed during sealing")
        if os.path.lexists(output_dir):
            raise GenerationPreparationError(
                f"prepared generation destination appeared during build: {output_dir}"
            )
        _rename_noreplace(stage, output_dir)
        _fsync_directory(output_dir.parent)
        return output_dir
    except Exception as exc:
        if connection is not None:
            connection.close()
        if isinstance(exc, GenerationPreparationError):
            if exc.staging_path is None:
                exc.staging_path = stage
            raise
        raise GenerationPreparationError(
            f"generation preparation failed; staging retained at {stage}: {exc}",
            staging_path=stage,
        ) from exc


def _validate_prepared_counts(prepared: PreparedGeneration) -> None:
    manifest = prepared.manifest
    sentinel = object()
    document_count = sample_count = chunk_count = 0
    previous_document: tuple[str, str, str] | None = None
    previous_sample: tuple[str, str, str, int] | None = None
    for document, sample in zip_longest(
        prepared.iter_documents(), prepared.iter_samples(), fillvalue=sentinel
    ):
        if document is sentinel or sample is sentinel:
            raise GenerationPreparationError(
                "prepared samples are not one-to-one with documents"
            )
        assert isinstance(document, DocumentRecord)
        assert isinstance(sample, SampleCheck)
        if not document.indexed or not document.content_complete:
            raise GenerationPreparationError(
                "prepared ledger contains excluded/incomplete content"
            )
        document_key = (
            document.source,
            document.document_id,
            document.version_id,
        )
        sample_key = (
            sample.source,
            sample.document_id,
            sample.version_id,
            sample.chunk_index,
        )
        if previous_document is not None and document_key <= previous_document:
            raise GenerationPreparationError(
                "prepared documents are not unique and sorted"
            )
        if previous_sample is not None and sample_key <= previous_sample:
            raise GenerationPreparationError(
                "prepared samples are not unique and sorted"
            )
        if sample.chunk_index != 0 or sample_key[:3] != document_key:
            raise GenerationPreparationError(
                "prepared samples must be one chunk-zero record per document"
            )
        if sample.point_id != store.point_id(
            document.source,
            document.document_id,
            0,
            version_id=document.version_id,
        ):
            raise GenerationPreparationError(
                "prepared sample has a non-deterministic point ID"
            )
        previous_document = document_key
        previous_sample = sample_key
        document_count += 1
        sample_count += 1
        chunk_count += document.expected_chunk_count
    if document_count != manifest.document_count:
        raise GenerationPreparationError("prepared document count mismatch")
    if sample_count != manifest.sample_count:
        raise GenerationPreparationError("prepared sample count mismatch")
    if chunk_count != manifest.chunk_count:
        raise GenerationPreparationError("prepared chunk accounting mismatch")


def _validate_prepared_source_state(
    value: Mapping[str, Any], manifest: GenerationManifest
) -> None:
    expected_keys = {
        "schema_version",
        "snapshot_id",
        "snapshot_sha256",
        "corpus_sha256",
        "structural_chunk_inventory",
        "source_state_evidence",
        "runs",
    }
    if set(value) != expected_keys:
        raise GenerationPreparationError("prepared source_state has invalid keys")
    if (
        value["schema_version"] != GENERATION_SCHEMA_VERSION
        or value["snapshot_id"] != manifest.corpus.name
        or value["snapshot_sha256"] != manifest.corpus.snapshot_sha256
        or not isinstance(value["corpus_sha256"], str)
        or not _SHA256_RE.fullmatch(value["corpus_sha256"])
    ):
        raise GenerationPreparationError("prepared source_state snapshot identity mismatch")
    structural_inventory = value["structural_chunk_inventory"]
    if (
        not isinstance(structural_inventory, dict)
        or set(structural_inventory)
        != {
            "sha256",
            "size_bytes",
            "identity_sha256",
            "record_count",
            "document_count",
            "chunk_count",
        }
        or structural_inventory["document_count"] != manifest.document_count
        or structural_inventory["chunk_count"] != manifest.chunk_count
        or any(
            not isinstance(structural_inventory[field], str)
            or not _SHA256_RE.fullmatch(structural_inventory[field])
            for field in ("sha256", "identity_sha256")
        )
        or any(
            isinstance(structural_inventory[field], bool)
            or not isinstance(structural_inventory[field], int)
            or structural_inventory[field] < minimum
            for field, minimum in (
                ("size_bytes", 1),
                ("record_count", 1),
                ("document_count", 0),
                ("chunk_count", 0),
            )
        )
    ):
        raise GenerationPreparationError(
            "prepared source_state structural chunk inventory is invalid"
        )
    evidence = value["source_state_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"sha256", "size_bytes"}:
        raise GenerationPreparationError("prepared source-state evidence is invalid")
    if (
        not isinstance(evidence["sha256"], str)
        or not _SHA256_RE.fullmatch(evidence["sha256"])
        or isinstance(evidence["size_bytes"], bool)
        or not isinstance(evidence["size_bytes"], int)
        or evidence["size_bytes"] < 0
    ):
        raise GenerationPreparationError("prepared source-state evidence is malformed")
    runs = value["runs"]
    if not isinstance(runs, list) or not runs:
        raise GenerationPreparationError("prepared source state has no attested runs")
    keys: list[tuple[str, str]] = []
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "source",
            "run_id",
            "items",
            "completion_record",
            "completed_at",
            "success_verified",
        }:
            raise GenerationPreparationError("prepared run attestation has invalid keys")
        source = run["source"]
        run_id = run["run_id"]
        if (
            source not in SOURCES_PRESENT
            or not isinstance(run_id, str)
            or not _RUN_ID_RE.fullmatch(run_id)
        ):
            raise GenerationPreparationError("prepared run identity is invalid")
        if run["success_verified"] is not True or not isinstance(
            run["completed_at"], str
        ):
            raise GenerationPreparationError("prepared run is not success-attested")
        try:
            parse_rfc3339_utc(run["completed_at"], field="source_state.completed_at")
        except ValueError as exc:
            raise GenerationPreparationError(
                "prepared run completion timestamp is invalid"
            ) from exc
        for field in ("items", "completion_record"):
            artifact = run[field]
            if not isinstance(artifact, dict) or set(artifact) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise GenerationPreparationError(
                    f"prepared run {field} binding is invalid"
                )
            if (
                artifact["path"]
                != f"{source}/runs/{run_id}/"
                + ("items.jsonl" if field == "items" else "run.json")
                or not isinstance(artifact["sha256"], str)
                or not _SHA256_RE.fullmatch(artifact["sha256"])
                or isinstance(artifact["size_bytes"], bool)
                or not isinstance(artifact["size_bytes"], int)
                or artifact["size_bytes"] < 0
            ):
                raise GenerationPreparationError(
                    f"prepared run {field} binding is malformed"
                )
        keys.append((source, run_id))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise GenerationPreparationError("prepared run inventory is not unique and sorted")
    if {source for source, _run_id in keys} != set(SOURCES_PRESENT):
        raise GenerationPreparationError("prepared run inventory lacks a required source")
    if keys != [(run.source, run.run_id) for run in manifest.covered_runs]:
        raise GenerationPreparationError("prepared runs differ from manifest.covered_runs")


def load_prepared_generation(root: Path | str) -> PreparedGeneration:
    """Verify an exact sealed prepared directory before publication."""
    directory = Path(root)
    _reject_symlink_components(directory)
    try:
        root_mode = directory.lstat().st_mode
        if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
            raise GenerationPreparationError(
                "prepared root must be a non-symlink directory"
            )
        entries = list(directory.iterdir())
    except OSError as exc:
        raise GenerationPreparationError(f"cannot inspect prepared root: {exc}") from exc
    expected_names = set(_PREPARED_FILES) | {CHECKSUM_FILENAME}
    if {entry.name for entry in entries} != expected_names:
        raise GenerationPreparationError(
            "prepared directory has missing or unexplained filesystem entries"
        )
    for entry in entries:
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise GenerationPreparationError(
                f"prepared artifact must be a regular non-symlink file: {entry.name}"
            )
    checksum_bytes, checksums_sha256 = _read_bound_file(
        directory / CHECKSUM_FILENAME, None
    )
    checksum_value = _strict_json_line(
        checksum_bytes, origin=str(directory / CHECKSUM_FILENAME)
    )
    checksums = ChecksumInventory.from_dict(checksum_value)
    if set(checksums.files) != set(_PREPARED_FILES):
        raise GenerationPreparationError(
            "prepared checksum inventory has missing or unexplained artifacts"
        )
    manifest_bytes, _manifest_sha = _read_bound_file(
        directory / MANIFEST_FILENAME,
        checksums.files[MANIFEST_FILENAME],
    )
    manifest = GenerationManifest.from_dict(
        _strict_json_line(manifest_bytes, origin="prepared manifest")
    )
    source_bytes, _source_sha = _read_bound_file(
        directory / SOURCE_STATE_FILENAME,
        checksums.files[SOURCE_STATE_FILENAME],
    )
    source_state = _strict_json_line(source_bytes, origin="prepared source_state")
    provenance_bytes, _provenance_sha = _read_bound_file(
        directory / PREPARATION_PROVENANCE_FILENAME,
        checksums.files[PREPARATION_PROVENANCE_FILENAME],
    )
    provenance = _strict_json_line(
        provenance_bytes,
        origin="preparation provenance",
    )
    if not isinstance(source_state, dict):
        raise GenerationPreparationError("prepared source_state must be an object")
    if not isinstance(provenance, dict):
        raise GenerationPreparationError("preparation provenance must be an object")
    if source_state_sha256(source_state) != manifest.source.state_sha256:
        raise GenerationPreparationError("prepared source_state hash mismatch")
    _validate_prepared_source_state(source_state, manifest)
    expected_provenance_keys = {
        "schema_version",
        "kind",
        "generation_id",
        "snapshot",
        "serving_collection",
        "physical_collection",
        "queried_collection",
        "collection_access_kind",
        "point_identity",
        "embed_binding_sha256",
        "vector_checksum_artifact_sha256",
        "vector_probe_sha256",
        "collection_configuration_sha256",
        "whole_collection_sha256",
        "dependency_lock_sha256",
        "runtime_identity_sha256",
        "runtime_identity_canonical_sha256",
        "runtime_identity",
        "final_image_digest",
        "code",
        "creation",
        "scan",
    }
    observed_provenance_keys = set(provenance)
    if observed_provenance_keys not in (
        expected_provenance_keys,
        expected_provenance_keys | {"materialization"},
    ):
        raise GenerationPreparationError("preparation provenance has invalid keys")
    expected_physical = physical_collection_name(manifest.generation_id)
    if (
        provenance["schema_version"] != GENERATION_SCHEMA_VERSION
        or provenance["kind"] != PREPARATION_KIND
        or provenance["generation_id"] != manifest.generation_id
        or provenance["serving_collection"] != SERVING_ALIAS
        or provenance["physical_collection"] != expected_physical
        or provenance["queried_collection"] != expected_physical
        or provenance["collection_access_kind"] != "direct_physical"
        or provenance["dependency_lock_sha256"] != manifest.dependency.lock_sha256
        or provenance["final_image_digest"] != manifest.dependency.image_digest
        or provenance["creation"] != manifest.to_dict()["creation"]
        or provenance["code"] != manifest.to_dict()["code"]
    ):
        raise GenerationPreparationError("preparation provenance identity mismatch")
    runtime_value = provenance["runtime_identity"]
    if not isinstance(runtime_value, dict) or runtime_value.get("status") != "validated":
        raise GenerationPreparationError("prepared runtime identity is not validated")
    if runtime_value.get("requirements_lock_sha256") != manifest.dependency.lock_sha256:
        raise GenerationPreparationError("prepared runtime/lock identity mismatch")
    if _canonical_sha256(runtime_value) != provenance[
        "runtime_identity_canonical_sha256"
    ]:
        raise GenerationPreparationError("prepared runtime identity hash mismatch")
    if not isinstance(provenance["runtime_identity_sha256"], str) or not _SHA256_RE.fullmatch(
        provenance["runtime_identity_sha256"]
    ):
        raise GenerationPreparationError("prepared runtime identity file hash is invalid")
    point_identity = provenance["point_identity"]
    if not isinstance(point_identity, dict) or point_identity.get(
        "retrieval_fingerprint_revision"
    ) != RETRIEVAL_FINGERPRINT_REVISION:
        raise GenerationPreparationError("prepared retrieval fingerprint revision mismatch")
    expected_point_identity = {
        "schema_version": manifest.schema_version,
        "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        "generation_id": manifest.generation_id,
        "embedding_model": manifest.model.embedding_model,
        "embedding_revision": manifest.model.embedding_revision,
        "model_revision": manifest.model.embedding_revision,
        "tokenizer_model": manifest.model.tokenizer_model,
        "tokenizer_revision": manifest.model.tokenizer_revision,
        "reranker_model": manifest.model.reranker_model,
        "reranker_revision": manifest.model.reranker_revision,
        "vector_space_id": manifest.vector_space.id,
        "chunking_fingerprint": manifest.chunking.fingerprint,
        "document_header": manifest.chunking.document_header,
        "retrieval_fingerprint": manifest.retrieval_fingerprint,
        "retrieval_fingerprint_revision": manifest.retrieval_fingerprint_revision,
    }
    if point_identity != expected_point_identity:
        raise GenerationPreparationError(
            "preparation point identity differs from generation manifest"
        )
    snapshot_identity = provenance["snapshot"]
    if not isinstance(snapshot_identity, dict) or snapshot_identity != {
        "snapshot_id": manifest.corpus.name,
        "snapshot_sha256": manifest.corpus.snapshot_sha256,
        "corpus_sha256": source_state["corpus_sha256"],
        "structural_chunk_inventory": source_state["structural_chunk_inventory"],
    }:
        raise GenerationPreparationError("preparation provenance snapshot mismatch")
    structural_inventory = snapshot_identity["structural_chunk_inventory"]
    if (
        not isinstance(structural_inventory, dict)
        or set(structural_inventory)
        != {
            "sha256",
            "size_bytes",
            "identity_sha256",
            "record_count",
            "document_count",
            "chunk_count",
        }
        or structural_inventory["document_count"] != manifest.document_count
        or structural_inventory["chunk_count"] != manifest.chunk_count
        or any(
            not isinstance(structural_inventory[field], str)
            or not _SHA256_RE.fullmatch(structural_inventory[field])
            for field in ("sha256", "identity_sha256")
        )
    ):
        raise GenerationPreparationError(
            "preparation provenance structural chunk inventory mismatch"
        )
    if provenance["scan"] != {
        "document_count": manifest.document_count,
        "chunk_count": manifest.chunk_count,
        "sample_count": manifest.sample_count,
    }:
        raise GenerationPreparationError("preparation provenance scan counts mismatch")
    digest_bytes, _digest_file_sha = _read_bound_file(
        directory / COLLECTION_DIGEST_FILENAME,
        checksums.files[COLLECTION_DIGEST_FILENAME],
    )
    collection_digest = CollectionDigest.from_dict(
        _strict_json_line(digest_bytes, origin="prepared collection digest")
    )
    if (
        collection_digest.point_count != manifest.chunk_count
        or collection_digest.physical_collection != expected_physical
        or provenance["embed_binding_sha256"]
        != collection_digest.embed_binding_sha256
        or provenance["vector_checksum_artifact_sha256"]
        != collection_digest.vector_checksum_artifact_sha256
        or provenance["vector_probe_sha256"]
        != collection_digest.vector_probe_sha256
        or provenance["collection_configuration_sha256"]
        != collection_digest.collection_configuration_sha256
        or provenance["whole_collection_sha256"]
        != collection_digest.collection_sha256
    ):
        raise GenerationPreparationError(
            "prepared collection digest/provenance identity mismatch"
        )
    if "materialization" in provenance:
        materialization = provenance["materialization"]
        expected_materialization_keys = {
            "report_sha256",
            "binding_sha256",
            "source_collection",
            "target_collection",
            "document_count",
            "point_count",
            "source_logical_vector_sha256",
            "target_logical_vector_sha256",
            "old_id_set_sha256",
            "new_id_set_sha256",
            "rekey_map_sha256",
            "court_extractor_revision",
            "provenance_strength",
        }
        if (
            not isinstance(materialization, dict)
            or set(materialization) != expected_materialization_keys
            or materialization["target_collection"] != expected_physical
            or materialization["document_count"] != manifest.document_count
            or materialization["point_count"] != manifest.chunk_count
            or materialization["source_logical_vector_sha256"]
            != materialization["target_logical_vector_sha256"]
            or materialization["provenance_strength"]
            != "legacy-empirically-attested"
        ):
            raise GenerationPreparationError(
                "prepared rematerialization provenance is invalid"
            )
        for field in (
            "report_sha256",
            "binding_sha256",
            "source_logical_vector_sha256",
            "target_logical_vector_sha256",
            "old_id_set_sha256",
            "new_id_set_sha256",
            "rekey_map_sha256",
        ):
            if not isinstance(materialization[field], str) or not _SHA256_RE.fullmatch(
                materialization[field]
            ):
                raise GenerationPreparationError(
                    "prepared rematerialization digest is invalid"
                )
    prepared = PreparedGeneration(
        root=directory,
        manifest=manifest,
        checksums=checksums,
        checksums_sha256=checksums_sha256,
        source_state=source_state,
        provenance=provenance,
        collection_digest=collection_digest,
    )
    _validate_prepared_counts(prepared)
    return prepared


def validate_prepared_configuration(prepared: PreparedGeneration, cfg: Config) -> None:
    """Revalidate current model/vector/chunk/retrieval configuration at publication."""
    manifest = prepared.manifest
    point_identity = store.generation_point_identity(
        dataclasses.replace(cfg, generation_id=manifest.generation_id)
    )
    if point_identity is None:
        raise GenerationPreparationError("current generation configuration is incomplete")
    expected = point_identity.as_payload()
    observed = prepared.provenance["point_identity"]
    for field, value in expected.items():
        if observed.get(field) != value:
            raise GenerationPreparationError(
                f"current configuration drifted at {field}"
            )
    if observed.get("retrieval_fingerprint_revision") != RETRIEVAL_FINGERPRINT_REVISION:
        raise GenerationPreparationError("current retrieval fingerprint revision drifted")


__all__ = [
    "PREPARATION_PROVENANCE_FILENAME",
    "PREPARATION_KIND",
    "SOURCE_STATE_FILENAME",
    "GenerationPreparationError",
    "PreparedGeneration",
    "ValidatedPreparationInputs",
    "load_prepared_generation",
    "prepare_generation",
    "validate_preparation_inputs",
    "validate_prepared_configuration",
]
