"""Atomic publisher for immutable, checksum-verified corpus generations.

This module never mutates an existing generation.  It builds a complete candidate in an
owner-only sibling staging directory, validates the strict generation format, writes the
manifest last, fsyncs the result, and renames the directory into place while holding a
local publisher lock.  A failed staging directory is retained as evidence and can never
be mistaken for a published generation because it has no final destination name.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO

from .generation import (
    CHECKSUM_ALGORITHM,
    CHECKSUM_FILENAME,
    DOCUMENTS_FILENAME,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    SAMPLE_CHECKS_FILENAME,
    ChecksumInventory,
    DocumentRecord,
    GenerationFormatError,
    GenerationManifest,
    SampleCheck,
    load_generation,
    validate_generation_id,
)

PROVENANCE_FILENAME = "provenance.json"
SOURCE_STATE_FILENAME = "source_state.json"
QUARANTINE_FILENAME = "quarantine.jsonl"

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class GenerationPublishError(RuntimeError):
    """A staged generation could not be validated or atomically published."""

    def __init__(self, message: str, *, staging_path: Path | None = None) -> None:
        super().__init__(message)
        self.staging_path = staging_path


class GenerationDestinationExists(GenerationPublishError):
    """The immutable destination or a conflicting filesystem entry already exists."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing a raced destination."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise GenerationPublishError(
            "atomic no-replace rename is unavailable; generation retained in staging",
            staging_path=source,
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise GenerationDestinationExists(
            f"generation destination appeared during build: {destination}",
            staging_path=source,
        )
    if error in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        raise GenerationPublishError(
            "filesystem does not support atomic no-replace generation publication",
            staging_path=source,
        )
    raise OSError(error, os.strerror(error), destination)


def _create_private_directory(path: Path) -> None:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    components = [cursor]
    for part in absolute.parts[1:]:
        cursor /= part
        components.append(cursor)

    def require_real_directory(directory: Path) -> bool:
        try:
            mode = directory.lstat().st_mode
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise GenerationPublishError(
                f"cannot inspect generation output path {directory}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise GenerationPublishError(f"not a real directory: {directory}")
        return True

    # Path.exists()/Path.is_dir() follow symlinks and only inspect the nearest existing
    # ancestor.  Walk every absolute component so an already-existing output root cannot
    # smuggle a symlinked parent into the immutable publisher.
    for directory in components:
        if require_real_directory(directory):
            continue
        try:
            directory.mkdir(mode=PRIVATE_DIRECTORY_MODE)
        except FileExistsError:
            pass
        except OSError as exc:
            raise GenerationPublishError(
                f"cannot create generation output directory {directory}: {exc}"
            ) from exc
        if not require_real_directory(directory):  # pragma: no cover - mkdir succeeded
            raise GenerationPublishError(f"generation output directory vanished: {directory}")

    # Recheck after creation to catch a component swapped while the path was built.
    for directory in components:
        if not require_real_directory(directory):
            raise GenerationPublishError(f"generation output directory vanished: {directory}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _publisher_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".generation-publish.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _canonical_json_bytes(value: object, *, pretty: bool) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        else:
            text = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
    except (TypeError, ValueError) as exc:
        raise GenerationFormatError(f"artifact is not strict JSON: {exc}") from exc
    return f"{text}\n".encode()


def _write_bytes(path: Path, data: bytes) -> None:
    """Create one fsynced owner-only file; existing paths are never overwritten."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write to {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _canonical_json_bytes(value, pretty=True))


@contextlib.contextmanager
def _private_binary_writer(path: Path) -> Iterator[BinaryIO]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    handle = os.fdopen(descriptor, "wb")
    try:
        os.fchmod(handle.fileno(), PRIVATE_FILE_MODE)
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()


def _write_json_line(handle: BinaryIO, value: object) -> None:
    handle.write(_canonical_json_bytes(value, pretty=False))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_state_sha256(source_state: Mapping[str, object]) -> str:
    """Hash the canonical source-state payload bound into ``manifest.source``."""
    if not isinstance(source_state, Mapping):
        raise GenerationFormatError("source_state must be a JSON object")
    canonical = _canonical_json_bytes(dict(source_state), pretty=False)
    return hashlib.sha256(canonical).hexdigest()


def _as_manifest(
    value: GenerationManifest | Mapping[str, object],
) -> GenerationManifest:
    if isinstance(value, GenerationManifest):
        data = value.to_dict()
        data["covered_runs"] = [dict(run) for run in data["covered_runs"]]
        return GenerationManifest.from_dict(data)
    if isinstance(value, Mapping):
        return GenerationManifest.from_dict(dict(value))
    raise TypeError("manifest must be a GenerationManifest or mapping")


def _as_document(
    value: DocumentRecord | Mapping[str, object], generation_id: str
) -> DocumentRecord:
    if isinstance(value, DocumentRecord):
        data = value.to_dict()
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise GenerationFormatError("document records must be objects")
    return DocumentRecord.from_dict(data, expected_generation_id=generation_id)


def _as_sample(
    value: SampleCheck | Mapping[str, object], generation_id: str
) -> SampleCheck:
    if isinstance(value, SampleCheck):
        data = value.to_dict()
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise GenerationFormatError("sample checks must be objects")
    return SampleCheck.from_dict(data, expected_generation_id=generation_id)


def _provenance(
    manifest: GenerationManifest,
    preparation_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    data = manifest.to_dict()
    provenance: dict[str, object] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": manifest.generation_id,
        "corpus": data["corpus"],
        "source": data["source"],
        "model": data["model"],
        "vector_space": data["vector_space"],
        "chunking": data["chunking"],
        "covered_runs": data["covered_runs"],
        "retrieval_fingerprint_revision": manifest.retrieval_fingerprint_revision,
        "retrieval_fingerprint": manifest.retrieval_fingerprint,
        "code": data["code"],
        "dependency": data["dependency"],
        "creation": data["creation"],
    }
    if preparation_provenance is not None:
        provenance["preparation"] = dict(preparation_provenance)
    return provenance


def _source_state_artifact(
    manifest: GenerationManifest, source_state: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": manifest.generation_id,
        "source": manifest.source.name,
        "state_sha256": manifest.source.state_sha256,
        "state": dict(source_state),
    }


def _quarantine_record(document: DocumentRecord) -> dict[str, object]:
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": document.generation_id,
        "source": document.source,
        "document_id": document.document_id,
        "version_id": document.version_id,
        "source_identity": document.source_identity,
        "content_hash": document.content_hash,
        "document_state_hash": document.document_state_hash,
        "expected_chunk_count": document.expected_chunk_count,
        "content_kind": document.content_kind,
        "extraction_status": document.extraction_status,
        "article_summary": document.article_summary,
        "exclusion_reason": document.exclusion_reason,
        "content_complete": document.content_complete,
        "refresh_deadline": document.refresh_deadline,
        "source_binary_url": document.source_binary_url,
    }


def _open_validation_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    os.chmod(path, PRIVATE_FILE_MODE)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE documents ("
        "source TEXT NOT NULL, document_id TEXT NOT NULL, version_id TEXT NOT NULL, "
        "expected_chunks INTEGER NOT NULL, indexed INTEGER NOT NULL, "
        "PRIMARY KEY (source, document_id, version_id))"
    )
    return connection


def _stream_documents(
    staging: Path,
    manifest: GenerationManifest,
    records: Iterable[DocumentRecord | Mapping[str, object]],
    index: sqlite3.Connection,
) -> tuple[int, int, int, int]:
    document_count = indexed_count = excluded_count = chunk_count = 0
    previous_key: tuple[str, str, str] | None = None
    document_path = staging / DOCUMENTS_FILENAME
    quarantine_path = staging / QUARANTINE_FILENAME
    with (
        _private_binary_writer(document_path) as document_file,
        _private_binary_writer(quarantine_path) as quarantine_file,
    ):
        for raw_record in records:
            record = _as_document(raw_record, manifest.generation_id)
            key = (record.source, record.document_id, record.version_id)
            if previous_key is not None and key <= previous_key:
                raise GenerationFormatError(
                    "documents must be strictly sorted by source, document_id, and version_id"
                )
            previous_key = key
            if record.indexed and not record.content_complete:
                raise GenerationFormatError(
                    "incomplete content must be explicitly excluded from indexing"
                )
            try:
                index.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (
                        record.source,
                        record.document_id,
                        record.version_id,
                        record.expected_chunk_count,
                        int(record.indexed),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise GenerationFormatError(
                    f"duplicate document identity: {key}"
                ) from exc
            _write_json_line(document_file, record.to_dict())
            document_count += 1
            if record.indexed:
                indexed_count += 1
                chunk_count += record.expected_chunk_count
            else:
                excluded_count += 1
                _write_json_line(quarantine_file, _quarantine_record(record))
    index.commit()
    return document_count, indexed_count, excluded_count, chunk_count


def _stream_samples(
    staging: Path,
    manifest: GenerationManifest,
    records: Iterable[SampleCheck | Mapping[str, object]],
    index: sqlite3.Connection,
) -> int:
    count = 0
    previous_key: tuple[str, str, str, int] | None = None
    path = staging / SAMPLE_CHECKS_FILENAME
    with _private_binary_writer(path) as sample_file:
        for raw_record in records:
            record = _as_sample(raw_record, manifest.generation_id)
            key = (
                record.source,
                record.document_id,
                record.version_id,
                record.chunk_index,
            )
            if previous_key is not None and key <= previous_key:
                raise GenerationFormatError(
                    "sample checks must be strictly sorted by source, document_id, "
                    "version_id, and chunk_index"
                )
            previous_key = key
            observed = index.execute(
                "SELECT expected_chunks, indexed FROM documents "
                "WHERE source = ? AND document_id = ? AND version_id = ?",
                (record.source, record.document_id, record.version_id),
            ).fetchone()
            if observed is None:
                raise GenerationFormatError(
                    f"sample references unknown document version: {key[:3]}"
                )
            expected_chunks, indexed = observed
            if not indexed or record.chunk_index >= expected_chunks:
                raise GenerationFormatError(
                    f"sample references non-indexed chunk: {key}"
                )
            _write_json_line(sample_file, record.to_dict())
            count += 1
    return count


def _validate_observed_counts(
    manifest: GenerationManifest,
    *,
    document_count: int,
    indexed_count: int,
    excluded_count: int,
    chunk_count: int,
    sample_count: int,
) -> None:
    observed = {
        "document_count": document_count,
        "indexed_document_count": indexed_count,
        "excluded_document_count": excluded_count,
        "chunk_count": chunk_count,
        "sample_count": sample_count,
    }
    expected = {name: getattr(manifest, name) for name in observed}
    if observed != expected:
        raise GenerationFormatError(
            f"manifest counts do not match streamed artifacts: expected={expected}, observed={observed}"
        )


def _checksum_inventory(staging: Path, manifest_bytes: bytes) -> ChecksumInventory:
    files = {
        DOCUMENTS_FILENAME: _sha256_file(staging / DOCUMENTS_FILENAME),
        SAMPLE_CHECKS_FILENAME: _sha256_file(staging / SAMPLE_CHECKS_FILENAME),
        PROVENANCE_FILENAME: _sha256_file(staging / PROVENANCE_FILENAME),
        SOURCE_STATE_FILENAME: _sha256_file(staging / SOURCE_STATE_FILENAME),
        QUARANTINE_FILENAME: _sha256_file(staging / QUARANTINE_FILENAME),
        MANIFEST_FILENAME: hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return ChecksumInventory.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "algorithm": CHECKSUM_ALGORITHM,
            "files": dict(sorted(files.items())),
        }
    )


def _validate_private_tree(staging: Path) -> None:
    if stat.S_IMODE(staging.stat().st_mode) != PRIVATE_DIRECTORY_MODE:
        raise GenerationPublishError(
            f"staging directory is not owner-only: {staging}", staging_path=staging
        )
    for path in staging.iterdir():
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != PRIVATE_FILE_MODE:
            raise GenerationPublishError(
                f"generation artifact is not a private regular file: {path}",
                staging_path=staging,
            )


def publish_generation(
    output_root: str | Path,
    generation_id: str,
    manifest: GenerationManifest | Mapping[str, object],
    documents: Iterable[DocumentRecord | Mapping[str, object]],
    sample_checks: Iterable[SampleCheck | Mapping[str, object]],
    source_state: Mapping[str, object],
    *,
    preparation_provenance: Mapping[str, object] | None = None,
) -> Path:
    """Build and atomically publish one immutable generation.

    ``generation_id`` is explicit and must match the manifest. Existing destinations are
    never reused or overwritten. Any failure after staging begins leaves the owner-only
    staging directory in place and exposes its path on ``GenerationPublishError``.
    """
    canonical_generation_id = validate_generation_id(generation_id)
    parsed_manifest = _as_manifest(manifest)
    if parsed_manifest.generation_id != canonical_generation_id:
        raise GenerationFormatError(
            "explicit generation_id does not match manifest.generation_id"
        )
    if not isinstance(source_state, Mapping):
        raise GenerationFormatError("source_state must be a JSON object")
    if source_state_sha256(source_state) != parsed_manifest.source.state_sha256:
        raise GenerationFormatError(
            "source_state does not match manifest.source.state_sha256"
        )
    if preparation_provenance is not None and not isinstance(
        preparation_provenance, Mapping
    ):
        raise GenerationFormatError("preparation_provenance must be a JSON object")

    root = Path(output_root)
    if root.name.lower() == "v1":
        raise GenerationPublishError(
            "snapshots/v1 is frozen and cannot be used as a generation output root"
        )
    _create_private_directory(root)
    root = root.resolve()
    target = root / canonical_generation_id

    with _publisher_lock(root):
        if _lexists(target):
            raise GenerationDestinationExists(
                f"generation destination already exists: {target}"
            )
        staging = root / (f".{canonical_generation_id}.staging-{uuid.uuid4().hex}")
        staging.mkdir(mode=PRIVATE_DIRECTORY_MODE)
        _fsync_directory(root)

        validation_path = staging / ".validation.sqlite"
        validation_index: sqlite3.Connection | None = None
        try:
            validation_index = _open_validation_index(validation_path)
            counts = _stream_documents(
                staging,
                parsed_manifest,
                documents,
                validation_index,
            )
            sample_count = _stream_samples(
                staging,
                parsed_manifest,
                sample_checks,
                validation_index,
            )
            _validate_observed_counts(
                parsed_manifest,
                document_count=counts[0],
                indexed_count=counts[1],
                excluded_count=counts[2],
                chunk_count=counts[3],
                sample_count=sample_count,
            )
            validation_index.close()
            validation_index = None
            validation_path.unlink()

            _write_json(
                staging / PROVENANCE_FILENAME,
                _provenance(parsed_manifest, preparation_provenance),
            )
            _write_json(
                staging / SOURCE_STATE_FILENAME,
                _source_state_artifact(parsed_manifest, source_state),
            )
            manifest_bytes = _canonical_json_bytes(
                parsed_manifest.to_dict(), pretty=True
            )
            inventory = _checksum_inventory(staging, manifest_bytes)
            _write_json(staging / CHECKSUM_FILENAME, inventory.to_dict())

            # Publication marker: the manifest is always the final staged artifact.
            _write_bytes(staging / MANIFEST_FILENAME, manifest_bytes)
            _fsync_directory(staging)

            loaded = load_generation(staging)
            if loaded.manifest != parsed_manifest:
                raise GenerationPublishError(
                    "published manifest changed during validation", staging_path=staging
                )
            if (
                sum(1 for _ in loaded.iter_documents())
                != parsed_manifest.document_count
            ):
                raise GenerationPublishError(
                    "document ledger changed during validation", staging_path=staging
                )
            if sum(1 for _ in loaded.iter_samples()) != parsed_manifest.sample_count:
                raise GenerationPublishError(
                    "sample ledger changed during validation", staging_path=staging
                )
            _validate_private_tree(staging)
            _fsync_directory(staging)

            if _lexists(target):
                raise GenerationDestinationExists(
                    f"generation destination appeared during build: {target}",
                    staging_path=staging,
                )
            _rename_noreplace(staging, target)
            _fsync_directory(root)
            return target
        except Exception as exc:
            if validation_index is not None:
                validation_index.close()
            if isinstance(exc, GenerationPublishError):
                if exc.staging_path is None:
                    exc.staging_path = staging
                raise
            raise GenerationPublishError(
                f"generation build failed; staging retained at {staging}: {exc}",
                staging_path=staging,
            ) from exc


__all__ = [
    "GenerationDestinationExists",
    "GenerationPublishError",
    "PROVENANCE_FILENAME",
    "QUARANTINE_FILENAME",
    "SOURCE_STATE_FILENAME",
    "publish_generation",
    "source_state_sha256",
]
