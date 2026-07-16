"""Fail-closed completion attestations for legal scraper runs.

This module deliberately contains no Scrapy signal wiring.  It is the small shared
contract used by spider startup, the completion extension, the combined runner, and
offline consumers.  Files named by an attestation are opened without following
symlinks, must be owner-private regular files, and are fsynced before their bytes are
claimed as durable.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlparse

COMPLETION_SCHEMA_VERSION = 1
SUPREMECOURT_FINISH_REASON = "closespider_timeout"
# Terminal publication is a tiny write-ahead log.  The full canonical terminal
# bytes are first made durable under the candidate name.  A second durable,
# create-only authorization binds those bytes to the source and run.  Only then may
# ``run.json`` be materialized.  Both write-ahead files are permanent: consumers
# require all three identities to agree, and startup can reconstruct ``run.json``
# from them after an interrupted/ambiguous rename.
TERMINAL_CANDIDATE_FILENAME = ".completion-terminal-candidate.json"
FINALIZATION_CLAIM_FILENAME = ".completion-finalized"
TERMINAL_RECOVERY_GUARD_FILENAME = ".completion-recovery-required"
_MAX_RECORD_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECOND_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")

_TERMINAL_AUTHORIZATION_KEYS = {
    "schema_version",
    "state",
    "source",
    "run_id",
    "candidate_filename",
    "terminal_size_bytes",
    "terminal_sha256",
}

_STARTUP_KEYS = {
    "schema_version",
    "run_id",
    "source",
    "spider",
    "start_date",
    "end_date",
    "started_at",
    "items_path",
    "latest_items_path",
    "log_path",
    "outcome",
    "quality_passed",
    "feeds_durable",
    "failure_count",
}
_TERMINAL_KEYS = _STARTUP_KEYS | {
    "finish_reason",
    "completed_at",
    "feed_outputs",
    "quality",
    "source_validation",
}
_FEED_OUTPUT_KEYS = {
    "durable",
    "configured_count",
    "success_count",
    "failure_count",
    "files",
}
_FEED_FILE_KEYS = {"role", "configured_uri", "path", "size_bytes", "sha256"}
_QUALITY_KEYS = {
    "passed",
    "quality_failures",
    "spider_errors",
    "spider_exceptions",
    "item_errors",
    "pagination_reconcilers",
    "pagination_reconciled",
}
_SUPREME_VALIDATION_KEYS = {
    "kind",
    "passed",
    "validator_schema_version",
    "run_dir",
    "run_id",
    "finish_reason",
    "items_sha256",
    "manifest_path",
    "manifest_sha256",
    "journal_path",
    "journal_sha256",
    "unresolved_failure_count",
}


class CompletionError(RuntimeError):
    """Completion evidence cannot be produced or trusted."""


class UnsafePathError(CompletionError):
    """A path failed a no-follow, file-type, ownership, or mode check."""


class CompletionAlreadyFinalized(CompletionError):
    """A run already has a terminal record or an exclusive finalization claim."""


class CompletionMaterializationPending(CompletionError):
    """A durable terminal decision exists but ``run.json`` needs recovery.

    Callers must not reinterpret this as a failed crawl or delete inactive staged
    dedup rows.  A later caller holding the source lock can reconstruct the
    authoritative path with :func:`recover_authorized_terminal`.
    """


@dataclass(frozen=True, slots=True)
class FileProof:
    """Durable byte identity of one local owner-private regular file."""

    path: Path
    size_bytes: int
    sha256: str

    def as_feed_row(self, *, role: str, configured_uri: str) -> dict[str, object]:
        return {
            "role": role,
            "configured_uri": configured_uri,
            "path": os.fspath(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FeedDurability:
    """Exact feed exporter counts and the durable local outputs they bind."""

    configured_count: int
    success_count: int
    failure_count: int
    files: tuple[dict[str, object], ...]

    @property
    def durable(self) -> bool:
        return (
            self.configured_count == 2
            and self.success_count == self.configured_count
            and self.failure_count == 0
            and len(self.files) == self.configured_count
        )

    @property
    def record(self) -> dict[str, object]:
        return {
            "durable": self.durable,
            "configured_count": self.configured_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "files": [dict(row) for row in self.files],
        }


@dataclass(frozen=True, slots=True)
class QualityEvaluation:
    """Pure, bounded crawl-quality result used by both runner and attester."""

    passed: bool
    quality_failures: int
    spider_errors: int
    spider_exceptions: int
    item_errors: int
    pagination_reconcilers: int
    pagination_reconciled: int
    issues: tuple[str, ...]

    @property
    def record(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "quality_failures": self.quality_failures,
            "spider_errors": self.spider_errors,
            "spider_exceptions": self.spider_exceptions,
            "item_errors": self.item_errors,
            "pagination_reconcilers": self.pagination_reconcilers,
            "pagination_reconciled": self.pagination_reconciled,
        }


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """Result of authoritative run publication and guarded latest copying."""

    run_path: Path
    latest_path: Path
    latest_updated: bool


@dataclass(frozen=True, slots=True)
class AuthorizedTerminal:
    """Permanent write-ahead evidence for one terminal publication decision."""

    record: dict[str, object]
    payload: bytes
    candidate: FileProof
    authorization: FileProof


def canonical_json_bytes(value: object) -> bytes:
    """Return the schema's deterministic UTF-8 JSON representation."""

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


def canonical_utc_format(value: datetime) -> str:
    """Format an aware datetime as canonical second-precision UTC."""

    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("completion timestamps must be timezone-aware datetimes")
    normalized = value.astimezone(UTC).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_utc_now() -> str:
    """Return the current canonical second-precision UTC timestamp."""

    return canonical_utc_format(datetime.now(UTC))


# Compatibility-friendly spelling used in some call sites.
canonical_utc_second = canonical_utc_format


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: str | os.PathLike[str]) -> Path:
    absolute = _absolute_lexical(path)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafePathError(f"cannot inspect path component {cursor}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise UnsafePathError(f"refusing symlink path component: {cursor}")
    return absolute


def _open_directory_nofollow(path: str | os.PathLike[str]) -> int:
    absolute = _absolute_lexical(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _ensure_real_directory(path: str | os.PathLike[str]) -> Path:
    """Create missing directories as 0700 without changing existing modes."""

    absolute = _reject_symlink_components(path)
    missing: list[Path] = []
    cursor = absolute
    while not os.path.lexists(cursor):
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if os.path.lexists(cursor):
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise UnsafePathError(f"directory ancestor is unsafe: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise UnsafePathError(f"created directory is unsafe: {directory}")
        mode = stat.S_IMODE(info.st_mode)
        if mode != 0o700 or info.st_uid != os.geteuid():
            raise UnsafePathError(
                f"created directory must be owned by this user with mode 0700: "
                f"{directory} (uid={info.st_uid}, mode={mode:04o})"
            )
        # Persist each newly-created directory entry before it becomes a parent for
        # another entry or a completion file.
        fsync_directory(directory.parent)
    _reject_symlink_components(absolute)
    descriptor = _open_directory_nofollow(absolute)
    os.close(descriptor)
    return absolute


def fsync_directory(path: str | os.PathLike[str]) -> None:
    """Fsync a real directory opened without following its final component."""

    absolute = _reject_symlink_components(path)
    descriptor = _open_directory_nofollow(absolute)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing completion evidence")
        view = view[written:]


def _private_temp(parent_fd: int, destination_name: str, payload: bytes) -> str:
    name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        # The caller cannot clean a name it never received.  Preserve the original
        # write/fsync exception even if best-effort cleanup itself encounters damage.
        try:
            os.unlink(name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    return name


def atomic_replace_private(path: str | os.PathLike[str], payload: bytes) -> None:
    """Atomically replace a regular destination from a private fsynced temp file."""

    destination = _reject_symlink_components(path)
    parent = _ensure_real_directory(destination.parent)
    parent_fd = _open_directory_nofollow(parent)
    temporary: str | None = None
    try:
        try:
            existing = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise UnsafePathError(
                f"replacement destination is not a regular non-symlink file: {destination}"
            )
        temporary = _private_temp(parent_fd, destination.name, payload)
        os.replace(
            temporary,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary = None
        os.fsync(parent_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _read_private_leaf(parent_fd: int, name: str, *, max_bytes: int) -> bytes:
    """Read one stable private leaf without fsyncing during recovery."""

    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or mode != 0o600
            or before.st_uid != os.geteuid()
        ):
            raise UnsafePathError(
                f"terminal destination must be owned by this user with mode 0600: "
                f"{name} (uid={before.st_uid}, mode={mode:04o})"
            )
        if before.st_size > max_bytes:
            raise CompletionError(
                f"terminal destination exceeds the {max_bytes}-byte limit"
            )
        payload = bytearray()
        while True:
            block = os.read(descriptor, 128 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > max_bytes:
                raise CompletionError(
                    f"terminal destination grew beyond the {max_bytes}-byte limit"
                )
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or after.st_size != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise CompletionError("terminal destination changed while reading")
        leaf = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(leaf.st_mode)
            or leaf.st_dev != before.st_dev
            or leaf.st_ino != before.st_ino
        ):
            raise CompletionError("terminal destination identity changed while reading")
        return bytes(payload)
    finally:
        os.close(descriptor)


def atomic_replace_terminal_private(
    path: str | os.PathLike[str],
    payload: bytes,
    startup_payload: bytes,
) -> None:
    """Materialize already-authorized terminal bytes at the authoritative path.

    ``startup_payload`` is the already-durable nonqualifying record.  A private
    same-directory hard link keeps that exact inode available until the terminal
    rename has been fsynced.  If the post-rename directory fsync fails, the hard link
    is atomically renamed back over the terminal record before the original error is
    re-raised.  A permanent candidate and authorization must have been made durable
    before calling this primitive; they, not an absence-based guard, define the
    irrevocable terminal decision and permit later recovery.
    """

    if len(payload) > _MAX_RECORD_BYTES or len(startup_payload) > _MAX_RECORD_BYTES:
        raise CompletionError("terminal/startup payload exceeds the fixed size bound")
    destination = _reject_symlink_components(path)
    parent = _ensure_real_directory(destination.parent)
    parent_fd = _open_directory_nofollow(parent)
    temporary: str | None = None
    backup: str | None = None
    cleanup_changed = False
    preserve_backup = False
    try:
        current = _read_private_leaf(
            parent_fd,
            destination.name,
            max_bytes=_MAX_RECORD_BYTES,
        )
        if current != startup_payload:
            raise CompletionError(
                "authoritative run metadata changed before terminal replacement"
            )

        temporary = _private_temp(parent_fd, destination.name, payload)
        backup = (
            f".{destination.name}.{secrets.token_hex(16)}.startup-backup"
        )
        os.link(
            destination.name,
            backup,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        cleanup_changed = True
        # Make the recovery inode durable before the startup name is replaced.
        os.fsync(parent_fd)

        os.replace(
            temporary,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary = None
        try:
            os.fsync(parent_fd)
        except BaseException as commit_error:
            restored = False
            try:
                os.replace(
                    backup,
                    destination.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                backup = None
                restored = True
            except BaseException:
                # The durable hard-link restore should be sufficient.  Retain a
                # private-write fallback so an unexpected rename failure still has
                # a chance to return the visible path to nonqualifying startup bytes.
                recovery: str | None = None
                try:
                    try:
                        recovery = _private_temp(
                            parent_fd,
                            destination.name,
                            startup_payload,
                        )
                        os.replace(
                            recovery,
                            destination.name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                        )
                        recovery = None
                        restored = True
                    except BaseException:
                        restored = False
                finally:
                    if recovery is not None:
                        try:
                            os.unlink(recovery, dir_fd=parent_fd)
                        except OSError:
                            pass
            try:
                os.fsync(parent_fd)
            except BaseException:
                # The live destination has still been restored.  Preserve the
                # original commit failure, which is the actionable root cause.
                pass
            try:
                restored = restored and (
                    _read_private_leaf(
                        parent_fd,
                        destination.name,
                        max_bytes=_MAX_RECORD_BYTES,
                    )
                    == startup_payload
                )
            except BaseException:
                restored = False
            if not restored:
                # The durable authorization was the commit decision.  If the
                # terminal rename is already the exact authorized bytes, reporting
                # a failure would create a split-brain outcome (strict consumers
                # accept while the publisher says it failed).  Treat this ambiguous
                # post-rename fsync as logical success; a crash can reconstruct the
                # name from the permanent candidate.
                try:
                    terminal_visible = (
                        _read_private_leaf(
                            parent_fd,
                            destination.name,
                            max_bytes=_MAX_RECORD_BYTES,
                        )
                        == payload
                    )
                except BaseException:
                    terminal_visible = False
                if terminal_visible:
                    return
                # Keep the exact durable startup inode for operator diagnostics.
                # The permanent authorized candidate remains the recovery source.
                preserve_backup = True
                raise CompletionError(
                    "terminal publication failed and startup restoration could not "
                    "be verified"
                ) from commit_error
            raise

        # The authoritative terminal rename is now durable.  Backup cleanup cannot
        # retroactively make that publication fail, so it is best effort.
        try:
            os.unlink(backup, dir_fd=parent_fd)
        except OSError:
            pass
        else:
            backup = None
            cleanup_changed = True
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    finally:
        leftovers = (temporary,) if preserve_backup else (temporary, backup)
        for leftover in leftovers:
            if leftover is None:
                continue
            try:
                os.unlink(leftover, dir_fd=parent_fd)
                cleanup_changed = True
            except OSError:
                pass
        if cleanup_changed:
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        # The namespace outcome is governed by the permanent authorization.
        # Descriptor cleanup must not reverse it into an apparent publication error.
        try:
            os.close(parent_fd)
        except BaseException:
            pass


def atomic_create_private(path: str | os.PathLike[str], payload: bytes) -> None:
    """Atomically create a private file and refuse every existing destination."""

    destination = _reject_symlink_components(path)
    parent = _ensure_real_directory(destination.parent)
    parent_fd = _open_directory_nofollow(parent)
    temporary: str | None = None
    try:
        temporary = _private_temp(parent_fd, destination.name, payload)
        try:
            os.link(
                temporary,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"completion destination already exists: {destination}") from exc
        os.unlink(temporary, dir_fd=parent_fd)
        temporary = None
        os.fsync(parent_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


@contextmanager
def source_lock(source_root: str | os.PathLike[str]):
    """Serialize startup/finalization operations for one source across processes."""

    root = _ensure_real_directory(source_root)
    root_fd = _open_directory_nofollow(root)
    descriptor: int | None = None
    try:
        root_info = os.fstat(root_fd)
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.geteuid():
            raise UnsafePathError(
                f"source root must be a directory owned by this user: {root}"
            )
        descriptor = os.open(
            ".completion.lock",
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
        ):
            raise UnsafePathError(f"source completion lock is not private: {root}")
        os.fsync(descriptor)
        os.fsync(root_fd)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException:
                pass
            try:
                os.close(descriptor)
            except BaseException:
                pass
        try:
            os.close(root_fd)
        except BaseException:
            pass


def build_startup_record(spider: object) -> dict[str, object]:
    """Build nonqualifying schema-v1 startup metadata from a configured spider."""

    source = str(getattr(spider, "name"))
    run_id = str(getattr(spider, "run_id"))
    started_at = getattr(spider, "started_at")
    if isinstance(started_at, datetime):
        started_at = canonical_utc_format(started_at)
    record: dict[str, object] = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "spider": source,
        "start_date": getattr(spider, "scraping_start_date").isoformat(),
        "end_date": getattr(spider, "scraping_end_date").isoformat(),
        "started_at": started_at,
        "items_path": os.fspath(_absolute_lexical(getattr(spider, "items_path"))),
        "latest_items_path": os.fspath(
            _absolute_lexical(getattr(spider, "latest_items_path"))
        ),
        "log_path": os.fspath(_absolute_lexical(getattr(spider, "log_path"))),
        "outcome": "started",
        "quality_passed": False,
        "feeds_durable": False,
        "failure_count": 0,
    }
    _validate_startup_record(record)
    return record


startup_record = build_startup_record


def publish_startup_metadata(
    run_path: str | os.PathLike[str],
    latest_path: str | os.PathLike[str],
    record: Mapping[str, object],
) -> None:
    """Invalidate ``latest`` first, then create the authoritative startup record."""

    normalized = dict(record)
    _validate_startup_record(normalized)
    items_path = Path(str(normalized["items_path"]))
    latest_items_path = Path(str(normalized["latest_items_path"]))
    if _absolute_lexical(run_path) != items_path.parent / "run.json":
        raise CompletionError("startup publication run path does not match its record")
    if _absolute_lexical(latest_path) != latest_items_path.parent / "run.json":
        raise CompletionError("startup publication latest path does not match its record")
    payload = canonical_json_bytes(normalized)
    if len(payload) > _MAX_RECORD_BYTES:
        raise CompletionError("startup record exceeds the fixed size bound")
    atomic_replace_private(latest_path, payload)
    atomic_create_private(run_path, payload)


def local_feed_path(uri: object) -> Path | None:
    """Return a canonical local feed path; unsupported schemes return ``None``."""

    text = str(uri)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.params or parsed.query or parsed.fragment:
            return None
        return _absolute_lexical(unquote(parsed.path))
    if parsed.scheme:
        return None
    return _absolute_lexical(text)


def hash_and_fsync_private_file(path: str | os.PathLike[str]) -> FileProof:
    """Fsync and hash a stable owner-private regular file, then fsync its parent."""

    absolute = _reject_symlink_components(path)
    parent_fd = _open_directory_nofollow(absolute.parent)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                absolute.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise UnsafePathError(
                f"cannot safely open private file {absolute}: {exc}"
            ) from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise UnsafePathError(f"not a regular feed file: {absolute}")
        mode = stat.S_IMODE(before.st_mode)
        if mode != 0o600 or before.st_uid != os.geteuid():
            raise UnsafePathError(
                f"feed file must be owned by this user with mode 0600; "
                f"uid={before.st_uid}, mode={mode:04o}: {absolute}"
            )
        os.fsync(descriptor)
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        if (
            size != before.st_size
            or after.st_size != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise CompletionError(f"feed file changed while hashing: {absolute}")
        leaf = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if leaf.st_dev != before.st_dev or leaf.st_ino != before.st_ino:
            raise CompletionError(f"feed file identity changed while hashing: {absolute}")
        os.fsync(parent_fd)
        return FileProof(absolute, size, digest.hexdigest())
    except FileNotFoundError as exc:
        raise CompletionError(f"configured feed was not materialized: {absolute}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _sum_stats(stats: Mapping[str, object], prefix: str) -> int:
    total = 0
    for key, value in stats.items():
        if str(key).startswith(prefix):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CompletionError(f"invalid nonnegative counter {key!r}: {value!r}")
            total += value
    return total


def attest_feed_outputs(
    feeds: Mapping[object, object] | Iterable[object],
    stats: Mapping[str, object],
    run_items_path: str | os.PathLike[str],
    latest_items_path: str | os.PathLike[str],
) -> FeedDurability:
    """Verify exact exporter success and durable identical run/latest feed bytes."""

    configured = list(feeds if not isinstance(feeds, Mapping) else feeds.keys())
    success_count = _sum_stats(stats, "feedexport/success_count/")
    failure_count = _sum_stats(stats, "feedexport/failed_count/")
    if len(configured) != 2:
        raise CompletionError(f"exactly two local feeds are required; configured={len(configured)}")
    if success_count != len(configured) or failure_count:
        raise CompletionError(
            "feed exporter counts do not prove success: "
            f"configured={len(configured)} successes={success_count} failures={failure_count}"
        )
    expected = {
        _absolute_lexical(run_items_path): "run",
        _absolute_lexical(latest_items_path): "latest",
    }
    if len(expected) != 2:
        raise CompletionError("run and latest feed paths must be distinct")
    rows: list[dict[str, object]] = []
    proofs: dict[str, FileProof] = {}
    seen_paths: set[Path] = set()
    for uri in configured:
        path = local_feed_path(uri)
        if path is None:
            raise CompletionError(f"unsupported nonlocal feed URI: {uri!s}")
        if path not in expected or path in seen_paths:
            raise CompletionError(f"configured feed does not uniquely match run/latest: {uri!s}")
        seen_paths.add(path)
        proof = hash_and_fsync_private_file(path)
        role = expected[path]
        proofs[role] = proof
        rows.append(proof.as_feed_row(role=role, configured_uri=str(uri)))
    run_proof = proofs.get("run")
    latest_proof = proofs.get("latest")
    if run_proof is None or latest_proof is None or (
        run_proof.size_bytes != latest_proof.size_bytes
        or run_proof.sha256 != latest_proof.sha256
    ):
        raise CompletionError("run and latest feeds do not contain identical bytes")
    rows.sort(key=lambda row: (str(row["role"]), str(row["configured_uri"]), str(row["path"])))
    return FeedDurability(len(configured), success_count, failure_count, tuple(rows))


def attest_materialized_outputs(
    run_items_path: str | os.PathLike[str],
    latest_items_path: str | os.PathLike[str],
) -> FeedDurability:
    """Attest a source's two custom-materialized outputs as logical local feeds."""

    run_path = _absolute_lexical(run_items_path)
    latest_path = _absolute_lexical(latest_items_path)
    if run_path == latest_path:
        raise CompletionError("run and latest materialized outputs must be distinct")
    rows = []
    proofs = {}
    for role, path in (("run", run_path), ("latest", latest_path)):
        proof = hash_and_fsync_private_file(path)
        proofs[role] = proof
        rows.append(
            proof.as_feed_row(role=role, configured_uri=os.fspath(path))
        )
    if (
        proofs["run"].size_bytes != proofs["latest"].size_bytes
        or proofs["run"].sha256 != proofs["latest"].sha256
    ):
        raise CompletionError("run and latest materialized outputs differ")
    rows.sort(key=lambda row: (str(row["role"]), str(row["configured_uri"]), str(row["path"])))
    return FeedDurability(2, 2, 0, tuple(rows))


def failed_feed_durability(
    *,
    configured_count: int = 2,
    success_count: int = 0,
    failure_count: int = 1,
) -> FeedDurability:
    """Create a bounded, explicitly non-durable feed summary for failure records."""

    for label, value in (
        ("configured_count", configured_count),
        ("success_count", success_count),
        ("failure_count", failure_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CompletionError(f"{label} must be a nonnegative integer")
    if failure_count == 0 and success_count == configured_count == 2:
        failure_count = 1
    return FeedDurability(configured_count, success_count, failure_count, ())


def _counter(value: object, *, label: str, issues: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.append(f"{label} is not a nonnegative counter")
        return 0
    return value


def evaluate_crawl_quality(
    stats: Mapping[str, object],
    source: str,
    finish_reason: object,
    *,
    spider_errors: int = 0,
    item_errors: int = 0,
    reconcilers: Mapping[object, object] | Iterable[object] | None = None,
) -> QualityEvaluation:
    """Purely evaluate bounded crawl counters and reconciler terminal state."""

    issues: list[str] = []
    expected_reason = SUPREMECOURT_FINISH_REASON if source == "supremecourt" else "finished"
    if finish_reason != expected_reason:
        issues.append(f"{source}: finish_reason={finish_reason!r}, expected {expected_reason!r}")
    quality_failures = _counter(
        stats.get("quality/failures", 0), label="quality/failures", issues=issues
    )
    observed_spider_errors = _counter(
        spider_errors, label="spider_errors", issues=issues
    )
    observed_item_errors = _counter(item_errors, label="item_errors", issues=issues)
    if "spider_exceptions/count" in stats:
        spider_exceptions = _counter(
            stats["spider_exceptions/count"],
            label="spider_exceptions/count",
            issues=issues,
        )
    else:
        spider_exceptions = 0
        for key, value in stats.items():
            if str(key).startswith("spider_exceptions/"):
                spider_exceptions += _counter(value, label=str(key), issues=issues)

    values: list[object]
    if reconcilers is None:
        values = []
    elif isinstance(reconcilers, Mapping):
        values = list(reconcilers.values())
    else:
        values = list(reconcilers)
    reconciled = 0
    for tracker in values:
        outcome = getattr(tracker, "_finalized", None)
        if outcome is not None:
            reconciled += 1
            if getattr(outcome, "ok", False) is not True:
                issues.append(
                    f"{source}: pagination reconciliation failed for "
                    f"{getattr(tracker, 'scope', '<unknown>')}"
                )
    if reconciled != len(values):
        issues.append(
            f"{source}: finalized {reconciled}/{len(values)} pagination reconciler(s)"
        )
    for label, count in (
        ("completeness-affecting failure(s)", quality_failures),
        ("spider error signal(s)", observed_spider_errors),
        ("callback exception(s)", spider_exceptions),
        ("item-pipeline error(s)", observed_item_errors),
    ):
        if count:
            issues.append(f"{source}: {count} {label}")
    bounded = tuple(issue[:500] for issue in issues[:32])
    return QualityEvaluation(
        passed=not bounded,
        quality_failures=quality_failures,
        spider_errors=observed_spider_errors,
        spider_exceptions=spider_exceptions,
        item_errors=observed_item_errors,
        pagination_reconcilers=len(values),
        pagination_reconciled=reconciled,
        issues=bounded,
    )


# Short alias retained for integration call sites.
evaluate_quality = evaluate_crawl_quality


def generic_source_validation() -> dict[str, object]:
    return {"kind": "generic", "passed": True}


def failed_source_validation(
    source: str | None = None,
    detail: object | None = None,
    *,
    kind: str | None = None,
) -> dict[str, object]:
    """Return the bounded source-validation shape used by terminal failures."""

    del detail  # Failure detail belongs in bounded logs, not the fixed attestation schema.
    selected = kind or (
        "supremecourt_partial_v1" if source == "supremecourt" else "generic"
    )
    return {"kind": str(selected)[:128], "passed": False}


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CompletionError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_bool(
    value: object, *, label: str, expected: bool | None = None
) -> bool:
    if type(value) is not bool:
        raise CompletionError(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise CompletionError(f"{label} must be {str(expected).lower()}")
    return value


def _require_int(
    value: object, *, label: str, expected: int | None = None
) -> int:
    if type(value) is not int or value < 0:
        raise CompletionError(f"{label} must be a nonnegative integer")
    if expected is not None and value != expected:
        raise CompletionError(f"{label} must be {expected}")
    return value


def _require_identity(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise CompletionError(f"{label} is unsafe or malformed")
    if value.lower() == "latest":
        raise CompletionError(f"{label} must identify an exact run")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CompletionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_canonical_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECOND_RE.fullmatch(value) is None:
        raise CompletionError(
            f"{label} must be canonical second-precision UTC (...Z)"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CompletionError(f"{label} is not a valid timestamp") from exc
    if canonical_utc_format(parsed) != value:
        raise CompletionError(f"{label} is not canonical")
    return parsed


def _require_canonical_date(value: object, *, label: str) -> date:
    if not isinstance(value, str):
        raise CompletionError(f"{label} must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CompletionError(f"{label} is not a valid date") from exc
    if parsed.isoformat() != value:
        raise CompletionError(f"{label} must use canonical YYYY-MM-DD")
    return parsed


def _require_canonical_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CompletionError(f"{label} must be a canonical absolute path")
    canonical = _absolute_lexical(value)
    if value != os.fspath(canonical):
        raise CompletionError(f"{label} must be a canonical absolute path")
    return canonical


def _validate_startup_record(record: Mapping[str, object]) -> None:
    _exact_keys(record, _STARTUP_KEYS, label="startup record")
    _require_int(
        record["schema_version"],
        label="startup.schema_version",
        expected=COMPLETION_SCHEMA_VERSION,
    )
    run_id = _require_identity(record["run_id"], label="startup.run_id")
    source = record["source"]
    if not isinstance(source, str) or not source or len(source) > 128:
        raise CompletionError("startup.source must be a bounded non-empty string")
    if record["spider"] != source:
        raise CompletionError("startup source/spider identity mismatch")
    if record["outcome"] != "started":
        raise CompletionError("startup.outcome must be 'started'")
    _require_bool(record["quality_passed"], label="startup.quality_passed", expected=False)
    _require_bool(record["feeds_durable"], label="startup.feeds_durable", expected=False)
    _require_int(record["failure_count"], label="startup.failure_count", expected=0)
    start = _require_canonical_date(record["start_date"], label="startup.start_date")
    end = _require_canonical_date(record["end_date"], label="startup.end_date")
    if start > end:
        raise CompletionError("startup date window is reversed")
    _require_canonical_timestamp(record["started_at"], label="startup.started_at")
    items_path = _require_canonical_path(
        record["items_path"], label="startup.items_path"
    )
    latest_path = _require_canonical_path(
        record["latest_items_path"], label="startup.latest_items_path"
    )
    log_path = _require_canonical_path(record["log_path"], label="startup.log_path")
    run_dir = items_path.parent
    if (
        items_path.name != "items.jsonl"
        or run_dir.name != run_id
        or run_dir.parent.name != "runs"
    ):
        raise CompletionError("startup.items_path does not bind its exact run layout")
    source_root = run_dir.parents[1]
    if source_root.name != source:
        raise CompletionError("startup.items_path does not bind its source")
    if latest_path != source_root / "latest/items.jsonl":
        raise CompletionError("startup.latest_items_path is inconsistent")
    if log_path != run_dir / "spider.log":
        raise CompletionError("startup.log_path is inconsistent")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CompletionError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise CompletionError(f"non-finite JSON number {value!r} in {label}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except CompletionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompletionError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionError(f"{label} must contain a JSON object")
    return value


def _read_private_bytes(
    path: str | os.PathLike[str], *, max_bytes: int | None = None
) -> tuple[bytes, FileProof]:
    """Read stable private bytes through one no-follow descriptor and fsync them."""

    absolute = _reject_symlink_components(path)
    parent_fd = _open_directory_nofollow(absolute.parent)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                absolute.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise UnsafePathError(
                f"cannot safely open private file {absolute}: {exc}"
            ) from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise UnsafePathError(f"not a regular file: {absolute}")
        mode = stat.S_IMODE(before.st_mode)
        if mode != 0o600 or before.st_uid != os.geteuid():
            raise UnsafePathError(
                f"file must be owned by this user with mode 0600; "
                f"uid={before.st_uid}, mode={mode:04o}: {absolute}"
            )
        if max_bytes is not None and before.st_size > max_bytes:
            raise CompletionError(f"file exceeds {max_bytes}-byte limit: {absolute}")
        os.fsync(descriptor)
        payload = bytearray()
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            digest.update(block)
            if max_bytes is not None and len(payload) > max_bytes:
                raise CompletionError(
                    f"file grew beyond {max_bytes}-byte limit: {absolute}"
                )
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or after.st_size != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise CompletionError(f"file changed while reading: {absolute}")
        leaf = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if leaf.st_dev != before.st_dev or leaf.st_ino != before.st_ino:
            raise CompletionError(f"file identity changed while reading: {absolute}")
        os.fsync(parent_fd)
        proof = FileProof(absolute, len(payload), digest.hexdigest())
        return bytes(payload), proof
    except FileNotFoundError as exc:
        raise CompletionError(f"required file is missing: {absolute}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _load_supremecourt_validator() -> ModuleType:
    module_name = "_legal_scraper_completion_supremecourt_validator"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    path = Path(__file__).resolve().parents[2] / "ingest/scripts/validate_supremecourt_partial.py"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("could not construct validator module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CompletionError(
            f"strict Supreme Court validator is unavailable: {path}: {exc}"
        ) from exc
    return module


def validate_supremecourt_source(
    run_dir: str | os.PathLike[str],
    finish_reason: object,
    items_proof: FileProof,
    *,
    validator: object | None = None,
) -> dict[str, object]:
    """Run and cross-check the unchanged strict four-hour Supreme Court validator."""

    run_path = _absolute_lexical(run_dir)
    if finish_reason != SUPREMECOURT_FINISH_REASON:
        raise CompletionError(
            "Supreme Court success requires finish_reason='closespider_timeout'"
        )
    if items_proof.path != run_path / "items.jsonl":
        raise CompletionError("Supreme Court items proof does not bind this run")
    manifest = hash_and_fsync_private_file(run_path / "partial_manifest.json")
    journal = hash_and_fsync_private_file(run_path / "items.journal.jsonl")
    module = validator or _load_supremecourt_validator()
    validate_run = getattr(module, "validate_run", None)
    if not callable(validate_run):
        raise CompletionError("strict Supreme Court validator has no validate_run()")
    try:
        report = validate_run(run_path)
    except Exception as exc:
        raise CompletionError(f"strict Supreme Court validation failed: {exc}") from exc
    if not isinstance(report, dict):
        raise CompletionError("strict Supreme Court validator returned a malformed report")
    expected = {
        "schema_version": 1,
        "valid": True,
        "run_id": run_path.name,
        "run_dir": os.fspath(run_path),
        "finish_reason": SUPREMECOURT_FINISH_REASON,
        "items_sha256": items_proof.sha256,
        "journal_sha256": journal.sha256,
        "unresolved_failure_count": 0,
    }
    for key, expected_value in expected.items():
        observed = report.get(key)
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise CompletionError(f"Supreme Court validator report mismatch for {key}")
    errors = report.get("errors")
    if type(errors) is not list or errors:
        raise CompletionError("Supreme Court validator report contains errors")
    # Detect mutation while the validator was parsing all three files.
    for original in (items_proof, manifest, journal):
        if hash_and_fsync_private_file(original.path) != original:
            raise CompletionError(
                f"Supreme Court validation input changed during validation: {original.path}"
            )
    return {
        "kind": "supremecourt_partial_v1",
        "passed": True,
        "validator_schema_version": 1,
        "run_dir": os.fspath(run_path),
        "run_id": run_path.name,
        "finish_reason": SUPREMECOURT_FINISH_REASON,
        "items_sha256": items_proof.sha256,
        "manifest_path": os.fspath(manifest.path),
        "manifest_sha256": manifest.sha256,
        "journal_path": os.fspath(journal.path),
        "journal_sha256": journal.sha256,
        "unresolved_failure_count": 0,
    }


def build_terminal_record(
    spider: object,
    *,
    finish_reason: str,
    quality: QualityEvaluation,
    feeds: FeedDurability,
    source_validation: Mapping[str, object],
    completed_at: datetime | str | None = None,
    outcome: str = "success",
    failure_count: int | None = None,
) -> dict[str, object]:
    """Build the exact fixed-key schema-v1 terminal record."""

    startup = build_startup_record(spider)
    if isinstance(completed_at, datetime):
        completed_text = canonical_utc_format(completed_at)
    elif completed_at is None:
        completed_text = canonical_utc_now()
    else:
        _require_canonical_timestamp(completed_at, label="completion.completed_at")
        completed_text = completed_at
    if outcome not in {"success", "failure"}:
        raise CompletionError("terminal outcome must be 'success' or 'failure'")
    if failure_count is None:
        observed = (
            quality.quality_failures
            + quality.spider_errors
            + quality.spider_exceptions
            + quality.item_errors
            + feeds.failure_count
        )
        failure_count = 0 if outcome == "success" else max(1, observed)
    _require_int(failure_count, label="completion.failure_count")
    success = outcome == "success"
    if success:
        if not quality.passed or not feeds.durable:
            raise CompletionError("successful terminal record requires quality and feed proof")
        if source_validation.get("passed") is not True:
            raise CompletionError("successful terminal record requires source validation")
        if failure_count:
            raise CompletionError("successful terminal record cannot contain failures")
    else:
        failure_count = max(1, failure_count)
    feed_record = feeds.record
    quality_record = quality.record
    source_record = dict(source_validation)
    if not success:
        feed_record["durable"] = False
        quality_record["passed"] = False
        source_record = {
            "kind": str(source_record.get("kind") or "generic")[:128],
            "passed": False,
        }
    record = dict(startup)
    record.update(
        {
            "outcome": outcome,
            # Every known terminal failure is explicitly nonqualifying even if a
            # subordinate operation happened to finish before another one failed.
            "quality_passed": quality.passed if success else False,
            "feeds_durable": feeds.durable if success else False,
            "failure_count": failure_count,
            "finish_reason": finish_reason,
            "completed_at": completed_text,
            "feed_outputs": feed_record,
            "quality": quality_record,
            "source_validation": source_record,
        }
    )
    _validate_terminal_mapping(record, require_success=success)
    return record


def _validate_quality_record(value: object, *, require_success: bool) -> None:
    if not isinstance(value, dict):
        raise CompletionError("completion.quality must be an object")
    _exact_keys(value, _QUALITY_KEYS, label="completion.quality")
    passed = _require_bool(value["passed"], label="completion.quality.passed")
    counts = {
        key: _require_int(value[key], label=f"completion.quality.{key}")
        for key in _QUALITY_KEYS - {"passed"}
    }
    if counts["pagination_reconciled"] > counts["pagination_reconcilers"]:
        raise CompletionError("completion.quality has too many reconciled scopes")
    if require_success:
        if not passed:
            raise CompletionError("successful completion requires quality.passed=true")
        for key in (
            "quality_failures",
            "spider_errors",
            "spider_exceptions",
            "item_errors",
        ):
            if counts[key] != 0:
                raise CompletionError(f"successful completion requires quality.{key}=0")
        if counts["pagination_reconciled"] != counts["pagination_reconcilers"]:
            raise CompletionError("not all pagination reconcilers reached terminal state")


def _validate_feed_record(
    value: object,
    *,
    require_success: bool,
    expected_run_path: Path | None = None,
    expected_latest_path: Path | None = None,
    expected_items: FileProof | None = None,
) -> None:
    if not isinstance(value, dict):
        raise CompletionError("completion.feed_outputs must be an object")
    _exact_keys(value, _FEED_OUTPUT_KEYS, label="completion.feed_outputs")
    durable = _require_bool(value["durable"], label="completion.feed_outputs.durable")
    configured = _require_int(
        value["configured_count"], label="completion.feed_outputs.configured_count"
    )
    successes = _require_int(
        value["success_count"], label="completion.feed_outputs.success_count"
    )
    failures = _require_int(
        value["failure_count"], label="completion.feed_outputs.failure_count"
    )
    files = value["files"]
    if not isinstance(files, list):
        raise CompletionError("completion.feed_outputs.files must be a list")
    if len(files) > 2:
        raise CompletionError("completion.feed_outputs.files exceeds the fixed bound")
    sort_key = lambda row: (  # noqa: E731 - kept adjacent to canonical check
        str(row.get("role", "")) if isinstance(row, dict) else "",
        str(row.get("configured_uri", "")) if isinstance(row, dict) else "",
        str(row.get("path", "")) if isinstance(row, dict) else "",
    )
    if files != sorted(files, key=sort_key):
        raise CompletionError("completion.feed_outputs.files is not sorted")
    expected_paths = None
    if expected_run_path is not None and expected_latest_path is not None:
        expected_paths = {"run": expected_run_path, "latest": expected_latest_path}
    seen: set[str] = set()
    for index, row in enumerate(files):
        label = f"completion.feed_outputs.files[{index}]"
        if not isinstance(row, dict):
            raise CompletionError(f"{label} must be an object")
        _exact_keys(row, _FEED_FILE_KEYS, label=label)
        role = row["role"]
        if role not in {"run", "latest"} or role in seen:
            raise CompletionError(f"{label}.role must uniquely identify run or latest")
        seen.add(role)
        path = _require_canonical_path(row["path"], label=f"{label}.path")
        if not isinstance(row["configured_uri"], str) or not row["configured_uri"]:
            raise CompletionError(f"{label}.configured_uri must be a non-empty string")
        configured_path = local_feed_path(row["configured_uri"])
        if configured_path is None or configured_path != path:
            raise CompletionError(f"{label}.configured_uri does not bind its path")
        size = _require_int(row["size_bytes"], label=f"{label}.size_bytes")
        digest = _require_sha256(row["sha256"], label=f"{label}.sha256")
        if expected_paths is not None and path != expected_paths[role]:
            raise CompletionError(f"{label}.path does not bind the expected {role} feed")
        if expected_items is not None and (
            size != expected_items.size_bytes or digest != expected_items.sha256
        ):
            raise CompletionError(f"{label} does not match exact run items bytes")
    if require_success:
        if not durable or configured != 2 or successes != 2 or failures != 0:
            raise CompletionError("successful completion lacks exact feed exporter proof")
        if len(files) != 2 or seen != {"run", "latest"}:
            raise CompletionError("successful completion requires run and latest feed rows")


def _validate_source_record(
    value: object, *, source: str, require_success: bool
) -> None:
    if not isinstance(value, dict):
        raise CompletionError("completion.source_validation must be an object")
    if not require_success:
        _exact_keys(value, {"kind", "passed"}, label="completion.source_validation")
        if (
            not isinstance(value["kind"], str)
            or not value["kind"]
            or len(value["kind"]) > 128
        ):
            raise CompletionError("completion.source_validation.kind is invalid")
        _require_bool(
            value["passed"],
            label="completion.source_validation.passed",
            expected=False,
        )
        return
    if source == "supremecourt":
        _exact_keys(value, _SUPREME_VALIDATION_KEYS, label="completion.source_validation")
        if value["kind"] != "supremecourt_partial_v1":
            raise CompletionError("invalid Supreme Court validation kind")
        _require_bool(
            value["passed"],
            label="completion.source_validation.passed",
            expected=True,
        )
        _require_int(
            value["validator_schema_version"],
            label="completion.source_validation.validator_schema_version",
            expected=1,
        )
        for key in ("run_dir", "manifest_path", "journal_path"):
            _require_canonical_path(
                value[key], label=f"completion.source_validation.{key}"
            )
        _require_identity(
            value["run_id"], label="completion.source_validation.run_id"
        )
        for key in ("items_sha256", "manifest_sha256", "journal_sha256"):
            _require_sha256(
                value[key], label=f"completion.source_validation.{key}"
            )
        _require_int(
            value["unresolved_failure_count"],
            label="completion.source_validation.unresolved_failure_count",
            expected=0,
        )
        if value["finish_reason"] != SUPREMECOURT_FINISH_REASON:
            raise CompletionError("Supreme Court source proof has wrong finish reason")
        return
    _exact_keys(value, {"kind", "passed"}, label="completion.source_validation")
    if value["kind"] != "generic":
        raise CompletionError("generic source validation has invalid kind")
    _require_bool(
        value["passed"], label="completion.source_validation.passed", expected=True
    )


def _validate_terminal_mapping(
    record: Mapping[str, object], *, require_success: bool | None = None
) -> None:
    _exact_keys(record, _TERMINAL_KEYS, label="completion record")
    _require_int(
        record["schema_version"],
        label="completion.schema_version",
        expected=COMPLETION_SCHEMA_VERSION,
    )
    _require_identity(record["run_id"], label="completion.run_id")
    source = record["source"]
    if not isinstance(source, str) or not source or len(source) > 128:
        raise CompletionError("completion.source must be a bounded string")
    if record["spider"] != source:
        raise CompletionError("completion source/spider identity mismatch")
    outcome = record["outcome"]
    if outcome not in {"success", "failure"}:
        raise CompletionError("completion outcome is not terminal")
    success = outcome == "success"
    if require_success is not None and success is not require_success:
        raise CompletionError("completion outcome does not match required terminal state")
    quality_passed = _require_bool(
        record["quality_passed"], label="completion.quality_passed"
    )
    feeds_durable = _require_bool(
        record["feeds_durable"], label="completion.feeds_durable"
    )
    failures = _require_int(record["failure_count"], label="completion.failure_count")
    if success:
        if not quality_passed or not feeds_durable or failures != 0:
            raise CompletionError("successful completion has inconsistent proof booleans")
    elif quality_passed or feeds_durable or failures < 1:
        raise CompletionError("terminal failure must be explicitly nonqualifying")
    start = _require_canonical_date(record["start_date"], label="completion.start_date")
    end = _require_canonical_date(record["end_date"], label="completion.end_date")
    if start > end:
        raise CompletionError("completion date window is reversed")
    started = _require_canonical_timestamp(
        record["started_at"], label="completion.started_at"
    )
    completed = _require_canonical_timestamp(
        record["completed_at"], label="completion.completed_at"
    )
    if completed < started:
        raise CompletionError("completion.completed_at precedes completion.started_at")
    if (
        not isinstance(record["finish_reason"], str)
        or not record["finish_reason"]
        or len(record["finish_reason"]) > 128
    ):
        raise CompletionError(
            "completion.finish_reason must be a bounded non-empty string"
        )
    for key in ("items_path", "latest_items_path", "log_path"):
        _require_canonical_path(record[key], label=f"completion.{key}")
    expected_reason = (
        SUPREMECOURT_FINISH_REASON if source == "supremecourt" else "finished"
    )
    if success and record["finish_reason"] != expected_reason:
        raise CompletionError(
            f"successful {source} completion requires finish_reason={expected_reason!r}"
        )
    _validate_feed_record(record["feed_outputs"], require_success=success)
    _validate_quality_record(record["quality"], require_success=success)
    _validate_source_record(
        record["source_validation"], source=source, require_success=success
    )


def _validate_success_binding(
    record: Mapping[str, object],
    *,
    record_path: Path,
    expected_source: str | None,
    expected_run_id: str | None,
    expected_items_path: str | os.PathLike[str] | None,
    validator: object | None,
    now: datetime | None,
) -> FileProof:
    """Prevalidate every fallible semantic/file binding for a success record."""

    _validate_terminal_mapping(record, require_success=True)
    source = str(record["source"])
    run_id = str(record["run_id"])
    if expected_source is not None and source != expected_source:
        raise CompletionError("completion source identity mismatch")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CompletionError("completion run identity mismatch")
    if record_path.name != "run.json" or record_path.parent.name != run_id:
        raise CompletionError("completion record is not the exact run-scoped run.json")
    _reject_terminal_recovery_guard(record_path.parent)
    items_path = (
        _absolute_lexical(expected_items_path)
        if expected_items_path is not None
        else record_path.parent / "items.jsonl"
    )
    if _require_canonical_path(
        record["items_path"], label="completion.items_path"
    ) != items_path:
        raise CompletionError("completion.items_path does not bind selected items")
    if items_path.parent != record_path.parent:
        raise CompletionError("completion record and selected items are from different runs")
    if run_id != items_path.parent.name:
        raise CompletionError("completion items path has wrong run identity")
    source_root = items_path.parent.parents[1]
    if source_root.name != source:
        raise CompletionError("completion items path has wrong source identity")
    latest_path = source_root / "latest/items.jsonl"
    log_path = items_path.parent / "spider.log"
    if _require_canonical_path(
        record["latest_items_path"], label="completion.latest_items_path"
    ) != latest_path:
        raise CompletionError("completion.latest_items_path is inconsistent")
    if _require_canonical_path(
        record["log_path"], label="completion.log_path"
    ) != log_path:
        raise CompletionError("completion.log_path is inconsistent")
    completed = _require_canonical_timestamp(
        record["completed_at"], label="completion.completed_at"
    )
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if completed > current:
        raise CompletionError("completion timestamp is in the future")
    items_proof = hash_and_fsync_private_file(items_path)
    _validate_feed_record(
        record["feed_outputs"],
        require_success=True,
        expected_run_path=items_path,
        expected_latest_path=latest_path,
        expected_items=items_proof,
    )
    source_validation = record["source_validation"]
    if source == "supremecourt":
        expected_validation = validate_supremecourt_source(
            items_path.parent,
            record["finish_reason"],
            items_proof,
            validator=validator,
        )
        if source_validation != expected_validation:
            raise CompletionError("Supreme Court completion source proof has drifted")
    # Rehash both authoritative inputs after all semantic/source validation.
    if hash_and_fsync_private_file(items_path) != items_proof:
        raise CompletionError("items.jsonl changed while verifying completion")
    _reject_terminal_recovery_guard(record_path.parent)
    return items_proof


def _reject_terminal_recovery_guard(run_dir: Path) -> None:
    """Reject a safely inspected publication-recovery guard of any kind."""

    directory_fd = _open_directory_nofollow(_reject_symlink_components(run_dir))
    try:
        try:
            info = os.stat(
                TERMINAL_RECOVERY_GUARD_FILENAME,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        mode = stat.S_IMODE(info.st_mode)
        if (
            not stat.S_ISREG(info.st_mode)
            or mode != 0o600
            or info.st_uid != os.geteuid()
        ):
            raise UnsafePathError(
                "terminal recovery guard is not a private regular file"
            )
        raise CompletionError(
            "terminal recovery guard is present; publication is ineligible"
        )
    finally:
        os.close(directory_fd)


def _terminal_authorization_payload(
    *, source: str, run_id: str, candidate: FileProof
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "state": "terminal_authorized",
            "source": source,
            "run_id": run_id,
            "candidate_filename": TERMINAL_CANDIDATE_FILENAME,
            "terminal_size_bytes": candidate.size_bytes,
            "terminal_sha256": candidate.sha256,
        }
    )


def _load_authorized_terminal(
    run_dir: str | os.PathLike[str],
    *,
    expected_source: str | None = None,
    expected_run_id: str | None = None,
) -> AuthorizedTerminal:
    """Load the permanent candidate and exact durable authorization binding."""

    directory = _reject_symlink_components(run_dir)
    authorization_payload, authorization_proof = _read_private_bytes(
        directory / FINALIZATION_CLAIM_FILENAME,
        max_bytes=_MAX_RECORD_BYTES,
    )
    authorization = _strict_json_object(
        authorization_payload,
        label=f"terminal authorization {directory / FINALIZATION_CLAIM_FILENAME}",
    )
    if authorization_payload != canonical_json_bytes(authorization):
        raise CompletionError("terminal authorization is not canonical JSON")
    _exact_keys(
        authorization,
        _TERMINAL_AUTHORIZATION_KEYS,
        label="terminal authorization",
    )
    _require_int(
        authorization["schema_version"],
        label="terminal authorization.schema_version",
        expected=COMPLETION_SCHEMA_VERSION,
    )
    if authorization["state"] != "terminal_authorized":
        raise CompletionError("terminal authorization has the wrong state")
    source = authorization["source"]
    if not isinstance(source, str) or not source or len(source) > 128:
        raise CompletionError("terminal authorization.source is invalid")
    run_id = _require_identity(
        authorization["run_id"], label="terminal authorization.run_id"
    )
    if expected_source is not None and source != expected_source:
        raise CompletionError("terminal authorization source identity mismatch")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CompletionError("terminal authorization run identity mismatch")
    if directory.name != run_id or directory.parent.name != "runs":
        raise CompletionError("terminal authorization does not bind an exact run directory")
    if directory.parents[1].name != source:
        raise CompletionError("terminal authorization does not bind its source directory")
    if authorization["candidate_filename"] != TERMINAL_CANDIDATE_FILENAME:
        raise CompletionError("terminal authorization names an unexpected candidate")
    authorized_size = _require_int(
        authorization["terminal_size_bytes"],
        label="terminal authorization.terminal_size_bytes",
    )
    if authorized_size > _MAX_RECORD_BYTES:
        raise CompletionError("authorized terminal candidate exceeds the fixed size bound")
    authorized_sha = _require_sha256(
        authorization["terminal_sha256"],
        label="terminal authorization.terminal_sha256",
    )

    candidate_payload, candidate_proof = _read_private_bytes(
        directory / TERMINAL_CANDIDATE_FILENAME,
        max_bytes=_MAX_RECORD_BYTES,
    )
    if (
        candidate_proof.size_bytes != authorized_size
        or candidate_proof.sha256 != authorized_sha
    ):
        raise CompletionError("terminal candidate does not match its authorization")
    candidate_record = _strict_json_object(
        candidate_payload,
        label=f"terminal candidate {candidate_proof.path}",
    )
    if candidate_payload != canonical_json_bytes(candidate_record):
        raise CompletionError("terminal candidate is not canonical schema-v1 JSON")
    _validate_terminal_mapping(candidate_record)
    if candidate_record["source"] != source or candidate_record["run_id"] != run_id:
        raise CompletionError("terminal candidate identity does not match authorization")
    items_path = _require_canonical_path(
        candidate_record["items_path"], label="terminal candidate.items_path"
    )
    if items_path != directory / "items.jsonl":
        raise CompletionError("terminal candidate does not bind its run directory")
    source_root = directory.parents[1]
    if _require_canonical_path(
        candidate_record["latest_items_path"],
        label="terminal candidate.latest_items_path",
    ) != source_root / "latest/items.jsonl":
        raise CompletionError("terminal candidate latest items path is inconsistent")
    if _require_canonical_path(
        candidate_record["log_path"], label="terminal candidate.log_path"
    ) != directory / "spider.log":
        raise CompletionError("terminal candidate log path is inconsistent")
    return AuthorizedTerminal(
        record=candidate_record,
        payload=candidate_payload,
        candidate=candidate_proof,
        authorization=authorization_proof,
    )


def _private_leaf_exists(directory: Path, name: str) -> bool:
    descriptor = _open_directory_nofollow(_reject_symlink_components(directory))
    try:
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError(f"completion write-ahead leaf is not regular: {directory / name}")
        if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
            raise UnsafePathError(
                f"completion write-ahead leaf is not private: {directory / name}"
            )
        return True
    finally:
        os.close(descriptor)


def terminal_authorization_exists(
    run_path: str | os.PathLike[str],
    *,
    expected_source: str | None = None,
    expected_run_id: str | None = None,
) -> bool:
    """Return whether an exact reconstructible terminal decision exists.

    This is a strict check: a malformed/partial authorization raises instead of
    being mistaken for absence.  It does not inspect or mutate ``run.json``.
    """

    path = _absolute_lexical(run_path)
    if path.name != "run.json":
        raise CompletionError("authorized terminal path must be run.json")
    if not _private_leaf_exists(path.parent, FINALIZATION_CLAIM_FILENAME):
        return False
    _load_authorized_terminal(
        path.parent,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
    )
    return True


def recover_authorized_terminal(
    run_path: str | os.PathLike[str],
    *,
    expected_source: str | None = None,
    expected_run_id: str | None = None,
) -> bool:
    """Materialize ``run.json`` from a permanent authorized candidate.

    The caller **must already hold** the per-source :func:`source_lock`; this helper
    intentionally does not reacquire it because flock recursion can deadlock.  It is
    designed for startup before pending-dedup reconciliation.  No authorization is
    a normal ``False`` result.  A partial/malformed authorization or conflicting
    terminal record fails closed.
    """

    path = _absolute_lexical(run_path)
    if path.name != "run.json":
        raise CompletionError("authorized terminal recovery path must be run.json")
    run_dir = _reject_symlink_components(path.parent)
    try:
        run_dir_info = run_dir.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(run_dir_info.st_mode):
        raise UnsafePathError(f"authorized terminal run path is not a directory: {run_dir}")
    if not _private_leaf_exists(run_dir, FINALIZATION_CLAIM_FILENAME):
        # A candidate without authorization is only an interrupted proposal; it is
        # deliberately not materialized and cannot qualify for strict consumption.
        return False
    authorized = _load_authorized_terminal(
        run_dir,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
    )
    _reject_terminal_recovery_guard(run_dir)
    # Make an authorization that was visible after an ambiguous create/fsync error
    # durable before acting on it.  The candidate file itself was fsynced before its
    # directory entry and is re-read above.
    fsync_directory(run_dir)

    try:
        current_payload, current_proof = _read_private_bytes(
            path, max_bytes=_MAX_RECORD_BYTES
        )
    except CompletionError:
        if os.path.lexists(path):
            raise
        try:
            atomic_replace_private(path, authorized.payload)
        except BaseException as materialize_error:
            raise CompletionMaterializationPending(
                "authorized terminal exists but missing run.json could not be materialized"
            ) from materialize_error
    else:
        if (
            current_payload == authorized.payload
            and current_proof.size_bytes == authorized.candidate.size_bytes
            and current_proof.sha256 == authorized.candidate.sha256
        ):
            fsync_directory(run_dir)
            return True
        current = _strict_json_object(current_payload, label=f"recovery record {path}")
        if current_payload != canonical_json_bytes(current):
            raise CompletionError("conflicting run.json is not canonical")
        if current.get("outcome") in {"success", "failure"}:
            raise CompletionAlreadyFinalized(
                "conflicting terminal run.json differs from its authorization"
            )
        _validate_startup_record(current)
        if (
            current["source"] != authorized.record["source"]
            or current["run_id"] != authorized.record["run_id"]
        ):
            raise CompletionError("startup record identity conflicts with authorization")
        immutable_startup_keys = _STARTUP_KEYS - {
            "outcome",
            "quality_passed",
            "feeds_durable",
            "failure_count",
        }
        drifted = sorted(
            key
            for key in immutable_startup_keys
            if current[key] != authorized.record[key]
        )
        if drifted:
            raise CompletionError(
                f"startup record conflicts with authorized field(s): {drifted}"
            )
        try:
            atomic_replace_terminal_private(
                path,
                authorized.payload,
                current_payload,
            )
        except BaseException as materialize_error:
            raise CompletionMaterializationPending(
                "authorized terminal exists but run.json recovery remains pending"
            ) from materialize_error

    recovered_payload, recovered_proof = _read_private_bytes(
        path, max_bytes=_MAX_RECORD_BYTES
    )
    if (
        recovered_payload != authorized.payload
        or recovered_proof.size_bytes != authorized.candidate.size_bytes
        or recovered_proof.sha256 != authorized.candidate.sha256
    ):
        raise CompletionMaterializationPending(
            "authorized terminal recovery did not produce exact run.json bytes"
        )
    fsync_directory(run_dir)
    return True


def verify_authorized_terminal_record(
    path: str | os.PathLike[str],
    expected_source: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, object]:
    """Verify the immutable WAL/run binding for either terminal outcome.

    This deliberately does not turn a structurally valid success into eligibility;
    :func:`verify_terminal_record` adds current item, quality, feed, and source proof.
    It exists so recovery can distinguish an exact authorized terminal *failure*
    from transient/ambiguous inability to prove success.
    """

    record_path = _absolute_lexical(path)
    if record_path.name != "run.json":
        raise CompletionError("authorized terminal record path must be run.json")
    _reject_terminal_recovery_guard(record_path.parent)
    authorized = _load_authorized_terminal(
        record_path.parent,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
    )
    record_payload, record_proof = _read_private_bytes(
        record_path, max_bytes=_MAX_RECORD_BYTES
    )
    if (
        record_payload != authorized.payload
        or record_proof.size_bytes != authorized.candidate.size_bytes
        or record_proof.sha256 != authorized.candidate.sha256
    ):
        raise CompletionError(
            "authoritative run.json does not exactly match its authorized candidate"
        )
    record = _strict_json_object(record_payload, label=f"completion record {path}")
    if record_payload != canonical_json_bytes(record):
        raise CompletionError("completion record bytes are not canonical schema-v1 JSON")
    _validate_terminal_mapping(record)
    payload_2, record_proof_2 = _read_private_bytes(
        record_proof.path, max_bytes=_MAX_RECORD_BYTES
    )
    if record_proof_2 != record_proof or payload_2 != record_payload:
        raise CompletionError("completion record changed while verifying")
    authorized_2 = _load_authorized_terminal(
        record_proof.path.parent,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
    )
    if authorized_2 != authorized:
        raise CompletionError("terminal candidate or authorization changed while verifying")
    _reject_terminal_recovery_guard(record_proof.path.parent)
    return record


def verify_terminal_record(
    path: str | os.PathLike[str],
    expected_source: str | None = None,
    expected_run_id: str | None = None,
    expected_items_path: str | os.PathLike[str] | None = None,
    *,
    validator: object | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Strictly reload a successful, authorized run and its current item binding."""

    record_path = _absolute_lexical(path)
    record = verify_authorized_terminal_record(
        record_path,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
    )
    _validate_success_binding(
        record,
        record_path=record_path,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
        expected_items_path=expected_items_path,
        validator=validator,
        now=now,
    )
    record_2 = verify_authorized_terminal_record(
        record_path,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
    )
    if record_2 != record:
        raise CompletionError("authorized terminal record changed while verifying success")
    return record


verify_completion_record = verify_terminal_record


def _load_startup_at(path: Path) -> dict[str, object]:
    payload, _proof = _read_private_bytes(path, max_bytes=_MAX_RECORD_BYTES)
    value = _strict_json_object(payload, label=f"startup record {path}")
    if value.get("outcome") in {"success", "failure"}:
        raise CompletionAlreadyFinalized(f"terminal completion already exists: {path}")
    if payload != canonical_json_bytes(value):
        raise CompletionError(f"startup record is not canonical: {path}")
    _validate_startup_record(value)
    return value


def _load_guard_metadata(path: Path) -> dict[str, object]:
    """Load canonical startup/terminal metadata solely for a latest identity guard."""

    payload, _proof = _read_private_bytes(path, max_bytes=_MAX_RECORD_BYTES)
    value = _strict_json_object(payload, label=f"latest guard record {path}")
    if payload != canonical_json_bytes(value):
        raise CompletionError(f"latest guard record is not canonical: {path}")
    outcome = value.get("outcome")
    if outcome == "started":
        _validate_startup_record(value)
    elif outcome in {"success", "failure"}:
        _validate_terminal_mapping(value)
    else:
        raise CompletionError(f"latest guard record has unknown outcome: {path}")
    return value


def publish_terminal_record(
    spider: object, record: Mapping[str, object]
) -> PublicationResult:
    """Authorize once, materialize run startup, then guard-copy bytes to latest.

    Publication is a write-ahead protocol: a permanent full-payload candidate is
    durable first, then a permanent exact authorization, and only then ``run.json``
    is replaced.  Therefore a publisher error before authorization can never create
    acceptable terminal evidence, while an interrupted later materialization is
    reconstructible without inventing historical bytes.
    """

    terminal = dict(record)
    _validate_terminal_mapping(terminal)
    run_path = _absolute_lexical(getattr(spider, "run_metadata_path"))
    latest_path = _absolute_lexical(getattr(spider, "latest_metadata_path"))
    run_id = str(getattr(spider, "run_id"))
    source = str(getattr(spider, "name"))
    if terminal["run_id"] != run_id or terminal["source"] != source:
        raise CompletionError("terminal record does not match the publishing spider")
    if run_path.parent.name != run_id:
        raise CompletionError("spider run metadata path has an inconsistent run_id")
    source_root = run_path.parent.parents[1]
    if source_root.name != source:
        raise CompletionError("spider run metadata path has an inconsistent source")
    payload = canonical_json_bytes(terminal)
    if len(payload) > _MAX_RECORD_BYTES:
        raise CompletionError("terminal completion record exceeds the fixed size bound")
    with source_lock(source_root):
        startup = _load_startup_at(run_path)
        if startup["run_id"] != run_id or startup["source"] != source:
            raise CompletionError("authoritative startup record identity mismatch")
        immutable_startup_keys = _STARTUP_KEYS - {
            "outcome",
            "quality_passed",
            "feeds_durable",
            "failure_count",
        }
        drifted = sorted(
            key for key in immutable_startup_keys if terminal[key] != startup[key]
        )
        if drifted:
            raise CompletionError(
                f"terminal record changed immutable startup field(s): {drifted}"
            )
        if terminal["outcome"] == "success":
            # Perform every semantic parse, hash, and source-specific validator call
            # before taking the one-shot claim.  After the claim, only the guarded
            # atomic publication primitive is allowed to touch authoritative run.json.
            _validate_success_binding(
                terminal,
                record_path=run_path,
                expected_source=source,
                expected_run_id=run_id,
                expected_items_path=getattr(spider, "items_path"),
                validator=None,
                now=None,
            )
        candidate_path = run_path.parent / TERMINAL_CANDIDATE_FILENAME
        claim_path = run_path.parent / FINALIZATION_CLAIM_FILENAME
        try:
            # The permanent full payload is the recovery source.  A hash-only
            # preauthorization would be unable to reconstruct run.json after a
            # namespace rollback.
            atomic_create_private(candidate_path, payload)
        except FileExistsError as exc:
            raise CompletionAlreadyFinalized(
                f"run finalization was already claimed: {run_id}"
            ) from exc
        candidate_payload, candidate_proof = _read_private_bytes(
            candidate_path, max_bytes=_MAX_RECORD_BYTES
        )
        if candidate_payload != payload:
            raise CompletionError("durable terminal candidate bytes changed")
        try:
            atomic_create_private(
                claim_path,
                _terminal_authorization_payload(
                    source=source,
                    run_id=run_id,
                    candidate=candidate_proof,
                ),
            )
        except FileExistsError as exc:
            raise CompletionAlreadyFinalized(
                f"run finalization was already authorized: {run_id}"
            ) from exc

        # Reload the two permanent files before touching the authoritative name.
        # This also rejects an authorization whose exact candidate binding drifted.
        authorized = _load_authorized_terminal(
            run_path.parent,
            expected_source=source,
            expected_run_id=run_id,
        )
        if authorized.payload != payload:
            raise CompletionError("authorized terminal differs from publication payload")
        try:
            atomic_replace_terminal_private(
                run_path,
                payload,
                canonical_json_bytes(startup),
            )
        except BaseException as exc:
            # Authorization is an irrevocable commit decision.  The atomic helper
            # normally handles this check itself, but retain the same audit
            # invariant around a replaced/instrumented helper: never report failure
            # while strict consumers can observe exact authorized terminal bytes.
            parent_fd = _open_directory_nofollow(run_path.parent)
            try:
                exact_after_error = (
                    _read_private_leaf(
                        parent_fd,
                        run_path.name,
                        max_bytes=_MAX_RECORD_BYTES,
                    )
                    == authorized.payload
                )
            except BaseException:
                exact_after_error = False
            finally:
                try:
                    os.close(parent_fd)
                except BaseException:
                    pass
            if not exact_after_error:
                # Otherwise run.json remains nonqualifying and locked startup
                # recovery must reconstruct it before pending dedup is touched.
                raise CompletionMaterializationPending(
                    f"terminal {source}/{run_id} is authorized but materialization is pending"
                ) from exc
        try:
            materialized_payload, materialized_proof = _read_private_bytes(
                run_path, max_bytes=_MAX_RECORD_BYTES
            )
        except BaseException as exc:
            raise CompletionMaterializationPending(
                f"terminal {source}/{run_id} materialization could not be durably confirmed"
            ) from exc
        exact_visible = (
            materialized_payload == authorized.payload
            and materialized_proof.size_bytes == authorized.candidate.size_bytes
            and materialized_proof.sha256 == authorized.candidate.sha256
        )
        if not exact_visible:
            raise CompletionMaterializationPending(
                f"terminal {source}/{run_id} does not match its authorized candidate"
            )
        if terminal["outcome"] == "success":
            # A successful publication is not returned until the same strict
            # consumer contract (including WAL rereads and current item/source
            # proofs) accepts it.  This keeps publication outcome and dedup
            # eligibility identical under composed filesystem faults.
            try:
                verify_terminal_record(
                    run_path,
                    expected_source=source,
                    expected_run_id=run_id,
                    expected_items_path=getattr(spider, "items_path"),
                )
            except BaseException as exc:
                raise CompletionMaterializationPending(
                    f"terminal {source}/{run_id} failed strict post-publication verification"
                ) from exc

        latest_updated = False
        try:
            latest = _load_guard_metadata(latest_path)
        except (CompletionError, OSError):
            latest = None
        if (
            latest is not None
            and latest.get("run_id") == run_id
            and latest.get("source") == source
        ):
            try:
                atomic_replace_private(latest_path, payload)
            except (CompletionError, OSError):
                # run.json is authoritative and already durable.  A damaged or
                # concurrently superseded mutable latest pointer cannot revoke it.
                latest_updated = False
            else:
                latest_updated = True
        return PublicationResult(run_path, latest_path, latest_updated)


def finalize_pagination_reconcilers(
    reconcilers: Mapping[object, object] | Iterable[object] | None,
) -> tuple[object, ...]:
    """Idempotently finalize every registered reconciler in deterministic order."""

    if reconcilers is None:
        return ()
    if isinstance(reconcilers, Mapping):
        values = [reconcilers[key] for key in sorted(reconcilers, key=str)]
    else:
        values = list(reconcilers)
    outcomes = []
    for tracker in values:
        finalize = getattr(tracker, "finalize", None)
        if not callable(finalize):
            raise CompletionError("registered pagination reconciler has no finalize()")
        outcomes.append(finalize())
    return tuple(outcomes)


__all__ = [
    "COMPLETION_SCHEMA_VERSION",
    "FINALIZATION_CLAIM_FILENAME",
    "SUPREMECOURT_FINISH_REASON",
    "TERMINAL_CANDIDATE_FILENAME",
    "TERMINAL_RECOVERY_GUARD_FILENAME",
    "AuthorizedTerminal",
    "CompletionAlreadyFinalized",
    "CompletionError",
    "CompletionMaterializationPending",
    "FeedDurability",
    "FileProof",
    "PublicationResult",
    "QualityEvaluation",
    "UnsafePathError",
    "atomic_create_private",
    "atomic_replace_private",
    "atomic_replace_terminal_private",
    "attest_feed_outputs",
    "attest_materialized_outputs",
    "build_startup_record",
    "build_terminal_record",
    "canonical_json_bytes",
    "canonical_utc_format",
    "canonical_utc_now",
    "canonical_utc_second",
    "evaluate_crawl_quality",
    "evaluate_quality",
    "failed_feed_durability",
    "failed_source_validation",
    "finalize_pagination_reconcilers",
    "fsync_directory",
    "generic_source_validation",
    "hash_and_fsync_private_file",
    "local_feed_path",
    "publish_startup_metadata",
    "publish_terminal_record",
    "recover_authorized_terminal",
    "source_lock",
    "startup_record",
    "terminal_authorization_exists",
    "validate_supremecourt_source",
    "verify_authorized_terminal_record",
    "verify_completion_record",
    "verify_terminal_record",
]
