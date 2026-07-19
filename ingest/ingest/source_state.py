"""Strict source-run attestation validation and evidence publication.

Production snapshots must be selected from an explicit ledger of exact crawl runs.  This
module is the single consumer and builder for that ledger: it validates completion schema
v1, binds it to immutable ``items.jsonl`` bytes, reruns the Supreme Court partial-crawl
validator, and publishes deterministic evidence without replacing an existing path.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlparse

from .artifacts import _run_success_reason

SOURCE_STATE_EVIDENCE_SCHEMA_VERSION = 1
CANDIDATE_SOURCE_STATE_EVIDENCE_SCHEMA_VERSION = 2
COMPLETION_SCHEMA_VERSION = 1
EVIDENCE_COMPLETION_SCHEMA_VERSION = 2
CANDIDATE_SNAPSHOT_ID = "v3_512_attested_20260715_01"
CANDIDATE_START_DATE = "1900-01-01"
CANDIDATE_END_DATE = "2026-07-15"
CANDIDATE_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1]
    / ".state/v3/source-evidence/v3_512_attested_20260715_01"
).absolute()
TERMINAL_CANDIDATE_FILENAME = ".completion-terminal-candidate.json"
FINALIZATION_CLAIM_FILENAME = ".completion-finalized"
TERMINAL_RECOVERY_GUARD_FILENAME = ".completion-recovery-required"
PRODUCTION_SOURCES = (
    "matsne",
    "napr",
    "ecd",
    "constcourt",
    "supremecourt",
    "tas",
    "tbappeal",
)
_SUPREME_OFFICIAL_CHAMBERS = frozenset(
    {
        "ადმინისტრაციულ საქმეთა პალატა",
        "სამოქალაქო საქმეთა პალატა",
        "სისხლის სამართლის საქმეთა პალატა",
    }
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_CONTRACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_UTC_SECOND_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_COMPLETION_BYTES = 1024 * 1024
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_MAX_AUTHORIZATION_BYTES = 16 * 1024
_AUTHORIZATION_SCHEMA_VERSION = 1
_AUTHORIZATION_STATE = "source_state_authorized"

_TERMINAL_AUTHORIZATION_KEYS = {
    "schema_version",
    "state",
    "source",
    "run_id",
    "candidate_filename",
    "terminal_size_bytes",
    "terminal_sha256",
}

_COMPLETION_KEYS = {
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
    "finish_reason",
    "completed_at",
    "feed_outputs",
    "quality",
    "source_validation",
}
_EVIDENCE_COMPLETION_KEYS = _COMPLETION_KEYS | {"crawl_contract"}
_CRAWL_CONTRACT_KEYS = {
    "kind",
    "start_date",
    "end_date",
    "source_arguments",
    "code_revision",
    "code_identity_sha256",
    "artifact_root",
    "http_cache_enabled",
    "cross_run_dedup_enabled",
    "shared_seen_sqlite_accessed",
    "within_run_duplicates",
    "discovery_limits",
    "terminal_status",
    "durable_outputs",
    "completed_at",
    "quality_passed",
    "feeds_durable",
    "failure_signals",
}
_ORDINARY_IDENTITY_FIELDS = {
    "matsne": ("document_id",),
    "ecd": ("decision_document_id",),
    "constcourt": ("legal_id",),
    "napr": ("document_id",),
    "tas": ("document_id",),
    "tbappeal": ("slug",),
}
_EXPECTED_DISCOVERY_LIMITS = {
    "matsne": {"max_pages": 20_000, "max_safe_pages": 90},
    "ecd": {"max_pages": 20_000, "page_size": 50},
    "constcourt": {"max_pages": 5_000, "page_size": 50},
    "napr": {"max_pages": 20_000, "page_size": 50},
    "tas": {"max_pages": 20_000, "page_size": 50},
    "tbappeal": {"max_pages": 1_000},
    "supremecourt": {
        "max_pages": 5_000,
        "max_window_days": 366,
        "page_size": 30,
        "target_max": 30,
        "target_min": 20,
    },
}
_IDENTITY_JOURNAL_KEYS = {"identity_sha256", "outcome", "sequence"}
_IDENTITY_JOURNAL_OUTCOMES = frozenset({"unique", "duplicate", "missing"})
_DURABLE_OUTPUT_KEYS = {"role", "path", "size_bytes", "sha256"}
_WITHIN_RUN_DUPLICATE_KEYS = {
    "observed",
    "unique",
    "duplicates",
    "missing_identity",
    "reconciled",
}
_FAILURE_SIGNAL_KEYS = {
    "quality_failures",
    "spider_errors",
    "spider_exceptions",
    "item_errors",
    "feed_failures",
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
_GENERIC_VALIDATION_KEYS = {"kind", "passed"}
_SUPREMECOURT_VALIDATION_KEYS = {
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


class SourceStateError(RuntimeError):
    """A selected crawl run or source-state ledger cannot be proven safe."""


class SourceStatePublicationPending(SourceStateError):
    """A durable authorization exists, but output stabilization is still pending."""


@dataclass(frozen=True, slots=True)
class FileAttestation:
    """Exact identity of one safely read regular file."""

    path: Path
    sha256: str
    size_bytes: int
    device: int | None = None
    inode: int | None = None


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    """A successful run bound to exact item and authorized completion bytes."""

    source: str
    run_id: str
    items: FileAttestation
    completion: FileAttestation
    terminal_candidate: FileAttestation
    terminal_authorization: FileAttestation
    completed_at: str
    completion_schema_version: int = COMPLETION_SCHEMA_VERSION

    def evidence_row(self) -> dict[str, str]:
        return {
            "source": self.source,
            "run_id": self.run_id,
            "items_sha256": self.items.sha256,
            "completion_record_sha256": self.completion.sha256,
        }


@dataclass(frozen=True, slots=True)
class LoadedEvidence:
    """Validated source-state input plus its own byte attestation."""

    runs: tuple[ValidatedRun, ...]
    evidence: FileAttestation
    schema_version: int = SOURCE_STATE_EVIDENCE_SCHEMA_VERSION
    supreme_coverage_chain: tuple[ValidatedRun, ...] = ()


@dataclass(frozen=True, slots=True)
class _AuthorizedEvidenceBundle:
    payload: bytes
    evidence: FileAttestation
    candidate: FileAttestation
    authorization: FileAttestation
    parent_device: int
    parent_inode: int


def canonical_json_bytes(value: object) -> bytes:
    """Serialize evidence deterministically; a final newline is part of the format."""

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


def _absolute_lexical(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: Path | str) -> Path:
    """Reject each existing symlink without resolving through an attacker path."""

    absolute = _absolute_lexical(path)
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SourceStateError(
                f"cannot inspect path component {cursor}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise SourceStateError(f"refusing symlink path component: {cursor}")
    return absolute


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    """Open an absolute directory one no-follow component at a time."""

    absolute = _absolute_lexical(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise SourceStateError(
            f"cannot open filesystem root for {label}: {exc}"
        ) from exc
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise SourceStateError(
                    f"cannot safely traverse {label} component {part!r}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _require_private_directory(path: Path, *, label: str) -> None:
    _reject_symlink_components(path)
    descriptor = _open_directory_nofollow(path, label=label)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise SourceStateError(f"{label} must be a real directory: {path}")
        mode = stat.S_IMODE(info.st_mode)
        if mode != 0o700 or info.st_uid != os.geteuid():
            raise SourceStateError(
                f"{label} must be owned by this user with mode 0700; "
                f"uid={info.st_uid}, mode={mode:04o}: {path}"
            )
    finally:
        os.close(descriptor)


def _reject_terminal_recovery_guard(run_dir: Path) -> None:
    """Reject a run whose terminal publication still requires recovery."""

    descriptor = _open_directory_nofollow(run_dir, label="selected run directory")
    try:
        try:
            os.stat(
                TERMINAL_RECOVERY_GUARD_FILENAME,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SourceStateError(
                "cannot inspect terminal publication recovery guard: "
                f"{run_dir / TERMINAL_RECOVERY_GUARD_FILENAME}: {exc}"
            ) from exc
    finally:
        os.close(descriptor)
    raise SourceStateError(
        "selected run has an incomplete terminal publication recovery guard: "
        f"{run_dir / TERMINAL_RECOVERY_GUARD_FILENAME}"
    )


def _read_private_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int | None = None,
    capture: bool = True,
    exact_mode: int | None = None,
) -> tuple[bytes, FileAttestation]:
    """Read and hash an owner-private regular file through a no-follow descriptor."""

    _reject_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = _open_directory_nofollow(path.parent, label=f"{label} parent")
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        os.close(parent_descriptor)
        raise SourceStateError(f"cannot safely open {label}: {path}: {exc}") from exc
    os.close(parent_descriptor)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SourceStateError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        mode = stat.S_IMODE(info.st_mode)
        if info.st_uid != os.geteuid():
            raise SourceStateError(f"{label} must be owned by the current user: {path}")
        if exact_mode is not None and mode != exact_mode:
            raise SourceStateError(
                f"{label} must have mode {exact_mode:04o}; mode is {mode:04o}: {path}"
            )
        if exact_mode is None and mode & 0o077:
            raise SourceStateError(
                f"{label} must be owner-private; mode is {mode:04o}: {path}"
            )
        if max_bytes is not None and info.st_size > max_bytes:
            raise SourceStateError(
                f"{label} exceeds the {max_bytes}-byte safety limit: {path}"
            )
        payload = bytearray() if capture else None
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if max_bytes is not None and size + len(block) > max_bytes:
                raise SourceStateError(
                    f"{label} exceeds the {max_bytes}-byte safety limit while reading: "
                    f"{path}"
                )
            if payload is not None:
                payload.extend(block)
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        if (
            size != info.st_size
            or after.st_size != info.st_size
            or after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
            or after.st_mtime_ns != info.st_mtime_ns
        ):
            raise SourceStateError(f"{label} changed while reading: {path}")
        return (
            bytes(payload) if payload is not None else b"",
            FileAttestation(
                path,
                digest.hexdigest(),
                size,
                device=info.st_dev,
                inode=info.st_ino,
            ),
        )
    finally:
        os.close(descriptor)


def iter_attested_lines(
    attestation: FileAttestation,
    *,
    label: str = "attested file",
) -> Iterator[bytes]:
    """Yield exact bytes from a no-follow open and verify the consumed stream.

    Snapshot construction uses this instead of reopening a validated pathname with
    ``Path.open``.  Any rename, growth, truncation, or same-path byte substitution is
    detected against the hash and size that were strictly validated before the build.
    """

    path = _reject_symlink_components(attestation.path)
    parent_descriptor = _open_directory_nofollow(path.parent, label=f"{label} parent")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise SourceStateError(f"cannot safely open {label}: {path}: {exc}") from exc
    finally:
        os.close(parent_descriptor)

    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
        ):
            raise SourceStateError(
                f"{label} must remain a current-user-owned mode-0600 regular file: "
                f"{path}"
            )
        if before.st_size != attestation.size_bytes:
            raise SourceStateError(f"{label} size changed before consumption: {path}")
        digest = hashlib.sha256()
        consumed = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for raw_line in handle:
                digest.update(raw_line)
                consumed += len(raw_line)
                yield raw_line
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or consumed != attestation.size_bytes
            or digest.hexdigest() != attestation.sha256
        ):
            raise SourceStateError(f"{label} bytes changed during consumption: {path}")
    finally:
        os.close(descriptor)


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def object_without_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SourceStateError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise SourceStateError(f"non-finite JSON number {value!r} in {label}")

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except SourceStateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SourceStateError(f"invalid JSON object in {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SourceStateError(f"JSON value must be an object in {label}")
    return parsed


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SourceStateError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_bool(value: object, *, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise SourceStateError(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise SourceStateError(f"{label} must be {str(expected).lower()}")
    return value


def _require_int(value: object, *, label: str, expected: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise SourceStateError(f"{label} must be a nonnegative integer")
    if expected is not None and value != expected:
        raise SourceStateError(f"{label} must be {expected}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceStateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ordinary_identity_sha256(
    source: str, item: Mapping[str, object], *, label: str
) -> str:
    fields = _ORDINARY_IDENTITY_FIELDS.get(source)
    if fields is None:
        raise SourceStateError(f"{source} has no frozen ordinary identity projection")
    parts: list[str] = []
    for field in fields:
        value = item.get(field)
        if value is None or value == "":
            raise SourceStateError(f"{label} lacks identity field {field!r}")
        parts.append(str(value).strip())
    identity = ":".join(parts)
    if not identity:
        raise SourceStateError(f"{label} has an empty durable identity")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _validate_ordinary_identity_evidence(
    *,
    source: str,
    items: FileAttestation,
    journal: FileAttestation,
    declared: Mapping[str, object],
) -> None:
    """Recompute ordinary feed and within-run identity counts from durable bytes."""

    feed_identities: set[str] = set()
    feed_count = 0
    for line_number, raw_line in enumerate(
        iter_attested_lines(items, label=f"{source} selected items.jsonl"), start=1
    ):
        if not raw_line.endswith(b"\n") or not raw_line.strip():
            raise SourceStateError(
                f"{source} items.jsonl line {line_number} is not a complete JSONL row"
            )
        item = _strict_json_object(
            raw_line, label=f"{source} items.jsonl line {line_number}"
        )
        identity_sha256 = _ordinary_identity_sha256(
            source, item, label=f"{source} items.jsonl line {line_number}"
        )
        if identity_sha256 in feed_identities:
            raise SourceStateError(f"{source} feed repeats an ordinary item identity")
        feed_identities.add(identity_sha256)
        feed_count += 1

    event_counts = {outcome: 0 for outcome in _IDENTITY_JOURNAL_OUTCOMES}
    unique_identities: set[str] = set()
    expected_sequence = 1
    for line_number, raw_line in enumerate(
        iter_attested_lines(journal, label=f"{source} identity journal"), start=1
    ):
        event = _strict_json_object(
            raw_line, label=f"{source} identity journal line {line_number}"
        )
        if raw_line != canonical_json_bytes(event):
            raise SourceStateError(
                f"{source} identity journal line {line_number} is not canonical"
            )
        _exact_keys(
            event,
            _IDENTITY_JOURNAL_KEYS,
            label=f"{source} identity journal line {line_number}",
        )
        if event["sequence"] != expected_sequence:
            raise SourceStateError(
                f"{source} identity journal sequence is not contiguous"
            )
        expected_sequence += 1
        outcome = event["outcome"]
        if outcome not in _IDENTITY_JOURNAL_OUTCOMES:
            raise SourceStateError(f"{source} identity journal outcome is invalid")
        identity_sha256 = event["identity_sha256"]
        if outcome == "missing":
            if identity_sha256 is not None:
                raise SourceStateError(
                    f"{source} missing-identity event carries an identity"
                )
        else:
            identity_sha256 = _require_sha256(
                identity_sha256,
                label=f"{source} identity journal identity_sha256",
            )
            if outcome == "unique":
                if identity_sha256 in unique_identities:
                    raise SourceStateError(
                        f"{source} identity journal repeats a unique identity"
                    )
                unique_identities.add(identity_sha256)
            elif identity_sha256 not in unique_identities:
                raise SourceStateError(
                    f"{source} duplicate event precedes its unique identity"
                )
        event_counts[str(outcome)] += 1

    expected_declared = {
        "observed": event_counts["unique"] + event_counts["duplicate"],
        "unique": event_counts["unique"],
        "duplicates": event_counts["duplicate"],
        "missing_identity": event_counts["missing"],
        "reconciled": event_counts["missing"] == 0,
    }
    if dict(declared) != expected_declared:
        raise SourceStateError(
            f"{source} declared within-run counts differ from durable identity evidence"
        )
    if feed_count != event_counts["unique"] or feed_identities != unique_identities:
        raise SourceStateError(
            f"{source} feed identities differ from unique identity decisions"
        )


def _load_terminal_authorization(
    run_dir: Path, *, source: str, run_id: str
) -> tuple[bytes, FileAttestation, FileAttestation]:
    """Read the permanent full candidate and its exact authorization binding."""

    authorization_payload, authorization_file = _read_private_regular(
        run_dir / FINALIZATION_CLAIM_FILENAME,
        label="terminal authorization",
        max_bytes=_MAX_COMPLETION_BYTES,
        exact_mode=0o600,
    )
    authorization = _strict_json_object(
        authorization_payload,
        label=f"terminal authorization {source}/{run_id}",
    )
    if authorization_payload != canonical_json_bytes(authorization):
        raise SourceStateError("terminal authorization is not canonically serialized")
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
        raise SourceStateError("terminal authorization has the wrong state")
    if authorization["source"] != source or authorization["run_id"] != run_id:
        raise SourceStateError("terminal authorization identity mismatch")
    if authorization["candidate_filename"] != TERMINAL_CANDIDATE_FILENAME:
        raise SourceStateError("terminal authorization names an unexpected candidate")
    authorized_size = _require_int(
        authorization["terminal_size_bytes"],
        label="terminal authorization.terminal_size_bytes",
    )
    if authorized_size > _MAX_COMPLETION_BYTES:
        raise SourceStateError("terminal candidate exceeds the fixed safety limit")
    authorized_sha = _require_sha256(
        authorization["terminal_sha256"],
        label="terminal authorization.terminal_sha256",
    )

    candidate_payload, candidate_file = _read_private_regular(
        run_dir / TERMINAL_CANDIDATE_FILENAME,
        label="terminal candidate",
        max_bytes=_MAX_COMPLETION_BYTES,
        exact_mode=0o600,
    )
    if (
        candidate_file.size_bytes != authorized_size
        or candidate_file.sha256 != authorized_sha
    ):
        raise SourceStateError("terminal candidate does not match its authorization")
    candidate = _strict_json_object(
        candidate_payload,
        label=f"terminal candidate {source}/{run_id}",
    )
    if candidate_payload != canonical_json_bytes(candidate):
        raise SourceStateError("terminal candidate is not canonically serialized")
    if candidate.get("source") != source or candidate.get("run_id") != run_id:
        raise SourceStateError("terminal candidate identity mismatch")
    return candidate_payload, candidate_file, authorization_file


def _canonical_date(value: object, *, label: str) -> date:
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        raise SourceStateError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SourceStateError(f"{label} is not a valid calendar date") from exc
    if parsed.isoformat() != value:
        raise SourceStateError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def _canonical_utc_second(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECOND_RE.fullmatch(value) is None:
        raise SourceStateError(f"{label} must be canonical second-precision UTC (...Z)")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceStateError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SourceStateError(f"{label} must be UTC")
    return parsed


def _canonical_path_value(value: object, *, expected: Path, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceStateError(f"{label} must be an absolute canonical path")
    if value != os.fspath(_absolute_lexical(value)) or Path(value) != expected:
        raise SourceStateError(f"{label} does not bind the selected run path")
    return value


def _local_configured_uri(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceStateError(f"{label} must be a non-empty local feed URI")
    parsed = urlparse(value)
    if parsed.scheme == "file":
        if (
            parsed.netloc not in {"", "localhost"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise SourceStateError(f"{label} must be a local file URI")
        return _absolute_lexical(unquote(parsed.path))
    if parsed.scheme:
        raise SourceStateError(f"{label} uses unsupported nonlocal feed scheme")
    return _absolute_lexical(value)


def _load_supremecourt_validator() -> ModuleType:
    """Load the repository's unchanged strict validator without network access."""

    module_name = "_legal_source_state_supremecourt_validator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "validate_supremecourt_partial.py"
    )
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("could not construct module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SourceStateError(
            f"strict Supreme Court validator is unavailable: {path}: {exc}"
        ) from exc
    return module


def _validate_feed_outputs(
    value: object,
    *,
    run_dir: Path,
    latest_dir: Path | None,
    items: FileAttestation,
) -> None:
    if not isinstance(value, dict):
        raise SourceStateError("completion.feed_outputs must be an object")
    _exact_keys(value, _FEED_OUTPUT_KEYS, label="completion.feed_outputs")
    _require_bool(
        value["durable"], label="completion.feed_outputs.durable", expected=True
    )
    expected_count = 1 if latest_dir is None else 2
    _require_int(
        value["configured_count"],
        label="completion.feed_outputs.configured_count",
        expected=expected_count,
    )
    _require_int(
        value["success_count"],
        label="completion.feed_outputs.success_count",
        expected=expected_count,
    )
    _require_int(
        value["failure_count"],
        label="completion.feed_outputs.failure_count",
        expected=0,
    )
    files = value["files"]
    if not isinstance(files, list) or len(files) != expected_count:
        raise SourceStateError(
            "completion.feed_outputs.files must contain exactly "
            f"{expected_count} row(s)"
        )

    def sort_key(row: object) -> tuple[str, str, str]:
        if not isinstance(row, dict):
            return ("", "", "")
        return (
            str(row.get("role", "")),
            str(row.get("configured_uri", "")),
            str(row.get("path", "")),
        )

    if files != sorted(files, key=sort_key):
        raise SourceStateError(
            "completion.feed_outputs.files must be deterministically sorted"
        )
    expected_paths = {"run": run_dir / "items.jsonl"}
    if latest_dir is not None:
        expected_paths["latest"] = latest_dir / "items.jsonl"
    seen_roles: set[str] = set()
    for index, raw in enumerate(files):
        label = f"completion.feed_outputs.files[{index}]"
        if not isinstance(raw, dict):
            raise SourceStateError(f"{label} must be an object")
        _exact_keys(raw, _FEED_FILE_KEYS, label=label)
        role = raw["role"]
        if role not in expected_paths or role in seen_roles:
            raise SourceStateError(f"{label}.role must uniquely identify run or latest")
        seen_roles.add(role)
        expected_path = expected_paths[role]
        _canonical_path_value(
            raw["path"], expected=expected_path, label=f"{label}.path"
        )
        if (
            _local_configured_uri(
                raw["configured_uri"], label=f"{label}.configured_uri"
            )
            != expected_path
        ):
            raise SourceStateError(
                f"{label}.configured_uri does not match its canonical path"
            )
        _require_int(
            raw["size_bytes"], label=f"{label}.size_bytes", expected=items.size_bytes
        )
        digest = _require_sha256(raw["sha256"], label=f"{label}.sha256")
        if digest != items.sha256:
            raise SourceStateError(
                f"{label}.sha256 does not match selected items.jsonl"
            )


def _validate_quality(value: object) -> None:
    if not isinstance(value, dict):
        raise SourceStateError("completion.quality must be an object")
    _exact_keys(value, _QUALITY_KEYS, label="completion.quality")
    _require_bool(value["passed"], label="completion.quality.passed", expected=True)
    for key in (
        "quality_failures",
        "spider_errors",
        "spider_exceptions",
        "item_errors",
    ):
        _require_int(value[key], label=f"completion.quality.{key}", expected=0)
    registered = _require_int(
        value["pagination_reconcilers"],
        label="completion.quality.pagination_reconcilers",
    )
    reconciled = _require_int(
        value["pagination_reconciled"],
        label="completion.quality.pagination_reconciled",
    )
    if reconciled != registered:
        raise SourceStateError(
            "completion.quality pagination reconcilers were not all finalized"
        )


def _validate_supremecourt_source(
    value: Mapping[str, object],
    *,
    run_dir: Path,
    items: FileAttestation,
    finish_reason: str = "closespider_timeout",
    require_terminal_coverage: bool = False,
) -> Mapping[str, object]:
    _exact_keys(
        value, _SUPREMECOURT_VALIDATION_KEYS, label="completion.source_validation"
    )
    if value["kind"] != "supremecourt_partial_v1":
        raise SourceStateError(
            "completion.source_validation.kind is invalid for supremecourt"
        )
    _require_bool(
        value["passed"], label="completion.source_validation.passed", expected=True
    )
    _require_int(
        value["validator_schema_version"],
        label="completion.source_validation.validator_schema_version",
        expected=1,
    )
    _canonical_path_value(
        value["run_dir"], expected=run_dir, label="completion.source_validation.run_dir"
    )
    if value["run_id"] != run_dir.name:
        raise SourceStateError("completion.source_validation.run_id mismatch")
    if value["finish_reason"] != finish_reason:
        raise SourceStateError("completion.source_validation.finish_reason drifted")
    if finish_reason not in {"closespider_timeout", "finished"}:
        raise SourceStateError("unsupported Supreme Court terminal reason")
    if (
        _require_sha256(
            value["items_sha256"], label="completion.source_validation.items_sha256"
        )
        != items.sha256
    ):
        raise SourceStateError("completion.source_validation.items_sha256 mismatch")
    manifest_payload, manifest = _read_private_regular(
        run_dir / "partial_manifest.json",
        label="Supreme Court partial manifest",
        capture=False,
    )
    journal_payload, journal = _read_private_regular(
        run_dir / "items.journal.jsonl",
        label="Supreme Court item journal",
        capture=False,
    )
    del manifest_payload, journal_payload
    _canonical_path_value(
        value["manifest_path"],
        expected=manifest.path,
        label="completion.source_validation.manifest_path",
    )
    _canonical_path_value(
        value["journal_path"],
        expected=journal.path,
        label="completion.source_validation.journal_path",
    )
    if (
        _require_sha256(
            value["manifest_sha256"],
            label="completion.source_validation.manifest_sha256",
        )
        != manifest.sha256
    ):
        raise SourceStateError("completion.source_validation.manifest_sha256 mismatch")
    if (
        _require_sha256(
            value["journal_sha256"], label="completion.source_validation.journal_sha256"
        )
        != journal.sha256
    ):
        raise SourceStateError("completion.source_validation.journal_sha256 mismatch")
    _require_int(
        value["unresolved_failure_count"],
        label="completion.source_validation.unresolved_failure_count",
        expected=0,
    )

    validator = _load_supremecourt_validator()
    validate_run = getattr(validator, "validate_run", None)
    if not callable(validate_run):
        raise SourceStateError("strict Supreme Court validator has no validate_run()")
    try:
        report = validate_run(run_dir)
    except Exception as exc:
        raise SourceStateError(
            f"strict Supreme Court validation failed: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise SourceStateError(
            "strict Supreme Court validator returned a malformed report"
        )
    exact_report_fields = {
        "schema_version": (int, value["validator_schema_version"]),
        "valid": (bool, True),
        "run_id": (str, run_dir.name),
        "run_dir": (str, os.fspath(run_dir)),
        "finish_reason": (str, value["finish_reason"]),
        "items_sha256": (str, items.sha256),
        "journal_sha256": (str, journal.sha256),
        "unresolved_failure_count": (int, 0),
    }
    for key, (expected_type, expected_value) in exact_report_fields.items():
        actual = report.get(key)
        if type(actual) is not expected_type or actual != expected_value:
            raise SourceStateError(
                f"strict Supreme Court validator report mismatch for {key}"
            )
    if type(report.get("errors")) is not list or report["errors"]:
        raise SourceStateError("strict Supreme Court validator report contains errors")
    if require_terminal_coverage:
        cursors = report.get("per_chamber_resume_cursors")
        if (
            report.get("lower_bound") != CANDIDATE_START_DATE
            or report.get("oldest_fully_completed_global_date_frontier")
            != CANDIDATE_START_DATE
            or not isinstance(cursors, dict)
            or set(cursors) != _SUPREME_OFFICIAL_CHAMBERS
            or any(cursor is not None for cursor in cursors.values())
            or report.get("unresolved_failure_count") != 0
        ):
            raise SourceStateError(
                "terminal Supreme Court run lacks exhausted-chamber, lower-bound, "
                "and zero-failure proof"
            )
    _manifest_payload_2, manifest_2 = _read_private_regular(
        manifest.path, label="Supreme Court partial manifest", capture=False
    )
    _journal_payload_2, journal_2 = _read_private_regular(
        journal.path, label="Supreme Court item journal", capture=False
    )
    if manifest_2 != manifest or journal_2 != journal:
        raise SourceStateError(
            "Supreme Court manifest or journal changed during strict validation"
        )
    return report


def _validate_crawl_contract(
    value: object,
    *,
    completion: Mapping[str, object],
    source: str,
    run_dir: Path,
    items: FileAttestation,
) -> str:
    """Independently verify the frozen evidence-crawl contract and durable bytes."""

    if not isinstance(value, dict):
        raise SourceStateError("completion.crawl_contract must be an object")
    _exact_keys(value, _CRAWL_CONTRACT_KEYS, label="completion.crawl_contract")
    if value["kind"] != "immutable_evidence_crawl_v1":
        raise SourceStateError("completion.crawl_contract.kind is invalid")
    for field in ("start_date", "end_date", "completed_at"):
        if value[field] != completion[field]:
            raise SourceStateError(f"completion.crawl_contract.{field} drifted")
    _canonical_path_value(
        value["artifact_root"],
        expected=CANDIDATE_ARTIFACT_ROOT,
        label="completion.crawl_contract.artifact_root",
    )
    revision = value["code_revision"]
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise SourceStateError(
            "completion.crawl_contract.code_revision must be an immutable hex revision"
        )
    _require_sha256(
        value["code_identity_sha256"],
        label="completion.crawl_contract.code_identity_sha256",
    )
    for field in (
        "http_cache_enabled",
        "cross_run_dedup_enabled",
        "shared_seen_sqlite_accessed",
    ):
        _require_bool(
            value[field],
            label=f"completion.crawl_contract.{field}",
            expected=False,
        )
    if value["terminal_status"] != completion["finish_reason"]:
        raise SourceStateError("completion.crawl_contract.terminal_status drifted")
    _require_bool(
        value["quality_passed"],
        label="completion.crawl_contract.quality_passed",
        expected=True,
    )
    _require_bool(
        value["feeds_durable"],
        label="completion.crawl_contract.feeds_durable",
        expected=True,
    )

    source_arguments = value["source_arguments"]
    if not isinstance(source_arguments, dict):
        raise SourceStateError(
            "completion.crawl_contract.source_arguments must be an object"
        )
    if source == "matsne":
        if source_arguments != {
            "doc_type": "all",
            "seed_file": None,
            "seed_ids_file": None,
            "seed_urls": None,
        }:
            raise SourceStateError(
                "Matsne evidence run is not full/default corpus mode"
            )
    elif source == "supremecourt":
        if set(source_arguments) != {
            "initial_window_days",
            "max_runtime_seconds",
            "parent_run_id",
        }:
            raise SourceStateError("Supreme Court source arguments are incomplete")
        initial_days = source_arguments["initial_window_days"]
        _require_int(
            initial_days,
            label="Supreme Court initial_window_days",
            expected=7,
        )
        _require_int(
            source_arguments["max_runtime_seconds"],
            label="Supreme Court max_runtime_seconds",
            expected=14_400,
        )
        parent = source_arguments["parent_run_id"]
        if parent != "none" and (
            not isinstance(parent, str)
            or _RUN_ID_RE.fullmatch(parent) is None
            or parent.lower() == "latest"
        ):
            raise SourceStateError("Supreme Court parent_run_id is unsafe")
    elif source_arguments:
        raise SourceStateError("ordinary evidence source arguments must be empty")

    limits = value["discovery_limits"]
    if not isinstance(limits, dict):
        raise SourceStateError(
            "completion.crawl_contract.discovery_limits must be an object"
        )
    if limits != _EXPECTED_DISCOVERY_LIMITS[source]:
        raise SourceStateError(
            f"{source} discovery limits do not match the frozen source contract"
        )
    for name, limit in limits.items():
        if not isinstance(name, str) or _CONTRACT_KEY_RE.fullmatch(name) is None:
            raise SourceStateError("crawl discovery-limit name is invalid")
        _require_int(limit, label=f"completion.crawl_contract.discovery_limits.{name}")

    duplicate_counts = value["within_run_duplicates"]
    if not isinstance(duplicate_counts, dict):
        raise SourceStateError(
            "completion.crawl_contract.within_run_duplicates must be an object"
        )
    _exact_keys(
        duplicate_counts,
        _WITHIN_RUN_DUPLICATE_KEYS,
        label="completion.crawl_contract.within_run_duplicates",
    )
    observed = _require_int(
        duplicate_counts["observed"],
        label="completion.crawl_contract.within_run_duplicates.observed",
    )
    unique = _require_int(
        duplicate_counts["unique"],
        label="completion.crawl_contract.within_run_duplicates.unique",
    )
    duplicates = _require_int(
        duplicate_counts["duplicates"],
        label="completion.crawl_contract.within_run_duplicates.duplicates",
    )
    missing = _require_int(
        duplicate_counts["missing_identity"],
        label="completion.crawl_contract.within_run_duplicates.missing_identity",
        expected=0,
    )
    reconciled = _require_bool(
        duplicate_counts["reconciled"],
        label="completion.crawl_contract.within_run_duplicates.reconciled",
        expected=True,
    )
    if not reconciled or missing or observed != unique + duplicates:
        raise SourceStateError("within-run duplicate counts do not reconcile exactly")

    failures = value["failure_signals"]
    if not isinstance(failures, dict):
        raise SourceStateError(
            "completion.crawl_contract.failure_signals must be an object"
        )
    _exact_keys(
        failures,
        _FAILURE_SIGNAL_KEYS,
        label="completion.crawl_contract.failure_signals",
    )
    expected_failures = {
        "quality_failures": completion["quality"]["quality_failures"],
        "spider_errors": completion["quality"]["spider_errors"],
        "spider_exceptions": completion["quality"]["spider_exceptions"],
        "item_errors": completion["quality"]["item_errors"],
        "feed_failures": completion["feed_outputs"]["failure_count"],
    }
    if failures != expected_failures or any(failures.values()):
        raise SourceStateError("completion.crawl_contract contains failure signals")

    outputs = value["durable_outputs"]
    if not isinstance(outputs, list):
        raise SourceStateError(
            "completion.crawl_contract.durable_outputs must be a list"
        )
    sort_key = lambda row: (  # noqa: E731 - adjacent canonical-order proof
        str(row.get("role", "")) if isinstance(row, dict) else "",
        str(row.get("path", "")) if isinstance(row, dict) else "",
    )
    if outputs != sorted(outputs, key=sort_key):
        raise SourceStateError(
            "completion.crawl_contract.durable_outputs is not sorted"
        )
    expected_roles = (
        {"identity_journal", "run"}
        if source != "supremecourt"
        else {"journal", "manifest", "run"}
    )
    observed_roles: set[str] = set()
    output_proofs: dict[str, FileAttestation] = {}
    for index, row in enumerate(outputs):
        label = f"completion.crawl_contract.durable_outputs[{index}]"
        if not isinstance(row, dict):
            raise SourceStateError(f"{label} must be an object")
        _exact_keys(row, _DURABLE_OUTPUT_KEYS, label=label)
        role = row["role"]
        if role not in expected_roles or role in observed_roles:
            raise SourceStateError(f"{label}.role is invalid or duplicated")
        observed_roles.add(role)
        expected_path = {
            "run": run_dir / "items.jsonl",
            "identity_journal": run_dir / "identity.journal.jsonl",
            "manifest": run_dir / "partial_manifest.json",
            "journal": run_dir / "items.journal.jsonl",
        }[role]
        _canonical_path_value(
            row["path"], expected=expected_path, label=f"{label}.path"
        )
        _payload, proof = _read_private_regular(
            expected_path,
            label=f"crawl-contract {role} output",
            capture=False,
            exact_mode=0o600,
        )
        _require_int(
            row["size_bytes"], label=f"{label}.size_bytes", expected=proof.size_bytes
        )
        if _require_sha256(row["sha256"], label=f"{label}.sha256") != proof.sha256:
            raise SourceStateError(f"{label}.sha256 does not match durable bytes")
        output_proofs[role] = proof
    if observed_roles != expected_roles:
        raise SourceStateError(
            "crawl contract does not bind the exact durable output set"
        )

    feed_row = completion["feed_outputs"]["files"][0]
    run_output = next(row for row in outputs if row["role"] == "run")
    expected_feed_projection = {
        "role": feed_row["role"],
        "path": feed_row["path"],
        "size_bytes": feed_row["size_bytes"],
        "sha256": feed_row["sha256"],
    }
    if run_output != expected_feed_projection:
        raise SourceStateError(
            "crawl-contract run feed identity drifted from feed_outputs"
        )
    if source != "supremecourt":
        _validate_ordinary_identity_evidence(
            source=source,
            items=items,
            journal=output_proofs["identity_journal"],
            declared=duplicate_counts,
        )
    return revision


def _validate_evidence_completion_record(
    completion: Mapping[str, object],
    *,
    source: str,
    run_id: str,
    run_dir: Path,
    items: FileAttestation,
    now: datetime | None,
) -> str:
    """Validate release-specific schema-v2 completion evidence without shared state."""

    _exact_keys(completion, _EVIDENCE_COMPLETION_KEYS, label="completion record")
    _require_int(
        completion["schema_version"],
        label="completion.schema_version",
        expected=EVIDENCE_COMPLETION_SCHEMA_VERSION,
    )
    if completion["source"] != source or completion["spider"] != source:
        raise SourceStateError(
            f"completion source identity mismatch for {source}/{run_id}"
        )
    if completion["outcome"] != "success":
        raise SourceStateError(
            f"completion outcome is not success for {source}/{run_id}"
        )
    _require_bool(
        completion["quality_passed"], label="completion.quality_passed", expected=True
    )
    _require_bool(
        completion["feeds_durable"], label="completion.feeds_durable", expected=True
    )
    _require_int(
        completion["failure_count"], label="completion.failure_count", expected=0
    )
    if (
        completion["start_date"] != CANDIDATE_START_DATE
        or completion["end_date"] != CANDIDATE_END_DATE
    ):
        raise SourceStateError(
            "evidence completion has the wrong frozen crawl interval"
        )
    _canonical_date(completion["start_date"], label="completion.start_date")
    _canonical_date(completion["end_date"], label="completion.end_date")
    started = _canonical_utc_second(
        completion["started_at"], label="completion.started_at"
    )
    completed = _canonical_utc_second(
        completion["completed_at"], label="completion.completed_at"
    )
    if completed < started:
        raise SourceStateError("completion.completed_at precedes completion.started_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if completed > current:
        raise SourceStateError(
            f"completion timestamp is in the future for {source}/{run_id}"
        )
    if run_dir.parents[2] != CANDIDATE_ARTIFACT_ROOT:
        raise SourceStateError(
            "evidence completion is outside the fixed candidate artifact root"
        )
    _canonical_path_value(
        completion["items_path"],
        expected=run_dir / "items.jsonl",
        label="completion.items_path",
    )
    if completion["latest_items_path"] is not None:
        raise SourceStateError("evidence completion must never bind latest_items_path")
    _canonical_path_value(
        completion["log_path"],
        expected=run_dir / "spider.log",
        label="completion.log_path",
    )
    _validate_feed_outputs(
        completion["feed_outputs"], run_dir=run_dir, latest_dir=None, items=items
    )
    _validate_quality(completion["quality"])

    source_validation = completion["source_validation"]
    if not isinstance(source_validation, dict):
        raise SourceStateError("completion.source_validation must be an object")
    finish_reason = completion["finish_reason"]
    if source == "supremecourt":
        if finish_reason not in {"closespider_timeout", "finished"}:
            raise SourceStateError(
                "Supreme Court evidence completion has an invalid finish reason"
            )
        _validate_supremecourt_source(
            source_validation,
            run_dir=run_dir,
            items=items,
            finish_reason=finish_reason,
            require_terminal_coverage=finish_reason == "finished",
        )
    else:
        if finish_reason != "finished":
            raise SourceStateError(
                "ordinary evidence completion requires finish_reason='finished'"
            )
        _exact_keys(
            source_validation,
            _GENERIC_VALIDATION_KEYS,
            label="completion.source_validation",
        )
        if source_validation["kind"] != "generic":
            raise SourceStateError(
                "completion.source_validation.kind must be 'generic'"
            )
        _require_bool(
            source_validation["passed"],
            label="completion.source_validation.passed",
            expected=True,
        )
    _validate_crawl_contract(
        completion["crawl_contract"],
        completion=completion,
        source=source,
        run_dir=run_dir,
        items=items,
    )
    return completion["completed_at"]  # type: ignore[return-value]


def validate_completion_record(
    completion: Mapping[str, object],
    *,
    source: str,
    run_id: str,
    run_dir: Path,
    items: FileAttestation,
    now: datetime | None = None,
) -> str:
    """Validate an ordinary or frozen-evidence completion attestation."""

    schema_version = completion.get("schema_version")
    if schema_version == EVIDENCE_COMPLETION_SCHEMA_VERSION:
        return _validate_evidence_completion_record(
            completion,
            source=source,
            run_id=run_id,
            run_dir=run_dir,
            items=items,
            now=now,
        )
    if schema_version != COMPLETION_SCHEMA_VERSION:
        raise SourceStateError(
            f"unsupported completion schema_version: {schema_version!r}"
        )
    _exact_keys(completion, _COMPLETION_KEYS, label="completion record")
    _require_int(
        completion["schema_version"],
        label="completion.schema_version",
        expected=COMPLETION_SCHEMA_VERSION,
    )
    if reason := _run_success_reason(completion, source, run_id):
        raise SourceStateError(
            f"completion record rejected for {source}/{run_id}: {reason}"
        )
    if completion["source"] != source or completion["spider"] != source:
        raise SourceStateError(
            f"completion source identity mismatch for {source}/{run_id}"
        )
    if completion["outcome"] != "success":
        raise SourceStateError(
            f"completion outcome is not success for {source}/{run_id}"
        )
    _require_bool(
        completion["quality_passed"], label="completion.quality_passed", expected=True
    )
    _require_bool(
        completion["feeds_durable"], label="completion.feeds_durable", expected=True
    )
    _require_int(
        completion["failure_count"], label="completion.failure_count", expected=0
    )
    start_date = _canonical_date(
        completion["start_date"], label="completion.start_date"
    )
    end_date = _canonical_date(completion["end_date"], label="completion.end_date")
    if start_date > end_date:
        raise SourceStateError("completion date window is reversed")
    started = _canonical_utc_second(
        completion["started_at"], label="completion.started_at"
    )
    completed = _canonical_utc_second(
        completion["completed_at"], label="completion.completed_at"
    )
    if completed < started:
        raise SourceStateError("completion.completed_at precedes completion.started_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if completed > current:
        raise SourceStateError(
            f"completion timestamp is in the future for {source}/{run_id}"
        )

    latest_dir = run_dir.parents[1] / "latest"
    _canonical_path_value(
        completion["items_path"],
        expected=run_dir / "items.jsonl",
        label="completion.items_path",
    )
    _canonical_path_value(
        completion["latest_items_path"],
        expected=latest_dir / "items.jsonl",
        label="completion.latest_items_path",
    )
    _canonical_path_value(
        completion["log_path"],
        expected=run_dir / "spider.log",
        label="completion.log_path",
    )
    _validate_feed_outputs(
        completion["feed_outputs"], run_dir=run_dir, latest_dir=latest_dir, items=items
    )
    _validate_quality(completion["quality"])

    source_validation = completion["source_validation"]
    if not isinstance(source_validation, dict):
        raise SourceStateError("completion.source_validation must be an object")
    if source == "supremecourt":
        if completion["finish_reason"] != "closespider_timeout":
            raise SourceStateError(
                "Supreme Court completion requires closespider_timeout"
            )
        _validate_supremecourt_source(
            source_validation,
            run_dir=run_dir,
            items=items,
            finish_reason="closespider_timeout",
        )
    else:
        if completion["finish_reason"] != "finished":
            raise SourceStateError(
                "ordinary completion requires finish_reason='finished'"
            )
        _exact_keys(
            source_validation,
            _GENERIC_VALIDATION_KEYS,
            label="completion.source_validation",
        )
        if source_validation["kind"] != "generic":
            raise SourceStateError(
                "completion.source_validation.kind must be 'generic'"
            )
        _require_bool(
            source_validation["passed"],
            label="completion.source_validation.passed",
            expected=True,
        )
    return completion["completed_at"]  # type: ignore[return-value]


def _validate_identity(source: object, run_id: object) -> tuple[str, str]:
    if source not in PRODUCTION_SOURCES:
        raise SourceStateError(f"unknown source: {source!r}")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise SourceStateError(f"unsafe run_id: {run_id!r}")
    if run_id.lower() == "latest":
        raise SourceStateError("'latest' is not an exact immutable run selection")
    return source, run_id  # type: ignore[return-value]


def validate_selected_run(
    artifacts_root: Path | str,
    source: object,
    run_id: object,
    *,
    expected_items_sha256: object | None = None,
    expected_completion_sha256: object | None = None,
    now: datetime | None = None,
) -> ValidatedRun:
    """Validate one exact run, including a second hash after semantic validation."""

    selected_source, selected_run_id = _validate_identity(source, run_id)
    root = _reject_symlink_components(artifacts_root)
    run_dir = _reject_symlink_components(
        root / selected_source / "runs" / selected_run_id
    )
    _require_private_directory(run_dir, label="selected run directory")
    _reject_terminal_recovery_guard(run_dir)
    candidate_payload, candidate_file, authorization_file = (
        _load_terminal_authorization(
            run_dir,
            source=selected_source,
            run_id=selected_run_id,
        )
    )
    items_payload, items = _read_private_regular(
        run_dir / "items.jsonl",
        label="selected items.jsonl",
        capture=False,
        exact_mode=0o600,
    )
    completion_payload, completion_file = _read_private_regular(
        run_dir / "run.json",
        label="selected completion record",
        max_bytes=_MAX_COMPLETION_BYTES,
        exact_mode=0o600,
    )
    del items_payload
    if (
        completion_payload != candidate_payload
        or completion_file.size_bytes != candidate_file.size_bytes
        or completion_file.sha256 != candidate_file.sha256
    ):
        raise SourceStateError(
            "selected run.json does not exactly match its authorized terminal "
            f"candidate: {selected_source}/{selected_run_id}"
        )
    if expected_items_sha256 is not None:
        expected = _require_sha256(expected_items_sha256, label="evidence.items_sha256")
        if items.sha256 != expected:
            raise SourceStateError(
                f"items.jsonl hash mismatch for {selected_source}/{selected_run_id}"
            )
    if expected_completion_sha256 is not None:
        expected = _require_sha256(
            expected_completion_sha256, label="evidence.completion_record_sha256"
        )
        if completion_file.sha256 != expected:
            raise SourceStateError(
                f"run.json hash mismatch for {selected_source}/{selected_run_id}"
            )
    completion = _strict_json_object(
        completion_payload,
        label=f"completion record {selected_source}/{selected_run_id}",
    )
    if completion_payload != canonical_json_bytes(completion):
        raise SourceStateError(
            f"completion record is not canonically serialized: "
            f"{selected_source}/{selected_run_id}"
        )
    completed_at = validate_completion_record(
        completion,
        source=selected_source,
        run_id=selected_run_id,
        run_dir=run_dir,
        items=items,
        now=now,
    )

    _items_payload_2, items_2 = _read_private_regular(
        items.path,
        label="selected items.jsonl",
        capture=False,
        exact_mode=0o600,
    )
    _completion_payload_2, completion_2 = _read_private_regular(
        completion_file.path,
        label="selected completion record",
        max_bytes=_MAX_COMPLETION_BYTES,
        capture=False,
        exact_mode=0o600,
    )
    candidate_payload_2, candidate_2, authorization_2 = _load_terminal_authorization(
        run_dir,
        source=selected_source,
        run_id=selected_run_id,
    )
    if items_2 != items:
        raise SourceStateError(
            f"selected items.jsonl changed during validation: {selected_source}/{selected_run_id}"
        )
    if completion_2 != completion_file:
        raise SourceStateError(
            f"completion record changed during validation: {selected_source}/{selected_run_id}"
        )
    if (
        candidate_payload_2 != candidate_payload
        or candidate_2 != candidate_file
        or authorization_2 != authorization_file
    ):
        raise SourceStateError(
            "terminal candidate or authorization changed during validation: "
            f"{selected_source}/{selected_run_id}"
        )
    _reject_terminal_recovery_guard(run_dir)
    return ValidatedRun(
        source=selected_source,
        run_id=selected_run_id,
        items=items,
        completion=completion_file,
        terminal_candidate=candidate_file,
        terminal_authorization=authorization_file,
        completed_at=completed_at,
        completion_schema_version=int(completion["schema_version"]),
    )


def parse_selection(value: str) -> tuple[str, str]:
    """Parse one CLI ``SOURCE:RUN_ID`` selection and reject aliases."""

    if not isinstance(value, str) or value.count(":") != 1:
        raise SourceStateError("--select must use exact SOURCE:RUN_ID syntax")
    source, run_id = value.split(":", 1)
    return _validate_identity(source, run_id)


def _validate_selections(
    selections: Iterable[tuple[object, object]],
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    covered: set[str] = set()
    for source, run_id in selections:
        identity = _validate_identity(source, run_id)
        if identity in seen:
            raise SourceStateError(
                f"duplicate run selection: {identity[0]}/{identity[1]}"
            )
        seen.add(identity)
        covered.add(identity[0])
        selected.append(identity)
    if not selected:
        raise SourceStateError(
            "at least one explicit --select SOURCE:RUN_ID is required"
        )
    missing = sorted(set(PRODUCTION_SOURCES) - covered)
    if missing:
        raise SourceStateError(
            f"source-state evidence must select all seven production sources; missing={missing}"
        )
    return sorted(selected)


def _rehash_validated(runs: Sequence[ValidatedRun]) -> None:
    for run in runs:
        _payload, items = _read_private_regular(
            run.items.path,
            label="selected items.jsonl",
            capture=False,
            exact_mode=0o600,
        )
        _payload, completion = _read_private_regular(
            run.completion.path,
            label="selected completion record",
            max_bytes=_MAX_COMPLETION_BYTES,
            capture=False,
            exact_mode=0o600,
        )
        candidate_payload, candidate = _read_private_regular(
            run.terminal_candidate.path,
            label="terminal candidate",
            max_bytes=_MAX_COMPLETION_BYTES,
            exact_mode=0o600,
        )
        _authorization_payload, authorization = _read_private_regular(
            run.terminal_authorization.path,
            label="terminal authorization",
            max_bytes=_MAX_COMPLETION_BYTES,
            exact_mode=0o600,
        )
        if items != run.items:
            raise SourceStateError(
                f"selected items.jsonl changed before publication: {run.source}/{run.run_id}"
            )
        if completion != run.completion:
            raise SourceStateError(
                f"completion record changed before publication: {run.source}/{run.run_id}"
            )
        if (
            candidate != run.terminal_candidate
            or candidate_payload
            != canonical_json_bytes(
                _strict_json_object(
                    candidate_payload,
                    label=f"terminal candidate {run.source}/{run.run_id}",
                )
            )
        ):
            raise SourceStateError(
                "terminal candidate changed before publication: "
                f"{run.source}/{run.run_id}"
            )
        if authorization != run.terminal_authorization:
            raise SourceStateError(
                "terminal authorization changed before publication: "
                f"{run.source}/{run.run_id}"
            )
        _reject_terminal_recovery_guard(run.completion.path.parent)


def _rename_noreplace(
    source_name: str,
    destination_name: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
    destination_display: Path,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SourceStateError("atomic renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            source_dir_fd,
            os.fsencode(source_name),
            destination_dir_fd,
            os.fsencode(destination_name),
            1,
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise SourceStateError(
            f"evidence destination already exists: {destination_display}"
        )
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise SourceStateError("atomic no-replace publication is unsupported")
    raise SourceStateError(
        f"cannot atomically publish evidence to {destination_display}: {os.strerror(error)}"
    )


def _open_publication_ancestor_nofollow(path: Path) -> int:
    """Open ``path`` while rejecting every uncontrolled ancestor directory."""

    absolute = _absolute_lexical(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        traversed = Path(absolute.anchor)
        for part in ("", *absolute.parts[1:]):
            if part:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
                traversed /= part
            info = os.fstat(descriptor)
            mode = stat.S_IMODE(info.st_mode)
            if mode & 0o022 and not mode & stat.S_ISVTX:
                raise SourceStateError(
                    "evidence ancestor is writable by other users without sticky "
                    f"protection: {traversed} mode={mode:04o}"
                )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_or_create_publication_parent(parent: Path) -> int:
    """Create missing parents one at a time and durably link every new directory."""

    parent = _reject_symlink_components(parent)
    missing: list[Path] = []
    cursor = parent
    while not os.path.lexists(cursor):
        missing.append(cursor)
        cursor = cursor.parent
    existing_descriptor = _open_publication_ancestor_nofollow(cursor)
    try:
        existing = os.fstat(existing_descriptor)
        mode = stat.S_IMODE(existing.st_mode)
        if mode & 0o022 and not mode & stat.S_ISVTX:
            raise SourceStateError(
                f"evidence ancestor is writable by other users: {cursor} mode={mode:04o}"
            )
        descriptor = existing_descriptor
        existing_descriptor = -1
        for directory in reversed(missing):
            try:
                os.mkdir(directory.name, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            os.fsync(descriptor)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(directory.name, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            child_info = os.fstat(descriptor)
            if stat.S_IMODE(child_info.st_mode) != 0o700:
                raise SourceStateError(
                    f"new evidence parent must have mode 0700: {directory}"
                )
        final = os.fstat(descriptor)
        final_mode = stat.S_IMODE(final.st_mode)
        if final.st_uid != os.geteuid():
            raise SourceStateError(
                f"evidence parent is not owned by the current user: {parent}"
            )
        if final_mode & 0o022 and not final_mode & stat.S_ISVTX:
            raise SourceStateError(
                f"evidence parent is writable by other users: {parent} mode={final_mode:04o}"
            )
        return descriptor
    except Exception:
        if existing_descriptor >= 0:
            os.close(existing_descriptor)
        elif "descriptor" in locals():
            os.close(descriptor)
        raise


def _fsync_publication_parent(parent_descriptor: int) -> None:
    os.fsync(parent_descriptor)


def _companion_paths(destination: Path) -> tuple[Path, Path]:
    leaf_digest = hashlib.sha256(os.fsencode(destination.name)).hexdigest()
    candidate = destination.with_name(f".source-state-{leaf_digest}.candidate")
    authorization = destination.with_name(
        f".source-state-{leaf_digest}.authorization.json"
    )
    return candidate, authorization


def source_state_evidence_companion_paths(
    destination: Path | str,
) -> tuple[Path, Path]:
    """Return the permanent candidate and authorization paths for an evidence path."""

    return _companion_paths(_absolute_lexical(destination))


def _authorization_value(
    destination: Path, candidate: Path, payload: bytes
) -> dict[str, object]:
    return {
        "schema_version": _AUTHORIZATION_SCHEMA_VERSION,
        "state": _AUTHORIZATION_STATE,
        "destination_leaf": destination.name,
        "candidate_leaf": candidate.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_private_regular_at(
    parent_descriptor: int,
    name: str,
    path: Path,
    *,
    label: str,
    max_bytes: int,
    capture: bool,
) -> tuple[bytes, FileAttestation]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise SourceStateError(f"cannot safely open {label}: {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SourceStateError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        mode = stat.S_IMODE(info.st_mode)
        if mode != 0o600:
            raise SourceStateError(
                f"{label} must have mode 0600; mode is {mode:04o}: {path}"
            )
        if info.st_uid != os.geteuid():
            raise SourceStateError(f"{label} is not owned by the current user: {path}")
        if info.st_size > max_bytes:
            raise SourceStateError(
                f"{label} exceeds the {max_bytes}-byte safety limit: {path}"
            )
        payload = bytearray() if capture else None
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if size + len(block) > max_bytes:
                raise SourceStateError(
                    f"{label} exceeds the {max_bytes}-byte safety limit while reading: "
                    f"{path}"
                )
            if payload is not None:
                payload.extend(block)
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        if (
            size != info.st_size
            or after.st_size != info.st_size
            or after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
            or after.st_mtime_ns != info.st_mtime_ns
        ):
            raise SourceStateError(f"{label} changed while reading: {path}")
        return (
            bytes(payload) if payload is not None else b"",
            FileAttestation(
                path,
                digest.hexdigest(),
                size,
                device=info.st_dev,
                inode=info.st_ino,
            ),
        )
    finally:
        os.close(descriptor)


def _load_authorized_bundle_at(
    parent_descriptor: int,
    destination: Path,
    *,
    expected_payload: bytes | None = None,
) -> _AuthorizedEvidenceBundle:
    candidate_path, authorization_path = _companion_paths(destination)
    evidence_payload, evidence = _read_private_regular_at(
        parent_descriptor,
        destination.name,
        destination,
        label="source-state evidence",
        max_bytes=_MAX_EVIDENCE_BYTES,
        capture=True,
    )
    _candidate_payload, candidate = _read_private_regular_at(
        parent_descriptor,
        candidate_path.name,
        candidate_path,
        label="source-state evidence candidate",
        max_bytes=_MAX_EVIDENCE_BYTES,
        capture=False,
    )
    authorization_payload, authorization = _read_private_regular_at(
        parent_descriptor,
        authorization_path.name,
        authorization_path,
        label="source-state evidence authorization",
        max_bytes=_MAX_AUTHORIZATION_BYTES,
        capture=True,
    )
    authorization_value = _strict_json_object(
        authorization_payload, label="source-state evidence authorization"
    )
    _exact_keys(
        authorization_value,
        {
            "schema_version",
            "state",
            "destination_leaf",
            "candidate_leaf",
            "size_bytes",
            "sha256",
        },
        label="source-state evidence authorization",
    )
    if authorization_payload != canonical_json_bytes(authorization_value):
        raise SourceStateError(
            "source-state evidence authorization is not canonically serialized"
        )
    _require_int(
        authorization_value["schema_version"],
        label="source-state evidence authorization.schema_version",
        expected=_AUTHORIZATION_SCHEMA_VERSION,
    )
    if authorization_value["state"] != _AUTHORIZATION_STATE:
        raise SourceStateError("source-state evidence authorization state is invalid")
    if authorization_value["destination_leaf"] != destination.name:
        raise SourceStateError(
            "source-state evidence authorization destination leaf mismatch"
        )
    if authorization_value["candidate_leaf"] != candidate_path.name:
        raise SourceStateError(
            "source-state evidence authorization candidate leaf mismatch"
        )
    size = _require_int(
        authorization_value["size_bytes"],
        label="source-state evidence authorization.size_bytes",
    )
    digest = _require_sha256(
        authorization_value["sha256"],
        label="source-state evidence authorization.sha256",
    )
    if (
        evidence.size_bytes != size
        or candidate.size_bytes != size
        or evidence.sha256 != digest
        or candidate.sha256 != digest
    ):
        raise SourceStateError(
            "source-state evidence, candidate, and authorization hash/size mismatch"
        )
    if evidence.device != candidate.device or evidence.inode != candidate.inode:
        raise SourceStateError(
            "source-state evidence is not the authorized candidate inode"
        )
    if expected_payload is not None and evidence_payload != expected_payload:
        raise SourceStateError(
            "authorized source-state evidence conflicts with requested exact bytes"
        )
    parent_info = os.fstat(parent_descriptor)
    return _AuthorizedEvidenceBundle(
        payload=evidence_payload,
        evidence=evidence,
        candidate=candidate,
        authorization=authorization,
        parent_device=parent_info.st_dev,
        parent_inode=parent_info.st_ino,
    )


def _load_authorized_bundle(
    destination: Path, *, expected_payload: bytes | None = None
) -> _AuthorizedEvidenceBundle:
    destination = _reject_symlink_components(destination)
    candidate, authorization = _companion_paths(destination)
    _reject_symlink_components(candidate)
    _reject_symlink_components(authorization)
    parent_descriptor = _open_publication_ancestor_nofollow(destination.parent)
    try:
        # Acceptance also stabilizes a previously authorized-but-pending hard link.  If
        # directory fsync is unavailable, the loader must reject until a later retry can
        # prove the output name durable.
        try:
            _fsync_publication_parent(parent_descriptor)
        except OSError as exc:
            raise SourceStateError(
                "source-state evidence output durability is still pending: "
                f"{destination}: {exc}"
            ) from exc
        return _load_authorized_bundle_at(
            parent_descriptor,
            destination,
            expected_payload=expected_payload,
        )
    finally:
        os.close(parent_descriptor)


class _PublicationConflict(SourceStateError):
    pass


def _name_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SourceStateError("short write while publishing source-state evidence")
        view = view[written:]


def _fsync_named_file_at(parent_descriptor: int, name: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_companion(
    parent_descriptor: int,
    permanent_path: Path,
    payload: bytes,
    *,
    label: str,
    max_bytes: int,
) -> FileAttestation:
    if _name_exists_at(parent_descriptor, permanent_path.name):
        try:
            existing_payload, existing = _read_private_regular_at(
                parent_descriptor,
                permanent_path.name,
                permanent_path,
                label=label,
                max_bytes=max_bytes,
                capture=True,
            )
        except SourceStateError as exc:
            raise _PublicationConflict(f"existing {label} is unsafe: {exc}") from exc
        if existing_payload != payload:
            raise _PublicationConflict(
                f"existing {label} conflicts with requested exact bytes"
            )
        _fsync_named_file_at(parent_descriptor, permanent_path.name)
        os.fsync(parent_descriptor)
        return existing

    temporary_name = f".{permanent_path.name}.{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    renamed = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        _write_descriptor(descriptor, payload)
        os.fsync(descriptor)
        _rename_noreplace(
            temporary_name,
            permanent_path.name,
            source_dir_fd=parent_descriptor,
            destination_dir_fd=parent_descriptor,
            destination_display=permanent_path,
        )
        renamed = True
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if not renamed:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
    existing_payload, existing = _read_private_regular_at(
        parent_descriptor,
        permanent_path.name,
        permanent_path,
        label=label,
        max_bytes=max_bytes,
        capture=True,
    )
    if existing_payload != payload:
        raise SourceStateError(f"new {label} differs from its fsynced temporary")
    return existing


def _authorization_is_exact_at(
    parent_descriptor: int,
    authorization_path: Path,
    authorization_payload: bytes,
) -> bool:
    if not _name_exists_at(parent_descriptor, authorization_path.name):
        return False
    try:
        observed, _attestation = _read_private_regular_at(
            parent_descriptor,
            authorization_path.name,
            authorization_path,
            label="source-state evidence authorization",
            max_bytes=_MAX_AUTHORIZATION_BYTES,
            capture=True,
        )
    except SourceStateError:
        return False
    return observed == authorization_payload


def _materialize_authorized_output(
    parent_descriptor: int,
    destination: Path,
    candidate_path: Path,
    payload: bytes,
) -> None:
    if not _name_exists_at(parent_descriptor, destination.name):
        try:
            os.link(
                candidate_path.name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
    try:
        _load_authorized_bundle_at(
            parent_descriptor, destination, expected_payload=payload
        )
    except SourceStateError as exc:
        raise _PublicationConflict(
            f"evidence destination or WAL conflicts with authorization: {exc}"
        ) from exc
    _fsync_publication_parent(parent_descriptor)
    held_info = os.fstat(parent_descriptor)
    lexical = _load_authorized_bundle(destination, expected_payload=payload)
    if (
        lexical.parent_device != held_info.st_dev
        or lexical.parent_inode != held_info.st_ino
    ):
        raise SourceStateError("evidence parent changed during authorized publication")


def _publish_create_only(destination: Path, payload: bytes) -> None:
    destination = _reject_symlink_components(destination)
    parent = destination.parent
    candidate_path, authorization_path = _companion_paths(destination)
    authorization_payload = canonical_json_bytes(
        _authorization_value(destination, candidate_path, payload)
    )
    parent_descriptor = _open_or_create_publication_parent(parent)
    try:
        output_exists = _name_exists_at(parent_descriptor, destination.name)
        candidate_exists = _name_exists_at(parent_descriptor, candidate_path.name)
        authorization_exists = _name_exists_at(
            parent_descriptor, authorization_path.name
        )
        if output_exists:
            raise _PublicationConflict(
                f"evidence destination already exists: {destination}"
            )
        if authorization_exists and not candidate_exists:
            raise _PublicationConflict(
                "source-state authorization exists without its permanent candidate"
            )

        _ensure_durable_companion(
            parent_descriptor,
            candidate_path,
            payload,
            label="source-state evidence candidate",
            max_bytes=_MAX_EVIDENCE_BYTES,
        )
        try:
            _ensure_durable_companion(
                parent_descriptor,
                authorization_path,
                authorization_payload,
                label="source-state evidence authorization",
                max_bytes=_MAX_AUTHORIZATION_BYTES,
            )
        except _PublicationConflict:
            raise
        except BaseException as first_error:
            try:
                _ensure_durable_companion(
                    parent_descriptor,
                    authorization_path,
                    authorization_payload,
                    label="source-state evidence authorization",
                    max_bytes=_MAX_AUTHORIZATION_BYTES,
                )
            except BaseException as recovery_error:
                if _authorization_is_exact_at(
                    parent_descriptor, authorization_path, authorization_payload
                ):
                    raise SourceStatePublicationPending(
                        "source-state authorization exists, but durability is pending"
                    ) from first_error
                raise first_error from recovery_error

        try:
            _materialize_authorized_output(
                parent_descriptor,
                destination,
                candidate_path,
                payload,
            )
        except _PublicationConflict:
            raise
        except BaseException as first_error:
            try:
                _materialize_authorized_output(
                    parent_descriptor,
                    destination,
                    candidate_path,
                    payload,
                )
            except _PublicationConflict:
                raise
            except BaseException:
                raise SourceStatePublicationPending(
                    "source-state output is authorized, but stabilization is pending"
                ) from first_error
    finally:
        try:
            os.close(parent_descriptor)
        except OSError:
            pass


def _completion_value(run: ValidatedRun) -> dict[str, object]:
    payload, proof = _read_private_regular(
        run.completion.path,
        label=f"selected completion record {run.source}/{run.run_id}",
        max_bytes=_MAX_COMPLETION_BYTES,
        exact_mode=0o600,
    )
    if proof != run.completion:
        raise SourceStateError(f"completion record changed: {run.source}/{run.run_id}")
    value = _strict_json_object(
        payload, label=f"selected completion record {run.source}/{run.run_id}"
    )
    if payload != canonical_json_bytes(value):
        raise SourceStateError(
            f"completion record is not canonical: {run.source}/{run.run_id}"
        )
    return value


def _validate_candidate_selections(
    selections: Iterable[tuple[object, object]],
) -> list[tuple[str, str]]:
    identities = _validate_selections(selections)
    if len(identities) != len(PRODUCTION_SOURCES):
        raise SourceStateError(
            "candidate ledger must contain exactly seven snapshot input rows"
        )
    counts = {source: 0 for source in PRODUCTION_SOURCES}
    for source, _run_id in identities:
        counts[source] += 1
    if any(count != 1 for count in counts.values()):
        raise SourceStateError(
            "candidate ledger must select exactly one final run per source"
        )
    return identities


def _validate_ordinary_run_directories(
    root: Path, runs: Sequence[ValidatedRun]
) -> None:
    """Require each ordinary source root to contain only its selected first run."""

    selected = {
        run.source: run.run_id
        for run in runs
        if run.source in _ORDINARY_IDENTITY_FIELDS
    }
    if set(selected) != set(_ORDINARY_IDENTITY_FIELDS):
        raise SourceStateError("candidate ordinary run set is incomplete")
    for source in sorted(_ORDINARY_IDENTITY_FIELDS):
        runs_dir = root / source / "runs"
        _require_private_directory(
            runs_dir, label=f"{source} ordinary evidence runs directory"
        )
        descriptor = _open_directory_nofollow(
            runs_dir, label=f"{source} ordinary evidence runs directory"
        )
        try:
            entries = sorted(os.listdir(descriptor))
            if entries != [selected[source]]:
                raise SourceStateError(
                    f"{source} must have exactly one ordinary evidence run; "
                    f"selected={selected[source]!r}, entries={entries[:8]!r}"
                )
            info = os.stat(entries[0], dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != os.geteuid()
            ):
                raise SourceStateError(
                    f"{source} ordinary evidence run directory is not private"
                )
        finally:
            os.close(descriptor)


def _parse_supreme_chain_selection(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise SourceStateError(
            "Supreme Court coverage chain IDs must be non-empty strings"
        )
    if ":" in value:
        source, run_id = parse_selection(value)
        if source != "supremecourt":
            raise SourceStateError("coverage chain may contain only Supreme Court runs")
        return source, run_id
    return _validate_identity("supremecourt", value)


def _validate_candidate_run_set(
    runs: Sequence[ValidatedRun], supreme_chain: Sequence[ValidatedRun]
) -> tuple[str, str]:
    if len(runs) != len(PRODUCTION_SOURCES):
        raise SourceStateError("candidate run set must contain exactly seven rows")
    if not supreme_chain:
        raise SourceStateError(
            "candidate ledger requires a Supreme Court coverage chain"
        )
    if any(
        run.completion_schema_version != EVIDENCE_COMPLETION_SCHEMA_VERSION
        for run in runs
    ):
        raise SourceStateError("candidate snapshot inputs require completion schema v2")
    if any(
        run.source != "supremecourt"
        or run.completion_schema_version != EVIDENCE_COMPLETION_SCHEMA_VERSION
        for run in supreme_chain
    ):
        raise SourceStateError("Supreme Court chain contains a non-evidence run")
    chain_ids = [run.run_id for run in supreme_chain]
    if len(chain_ids) != len(set(chain_ids)):
        raise SourceStateError("Supreme Court coverage chain repeats a run")
    final_supreme = next(run for run in runs if run.source == "supremecourt")
    if supreme_chain[-1].run_id != final_supreme.run_id:
        raise SourceStateError(
            "the seven-row ledger must use the final cumulative Supreme Court chain feed"
        )

    revisions: set[str] = set()
    code_identities: set[str] = set()
    for run in runs:
        completion = _completion_value(run)
        contract = completion["crawl_contract"]
        assert isinstance(contract, dict)
        revision = contract["code_revision"]
        assert isinstance(revision, str)
        revisions.add(revision)
        code_identity = contract["code_identity_sha256"]
        assert isinstance(code_identity, str)
        code_identities.add(code_identity)
    for index, run in enumerate(supreme_chain):
        completion = _completion_value(run)
        contract = completion["crawl_contract"]
        assert isinstance(contract, dict)
        chain_revision = contract["code_revision"]
        assert isinstance(chain_revision, str)
        revisions.add(chain_revision)
        chain_code_identity = contract["code_identity_sha256"]
        assert isinstance(chain_code_identity, str)
        code_identities.add(chain_code_identity)
        arguments = contract["source_arguments"]
        assert isinstance(arguments, dict)
        expected_parent = "none" if index == 0 else supreme_chain[index - 1].run_id
        if arguments["parent_run_id"] != expected_parent:
            raise SourceStateError(
                "Supreme Court coverage chain is not an explicit contiguous parent chain"
            )
        if (
            index < len(supreme_chain) - 1
            and completion["finish_reason"] != "closespider_timeout"
        ):
            raise SourceStateError(
                "non-terminal Supreme Court coverage runs require the strict timeout close"
            )
        source_validation = completion["source_validation"]
        assert isinstance(source_validation, dict)
        report = _validate_supremecourt_source(
            source_validation,
            run_dir=run.items.path.parent,
            items=run.items,
            finish_reason=str(completion["finish_reason"]),
            require_terminal_coverage=index == len(supreme_chain) - 1,
        )
        resume_parent = report.get("resume_parent")
        if index == 0:
            if resume_parent is not None:
                raise SourceStateError(
                    "first Supreme Court coverage run is not parent-free"
                )
        elif (
            not isinstance(resume_parent, dict)
            or resume_parent.get("run_id") != expected_parent
        ):
            raise SourceStateError(
                "Supreme Court validator parent proof does not match the ordered chain"
            )
    if len(revisions) != 1:
        raise SourceStateError("candidate crawl code revisions are not identical")
    if len(code_identities) != 1:
        raise SourceStateError("candidate crawl code identities are not identical")
    return next(iter(revisions)), next(iter(code_identities))


def build_candidate_source_state_evidence(
    artifacts_root: Path | str,
    output_path: Path | str,
    selections: Iterable[tuple[object, object] | str],
    supreme_coverage_chain: Iterable[str],
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create the frozen seven-row ledger and its explicit Supreme coverage chain."""

    root = _absolute_lexical(artifacts_root)
    if root != CANDIDATE_ARTIFACT_ROOT:
        raise SourceStateError(
            "candidate evidence must be read from the fixed release-specific artifact root"
        )
    parsed = [
        parse_selection(selection) if isinstance(selection, str) else selection
        for selection in selections
    ]
    identities = _validate_candidate_selections(parsed)
    runs = [
        validate_selected_run(root, source, run_id, now=now)
        for source, run_id in identities
    ]
    _validate_ordinary_run_directories(root, runs)
    chain_identities = [
        _parse_supreme_chain_selection(value) for value in supreme_coverage_chain
    ]
    chain_runs_by_id = {run.run_id: run for run in runs if run.source == "supremecourt"}
    supreme_chain: list[ValidatedRun] = []
    for source, run_id in chain_identities:
        run = chain_runs_by_id.get(run_id)
        if run is None:
            run = validate_selected_run(root, source, run_id, now=now)
        supreme_chain.append(run)
    code_revision, code_identity_sha256 = _validate_candidate_run_set(
        runs, supreme_chain
    )
    evidence: dict[str, object] = {
        "schema_version": CANDIDATE_SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
        "snapshot_id": CANDIDATE_SNAPSHOT_ID,
        "crawl_interval": {
            "start_date": CANDIDATE_START_DATE,
            "end_date": CANDIDATE_END_DATE,
        },
        "artifact_root": os.fspath(CANDIDATE_ARTIFACT_ROOT),
        "code_revision": code_revision,
        "code_identity_sha256": code_identity_sha256,
        "runs": [run.evidence_row() for run in runs],
        "supremecourt_coverage_chain": [run.evidence_row() for run in supreme_chain],
    }
    payload = canonical_json_bytes(evidence)
    _rehash_validated(tuple(runs) + tuple(supreme_chain))
    _validate_ordinary_run_directories(root, runs)
    _publish_create_only(_absolute_lexical(output_path), payload)
    return evidence


def build_source_state_evidence(
    artifacts_root: Path | str,
    output_path: Path | str,
    selections: Iterable[tuple[object, object] | str],
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate explicit exact runs and publish a deterministic schema-v1 ledger."""

    parsed: list[tuple[object, object]] = []
    for selection in selections:
        parsed.append(
            parse_selection(selection) if isinstance(selection, str) else selection
        )
    identities = _validate_selections(parsed)
    runs = [
        validate_selected_run(artifacts_root, source, run_id, now=now)
        for source, run_id in identities
    ]
    if any(run.completion_schema_version != COMPLETION_SCHEMA_VERSION for run in runs):
        raise SourceStateError(
            "schema-v2 evidence runs require build_candidate_source_state_evidence()"
        )
    evidence: dict[str, object] = {
        "schema_version": SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
        "runs": [run.evidence_row() for run in runs],
    }
    payload = canonical_json_bytes(evidence)
    _rehash_validated(runs)
    _publish_create_only(_absolute_lexical(output_path), payload)
    return evidence


def _validate_evidence_row(
    raw: object, *, label: str, required_source: str | None = None
) -> tuple[tuple[str, str], str, str]:
    if not isinstance(raw, dict):
        raise SourceStateError(f"{label} must be an object")
    _exact_keys(
        raw,
        {"source", "run_id", "items_sha256", "completion_record_sha256"},
        label=label,
    )
    identity = _validate_identity(raw["source"], raw["run_id"])
    if required_source is not None and identity[0] != required_source:
        raise SourceStateError(f"{label} must select only {required_source}")
    items_sha = _require_sha256(raw["items_sha256"], label=f"{label}.items_sha256")
    completion_sha = _require_sha256(
        raw["completion_record_sha256"],
        label=f"{label}.completion_record_sha256",
    )
    return identity, items_sha, completion_sha


def _revalidate_evidence_bundle(
    bundle: _AuthorizedEvidenceBundle, evidence_file: FileAttestation
) -> None:
    bundle_2 = _load_authorized_bundle(evidence_file.path)
    if (
        bundle_2.evidence != bundle.evidence
        or bundle_2.candidate != bundle.candidate
        or bundle_2.authorization != bundle.authorization
        or bundle_2.parent_device != bundle.parent_device
        or bundle_2.parent_inode != bundle.parent_inode
    ):
        raise SourceStateError(
            "source-state evidence or its authorization changed while validating runs"
        )


def _load_candidate_source_state_evidence(
    artifacts_root: Path | str,
    evidence: Mapping[str, object],
    *,
    bundle: _AuthorizedEvidenceBundle,
    now: datetime | None,
) -> LoadedEvidence:
    _exact_keys(
        evidence,
        {
            "schema_version",
            "snapshot_id",
            "crawl_interval",
            "artifact_root",
            "code_revision",
            "code_identity_sha256",
            "runs",
            "supremecourt_coverage_chain",
        },
        label="candidate source-state evidence",
    )
    _require_int(
        evidence["schema_version"],
        label="candidate source-state evidence.schema_version",
        expected=CANDIDATE_SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
    )
    if evidence["snapshot_id"] != CANDIDATE_SNAPSHOT_ID:
        raise SourceStateError(
            "candidate source-state evidence snapshot identity drifted"
        )
    interval = evidence["crawl_interval"]
    if not isinstance(interval, dict):
        raise SourceStateError("candidate crawl_interval must be an object")
    _exact_keys(interval, {"start_date", "end_date"}, label="candidate crawl_interval")
    if interval != {
        "start_date": CANDIDATE_START_DATE,
        "end_date": CANDIDATE_END_DATE,
    }:
        raise SourceStateError("candidate source-state evidence crawl interval drifted")
    root = _absolute_lexical(artifacts_root)
    if root != CANDIDATE_ARTIFACT_ROOT:
        raise SourceStateError(
            "candidate source-state loader received the wrong artifact root"
        )
    _canonical_path_value(
        evidence["artifact_root"],
        expected=root,
        label="candidate source-state evidence.artifact_root",
    )
    declared_revision = evidence["code_revision"]
    if (
        not isinstance(declared_revision, str)
        or _REVISION_RE.fullmatch(declared_revision) is None
    ):
        raise SourceStateError(
            "candidate source-state evidence.code_revision is invalid"
        )
    declared_code_identity = _require_sha256(
        evidence["code_identity_sha256"],
        label="candidate source-state evidence.code_identity_sha256",
    )

    rows = evidence["runs"]
    if not isinstance(rows, list):
        raise SourceStateError("candidate source-state evidence.runs must be a list")
    parsed_rows = [
        _validate_evidence_row(raw, label=f"candidate runs[{index}]")
        for index, raw in enumerate(rows)
    ]
    identities = _validate_candidate_selections(
        identity for identity, _a, _b in parsed_rows
    )
    if [identity for identity, _a, _b in parsed_rows] != identities:
        raise SourceStateError("candidate source-state runs are not canonically sorted")
    runs = [
        validate_selected_run(
            root,
            source,
            run_id,
            expected_items_sha256=items_sha,
            expected_completion_sha256=completion_sha,
            now=now,
        )
        for (source, run_id), items_sha, completion_sha in parsed_rows
    ]
    _validate_ordinary_run_directories(root, runs)

    chain_rows = evidence["supremecourt_coverage_chain"]
    if not isinstance(chain_rows, list) or not chain_rows:
        raise SourceStateError(
            "candidate Supreme Court coverage chain must be non-empty"
        )
    parsed_chain = [
        _validate_evidence_row(
            raw,
            label=f"candidate supremecourt_coverage_chain[{index}]",
            required_source="supremecourt",
        )
        for index, raw in enumerate(chain_rows)
    ]
    selected_by_id = {run.run_id: run for run in runs if run.source == "supremecourt"}
    supreme_chain: list[ValidatedRun] = []
    for (source, run_id), items_sha, completion_sha in parsed_chain:
        run = selected_by_id.get(run_id)
        if run is not None:
            if run.items.sha256 != items_sha or run.completion.sha256 != completion_sha:
                raise SourceStateError(
                    "final Supreme Court row has inconsistent hashes"
                )
        else:
            run = validate_selected_run(
                root,
                source,
                run_id,
                expected_items_sha256=items_sha,
                expected_completion_sha256=completion_sha,
                now=now,
            )
        supreme_chain.append(run)
    observed_revision, observed_code_identity = _validate_candidate_run_set(
        runs, supreme_chain
    )
    if observed_revision != declared_revision:
        raise SourceStateError(
            "candidate evidence code revision drifted from completions"
        )
    if observed_code_identity != declared_code_identity:
        raise SourceStateError(
            "candidate evidence code identity drifted from completions"
        )
    _rehash_validated(tuple(runs) + tuple(supreme_chain))
    _validate_ordinary_run_directories(root, runs)
    _revalidate_evidence_bundle(bundle, bundle.evidence)
    return LoadedEvidence(
        runs=tuple(runs),
        evidence=bundle.evidence,
        schema_version=CANDIDATE_SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
        supreme_coverage_chain=tuple(supreme_chain),
    )


def load_source_state_evidence(
    artifacts_root: Path | str,
    evidence_path: Path | str,
    *,
    now: datetime | None = None,
) -> LoadedEvidence:
    """Strictly load a reviewed ledger and revalidate every selected run."""

    path = _reject_symlink_components(evidence_path)
    bundle = _load_authorized_bundle(path)
    payload = bundle.payload
    evidence_file = bundle.evidence
    evidence = _strict_json_object(payload, label="source-state evidence")
    if payload != canonical_json_bytes(evidence):
        raise SourceStateError("source-state evidence is not canonically serialized")
    if evidence.get("schema_version") == CANDIDATE_SOURCE_STATE_EVIDENCE_SCHEMA_VERSION:
        return _load_candidate_source_state_evidence(
            artifacts_root,
            evidence,
            bundle=bundle,
            now=now,
        )
    _exact_keys(
        evidence,
        {"schema_version", "runs"},
        label="source-state evidence",
    )
    try:
        _require_int(
            evidence["schema_version"],
            label="source-state evidence.schema_version",
            expected=SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
        )
    except SourceStateError as exc:
        raise SourceStateError(
            "unsupported source-state evidence schema_version: "
            f"{evidence['schema_version']!r}"
        ) from exc
    rows = evidence["runs"]
    if not isinstance(rows, list) or not rows:
        raise SourceStateError("source-state evidence runs must be a non-empty list")
    selections: list[tuple[object, object]] = []
    normalized_rows: list[dict[str, object]] = []
    expected_by_identity: dict[tuple[str, str], tuple[object, object]] = {}
    for index, raw in enumerate(rows):
        label = f"source-state evidence runs[{index}]"
        if not isinstance(raw, dict):
            raise SourceStateError(f"{label} must be an object")
        _exact_keys(
            raw,
            {"source", "run_id", "items_sha256", "completion_record_sha256"},
            label=label,
        )
        identity = _validate_identity(raw["source"], raw["run_id"])
        _require_sha256(raw["items_sha256"], label=f"{label}.items_sha256")
        _require_sha256(
            raw["completion_record_sha256"],
            label=f"{label}.completion_record_sha256",
        )
        selections.append(identity)
        normalized_rows.append(raw)
        expected_by_identity[identity] = (
            raw["items_sha256"],
            raw["completion_record_sha256"],
        )
    identities = _validate_selections(selections)
    if [(row["source"], row["run_id"]) for row in normalized_rows] != identities:
        raise SourceStateError(
            "source-state evidence runs must be sorted by source and run_id"
        )
    runs = []
    for source, run_id in identities:
        items_sha, completion_sha = expected_by_identity[(source, run_id)]
        runs.append(
            validate_selected_run(
                artifacts_root,
                source,
                run_id,
                expected_items_sha256=items_sha,
                expected_completion_sha256=completion_sha,
                now=now,
            )
        )
    if any(run.completion_schema_version != COMPLETION_SCHEMA_VERSION for run in runs):
        raise SourceStateError(
            "schema-v1 source-state evidence cannot contain schema-v2 crawl completions"
        )
    _rehash_validated(runs)
    bundle_2 = _load_authorized_bundle(evidence_file.path)
    if (
        bundle_2.evidence != bundle.evidence
        or bundle_2.candidate != bundle.candidate
        or bundle_2.authorization != bundle.authorization
        or bundle_2.parent_device != bundle.parent_device
        or bundle_2.parent_inode != bundle.parent_inode
    ):
        raise SourceStateError(
            "source-state evidence or its authorization changed while validating runs"
        )
    return LoadedEvidence(tuple(runs), evidence_file)


__all__ = [
    "CANDIDATE_ARTIFACT_ROOT",
    "CANDIDATE_END_DATE",
    "CANDIDATE_SNAPSHOT_ID",
    "CANDIDATE_SOURCE_STATE_EVIDENCE_SCHEMA_VERSION",
    "CANDIDATE_START_DATE",
    "COMPLETION_SCHEMA_VERSION",
    "EVIDENCE_COMPLETION_SCHEMA_VERSION",
    "PRODUCTION_SOURCES",
    "SOURCE_STATE_EVIDENCE_SCHEMA_VERSION",
    "TERMINAL_RECOVERY_GUARD_FILENAME",
    "FileAttestation",
    "LoadedEvidence",
    "SourceStateError",
    "SourceStatePublicationPending",
    "ValidatedRun",
    "build_candidate_source_state_evidence",
    "build_source_state_evidence",
    "canonical_json_bytes",
    "iter_attested_lines",
    "load_source_state_evidence",
    "parse_selection",
    "source_state_evidence_companion_paths",
    "validate_completion_record",
    "validate_selected_run",
]
