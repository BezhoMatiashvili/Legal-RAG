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
COMPLETION_SCHEMA_VERSION = 1
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

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
    """A schema-v1 successful run bound to exact item and completion bytes."""

    source: str
    run_id: str
    items: FileAttestation
    completion: FileAttestation
    terminal_candidate: FileAttestation
    terminal_authorization: FileAttestation
    completed_at: str

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
            raise SourceStateError(f"cannot inspect path component {cursor}: {exc}") from exc
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
        raise SourceStateError(f"cannot open filesystem root for {label}: {exc}") from exc
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
            raise SourceStateError(f"{label} must be a regular non-symlink file: {path}")
        mode = stat.S_IMODE(info.st_mode)
        if info.st_uid != os.geteuid():
            raise SourceStateError(
                f"{label} must be owned by the current user: {path}"
            )
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
        if parsed.netloc not in {"", "localhost"} or parsed.params or parsed.query or parsed.fragment:
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
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_supremecourt_partial.py"
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
    latest_dir: Path,
    items: FileAttestation,
) -> None:
    if not isinstance(value, dict):
        raise SourceStateError("completion.feed_outputs must be an object")
    _exact_keys(value, _FEED_OUTPUT_KEYS, label="completion.feed_outputs")
    _require_bool(value["durable"], label="completion.feed_outputs.durable", expected=True)
    _require_int(value["configured_count"], label="completion.feed_outputs.configured_count", expected=2)
    _require_int(value["success_count"], label="completion.feed_outputs.success_count", expected=2)
    _require_int(value["failure_count"], label="completion.feed_outputs.failure_count", expected=0)
    files = value["files"]
    if not isinstance(files, list) or len(files) != 2:
        raise SourceStateError("completion.feed_outputs.files must contain exactly two rows")
    def sort_key(row: object) -> tuple[str, str, str]:
        if not isinstance(row, dict):
            return ("", "", "")
        return (
            str(row.get("role", "")),
            str(row.get("configured_uri", "")),
            str(row.get("path", "")),
        )
    if files != sorted(files, key=sort_key):
        raise SourceStateError("completion.feed_outputs.files must be deterministically sorted")
    expected_paths = {
        "run": run_dir / "items.jsonl",
        "latest": latest_dir / "items.jsonl",
    }
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
        _canonical_path_value(raw["path"], expected=expected_path, label=f"{label}.path")
        if _local_configured_uri(raw["configured_uri"], label=f"{label}.configured_uri") != expected_path:
            raise SourceStateError(f"{label}.configured_uri does not match its canonical path")
        _require_int(raw["size_bytes"], label=f"{label}.size_bytes", expected=items.size_bytes)
        digest = _require_sha256(raw["sha256"], label=f"{label}.sha256")
        if digest != items.sha256:
            raise SourceStateError(f"{label}.sha256 does not match selected items.jsonl")


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
        raise SourceStateError("completion.quality pagination reconcilers were not all finalized")


def _validate_supremecourt_source(
    value: Mapping[str, object], *, run_dir: Path, items: FileAttestation
) -> None:
    _exact_keys(value, _SUPREMECOURT_VALIDATION_KEYS, label="completion.source_validation")
    if value["kind"] != "supremecourt_partial_v1":
        raise SourceStateError("completion.source_validation.kind is invalid for supremecourt")
    _require_bool(value["passed"], label="completion.source_validation.passed", expected=True)
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
    if value["finish_reason"] != "closespider_timeout":
        raise SourceStateError("completion.source_validation.finish_reason is not strict timeout")
    if _require_sha256(
        value["items_sha256"], label="completion.source_validation.items_sha256"
    ) != items.sha256:
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
    if _require_sha256(
        value["manifest_sha256"], label="completion.source_validation.manifest_sha256"
    ) != manifest.sha256:
        raise SourceStateError("completion.source_validation.manifest_sha256 mismatch")
    if _require_sha256(
        value["journal_sha256"], label="completion.source_validation.journal_sha256"
    ) != journal.sha256:
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
        raise SourceStateError(f"strict Supreme Court validation failed: {exc}") from exc
    if not isinstance(report, dict):
        raise SourceStateError("strict Supreme Court validator returned a malformed report")
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


def validate_completion_record(
    completion: Mapping[str, object],
    *,
    source: str,
    run_id: str,
    run_dir: Path,
    items: FileAttestation,
    now: datetime | None = None,
) -> str:
    """Validate exact completion schema v1 and return canonical ``completed_at``."""

    _exact_keys(completion, _COMPLETION_KEYS, label="completion record")
    try:
        _require_int(
            completion["schema_version"],
            label="completion.schema_version",
            expected=COMPLETION_SCHEMA_VERSION,
        )
    except SourceStateError as exc:
        raise SourceStateError(
            f"unsupported completion schema_version: {completion['schema_version']!r}"
        ) from exc
    if reason := _run_success_reason(completion, source, run_id):
        raise SourceStateError(
            f"completion record rejected for {source}/{run_id}: {reason}"
        )
    if completion["source"] != source or completion["spider"] != source:
        raise SourceStateError(f"completion source identity mismatch for {source}/{run_id}")
    if completion["outcome"] != "success":
        raise SourceStateError(f"completion outcome is not success for {source}/{run_id}")
    _require_bool(completion["quality_passed"], label="completion.quality_passed", expected=True)
    _require_bool(completion["feeds_durable"], label="completion.feeds_durable", expected=True)
    _require_int(completion["failure_count"], label="completion.failure_count", expected=0)
    start_date = _canonical_date(completion["start_date"], label="completion.start_date")
    end_date = _canonical_date(completion["end_date"], label="completion.end_date")
    if start_date > end_date:
        raise SourceStateError("completion date window is reversed")
    started = _canonical_utc_second(completion["started_at"], label="completion.started_at")
    completed = _canonical_utc_second(completion["completed_at"], label="completion.completed_at")
    if completed < started:
        raise SourceStateError("completion.completed_at precedes completion.started_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if completed > current:
        raise SourceStateError(f"completion timestamp is in the future for {source}/{run_id}")

    latest_dir = run_dir.parents[1] / "latest"
    _canonical_path_value(
        completion["items_path"], expected=run_dir / "items.jsonl", label="completion.items_path"
    )
    _canonical_path_value(
        completion["latest_items_path"],
        expected=latest_dir / "items.jsonl",
        label="completion.latest_items_path",
    )
    _canonical_path_value(
        completion["log_path"], expected=run_dir / "spider.log", label="completion.log_path"
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
            raise SourceStateError("Supreme Court completion requires closespider_timeout")
        _validate_supremecourt_source(source_validation, run_dir=run_dir, items=items)
    else:
        if completion["finish_reason"] != "finished":
            raise SourceStateError("ordinary completion requires finish_reason='finished'")
        _exact_keys(
            source_validation,
            _GENERIC_VALIDATION_KEYS,
            label="completion.source_validation",
        )
        if source_validation["kind"] != "generic":
            raise SourceStateError("completion.source_validation.kind must be 'generic'")
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
    candidate_payload_2, candidate_2, authorization_2 = (
        _load_terminal_authorization(
            run_dir,
            source=selected_source,
            run_id=selected_run_id,
        )
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
            raise SourceStateError(f"duplicate run selection: {identity[0]}/{identity[1]}")
        seen.add(identity)
        covered.add(identity[0])
        selected.append(identity)
    if not selected:
        raise SourceStateError("at least one explicit --select SOURCE:RUN_ID is required")
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
            or candidate_payload != canonical_json_bytes(
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
            raise SourceStateError(f"evidence parent is not owned by the current user: {parent}")
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
            raise SourceStateError(f"{label} must be a regular non-symlink file: {path}")
        mode = stat.S_IMODE(info.st_mode)
        if mode != 0o600:
            raise SourceStateError(
                f"{label} must have mode 0600; mode is {mode:04o}: {path}"
            )
        if info.st_uid != os.geteuid():
            raise SourceStateError(
                f"{label} is not owned by the current user: {path}"
            )
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
        parsed.append(parse_selection(selection) if isinstance(selection, str) else selection)
    identities = _validate_selections(parsed)
    runs = [
        validate_selected_run(artifacts_root, source, run_id, now=now)
        for source, run_id in identities
    ]
    evidence: dict[str, object] = {
        "schema_version": SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
        "runs": [run.evidence_row() for run in runs],
    }
    payload = canonical_json_bytes(evidence)
    _rehash_validated(runs)
    _publish_create_only(_absolute_lexical(output_path), payload)
    return evidence


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
        raise SourceStateError(
            "source-state evidence is not canonically serialized"
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
        raise SourceStateError("source-state evidence runs must be sorted by source and run_id")
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
    "COMPLETION_SCHEMA_VERSION",
    "PRODUCTION_SOURCES",
    "SOURCE_STATE_EVIDENCE_SCHEMA_VERSION",
    "TERMINAL_RECOVERY_GUARD_FILENAME",
    "FileAttestation",
    "LoadedEvidence",
    "SourceStateError",
    "SourceStatePublicationPending",
    "ValidatedRun",
    "build_source_state_evidence",
    "canonical_json_bytes",
    "iter_attested_lines",
    "load_source_state_evidence",
    "parse_selection",
    "source_state_evidence_companion_paths",
    "validate_completion_record",
    "validate_selected_run",
]
