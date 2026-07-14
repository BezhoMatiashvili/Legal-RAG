"""Private artifact I/O, bounded log rotation, and fail-closed pruning plans.

The pruning API deliberately separates planning from deletion.  A raw crawl run is
eligible only when its own completion record is unambiguously successful, its retention
period has elapsed, and an explicitly verified immutable generation lists the exact
``(source, run_id)`` in ``covered_runs``.  Legacy runs and generation manifests that do
not carry those proofs are retained.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
DEFAULT_ROTATE_BYTES = 50 * 1024 * 1024
DEFAULT_ROTATE_BACKUPS = 10
DEFAULT_RAW_RETENTION_DAYS = 180
TAS_RAW_RETENTION_DAYS = 30

_MAX_METADATA_BYTES = 1024 * 1024
_MAX_PRUNE_PLAN_BYTES = 64 * 1024 * 1024
PRUNE_PLAN_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUCCESS = "success"
_VERIFICATION_GATES = ("coverage", "integrity", "freshness", "quality")
_PROTECTED_NAME_PARTS = (
    "dead-letter",
    "dead_letter",
    "failed",
    "failure",
    "quarantine",
    "repair",
    "snapshot",
)


class ArtifactSafetyError(RuntimeError):
    """Raised when an artifact operation cannot prove that it is safe."""


@dataclass(frozen=True, order=True, slots=True)
class RunIdentity:
    """Stable identity of one raw source run."""

    source: str
    run_id: str


@dataclass(frozen=True, slots=True)
class VerifiedGenerationCoverage:
    """Exact raw-run coverage asserted by a verified immutable generation."""

    generation_id: str
    manifest_path: Path
    covered_runs: tuple[RunIdentity, ...]


@dataclass(frozen=True, slots=True)
class PruneCandidate:
    """A raw run that passed every eligibility proof at planning time."""

    relative_path: str
    source: str
    run_id: str
    completed_at: str
    eligible_at: str
    retention_days: int
    covered_by: tuple[str, ...]
    metadata_sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "source": self.source,
            "run_id": self.run_id,
            "completed_at": self.completed_at,
            "eligible_at": self.eligible_at,
            "retention_days": self.retention_days,
            "covered_by": list(self.covered_by),
            "metadata_sha256": self.metadata_sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class RetainedRun:
    """A raw run excluded from a pruning plan, with one deterministic reason."""

    relative_path: str
    source: str
    run_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "source": self.source,
            "run_id": self.run_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """Immutable, deterministic plan; constructing it never deletes artifacts."""

    artifacts_root: Path
    generations_root: Path
    planned_at: str
    verified_generations: tuple[str, ...]
    candidates: tuple[PruneCandidate, ...]
    retained: tuple[RetainedRun, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PRUNE_PLAN_SCHEMA_VERSION,
            "planned_at": self.planned_at,
            "artifacts_root": str(self.artifacts_root),
            "generations_root": str(self.generations_root),
            "verified_generations": list(self.verified_generations),
            "candidate_count": len(self.candidates),
            "retained_count": len(self.retained),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "retained": [decision.to_dict() for decision in self.retained],
        }


def _create_private_parents(directory: Path) -> None:
    """Create missing directories as 0700 without chmod-ing existing directories."""
    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
        raise ArtifactSafetyError(f"artifact parent is not a real directory: {cursor}")
    for path in reversed(missing):
        try:
            path.mkdir(mode=PRIVATE_DIRECTORY_MODE)
        except FileExistsError:  # another local writer won the creation race
            pass
        if path.is_symlink() or not path.is_dir():
            raise ArtifactSafetyError(
                f"artifact parent is not a real directory: {path}"
            )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomically replace ``path`` with an fsynced 0600 text file.

    Missing parent directories are created as 0700.  Existing directories are never
    chmod-ed, which avoids silently changing user-owned artifacts.
    """
    target = Path(path)
    _create_private_parents(target.parent)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(target)

    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        handle = os.fdopen(fd, "w", encoding=encoding)
        fd = -1  # ownership transferred to ``handle`` even if the write raises
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, PRIVATE_FILE_MODE)
        _fsync_directory(target.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return target


def atomic_write_json(path: Path | str, value: object) -> Path:
    """Atomically write stable, UTF-8 JSON with owner-only permissions."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(path, payload)


def _atomic_create_private_json(path: Path, value: object) -> Path:
    """Publish a complete owner-only JSON file without replacing an existing path."""
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _create_private_parents(path.parent)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return path


def _validate_rotation_options(max_bytes: int, backups: int) -> None:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if backups <= 0:
        raise ValueError("backups must be positive")


def _regular_file_stat(path: Path) -> os.stat_result | None:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(result.st_mode):
        raise ArtifactSafetyError(f"refusing non-regular artifact file: {path}")
    return result


@contextlib.contextmanager
def _rotation_lock(path: Path) -> Iterator[None]:
    _create_private_parents(path.parent)
    lock_path = path.with_name(f".{path.name}.rotate.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _rotate_unlocked(path: Path, backups: int) -> bool:
    if _regular_file_stat(path) is None:
        return False

    oldest = path.with_name(f"{path.name}.{backups}")
    if _regular_file_stat(oldest) is not None:
        oldest.unlink()
    for index in range(backups - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if _regular_file_stat(source) is None:
            continue
        destination = path.with_name(f"{path.name}.{index + 1}")
        if destination.exists() and _regular_file_stat(destination) is None:
            raise ArtifactSafetyError(f"refusing unsafe rotation target: {destination}")
        os.replace(source, destination)
        os.chmod(destination, PRIVATE_FILE_MODE)

    first = path.with_name(f"{path.name}.1")
    if first.exists() and _regular_file_stat(first) is None:
        raise ArtifactSafetyError(f"refusing unsafe rotation target: {first}")
    os.replace(path, first)
    os.chmod(first, PRIVATE_FILE_MODE)
    _fsync_directory(path.parent)
    return True


def rotate_file(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_ROTATE_BYTES,
    backups: int = DEFAULT_ROTATE_BACKUPS,
) -> bool:
    """Rotate a regular file at ``max_bytes``, retaining ``backups`` archives."""
    _validate_rotation_options(max_bytes, backups)
    target = Path(path)
    with _rotation_lock(target):
        current = _regular_file_stat(target)
        if current is None or current.st_size < max_bytes:
            return False
        return _rotate_unlocked(target, backups)


def append_rotating_text(
    path: Path | str,
    text: str,
    *,
    max_bytes: int = DEFAULT_ROTATE_BYTES,
    backups: int = DEFAULT_ROTATE_BACKUPS,
    encoding: str = "utf-8",
) -> Path:
    """Append text to a private file, rotating before it would exceed the limit."""
    _validate_rotation_options(max_bytes, backups)
    target = Path(path)
    data = text.encode(encoding)
    with _rotation_lock(target):
        current = _regular_file_stat(target)
        if (
            current is not None
            and current.st_size
            and current.st_size + len(data) > max_bytes
        ):
            _rotate_unlocked(target, backups)

        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags, PRIVATE_FILE_MODE)
        try:
            os.fchmod(fd, PRIVATE_FILE_MODE)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(target.parent)
    return target


def parse_utc_datetime(value: str) -> datetime:
    """Parse an explicitly timezone-aware ISO timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path) -> dict[str, Any]:
    file_stat = _regular_file_stat(path)
    if file_stat is None:
        raise FileNotFoundError(path)
    if file_stat.st_size > _MAX_METADATA_BYTES:
        raise ArtifactSafetyError(
            f"metadata exceeds {_MAX_METADATA_BYTES} bytes: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_component(value: object) -> str | None:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return None
    if "/" in value or "\\" in value or "\x00" in value:
        return None
    return value


def _parse_covered_runs(value: object) -> tuple[RunIdentity, ...] | None:
    records: list[RunIdentity] = []
    if isinstance(value, Mapping):
        for source, run_ids in value.items():
            safe_source = _safe_component(source)
            if (
                safe_source is None
                or not isinstance(run_ids, Sequence)
                or isinstance(run_ids, (str, bytes))
            ):
                return None
            for run_id in run_ids:
                safe_run_id = _safe_component(run_id)
                if safe_run_id is None:
                    return None
                records.append(RunIdentity(safe_source, safe_run_id))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if not isinstance(item, Mapping):
                return None
            safe_source = _safe_component(item.get("source"))
            safe_run_id = _safe_component(item.get("run_id"))
            if safe_source is None or safe_run_id is None:
                return None
            if item.get("complete") is False:
                return None
            records.append(RunIdentity(safe_source, safe_run_id))
    else:
        return None

    unique = set(records)
    if len(unique) != len(records):
        return None
    return tuple(sorted(unique))


def _gate_ok(report: Mapping[str, object], gate: str) -> bool:
    value = report.get(gate)
    if not isinstance(value, Mapping) or value.get("ok") is not True:
        return False
    issue_count = value.get("issue_count")
    examples = value.get("examples")
    return type(issue_count) is int and issue_count >= 0 and isinstance(examples, list)


def _matching_verification_report(
    generation_dir: Path,
    manifest: Mapping[str, object],
    covered_runs: tuple[RunIdentity, ...],
) -> bool:
    path = generation_dir.with_name(f"{generation_dir.name}.verification.json")
    generation_id = manifest.get("generation_id")
    if not path.exists():
        return False
    try:
        report = _read_json_object(path)
    except (ArtifactSafetyError, OSError, ValueError, json.JSONDecodeError):
        return False
    if report.get("schema_version") != 1:
        return False
    if report.get("generation_id") != generation_id or report.get("ok") is not True:
        return False
    try:
        parse_utc_datetime(report["verified_at"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return False
    manifest_path = generation_dir / "manifest.json"
    if report.get("manifest_sha256") != _metadata_hash(manifest_path):
        return False
    if not isinstance(report.get("stats"), Mapping):
        return False
    if not all(_gate_ok(report, gate) for gate in _VERIFICATION_GATES):
        return False
    if report.get("covered_runs") != manifest.get("covered_runs"):
        return False
    report_runs = _parse_covered_runs(report.get("covered_runs"))
    return report_runs == covered_runs


def load_verified_generation_coverage(
    generations_root: Path | str,
) -> tuple[VerifiedGenerationCoverage, ...]:
    """Load exact ``covered_runs`` only from explicitly verified generations.

    The sibling ``<generation>.verification.json`` must bind the exact manifest SHA and
    report all four gates green.  Missing/invalid ``covered_runs`` invalidates the entire
    generation.  Both ``{source: [run_ids]}`` and ``[{source, run_id}]`` are accepted.
    """
    root = Path(generations_root)
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise ArtifactSafetyError(f"generations root is not a real directory: {root}")

    verified: list[VerifiedGenerationCoverage] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
            continue
        try:
            manifest = _read_json_object(manifest_path)
        except (ArtifactSafetyError, OSError, ValueError, json.JSONDecodeError):
            continue
        generation_id = _safe_component(manifest.get("generation_id"))
        if (
            manifest.get("schema_version") != 1
            or generation_id is None
            or generation_id != manifest_path.parent.name
        ):
            continue
        covered_runs = _parse_covered_runs(manifest.get("covered_runs"))
        if covered_runs is None:
            continue
        if not _matching_verification_report(
            manifest_path.parent, manifest, covered_runs
        ):
            continue
        verified.append(
            VerifiedGenerationCoverage(
                generation_id=generation_id,
                manifest_path=manifest_path.resolve(),
                covered_runs=covered_runs,
            )
        )
    return tuple(sorted(verified, key=lambda item: item.generation_id))


def _coverage_index(
    generations: Sequence[VerifiedGenerationCoverage],
) -> dict[RunIdentity, tuple[str, ...]]:
    mutable: dict[RunIdentity, set[str]] = {}
    for generation in generations:
        for run in generation.covered_runs:
            mutable.setdefault(run, set()).add(generation.generation_id)
    return {run: tuple(sorted(ids)) for run, ids in mutable.items()}


def _protected_name(name: str) -> bool:
    normalized = name.lower().replace(" ", "_")
    return any(part in normalized for part in _PROTECTED_NAME_PARTS)


def _contains_protected_evidence(run_dir: Path) -> bool:
    for path in run_dir.rglob("*"):
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            return True
        if not (stat.S_ISREG(file_stat.st_mode) or stat.S_ISDIR(file_stat.st_mode)):
            return True
        relative = path.relative_to(run_dir)
        if any(_protected_name(part) for part in relative.parts):
            return True
    return False


def _truthy_failure_signal(value: object) -> bool:
    if value in (None, False, 0, ""):
        return False
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return False
    return True


def _metadata_has_failures(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(token in normalized for token in ("error", "failed", "failure")):
                if _truthy_failure_signal(nested):
                    return True
            if _metadata_has_failures(nested):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_metadata_has_failures(item) for item in value)
    return False


def _required_true(
    metadata: Mapping[str, object],
    direct_key: str,
    nested_keys: Sequence[tuple[str, str]],
) -> bool:
    values: list[object] = []
    if direct_key in metadata:
        values.append(metadata[direct_key])
    for parent_key, child_key in nested_keys:
        parent = metadata.get(parent_key)
        if isinstance(parent, Mapping) and child_key in parent:
            values.append(parent[child_key])
    return bool(values) and all(value is True for value in values)


def _run_success_reason(
    metadata: Mapping[str, object], source: str, run_id: str
) -> str | None:
    if metadata.get("run_id") != run_id:
        return "run_identity_mismatch"
    recorded_sources = [
        metadata[key] for key in ("source", "spider") if key in metadata
    ]
    if not recorded_sources or any(value != source for value in recorded_sources):
        return "source_identity_mismatch"

    outcomes = [metadata[key] for key in ("outcome", "status") if key in metadata]
    if not outcomes or any(
        not isinstance(value, str) or value.lower() != _SUCCESS for value in outcomes
    ):
        return "run_not_successful"
    if not _required_true(
        metadata,
        "quality_passed",
        (("quality", "passed"), ("quality_gate", "passed")),
    ):
        return "quality_not_passed"
    if not _required_true(
        metadata,
        "feeds_durable",
        (("feeds", "durable"), ("feed_outputs", "durable")),
    ):
        return "feeds_not_durable"
    if _metadata_has_failures(metadata):
        return "failure_recorded"
    return None


def _directory_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        result = path.lstat()
        if stat.S_ISLNK(result.st_mode):
            raise ArtifactSafetyError(f"refusing symlink in raw run: {path}")
        if stat.S_ISREG(result.st_mode):
            total += result.st_size
    return total


def _metadata_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retention_days(source: str) -> int:
    return (
        TAS_RAW_RETENTION_DAYS
        if source.lower() == "tas"
        else DEFAULT_RAW_RETENTION_DAYS
    )


def _retained(source: str, run_id: str, reason: str) -> RetainedRun:
    return RetainedRun(
        relative_path=f"{source}/runs/{run_id}",
        source=source,
        run_id=run_id,
        reason=reason,
    )


def _assess_run(
    run_dir: Path,
    source: str,
    run_id: str,
    now: datetime,
    coverage: Mapping[RunIdentity, tuple[str, ...]],
) -> PruneCandidate | RetainedRun:
    if _protected_name(source) or _protected_name(run_id):
        return _retained(source, run_id, "protected_namespace")
    if run_dir.is_symlink() or not run_dir.is_dir():
        return _retained(source, run_id, "unsafe_run_path")

    metadata_path = run_dir / "run.json"
    items_path = run_dir / "items.jsonl"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        return _retained(source, run_id, "completion_record_missing")
    if items_path.is_symlink() or not items_path.is_file():
        return _retained(source, run_id, "raw_items_missing")
    try:
        metadata = _read_json_object(metadata_path)
    except (ArtifactSafetyError, OSError, ValueError, json.JSONDecodeError):
        return _retained(source, run_id, "completion_record_invalid")
    if _contains_protected_evidence(run_dir):
        return _retained(source, run_id, "protected_evidence_present")
    if reason := _run_success_reason(metadata, source, run_id):
        return _retained(source, run_id, reason)

    completed_value = metadata.get("completed_at")
    try:
        completed_at = parse_utc_datetime(completed_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _retained(source, run_id, "completion_timestamp_invalid")
    if completed_at > now:
        return _retained(source, run_id, "completion_timestamp_in_future")

    covered_by = coverage.get(RunIdentity(source, run_id), ())
    if not covered_by:
        return _retained(source, run_id, "not_covered_by_verified_generation")

    retention_days = _retention_days(source)
    eligible_at = completed_at + timedelta(days=retention_days)
    if now < eligible_at:
        return _retained(source, run_id, "retention_period_not_elapsed")

    return PruneCandidate(
        relative_path=f"{source}/runs/{run_id}",
        source=source,
        run_id=run_id,
        completed_at=_iso_utc(completed_at),
        eligible_at=_iso_utc(eligible_at),
        retention_days=retention_days,
        covered_by=covered_by,
        metadata_sha256=_metadata_hash(metadata_path),
        size_bytes=_directory_size(run_dir),
    )


def build_prune_plan(
    artifacts_root: Path | str,
    generations_root: Path | str,
    *,
    now: datetime | None = None,
) -> PrunePlan:
    """Build a deterministic, read-only pruning plan for raw crawl run directories."""
    root = Path(artifacts_root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactSafetyError(f"artifacts root is not a real directory: {root}")
    root = root.resolve()
    generation_root = Path(generations_root).resolve()
    requested_now = now or datetime.now(UTC)
    if requested_now.tzinfo is None:
        raise ValueError("now must include a timezone")
    planned_at = requested_now.astimezone(UTC)

    generations = load_verified_generation_coverage(generation_root)
    coverage = _coverage_index(generations)
    candidates: list[PruneCandidate] = []
    retained: list[RetainedRun] = []

    for source_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if source_dir.is_symlink() or not source_dir.is_dir():
            continue
        runs_dir = source_dir / "runs"
        if not runs_dir.exists():
            continue
        if runs_dir.is_symlink() or not runs_dir.is_dir():
            raise ArtifactSafetyError(f"runs path is not a real directory: {runs_dir}")
        for run_dir in sorted(runs_dir.iterdir(), key=lambda path: path.name):
            if not run_dir.is_dir() and not run_dir.is_symlink():
                continue
            decision = _assess_run(
                run_dir,
                source_dir.name,
                run_dir.name,
                planned_at,
                coverage,
            )
            if isinstance(decision, PruneCandidate):
                candidates.append(decision)
            else:
                retained.append(decision)

    return PrunePlan(
        artifacts_root=root,
        generations_root=generation_root,
        planned_at=_iso_utc(planned_at),
        verified_generations=tuple(item.generation_id for item in generations),
        candidates=tuple(sorted(candidates, key=lambda item: item.relative_path)),
        retained=tuple(sorted(retained, key=lambda item: item.relative_path)),
    )


def _prune_plan_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prune_plan_document(plan: PrunePlan) -> dict[str, object]:
    """Return the canonical persisted plan plus a digest of every plan field."""
    value = plan.to_dict()
    return {**value, "plan_sha256": _prune_plan_digest(value)}


def write_prune_plan(path: Path | str, plan: PrunePlan) -> Path:
    """Persist a new 0600 pruning plan, refusing overwrite and partial publication."""
    return _atomic_create_private_json(Path(path), prune_plan_document(plan))


def _plan_component(value: object, field: str) -> str:
    result = _safe_component(value)
    if result is None:
        raise ArtifactSafetyError(f"invalid prune plan {field}")
    return result


def _plan_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ArtifactSafetyError(f"invalid prune plan {field}")
    try:
        normalized = _iso_utc(parse_utc_datetime(value))
    except ValueError as exc:
        raise ArtifactSafetyError(f"invalid prune plan {field}") from exc
    if normalized != value:
        raise ArtifactSafetyError(f"non-canonical prune plan {field}")
    return value


def _load_prune_candidate(value: object) -> PruneCandidate:
    fields = {
        "relative_path",
        "source",
        "run_id",
        "completed_at",
        "eligible_at",
        "retention_days",
        "covered_by",
        "metadata_sha256",
        "size_bytes",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ArtifactSafetyError("invalid prune candidate schema")
    source = _plan_component(value["source"], "candidate source")
    run_id = _plan_component(value["run_id"], "candidate run_id")
    relative_path = value["relative_path"]
    if relative_path != f"{source}/runs/{run_id}":
        raise ArtifactSafetyError("invalid prune candidate relative_path")
    retention_days = value["retention_days"]
    size_bytes = value["size_bytes"]
    covered_by = value["covered_by"]
    metadata_sha256 = value["metadata_sha256"]
    if type(retention_days) is not int or retention_days < 1:
        raise ArtifactSafetyError("invalid prune candidate retention_days")
    if type(size_bytes) is not int or size_bytes < 0:
        raise ArtifactSafetyError("invalid prune candidate size_bytes")
    if (
        not isinstance(covered_by, list)
        or not covered_by
        or any(_safe_component(item) is None for item in covered_by)
        or covered_by != sorted(set(covered_by))
    ):
        raise ArtifactSafetyError("invalid prune candidate covered_by")
    if not isinstance(metadata_sha256, str) or not _SHA256_RE.fullmatch(
        metadata_sha256
    ):
        raise ArtifactSafetyError("invalid prune candidate metadata_sha256")
    return PruneCandidate(
        relative_path=relative_path,
        source=source,
        run_id=run_id,
        completed_at=_plan_timestamp(value["completed_at"], "completed_at"),
        eligible_at=_plan_timestamp(value["eligible_at"], "eligible_at"),
        retention_days=retention_days,
        covered_by=tuple(covered_by),
        metadata_sha256=metadata_sha256,
        size_bytes=size_bytes,
    )


def _load_retained_run(value: object) -> RetainedRun:
    fields = {"relative_path", "source", "run_id", "reason"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ArtifactSafetyError("invalid retained-run schema")
    source = _plan_component(value["source"], "retained source")
    run_id = _plan_component(value["run_id"], "retained run_id")
    relative_path = value["relative_path"]
    reason = value["reason"]
    if relative_path != f"{source}/runs/{run_id}":
        raise ArtifactSafetyError("invalid retained-run relative_path")
    if not isinstance(reason, str) or not reason:
        raise ArtifactSafetyError("invalid retained-run reason")
    return RetainedRun(relative_path, source, run_id, reason)


def load_prune_plan(path: Path | str) -> PrunePlan:
    """Load and validate an owner-only, digest-bound pruning plan artifact."""
    source = Path(path)
    try:
        file_stat = source.lstat()
    except FileNotFoundError:
        raise ArtifactSafetyError(f"prune plan does not exist: {source}") from None
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != os.geteuid()
        or stat.S_IMODE(file_stat.st_mode) & 0o077
    ):
        raise ArtifactSafetyError(
            "prune plan must be an owner-owned regular file with no group/other access"
        )
    if file_stat.st_size < 2 or file_stat.st_size > _MAX_PRUNE_PLAN_BYTES:
        raise ArtifactSafetyError("prune plan size is outside the accepted bounds")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactSafetyError(f"cannot read prune plan: {exc}") from exc
    expected_fields = {
        "schema_version",
        "planned_at",
        "artifacts_root",
        "generations_root",
        "verified_generations",
        "candidate_count",
        "retained_count",
        "candidates",
        "retained",
        "plan_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ArtifactSafetyError("invalid prune plan schema")
    if value["schema_version"] != PRUNE_PLAN_SCHEMA_VERSION:
        raise ArtifactSafetyError("unsupported prune plan schema_version")
    supplied_digest = value.pop("plan_sha256")
    if (
        not isinstance(supplied_digest, str)
        or not _SHA256_RE.fullmatch(supplied_digest)
        or _prune_plan_digest(value) != supplied_digest
    ):
        raise ArtifactSafetyError("prune plan digest mismatch")

    def resolved_root(field: str) -> Path:
        raw = value[field]
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ArtifactSafetyError(f"invalid prune plan {field}")
        resolved = Path(raw).resolve()
        if str(resolved) != raw:
            raise ArtifactSafetyError(f"non-canonical prune plan {field}")
        return resolved

    verified = value["verified_generations"]
    if (
        not isinstance(verified, list)
        or any(_safe_component(item) is None for item in verified)
        or verified != sorted(set(verified))
    ):
        raise ArtifactSafetyError("invalid verified_generations")
    candidates_raw = value["candidates"]
    retained_raw = value["retained"]
    if not isinstance(candidates_raw, list) or not isinstance(retained_raw, list):
        raise ArtifactSafetyError("invalid prune plan record lists")
    candidates = tuple(_load_prune_candidate(item) for item in candidates_raw)
    retained = tuple(_load_retained_run(item) for item in retained_raw)
    if value["candidate_count"] != len(candidates):
        raise ArtifactSafetyError("prune plan candidate_count mismatch")
    if value["retained_count"] != len(retained):
        raise ArtifactSafetyError("prune plan retained_count mismatch")
    if [item.relative_path for item in candidates] != sorted(
        {item.relative_path for item in candidates}
    ):
        raise ArtifactSafetyError("prune candidates are not unique and sorted")
    if [item.relative_path for item in retained] != sorted(
        {item.relative_path for item in retained}
    ):
        raise ArtifactSafetyError("retained runs are not unique and sorted")
    return PrunePlan(
        artifacts_root=resolved_root("artifacts_root"),
        generations_root=resolved_root("generations_root"),
        planned_at=_plan_timestamp(value["planned_at"], "planned_at"),
        verified_generations=tuple(verified),
        candidates=candidates,
        retained=retained,
    )


def apply_prune_plan(
    plan_path: Path | str, *, approved: bool = False
) -> tuple[str, ...]:
    """Delete exactly a persisted plan's candidates after explicit approval.

    The CLI adds the second, environment-variable approval gate. This function accepts
    only an already-persisted plan and rebuilds it at the original timestamp before
    deleting anything, so changed completion or verification evidence aborts atomically.
    """
    if approved is not True:
        raise PermissionError("artifact pruning requires explicit approval")
    plan = load_prune_plan(plan_path)

    fresh = build_prune_plan(
        plan.artifacts_root,
        plan.generations_root,
        now=parse_utc_datetime(plan.planned_at),
    )
    fresh_candidates = {item.relative_path: item for item in fresh.candidates}
    for candidate in plan.candidates:
        if fresh_candidates.get(candidate.relative_path) != candidate:
            raise ArtifactSafetyError(
                f"prune candidate changed after planning: {candidate.relative_path}"
            )

    paths: list[tuple[str, Path]] = []
    for candidate in plan.candidates:
        expected_relative = f"{candidate.source}/runs/{candidate.run_id}"
        if candidate.relative_path != expected_relative:
            raise ArtifactSafetyError(
                f"invalid candidate path: {candidate.relative_path}"
            )
        path = plan.artifacts_root / candidate.source / "runs" / candidate.run_id
        if path.is_symlink() or not path.is_dir():
            raise ArtifactSafetyError(f"candidate path became unsafe: {path}")
        paths.append((candidate.relative_path, path))

    deleted: list[str] = []
    for relative_path, path in paths:
        parent = path.parent
        shutil.rmtree(path)
        _fsync_directory(parent)
        deleted.append(relative_path)
    return tuple(deleted)


__all__ = [
    "ArtifactSafetyError",
    "DEFAULT_RAW_RETENTION_DAYS",
    "DEFAULT_ROTATE_BACKUPS",
    "DEFAULT_ROTATE_BYTES",
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "PRUNE_PLAN_SCHEMA_VERSION",
    "PruneCandidate",
    "PrunePlan",
    "RunIdentity",
    "TAS_RAW_RETENTION_DAYS",
    "VerifiedGenerationCoverage",
    "append_rotating_text",
    "apply_prune_plan",
    "atomic_write_json",
    "atomic_write_text",
    "build_prune_plan",
    "load_verified_generation_coverage",
    "load_prune_plan",
    "parse_utc_datetime",
    "prune_plan_document",
    "rotate_file",
    "write_prune_plan",
]
