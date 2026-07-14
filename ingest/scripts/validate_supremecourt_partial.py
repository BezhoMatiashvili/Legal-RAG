#!/usr/bin/env python
"""Strict, offline validation for a timed Supreme Court crawl artifact.

The timed crawler writes a cumulative ``items.jsonl``, an append-only journal containing
only this run's new items, and ``partial_manifest.json``.  This module proves that those
three files agree before a tokenizer, model, Qdrant, or remote service is allowed to see
the artifact.

Usage (from ``ingest/``)::

    .venv/bin/python scripts/validate_supremecourt_partial.py \
        --run-dir ../artifacts/supremecourt/runs/<run-id>

Both success and failure are emitted as a single JSON object on stdout.  Exit status zero
means the artifact passed every gate; status one means it must not be embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import BinaryIO, Iterator

SCHEMA_VERSION = 1
EXPECTED_RUNTIME_SECONDS = 14_400
EXPECTED_FINISH_REASON = "closespider_timeout"
TARGET_LITERAL = "330100122006207137"
BASE_URL = "https://www.supremecourt.ge"

CHAMBER_TO_PALATA = {
    "ადმინისტრაციულ საქმეთა პალატა": "0",
    "სამოქალაქო საქმეთა პალატა": "1",
    "სისხლის სამართლის საქმეთა პალატა": "2",
}
OFFICIAL_CHAMBERS = frozenset(CHAMBER_TO_PALATA)

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CASE_ID_RE = re.compile(r"[0-9]+")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_METADATA_KINDS = {
    "article_summary",
    "legacy_unlabeled",
    "legacy_unknown",
    "metadata only",
    "metadata-only",
    "metadata_only",
}
_INCOMPLETE_STATUSES = {"empty", "failed", "malformed", "metadata_only"}


class ValidationError(RuntimeError):
    """The run artifact failed one or more deterministic validation gates."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__(
            "Supreme Court partial artifact is invalid: " + "; ".join(errors)
        )


@dataclass(frozen=True)
class _Record:
    identity: str
    case_id: str
    chamber: str
    decision_date: date
    fingerprint: str
    target_body_hit: bool


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _loads(payload: str | bytes, *, label: str) -> object:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationError([f"{label}: invalid UTF-8 JSON: {exc}"]) from exc


@contextmanager
def _open_private_regular(path: Path) -> Iterator[BinaryIO]:
    """Open an owner-private regular file without following its final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationError([f"{path.name}: cannot safely open file: {exc}"]) from exc
    try:
        file_stat = os.fstat(descriptor)
        errors = []
        if not stat.S_ISREG(file_stat.st_mode):
            errors.append(f"{path.name}: must be a regular non-symlink file")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            errors.append(
                f"{path.name}: must be owner-private; mode is "
                f"{stat.S_IMODE(file_stat.st_mode):04o}"
            )
        if errors:
            raise ValidationError(errors)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)


def _read_private_regular(path: Path) -> bytes:
    with _open_private_regular(path) as handle:
        return handle.read()


def _iso_date(value: object, *, label: str, errors: list[str]) -> date | None:
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        errors.append(f"{label}: expected canonical ISO date YYYY-MM-DD")
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        errors.append(f"{label}: invalid calendar date {value!r}")
        return None
    if parsed.isoformat() != value:
        errors.append(f"{label}: non-canonical ISO date {value!r}")
        return None
    return parsed


def _iso_datetime(value: object, *, label: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: expected a final ISO timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label}: invalid ISO timestamp {value!r}")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(f"{label}: timestamp must include a UTC offset")
        return None
    return parsed


def _canonical_fingerprint(item: dict) -> str:
    payload = json.dumps(
        item,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_explicitly_incomplete(value: object) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return not bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"1", "true", "yes"}
    return value is not None


def _record_from_item(
    item: object,
    *,
    label: str,
    errors: list[str],
) -> _Record | None:
    if not isinstance(item, dict):
        errors.append(f"{label}: JSON value must be an object")
        return None

    case_id = item.get("case_id")
    chamber = item.get("chamber")
    decision_date = _iso_date(item.get("date"), label=f"{label}.date", errors=errors)
    valid = True
    if not isinstance(case_id, str) or _CASE_ID_RE.fullmatch(case_id) is None:
        errors.append(f"{label}.case_id: expected a nonempty decimal string")
        valid = False
    if chamber not in OFFICIAL_CHAMBERS:
        errors.append(f"{label}.chamber: unofficial chamber {chamber!r}")
        valid = False

    if valid:
        expected_url = (
            f"{BASE_URL}/ka/fullcase/{case_id}/{CHAMBER_TO_PALATA[str(chamber)]}"
        )
        if item.get("source_url") != expected_url:
            errors.append(
                f"{label}.source_url: expected exact official fullcase URL "
                f"{expected_url!r}"
            )

    body = item.get("body_markdown")
    if not isinstance(body, str) or not body.strip():
        errors.append(f"{label}.body_markdown: nonempty full text is required")
        body = ""

    if "content_complete" in item and _is_explicitly_incomplete(
        item["content_complete"]
    ):
        errors.append(f"{label}: content_complete marks the record incomplete")
    content_kind = item.get("content_kind")
    if (
        isinstance(content_kind, str)
        and content_kind.strip().lower() in _METADATA_KINDS
    ):
        errors.append(f"{label}: metadata-only content_kind is forbidden")
    if item.get("metadata_only") is True:
        errors.append(f"{label}: metadata_only=true is forbidden")
    extraction_status = item.get("extraction_status")
    if (
        isinstance(extraction_status, str)
        and extraction_status.strip().lower() in _INCOMPLETE_STATUSES
    ):
        errors.append(f"{label}: incomplete extraction_status is forbidden")

    case_number = item.get("case_number")
    if isinstance(case_number, str) and TARGET_LITERAL in case_number:
        errors.append(
            f"{label}.case_number: first-instance number {TARGET_LITERAL} "
            "must not be treated as a Supreme Court case number"
        )
    for field, value in item.items():
        if field in {"body_markdown", "case_number"}:
            continue
        if isinstance(value, str) and TARGET_LITERAL in value:
            errors.append(
                f"{label}.{field}: target literal is allowed only in body_markdown"
            )

    if not valid or decision_date is None:
        return None
    identity = f"{case_id}:{chamber}"
    try:
        fingerprint = _canonical_fingerprint(item)
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}: cannot canonicalize JSON object: {exc}")
        return None
    return _Record(
        identity=identity,
        case_id=case_id,
        chamber=str(chamber),
        decision_date=decision_date,
        fingerprint=fingerprint,
        target_body_hit=TARGET_LITERAL in body,
    )


def _read_jsonl(path: Path) -> tuple[list[_Record], str]:
    records: list[_Record] = []
    errors: list[str] = []
    digest = hashlib.sha256()
    with _open_private_regular(path) as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            label = f"{path.name}:{line_number}"
            if not raw_line.endswith(b"\n"):
                errors.append(f"{label}: JSONL record must end with a newline")
            if not raw_line.strip():
                errors.append(f"{label}: blank JSONL records are forbidden")
                continue
            try:
                item = _loads(raw_line, label=label)
            except ValidationError as exc:
                errors.extend(exc.errors)
                continue
            record = _record_from_item(item, label=label, errors=errors)
            if record is not None:
                records.append(record)
    if errors:
        raise ValidationError(errors)
    return records, digest.hexdigest()


def _require_unique(records: list[_Record], *, label: str, errors: list[str]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.identity in seen:
            errors.append(f"{label}: duplicate official identity {record.identity!r}")
        seen.add(record.identity)


def _manifest_int(
    mapping: dict, field: str, *, label: str, errors: list[str]
) -> int | None:
    value = mapping.get(field)
    if type(value) is not int or value < 0:  # bool is deliberately not an integer here.
        errors.append(f"{label}.{field}: expected a nonnegative integer")
        return None
    return value


def _resume_start_state(
    manifest: dict,
    *,
    run_path: Path,
    frontier: date | None,
    lower_bound: date | None,
    started_at: datetime | None,
    visited: frozenset[Path],
    errors: list[str],
) -> tuple[dict[str, date | None], bool, dict[str, str] | None]:
    """Verify a parent chain and return trusted cursors and parent fingerprints."""

    fallback = {chamber: frontier for chamber in CHAMBER_TO_PALATA}
    raw_starts = manifest.get("per_chamber_start_cursors")
    parent = manifest.get("resume_parent")
    if raw_starts is None:
        # Compatibility for the already-running first acceptance crawl: schema-1 originally
        # implied that every chamber started at frontier_start_date.
        if parent is not None:
            errors.append(
                "manifest.resume_parent: legacy manifest without start cursors cannot resume"
            )
        return fallback, False, None
    if not isinstance(raw_starts, dict) or set(raw_starts) != OFFICIAL_CHAMBERS:
        errors.append(
            "manifest.per_chamber_start_cursors: keys must be exactly the three "
            "official chambers"
        )
        return fallback, parent is not None, None

    starts: dict[str, date | None] = {}
    for chamber in CHAMBER_TO_PALATA:
        raw = raw_starts[chamber]
        if raw is None:
            starts[chamber] = None
            continue
        parsed = _iso_date(
            raw,
            label=f"manifest.per_chamber_start_cursors[{chamber!r}]",
            errors=errors,
        )
        if (
            parsed is not None
            and frontier is not None
            and lower_bound is not None
            and not lower_bound <= parsed <= frontier
        ):
            errors.append(
                f"manifest.per_chamber_start_cursors[{chamber!r}]: "
                "lies outside manifest bounds"
            )
        starts[chamber] = parsed

    if parent is None:
        for chamber, start in starts.items():
            if start != frontier:
                errors.append(
                    f"manifest.per_chamber_start_cursors[{chamber!r}]: a fresh run "
                    "must start at frontier_start_date"
                )
        return starts, False, None
    if not isinstance(parent, dict):
        errors.append("manifest.resume_parent: expected null or an object")
        return starts, True, None
    required = {"run_id", "manifest_file", "manifest_sha256", "items_sha256"}
    if set(parent) != required:
        errors.append(
            "manifest.resume_parent: keys must be exactly " + ", ".join(sorted(required))
        )
        return starts, True, None

    parent_run_id = parent.get("run_id")
    if (
        not isinstance(parent_run_id, str)
        or _RUN_ID_RE.fullmatch(parent_run_id) is None
        or parent_run_id == run_path.name
    ):
        errors.append("manifest.resume_parent.run_id: unsafe, missing, or cyclic")
        return starts, True, None
    expected_parent_path = run_path.parent / parent_run_id / "partial_manifest.json"
    manifest_file = parent.get("manifest_file")
    if (
        not isinstance(manifest_file, str)
        or not Path(manifest_file).is_absolute()
        or _absolute_lexical(Path(manifest_file)) != expected_parent_path
    ):
        errors.append(
            "manifest.resume_parent.manifest_file: must be the exact contained "
            "run-scoped parent manifest"
        )
        return starts, True, None
    if expected_parent_path.parent in visited:
        errors.append("manifest.resume_parent: cycle detected in resume chain")
        return starts, True, None

    try:
        parent_payload = _read_private_regular(expected_parent_path)
    except ValidationError as exc:
        errors.extend(f"resume parent: {error}" for error in exc.errors)
        return starts, True, None
    expected_manifest_sha = parent.get("manifest_sha256")
    actual_manifest_sha = hashlib.sha256(parent_payload).hexdigest()
    if (
        not isinstance(expected_manifest_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha) is None
        or expected_manifest_sha != actual_manifest_sha
    ):
        errors.append("manifest.resume_parent.manifest_sha256: parent bytes differ")
        return starts, True, None

    try:
        parent_report = validate_run(expected_parent_path.parent, _visited=visited)
    except ValidationError as exc:
        errors.extend(f"resume parent: {error}" for error in exc.errors)
        return starts, True, None
    if (
        frontier is not None
        and lower_bound is not None
        and (
            parent_report.get("frontier_start_date") != frontier.isoformat()
            or parent_report.get("lower_bound") != lower_bound.isoformat()
        )
    ):
        errors.append(
            "manifest.resume_parent: validated parent date scope differs from child"
        )
    if parent.get("items_sha256") != parent_report.get("items_sha256"):
        errors.append("manifest.resume_parent.items_sha256: parent items digest differs")
    parent_cursors = parent_report.get("per_chamber_resume_cursors")
    declared_starts = {
        chamber: value.isoformat() if value is not None else None
        for chamber, value in starts.items()
    }
    if declared_starts != parent_cursors:
        errors.append(
            "manifest.per_chamber_start_cursors: must exactly equal the validated "
            "parent's derived resume cursors"
        )
    parent_finished = parent_report.get("finished_at")
    if (
        started_at is not None
        and isinstance(parent_finished, str)
        and datetime.fromisoformat(parent_finished) >= started_at
    ):
        errors.append("manifest.resume_parent: parent did not finish before child started")
    parent_fingerprints: dict[str, str] | None = None
    try:
        parent_records, parent_items_sha = _read_jsonl(
            expected_parent_path.parent / "items.jsonl"
        )
    except ValidationError as exc:
        errors.extend(f"resume parent: {error}" for error in exc.errors)
    else:
        if parent_items_sha != parent_report.get("items_sha256"):
            errors.append("manifest.resume_parent: parent items changed during validation")
        else:
            parent_fingerprints = {
                record.identity: record.fingerprint for record in parent_records
            }
    return starts, True, parent_fingerprints


def _validate_completed_windows(
    manifest: dict,
    *,
    frontier: date | None,
    lower_bound: date | None,
    start_cursors: dict[str, date | None],
    has_resume_parent: bool,
    errors: list[str],
) -> tuple[dict[str, str | None], str | None, int, int, int]:
    """Validate window evidence and independently derive every durable frontier.

    ``resume_cursor`` is operational state: trusting the manifest's own copy would let a
    stale or fabricated cursor skip uncrawled dates on the next run.  Only failure-free
    ``completed`` intervals count toward a cursor.  Split parents and incomplete leaves are
    retained as audit evidence but never advance it.
    """

    required_fields = {
        "id",
        "chamber",
        "start",
        "end",
        "status",
        "authoritative_total",
        "known",
        "new",
        "failures",
    }
    windows = manifest.get("completed_windows")
    if not isinstance(windows, list):
        errors.append("manifest.completed_windows: expected a list")
        windows = []

    seen_ids: set[str] = set()
    leaf_intervals: dict[str, list[tuple[date, date, str]]] = {
        chamber: [] for chamber in CHAMBER_TO_PALATA
    }
    completed_intervals: dict[str, list[tuple[date, date]]] = {
        chamber: [] for chamber in CHAMBER_TO_PALATA
    }
    completed_known = 0
    completed_new = 0
    represented_failures = 0

    for index, value in enumerate(windows):
        label = f"manifest.completed_windows[{index}]"
        if not isinstance(value, dict):
            errors.append(f"{label}: expected an object")
            continue
        if set(value) != required_fields:
            errors.append(
                f"{label}: keys must be exactly {', '.join(sorted(required_fields))}"
            )

        window_id = value.get("id")
        if not isinstance(window_id, str) or not window_id:
            errors.append(f"{label}.id: expected a nonempty string")
        elif window_id in seen_ids:
            errors.append(f"{label}.id: duplicate window id {window_id!r}")
        else:
            seen_ids.add(window_id)

        chamber = value.get("chamber")
        if chamber not in OFFICIAL_CHAMBERS:
            errors.append(f"{label}.chamber: unofficial chamber {chamber!r}")
            chamber = None
        start = _iso_date(value.get("start"), label=f"{label}.start", errors=errors)
        end = _iso_date(value.get("end"), label=f"{label}.end", errors=errors)
        if start is not None and end is not None and start > end:
            errors.append(f"{label}: start date exceeds end date")
        if (
            start is not None
            and end is not None
            and lower_bound is not None
            and frontier is not None
            and not lower_bound <= start <= end <= frontier
        ):
            errors.append(f"{label}: interval lies outside manifest bounds")
        chamber_start = start_cursors.get(str(chamber)) if chamber is not None else None
        if (
            chamber is not None
            and start is not None
            and end is not None
            and (
                chamber_start is None
                or end > chamber_start
            )
        ):
            errors.append(
                f"{label}: interval exceeds the chamber's recorded start cursor"
            )

        status = value.get("status")
        if status not in {"completed", "incomplete", "split"}:
            errors.append(
                f"{label}.status: expected 'completed', 'incomplete', or 'split'"
            )

        counts: dict[str, int | None] = {}
        for field in ("known", "new", "failures"):
            counts[field] = _manifest_int(value, field, label=label, errors=errors)
        authoritative_total = value.get("authoritative_total")
        if authoritative_total is not None and (
            type(authoritative_total) is not int or authoritative_total < 0
        ):
            errors.append(
                f"{label}.authoritative_total: expected null or a nonnegative integer"
            )
            authoritative_total = None

        known = counts["known"]
        new = counts["new"]
        failures = counts["failures"]
        if failures is not None:
            represented_failures += failures

        structurally_complete = True
        if status == "completed":
            if authoritative_total is None:
                errors.append(
                    f"{label}.authoritative_total: completed window requires an integer"
                )
                structurally_complete = False
            if failures != 0:
                errors.append(f"{label}: completed window must have zero failures")
                structurally_complete = False
            if (
                authoritative_total is not None
                and known is not None
                and new is not None
                and known + new != authoritative_total
            ):
                errors.append(
                    f"{label}: completed known + new must equal authoritative_total"
                )
                structurally_complete = False
            if known is not None:
                completed_known += known
            if new is not None:
                completed_new += new
        elif status == "incomplete":
            if failures == 0:
                errors.append(f"{label}: incomplete window must retain a failure")
        elif status == "split":
            if authoritative_total is None or authoritative_total <= 30:
                errors.append(
                    f"{label}: split window requires authoritative_total greater than 30"
                )
            if any(count != 0 for count in (known, new, failures)):
                errors.append(f"{label}: split parent counts must all be zero")

        valid_interval = (
            chamber is not None
            and start is not None
            and end is not None
            and start <= end
            and (
                lower_bound is None
                or frontier is None
                or lower_bound <= start <= end <= frontier
            )
            and chamber_start is not None
            and end <= chamber_start
        )
        if valid_interval and status in {"completed", "incomplete"}:
            leaf_intervals[str(chamber)].append((start, end, str(window_id)))
        if valid_interval and status == "completed" and structurally_complete:
            completed_intervals[str(chamber)].append((start, end))

    # Non-split planner leaves for a chamber are disjoint.  Rejecting overlap prevents a
    # malformed collection of windows from being used to manufacture a contiguous cursor.
    for chamber, intervals in leaf_intervals.items():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] <= previous[1]:
                errors.append(
                    "manifest.completed_windows: overlapping leaf intervals for "
                    f"{chamber!r}: {previous[2]!r} and {current[2]!r}"
                )

    derived_cursors: dict[str, str | None] = {
        chamber: None for chamber in CHAMBER_TO_PALATA
    }
    derived_oldest: list[date] = []
    if frontier is not None and lower_bound is not None:
        for chamber in CHAMBER_TO_PALATA:
            start_cursor = start_cursors[chamber]
            if start_cursor is None:
                derived_cursors[chamber] = None
                derived_oldest.append(lower_bound)
                continue
            cursor = start_cursor
            intervals = completed_intervals[chamber]
            while cursor >= lower_bound:
                covering = [start for start, end in intervals if start <= cursor <= end]
                if not covering:
                    break
                cursor = min(covering) - date.resolution
            if cursor < lower_bound:
                derived_cursors[chamber] = None
                derived_oldest.append(lower_bound)
            else:
                derived_cursors[chamber] = cursor.isoformat()
                if cursor != frontier or has_resume_parent:
                    derived_oldest.append(cursor + date.resolution)

    derived_global = (
        max(derived_oldest).isoformat()
        if len(derived_oldest) == len(CHAMBER_TO_PALATA)
        else None
    )
    return (
        derived_cursors,
        derived_global,
        completed_known,
        completed_new,
        represented_failures,
    )


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def validate_run(
    run_dir: str | Path, *, _visited: frozenset[Path] | None = None
) -> dict:
    """Validate ``run_dir`` and return a bounded, machine-readable success report.

    The function performs no network, model, or database access.  It raises
    :class:`ValidationError` on any mismatch.
    """

    run_path = _absolute_lexical(Path(run_dir).expanduser())
    visited = _visited or frozenset()
    if run_path in visited:
        raise ValidationError(["resume parent: cycle detected in resume chain"])
    visited = visited | {run_path}
    try:
        run_stat = run_path.lstat()
    except OSError as exc:
        raise ValidationError([f"run_dir: cannot inspect {run_path}: {exc}"]) from exc
    if stat.S_ISLNK(run_stat.st_mode) or not stat.S_ISDIR(run_stat.st_mode):
        raise ValidationError(["run_dir: must be a real, non-symlink directory"])

    manifest_path = run_path / "partial_manifest.json"
    items_path = run_path / "items.jsonl"
    journal_path = run_path / "items.journal.jsonl"
    manifest = _loads(_read_private_regular(manifest_path), label=manifest_path.name)
    if not isinstance(manifest, dict):
        raise ValidationError(["partial_manifest.json: root must be an object"])

    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest.schema_version: expected {SCHEMA_VERSION}")
    if manifest.get("run_id") != run_path.name:
        errors.append("manifest.run_id: must exactly match the run directory name")
    if manifest.get("finish_reason") != EXPECTED_FINISH_REASON:
        errors.append(
            "manifest.finish_reason: expected graceful four-hour timeout "
            f"{EXPECTED_FINISH_REASON!r}"
        )
    if manifest.get("max_runtime_seconds") != EXPECTED_RUNTIME_SECONDS:
        errors.append(
            f"manifest.max_runtime_seconds: expected {EXPECTED_RUNTIME_SECONDS}"
        )
    if manifest.get("partial_by_design") is not True:
        errors.append("manifest.partial_by_design: expected true")
    if manifest.get("date_order") != "newest_first":
        errors.append("manifest.date_order: expected 'newest_first'")

    started_at = _iso_datetime(
        manifest.get("started_at"), label="manifest.started_at", errors=errors
    )
    finished_at = _iso_datetime(
        manifest.get("finished_at"), label="manifest.finished_at", errors=errors
    )
    if started_at is not None and finished_at is not None and finished_at < started_at:
        errors.append("manifest.finished_at: precedes started_at")
    elapsed_time = manifest.get("elapsed_time_seconds")
    if (
        isinstance(elapsed_time, bool)
        or not isinstance(elapsed_time, (int, float))
        or not math.isfinite(elapsed_time)
        or elapsed_time < EXPECTED_RUNTIME_SECONDS
    ):
        errors.append(
            "manifest.elapsed_time_seconds: expected a finite numeric duration "
            f">= {EXPECTED_RUNTIME_SECONDS}"
        )

    frontier = _iso_date(
        manifest.get("frontier_start_date"),
        label="manifest.frontier_start_date",
        errors=errors,
    )
    lower_bound = _iso_date(
        manifest.get("lower_bound"), label="manifest.lower_bound", errors=errors
    )
    if frontier is not None and lower_bound is not None and lower_bound > frontier:
        errors.append("manifest.lower_bound: exceeds frontier_start_date")
    start_cursors, has_resume_parent, parent_fingerprints = _resume_start_state(
        manifest,
        run_path=run_path,
        frontier=frontier,
        lower_bound=lower_bound,
        started_at=started_at,
        visited=visited,
        errors=errors,
    )

    declared_items = manifest.get("items_file")
    declared_journal = manifest.get("journal_file")
    if (
        not isinstance(declared_items, str)
        or _absolute_lexical(Path(declared_items).expanduser()) != items_path
    ):
        errors.append("manifest.items_file: must name this run's exact items.jsonl")
    if (
        not isinstance(declared_journal, str)
        or _absolute_lexical(Path(declared_journal).expanduser()) != journal_path
    ):
        errors.append(
            "manifest.journal_file: must name this run's exact items.journal.jsonl"
        )

    items, items_sha256 = _read_jsonl(items_path)
    journal, _journal_sha256 = _read_jsonl(journal_path)
    _require_unique(items, label="items.jsonl", errors=errors)
    _require_unique(journal, label="items.journal.jsonl", errors=errors)

    expected_items_sha = manifest.get("items_sha256")
    if (
        not isinstance(expected_items_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_items_sha) is None
        or expected_items_sha != items_sha256
    ):
        errors.append("manifest.items_sha256: does not match exact items.jsonl bytes")

    observed_order = [
        (record.decision_date, record.chamber, record.case_id) for record in items
    ]
    if observed_order != sorted(observed_order, reverse=True):
        errors.append(
            "items.jsonl: order does not match crawler materialization "
            "(date, chamber, case_id descending)"
        )
    journal_dates = [record.decision_date for record in journal]
    if journal_dates != sorted(journal_dates, reverse=True):
        errors.append("items.journal.jsonl: dates must be non-increasing")

    item_by_identity = {record.identity: record for record in items}
    journal_by_identity = {record.identity: record for record in journal}
    if parent_fingerprints is not None:
        for identity, parent_fingerprint in parent_fingerprints.items():
            child_record = item_by_identity.get(identity)
            if child_record is None:
                errors.append(
                    "items.jsonl: resumed cumulative artifact dropped parent identity "
                    f"{identity!r}"
                )
            elif child_record.fingerprint != parent_fingerprint:
                errors.append(
                    "items.jsonl: resumed cumulative artifact changed parent identity "
                    f"{identity!r}"
                )
    for identity, journal_record in journal_by_identity.items():
        item_record = item_by_identity.get(identity)
        if item_record is None:
            errors.append(
                f"items.journal.jsonl: identity {identity!r} is absent from items.jsonl"
            )
        elif item_record.fingerprint != journal_record.fingerprint:
            errors.append(
                f"items.journal.jsonl: record {identity!r} differs from items.jsonl"
            )

    total_items = _manifest_int(
        manifest, "total_items", label="manifest", errors=errors
    )
    known_items = _manifest_int(
        manifest, "known_items", label="manifest", errors=errors
    )
    new_items = _manifest_int(manifest, "new_items", label="manifest", errors=errors)
    known_encountered = _manifest_int(
        manifest, "known_encountered", label="manifest", errors=errors
    )
    if total_items is not None and total_items != len(items):
        errors.append(
            f"manifest.total_items: expected exact item count {len(items)}, "
            f"got {total_items}"
        )
    if new_items is not None and new_items != len(journal):
        errors.append(
            f"manifest.new_items: expected exact journal count {len(journal)}, "
            f"got {new_items}"
        )
    derived_known = len(items) - len(journal_by_identity)
    if known_items is not None and known_items != derived_known:
        errors.append(
            f"manifest.known_items: expected cumulative-minus-journal count "
            f"{derived_known}, got {known_items}"
        )
    if (
        total_items is not None
        and known_items is not None
        and new_items is not None
        and known_items + new_items != total_items
    ):
        errors.append("manifest counts: known_items + new_items must equal total_items")

    per_chamber = manifest.get("per_chamber")
    if not isinstance(per_chamber, dict):
        errors.append("manifest.per_chamber: expected an object")
        per_chamber = {}
    if set(per_chamber) != OFFICIAL_CHAMBERS:
        errors.append(
            "manifest.per_chamber: keys must be exactly the three official chambers"
        )

    derived_per_chamber: dict[str, dict] = {}
    declared_chamber_cursors: dict[str, str | None] = {}
    for chamber in CHAMBER_TO_PALATA:
        chamber_items = [record for record in items if record.chamber == chamber]
        chamber_new = [record for record in journal if record.chamber == chamber]
        chamber_dates = [record.decision_date for record in chamber_items]
        expected = {
            "known_items": len(chamber_items) - len(chamber_new),
            "new_items": len(chamber_new),
            "total_items": len(chamber_items),
            "newest_date": max(chamber_dates).isoformat() if chamber_dates else None,
            "oldest_date": min(chamber_dates).isoformat() if chamber_dates else None,
        }
        derived_per_chamber[chamber] = dict(expected)
        actual = per_chamber.get(chamber)
        if not isinstance(actual, dict):
            errors.append(f"manifest.per_chamber[{chamber!r}]: expected an object")
            continue
        for field, value in expected.items():
            if actual.get(field) != value:
                errors.append(
                    f"manifest.per_chamber[{chamber!r}].{field}: "
                    f"expected {value!r}, got {actual.get(field)!r}"
                )
        cursor = actual.get("resume_cursor")
        if cursor is not None:
            _iso_date(
                cursor,
                label=f"manifest.per_chamber[{chamber!r}].resume_cursor",
                errors=errors,
            )
        declared_chamber_cursors[chamber] = cursor

    (
        expected_cursors,
        expected_global_frontier,
        completed_known,
        completed_new,
        represented_window_failures,
    ) = _validate_completed_windows(
        manifest,
        frontier=frontier,
        lower_bound=lower_bound,
        start_cursors=start_cursors,
        has_resume_parent=has_resume_parent,
        errors=errors,
    )
    for chamber in CHAMBER_TO_PALATA:
        declared = declared_chamber_cursors.get(chamber)
        expected = expected_cursors[chamber]
        if declared != expected:
            errors.append(
                f"manifest.per_chamber[{chamber!r}].resume_cursor: "
                f"completed windows derive {expected!r}, got {declared!r}"
            )
        derived_per_chamber[chamber]["resume_cursor"] = expected
    if completed_new > len(journal):
        errors.append(
            "manifest.completed_windows: completed new count exceeds journal records"
        )
    if known_encountered is not None and completed_known > known_encountered:
        errors.append(
            "manifest.completed_windows: completed known count exceeds "
            "manifest.known_encountered"
        )

    resume_cursors = manifest.get("per_chamber_resume_cursors")
    if resume_cursors != expected_cursors:
        errors.append(
            "manifest.per_chamber_resume_cursors: must exactly match cursors derived "
            "from completed windows"
        )
    global_frontier = manifest.get("oldest_fully_completed_global_date_frontier")
    parsed_global_frontier = _iso_date(
        global_frontier,
        label="manifest.oldest_fully_completed_global_date_frontier",
        errors=errors,
    )
    if (
        parsed_global_frontier is not None
        and frontier is not None
        and lower_bound is not None
        and not lower_bound <= parsed_global_frontier <= frontier
    ):
        errors.append(
            "manifest.oldest_fully_completed_global_date_frontier: "
            "lies outside manifest bounds"
        )
    if global_frontier != expected_global_frontier:
        errors.append(
            "manifest.oldest_fully_completed_global_date_frontier: completed windows "
            f"derive {expected_global_frontier!r}, got {global_frontier!r}"
        )

    retries = manifest.get("retries")
    retry_fields = {"total", "retry_after", "parse", "detail_parse"}
    if not isinstance(retries, dict) or set(retries) != retry_fields:
        errors.append(
            "manifest.retries: keys must be exactly total, retry_after, parse, "
            "detail_parse"
        )
        retries = {}
    validated_retries = {}
    for field in sorted(retry_fields):
        value = retries.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"manifest.retries.{field}: expected a nonnegative integer")
        else:
            validated_retries[field] = value

    unresolved_count = _manifest_int(
        manifest,
        "unresolved_failure_count",
        label="manifest",
        errors=errors,
    )
    unresolved = manifest.get("unresolved_failures")
    if not isinstance(unresolved, list):
        errors.append("manifest.unresolved_failures: expected a list")
        unresolved = []
    elif any(not isinstance(record, dict) for record in unresolved):
        errors.append("manifest.unresolved_failures: every record must be an object")
    unresolved_truncated = manifest.get("unresolved_failures_truncated")
    if not isinstance(unresolved_truncated, bool):
        errors.append("manifest.unresolved_failures_truncated: expected a boolean")
    elif unresolved_count is not None:
        if unresolved_truncated and unresolved_count <= len(unresolved):
            errors.append(
                "manifest unresolved failures: truncated=true requires count to "
                "exceed the retained list length"
            )
        if not unresolved_truncated and unresolved_count != len(unresolved):
            errors.append(
                "manifest unresolved failures: untruncated count must equal the "
                "retained list length"
            )
        if represented_window_failures > unresolved_count:
            errors.append(
                "manifest.completed_windows: represented failures exceed "
                "unresolved_failure_count"
            )

    if frontier is not None and lower_bound is not None:
        for record in items:
            if not lower_bound <= record.decision_date <= frontier:
                errors.append(
                    f"items.jsonl: {record.identity!r} date "
                    f"{record.decision_date.isoformat()} lies outside manifest bounds"
                )

    target_body_hits = sum(record.target_body_hit for record in items)
    if target_body_hits == 0:
        errors.append(
            f"items.jsonl: target first-instance literal {TARGET_LITERAL} "
            "was not found in any full decision body"
        )

    if errors:
        raise ValidationError(errors)

    all_dates = [record.decision_date for record in items]
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "run_id": manifest["run_id"],
        "run_dir": str(run_path),
        "started_at": manifest["started_at"],
        "finished_at": manifest["finished_at"],
        "finish_reason": manifest["finish_reason"],
        "frontier_start_date": manifest["frontier_start_date"],
        "lower_bound": manifest["lower_bound"],
        "max_runtime_seconds": manifest["max_runtime_seconds"],
        "elapsed_time_seconds": elapsed_time,
        "partial_by_design": True,
        "per_chamber_start_cursors": {
            chamber: value.isoformat() if value is not None else None
            for chamber, value in start_cursors.items()
        },
        "resume_parent": manifest.get("resume_parent"),
        "items_sha256": items_sha256,
        "journal_sha256": _journal_sha256,
        "total_items": len(items),
        "known_items": derived_known,
        "new_items": len(journal),
        "journal_items": len(journal),
        "newest_date": max(all_dates).isoformat() if all_dates else None,
        "oldest_date": min(all_dates).isoformat() if all_dates else None,
        "per_chamber": derived_per_chamber,
        "oldest_fully_completed_global_date_frontier": global_frontier,
        "per_chamber_resume_cursors": expected_cursors,
        "retries": validated_retries,
        "unresolved_failure_count": unresolved_count,
        "target_literal": TARGET_LITERAL,
        "target_body_hits": target_body_hits,
        "target_case_number_hits": 0,
        "errors": [],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="final artifacts/supremecourt/runs/<run-id> directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_run(args.run_dir)
    except ValidationError as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "run_dir": str(_absolute_lexical(args.run_dir.expanduser())),
            "errors": list(exc.errors),
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
