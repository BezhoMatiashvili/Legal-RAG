"""Create-only early-stop handoff for the immutable 512-token candidate.

This artifact is used only when the mandatory release-input gate is inconclusive.  It
cannot represent a completed crawl, collection, evaluation, or promotion and deliberately
uses explicit unavailable markers instead of placeholder hashes or invented metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .release_inputs import (
    GENERATION_ID,
    PHYSICAL_COLLECTION,
    RELEASE_BUNDLE_PATH,
    SNAPSHOT_ID,
    _current_repository_revision,
)

SCHEMA_VERSION = "immutable-512-inconclusive-handoff/v1"
METRICS = (
    "success_at_1",
    "success_at_5",
    "success_at_10",
    "candidate_recall_at_50",
    "candidate_recall_at_80",
    "required_evidence_recall_at_10",
    "correct_document_identity",
    "passage_accuracy",
    "evidence_span_coverage_at_10",
    "context_duplication",
    "context_noise",
)
RETRIEVAL_ONLY_DISCLAIMER = (
    "This verdict concerns retrieval and canonical evidence selection only; it does not "
    "establish legal-answer accuracy."
)
INITIAL_REPOSITORY_HEAD = "1e8a0d78659503a5b3572114d81b430bd98f1952"
INITIAL_INVENTORY_SHA256 = (
    "1015e6dc705d32824ca77ba8a7b8c8a3b04f67eff7eab47c62280044cf38ba9d"
)
INITIAL_STATUS_SHA256 = (
    "9a068a94033f11cc0e4a58d0cdd2f16f406886ad839c163e9fcdb4ae6682b39a"
)
INITIAL_SYMBOLS_SHA256 = (
    "11737db8bed587ac2ceae358ed38e680da49267c0bf41e799b13ae3c86331afa"
)
INITIAL_PRESENTATION_SHA256 = (
    "87e5d1b07e79cbcad22ddeaed0b2dbd51301f75c198f47b3f7f62007f3fa37dd"
)
INITIAL_PRESENTATION_FILES = (
    "accuracy-first-demo/README.md",
    "accuracy-first-demo/demo-script.md",
    "accuracy-first-demo/limitations.md",
    "accuracy-first-demo/questions.jsonl",
    "accuracy-first-demo/questions.sha256",
    "accuracy-first-demo/reviewer-checklist.md",
    "accuracy-first-demo/run-1.json",
    "accuracy-first-demo/run-2.json",
    "accuracy-first-demo/scorecard.json",
    "accuracy-first-demo/scorecard.md",
    "accuracy-first-demo/slides-outline.md",
    "accuracy-first-demo/talk-track.md",
)
_VALIDATION_REPORT_KEYS = {
    "schema_version",
    "generation_id",
    "status",
    "ceiling",
    "checked_at",
    "bundle",
    "reason",
}
_INVENTORY_KEYS = {
    "schema_version",
    "recorded_at",
    "repository_head",
    "git_status_porcelain_v2_sha256",
    "git_status_entry_count",
    "untracked_entry_count",
    "deleted_entry_count",
    "index_entry_count",
    "accepted_generated_symbols_sha256",
    "presentation",
    "preservation_contract",
}
_UTC_SECONDS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CODE_MAP_CHECK_NAME = "code map regeneration and check"


class CandidateHandoffError(ValueError):
    """The early-stop evidence is malformed or the safety boundary drifted."""


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            info = cursor.lstat()
        except OSError as exc:
            raise CandidateHandoffError(f"cannot inspect {label}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise CandidateHandoffError(f"{label} path contains a symlink")
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise CandidateHandoffError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CandidateHandoffError(f"{label} must be a real regular file")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CandidateHandoffError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        raw = absolute.read_bytes()
    except OSError as exc:
        raise CandidateHandoffError(f"cannot read {label}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateHandoffError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateHandoffError(f"{label} must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CandidateHandoffError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _validate_stop_report(
    value: Mapping[str, Any], *, path: Path, repo_root: Path
) -> str:
    _exact_keys(value, _VALIDATION_REPORT_KEYS, label="release-input report")
    expected_parent = (
        repo_root / "ingest/.state/v3/release-safety" / GENERATION_ID
    ).absolute()
    absolute = path.expanduser().absolute()
    if (
        absolute.parent != expected_parent
        or re.fullmatch(r"release-input-validation-[0-9]+\.json", absolute.name) is None
    ):
        raise CandidateHandoffError(
            "release-input report is not in the immutable candidate safety root"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise CandidateHandoffError("release-input report schema_version is invalid")
    expected = {
        "generation_id": GENERATION_ID,
        "status": "inconclusive",
        "ceiling": "no_network_or_paid_work",
        "bundle": RELEASE_BUNDLE_PATH,
    }
    if any(
        value[field] != expected_value for field, expected_value in expected.items()
    ):
        raise CandidateHandoffError(
            "handoff requires the exact official inconclusive pre-network release gate"
        )
    checked_at = value["checked_at"]
    if not isinstance(checked_at, str) or _UTC_SECONDS_RE.fullmatch(checked_at) is None:
        raise CandidateHandoffError(
            "release-input report checked_at is not UTC seconds"
        )
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
        raise CandidateHandoffError(
            "release-input gate lacks a concrete bounded reason"
        )
    return reason


def _validate_initial_inventory(
    value: Mapping[str, Any], *, digest: str, path: Path, repo_root: Path
) -> Mapping[str, Any]:
    expected_path = (
        repo_root
        / "ingest/.state/v3/release-safety"
        / GENERATION_ID
        / "initial-worktree-inventory.json"
    ).absolute()
    if path.expanduser().absolute() != expected_path:
        raise CandidateHandoffError(
            "initial safety inventory is not the official candidate inventory"
        )
    if digest != INITIAL_INVENTORY_SHA256:
        raise CandidateHandoffError("initial safety inventory bytes are not official")
    _exact_keys(value, _INVENTORY_KEYS, label="initial safety inventory")
    expected_scalars: dict[str, object] = {
        "schema_version": 1,
        "recorded_at": "2026-07-17T12:41:00Z",
        "repository_head": INITIAL_REPOSITORY_HEAD,
        "git_status_porcelain_v2_sha256": INITIAL_STATUS_SHA256,
        "git_status_entry_count": 166,
        "untracked_entry_count": 59,
        "deleted_entry_count": 8,
        "index_entry_count": 8,
        "accepted_generated_symbols_sha256": INITIAL_SYMBOLS_SHA256,
    }
    if any(
        type(value[field]) is not type(expected) or value[field] != expected
        for field, expected in expected_scalars.items()
    ):
        raise CandidateHandoffError("initial safety inventory identity drifted")
    presentation = value["presentation"]
    if not isinstance(presentation, Mapping):
        raise CandidateHandoffError(
            "initial safety inventory lacks presentation evidence"
        )
    _exact_keys(
        presentation,
        {"tree_sha256", "file_count", "files"},
        label="presentation inventory",
    )
    if (
        presentation["tree_sha256"] != INITIAL_PRESENTATION_SHA256
        or type(presentation["file_count"]) is not int
        or presentation["file_count"] != len(INITIAL_PRESENTATION_FILES)
        or presentation["files"] != list(INITIAL_PRESENTATION_FILES)
    ):
        raise CandidateHandoffError("initial presentation inventory drifted")
    contract = value["preservation_contract"]
    expected_contract = {
        "preserve_all_existing_changes_and_deletions": True,
        "presentation_tree_must_remain_unchanged": True,
        "generated_symbols_working_copy_accepted": True,
    }
    if (
        not isinstance(contract, Mapping)
        or any(
            type(contract.get(field)) is not bool or contract.get(field) is not expected
            for field, expected in expected_contract.items()
        )
        or set(contract) != set(expected_contract)
    ):
        raise CandidateHandoffError("initial preservation contract drifted")
    return presentation


def _sha256_regular_file(path: Path, *, label: str) -> str:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CandidateHandoffError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CandidateHandoffError(f"{label} must be a real regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CandidateHandoffError(f"cannot read {label}: {exc}") from exc
    return digest.hexdigest()


def _assert_candidate_artifacts_absent(repo_root: Path) -> dict[str, str]:
    paths = {
        "source_evidence": repo_root / "ingest/.state/v3/source-evidence" / SNAPSHOT_ID,
        "snapshot": repo_root / "ingest/snapshots/v3" / SNAPSHOT_ID,
        "generation": repo_root / "artifacts/generations" / GENERATION_ID,
        "gpu_workflow": repo_root / "ingest/.state/v3/gpu" / GENERATION_ID,
        "evaluation": repo_root / "ingest/.state/v3/evaluation" / GENERATION_ID,
        "prepared_generation": repo_root / "ingest/.state/v3/prepared" / GENERATION_ID,
        "embed_state": repo_root / "ingest/.state/embed" / GENERATION_ID,
        "verification_sidecar_1": repo_root
        / "artifacts/generations"
        / f"{GENERATION_ID}.physical-v3-512-01.verification.json",
        "verification_sidecar_2": repo_root
        / "artifacts/generations"
        / f"{GENERATION_ID}.physical-v3-512-02.verification.json",
    }
    observed: dict[str, str] = {}
    for label, path in paths.items():
        absolute = path.absolute()
        cursor = Path(absolute.anchor)
        for component in absolute.parts[1:-1]:
            cursor /= component
            if os.path.lexists(cursor) and cursor.is_symlink():
                raise CandidateHandoffError(
                    f"candidate artifact probe path contains a symlink: {label}"
                )
        if os.path.lexists(absolute):
            raise CandidateHandoffError(
                f"candidate artifact exists despite the pre-network stop: {label}"
            )
        observed[label] = str(absolute)
    return observed


def _normalize_checks(
    checks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(checks):
        row = dict(raw)
        if set(row) != {"name", "command", "status", "result"}:
            raise CandidateHandoffError(f"check {index} has the wrong fields")
        if row["status"] not in {"passed", "failed", "blocked", "not_run"}:
            raise CandidateHandoffError(f"check {index} has an invalid status")
        if not all(isinstance(row[field], str) and row[field] for field in row):
            raise CandidateHandoffError(f"check {index} contains an empty field")
        normalized.append(row)
    return normalized


def _validate_code_map_transition(
    checks: Sequence[Mapping[str, object]],
    *,
    current_sha256: str,
    expected_current_sha256: str | None,
) -> tuple[str, Mapping[str, object]]:
    if expected_current_sha256 is None:
        if current_sha256 != INITIAL_SYMBOLS_SHA256:
            raise CandidateHandoffError(
                "changed generated symbols require an explicit expected current SHA-256"
            )
        expected_current_sha256 = INITIAL_SYMBOLS_SHA256
    if _SHA256_RE.fullmatch(expected_current_sha256) is None:
        raise CandidateHandoffError(
            "expected current generated-symbols SHA-256 is invalid"
        )
    if current_sha256 != expected_current_sha256:
        raise CandidateHandoffError(
            "current generated symbols do not match the reviewed expected SHA-256"
        )
    matching = [row for row in checks if row["name"] == CODE_MAP_CHECK_NAME]
    if len(matching) != 1:
        raise CandidateHandoffError(
            f"exactly one {CODE_MAP_CHECK_NAME!r} check is required"
        )
    check = matching[0]
    expected_result = (
        f"current_symbols_sha256={current_sha256}; regeneration=passed; check=passed"
    )
    if check["status"] != "passed" or check["result"] != expected_result:
        raise CandidateHandoffError(
            "code map regeneration/check evidence is not passed and digest-bound"
        )
    return expected_current_sha256, check


def _presentation_hash(
    repo_root: Path, expected_files: Sequence[object]
) -> tuple[str, list[str]]:
    root = repo_root / "presentation"
    if root.is_symlink() or not root.is_dir():
        raise CandidateHandoffError("presentation/ must remain a real directory")
    declared = [str(value) for value in expected_files]
    if any(
        not value or Path(value).is_absolute() or ".." in Path(value).parts
        for value in declared
    ):
        raise CandidateHandoffError(
            "initial presentation inventory contains an unsafe path"
        )
    observed = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    if observed != declared:
        raise CandidateHandoffError("presentation tree file inventory changed")
    sha_lines = bytearray()
    for relative in observed:
        path = root / relative
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise CandidateHandoffError("presentation tree contains an unsafe entry")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sha_lines.extend(f"{digest}  presentation/{relative}\n".encode("utf-8"))
    return hashlib.sha256(sha_lines).hexdigest(), observed


def _create_json(path: Path, value: object) -> Path:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"handoff already exists: {destination}")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"handoff already exists: {destination}") from None
        temporary.unlink()
        parent = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def create_inconclusive_handoff(
    *,
    validation_report: Path,
    safety_inventory: Path,
    checks: Sequence[Mapping[str, object]],
    output: Path,
    repo_root: Path,
    created_at: datetime | None = None,
    expected_current_symbols_sha256: str | None = None,
) -> Path:
    """Create a fail-closed handoff after the pre-network release gate stops."""

    resolved_repo_root = repo_root.expanduser().absolute()
    validation, validation_sha = _read_json(
        validation_report, label="release-input validation report"
    )
    reason = _validate_stop_report(
        validation, path=validation_report, repo_root=resolved_repo_root
    )
    safety, safety_sha = _read_json(safety_inventory, label="initial safety inventory")
    presentation = _validate_initial_inventory(
        safety,
        digest=safety_sha,
        path=safety_inventory,
        repo_root=resolved_repo_root,
    )
    if _current_repository_revision(resolved_repo_root) != INITIAL_REPOSITORY_HEAD:
        raise CandidateHandoffError(
            "repository HEAD changed from the initial inventory"
        )
    symbols_sha = _sha256_regular_file(
        resolved_repo_root / "memory-bank/generated/symbols.md",
        label="current generated symbols working copy",
    )
    current_presentation_hash, current_files = _presentation_hash(
        resolved_repo_root, presentation["files"]
    )
    if current_presentation_hash != INITIAL_PRESENTATION_SHA256:
        raise CandidateHandoffError("presentation tree bytes changed")
    absent_paths = _assert_candidate_artifacts_absent(resolved_repo_root)
    normalized_checks = _normalize_checks(checks)
    expected_symbols_sha, code_map_check = _validate_code_map_transition(
        normalized_checks,
        current_sha256=symbols_sha,
        expected_current_sha256=expected_current_symbols_sha256,
    )

    unavailable_hashes = {
        name: {"status": "unavailable", "sha256": None}
        for name in (
            "snapshot",
            "generation",
            "configuration",
            "whole_collection",
            "vector_checksum_artifact",
            "qrel_adapter",
            "verification_1",
            "verification_2",
            "production_repeat_1",
            "production_repeat_2",
            "accuracy_strict_repeat_1",
            "accuracy_strict_repeat_2",
            "comparison",
        )
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at": (created_at or datetime.now(UTC))
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "inconclusive",
        "release": {
            "snapshot_id": SNAPSHOT_ID,
            "generation_id": GENERATION_ID,
            "physical_collection": PHYSICAL_COLLECTION,
        },
        "stop_gate": {
            "ceiling": "no_network_or_paid_work",
            "reason": reason,
            "report_path": str(validation_report.expanduser().absolute()),
            "report_sha256": validation_sha,
        },
        "safety": {
            "initial_inventory_path": str(safety_inventory.expanduser().absolute()),
            "initial_inventory_sha256": safety_sha,
            "initial_presentation_sha256": presentation["tree_sha256"],
            "current_presentation_sha256": current_presentation_hash,
            "presentation_file_count": len(current_files),
            "presentation_unchanged": True,
            "repository_head": INITIAL_REPOSITORY_HEAD,
            "repository_head_unchanged": True,
            "initial_git_status_porcelain_v2_sha256": INITIAL_STATUS_SHA256,
            "initial_accepted_generated_symbols_sha256": INITIAL_SYMBOLS_SHA256,
            "current_generated_symbols_sha256": symbols_sha,
            "expected_current_generated_symbols_sha256": expected_symbols_sha,
            "generated_symbols_changed_since_initial": (
                symbols_sha != INITIAL_SYMBOLS_SHA256
            ),
            "code_map_regeneration_check": code_map_check,
            "preservation_basis": (
                "The immutable inventory hash-binds the accepted starting copy. The "
                "current copy is accepted only because its reviewed SHA-256 matches "
                "explicit passed regeneration and --check evidence."
            ),
            "aggregate_inventory_validated": True,
            "individual_initial_worktree_entry_preservation": {
                "status": "not_independently_verifiable",
                "reason": (
                    "the immutable initial inventory contains aggregate status counts and "
                    "a hash, not the individual porcelain-v2 entries"
                ),
            },
            "verified_absent_local_candidate_paths": absent_paths,
        },
        "artifacts_and_hashes": unavailable_hashes,
        "source_and_quarantine_counts": {
            "status": "unavailable",
            "reason": (
                "the release gate stopped before authorization and all candidate-local "
                "evidence roots were verified absent"
            ),
        },
        "metrics": {
            metric: {
                "baseline": None,
                "candidate": None,
                "paired_delta": None,
                "confidence_interval_95": None,
                "sample_count": 0,
                "status": "unavailable",
            }
            for metric in METRICS
        },
        "reproducibility": {
            "status": "not_run",
            "production_repeats": 0,
            "accuracy_strict_repeats": 0,
        },
        "commands_and_checks": normalized_checks,
        "external_resource_usage": {
            "scope": "actions_performed_in_this_session",
            "network_crawls": 0,
            "model_downloads": 0,
            "paid_compute_usd": 0.0,
            "remote_gpu_hours": 0.0,
            "qdrant_mutations": 0,
            "local_restore_attempts": 0,
            "external_state_observability": (
                "not_observable_from_supplied_local_evidence"
            ),
        },
        "remaining_limitations": [
            reason,
            "No local qualifying source ledger or sealed candidate snapshot exists.",
            (
                "Remote provider usage and Qdrant state are not observable from the "
                "supplied local evidence."
            ),
            (
                "The aggregate initial worktree inventory cannot independently prove "
                "preservation of each original dirty-tree entry."
            ),
        ],
        "next_experiment": {
            "status": "not_selectable",
            "reason": "measured error attribution is unavailable before paired evaluation",
            "selected_experiment": None,
        },
        "promotion": {
            "scope": "actions_performed_in_this_session",
            "plan_created": False,
            "alias_operations": 0,
            "legacy_collection_touched": False,
            "external_alias_state_observability": (
                "not_observable_from_supplied_local_evidence"
            ),
        },
        "verdict": "inconclusive",
        "scope_disclaimer": RETRIEVAL_ONLY_DISCLAIMER,
    }
    return _create_json(output, artifact)


__all__ = [
    "CandidateHandoffError",
    "METRICS",
    "SCHEMA_VERSION",
    "create_inconclusive_handoff",
]
