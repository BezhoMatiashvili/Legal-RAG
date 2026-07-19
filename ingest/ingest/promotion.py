"""Persisted, reversible blue/green promotion state machine.

The executor is backend-agnostic: production adapters must implement ``PromotionBackend``
and an atomic alias switch, while tests use an in-memory fake.  The state machine never
deletes a collection.  It restores and verifies a generation-specific physical candidate,
switches the stable alias forward, switches it back to prove rollback, and leaves the
previous collection serving unless a second explicit forward approval is supplied.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .artifacts import atomic_write_json, load_verified_generation_coverage
from .config import RETRIEVAL_FINGERPRINT_REVISION
from .generation import (
    MANIFEST_FILENAME,
    GenerationFormatError,
    load_generation,
    parse_rfc3339_utc,
    validate_generation_id,
)

PROMOTION_SCHEMA_VERSION = 1
SERVING_ALIAS = "georgian_legal"
PHYSICAL_COLLECTION_PREFIX = "georgian_legal__gen_"
PROMOTION_APPROVAL_ENV = "PROMOTION_APPROVED"
FORWARD_APPROVAL_ENV = "PROMOTION_FORWARD_APPROVED"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,127}$")
_PHASES = frozenset(
    {
        "planned",
        "candidate_restored",
        "candidate_verified",
        "checks_passed",
        "forward_switched",
        "rollback_switched",
        "rollback_proven",
        "completed_forward",
        "failed",
    }
)


class PromotionError(RuntimeError):
    """Base class for fail-closed promotion failures."""


class PromotionPreconditionError(PromotionError):
    """Candidate identity, integrity, readiness, or alias state is incompatible."""


class PromotionLockedError(PromotionError):
    """Another local publisher already owns the promotion lock."""


class PromotionPlanExists(PromotionError):
    """An immutable promotion plan already exists at the requested path."""


def refuse_frozen_candidate_promotion(
    generation_id: str, physical_collection: str
) -> None:
    """Keep the exact 512-token candidate outside every alias-capable workflow."""

    # Import lazily so the generic promotion module remains independent of release-input
    # validation at import time.
    from .release_inputs import GENERATION_ID, PHYSICAL_COLLECTION

    if generation_id == GENERATION_ID and physical_collection == PHYSICAL_COLLECTION:
        raise PromotionPreconditionError(
            "the frozen 512-token candidate cannot enter a promotion plan or restore; "
            "this release explicitly forbids promotion and alias operations"
        )


def _require_string(value: object, field: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(char) < 32 for char in value)
    ):
        raise PromotionError(f"{field} must be a non-empty safe string")
    return value


def _require_model_name(value: object, field: str) -> str:
    model = _require_string(value, field)
    if not model.strip():
        raise PromotionError(f"{field} must be a non-empty model identity")
    return model


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PromotionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, field: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise PromotionError(
            f"{field} must be an immutable lowercase hexadecimal revision"
        )
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PromotionError(f"{field} must be an integer >= 0")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise PromotionError(f"{field} must be a boolean")
    return value


def _require_timestamp(value: object, field: str) -> str:
    try:
        parse_rfc3339_utc(value, field=field)
    except GenerationFormatError as exc:
        raise PromotionError(str(exc)) from exc
    return str(value)


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise PromotionError(
            f"{field} keys mismatch: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise PromotionError(f"promotion artifact is not strict JSON: {exc}") from exc


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise PromotionError("promotion timestamps must be timezone-aware")
    return (
        value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def physical_collection_name(generation_id: str) -> str:
    """Return the only permitted physical collection name for a generation."""
    return f"{PHYSICAL_COLLECTION_PREFIX}{validate_generation_id(generation_id)}"


@dataclass(frozen=True, slots=True)
class ExpectedCollectionIdentity:
    payload_schema_version: int
    points_count: int
    dense_name: str
    dense_dimension: int
    distance: str
    sparse_name: str
    embedding_model: str
    embedding_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    reranker_model: str
    reranker_revision: str
    vector_space_id: str
    chunking_fingerprint: str
    document_header: bool
    retrieval_fingerprint_revision: int
    retrieval_fingerprint: str

    @classmethod
    def from_dict(cls, value: object) -> ExpectedCollectionIdentity:
        if not isinstance(value, Mapping):
            raise PromotionError("expected_collection must be an object")
        expected = {
            "payload_schema_version",
            "points_count",
            "dense_name",
            "dense_dimension",
            "distance",
            "sparse_name",
            "embedding_model",
            "embedding_revision",
            "tokenizer_model",
            "tokenizer_revision",
            "reranker_model",
            "reranker_revision",
            "vector_space_id",
            "chunking_fingerprint",
            "document_header",
            "retrieval_fingerprint_revision",
            "retrieval_fingerprint",
        }
        _exact_keys(value, expected, "expected_collection")
        identity = cls(
            payload_schema_version=_require_int(
                value["payload_schema_version"], "payload_schema_version"
            ),
            points_count=_require_int(value["points_count"], "points_count"),
            dense_name=_require_string(value["dense_name"], "dense_name"),
            dense_dimension=_require_int(value["dense_dimension"], "dense_dimension"),
            distance=_require_string(value["distance"], "distance").lower(),
            sparse_name=_require_string(value["sparse_name"], "sparse_name"),
            embedding_model=_require_model_name(
                value["embedding_model"], "embedding_model"
            ),
            embedding_revision=_require_revision(
                value["embedding_revision"], "embedding_revision"
            ),
            tokenizer_model=_require_model_name(
                value["tokenizer_model"], "tokenizer_model"
            ),
            tokenizer_revision=_require_revision(
                value["tokenizer_revision"], "tokenizer_revision"
            ),
            reranker_model=_require_model_name(
                value["reranker_model"], "reranker_model"
            ),
            reranker_revision=_require_revision(
                value["reranker_revision"], "reranker_revision"
            ),
            vector_space_id=_require_sha256(
                value["vector_space_id"], "vector_space_id"
            ),
            chunking_fingerprint=_require_sha256(
                value["chunking_fingerprint"], "chunking_fingerprint"
            ),
            document_header=_require_bool(value["document_header"], "document_header"),
            retrieval_fingerprint_revision=_require_int(
                value["retrieval_fingerprint_revision"],
                "retrieval_fingerprint_revision",
            ),
            retrieval_fingerprint=_require_sha256(
                value["retrieval_fingerprint"], "retrieval_fingerprint"
            ),
        )
        if identity.retrieval_fingerprint_revision != RETRIEVAL_FINGERPRINT_REVISION:
            raise PromotionError(
                "retrieval_fingerprint_revision must equal "
                f"{RETRIEVAL_FINGERPRINT_REVISION}"
            )
        return identity

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    schema_version: int
    promotion_id: str
    generation_id: str
    manifest_sha256: str
    verification_report_sha256: str
    snapshot_ref: str
    snapshot_sha256: str
    serving_alias: str
    physical_collection: str
    expected_collection: ExpectedCollectionIdentity
    created_at: str
    created_by: str

    @classmethod
    def from_dict(cls, value: object) -> PromotionPlan:
        if not isinstance(value, Mapping):
            raise PromotionError("promotion plan must be an object")
        expected = {
            "schema_version",
            "promotion_id",
            "generation_id",
            "manifest_sha256",
            "verification_report_sha256",
            "snapshot_ref",
            "snapshot_sha256",
            "serving_alias",
            "physical_collection",
            "expected_collection",
            "created_at",
            "created_by",
        }
        _exact_keys(value, expected, "promotion plan")
        if value["schema_version"] != PROMOTION_SCHEMA_VERSION:
            raise PromotionError("unsupported promotion plan schema_version")
        promotion_id = _require_string(
            value["promotion_id"], "promotion_id", maximum=128
        )
        if not _ID_RE.fullmatch(promotion_id):
            raise PromotionError("promotion_id has an invalid format")
        generation_id = validate_generation_id(value["generation_id"])
        physical = _require_string(value["physical_collection"], "physical_collection")
        if physical != physical_collection_name(generation_id):
            raise PromotionError(
                "physical_collection is not derived from generation_id"
            )
        refuse_frozen_candidate_promotion(generation_id, physical)
        if value["serving_alias"] != SERVING_ALIAS:
            raise PromotionError(f"serving_alias must be {SERVING_ALIAS!r}")
        return cls(
            schema_version=PROMOTION_SCHEMA_VERSION,
            promotion_id=promotion_id,
            generation_id=generation_id,
            manifest_sha256=_require_sha256(
                value["manifest_sha256"], "manifest_sha256"
            ),
            verification_report_sha256=_require_sha256(
                value["verification_report_sha256"],
                "verification_report_sha256",
            ),
            snapshot_ref=_require_string(
                value["snapshot_ref"], "snapshot_ref", maximum=8192
            ),
            snapshot_sha256=_require_sha256(
                value["snapshot_sha256"], "snapshot_sha256"
            ),
            serving_alias=SERVING_ALIAS,
            physical_collection=physical,
            expected_collection=ExpectedCollectionIdentity.from_dict(
                value["expected_collection"]
            ),
            created_at=_require_timestamp(value["created_at"], "created_at"),
            created_by=_require_string(value["created_by"], "created_by"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "promotion_id": self.promotion_id,
            "generation_id": self.generation_id,
            "manifest_sha256": self.manifest_sha256,
            "verification_report_sha256": self.verification_report_sha256,
            "snapshot_ref": self.snapshot_ref,
            "snapshot_sha256": self.snapshot_sha256,
            "serving_alias": self.serving_alias,
            "physical_collection": self.physical_collection,
            "expected_collection": self.expected_collection.to_dict(),
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class CollectionInspection:
    """Backend observation used for exact candidate compatibility checks."""

    name: str
    payload_schema_version: int
    points_count: int
    dense_name: str
    dense_dimension: int
    distance: str
    sparse_name: str
    generation_id: str
    manifest_sha256: str
    embedding_model: str
    embedding_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    reranker_model: str
    reranker_revision: str
    vector_space_id: str
    chunking_fingerprint: str
    document_header: bool
    retrieval_fingerprint_revision: int
    retrieval_fingerprint: str
    optimizer_status: str
    integrity_ok: bool
    verification_provenance_sha256: str | None = None


class PromotionBackend(Protocol):
    """Mutation boundary; implementations must not overwrite/delete collections."""

    def restore_candidate(self, plan: PromotionPlan) -> None:
        """Restore or idempotently retain only ``plan.physical_collection``."""

    def wait_for_green(
        self, collection: str, timeout_seconds: float
    ) -> CollectionInspection:
        """Wait for optimizer green and return the final exact inspection."""

    def smoke(self, collection: str) -> bool:
        """Run bounded retrieval smoke checks against a physical collection."""

    def readiness(self, collection: str) -> bool:
        """Confirm the named physical collection can safely serve queries."""

    def alias_target(self, alias: str) -> str | None:
        """Return the alias's current physical target without mutation."""

    def switch_alias(self, alias: str, collection: str) -> None:
        """Atomically switch one stable alias; never delete either collection."""


def create_promotion_plan(
    generation_root: str | Path,
    *,
    snapshot_ref: str,
    snapshot_sha256: str,
    created_by: str,
    created_at: datetime | None = None,
    promotion_id: str | None = None,
) -> PromotionPlan:
    """Create an in-memory plan only after local generation + four-gate verification."""
    supplied_root = Path(generation_root)
    if supplied_root.is_symlink():
        raise PromotionPreconditionError("generation root cannot be a symlink")
    root = supplied_root.resolve()
    _require_private_generation_tree(root)
    artifacts = load_generation(root)
    manifest = artifacts.manifest
    physical_collection = physical_collection_name(manifest.generation_id)
    # Refuse before building any alias-capable plan or consulting its promotion sidecar.
    refuse_frozen_candidate_promotion(manifest.generation_id, physical_collection)
    sidecar = root.with_name(f"{root.name}.verification.json")
    _require_private_regular_file(sidecar, "verification sidecar")
    verification_report_sha256 = _file_sha256(sidecar)
    verified = load_verified_generation_coverage(root.parent)
    proof = next(
        (
            item
            for item in verified
            if item.generation_id == manifest.generation_id
            and item.manifest_path == (root / MANIFEST_FILENAME).resolve()
        ),
        None,
    )
    if proof is None:
        raise PromotionPreconditionError(
            "generation lacks a matching all-green verification sidecar"
        )
    if _file_sha256(sidecar) != verification_report_sha256:
        raise PromotionPreconditionError(
            "verification sidecar changed while the promotion plan was built"
        )
    manifest_sha256 = artifacts.checksums.files[MANIFEST_FILENAME]
    if _file_sha256(root / MANIFEST_FILENAME) != manifest_sha256:
        raise PromotionPreconditionError(
            "generation manifest changed while the promotion plan was built"
        )
    generated_promotion_id = promotion_id or f"promotion-{uuid.uuid4().hex}"
    vector = manifest.vector_space
    model = manifest.model
    chunking = manifest.chunking
    plan = PromotionPlan(
        schema_version=PROMOTION_SCHEMA_VERSION,
        promotion_id=generated_promotion_id,
        generation_id=manifest.generation_id,
        manifest_sha256=manifest_sha256,
        verification_report_sha256=verification_report_sha256,
        snapshot_ref=snapshot_ref,
        snapshot_sha256=snapshot_sha256,
        serving_alias=SERVING_ALIAS,
        physical_collection=physical_collection,
        expected_collection=ExpectedCollectionIdentity(
            payload_schema_version=manifest.schema_version,
            points_count=manifest.chunk_count,
            dense_name=vector.dense_name,
            dense_dimension=vector.dense_dimension,
            distance=vector.distance,
            sparse_name=vector.sparse_name,
            embedding_model=model.embedding_model,
            embedding_revision=model.embedding_revision,
            tokenizer_model=model.tokenizer_model,
            tokenizer_revision=model.tokenizer_revision,
            reranker_model=model.reranker_model,
            reranker_revision=model.reranker_revision,
            vector_space_id=vector.id,
            chunking_fingerprint=chunking.fingerprint,
            document_header=chunking.document_header,
            retrieval_fingerprint_revision=manifest.retrieval_fingerprint_revision,
            retrieval_fingerprint=manifest.retrieval_fingerprint,
        ),
        created_at=_format_utc(created_at or datetime.now(UTC)),
        created_by=created_by,
    )
    return PromotionPlan.from_dict(plan.to_dict())


def promotion_plan_sha256(plan: PromotionPlan) -> str:
    return hashlib.sha256(_canonical_bytes(plan.to_dict())).hexdigest()


def write_promotion_plan(path: str | Path, plan: PromotionPlan) -> Path:
    """Persist a new owner-only plan, refusing replacement of any existing entry."""
    validated = PromotionPlan.from_dict(plan.to_dict())
    destination = Path(path)
    if os.path.lexists(destination):
        raise PromotionPlanExists(f"promotion plan already exists: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.publish-{uuid.uuid4().hex}.tmp"
    )
    atomic_write_json(temporary, validated.to_dict())
    try:
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise PromotionPlanExists(
                f"promotion plan already exists: {destination}"
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    if stat_mode(destination) != 0o600:
        raise PromotionError(f"promotion plan is not owner-only: {destination}")
    return destination


def _load_json(path: Path) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PromotionError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    _require_private_regular_file(path, "promotion artifact")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot load promotion artifact {path}: {exc}") from exc


def load_promotion_plan(path: str | Path) -> PromotionPlan:
    return PromotionPlan.from_dict(_load_json(Path(path)))


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _require_private_regular_file(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PromotionPreconditionError(
            f"cannot stat {description} {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise PromotionPreconditionError(f"{description} is not a regular file: {path}")
    if stat.S_IMODE(mode) != 0o600:
        raise PromotionPreconditionError(f"{description} is not owner-only: {path}")


def _require_private_generation_tree(root: Path) -> None:
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise PromotionPreconditionError(
            f"cannot stat generation root {root}: {exc}"
        ) from exc
    if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) != 0o700:
        raise PromotionPreconditionError(
            f"generation root must be an owner-only real directory: {root}"
        )
    try:
        artifacts = tuple(root.iterdir())
    except OSError as exc:
        raise PromotionPreconditionError(
            f"cannot inspect generation root {root}: {exc}"
        ) from exc
    for artifact in artifacts:
        _require_private_regular_file(artifact, "generation artifact")


def _create_private_directories(directory: Path) -> None:
    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
        raise PromotionPreconditionError(
            f"promotion parent is not a real directory: {cursor}"
        )
    for path in reversed(missing):
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if path.is_symlink() or not path.is_dir():
            raise PromotionPreconditionError(
                f"promotion parent is not a real directory: {path}"
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PromotionState:
    schema_version: int
    promotion_id: str
    plan_sha256: str
    phase: str
    previous_collection: str
    candidate_collection: str
    candidate_restored: bool
    candidate_verified: bool
    candidate_verification_sha256: str | None
    checks_passed: bool
    forward_switched: bool
    rollback_switched: bool
    rollback_proven: bool
    final_forward_switched: bool
    updated_at: str
    events: tuple[str, ...]
    last_error: str | None

    @classmethod
    def from_dict(cls, value: object) -> PromotionState:
        if not isinstance(value, Mapping):
            raise PromotionError("promotion state must be an object")
        expected = {
            "schema_version",
            "promotion_id",
            "plan_sha256",
            "phase",
            "previous_collection",
            "candidate_collection",
            "candidate_restored",
            "candidate_verified",
            "candidate_verification_sha256",
            "checks_passed",
            "forward_switched",
            "rollback_switched",
            "rollback_proven",
            "final_forward_switched",
            "updated_at",
            "events",
            "last_error",
        }
        _exact_keys(value, expected, "promotion state")
        if value["schema_version"] != PROMOTION_SCHEMA_VERSION:
            raise PromotionError("unsupported promotion state schema_version")
        phase = _require_string(value["phase"], "phase")
        if phase not in _PHASES:
            raise PromotionError(f"invalid promotion phase: {phase}")
        events = value["events"]
        if not isinstance(events, list) or not all(
            isinstance(event, str) and event for event in events
        ):
            raise PromotionError("promotion events must be non-empty strings")
        last_error = value["last_error"]
        if last_error is not None:
            last_error = _require_string(last_error, "last_error", maximum=1000)
        promotion_id = _require_string(value["promotion_id"], "promotion_id")
        if not _ID_RE.fullmatch(promotion_id):
            raise PromotionError("promotion state promotion_id has an invalid format")
        candidate_verification_sha256 = value["candidate_verification_sha256"]
        if candidate_verification_sha256 is not None:
            candidate_verification_sha256 = _require_sha256(
                candidate_verification_sha256,
                "candidate_verification_sha256",
            )
        state = cls(
            schema_version=PROMOTION_SCHEMA_VERSION,
            promotion_id=promotion_id,
            plan_sha256=_require_sha256(value["plan_sha256"], "plan_sha256"),
            phase=phase,
            previous_collection=_require_string(
                value["previous_collection"], "previous_collection"
            ),
            candidate_collection=_require_string(
                value["candidate_collection"], "candidate_collection"
            ),
            candidate_restored=_require_bool(
                value["candidate_restored"], "candidate_restored"
            ),
            candidate_verified=_require_bool(
                value["candidate_verified"], "candidate_verified"
            ),
            candidate_verification_sha256=candidate_verification_sha256,
            checks_passed=_require_bool(value["checks_passed"], "checks_passed"),
            forward_switched=_require_bool(
                value["forward_switched"], "forward_switched"
            ),
            rollback_switched=_require_bool(
                value["rollback_switched"], "rollback_switched"
            ),
            rollback_proven=_require_bool(value["rollback_proven"], "rollback_proven"),
            final_forward_switched=_require_bool(
                value["final_forward_switched"], "final_forward_switched"
            ),
            updated_at=_require_timestamp(value["updated_at"], "updated_at"),
            events=tuple(events),
            last_error=last_error,
        )
        if state.previous_collection == state.candidate_collection:
            raise PromotionError(
                "promotion state previous and candidate collections must differ"
            )
        if state.candidate_verified != (
            state.candidate_verification_sha256 is not None
        ):
            raise PromotionError(
                "candidate verification flag and provenance digest must agree"
            )
        ordered_flags = (
            state.candidate_restored,
            state.candidate_verified,
            state.checks_passed,
            state.forward_switched,
            state.rollback_switched,
            state.rollback_proven,
            state.final_forward_switched,
        )
        if any(
            ordered_flags[index] and not ordered_flags[index - 1]
            for index in range(1, len(ordered_flags))
        ):
            raise PromotionError("promotion state contains an impossible flag sequence")
        phase_requirement = {
            "planned": not any(ordered_flags),
            "candidate_restored": state.candidate_restored,
            "candidate_verified": state.candidate_verified,
            "checks_passed": state.checks_passed,
            "forward_switched": state.forward_switched,
            "rollback_switched": state.rollback_switched,
            "rollback_proven": (
                state.rollback_proven and not state.final_forward_switched
            ),
            "completed_forward": state.final_forward_switched,
            "failed": True,
        }
        if not phase_requirement[state.phase]:
            raise PromotionError(
                f"promotion phase {state.phase!r} conflicts with persisted flags"
            )
        return state

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["events"] = list(self.events)
        return data


def load_promotion_state(path: str | Path) -> PromotionState | None:
    state_path = Path(path)
    if not state_path.exists():
        return None
    return PromotionState.from_dict(_load_json(state_path))


def _write_state(path: Path, state: PromotionState) -> None:
    atomic_write_json(path, state.to_dict())
    if stat_mode(path) != 0o600:
        raise PromotionError(f"promotion state is not owner-only: {path}")


@contextlib.contextmanager
def promotion_lock(path: str | Path) -> Iterator[None]:
    """Acquire the non-blocking owner-only local promotion lock."""
    lock_path = Path(path)
    _create_private_directories(lock_path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PromotionLockedError(f"promotion lock is held: {lock_path}") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _initial_state(
    plan: PromotionPlan, plan_hash: str, previous: str, now: datetime
) -> PromotionState:
    return PromotionState(
        schema_version=PROMOTION_SCHEMA_VERSION,
        promotion_id=plan.promotion_id,
        plan_sha256=plan_hash,
        phase="planned",
        previous_collection=previous,
        candidate_collection=plan.physical_collection,
        candidate_restored=False,
        candidate_verified=False,
        candidate_verification_sha256=None,
        checks_passed=False,
        forward_switched=False,
        rollback_switched=False,
        rollback_proven=False,
        final_forward_switched=False,
        updated_at=_format_utc(now),
        events=("plan_loaded",),
        last_error=None,
    )


def _advance(
    state: PromotionState,
    now: datetime,
    event: str,
    *,
    phase: str | None = None,
    **changes: object,
) -> PromotionState:
    return replace(
        state,
        phase=phase or state.phase,
        updated_at=_format_utc(now),
        events=(*state.events, event),
        last_error=None,
        **changes,
    )


def _candidate_mismatches(
    plan: PromotionPlan, inspection: CollectionInspection
) -> list[str]:
    expected = plan.expected_collection
    comparisons = {
        "name": (inspection.name, plan.physical_collection),
        "payload_schema_version": (
            inspection.payload_schema_version,
            expected.payload_schema_version,
        ),
        "points_count": (inspection.points_count, expected.points_count),
        "dense_name": (inspection.dense_name, expected.dense_name),
        "dense_dimension": (inspection.dense_dimension, expected.dense_dimension),
        "distance": (inspection.distance.lower(), expected.distance.lower()),
        "sparse_name": (inspection.sparse_name, expected.sparse_name),
        "generation_id": (inspection.generation_id, plan.generation_id),
        "manifest_sha256": (inspection.manifest_sha256, plan.manifest_sha256),
        "embedding_model": (inspection.embedding_model, expected.embedding_model),
        "embedding_revision": (
            inspection.embedding_revision,
            expected.embedding_revision,
        ),
        "tokenizer_model": (
            inspection.tokenizer_model,
            expected.tokenizer_model,
        ),
        "tokenizer_revision": (
            inspection.tokenizer_revision,
            expected.tokenizer_revision,
        ),
        "reranker_model": (
            inspection.reranker_model,
            expected.reranker_model,
        ),
        "reranker_revision": (
            inspection.reranker_revision,
            expected.reranker_revision,
        ),
        "vector_space_id": (inspection.vector_space_id, expected.vector_space_id),
        "chunking_fingerprint": (
            inspection.chunking_fingerprint,
            expected.chunking_fingerprint,
        ),
        "document_header": (
            inspection.document_header,
            expected.document_header,
        ),
        "retrieval_fingerprint_revision": (
            inspection.retrieval_fingerprint_revision,
            expected.retrieval_fingerprint_revision,
        ),
        "retrieval_fingerprint": (
            inspection.retrieval_fingerprint,
            expected.retrieval_fingerprint,
        ),
        "integrity_ok": (inspection.integrity_ok, True),
        "optimizer_status": (inspection.optimizer_status.lower(), "green"),
    }
    return [
        f"{field}: expected {expected_value!r}, got {actual!r}"
        for field, (actual, expected_value) in comparisons.items()
        if actual != expected_value
    ]


def _assert_alias(backend: PromotionBackend, alias: str, expected: str) -> None:
    actual = backend.alias_target(alias)
    if actual != expected:
        raise PromotionPreconditionError(
            f"alias {alias!r} expected {expected!r}, got {actual!r}"
        )


def _persist_advance(
    state_path: Path,
    state: PromotionState,
    now: Callable[[], datetime],
    event: str,
    *,
    phase: str | None = None,
    **changes: object,
) -> PromotionState:
    updated = _advance(state, now(), event, phase=phase, **changes)
    _write_state(state_path, updated)
    return updated


def _attempt_emergency_rollback(
    state: PromotionState,
    plan: PromotionPlan,
    backend: PromotionBackend,
) -> tuple[PromotionState, str | None]:
    """Best-effort containment after an uncertain alias mutation.

    A transport error can occur after Qdrant committed an atomic alias batch.  When the
    previous target was already proven ready, reconcile the remote alias and put it back
    before returning the original failure.  A completed, persisted final-forward decision
    is never implicitly reversed.
    """
    if not state.checks_passed or state.final_forward_switched:
        return state, None

    try:
        current = backend.alias_target(plan.serving_alias)
    except Exception as exc:  # noqa: BLE001 - retain the original promotion failure
        return state, f"emergency rollback alias check failed: {type(exc).__name__}"

    if current == state.previous_collection:
        if not state.forward_switched:
            return state, None
        try:
            ready = backend.readiness(state.previous_collection)
        except Exception as exc:  # noqa: BLE001 - retain original failure
            return state, f"emergency rollback readiness failed: {type(exc).__name__}"
        if ready is not True:
            return state, "emergency rollback target is not ready"
        return (
            replace(
                state,
                rollback_switched=True,
                rollback_proven=True,
                events=(*state.events, "emergency_rollback_reconciled"),
            ),
            None,
        )
    if current != plan.physical_collection:
        return state, f"emergency rollback found unexpected alias target: {current!r}"

    try:
        if backend.readiness(state.previous_collection) is not True:
            return state, "emergency rollback target is not ready"
        backend.switch_alias(plan.serving_alias, state.previous_collection)
        _assert_alias(backend, plan.serving_alias, state.previous_collection)
        switched = replace(
            state,
            forward_switched=True,
            rollback_switched=True,
            events=(*state.events, "emergency_rollback_alias_switched"),
        )
        if backend.readiness(state.previous_collection) is not True:
            return switched, "emergency rollback target failed its post-switch check"
        return replace(switched, rollback_proven=True), None
    except Exception as exc:  # noqa: BLE001 - retain the original promotion failure
        return state, f"emergency rollback failed: {type(exc).__name__}"


def execute_promotion(
    plan_path: str | Path,
    state_path: str | Path,
    backend: PromotionBackend,
    *,
    forward_after_rollback: bool = False,
    optimizer_timeout_seconds: float = 900.0,
    now: Callable[[], datetime] | None = None,
) -> PromotionState:
    """Execute or safely resume the reversible promotion protocol.

    The previous collection remains the final alias target by default. Setting
    ``forward_after_rollback`` is honored only after a persisted rollback proof.
    """
    if (
        isinstance(optimizer_timeout_seconds, bool)
        or not isinstance(optimizer_timeout_seconds, (int, float))
        or not math.isfinite(optimizer_timeout_seconds)
        or optimizer_timeout_seconds <= 0
    ):
        raise PromotionPreconditionError(
            "optimizer_timeout_seconds must be finite and positive"
        )
    clock = now or (lambda: datetime.now(UTC))
    plan_file = Path(plan_path)
    state_file = Path(state_path)
    lock_file = state_file.parent / ".promotion.lock"
    with promotion_lock(lock_file):
        plan = load_promotion_plan(plan_file)
        plan_hash = promotion_plan_sha256(plan)
        state = load_promotion_state(state_file)
        if state is None:
            previous = backend.alias_target(plan.serving_alias)
            if previous is None:
                raise PromotionPreconditionError(
                    "stable serving alias has no rollback target; maintenance migration required"
                )
            if previous == plan.physical_collection:
                raise PromotionPreconditionError(
                    "candidate already serves the alias but no rollback state exists"
                )
            state = _initial_state(plan, plan_hash, previous, clock())
            _write_state(state_file, state)
        else:
            if (
                state.promotion_id != plan.promotion_id
                or state.plan_sha256 != plan_hash
            ):
                raise PromotionPreconditionError(
                    "promotion state is bound to a different immutable plan"
                )
            if state.candidate_collection != plan.physical_collection:
                raise PromotionPreconditionError(
                    "promotion state candidate differs from the plan"
                )

        try:
            backend.restore_candidate(plan)
            state = _persist_advance(
                state_file,
                state,
                clock,
                "candidate_restored",
                phase="candidate_restored",
                candidate_restored=True,
            )
            inspection = backend.wait_for_green(
                plan.physical_collection, optimizer_timeout_seconds
            )
            mismatches = _candidate_mismatches(plan, inspection)
            if mismatches:
                raise PromotionPreconditionError(
                    "candidate compatibility failed: " + "; ".join(mismatches[:20])
                )
            verification_sha256 = inspection.verification_provenance_sha256
            if (
                not isinstance(verification_sha256, str)
                or not _SHA256_RE.fullmatch(verification_sha256)
            ):
                raise PromotionPreconditionError(
                    "candidate verification lacks a persisted provenance digest"
                )
            state = _persist_advance(
                state_file,
                state,
                clock,
                "candidate_verified",
                phase="candidate_verified",
                candidate_verified=True,
                candidate_verification_sha256=verification_sha256,
            )
            if not backend.smoke(plan.physical_collection):
                raise PromotionPreconditionError("candidate smoke checks failed")
            if not backend.readiness(plan.physical_collection):
                raise PromotionPreconditionError("candidate readiness checks failed")
            if not backend.readiness(state.previous_collection):
                raise PromotionPreconditionError(
                    "previous collection is not ready for rollback"
                )
            state = _persist_advance(
                state_file,
                state,
                clock,
                "pre_switch_checks_passed",
                phase="checks_passed",
                checks_passed=True,
            )

            current = backend.alias_target(plan.serving_alias)
            allowed = {state.previous_collection, plan.physical_collection}
            if current not in allowed:
                raise PromotionPreconditionError(
                    f"alias target changed outside this plan: {current!r}"
                )

            if state.rollback_proven:
                if state.final_forward_switched:
                    if not forward_after_rollback:
                        raise PromotionPreconditionError(
                            "promotion already finalized forward; refusing implicit rollback"
                        )
                    _assert_alias(backend, plan.serving_alias, plan.physical_collection)
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "final_forward_reconfirmed",
                        phase="completed_forward",
                    )
                    return state
                if current == plan.physical_collection:
                    if not forward_after_rollback:
                        backend.switch_alias(
                            plan.serving_alias, state.previous_collection
                        )
                        _assert_alias(
                            backend,
                            plan.serving_alias,
                            state.previous_collection,
                        )
                        if not backend.readiness(state.previous_collection):
                            raise PromotionPreconditionError(
                                "recovered rollback target is not ready"
                            )
                        state = _persist_advance(
                            state_file,
                            state,
                            clock,
                            "unconfirmed_final_forward_rolled_back",
                            phase="rollback_proven",
                            final_forward_switched=False,
                        )
                        return state
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "final_forward_reconciled",
                        phase="completed_forward",
                        final_forward_switched=True,
                    )
                    return state
                _assert_alias(backend, plan.serving_alias, state.previous_collection)
                if not forward_after_rollback:
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "rollback_proof_reconfirmed",
                        phase="rollback_proven",
                    )
            else:
                if current == state.previous_collection and not state.forward_switched:
                    backend.switch_alias(plan.serving_alias, plan.physical_collection)
                    _assert_alias(backend, plan.serving_alias, plan.physical_collection)
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "forward_alias_switched",
                        phase="forward_switched",
                        forward_switched=True,
                    )
                    current = plan.physical_collection
                elif current == plan.physical_collection and not state.forward_switched:
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "forward_alias_reconciled",
                        phase="forward_switched",
                        forward_switched=True,
                    )
                elif current == state.previous_collection and state.forward_switched:
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "rollback_alias_reconciled",
                        phase="rollback_switched",
                        rollback_switched=True,
                    )

                current = backend.alias_target(plan.serving_alias)
                if current == plan.physical_collection:
                    backend.switch_alias(plan.serving_alias, state.previous_collection)
                    _assert_alias(
                        backend, plan.serving_alias, state.previous_collection
                    )
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "rollback_alias_switched",
                        phase="rollback_switched",
                        rollback_switched=True,
                    )
                _assert_alias(backend, plan.serving_alias, state.previous_collection)
                if not state.forward_switched or not state.rollback_switched:
                    raise PromotionPreconditionError(
                        "rollback proof requires observed forward and backward switches"
                    )
                if not backend.readiness(state.previous_collection):
                    raise PromotionPreconditionError("rollback readiness proof failed")
                state = _persist_advance(
                    state_file,
                    state,
                    clock,
                    "rollback_proven",
                    phase="rollback_proven",
                    rollback_proven=True,
                )

            if forward_after_rollback:
                if not state.rollback_proven:
                    raise PromotionPreconditionError(
                        "forward completion requires persisted rollback proof"
                    )
                backend.switch_alias(plan.serving_alias, plan.physical_collection)
                _assert_alias(backend, plan.serving_alias, plan.physical_collection)
                if not backend.readiness(plan.physical_collection):
                    backend.switch_alias(plan.serving_alias, state.previous_collection)
                    _assert_alias(
                        backend, plan.serving_alias, state.previous_collection
                    )
                    state = _persist_advance(
                        state_file,
                        state,
                        clock,
                        "final_forward_readiness_rollback",
                        phase="rollback_proven",
                        final_forward_switched=False,
                    )
                    raise PromotionPreconditionError(
                        "candidate readiness failed after final forward switch"
                    )
                state = _persist_advance(
                    state_file,
                    state,
                    clock,
                    "final_forward_switched",
                    phase="completed_forward",
                    final_forward_switched=True,
                )
            return state
        except Exception as exc:
            state, rollback_error = _attempt_emergency_rollback(
                state,
                plan,
                backend,
            )
            error = str(exc)[:700] or type(exc).__name__
            if rollback_error:
                error = f"{error}; {rollback_error}"[:1000]
            failed = replace(
                state,
                phase="failed",
                updated_at=_format_utc(clock()),
                events=(*state.events, "failed"),
                last_error=error,
            )
            _write_state(state_file, failed)
            raise


__all__ = [
    "CollectionInspection",
    "ExpectedCollectionIdentity",
    "FORWARD_APPROVAL_ENV",
    "PHYSICAL_COLLECTION_PREFIX",
    "PROMOTION_APPROVAL_ENV",
    "PromotionBackend",
    "PromotionError",
    "PromotionLockedError",
    "PromotionPlan",
    "PromotionPlanExists",
    "PromotionPreconditionError",
    "PromotionState",
    "SERVING_ALIAS",
    "create_promotion_plan",
    "execute_promotion",
    "load_promotion_plan",
    "load_promotion_state",
    "physical_collection_name",
    "promotion_lock",
    "promotion_plan_sha256",
    "write_promotion_plan",
]
