"""Fail-closed embedding of a sealed corpus snapshot into one physical generation.

The production path deliberately has no implicit snapshot, collection, or resume state.
Before a model is loaded or Qdrant is contacted, callers verify the sealed snapshot and
the complete immutable generation/model identity.  A create-only binding then ties local
progress to that exact tuple; per-source/per-shard checkpoints advance only after a
``wait=True`` upsert returns successfully.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import chunk_inventory
from . import qdrant_store as store
from .chunking import Chunk, build_embed_text
from .config import RETRIEVAL_FINGERPRINT_REVISION, Config
from .court_extract import EXTRACTOR_REVISION
from .generation import CANONICAL_PAYLOAD_REVISION, GENERATION_SCHEMA_VERSION
from .pipeline import _build_doc_points, _document_state_hash, _prepare_doc_for_index
from .sources import (
    COURT_CANONICAL_FIELDS,
    NORMALIZER_REVISION,
    CanonicalDoc,
    PageBoundary,
    court_metadata_for_document,
)

logger = logging.getLogger("ingest.embed_job")

SOURCES = (
    "matsne",
    "napr",
    "ecd",
    "constcourt",
    "supremecourt",
    "tas",
    "tbappeal",
)

BINDING_SCHEMA_VERSION = 3
CHECKPOINT_SCHEMA_VERSION = 1
COORDINATOR_SCHEMA_VERSION = 1
INITIALIZATION_SCHEMA_VERSION = 2
VECTOR_CHECKSUM_SCHEMA_VERSION = 2
EXPECTED_CANDIDATE_CHUNKING = {
    "tokens": 512,
    "overlap": 80,
    "min_tokens": 64,
}
RESUME_RETRIEVE_BATCH_SIZE = 256

# This ordered KA+EN suite is release identity, not a display-only smoke sentence.  Do not
# reorder or edit it without a schema/revision change: exact dense and learned-sparse float32
# bytes from every probe are covered by the checksum artifact.
CHECKSUM_PROBE_SUITE: tuple[tuple[str, str], ...] = (
    (
        "ka_statute",
        "საქართველოს კანონი — მუხლი 1. ეს წინადადება ამოწმებს სამართლებრივ ძიებას.",
    ),
    (
        "ka_court",
        "საკონსტიტუციო სასამართლოს გადაწყვეტილება და საქმის ზუსტი ნომერი.",
    ),
    (
        "en_legal",
        "Article 1 of the Georgian law fixes this multilingual legal retrieval probe.",
    ),
)
# Kept as a compatibility alias for old callers that display one sentence.  The digest itself
# always covers the complete CHECKSUM_PROBE_SUITE and both vector families.
CHECKSUM_SENTENCE = CHECKSUM_PROBE_SUITE[0][1]
VECTOR_CHECKSUM_ENCODING = "little-endian-float32+dense;sorted-uint64-index-float32+sparse"
VECTOR_CHECKSUM_DOMAIN = b"georgian-legal-vector-probe-suite-v2\0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EmbedStateError(RuntimeError):
    """Snapshot, binding, collection, or checkpoint state is unsafe to use."""


@dataclass(frozen=True, slots=True)
class SealedSnapshot:
    root: Path
    docs: Path
    snapshot_id: str
    snapshot_sha256: str
    corpus_sha256: str
    sources: tuple[str, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EmbedBinding:
    path: Path
    value: Mapping[str, Any]

    @property
    def variant_id(self) -> str:
        return str(self.value["variant_id"])


@dataclass(frozen=True, slots=True)
class VectorChecksumReference:
    path: Path
    file_sha256: str
    probe_sha256: str
    probes: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VectorChecksumComparison:
    path: Path
    file_sha256: str
    cpu_artifact_sha256: str
    runtime_artifact_sha256: str
    dense_cosines: tuple[float, ...]
    minimum_cosine: float


@dataclass(frozen=True, slots=True)
class EmbedCoordinator:
    path: Path
    value: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EmbedInitialization:
    path: Path
    value: Mapping[str, Any]
    recovered: bool


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EmbedStateError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _strict_json(text: str, *, origin: str) -> Any:
    def reject_constant(value: str) -> None:
        raise EmbedStateError(f"{origin}: non-finite JSON number {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except EmbedStateError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise EmbedStateError(f"{origin}: invalid JSON: {exc}") from exc


def _canonical_json_bytes(value: object) -> bytes:
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
        raise EmbedStateError(f"state is not canonical JSON: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    _require_real_directory_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_components(path: Path) -> Iterator[Path]:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        yield current


def _require_real_directory_chain(path: Path, *, allow_missing: bool = False) -> None:
    """Reject every symlink/non-directory component instead of following parent links."""

    for component in _path_components(path):
        try:
            info = component.lstat()
        except FileNotFoundError as exc:
            if allow_missing:
                return
            raise EmbedStateError(f"state directory is absent: {component}") from exc
        except OSError as exc:
            raise EmbedStateError(f"cannot inspect state directory {component}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise EmbedStateError(
                f"state path contains a symlink or non-directory component: {component}"
            )


def _create_private_directories(path: Path) -> None:
    """Create a directory chain while refusing all existing symlink components."""

    for component in _path_components(path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            try:
                component.mkdir(mode=0o700)
            except FileExistsError:
                pass
            try:
                info = component.lstat()
            except OSError as exc:
                raise EmbedStateError(
                    f"cannot verify created state directory {component}: {exc}"
                ) from exc
        except OSError as exc:
            raise EmbedStateError(f"cannot inspect state directory {component}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise EmbedStateError(
                f"state path contains a symlink or non-directory component: {component}"
            )


def _create_private_json(path: Path, value: object) -> None:
    """Create one owner-only JSON file and never follow or replace an existing path."""

    _create_private_directories(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise EmbedStateError(f"immutable state already exists: {path}") from exc
    try:
        data = _canonical_json_bytes(value)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _load_private_json(path: Path, *, label: str) -> dict[str, Any]:
    _require_real_directory_chain(path.parent, allow_missing=True)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise EmbedStateError(f"{label} is absent: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EmbedStateError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = _strict_json(path.read_text(encoding="utf-8"), origin=str(path))
    except OSError as exc:
        raise EmbedStateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EmbedStateError(f"{label} must be a JSON object: {path}")
    return value


def _replace_checkpoint(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_previous: Mapping[str, Any],
) -> None:
    """Replace a validated checkpoint without ever manufacturing missing resume state."""

    actual = _load_private_json(path, label="embed checkpoint")
    if actual != dict(expected_previous):
        raise EmbedStateError(
            f"embed checkpoint changed or mismatched while running: {path}"
        )
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    _create_private_json(tmp, value)
    try:
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def verify_snapshot_docs(snapshot_docs: str | Path) -> SealedSnapshot:
    """Verify an explicit production snapshot and return its sealed identity.

    ``snapshot_docs`` must be the literal ``docs`` child of the sealed snapshot root.
    Preflight snapshots are intentionally rejected by the snapshot validator.
    """

    supplied = Path(snapshot_docs).expanduser()
    if supplied.name != "docs":
        raise EmbedStateError("--snapshot-docs must name the snapshot's exact docs directory")
    try:
        info = supplied.lstat()
    except OSError as exc:
        raise EmbedStateError(f"cannot stat snapshot docs {supplied}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EmbedStateError("--snapshot-docs must be a real, non-symlink directory")
    root = supplied.parent
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise EmbedStateError(f"cannot stat snapshot root {root}: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise EmbedStateError("snapshot root must be a real, non-symlink directory")

    # Imported only after the path shape has been checked; this is still entirely offline.
    from . import snapshot

    try:
        manifest = snapshot.verify_sealed_snapshot(
            root,
            allow_preflight=False,
            require_all_sources=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize snapshot validation failures
        raise EmbedStateError(f"sealed snapshot verification failed: {exc}") from exc

    snapshot_id = manifest.get("snapshot_id")
    snapshot_sha256 = manifest.get("snapshot_sha256")
    corpus_sha256 = manifest.get("corpus_sha256")
    build = manifest.get("build")
    raw_sources = build.get("sources") if isinstance(build, dict) else None
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise EmbedStateError("sealed snapshot manifest has invalid snapshot_id")
    if not isinstance(snapshot_sha256, str) or not _SHA256_RE.fullmatch(snapshot_sha256):
        raise EmbedStateError("sealed snapshot manifest has invalid snapshot_sha256")
    if not isinstance(corpus_sha256, str) or not _SHA256_RE.fullmatch(corpus_sha256):
        raise EmbedStateError("sealed snapshot manifest has invalid corpus_sha256")
    if (
        not isinstance(raw_sources, list)
        or len(raw_sources) != len(SOURCES)
        or set(raw_sources) != set(SOURCES)
        or any(not isinstance(item, str) for item in raw_sources)
    ):
        raise EmbedStateError("sealed snapshot must contain exactly all seven sources")
    if manifest.get("preflight") is not False:
        raise EmbedStateError("preflight snapshots cannot be embedded")
    structural_inventory = manifest.get("structural_chunk_inventory")
    if (
        not isinstance(structural_inventory, Mapping)
        or structural_inventory.get("status")
        != chunk_inventory.CHUNK_INVENTORY_STATUS_AVAILABLE
    ):
        raise EmbedStateError(
            "immutable embedding requires an available structural chunk inventory"
        )

    resolved_root = root.resolve(strict=True)
    resolved_docs = supplied.resolve(strict=True)
    if resolved_docs.parent != resolved_root:
        raise EmbedStateError("snapshot docs directory escapes its sealed snapshot root")
    return SealedSnapshot(
        root=resolved_root,
        docs=resolved_docs,
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        corpus_sha256=corpus_sha256,
        sources=tuple(raw_sources),
        manifest=manifest,
    )


def validate_snapshot_build_config(sealed: SealedSnapshot, cfg: Config) -> None:
    """Require the sealed snapshot build tuple to equal the immutable embed config.

    This check is intentionally offline and must run before a binding is created, Qdrant is
    contacted, or a model/tokenizer is loaded.  The candidate contract is specifically the
    512/80/64 chunk profile; a separately sealed snapshot/config pair with different knobs
    is not silently admitted under this release path.
    """

    build = sealed.manifest.get("build")
    if not isinstance(build, Mapping):
        raise EmbedStateError("sealed snapshot manifest.build must be an object")
    tokenizer = build.get("tokenizer")
    chunk = build.get("chunk")
    if not isinstance(tokenizer, Mapping):
        raise EmbedStateError("sealed snapshot manifest.build.tokenizer must be an object")
    if not isinstance(chunk, Mapping):
        raise EmbedStateError("sealed snapshot manifest.build.chunk must be an object")

    expected = {
        "embed_model": cfg.embed_model,
        "embedding_revision": cfg.embedding_revision,
        "tokenizer.model": cfg.tokenizer_model,
        "tokenizer.revision": cfg.tokenizer_revision,
        "chunk.tokens": cfg.chunk_tokens,
        "chunk.overlap": cfg.chunk_overlap,
        "chunk.min_tokens": cfg.chunk_min_tokens,
        "document_header": cfg.embed_header_v2,
    }
    actual = {
        "embed_model": build.get("embed_model"),
        "embedding_revision": build.get("embedding_revision"),
        "tokenizer.model": tokenizer.get("model"),
        "tokenizer.revision": tokenizer.get("revision"),
        "chunk.tokens": chunk.get("tokens"),
        "chunk.overlap": chunk.get("overlap"),
        "chunk.min_tokens": chunk.get("min_tokens"),
        "document_header": build.get("document_header"),
    }
    mismatches = [
        f"{field}: snapshot={actual[field]!r}, config={value!r}"
        for field, value in expected.items()
        if actual[field] != value
    ]
    snapshot_extractor_revision = build.get("court_extractor_revision")
    if (
        snapshot_extractor_revision is not None
        and snapshot_extractor_revision != EXTRACTOR_REVISION
    ):
        mismatches.append(
            "court_extractor_revision: "
            f"snapshot={snapshot_extractor_revision!r}, expected={EXTRACTOR_REVISION!r}"
        )
    configured_chunking = {
        "tokens": cfg.chunk_tokens,
        "overlap": cfg.chunk_overlap,
        "min_tokens": cfg.chunk_min_tokens,
    }
    if configured_chunking != EXPECTED_CANDIDATE_CHUNKING:
        mismatches.append(
            "candidate chunk contract: "
            f"config={configured_chunking!r}, expected={EXPECTED_CANDIDATE_CHUNKING!r}"
        )
    if mismatches:
        raise EmbedStateError(
            "sealed snapshot build configuration mismatch: " + "; ".join(mismatches)
        )
    try:
        expected_inventory_identity = chunk_inventory.chunk_inventory_identity(
            sources=sealed.sources,
            tokenizer_model=cfg.tokenizer_model,
            tokenizer_revision=cfg.tokenizer_revision,
            max_tokens=cfg.chunk_tokens,
            overlap_tokens=cfg.chunk_overlap,
            min_tokens=cfg.chunk_min_tokens,
            document_header=cfg.embed_header_v2,
        )
        parsed_inventory = chunk_inventory.validate_manifest_entry(
            sealed.manifest.get("structural_chunk_inventory"),
            expected_identity=expected_inventory_identity,
        )
    except chunk_inventory.ChunkInventoryError as exc:
        raise EmbedStateError(f"sealed structural chunk inventory is invalid: {exc}") from exc
    if (
        parsed_inventory["status"]
        != chunk_inventory.CHUNK_INVENTORY_STATUS_AVAILABLE
    ):
        raise EmbedStateError(
            "immutable embedding requires an available structural chunk inventory"
        )


def recompute_snapshot_chunk_inventory(
    sealed: SealedSnapshot,
    cfg: Config,
    count_tokens,
) -> chunk_inventory.StructuralChunkInventory:
    """Recompute and compare the exact sealed chunk inventory without Qdrant access."""

    validate_snapshot_build_config(sealed, cfg)
    expected = sealed.manifest["structural_chunk_inventory"]
    try:
        observed = chunk_inventory.compute_structural_chunk_inventory(
            sealed.docs,
            sources=sealed.sources,
            tokenizer_model=cfg.tokenizer_model,
            tokenizer_revision=cfg.tokenizer_revision,
            max_tokens=cfg.chunk_tokens,
            overlap_tokens=cfg.chunk_overlap,
            min_tokens=cfg.chunk_min_tokens,
            document_header=cfg.embed_header_v2,
            count_tokens=count_tokens,
        )
        chunk_inventory.compare_recomputed_inventory(expected, observed)
    except chunk_inventory.ChunkInventoryError as exc:
        raise EmbedStateError(f"structural chunk inventory verification failed: {exc}") from exc
    return observed


def _required_string(record: Mapping[str, Any], field: str) -> str:
    if field not in record:
        raise EmbedStateError(f"snapshot record is missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str) or not value:
        raise EmbedStateError(f"snapshot record field {field!r} must be a non-empty string")
    return value


def _snapshot_court_metadata(record: Mapping[str, Any]) -> dict[str, object]:
    """Accept a legacy record or validate one complete serialized court bundle."""

    present = set(COURT_CANONICAL_FIELDS).intersection(record)
    expected_fields = set(COURT_CANONICAL_FIELDS)
    if present and present != expected_fields:
        missing = sorted(expected_fields - present)
        raise EmbedStateError(
            f"snapshot court extraction bundle is partial; missing={missing}"
        )
    promoted = record.get("promoted") or {}
    if not isinstance(promoted, dict):
        raise EmbedStateError("snapshot record promoted must be an object")
    expected = court_metadata_for_document(
        str(record.get("source") or ""),
        str(record.get("body_markdown") or ""),
        promoted,
    )
    if not present:
        return expected

    for field in ("judges", "judges_raw"):
        value = record[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise EmbedStateError(f"snapshot record field {field!r} must be a string list")
    observed = {
        field: tuple(record[field]) if field in {"judges", "judges_raw"} else record[field]
        for field in COURT_CANONICAL_FIELDS
    }
    if observed != expected:
        mismatches = sorted(
            field for field in COURT_CANONICAL_FIELDS
            if observed[field] != expected[field]
        )
        raise EmbedStateError(
            "snapshot court extraction bundle does not match deterministic "
            f"recomputation; fields={mismatches}"
        )
    return expected


def _validate_strict_snapshot_record(record: Mapping[str, Any]) -> None:
    source = _required_string(record, "source")
    if source not in SOURCES:
        raise EmbedStateError(f"snapshot record has unsupported source {source!r}")
    _required_string(record, "document_id")
    body = _required_string(record, "body_markdown")
    if not body.strip():
        raise EmbedStateError("snapshot record body_markdown is empty")
    _required_string(record, "language")
    fingerprint = _required_string(record, "source_fingerprint")
    if not _SHA256_RE.fullmatch(fingerprint):
        raise EmbedStateError("snapshot record source_fingerprint must be SHA-256")
    _required_string(record, "normalizer_revision")
    _required_string(record, "version_id")
    _required_string(record, "version_id_kind")
    _required_string(record, "content_kind")
    if record.get("content_complete") is not True:
        raise EmbedStateError("snapshot record must attest content_complete=true")
    if record.get("extraction_status") != "full_text":
        raise EmbedStateError("snapshot record must attest extraction_status='full_text'")
    _required_string(record, "version_lineage_status")
    if not isinstance(record.get("version_lineage_complete"), bool):
        raise EmbedStateError("snapshot record must explicitly set version_lineage_complete")
    authority = _required_string(record, "source_authority")
    if authority not in {"official", "primary_official"}:
        raise EmbedStateError("snapshot record source_authority is not official")
    _required_string(record, "official_url")
    if "freshness_sla_met" not in record or not (
        record["freshness_sla_met"] is None
        or isinstance(record["freshness_sla_met"], bool)
    ):
        raise EmbedStateError("snapshot record must explicitly set freshness_sla_met")
    for field in ("supersedes", "consolidated_dates"):
        value = record.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise EmbedStateError(f"snapshot record field {field!r} must be a string list")
    if record.get("admissible") is not True:
        raise EmbedStateError("snapshot record must attest admissible=true")
    _snapshot_page_boundaries(record)
    _snapshot_court_metadata(record)


def _snapshot_page_boundaries(record: Mapping[str, Any]) -> tuple[PageBoundary, ...]:
    raw = record.get("page_boundaries")
    reason = record.get("page_coordinate_reason")
    body = record.get("body_markdown")
    if not isinstance(raw, list):
        raise EmbedStateError("snapshot record page_boundaries must be an array")
    if not isinstance(reason, str) or not reason:
        raise EmbedStateError("snapshot record page_coordinate_reason is required")
    if not isinstance(body, str):
        raise EmbedStateError("snapshot record body_markdown is invalid")
    boundaries: list[PageBoundary] = []
    previous_end = 0
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) not in (
            {"page", "char_start", "char_end"},
            {"page_number", "char_start", "char_end"},
        ):
            raise EmbedStateError(
                f"snapshot record page_boundaries[{index}] has invalid keys"
            )
        page = value.get("page", value.get("page_number"))
        start = value["char_start"]
        end = value["char_end"]
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (page, start, end)
        ):
            raise EmbedStateError(
                f"snapshot record page_boundaries[{index}] values are invalid"
            )
        if page != index + 1 or start < previous_end or end < start or end > len(body):
            raise EmbedStateError(
                f"snapshot record page_boundaries[{index}] range is invalid"
            )
        boundaries.append(PageBoundary(page=page, char_start=start, char_end=end))
        previous_end = end
    if boundaries and reason != "exact_pdf_text":
        raise EmbedStateError(
            "snapshot records with page boundaries require exact_pdf_text coordinates"
        )
    if not boundaries and reason != "source_not_paginated":
        raise EmbedStateError(
            "admissible snapshot records without page boundaries must be source_not_paginated"
        )
    return tuple(boundaries)


def snapshot_doc_to_canonical(
    d: Mapping[str, Any], *, strict: bool = False
) -> CanonicalDoc:
    """Rebuild a canonical document; production snapshot iteration is always strict."""

    if strict:
        _validate_strict_snapshot_record(d)
    court_metadata = _snapshot_court_metadata(d)
    return CanonicalDoc(
        source=d["source"],
        document_id=d["document_id"],
        title=d.get("title"),
        date=d.get("date"),
        date_raw=d.get("date_raw"),
        language=d.get("language") or "ka",
        document_type=d.get("document_type"),
        court=d.get("court"),
        source_url=d.get("source_url"),
        document_number=d.get("document_number"),
        registration_code=d.get("registration_code"),
        parties=d.get("parties"),
        status=d.get("status"),
        status_raw=d.get("status_raw"),
        in_force_date=d.get("in_force_date"),
        expiry_date=d.get("expiry_date"),
        body_markdown=d["body_markdown"],
        extra={},
        promoted=d.get("promoted") or {},
        is_consolidated=d.get("is_consolidated"),
        consolidated_count=d.get("consolidated_count"),
        content_kind=(d["content_kind"] if strict else d.get("content_kind") or "full_text"),
        content_complete=(d["content_complete"] if strict else bool(d.get("content_complete", True))),
        extraction_status=(
            d["extraction_status"] if strict else d.get("extraction_status") or "full_text"
        ),
        source_binary_url=d.get("source_binary_url"),
        article_summary=d.get("article_summary"),
        source_fingerprint=(d["source_fingerprint"] if strict else d.get("source_fingerprint")),
        normalizer_revision=(
            d["normalizer_revision"]
            if strict
            else d.get("normalizer_revision") or NORMALIZER_REVISION
        ),
        version_id=(d["version_id"] if strict else d.get("version_id")),
        version_id_kind=(
            d["version_id_kind"] if strict else d.get("version_id_kind") or "derived"
        ),
        supersedes=tuple(d.get("supersedes") or ()),
        effective_from=d.get("effective_from"),
        effective_to=d.get("effective_to"),
        repeal_date=d.get("repeal_date"),
        consolidation_status=d.get("consolidation_status"),
        version_lineage_status=(
            d["version_lineage_status"]
            if strict
            else d.get("version_lineage_status") or "unknown"
        ),
        version_lineage_complete=(
            d["version_lineage_complete"]
            if strict
            else bool(d.get("version_lineage_complete", False))
        ),
        consolidated_dates=tuple(d.get("consolidated_dates") or ()),
        official_url=(d["official_url"] if strict else d.get("official_url") or d.get("source_url")),
        official_binary_url=d.get("official_binary_url") or d.get("source_binary_url"),
        official_html_url=d.get("official_html_url") or d.get("official_url") or d.get("source_url"),
        official_pdf_url=d.get("official_pdf_url"),
        source_authority=(
            d["source_authority"] if strict else d.get("source_authority") or "primary_official"
        ),
        freshness_sla_met=d.get("freshness_sla_met"),
        admissible=(d["admissible"] if strict else bool(d.get("admissible", True))),
        page_boundaries=(
            _snapshot_page_boundaries(d)
            if strict
            else tuple(
                PageBoundary(
                    page=int(value.get("page", value.get("page_number"))),
                    char_start=int(value["char_start"]),
                    char_end=int(value["char_end"]),
                )
                for value in (d.get("page_boundaries") or [])
            )
        ),
        page_coordinate_reason=(
            d["page_coordinate_reason"]
            if strict
            else d.get("page_coordinate_reason") or "source_not_paginated"
        ),
        **court_metadata,
    )


def iter_snapshot_docs(
    source: str,
    *,
    root: Path,
    limit: int | None = None,
    strict: bool = True,
) -> Iterator[CanonicalDoc]:
    if source not in SOURCES:
        raise EmbedStateError(f"unsupported snapshot source {source!r}")
    path = root / f"{source}.jsonl"
    try:
        info = path.lstat()
    except OSError as exc:
        raise EmbedStateError(f"cannot stat snapshot source file {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EmbedStateError(f"snapshot source must be a regular non-symlink file: {path}")
    emitted = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise EmbedStateError(f"cannot open snapshot source {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _strict_json(line, origin=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise EmbedStateError(f"{path}:{line_number}: record must be a JSON object")
            try:
                doc = snapshot_doc_to_canonical(value, strict=strict)
            except Exception as exc:  # noqa: BLE001 - add immutable source cursor context
                raise EmbedStateError(f"{path}:{line_number}: invalid snapshot record: {exc}") from exc
            if doc.source != source:
                raise EmbedStateError(
                    f"{path}:{line_number}: record source {doc.source!r} does not match {source!r}"
                )
            yield doc
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def load_snapshot_docs(
    source: str, ids: set[str], *, root: Path, strict: bool = True
) -> list[CanonicalDoc]:
    """Load a specific set of document IDs from one explicit snapshot directory."""

    out: list[CanonicalDoc] = []
    remaining = set(ids)
    for doc in iter_snapshot_docs(source, root=root, strict=strict):
        if doc.document_id in remaining:
            out.append(doc)
            remaining.discard(doc.document_id)
            if not remaining:
                break
    return out


def _length_prefixed(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _float32_bytes(values: Sequence[Any], *, label: str) -> bytes:
    encoded = bytearray()
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbedStateError(f"{label}[{index}] is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise EmbedStateError(f"{label}[{index}] is not finite")
        try:
            encoded.extend(struct.pack("<f", number))
        except (OverflowError, struct.error) as exc:
            raise EmbedStateError(f"{label}[{index}] is not float32") from exc
    return bytes(encoded)


def _normalise_sparse(value: Any, *, label: str) -> tuple[list[int], list[float]]:
    indices = getattr(value, "indices", None)
    weights = getattr(value, "values", None)
    if isinstance(value, Mapping):
        indices = value.get("indices")
        weights = value.get("values")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes, bytearray))
        or not isinstance(weights, Sequence)
        or isinstance(weights, (str, bytes, bytearray))
        or len(indices) != len(weights)
    ):
        raise EmbedStateError(f"{label} is not an index/value sparse vector")
    pairs: list[tuple[int, float]] = []
    for position, (raw_index, raw_weight) in enumerate(
        zip(indices, weights, strict=True)
    ):
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or raw_index > (2**64 - 1)
        ):
            raise EmbedStateError(f"{label}.indices[{position}] is invalid")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise EmbedStateError(f"{label}.values[{position}] is not numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise EmbedStateError(f"{label}.values[{position}] is not finite")
        pairs.append((raw_index, weight))
    pairs.sort(key=lambda item: item[0])
    if any(left[0] == right[0] for left, right in zip(pairs, pairs[1:])):
        raise EmbedStateError(f"{label} contains duplicate indices")
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _probe_digest(probes: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(VECTOR_CHECKSUM_DOMAIN)
    digest.update(struct.pack("<Q", len(probes)))
    for expected, probe in zip(CHECKSUM_PROBE_SUITE, probes, strict=True):
        expected_id, expected_text = expected
        if probe.get("id") != expected_id or probe.get("text") != expected_text:
            raise EmbedStateError("vector checksum probe suite identity mismatch")
        dense = probe.get("dense")
        if not isinstance(dense, Sequence) or isinstance(
            dense, (str, bytes, bytearray)
        ):
            raise EmbedStateError(f"vector checksum probe {expected_id!r} lacks dense values")
        sparse_indices, sparse_values = _normalise_sparse(
            probe.get("sparse"), label=f"probe[{expected_id}].sparse"
        )
        dense_bytes = _float32_bytes(dense, label=f"probe[{expected_id}].dense")
        digest.update(_length_prefixed(expected_id.encode("utf-8")))
        digest.update(_length_prefixed(expected_text.encode("utf-8")))
        digest.update(struct.pack("<Q", len(dense)))
        digest.update(dense_bytes)
        digest.update(struct.pack("<Q", len(sparse_indices)))
        for sparse_index, sparse_weight in zip(
            sparse_indices, sparse_values, strict=True
        ):
            digest.update(struct.pack("<Q", sparse_index))
            digest.update(_float32_bytes([sparse_weight], label="sparse weight"))
    return digest.hexdigest()


def vector_checksum_value(embedder) -> dict[str, Any]:
    """Encode the fixed suite and hash exact dense+sparse little-endian float32 bytes."""

    probes: list[dict[str, Any]] = []
    for probe_id, text in CHECKSUM_PROBE_SUITE:
        embedded = embedder.encode_query(text)
        dense = [float(value) for value in embedded.dense]
        sparse_indices, sparse_values = _normalise_sparse(
            embedded.sparse, label=f"probe[{probe_id}].sparse"
        )
        probes.append(
            {
                "id": probe_id,
                "text": text,
                "dense": dense,
                "sparse": {
                    "indices": sparse_indices,
                    "values": sparse_values,
                },
            }
        )
    return {
        "schema_version": VECTOR_CHECKSUM_SCHEMA_VERSION,
        "algorithm": "sha256",
        "encoding": VECTOR_CHECKSUM_ENCODING,
        "probe_sha256": _probe_digest(probes),
        "probes": probes,
    }


def dense_checksum(embedder) -> tuple[str, list[float]]:
    """Compatibility wrapper returning the full suite digest and first dense vector."""

    value = vector_checksum_value(embedder)
    return str(value["probe_sha256"]), list(value["probes"][0]["dense"])


def checksum_cosine(cpu_vec: list[float], gpu_vec: list[float]) -> float:
    """Cosine similarity between two equal-length checksum vectors."""

    if not cpu_vec or len(cpu_vec) != len(gpu_vec):
        raise ValueError("checksum vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(cpu_vec, gpu_vec, strict=True))
    na = math.sqrt(sum(a * a for a in cpu_vec))
    nb = math.sqrt(sum(b * b for b in gpu_vec))
    if not na or not nb:
        raise ValueError("checksum vectors must have non-zero norm")
    return dot / (na * nb)


def checksum_suite_cosines(
    left: VectorChecksumReference | Mapping[str, Any],
    right: VectorChecksumReference | Mapping[str, Any],
) -> tuple[float, ...]:
    """Return dense CPU/GPU cosine values for the complete ordered probe suite."""

    left_probes = left.probes if isinstance(left, VectorChecksumReference) else left["probes"]
    right_probes = (
        right.probes if isinstance(right, VectorChecksumReference) else right["probes"]
    )
    if len(left_probes) != len(CHECKSUM_PROBE_SUITE) or len(right_probes) != len(
        CHECKSUM_PROBE_SUITE
    ):
        raise EmbedStateError("vector checksum probe suite length mismatch")
    return tuple(
        checksum_cosine(
            [float(value) for value in left_probe["dense"]],
            [float(value) for value in right_probe["dense"]],
        )
        for left_probe, right_probe in zip(left_probes, right_probes, strict=True)
    )


def save_checksum_comparison(
    cpu: VectorChecksumReference,
    runtime: VectorChecksumReference,
    path: str | Path,
    *,
    minimum_cosine: float = 0.999,
) -> VectorChecksumComparison:
    """Create the immutable CPU/runtime dense-cosine gate for the exact probe suite."""

    if (
        isinstance(minimum_cosine, bool)
        or not isinstance(minimum_cosine, (int, float))
        or not math.isfinite(float(minimum_cosine))
        or not 0.0 < float(minimum_cosine) <= 1.0
    ):
        raise EmbedStateError("minimum checksum cosine must be in (0, 1]")
    cosines = checksum_suite_cosines(cpu, runtime)
    minimum = float(minimum_cosine)
    if any(not math.isfinite(value) or value < minimum for value in cosines):
        raise EmbedStateError(
            "CPU/runtime vector checksum cosine gate failed: "
            f"minimum={minimum}, observed={list(cosines)}"
        )
    value = {
        "schema_version": VECTOR_CHECKSUM_SCHEMA_VERSION,
        "kind": "cpu-runtime-vector-checksum-comparison",
        "algorithm": "cosine",
        "probe_suite_sha256": hashlib.sha256(
            _canonical_json_bytes(list(CHECKSUM_PROBE_SUITE))
        ).hexdigest(),
        "cpu": {
            "artifact_sha256": cpu.file_sha256,
            "probe_sha256": cpu.probe_sha256,
        },
        "runtime": {
            "artifact_sha256": runtime.file_sha256,
            "probe_sha256": runtime.probe_sha256,
        },
        "minimum_cosine": minimum,
        "dense_cosines": list(cosines),
        "passed": True,
    }
    output = Path(path).expanduser().resolve(strict=False)
    _create_private_json(output, value)
    return load_checksum_comparison(output, cpu=cpu, runtime=runtime)


def load_checksum_comparison(
    path: str | Path,
    *,
    cpu: VectorChecksumReference,
    runtime: VectorChecksumReference,
) -> VectorChecksumComparison:
    """Reload and independently reproduce a CPU/runtime checksum gate artifact."""

    artifact = Path(path).expanduser().resolve(strict=True)
    value = _load_private_json(artifact, label="vector checksum comparison")
    expected_keys = {
        "schema_version",
        "kind",
        "algorithm",
        "probe_suite_sha256",
        "cpu",
        "runtime",
        "minimum_cosine",
        "dense_cosines",
        "passed",
    }
    if set(value) != expected_keys:
        raise EmbedStateError("vector checksum comparison has invalid keys")
    expected_suite_sha = hashlib.sha256(
        _canonical_json_bytes(list(CHECKSUM_PROBE_SUITE))
    ).hexdigest()
    cpu_identity = value.get("cpu")
    runtime_identity = value.get("runtime")
    if (
        value["schema_version"] != VECTOR_CHECKSUM_SCHEMA_VERSION
        or value["kind"] != "cpu-runtime-vector-checksum-comparison"
        or value["algorithm"] != "cosine"
        or value["probe_suite_sha256"] != expected_suite_sha
        or value["passed"] is not True
        or not isinstance(cpu_identity, dict)
        or set(cpu_identity) != {"artifact_sha256", "probe_sha256"}
        or not isinstance(runtime_identity, dict)
        or set(runtime_identity) != {"artifact_sha256", "probe_sha256"}
        or cpu_identity
        != {
            "artifact_sha256": cpu.file_sha256,
            "probe_sha256": cpu.probe_sha256,
        }
        or runtime_identity
        != {
            "artifact_sha256": runtime.file_sha256,
            "probe_sha256": runtime.probe_sha256,
        }
    ):
        raise EmbedStateError("vector checksum comparison identity mismatch")
    minimum = value.get("minimum_cosine")
    stored_cosines = value.get("dense_cosines")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or not 0.0 < float(minimum) <= 1.0
        or not isinstance(stored_cosines, list)
        or len(stored_cosines) != len(CHECKSUM_PROBE_SUITE)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in stored_cosines
        )
    ):
        raise EmbedStateError("vector checksum comparison values are invalid")
    observed = checksum_suite_cosines(cpu, runtime)
    if list(observed) != stored_cosines or any(
        value < float(minimum) for value in observed
    ):
        raise EmbedStateError("vector checksum comparison no longer reproduces")
    return VectorChecksumComparison(
        path=artifact,
        file_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        cpu_artifact_sha256=cpu.file_sha256,
        runtime_artifact_sha256=runtime.file_sha256,
        dense_cosines=observed,
        minimum_cosine=float(minimum),
    )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def save_checksum_reference(
    embedder,
    path: Path,
    *,
    snapshot_root: Path | None = None,
) -> str:
    """Create one explicit checksum artifact; never overwrite or write into a snapshot."""

    output = path.expanduser().resolve(strict=False)
    if snapshot_root is not None and _is_within(output, snapshot_root.resolve(strict=True)):
        raise EmbedStateError("--checksum-output must be outside the immutable snapshot")
    value = vector_checksum_value(embedder)
    _create_private_json(output, value)
    return str(value["probe_sha256"])


def load_checksum_reference(path: str | Path) -> VectorChecksumReference:
    """Validate and hash one create-only vector checksum artifact."""

    artifact = Path(path).expanduser().resolve(strict=True)
    value = _load_private_json(artifact, label="vector checksum artifact")
    if set(value) != {
        "schema_version",
        "algorithm",
        "encoding",
        "probe_sha256",
        "probes",
    }:
        raise EmbedStateError("vector checksum artifact has invalid keys")
    if (
        value["schema_version"] != VECTOR_CHECKSUM_SCHEMA_VERSION
        or value["algorithm"] != "sha256"
        or value["encoding"] != VECTOR_CHECKSUM_ENCODING
        or not isinstance(value["probe_sha256"], str)
        or not _SHA256_RE.fullmatch(value["probe_sha256"])
        or not isinstance(value["probes"], list)
        or len(value["probes"]) != len(CHECKSUM_PROBE_SUITE)
        or any(not isinstance(probe, dict) for probe in value["probes"])
    ):
        raise EmbedStateError("vector checksum artifact identity is invalid")
    observed_probe_sha = _probe_digest(value["probes"])
    if observed_probe_sha != value["probe_sha256"]:
        raise EmbedStateError("vector checksum artifact digest mismatch")
    file_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return VectorChecksumReference(
        path=artifact,
        file_sha256=file_digest,
        probe_sha256=observed_probe_sha,
        probes=tuple(value["probes"]),
    )


def _filesystem_id(value: os.statvfs_result) -> int | None:
    raw = getattr(value, "f_fsid", None)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _directory_volume_evidence(
    path: str | Path,
    *,
    label: str,
    allow_missing_leaf: bool,
) -> dict[str, Any]:
    requested = Path(path).expanduser().absolute()
    _require_real_directory_chain(requested, allow_missing=allow_missing_leaf)
    probe = requested
    while not probe.exists():
        if not allow_missing_leaf or probe == probe.parent:
            raise EmbedStateError(f"{label} is absent: {requested}")
        probe = probe.parent
    try:
        info = probe.lstat()
        volume = os.statvfs(probe)
    except OSError as exc:
        raise EmbedStateError(f"cannot inspect {label} volume: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EmbedStateError(f"{label} must resolve through real directories")
    return {
        "requested_path": str(requested),
        "st_dev": int(info.st_dev),
        "f_fsid": _filesystem_id(volume),
    }


def storage_identity_descriptor(
    path: str | Path | None,
    *,
    checkpoint_root: str | Path | None = None,
    qdrant_storage_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Describe one mounted persistent volume using non-copyable local evidence.

    The identity file's exact content, inode and device are bound together with the
    filesystem ID and the two persistent roots.  Copying the same bytes to another
    inode/filesystem therefore cannot satisfy resume.  Device/fsid/inode changes after
    a legitimate remount conservatively fail and require a new reviewed workflow.
    """

    if path is None:
        return None
    identity_path = Path(path).expanduser().absolute()
    _require_real_directory_chain(identity_path.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(identity_path, flags)
        info = os.fstat(descriptor)
        volume = os.fstatvfs(descriptor)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    except OSError as exc:
        raise EmbedStateError(f"cannot inspect storage identity {identity_path}: {exc}") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise EmbedStateError("storage identity must be a regular non-symlink file")
    identity_volume = {
        "st_dev": int(info.st_dev),
        "f_fsid": _filesystem_id(volume),
    }
    value: dict[str, Any] = {
        "schema_version": 2,
        "content_sha256": digest.hexdigest(),
        "identity_file": {
            "path": str(identity_path),
            "st_dev": int(info.st_dev),
            "st_ino": int(info.st_ino),
            "f_fsid": _filesystem_id(volume),
        },
        "checkpoint_root": None,
        "qdrant_storage_root": None,
    }
    for field, supplied, allow_missing in (
        ("checkpoint_root", checkpoint_root, True),
        ("qdrant_storage_root", qdrant_storage_root, False),
    ):
        if supplied is None:
            continue
        evidence = _directory_volume_evidence(
            supplied,
            label=field.replace("_", " "),
            allow_missing_leaf=allow_missing,
        )
        if {
            "st_dev": evidence["st_dev"],
            "f_fsid": evidence["f_fsid"],
        } != identity_volume:
            raise EmbedStateError(
                "storage identity, checkpoint binding root, and Qdrant storage root "
                "must share the same device/filesystem ID"
            )
        value[field] = evidence
    if (checkpoint_root is None) != (qdrant_storage_root is None):
        raise EmbedStateError(
            "checkpoint_root and qdrant_storage_root must be supplied together"
        )
    return value


def storage_identity_sha256(
    path: str | Path | None,
    *,
    checkpoint_root: str | Path | None = None,
    qdrant_storage_root: str | Path | None = None,
) -> str | None:
    """Hash the canonical mounted-volume descriptor used by binding/resume."""

    descriptor = storage_identity_descriptor(
        path,
        checkpoint_root=checkpoint_root,
        qdrant_storage_root=qdrant_storage_root,
    )
    return (
        hashlib.sha256(_canonical_json_bytes(descriptor)).hexdigest()
        if descriptor is not None
        else None
    )


def _embedding_runtime_value(cfg: Config) -> dict[str, object]:
    if cfg.embed_device is not None and (
        not isinstance(cfg.embed_device, str) or not cfg.embed_device.strip()
    ):
        raise EmbedStateError("embedding device must be null or a non-empty string")
    if not isinstance(cfg.embed_use_fp16, bool):
        raise EmbedStateError("embedding fp16 mode must be boolean")
    if (
        isinstance(cfg.embed_batch_size, bool)
        or not isinstance(cfg.embed_batch_size, int)
        or cfg.embed_batch_size < 1
    ):
        raise EmbedStateError("embedding batch size must be a positive integer")
    return {
        "device": cfg.embed_device,
        "use_fp16": cfg.embed_use_fp16,
        "batch_size": cfg.embed_batch_size,
    }


def _binding_value(
    cfg: Config,
    sealed: SealedSnapshot,
    *,
    checksum: VectorChecksumReference,
    worker_count: int,
    collection_configuration: Mapping[str, Any],
    collection_configuration_sha256: str,
    storage_identity_sha256: str | None = None,
    reviewed_plan_sha256: str | None = None,
) -> dict[str, Any]:
    validate_snapshot_build_config(sealed, cfg)
    identity = store.validate_generation_identity(cfg)
    assert cfg.generation_id is not None
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise EmbedStateError("worker_count must be an integer >= 1")
    if not _SHA256_RE.fullmatch(collection_configuration_sha256):
        raise EmbedStateError("collection configuration SHA-256 is invalid")
    if (
        store.collection_configuration_sha256(collection_configuration)
        != collection_configuration_sha256
    ):
        raise EmbedStateError(
            "collection configuration SHA-256 does not match its exact value"
        )
    if storage_identity_sha256 is not None and not _SHA256_RE.fullmatch(
        storage_identity_sha256
    ):
        raise EmbedStateError("storage identity SHA-256 is invalid")
    if reviewed_plan_sha256 is not None and not _SHA256_RE.fullmatch(
        reviewed_plan_sha256
    ):
        raise EmbedStateError("reviewed plan SHA-256 is invalid")
    material: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "generation_id": cfg.generation_id,
        "physical_collection": cfg.collection_name,
        "snapshot": {
            "snapshot_id": sealed.snapshot_id,
            "snapshot_sha256": sealed.snapshot_sha256,
            "corpus_sha256": sealed.corpus_sha256,
            "root": str(sealed.root),
            "docs": str(sealed.docs),
            "structural_chunk_inventory": {
                field: sealed.manifest["structural_chunk_inventory"][field]
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
        "models": {
            "embedding": {"name": identity.embedding_model, "revision": identity.embedding_revision},
            "tokenizer": {"name": identity.tokenizer_model, "revision": identity.tokenizer_revision},
            "reranker": {"name": identity.reranker_model, "revision": identity.reranker_revision},
        },
        "embedding_runtime": _embedding_runtime_value(cfg),
        "vector_space_id": identity.vector_space_id,
        "vector_checksum": {
            "schema_version": VECTOR_CHECKSUM_SCHEMA_VERSION,
            "artifact_sha256": checksum.file_sha256,
            "probe_sha256": checksum.probe_sha256,
            "encoding": VECTOR_CHECKSUM_ENCODING,
        },
        "collection_configuration": {
            "sha256": collection_configuration_sha256,
            "value": dict(collection_configuration),
        },
        "storage_identity_sha256": storage_identity_sha256,
        "reviewed_plan_sha256": reviewed_plan_sha256,
        "worker_count": worker_count,
        "chunking": {
            "fingerprint": identity.chunking_fingerprint,
            "max_tokens": cfg.chunk_tokens,
            "overlap_tokens": cfg.chunk_overlap,
            "min_tokens": cfg.chunk_min_tokens,
            "document_header": cfg.embed_header_v2,
        },
        "retrieval": {
            "fingerprint_revision": RETRIEVAL_FINGERPRINT_REVISION,
            "fingerprint_sha256": identity.retrieval_fingerprint,
        },
        "payload_schema": {
            "generation_schema_version": GENERATION_SCHEMA_VERSION,
            "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        },
    }
    variant_id = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
    material["variant_id"] = variant_id
    return material


def binding_path(cfg: Config) -> Path:
    if cfg.generation_id is None:
        raise EmbedStateError("GENERATION_ID is required for an embed binding")
    return cfg.state_dir / "embed" / cfg.generation_id / "binding.json"


def initialization_path(cfg: Config) -> Path:
    if cfg.generation_id is None:
        raise EmbedStateError("GENERATION_ID is required for coordinator initialization")
    return cfg.state_dir / "embed" / cfg.generation_id / "initialization.json"


def _initialization_value(
    cfg: Config,
    sealed: SealedSnapshot,
    *,
    checksum: VectorChecksumReference,
    worker_count: int,
    collection_configuration: Mapping[str, Any],
    collection_configuration_sha256: str,
    storage_identity_sha256: str | None,
    reviewed_plan_sha256: str | None,
) -> dict[str, Any]:
    validate_snapshot_build_config(sealed, cfg)
    identity = store.validate_generation_identity(cfg)
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise EmbedStateError("worker_count must be an integer >= 1")
    if not _SHA256_RE.fullmatch(collection_configuration_sha256):
        raise EmbedStateError("collection configuration SHA-256 is invalid")
    if (
        store.collection_configuration_sha256(collection_configuration)
        != collection_configuration_sha256
    ):
        raise EmbedStateError(
            "collection configuration SHA-256 does not match its exact value"
        )
    reviewed_configuration = store.expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )
    if dict(collection_configuration) != reviewed_configuration:
        raise EmbedStateError(
            "initialization collection configuration differs from the reviewed "
            "complete profile"
        )
    for label, digest in (
        ("storage identity", storage_identity_sha256),
        ("reviewed plan", reviewed_plan_sha256),
    ):
        if digest is not None and not _SHA256_RE.fullmatch(digest):
            raise EmbedStateError(f"{label} SHA-256 is invalid")
    return {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "kind": "immutable-embed-coordinator-initialization",
        "generation_id": identity.generation_id,
        "physical_collection": cfg.collection_name,
        "snapshot": {
            "snapshot_id": sealed.snapshot_id,
            "snapshot_sha256": sealed.snapshot_sha256,
            "corpus_sha256": sealed.corpus_sha256,
            "structural_chunk_inventory": {
                field: sealed.manifest["structural_chunk_inventory"][field]
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
        "point_identity": identity.as_payload(),
        "embedding_runtime": _embedding_runtime_value(cfg),
        "vector_checksum": {
            "artifact_sha256": checksum.file_sha256,
            "probe_sha256": checksum.probe_sha256,
        },
        "collection_configuration": {
            "sha256": collection_configuration_sha256,
            "value": dict(collection_configuration),
        },
        "storage_identity_sha256": storage_identity_sha256,
        "reviewed_plan_sha256": reviewed_plan_sha256,
        "worker_count": worker_count,
    }


def prepare_initialization(
    cfg: Config,
    sealed: SealedSnapshot,
    *,
    checksum: VectorChecksumReference,
    worker_count: int,
    collection_configuration: Mapping[str, Any],
    collection_configuration_sha256: str,
    storage_identity_sha256: str | None,
    reviewed_plan_sha256: str | None = None,
) -> EmbedInitialization:
    """Create or validate the durable pre-collection initialization intent.

    The caller must refuse an already-existing collection before creating a new intent.
    Once present, this exact marker authorizes recovery of only the same still-empty
    create-only target after a crash.
    """

    expected = _initialization_value(
        cfg,
        sealed,
        checksum=checksum,
        worker_count=worker_count,
        collection_configuration=collection_configuration,
        collection_configuration_sha256=collection_configuration_sha256,
        storage_identity_sha256=storage_identity_sha256,
        reviewed_plan_sha256=reviewed_plan_sha256,
    )
    path = initialization_path(cfg)
    if os.path.lexists(path):
        actual = _load_private_json(path, label="embed initialization")
        if actual != expected:
            raise EmbedStateError(
                "embed initialization intent does not match this exact launch"
            )
        return EmbedInitialization(path=path, value=actual, recovered=True)
    _create_private_json(path, expected)
    return EmbedInitialization(path=path, value=expected, recovered=False)


def prepare_binding(
    cfg: Config,
    sealed: SealedSnapshot,
    *,
    checksum: VectorChecksumReference,
    worker_count: int,
    collection_configuration: Mapping[str, Any],
    collection_configuration_sha256: str,
    storage_identity_sha256: str | None = None,
    reviewed_plan_sha256: str | None = None,
    resume: bool,
    recover_initialization: bool = False,
) -> EmbedBinding:
    """Create a fresh binding or require an exact existing binding for resume."""

    expected = _binding_value(
        cfg,
        sealed,
        checksum=checksum,
        worker_count=worker_count,
        collection_configuration=collection_configuration,
        collection_configuration_sha256=collection_configuration_sha256,
        storage_identity_sha256=storage_identity_sha256,
        reviewed_plan_sha256=reviewed_plan_sha256,
    )
    path = binding_path(cfg)
    if resume:
        actual = _load_private_json(path, label="embed binding")
        if actual != expected:
            raise EmbedStateError(
                "embed binding does not match the selected snapshot, physical collection, "
                "models, chunking, retrieval identity, or payload schema"
            )
    else:
        if recover_initialization and os.path.lexists(path):
            actual = _load_private_json(path, label="embed binding")
            if actual != expected:
                raise EmbedStateError(
                    "stranded embed binding does not match recoverable initialization"
                )
        else:
            _create_private_json(path, expected)
            actual = expected
    return EmbedBinding(path=path, value=actual)


def binding_sha256(binding: EmbedBinding) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(binding.value))).hexdigest()


def coordinator_path(binding: EmbedBinding) -> Path:
    return binding.path.parent / "coordinator.json"


def checkpoint_path(
    binding: EmbedBinding,
    source: str,
    shard: tuple[int, int] | None = None,
) -> Path:
    if source not in SOURCES:
        raise EmbedStateError(f"unsupported checkpoint source {source!r}")
    shard_i, shard_n = shard if shard is not None else (0, 1)
    if not (isinstance(shard_i, int) and isinstance(shard_n, int) and 0 <= shard_i < shard_n):
        raise EmbedStateError(f"invalid shard ({shard_i!r}, {shard_n!r})")
    filename = f"{source}.shard{shard_i}of{shard_n}.json"
    return (
        binding.path.parent
        / "variants"
        / binding.variant_id
        / "checkpoints"
        / filename
    )


def _initial_checkpoint(
    binding: EmbedBinding,
    source: str,
    shard: tuple[int, int] | None,
) -> dict[str, Any]:
    shard_i, shard_n = shard if shard is not None else (0, 1)
    snapshot = binding.value["snapshot"]
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generation_id": binding.value["generation_id"],
        "variant_id": binding.variant_id,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "physical_collection": binding.value["physical_collection"],
        "source": source,
        "shard": {"index": shard_i, "count": shard_n},
        "cursor": None,
        "documents_completed": 0,
        "chunks_completed": 0,
        "complete": False,
    }


def _validate_checkpoint(
    value: Mapping[str, Any],
    expected_initial: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    expected_keys = set(expected_initial)
    if set(value) != expected_keys:
        raise EmbedStateError(
            f"embed checkpoint keys mismatch at {path}: "
            f"missing={sorted(expected_keys - set(value))}, "
            f"unknown={sorted(set(value) - expected_keys)}"
        )
    for field in (
        "schema_version",
        "generation_id",
        "variant_id",
        "snapshot_sha256",
        "physical_collection",
        "source",
        "shard",
    ):
        if value[field] != expected_initial[field]:
            raise EmbedStateError(f"embed checkpoint {field} mismatch at {path}")
    docs = value["documents_completed"]
    chunks = value["chunks_completed"]
    if (
        not isinstance(docs, int)
        or isinstance(docs, bool)
        or docs < 0
        or not isinstance(chunks, int)
        or isinstance(chunks, bool)
        or chunks < 0
    ):
        raise EmbedStateError(f"embed checkpoint counters are invalid at {path}")
    if not isinstance(value["complete"], bool):
        raise EmbedStateError(f"embed checkpoint complete flag is invalid at {path}")
    cursor = value["cursor"]
    if cursor is None:
        if docs != 0 or chunks != 0:
            raise EmbedStateError(f"empty checkpoint cursor has non-zero counters at {path}")
    else:
        cursor_keys = {"global_index", "source", "document_id", "version_id"}
        if not isinstance(cursor, dict) or set(cursor) != cursor_keys:
            raise EmbedStateError(f"embed checkpoint cursor shape is invalid at {path}")
        global_index = cursor["global_index"]
        shard_value = expected_initial["shard"]
        if (
            not isinstance(global_index, int)
            or isinstance(global_index, bool)
            or global_index < 0
            or global_index % shard_value["count"] != shard_value["index"]
            or cursor["source"] != expected_initial["source"]
            or not isinstance(cursor["document_id"], str)
            or not cursor["document_id"]
            or not isinstance(cursor["version_id"], str)
            or not cursor["version_id"]
        ):
            raise EmbedStateError(f"embed checkpoint cursor values are invalid at {path}")
        if docs == 0 or chunks == 0:
            raise EmbedStateError(f"non-empty checkpoint cursor has empty counters at {path}")
    return dict(value)


def _open_checkpoint(
    binding: EmbedBinding,
    source: str,
    shard: tuple[int, int] | None,
    *,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    path = checkpoint_path(binding, source, shard)
    initial = _initial_checkpoint(binding, source, shard)
    if resume:
        actual = _load_private_json(path, label="embed checkpoint")
        return path, _validate_checkpoint(actual, initial, path=path)
    _create_private_json(path, initial)
    return path, initial


def preflight_checkpoints(
    binding: EmbedBinding,
    sources: list[str] | tuple[str, ...],
    shard: tuple[int, int] | None,
    *,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    """Create fresh cursors or validate resume cursors before client/model access."""

    prepared: dict[str, dict[str, Any]] = {}
    pending_fresh: list[tuple[str, Path, dict[str, Any]]] = []
    for source in sources:
        path = checkpoint_path(binding, source, shard)
        initial = _initial_checkpoint(binding, source, shard)
        if resume:
            actual = _load_private_json(path, label="embed checkpoint")
            prepared[source] = _validate_checkpoint(actual, initial, path=path)
            continue
        try:
            path.lstat()
        except FileNotFoundError:
            pending_fresh.append((source, path, initial))
            continue
        except OSError as exc:
            raise EmbedStateError(f"cannot inspect embed checkpoint {path}: {exc}") from exc
        raise EmbedStateError(
            f"fresh embed refuses existing checkpoint {path}; use --resume"
        )
    for source, path, initial in pending_fresh:
        _create_private_json(path, initial)
        prepared[source] = initial
    return prepared


def _coordinator_value(
    binding: EmbedBinding,
    sources: Sequence[str],
) -> dict[str, Any]:
    worker_count = binding.value.get("worker_count")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise EmbedStateError("embed binding has invalid worker_count")
    canonical_sources = tuple(sources)
    if canonical_sources != SOURCES:
        raise EmbedStateError("coordinator must initialize exactly all seven sources")
    checkpoints: list[dict[str, Any]] = []
    for shard_index in range(worker_count):
        shard = (shard_index, worker_count)
        for source in canonical_sources:
            initial = _initial_checkpoint(binding, source, shard)
            path = checkpoint_path(binding, source, shard)
            checkpoints.append(
                {
                    "source": source,
                    "shard": {"index": shard_index, "count": worker_count},
                    "path": str(path.relative_to(binding.path.parent)),
                    "initial_sha256": hashlib.sha256(
                        _canonical_json_bytes(initial)
                    ).hexdigest(),
                }
            )
    return {
        "schema_version": COORDINATOR_SCHEMA_VERSION,
        "generation_id": binding.value["generation_id"],
        "variant_id": binding.variant_id,
        "physical_collection": binding.value["physical_collection"],
        "binding_sha256": binding_sha256(binding),
        "worker_count": worker_count,
        "sources": list(canonical_sources),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
    }


def initialize_coordinator(
    binding: EmbedBinding,
    *,
    sources: Sequence[str] = SOURCES,
    recover: bool = False,
) -> EmbedCoordinator:
    """Create every zero checkpoint, then publish one immutable coordinator marker.

    The marker is written last.  A crash can therefore leave diagnostic state but can never
    make a partial checkpoint topology eligible for a worker resume.
    """

    expected = _coordinator_value(binding, sources)
    path = coordinator_path(binding)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise EmbedStateError(f"cannot inspect coordinator state {path}: {exc}") from exc
    else:
        if recover:
            return load_coordinator(binding)
        raise EmbedStateError(f"immutable coordinator already exists: {path}")

    worker_count = int(binding.value["worker_count"])
    # Refuse all collisions before creating the first checkpoint unless an exact
    # marker-absent recovery is requested. Workers can never start without the marker,
    # so every recovered checkpoint must still equal its immutable zero state.
    for shard_index in range(worker_count):
        for source in sources:
            checkpoint = checkpoint_path(
                binding, source, (shard_index, worker_count)
            )
            if os.path.lexists(checkpoint) and not recover:
                raise EmbedStateError(
                    f"fresh coordinator refuses existing checkpoint {checkpoint}"
                )
    for shard_index in range(worker_count):
        shard = (shard_index, worker_count)
        for source in sources:
            checkpoint = checkpoint_path(binding, source, shard)
            initial = _initial_checkpoint(binding, source, shard)
            if os.path.lexists(checkpoint):
                actual = _load_private_json(checkpoint, label="embed checkpoint")
                _validate_checkpoint(actual, initial, path=checkpoint)
                if actual != initial:
                    raise EmbedStateError(
                        "marker-absent coordinator recovery found a non-zero checkpoint: "
                        f"{checkpoint}"
                    )
            else:
                _create_private_json(checkpoint, initial)
    _create_private_json(path, expected)
    coordinator = load_coordinator(binding)
    # Before returning from initialization, all checkpoints must still be exact zero state.
    for shard_index in range(worker_count):
        for source in sources:
            checkpoint = checkpoint_path(
                binding, source, (shard_index, worker_count)
            )
            actual = _load_private_json(checkpoint, label="embed checkpoint")
            initial = _initial_checkpoint(
                binding, source, (shard_index, worker_count)
            )
            if actual != initial:
                raise EmbedStateError(
                    f"checkpoint changed during coordinator initialization: {checkpoint}"
                )
    return coordinator


def load_coordinator(binding: EmbedBinding) -> EmbedCoordinator:
    """Validate the immutable topology marker and every named checkpoint identity."""

    path = coordinator_path(binding)
    actual = _load_private_json(path, label="embed coordinator")
    expected = _coordinator_value(binding, SOURCES)
    if actual != expected:
        raise EmbedStateError("embed coordinator does not match binding/topology")
    root = binding.path.parent.resolve(strict=True)
    for entry in actual["checkpoints"]:
        checkpoint = (binding.path.parent / entry["path"]).resolve(strict=True)
        if checkpoint.parent.parent != root / "variants" / binding.variant_id:
            raise EmbedStateError("coordinator checkpoint escapes the binding state root")
        shard = entry["shard"]
        initial = _initial_checkpoint(
            binding,
            entry["source"],
            (shard["index"], shard["count"]),
        )
        value = _load_private_json(checkpoint, label="embed checkpoint")
        _validate_checkpoint(value, initial, path=checkpoint)
    return EmbedCoordinator(path=path, value=actual)


def _batched(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _retrieve_exact_points(client, cfg: Config, point_ids: list[str]) -> dict[str, Any]:
    """Read exactly the requested deterministic IDs and reject missing/extra/duplicates."""

    records: dict[str, Any] = {}
    for batch in _batched(point_ids, RESUME_RETRIEVE_BATCH_SIZE):
        try:
            result = client.retrieve(
                collection_name=cfg.collection_name,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 - unavailable proof must fail closed
            raise EmbedStateError(
                f"cannot retrieve acknowledged candidate points: {exc}"
            ) from exc
        if not isinstance(result, (list, tuple)):
            raise EmbedStateError("Qdrant retrieve returned an invalid point sequence")
        requested = set(batch)
        for record in result:
            point_id = str(getattr(record, "id", ""))
            if point_id not in requested:
                raise EmbedStateError(
                    f"Qdrant retrieve returned unrequested point {point_id!r}"
                )
            if point_id in records:
                raise EmbedStateError(
                    f"Qdrant retrieve returned duplicate point {point_id!r}"
                )
            records[point_id] = record
    missing = sorted(set(point_ids) - set(records))
    if missing:
        sample = ", ".join(missing[:3])
        raise EmbedStateError(
            f"acknowledged deterministic points are missing from Qdrant: {sample}"
        )
    return records


def _inventory_chunk(
    doc: CanonicalDoc,
    value: Mapping[str, Any],
    *,
    count_tokens,
) -> Chunk:
    start = value["char_start"]
    end = value["char_end"]
    text = doc.body_markdown[start:end]
    encoded = text.encode("utf-8")
    passage_sha = hashlib.sha256(encoded).hexdigest()
    if (
        passage_sha != value["canonical_passage_sha256"]
        or len(text) != value["canonical_passage_char_length"]
        or len(encoded) != value["canonical_passage_utf8_length"]
        or count_tokens(text) != value["token_count"]
    ):
        raise EmbedStateError(
            f"sealed inventory passage differs from snapshot body for "
            f"{doc.source}:{doc.document_id}:{doc.version_id}:{value['chunk_index']}"
        )
    structure = value["structure"]
    page = value["page"]
    return Chunk(
        text=text,
        chunk_index=value["chunk_index"],
        heading_path=list(structure["heading_path"]),
        token_count=value["token_count"],
        char_start=start,
        char_end=end,
        canonical_text=text,
        passage_hash=passage_sha,
        article_id=structure["article_id"],
        article_label=structure["article_label"],
        article_start=structure["article_start"],
        clause=structure["clause"],
        clause_id=structure["clause_id"],
        subarticle=structure["subarticle"],
        chapter=structure["chapter"],
        parent_id=structure["parent_id"],
        clause_ids=tuple(structure["clause_ids"]),
        subarticle_ids=tuple(structure["subarticle_ids"]),
        page_start=page["page_start"],
        page_end=page["page_end"],
        page_coordinate_reason=page["page_coordinate_reason"],
        article_start_chunk_index=structure["article_start_chunk_index"],
        parent_chunk_index=structure["parent_chunk_index"],
        chunker_revision=structure["chunker_revision"],
    )


def _validate_acknowledged_inventory_point(
    cfg: Config,
    doc: CanonicalDoc,
    inventory_chunk: Mapping[str, Any],
    record: Any,
    *,
    document_chunk_count: int,
    document_state_hash: str,
    count_tokens,
) -> None:
    chunk = _inventory_chunk(doc, inventory_chunk, count_tokens=count_tokens)
    expected = store.build_payload(
        doc,
        chunk,
        document_chunk_count=document_chunk_count,
        document_state_hash=document_state_hash,
        cfg=cfg,
    )
    # Qdrant's JSON boundary canonicalizes any promoted tuple values to arrays.
    expected = json.loads(
        json.dumps(
            expected,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    payload = getattr(record, "payload", None)
    if not isinstance(payload, Mapping):
        raise EmbedStateError(
            f"acknowledged point {getattr(record, 'id', None)!r} lacks payload"
        )
    actual = dict(payload)
    if actual != expected:
        mismatches = sorted(
            field
            for field in set(actual) | set(expected)
            if actual.get(field) != expected.get(field)
            or (field in actual) != (field in expected)
        )
        raise EmbedStateError(
            f"acknowledged point {getattr(record, 'id', None)!r} exact payload "
            "mismatch: " + ", ".join(mismatches[:20])
        )

    header: dict[str, Any] = {}
    if cfg.embed_header_v2:
        header = {
            "document_number": actual["document_number"],
            "date": actual["date"] or actual["date_raw"],
            "status": actual["status"],
            "is_consolidated": actual["is_consolidated"],
        }
    embed_text = build_embed_text(
        actual["text"],
        title=actual["title"],
        document_type=actual["document_type"],
        heading_path=actual["heading_path"],
        **header,
    )
    embed_bytes = embed_text.encode("utf-8")
    embed_identity = {
        "sha256": hashlib.sha256(embed_bytes).hexdigest(),
        "char_length": len(embed_text),
        "utf8_length": len(embed_bytes),
        "token_count": count_tokens(embed_text),
    }
    if embed_identity != inventory_chunk["embed_input"]:
        raise EmbedStateError(
            f"acknowledged point {getattr(record, 'id', None)!r} embed-input "
            "identity differs from sealed inventory"
        )


def verify_resume_checkpoint_points(
    cfg: Config,
    client,
    sealed: SealedSnapshot,
    checkpoints: Mapping[str, Mapping[str, Any]],
    *,
    count_tokens,
) -> None:
    """Replay every acknowledged point against the exact sealed structural inventory.

    This gate runs before model loading or any resumed upsert.  It streams and rehashes the
    complete inventory, pairs every inventory document with the independently sealed source
    feed, and exact-compares each checkpoint-skipped point's complete deterministic payload
    and context-enriched encoder-input identity.
    """

    validate_snapshot_build_config(sealed, cfg)
    expected_identity = chunk_inventory.chunk_inventory_identity(
        sources=SOURCES,
        tokenizer_model=cfg.tokenizer_model,
        tokenizer_revision=cfg.tokenizer_revision,
        max_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        document_header=cfg.embed_header_v2,
    )
    manifest_entry = sealed.manifest["structural_chunk_inventory"]
    inventory_path = sealed.root / str(manifest_entry["path"])
    source_iterators = {
        source: iter(iter_snapshot_docs(source, root=sealed.docs, strict=True))
        for source in SOURCES
    }
    source_indexes = dict.fromkeys(SOURCES, 0)
    cursor_found = {
        source: checkpoint["cursor"] is None
        for source, checkpoint in checkpoints.items()
    }
    document_totals = dict.fromkeys(checkpoints, 0)
    chunk_totals = dict.fromkeys(checkpoints, 0)
    pending: list[tuple[str, CanonicalDoc, Mapping[str, Any], int, str]] = []
    pending_ids: set[str] = set()

    def prove_pending() -> None:
        if not pending:
            return
        ids = [item[0] for item in pending]
        records = _retrieve_exact_points(client, cfg, ids)
        for point_id, doc, chunk_value, chunk_count, state_hash in pending:
            _validate_acknowledged_inventory_point(
                cfg,
                doc,
                chunk_value,
                records[point_id],
                document_chunk_count=chunk_count,
                document_state_hash=state_hash,
                count_tokens=count_tokens,
            )
        pending.clear()
        pending_ids.clear()

    try:
        rows = chunk_inventory.iter_validated_inventory(
            inventory_path,
            manifest_entry=manifest_entry,
            expected_identity=expected_identity,
        )
        for row in rows:
            source = row["source"]
            global_index = source_indexes[source]
            source_indexes[source] += 1
            try:
                doc = next(source_iterators[source])
            except StopIteration as exc:
                raise EmbedStateError(
                    f"sealed inventory has an extra document for source {source!r}"
                ) from exc
            snapshot_doc = doc
            boundaries = [boundary.to_dict() for boundary in snapshot_doc.page_boundaries]
            body_bytes = snapshot_doc.body_markdown.encode("utf-8")
            if (
                row["document_id"] != snapshot_doc.document_id
                or row["version_id"] != snapshot_doc.version_id
                or row["canonical_content_sha256"]
                != hashlib.sha256(body_bytes).hexdigest()
                or row["canonical_body_char_length"] != len(snapshot_doc.body_markdown)
                or row["canonical_body_utf8_length"] != len(body_bytes)
                or row["page_boundaries"] != boundaries
                or row["page_coordinate_reason"]
                != snapshot_doc.page_coordinate_reason
            ):
                raise EmbedStateError(
                    f"sealed inventory/document mismatch at {source}[{global_index}]"
                )
            # The inventory binds the sealed source record.  The indexed payload then
            # derives its canonical version identity from that exact record.
            doc = _prepare_doc_for_index(snapshot_doc)

            checkpoint = checkpoints.get(source)
            if checkpoint is None:
                continue
            cursor = checkpoint["cursor"]
            resume_index = cursor["global_index"] if cursor is not None else -1
            if global_index == resume_index:
                expected_cursor = {
                    "global_index": global_index,
                    "source": doc.source,
                    "document_id": doc.document_id,
                    "version_id": doc.version_id,
                }
                if cursor != expected_cursor:
                    raise EmbedStateError(
                        f"resume cursor no longer matches snapshot at "
                        f"{source}[{global_index}]"
                    )
                cursor_found[source] = True
            assigned = (
                global_index % checkpoint["shard"]["count"]
                == checkpoint["shard"]["index"]
            )
            acknowledged = assigned and global_index <= resume_index
            if checkpoint["complete"] and assigned and not acknowledged:
                raise EmbedStateError(
                    f"completed checkpoint for {source!r} stops before an assigned "
                    "inventory document"
                )
            if not acknowledged:
                continue

            chunks = row["chunks"]
            document_totals[source] += 1
            chunk_totals[source] += len(chunks)
            state_hash = _document_state_hash(cfg, doc=doc)
            for chunk_value in chunks:
                point_id = store.point_id(
                    source,
                    doc.document_id,
                    chunk_value["chunk_index"],
                    version_id=doc.version_id,
                )
                if point_id in pending_ids:
                    raise EmbedStateError(
                        "snapshot checkpoint prefix has duplicate chunk identity"
                    )
                pending.append(
                    (point_id, doc, chunk_value, len(chunks), state_hash)
                )
                pending_ids.add(point_id)
                if len(pending) >= RESUME_RETRIEVE_BATCH_SIZE:
                    prove_pending()
        prove_pending()
    except chunk_inventory.ChunkInventoryError as exc:
        raise EmbedStateError(f"cannot replay sealed structural inventory: {exc}") from exc

    for source, iterator in source_iterators.items():
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise EmbedStateError(
                f"sealed snapshot has a document absent from inventory for {source!r}"
            )
    for source, checkpoint in checkpoints.items():
        if not cursor_found[source]:
            raise EmbedStateError(
                f"resume cursor was not found in snapshot source {source!r}"
            )
        if document_totals[source] != checkpoint["documents_completed"]:
            raise EmbedStateError(
                f"checkpoint document count mismatch for {source!r}: "
                f"state={checkpoint['documents_completed']}, "
                f"inventory={document_totals[source]}"
            )
        if chunk_totals[source] != checkpoint["chunks_completed"]:
            raise EmbedStateError(
                f"checkpoint chunk count mismatch for {source!r}: "
                f"state={checkpoint['chunks_completed']}, "
                f"inventory={chunk_totals[source]}"
            )


def embed_docs(
    cfg: Config,
    client,
    embedder,
    count_tokens,
    docs: list[CanonicalDoc],
    *,
    batch_size: int = 256,
    mutation_capability: store.ReviewedMutationCapability | None = None,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> tuple[int, int, int]:
    """Embed an explicit list; any conversion/chunk/embedding failure is fatal."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    store.require_reviewed_mutation_capability(
        cfg,
        mutation_capability,
        operation="upsert",
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    pending: list[Any] = []
    n_docs = n_chunks = 0
    for doc in docs:
        try:
            points, chunk_count = _build_doc_points(cfg, embedder, count_tokens, doc)
        except Exception as exc:  # noqa: BLE001 - immutable builds never skip
            raise EmbedStateError(
                f"embedding failed for {doc.source}:{doc.document_id}:{doc.version_id}: {exc}"
            ) from exc
        if (
            not points
            or not isinstance(chunk_count, int)
            or chunk_count <= 0
            or len(points) != chunk_count
        ):
            raise EmbedStateError(
                f"embedding produced invalid chunk accounting for "
                f"{doc.source}:{doc.document_id}:{doc.version_id}"
            )
        pending.extend(points)
        n_docs += 1
        n_chunks += chunk_count
        if len(pending) >= batch_size:
            store.upsert_points(
                client,
                cfg.collection_name,
                pending,
                wait=True,
                mutation_capability=mutation_capability,
                reviewed_plan_sha256=reviewed_plan_sha256,
                launch_evidence_sha256=launch_evidence_sha256,
                storage_identity_sha256=storage_identity_sha256,
            )
            pending.clear()
    if pending:
        store.upsert_points(
            client,
            cfg.collection_name,
            pending,
            wait=True,
            mutation_capability=mutation_capability,
            reviewed_plan_sha256=reviewed_plan_sha256,
            launch_evidence_sha256=launch_evidence_sha256,
            storage_identity_sha256=storage_identity_sha256,
        )
    return n_docs, n_chunks, 0


def embed_source_resumable(
    cfg: Config,
    client,
    embedder,
    count_tokens,
    source: str,
    *,
    binding: EmbedBinding,
    snapshot_docs: Path,
    resume: bool,
    prepared_checkpoint: Mapping[str, Any] | None = None,
    batch_size: int = 256,
    shard: tuple[int, int] | None = None,
    mutation_capability: store.ReviewedMutationCapability | None = None,
    launch_evidence_sha256: str | None = None,
) -> tuple[int, int, int]:
    """Embed one source and durably advance an exact source/version/global cursor."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    reviewed_plan_sha256 = binding.value.get("reviewed_plan_sha256")
    storage_identity_sha256 = binding.value.get("storage_identity_sha256")
    store.require_reviewed_mutation_capability(
        cfg,
        mutation_capability,
        operation="upsert",
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    bound_docs = Path(str(binding.value["snapshot"]["docs"]))
    if snapshot_docs.resolve(strict=True) != bound_docs:
        raise EmbedStateError("snapshot docs do not match the immutable embed binding")
    if cfg.collection_name != binding.value["physical_collection"]:
        raise EmbedStateError("configured collection does not match the immutable embed binding")

    shard_i, shard_n = shard if shard is not None else (0, 1)
    if prepared_checkpoint is None:
        path, checkpoint = _open_checkpoint(
            binding,
            source,
            shard,
            resume=resume,
        )
    else:
        path = checkpoint_path(binding, source, shard)
        initial = _initial_checkpoint(binding, source, shard)
        actual = _load_private_json(path, label="embed checkpoint")
        checkpoint = _validate_checkpoint(actual, initial, path=path)
        if checkpoint != dict(prepared_checkpoint):
            raise EmbedStateError(
                f"embed checkpoint changed after startup validation: {path}"
            )
    if checkpoint["complete"]:
        return 0, 0, 0

    cursor = checkpoint["cursor"]
    resume_index = cursor["global_index"] if cursor is not None else -1
    cursor_found = cursor is None
    pending: list[Any] = []
    pending_docs = pending_chunks = 0
    pending_cursor: dict[str, Any] | None = None
    invocation_docs = invocation_chunks = 0

    def flush() -> None:
        nonlocal checkpoint, pending_docs, pending_chunks, pending_cursor
        if not pending:
            return
        # Qdrant's wait=True contract means a successful return is the durability
        # acknowledgement.  The checkpoint is replaced only after that return.
        store.upsert_points(
            client,
            cfg.collection_name,
            pending,
            wait=True,
            mutation_capability=mutation_capability,
            reviewed_plan_sha256=reviewed_plan_sha256,
            launch_evidence_sha256=launch_evidence_sha256,
            storage_identity_sha256=storage_identity_sha256,
        )
        assert pending_cursor is not None
        updated = dict(checkpoint)
        updated["cursor"] = dict(pending_cursor)
        updated["documents_completed"] += pending_docs
        updated["chunks_completed"] += pending_chunks
        _replace_checkpoint(path, updated, expected_previous=checkpoint)
        checkpoint = updated
        pending.clear()
        pending_docs = 0
        pending_chunks = 0
        pending_cursor = None

    for global_index, doc in enumerate(
        iter_snapshot_docs(source, root=snapshot_docs, strict=True)
    ):
        if global_index <= resume_index:
            if global_index == resume_index:
                expected_cursor = {
                    "global_index": global_index,
                    "source": doc.source,
                    "document_id": doc.document_id,
                    "version_id": doc.version_id,
                }
                if expected_cursor != cursor:
                    raise EmbedStateError(
                        f"resume cursor no longer matches snapshot at {source}[{global_index}]"
                    )
                cursor_found = True
            continue
        if not cursor_found:
            raise EmbedStateError(f"resume cursor was not found in snapshot source {source!r}")
        if global_index % shard_n != shard_i:
            continue
        if not isinstance(doc.version_id, str) or not doc.version_id:
            raise EmbedStateError(
                f"snapshot document {source}:{doc.document_id} has no immutable version_id"
            )
        try:
            points, chunk_count = _build_doc_points(cfg, embedder, count_tokens, doc)
        except Exception as exc:  # noqa: BLE001 - immutable builds never skip
            raise EmbedStateError(
                f"embedding failed for {source}:{doc.document_id}:{doc.version_id}: {exc}"
            ) from exc
        if (
            not points
            or not isinstance(chunk_count, int)
            or chunk_count <= 0
            or len(points) != chunk_count
        ):
            raise EmbedStateError(
                f"embedding produced invalid chunk accounting for "
                f"{source}:{doc.document_id}:{doc.version_id}"
            )
        pending.extend(points)
        pending_docs += 1
        pending_chunks += chunk_count
        invocation_docs += 1
        invocation_chunks += chunk_count
        pending_cursor = {
            "global_index": global_index,
            "source": source,
            "document_id": doc.document_id,
            "version_id": doc.version_id,
        }
        if len(pending) >= batch_size:
            flush()

    if not cursor_found:
        raise EmbedStateError(f"resume cursor was not found in snapshot source {source!r}")
    flush()
    completed = dict(checkpoint)
    completed["complete"] = True
    _replace_checkpoint(path, completed, expected_previous=checkpoint)
    return invocation_docs, invocation_chunks, 0
