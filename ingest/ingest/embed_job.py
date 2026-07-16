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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import qdrant_store as store
from .config import RETRIEVAL_FINGERPRINT_REVISION, Config
from .dedup import content_hash
from .generation import CANONICAL_PAYLOAD_REVISION, GENERATION_SCHEMA_VERSION
from .pipeline import _build_doc_points, _document_state_hash, _prepare_doc_for_index
from .sources import NORMALIZER_REVISION, CanonicalDoc

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

BINDING_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
EXPECTED_CANDIDATE_CHUNKING = {
    "tokens": 512,
    "overlap": 80,
    "min_tokens": 64,
}
RESUME_RETRIEVE_BATCH_SIZE = 256

# Fixed KA+EN checksum sentence for the CPU-vs-GPU vector-space identity check.
CHECKSUM_SENTENCE = "საქართველოს კანონი — Article 1: this sentence pins the BGE-M3 vector space."

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
    }
    actual = {
        "embed_model": build.get("embed_model"),
        "embedding_revision": build.get("embedding_revision"),
        "tokenizer.model": tokenizer.get("model"),
        "tokenizer.revision": tokenizer.get("revision"),
        "chunk.tokens": chunk.get("tokens"),
        "chunk.overlap": chunk.get("overlap"),
        "chunk.min_tokens": chunk.get("min_tokens"),
    }
    mismatches = [
        f"{field}: snapshot={actual[field]!r}, config={value!r}"
        for field, value in expected.items()
        if actual[field] != value
    ]
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


def _required_string(record: Mapping[str, Any], field: str) -> str:
    if field not in record:
        raise EmbedStateError(f"snapshot record is missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str) or not value:
        raise EmbedStateError(f"snapshot record field {field!r} must be a non-empty string")
    return value


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


def snapshot_doc_to_canonical(
    d: Mapping[str, Any], *, strict: bool = False
) -> CanonicalDoc:
    """Rebuild a canonical document; production snapshot iteration is always strict."""

    if strict:
        _validate_strict_snapshot_record(d)
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


def dense_checksum(embedder) -> tuple[str, list[float]]:
    """Return a display digest and full vector for the fixed vector-space sentence."""

    vec = [float(x) for x in embedder.encode_query(CHECKSUM_SENTENCE).dense]
    digest = hashlib.sha256(
        json.dumps([round(x, 3) for x in vec], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return digest, vec


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
    digest, vec = dense_checksum(embedder)
    _create_private_json(
        output,
        {"sha": digest, "sentence": CHECKSUM_SENTENCE, "dense": vec},
    )
    return digest


def _binding_value(cfg: Config, sealed: SealedSnapshot) -> dict[str, Any]:
    validate_snapshot_build_config(sealed, cfg)
    identity = store.validate_generation_identity(cfg)
    assert cfg.generation_id is not None
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
        },
        "models": {
            "embedding": {"name": identity.embedding_model, "revision": identity.embedding_revision},
            "tokenizer": {"name": identity.tokenizer_model, "revision": identity.tokenizer_revision},
            "reranker": {"name": identity.reranker_model, "revision": identity.reranker_revision},
        },
        "vector_space_id": identity.vector_space_id,
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


def prepare_binding(cfg: Config, sealed: SealedSnapshot, *, resume: bool) -> EmbedBinding:
    """Create a fresh binding or require an exact existing binding for resume."""

    expected = _binding_value(cfg, sealed)
    path = binding_path(cfg)
    if resume:
        actual = _load_private_json(path, label="embed binding")
        if actual != expected:
            raise EmbedStateError(
                "embed binding does not match the selected snapshot, physical collection, "
                "models, chunking, retrieval identity, or payload schema"
            )
    else:
        _create_private_json(path, expected)
        actual = expected
    return EmbedBinding(path=path, value=actual)


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


_RESUME_PROOF_PAYLOAD_FIELDS = (
    "schema_version",
    "canonical_payload_revision",
    "generation_id",
    "embedding_model",
    "embedding_revision",
    "model_revision",
    "tokenizer_model",
    "tokenizer_revision",
    "reranker_model",
    "reranker_revision",
    "vector_space_id",
    "chunking_fingerprint",
    "document_header",
    "retrieval_fingerprint",
    "retrieval_fingerprint_revision",
    "source",
    "document_id",
    "version_id",
    "chunk_index",
    "document_chunk_count",
    "document_state_hash",
    "content_hash",
    "canonical_content_hash",
    "source_fingerprint",
    "normalizer_revision",
    "content_complete",
    "extraction_status",
    "source_authority",
    "canonical_text_exact",
    "passage_hash",
)


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
                with_payload=list(_RESUME_PROOF_PAYLOAD_FIELDS),
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


def _checkpoint_prefix_documents(
    sealed: SealedSnapshot,
    source: str,
    checkpoint: Mapping[str, Any],
) -> Iterator[CanonicalDoc]:
    """Replay exactly the shard documents a checkpoint claims it will skip."""

    cursor = checkpoint["cursor"]
    shard = checkpoint["shard"]
    shard_i = shard["index"]
    shard_n = shard["count"]
    complete = checkpoint["complete"]
    if cursor is None and not complete:
        return
    resume_index = cursor["global_index"] if cursor is not None else -1
    cursor_found = cursor is None
    for global_index, doc in enumerate(
        iter_snapshot_docs(source, root=sealed.docs, strict=True)
    ):
        assigned = global_index % shard_n == shard_i
        if global_index <= resume_index:
            if assigned:
                yield doc
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
        if not complete:
            break
        if assigned:
            raise EmbedStateError(
                f"completed checkpoint for {source!r} stops before an assigned snapshot document"
            )
    if not cursor_found:
        raise EmbedStateError(f"resume cursor was not found in snapshot source {source!r}")


def _validate_acknowledged_point(
    doc: CanonicalDoc,
    record: Any,
    *,
    chunk_index: int,
    document_chunk_count: int,
    document_state_hash: str,
    identity: Mapping[str, Any],
    body_hash: str,
) -> None:
    payload = getattr(record, "payload", None)
    if not isinstance(payload, Mapping):
        raise EmbedStateError(f"acknowledged point {getattr(record, 'id', None)!r} lacks payload")
    expected = {
        **identity,
        "source": doc.source,
        "document_id": doc.document_id,
        "version_id": doc.version_id,
        "chunk_index": chunk_index,
        "document_chunk_count": document_chunk_count,
        "document_state_hash": document_state_hash,
        "content_hash": body_hash,
        "canonical_content_hash": body_hash,
        "source_fingerprint": doc.source_fingerprint,
        "normalizer_revision": doc.normalizer_revision,
        "content_complete": True,
        "extraction_status": "full_text",
        "source_authority": doc.source_authority,
        "canonical_text_exact": True,
    }
    mismatches = [
        field for field, value in expected.items() if payload.get(field) != value
    ]
    if mismatches:
        raise EmbedStateError(
            f"acknowledged point {getattr(record, 'id', None)!r} payload mismatch: "
            + ", ".join(sorted(mismatches))
        )
    passage_hash = payload.get("passage_hash")
    if not isinstance(passage_hash, str) or not _SHA256_RE.fullmatch(passage_hash):
        raise EmbedStateError(
            f"acknowledged point {getattr(record, 'id', None)!r} has invalid passage_hash"
        )


def verify_resume_checkpoint_points(
    cfg: Config,
    client,
    sealed: SealedSnapshot,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove every checkpoint-skipped deterministic chunk exists and is contiguous.

    The earlier collection-wide identity count cannot prove set membership: an unrelated
    same-identity point could mask deletion of an acknowledged point.  This read-only gate
    replays each exact snapshot prefix, retrieves chunk zero to obtain the sealed document's
    recorded chunk count, then retrieves every deterministic UUID in that contiguous range.
    """

    validate_snapshot_build_config(sealed, cfg)
    identity = store.validate_generation_identity(cfg).as_payload()
    for source, checkpoint in checkpoints.items():
        document_total = 0
        chunk_total = 0
        doc_batch: list[CanonicalDoc] = []

        def prove_batch() -> None:
            nonlocal document_total, chunk_total
            if not doc_batch:
                return
            prepared_docs = [_prepare_doc_for_index(doc) for doc in doc_batch]
            zero_ids = [
                store.point_id(
                    doc.source,
                    doc.document_id,
                    0,
                    version_id=doc.version_id,
                )
                for doc in prepared_docs
            ]
            if len(set(zero_ids)) != len(zero_ids):
                raise EmbedStateError("snapshot checkpoint prefix has duplicate point identity")
            zero_records = _retrieve_exact_points(client, cfg, zero_ids)
            remaining: list[tuple[str, CanonicalDoc, int, int, str, str]] = []
            for doc, zero_id in zip(prepared_docs, zero_ids, strict=True):
                zero = zero_records[zero_id]
                payload = getattr(zero, "payload", None)
                count = payload.get("document_chunk_count") if isinstance(payload, Mapping) else None
                if (
                    not isinstance(count, int)
                    or isinstance(count, bool)
                    or count <= 0
                    or chunk_total + count > checkpoint["chunks_completed"]
                ):
                    raise EmbedStateError(
                        f"acknowledged chunk count is invalid for "
                        f"{doc.source}:{doc.document_id}:{doc.version_id}"
                    )
                state_hash = _document_state_hash(cfg, doc=doc)
                body_hash = content_hash(doc.body_markdown)
                _validate_acknowledged_point(
                    doc,
                    zero,
                    chunk_index=0,
                    document_chunk_count=count,
                    document_state_hash=state_hash,
                    identity=identity,
                    body_hash=body_hash,
                )
                for chunk_index in range(1, count):
                    point_id = store.point_id(
                        doc.source,
                        doc.document_id,
                        chunk_index,
                        version_id=doc.version_id,
                    )
                    remaining.append(
                        (point_id, doc, chunk_index, count, state_hash, body_hash)
                    )
                document_total += 1
                chunk_total += count
            remaining_ids = [item[0] for item in remaining]
            if len(set(remaining_ids)) != len(remaining_ids):
                raise EmbedStateError("snapshot checkpoint prefix has duplicate chunk identity")
            records = _retrieve_exact_points(client, cfg, remaining_ids)
            for point_id, doc, chunk_index, count, state_hash, body_hash in remaining:
                _validate_acknowledged_point(
                    doc,
                    records[point_id],
                    chunk_index=chunk_index,
                    document_chunk_count=count,
                    document_state_hash=state_hash,
                    identity=identity,
                    body_hash=body_hash,
                )
            doc_batch.clear()

        for doc in _checkpoint_prefix_documents(sealed, source, checkpoint):
            doc_batch.append(doc)
            if len(doc_batch) >= RESUME_RETRIEVE_BATCH_SIZE:
                prove_batch()
        prove_batch()
        if document_total != checkpoint["documents_completed"]:
            raise EmbedStateError(
                f"checkpoint document count mismatch for {source!r}: "
                f"state={checkpoint['documents_completed']}, snapshot={document_total}"
            )
        if chunk_total != checkpoint["chunks_completed"]:
            raise EmbedStateError(
                f"checkpoint chunk count mismatch for {source!r}: "
                f"state={checkpoint['chunks_completed']}, Qdrant={chunk_total}"
            )


def embed_docs(
    cfg: Config,
    client,
    embedder,
    count_tokens,
    docs: list[CanonicalDoc],
    *,
    batch_size: int = 256,
) -> tuple[int, int, int]:
    """Embed an explicit list; any conversion/chunk/embedding failure is fatal."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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
            store.upsert_points(client, cfg.collection_name, pending, wait=True)
            pending.clear()
    if pending:
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
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
) -> tuple[int, int, int]:
    """Embed one source and durably advance an exact source/version/global cursor."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
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
