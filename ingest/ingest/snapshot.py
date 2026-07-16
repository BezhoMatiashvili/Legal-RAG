"""Build and verify immutable clean-corpus snapshots.

Production snapshots are selected from an explicit, hash-attested run ledger.  A
snapshot is assembled in a private staging directory, sealed with a complete file
inventory, fsynced, and published with an atomic no-replace rename.  The deliberately
separate ``--preflight`` mode is useful for local corpus inspection, but its manifest is
marked and downstream production consumers must reject it.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import dedup, hygiene, structure
from .artifacts import parse_utc_datetime
from .config import REPO_ROOT, Config
from .source_state import (
    FileAttestation,
    PRODUCTION_SOURCES,
    SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
    SourceStateError,
    iter_attested_lines,
    load_source_state_evidence,
)
from .sources import SOURCES, finalize_canonical_text, normalize

SNAPSHOT_MANIFEST_SCHEMA_VERSION = 2
SNAPSHOT_PIPELINE_VERSION = "2"
SOURCES_PRESENT = PRODUCTION_SOURCES
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "ingest" / "snapshots"
FROZEN_V1_ROOT = DEFAULT_SNAPSHOT_ROOT / "v1"

_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,127}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_MANIFEST_KEYS = {
    "schema_version",
    "snapshot_id",
    "pipeline_version",
    "preflight",
    "created_at",
    "config_hash",
    "build",
    "source_state_evidence",
    "runs",
    "totals",
    "sources",
    "files",
    "corpus_sha256",
    "snapshot_sha256",
}


class SnapshotSafetyError(RuntimeError):
    """Raised when a snapshot cannot be proven safe and immutable."""


def _run_files_desc(source: str, artifacts_root: Path) -> list[tuple[str, str]]:
    """Legacy delta-builder discovery helper (new sealed builds never call this)."""
    paths: list[tuple[str, str]] = []
    for path in (artifacts_root / source / "runs").glob("*/items.jsonl"):
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
            paths.append((path.parent.name, str(path)))
    paths.sort(key=lambda value: value[0], reverse=True)
    return paths


@dataclass(frozen=True, slots=True)
class _RunInput:
    source: str
    run_id: str
    items_path: Path
    items_sha256: str
    items_size_bytes: int
    completion_path: Path | None = None
    completion_sha256: str | None = None
    completion_size_bytes: int | None = None
    completed_at: str | None = None

    @property
    def success_verified(self) -> bool:
        return self.completion_path is not None

    def to_manifest(self) -> dict[str, object]:
        completion: dict[str, object] | None = None
        if self.completion_path is not None:
            completion = {
                "path": f"{self.source}/runs/{self.run_id}/run.json",
                "sha256": self.completion_sha256,
                "size_bytes": self.completion_size_bytes,
            }
        return {
            "source": self.source,
            "run_id": self.run_id,
            "items": {
                "path": f"{self.source}/runs/{self.run_id}/items.jsonl",
                "sha256": self.items_sha256,
                "size_bytes": self.items_size_bytes,
            },
            "completion_record": completion,
            "completed_at": self.completed_at,
            "success_verified": self.success_verified,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    """Reject every existing symlink component, including a broken final symlink."""
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise SnapshotSafetyError(f"refusing symlink path component: {cursor}")


def _hash_regular_file(path: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SnapshotSafetyError(f"required file is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SnapshotSafetyError(f"required path is not a regular non-symlink file: {path}")
    if max_bytes is not None and info.st_size > max_bytes:
        raise SnapshotSafetyError(f"file exceeds the {max_bytes}-byte safety limit: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    if size != info.st_size:
        raise SnapshotSafetyError(f"file changed while hashing: {path}")
    return digest.hexdigest(), size


def _read_json_object(path: Path, *, max_bytes: int = 1024 * 1024) -> dict[str, object]:
    _digest, size = _hash_regular_file(path, max_bytes=max_bytes)

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotSafetyError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise SnapshotSafetyError(f"non-finite JSON number {value!r}: {path}")

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotSafetyError(f"invalid JSON object: {path}: {exc}") from exc
    if len(raw) != size:
        raise SnapshotSafetyError(f"file changed while reading: {path}")
    if not isinstance(value, dict):
        raise SnapshotSafetyError(f"JSON value must be an object: {path}")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SnapshotSafetyError(f"{field} keys mismatch: missing={missing}, extra={extra}")


def _validate_snapshot_id(snapshot_id: str) -> None:
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID_RE.fullmatch(snapshot_id):
        raise SnapshotSafetyError(
            "snapshot_id must be 3-128 lowercase letters, digits, underscores, or hyphens"
        )
    if snapshot_id.startswith("v1"):
        raise SnapshotSafetyError("legacy v1* snapshot identifiers are forbidden")


def _iso_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_source_state_evidence(
    cfg: Config, evidence_path: Path
) -> tuple[list[_RunInput], dict[str, object]]:
    """Load exact run selections through the shared schema-v1 attestation consumer."""

    try:
        loaded = load_source_state_evidence(cfg.artifacts_root, evidence_path)
    except SourceStateError as exc:
        raise SnapshotSafetyError(str(exc)) from exc
    selected = [
        _RunInput(
            source=run.source,
            run_id=run.run_id,
            items_path=run.items.path,
            items_sha256=run.items.sha256,
            items_size_bytes=run.items.size_bytes,
            completion_path=run.completion.path,
            completion_sha256=run.completion.sha256,
            completion_size_bytes=run.completion.size_bytes,
            completed_at=run.completed_at,
        )
        for run in loaded.runs
    ]
    return selected, {
        "sha256": loaded.evidence.sha256,
        "size_bytes": loaded.evidence.size_bytes,
    }


def _discover_preflight_inputs(cfg: Config) -> list[_RunInput]:
    """Discover raw runs without interpreting them as success evidence."""
    selected: list[_RunInput] = []
    for source in SOURCES_PRESENT:
        runs_root = cfg.artifacts_root / source / "runs"
        if not os.path.lexists(runs_root):
            continue
        _reject_symlink_components(runs_root)
        if not runs_root.is_dir():
            raise SnapshotSafetyError(f"runs path is not a directory: {runs_root}")
        for run_dir in sorted(runs_root.iterdir(), key=lambda path: path.name):
            if run_dir.is_symlink():
                raise SnapshotSafetyError(f"refusing symlink run directory: {run_dir}")
            if not run_dir.is_dir():
                continue
            if not _RUN_ID_RE.fullmatch(run_dir.name):
                raise SnapshotSafetyError(f"unsafe run directory name: {run_dir.name!r}")
            items_path = run_dir / "items.jsonl"
            if not os.path.lexists(items_path):
                continue
            items_sha, items_size = _hash_regular_file(items_path)
            selected.append(
                _RunInput(
                    source=source,
                    run_id=run_dir.name,
                    items_path=items_path,
                    items_sha256=items_sha,
                    items_size_bytes=items_size,
                )
            )
    selected.sort(key=lambda item: (item.source, item.run_id))
    return selected


def config_hash(cfg: Config) -> str:
    """Hash the parameters that define canonical snapshot records."""
    material = {
        "pipeline_version": SNAPSHOT_PIPELINE_VERSION,
        "logic_rev": "r2",
        "hygiene": {
            "near_empty_chars": hygiene.NEAR_EMPTY_CHARS,
            "mojibake_ratio": hygiene.MOJIBAKE_RATIO,
        },
        "dedup": {
            "shingle_k": dedup.SHINGLE_K,
            "minhash_perms": dedup.MINHASH_PERMS,
            "near_dup_threshold": dedup.NEAR_DUP_THRESHOLD,
            "large_amendment_cluster": dedup.LARGE_AMENDMENT_CLUSTER,
        },
        "chunk": {
            "tokens": cfg.chunk_tokens,
            "overlap": cfg.chunk_overlap,
            "min_tokens": cfg.chunk_min_tokens,
        },
        "embed_model": cfg.embed_model,
        "embedding_revision": cfg.embedding_revision,
        "tokenizer_model": cfg.tokenizer_model,
        "tokenizer_revision": cfg.tokenizer_revision,
        "sources": sorted(SOURCES),
    }
    return _sha256_bytes(_canonical_json_bytes(material))[:16]


@dataclass
class SourceStats:
    raw_lines: int = 0
    unique: int = 0
    malformed: int = 0
    clean: int = 0
    quarantined: Counter = None  # type: ignore[assignment]  # reason -> count
    nul_docs: int = 0
    control_docs: int = 0
    mojibake_docs: int = 0
    nfc_changed: int = 0
    char_lens: list = None  # type: ignore[assignment]
    structure: Counter = None  # type: ignore[assignment]  # marker -> docs
    kinds: Counter = None  # type: ignore[assignment]  # primary_kind -> docs
    doc_types: Counter = None  # type: ignore[assignment]
    exact: dict = None  # type: ignore[assignment]
    amendment: dict = None  # type: ignore[assignment]
    near: dict = None  # type: ignore[assignment]
    token_pctl: dict | None = None
    pii_fields: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.quarantined = Counter()
        self.char_lens = []
        self.structure = Counter()
        self.kinds = Counter()
        self.doc_types = Counter()
        self.pii_fields = Counter()


def _snapshot_record(
    source: str,
    doc,
    clean_body: str,
    chash: str,
    sinfo,
    run_label: str,
    *,
    snapshot_id: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "snapshot_version": SNAPSHOT_PIPELINE_VERSION,
        "doc_id": f"{source}:{doc.document_id}:{doc.version_id}",
        "source": source,
        "document_id": doc.document_id,
        "content_hash": chash,
        "title": doc.title,
        "date": doc.date,
        "date_raw": doc.date_raw,
        "language": doc.language,
        "document_type": doc.document_type,
        "court": doc.court,
        "source_url": doc.source_url,
        "source_binary_url": doc.source_binary_url,
        "document_number": doc.document_number,
        "registration_code": doc.registration_code,
        "parties": doc.parties,
        "status": doc.status,
        "status_raw": doc.status_raw,
        "in_force_date": doc.in_force_date,
        "expiry_date": doc.expiry_date,
        "is_consolidated": doc.is_consolidated,
        "consolidated_count": doc.consolidated_count,
        "content_kind": doc.content_kind,
        "content_complete": doc.content_complete,
        "extraction_status": doc.extraction_status,
        "article_summary": doc.article_summary,
        "source_fingerprint": doc.source_fingerprint,
        "normalizer_revision": doc.normalizer_revision,
        "version_id": doc.version_id,
        "version_id_kind": doc.version_id_kind,
        "supersedes": list(doc.supersedes),
        "effective_from": doc.effective_from,
        "effective_to": doc.effective_to,
        "repeal_date": doc.repeal_date,
        "consolidation_status": doc.consolidation_status,
        "version_lineage_status": doc.version_lineage_status,
        "version_lineage_complete": doc.version_lineage_complete,
        "consolidated_dates": list(doc.consolidated_dates),
        "official_url": doc.official_url,
        "official_binary_url": doc.official_binary_url,
        "official_html_url": doc.official_html_url,
        "official_pdf_url": doc.official_pdf_url,
        "source_authority": doc.source_authority,
        "freshness_sla_met": doc.freshness_sla_met,
        "promoted": doc.promoted,
        "structure": {
            "primary_kind": sinfo.primary_kind,
            "has_article": sinfo.has_article,
            "has_heading": sinfo.has_heading,
            "has_num_clause": sinfo.has_num_clause,
            "article_count": sinfo.article_count,
        },
        "body_char_len": len(clean_body),
        "source_run": run_label,
        "body_markdown": clean_body,
    }
    if snapshot_id is not None:
        record["snapshot_id"] = snapshot_id
    return record


def _create_output_root(path: Path) -> Path:
    path = path.expanduser().absolute()
    _reject_symlink_components(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_components(path)
    if not path.is_dir():
        raise SnapshotSafetyError(f"output root is not a directory: {path}")
    return path


def _validate_build_request(
    cfg: Config,
    *,
    snapshot_id: str,
    output_root: Path,
    sources: Sequence[str],
    source_state_evidence: Path | None,
    preflight: bool,
    limit: int | None,
    token_sample: int,
) -> Path:
    _validate_snapshot_id(snapshot_id)
    if tuple(sources) != SOURCES_PRESENT:
        raise SnapshotSafetyError(
            "sealed snapshots require all seven sources in canonical order"
        )
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise SnapshotSafetyError("limit must be a positive integer")
    if limit is not None and not preflight:
        raise SnapshotSafetyError("--limit is permitted only with --preflight")
    if not preflight and source_state_evidence is None:
        raise SnapshotSafetyError(
            "production snapshots require --source-state-evidence"
        )
    if not isinstance(token_sample, int) or token_sample < 0:
        raise SnapshotSafetyError("token_sample must be a non-negative integer")
    if token_sample > 0 and (
        cfg.tokenizer_revision is None
        or not _IMMUTABLE_REVISION_RE.fullmatch(cfg.tokenizer_revision)
    ):
        raise SnapshotSafetyError(
            "positive token sampling requires an immutable TOKENIZER_REVISION"
        )

    output_root = output_root.expanduser().absolute()
    _reject_symlink_components(output_root)
    destination = output_root / snapshot_id
    resolved_destination = destination.resolve(strict=False)
    frozen = FROZEN_V1_ROOT.resolve(strict=False)
    if resolved_destination == frozen or _is_relative_to(resolved_destination, frozen):
        raise SnapshotSafetyError("refusing to publish into frozen ingest/snapshots/v1")
    preflight_root = (cfg.state_dir / "v3").resolve(strict=False)
    if preflight:
        if not _is_relative_to(resolved_destination, preflight_root):
            raise SnapshotSafetyError(
                "preflight snapshots must be published below ingest/.state/v3"
            )
    elif _is_relative_to(resolved_destination, preflight_root):
        raise SnapshotSafetyError(
            "only --preflight snapshots may be published below ingest/.state/v3"
        )
    if os.path.lexists(destination):
        raise SnapshotSafetyError(f"snapshot destination already exists: {destination}")
    return output_root


def _make_token_counter(cfg: Config, token_sample: int):
    if token_sample == 0:
        return None
    from .embedding import make_token_counter

    # Deliberately propagate download/import/tokenizer errors.  A sampled profile that
    # silently omitted its tokenizer result would not be an attested build.
    return make_token_counter(cfg.tokenizer_model, cfg.tokenizer_revision)


def _runs_by_source(runs: Sequence[_RunInput]) -> dict[str, list[_RunInput]]:
    grouped = {source: [] for source in SOURCES_PRESENT}
    for run in runs:
        grouped[run.source].append(run)
    for source in grouped:
        grouped[source].sort(key=lambda run: run.run_id, reverse=True)
    return grouped


def _iter_run_lines(run: _RunInput):
    """Read attested production bytes through the same no-follow hash binding."""

    if run.success_verified:
        attestation = FileAttestation(
            path=run.items_path,
            sha256=run.items_sha256,
            size_bytes=run.items_size_bytes,
        )
        try:
            for raw_line in iter_attested_lines(
                attestation,
                label=f"selected items.jsonl {run.source}/{run.run_id}",
            ):
                yield raw_line.decode("utf-8", errors="replace")
        except SourceStateError as exc:
            raise SnapshotSafetyError(str(exc)) from exc
        return
    with run.items_path.open(encoding="utf-8", errors="replace") as items_fp:
        yield from items_fp


def _build_corpus(
    *,
    snapshot_id: str,
    out_dir: Path,
    runs: Sequence[_RunInput],
    limit: int | None,
    near_dup: bool,
    token_sample: int,
    token_counter,
) -> dict[str, SourceStats]:
    (out_dir / "docs").mkdir(mode=0o700)
    (out_dir / "reports").mkdir(mode=0o700)
    grouped = _runs_by_source(runs)
    stats: dict[str, SourceStats] = {}
    quarantine_path = out_dir / "quarantine.jsonl"
    with quarantine_path.open("w", encoding="utf-8") as quarantine_fp:
        for source in SOURCES_PRESENT:
            source_stats = SourceStats()
            seen: set[tuple[str, str]] = set()
            exact_map: list[tuple[str, str]] = []
            amend_map: list[tuple[str, str]] = []
            docs_path = out_dir / "docs" / f"{source}.jsonl"
            stop = False
            with docs_path.open("w", encoding="utf-8") as docs_fp:
                for run in grouped[source]:
                    for line in _iter_run_lines(run):
                            line = line.strip()
                            if not line:
                                continue
                            source_stats.raw_lines += 1
                            try:
                                item = json.loads(line)
                                doc = normalize(source, item)
                            except Exception:
                                source_stats.malformed += 1
                                continue
                            raw = doc.body_markdown
                            clean_body = hygiene.clean_text(raw)
                            doc = finalize_canonical_text(doc, clean_body)
                            version_key = (doc.document_id, str(doc.version_id))
                            if version_key in seen:
                                continue
                            if limit is not None and source_stats.unique >= limit:
                                stop = True
                                break
                            seen.add(version_key)
                            source_stats.unique += 1
                            report = hygiene.assess(raw)
                            source_stats.nul_docs += int(bool(report.nul_chars))
                            source_stats.control_docs += int(bool(report.control_chars))
                            source_stats.mojibake_docs += int(bool(report.replacement_chars))
                            source_stats.nfc_changed += int(report.changed_by_nfc)
                            source_stats.doc_types[doc.document_type] += 1
                            for key in doc.promoted:
                                source_stats.pii_fields[key] += 1
                            doc_id = f"{source}:{doc.document_id}:{doc.version_id}"

                            incomplete_reason = None
                            if not doc.content_complete:
                                incomplete_reason = (
                                    "incomplete_content:"
                                    f"{doc.content_kind}:{doc.extraction_status}"
                                )
                            if incomplete_reason or not report.is_usable:
                                reason = (
                                    report.quarantine_reason
                                    if not report.is_usable
                                    else incomplete_reason
                                )
                                source_stats.quarantined[reason] += 1
                                quarantine_fp.write(
                                    json.dumps(
                                        {
                                            "doc_id": doc_id,
                                            "source": source,
                                            "document_id": doc.document_id,
                                            "reason": reason,
                                            "title": doc.title,
                                            "source_run": run.run_id,
                                            "content_kind": doc.content_kind,
                                            "content_complete": doc.content_complete,
                                            "extraction_status": doc.extraction_status,
                                            "source_binary_url": doc.source_binary_url,
                                            "damage": {
                                                "length": report.length,
                                                "meaningful": report.meaningful_chars,
                                                "nul": report.nul_chars,
                                                "control": report.control_chars,
                                                "replacement": report.replacement_chars,
                                            },
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                continue

                            content_hash = dedup.content_hash(clean_body)
                            structure_info = structure.detect(clean_body)
                            source_stats.clean += 1
                            source_stats.char_lens.append(len(clean_body))
                            source_stats.kinds[structure_info.primary_kind] += 1
                            for marker, present in (
                                ("article", structure_info.has_article),
                                ("heading", structure_info.has_heading),
                                ("num_clause", structure_info.has_num_clause),
                                ("chapter", structure_info.has_chapter),
                            ):
                                if present:
                                    source_stats.structure[marker] += 1
                            exact_map.append((doc_id, content_hash))
                            if doc.registration_code:
                                amend_map.append((doc_id, doc.registration_code))
                            docs_fp.write(
                                json.dumps(
                                    _snapshot_record(
                                        source,
                                        doc,
                                        clean_body,
                                        content_hash,
                                        structure_info,
                                        run.run_id,
                                        snapshot_id=snapshot_id,
                                    ),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    if stop:
                        break

            exact = dedup.cluster_by_key(exact_map, "exact")
            amendments = dedup.cluster_by_key(amend_map, "amendment")
            source_stats.exact = dedup.cluster_stats(exact)
            source_stats.amendment = dedup.cluster_stats(amendments)
            source_stats.near = (
                _near_dup_for_source(out_dir, source)
                if near_dup
                else {"skipped": True}
            )
            source_stats.token_pctl = (
                _token_pctls(out_dir, source, token_counter, token_sample)
                if token_counter is not None
                else None
            )
            stats[source] = source_stats
            print(
                f"  {source}: {source_stats.clean} clean, "
                f"{sum(source_stats.quarantined.values())} quarantined, "
                f"{source_stats.malformed} malformed "
                f"(exact-dups={source_stats.exact['clusters']}, "
                f"amend={source_stats.amendment['clusters']})"
            )
    return stats


def _token_pctls(out_dir: Path, source: str, counter, sample: int) -> dict | None:
    lengths: list[int] = []
    path = out_dir / "docs" / f"{source}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= sample:
                break
            record = json.loads(line)
            lengths.append(counter(record["body_markdown"]))
    lengths.sort()
    if not lengths:
        return None

    def percentile(percent: int) -> int:
        return lengths[min(len(lengths) - 1, int(percent / 100 * len(lengths)))]

    return {
        "sample": len(lengths),
        "p50": percentile(50),
        "p90": percentile(90),
        "p99": percentile(99),
        "max": lengths[-1],
    }


def _near_dup_for_source(out_dir: Path, source: str) -> dict:
    def rows():
        path = out_dir / "docs" / f"{source}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield record["doc_id"], record["body_markdown"]

    result = dedup.near_dup_clusters(rows())
    if result.skipped:
        return {"skipped": True, "reason": result.reason}
    return {"skipped": False, **dedup.cluster_stats(result.clusters)}


def _pctl(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(percentile / 100 * len(ordered)))]


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _write_reports(
    out_dir: Path,
    *,
    snapshot_id: str,
    created_at: str,
    config_digest: str,
    tokenizer_model: str,
    stats: Mapping[str, SourceStats],
) -> None:
    profile = [
        f"# Corpus profile — snapshot {snapshot_id}",
        "",
        f"Config hash: `{config_digest}` · created {created_at}",
        "",
        "| source | clean | quarantined | malformed | NUL | ctrl | mojibake | "
        "non-NFC | char p50/p90/p99/max |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--|",
    ]
    for source in SOURCES_PRESENT:
        source_stats = stats[source]
        profile.append(
            f"| {source} | {source_stats.clean} | "
            f"{sum(source_stats.quarantined.values())} | {source_stats.malformed} | "
            f"{source_stats.nul_docs} | {source_stats.control_docs} | "
            f"{source_stats.mojibake_docs} | {source_stats.nfc_changed} | "
            f"{_pctl(source_stats.char_lens, 50)}/"
            f"{_pctl(source_stats.char_lens, 90)}/"
            f"{_pctl(source_stats.char_lens, 99)}/"
            f"{max(source_stats.char_lens) if source_stats.char_lens else 0} |"
        )
    profile += ["", "## Structure coverage (% of clean docs) & primary kind", ""]
    for source in SOURCES_PRESENT:
        source_stats = stats[source]
        coverage = ", ".join(
            f"{key} {100 * count / max(1, source_stats.clean):.0f}%"
            for key, count in source_stats.structure.most_common()
        )
        kinds = ", ".join(
            f"{key}={count}" for key, count in source_stats.kinds.most_common()
        )
        profile.append(f"- **{source}**: {coverage or '—'} · kinds: {kinds}")
        if source_stats.token_pctl:
            token = source_stats.token_pctl
            profile.append(
                f"    - {tokenizer_model} tokens (sample {token['sample']}): "
                f"p50={token['p50']} p90={token['p90']} p99={token['p99']} "
                f"max={token['max']}"
            )
    profile += ["", "## PII field presence (counts only — values never exported)", ""]
    for source in SOURCES_PRESENT:
        source_stats = stats[source]
        if source_stats.pii_fields:
            profile.append(
                f"- **{source}**: "
                + ", ".join(
                    f"{key}={count}"
                    for key, count in source_stats.pii_fields.most_common()
                )
            )
    _write_text(out_dir / "reports" / "profile.md", "\n".join(profile) + "\n")

    dedup_report = [
        f"# Dedup report — snapshot {snapshot_id}",
        "",
        "Documents are reported, never deleted.",
        "",
        "| source | exact clusters (docs) | amendment clusters (docs, suspect) | "
        "near-dup |",
        "|---|--|--|--|",
    ]
    for source in SOURCES_PRESENT:
        source_stats = stats[source]
        near = (
            "skipped"
            if source_stats.near.get("skipped")
            else f"{source_stats.near['clusters']} clusters "
            f"({source_stats.near['docs_in_clusters']} docs)"
        )
        dedup_report.append(
            f"| {source} | {source_stats.exact['clusters']} "
            f"({source_stats.exact['docs_in_clusters']}) | "
            f"{source_stats.amendment['clusters']} "
            f"({source_stats.amendment['docs_in_clusters']}, "
            f"{source_stats.amendment['suspect']} suspect) | {near} |"
        )
    _write_text(out_dir / "reports" / "dedup.md", "\n".join(dedup_report) + "\n")

    reasons = sorted(
        {reason for source_stats in stats.values() for reason in source_stats.quarantined}
    )
    columns = reasons or ["none"]
    quarantine_report = [
        f"# Quarantine report — snapshot {snapshot_id}",
        "",
        "Quarantined documents are excluded from clean docs and retained in "
        "`quarantine.jsonl`.",
        "",
        "| source | " + " | ".join(columns) + " |",
        "|---|" + "|".join("--" for _ in columns) + "|",
    ]
    for source in SOURCES_PRESENT:
        source_stats = stats[source]
        quarantine_report.append(
            f"| {source} | "
            + " | ".join(str(source_stats.quarantined.get(reason, 0)) for reason in columns)
            + " |"
        )
    _write_text(
        out_dir / "reports" / "quarantine.md",
        "\n".join(quarantine_report) + "\n",
    )


def _file_inventory(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SnapshotSafetyError(f"refusing symlink in staged snapshot: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise SnapshotSafetyError(f"refusing non-regular snapshot file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        digest, size = _hash_regular_file(path)
        entries.append({"path": relative, "sha256": digest, "size_bytes": size})
    return entries


def inventory_sha256(entries: Sequence[Mapping[str, object]]) -> str:
    """Hash an ordered file-inventory subset using the snapshot canonical encoding."""
    return _sha256_bytes(_canonical_json_bytes(list(entries)))


def snapshot_manifest_sha256(manifest: Mapping[str, object]) -> str:
    """Hash every manifest field except the self-referential aggregate digest."""
    material = dict(manifest)
    material.pop("snapshot_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def _fsync_tree(root: Path) -> None:
    files: list[Path] = []
    directories: list[Path] = [root]
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SnapshotSafetyError(f"refusing symlink while sealing snapshot: {path}")
        if stat.S_ISREG(info.st_mode):
            files.append(path)
        elif stat.S_ISDIR(info.st_mode):
            directories.append(path)
        else:
            raise SnapshotSafetyError(f"refusing special file while sealing snapshot: {path}")
    for path in files:
        path.chmod(0o600)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o700)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing an existing name."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SnapshotSafetyError("atomic renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise SnapshotSafetyError(f"snapshot destination already exists: {destination}")
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise SnapshotSafetyError("atomic no-replace rename is unsupported")
    raise OSError(error, os.strerror(error), str(destination))


def _assert_inputs_unchanged(
    runs: Sequence[_RunInput],
    evidence_path: Path | None,
    evidence_attestation: Mapping[str, object] | None,
) -> None:
    for run in runs:
        digest, size = _hash_regular_file(run.items_path)
        if (digest, size) != (run.items_sha256, run.items_size_bytes):
            raise SnapshotSafetyError(
                f"selected items.jsonl changed during build: {run.source}/{run.run_id}"
            )
        if run.completion_path is not None:
            digest, size = _hash_regular_file(run.completion_path)
            if (digest, size) != (run.completion_sha256, run.completion_size_bytes):
                raise SnapshotSafetyError(
                    f"completion record changed during build: {run.source}/{run.run_id}"
                )
    if evidence_path is not None and evidence_attestation is not None:
        digest, size = _hash_regular_file(evidence_path)
        if (digest, size) != (
            evidence_attestation["sha256"],
            evidence_attestation["size_bytes"],
        ):
            raise SnapshotSafetyError("source-state evidence changed during build")


def _validate_inventory_entry(value: object, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SnapshotSafetyError(f"files[{index}] must be an object")
    _exact_keys(value, {"path", "sha256", "size_bytes"}, field=f"files[{index}]")
    path = value["path"]
    digest = value["sha256"]
    size = value["size_bytes"]
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise SnapshotSafetyError(f"files[{index}].path is invalid")
    parsed = Path(path)
    if parsed.as_posix() != path or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SnapshotSafetyError(f"files[{index}].path is not canonical")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise SnapshotSafetyError(f"files[{index}].sha256 is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise SnapshotSafetyError(f"files[{index}].size_bytes is invalid")
    return value


def _verify_sealed_snapshot_at(
    snapshot_root: Path,
    *,
    allow_preflight: bool,
    require_all_sources: bool,
    expected_snapshot_id: str,
    validate_snapshot_id: bool = True,
) -> dict[str, object]:
    _reject_symlink_components(snapshot_root)
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise SnapshotSafetyError(f"snapshot root is not a real directory: {snapshot_root}")
    manifest_path = snapshot_root / "manifest.json"
    manifest = _read_json_object(manifest_path, max_bytes=16 * 1024 * 1024)
    _exact_keys(manifest, _MANIFEST_KEYS, field="snapshot manifest")
    if manifest["schema_version"] != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise SnapshotSafetyError("unsupported snapshot manifest schema_version")
    if manifest["pipeline_version"] != SNAPSHOT_PIPELINE_VERSION:
        raise SnapshotSafetyError("unsupported snapshot pipeline_version")
    if manifest["snapshot_id"] != expected_snapshot_id:
        raise SnapshotSafetyError("snapshot_id does not match its immutable directory")
    if validate_snapshot_id:
        _validate_snapshot_id(expected_snapshot_id)
    preflight = manifest["preflight"]
    if not isinstance(preflight, bool):
        raise SnapshotSafetyError("manifest.preflight must be boolean")
    if preflight and not allow_preflight:
        raise SnapshotSafetyError("preflight snapshots are not valid production inputs")
    if not isinstance(manifest["config_hash"], str) or not re.fullmatch(
        r"[0-9a-f]{16}", manifest["config_hash"]
    ):
        raise SnapshotSafetyError("manifest.config_hash is invalid")
    try:
        parse_utc_datetime(manifest["created_at"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SnapshotSafetyError("manifest.created_at is invalid") from exc

    build = manifest["build"]
    if not isinstance(build, dict):
        raise SnapshotSafetyError("manifest.build must be an object")
    _exact_keys(
        build,
        {
            "sources",
            "limit",
            "near_dup",
            "token_sample",
            "tokenizer",
            "chunk",
            "embed_model",
            "embedding_revision",
        },
        field="manifest.build",
    )
    if require_all_sources and build["sources"] != list(SOURCES_PRESENT):
        raise SnapshotSafetyError("manifest.build.sources must contain all seven sources")
    if not preflight and build["limit"] is not None:
        raise SnapshotSafetyError("production snapshot manifest cannot contain a limit")
    token_sample = build["token_sample"]
    tokenizer = build["tokenizer"]
    if not isinstance(token_sample, int) or isinstance(token_sample, bool) or token_sample < 0:
        raise SnapshotSafetyError("manifest.build.token_sample is invalid")
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"model", "revision"}:
        raise SnapshotSafetyError("manifest.build.tokenizer is invalid")
    tokenizer_revision = tokenizer["revision"]
    if tokenizer_revision is not None and (
        not isinstance(tokenizer_revision, str)
        or not _IMMUTABLE_REVISION_RE.fullmatch(tokenizer_revision)
    ):
        raise SnapshotSafetyError("snapshot tokenizer revision is not immutable")
    if token_sample > 0 and (
        not isinstance(tokenizer_revision, str)
        or not _IMMUTABLE_REVISION_RE.fullmatch(tokenizer_revision)
    ):
        raise SnapshotSafetyError("sampled snapshot lacks an immutable tokenizer revision")

    source_stats = manifest["sources"]
    if not isinstance(source_stats, dict):
        raise SnapshotSafetyError("manifest.sources must be an object")
    if require_all_sources and set(source_stats) != set(SOURCES_PRESENT):
        raise SnapshotSafetyError("manifest.sources must contain all seven sources")
    runs = manifest["runs"]
    if not isinstance(runs, list):
        raise SnapshotSafetyError("manifest.runs must be a list")
    run_keys: list[tuple[str, str]] = []
    runs_by_source: dict[str, list[str]] = {source: [] for source in SOURCES_PRESENT}
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise SnapshotSafetyError(f"manifest.runs[{index}] must be an object")
        _exact_keys(
            run,
            {
                "source",
                "run_id",
                "items",
                "completion_record",
                "completed_at",
                "success_verified",
            },
            field=f"manifest.runs[{index}]",
        )
        source = run["source"]
        run_id = run["run_id"]
        if source not in SOURCES_PRESENT or not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(
            run_id
        ):
            raise SnapshotSafetyError(f"manifest.runs[{index}] identity is invalid")
        key = (source, run_id)
        if key in run_keys:
            raise SnapshotSafetyError(f"duplicate run inventory entry: {source}/{run_id}")
        run_keys.append(key)
        runs_by_source[source].append(run_id)
        items = run["items"]
        if not isinstance(items, dict):
            raise SnapshotSafetyError(f"manifest.runs[{index}].items is invalid")
        _exact_keys(items, {"path", "sha256", "size_bytes"}, field="run items")
        if items["path"] != f"{source}/runs/{run_id}/items.jsonl":
            raise SnapshotSafetyError(f"manifest.runs[{index}].items.path is invalid")
        if not isinstance(items["sha256"], str) or not _SHA256_RE.fullmatch(
            items["sha256"]
        ):
            raise SnapshotSafetyError(f"manifest.runs[{index}].items.sha256 is invalid")
        if not isinstance(items["size_bytes"], int) or isinstance(
            items["size_bytes"], bool
        ) or items["size_bytes"] < 0:
            raise SnapshotSafetyError(f"manifest.runs[{index}].items.size_bytes is invalid")
        if preflight:
            if run["success_verified"] is not False:
                raise SnapshotSafetyError("preflight run cannot claim success verification")
            if run["completion_record"] is not None or run["completed_at"] is not None:
                raise SnapshotSafetyError("preflight run cannot carry completion attestation")
        else:
            if run["success_verified"] is not True:
                raise SnapshotSafetyError("production run lacks verified success")
            completion = run["completion_record"]
            if not isinstance(completion, dict):
                raise SnapshotSafetyError("production run lacks completion record binding")
            _exact_keys(
                completion,
                {"path", "sha256", "size_bytes"},
                field="run completion_record",
            )
            if completion["path"] != f"{source}/runs/{run_id}/run.json":
                raise SnapshotSafetyError("completion record path is invalid")
            if not isinstance(completion["sha256"], str) or not _SHA256_RE.fullmatch(
                completion["sha256"]
            ):
                raise SnapshotSafetyError("completion record hash is invalid")
            if not isinstance(completion["size_bytes"], int) or isinstance(
                completion["size_bytes"], bool
            ) or completion["size_bytes"] < 0:
                raise SnapshotSafetyError("completion record size is invalid")
            try:
                parse_utc_datetime(run["completed_at"])  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise SnapshotSafetyError("completion timestamp is invalid") from exc
    if run_keys != sorted(run_keys):
        raise SnapshotSafetyError("manifest.runs must be sorted by source and run_id")
    missing_run_sources = [
        source for source in SOURCES_PRESENT if not runs_by_source[source]
    ]
    if not preflight and missing_run_sources:
        raise SnapshotSafetyError(
            "production run inventory must contain every source; "
            f"missing={missing_run_sources}"
        )
    for source, run_ids in runs_by_source.items():
        stats = source_stats.get(source)
        if not isinstance(stats, dict) or stats.get("run_ids") != sorted(run_ids):
            raise SnapshotSafetyError(f"source run inventory mismatch for {source}")

    evidence = manifest["source_state_evidence"]
    if preflight:
        if evidence is not None:
            raise SnapshotSafetyError("preflight manifest cannot claim production evidence")
    else:
        if not isinstance(evidence, dict):
            raise SnapshotSafetyError("production manifest lacks source-state evidence")
        _exact_keys(evidence, {"sha256", "size_bytes"}, field="source_state_evidence")
        if not isinstance(evidence["sha256"], str) or not _SHA256_RE.fullmatch(
            evidence["sha256"]
        ):
            raise SnapshotSafetyError("source-state evidence hash is invalid")
        if not isinstance(evidence["size_bytes"], int) or isinstance(
            evidence["size_bytes"], bool
        ) or evidence["size_bytes"] < 0:
            raise SnapshotSafetyError("source-state evidence size is invalid")

    files_raw = manifest["files"]
    if not isinstance(files_raw, list):
        raise SnapshotSafetyError("manifest.files must be a list")
    files = [_validate_inventory_entry(value, index) for index, value in enumerate(files_raw)]
    paths = [entry["path"] for entry in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SnapshotSafetyError("manifest.files must have unique sorted paths")
    required = {"quarantine.jsonl", "reports/profile.md", "reports/dedup.md", "reports/quarantine.md"}
    required.update(f"docs/{source}.jsonl" for source in SOURCES_PRESENT)
    if require_all_sources and not required.issubset(set(paths)):
        raise SnapshotSafetyError("snapshot file inventory is missing required corpus files")
    actual = _file_inventory(snapshot_root)
    if actual != files:
        raise SnapshotSafetyError("snapshot file inventory does not match exact on-disk bytes")
    docs_entries = [entry for entry in files if str(entry["path"]).startswith("docs/")]
    if manifest["corpus_sha256"] != inventory_sha256(docs_entries):
        raise SnapshotSafetyError("snapshot corpus_sha256 mismatch")
    expected_snapshot_sha = snapshot_manifest_sha256(manifest)
    if manifest["snapshot_sha256"] != expected_snapshot_sha:
        raise SnapshotSafetyError("snapshot snapshot_sha256 mismatch")
    return manifest


def verify_sealed_snapshot(
    snapshot_root: Path | str,
    *,
    allow_preflight: bool = False,
    require_all_sources: bool = True,
) -> dict[str, object]:
    """Verify an immutable snapshot and return its parsed sealed manifest.

    Production consumers should keep the default ``allow_preflight=False``.  The
    verifier checks the complete exact file set, every recorded size/hash, both
    aggregate digests, source coverage, run bindings, and manifest self-consistency.
    """
    root = Path(snapshot_root).expanduser().absolute()
    return _verify_sealed_snapshot_at(
        root,
        allow_preflight=allow_preflight,
        require_all_sources=require_all_sources,
        expected_snapshot_id=root.name,
    )


def build_snapshot(
    cfg: Config,
    *,
    snapshot_id: str,
    output_root: Path,
    source_state_evidence: Path | None = None,
    preflight: bool = False,
    sources: Sequence[str] = SOURCES_PRESENT,
    limit: int | None = None,
    near_dup: bool = True,
    token_sample: int = 2000,
) -> dict[str, object]:
    """Build, seal, and immutably publish one snapshot."""
    output_root = _validate_build_request(
        cfg,
        snapshot_id=snapshot_id,
        output_root=output_root,
        sources=sources,
        source_state_evidence=source_state_evidence,
        preflight=preflight,
        limit=limit,
        token_sample=token_sample,
    )
    evidence_path = (
        source_state_evidence.expanduser().absolute()
        if source_state_evidence is not None
        else None
    )
    if preflight:
        if evidence_path is not None:
            raise SnapshotSafetyError(
                "--source-state-evidence is reserved for production snapshots"
            )
        runs = _discover_preflight_inputs(cfg)
        evidence_attestation = None
    else:
        assert evidence_path is not None
        runs, evidence_attestation = _load_source_state_evidence(cfg, evidence_path)

    # A zero sample is intentionally fully offline: do not import the embedding module.
    token_counter = _make_token_counter(cfg, token_sample)
    output_root = _create_output_root(output_root)
    destination = output_root / snapshot_id
    if os.path.lexists(destination):
        raise SnapshotSafetyError(f"snapshot destination already exists: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=output_root))
    published = False
    try:
        created_at = _iso_utc(datetime.now(timezone.utc))
        stats = _build_corpus(
            snapshot_id=snapshot_id,
            out_dir=stage,
            runs=runs,
            limit=limit,
            near_dup=near_dup,
            token_sample=token_sample,
            token_counter=token_counter,
        )
        config_digest = config_hash(cfg)
        _write_reports(
            stage,
            snapshot_id=snapshot_id,
            created_at=created_at,
            config_digest=config_digest,
            tokenizer_model=cfg.tokenizer_model,
            stats=stats,
        )
        _assert_inputs_unchanged(runs, evidence_path, evidence_attestation)
        files = _file_inventory(stage)
        docs_entries = [entry for entry in files if str(entry["path"]).startswith("docs/")]
        total_clean = sum(source_stats.clean for source_stats in stats.values())
        total_quarantined = sum(
            sum(source_stats.quarantined.values()) for source_stats in stats.values()
        )
        total_malformed = sum(source_stats.malformed for source_stats in stats.values())
        run_ids_by_source = {
            source: sorted(run.run_id for run in runs if run.source == source)
            for source in SOURCES_PRESENT
        }
        manifest: dict[str, object] = {
            "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "pipeline_version": SNAPSHOT_PIPELINE_VERSION,
            "preflight": preflight,
            "created_at": created_at,
            "config_hash": config_digest,
            "build": {
                "sources": list(SOURCES_PRESENT),
                "limit": limit,
                "near_dup": near_dup,
                "token_sample": token_sample,
                "tokenizer": {
                    "model": cfg.tokenizer_model,
                    # Record the configured immutable identity even when sampling is
                    # disabled.  ``token_sample=0`` remains fully offline; a later embed
                    # can still prove the snapshot was sealed under the same tokenizer.
                    "revision": cfg.tokenizer_revision,
                },
                "chunk": {
                    "tokens": cfg.chunk_tokens,
                    "overlap": cfg.chunk_overlap,
                    "min_tokens": cfg.chunk_min_tokens,
                },
                "embed_model": cfg.embed_model,
                "embedding_revision": cfg.embedding_revision,
            },
            "source_state_evidence": evidence_attestation,
            "runs": [run.to_manifest() for run in sorted(runs, key=lambda item: (item.source, item.run_id))],
            "totals": {
                "clean": total_clean,
                "quarantined": total_quarantined,
                "malformed": total_malformed,
            },
            "sources": {
                source: {
                    "run_ids": run_ids_by_source[source],
                    "raw_lines": stats[source].raw_lines,
                    "unique": stats[source].unique,
                    "clean": stats[source].clean,
                    "malformed": stats[source].malformed,
                    "quarantined": dict(stats[source].quarantined),
                    "damage": {
                        "nul": stats[source].nul_docs,
                        "control": stats[source].control_docs,
                        "mojibake": stats[source].mojibake_docs,
                        "nfc_changed": stats[source].nfc_changed,
                    },
                    "dedup": {
                        "exact": stats[source].exact,
                        "amendment": stats[source].amendment,
                        "near": stats[source].near,
                    },
                }
                for source in SOURCES_PRESENT
            },
            "files": files,
            "corpus_sha256": inventory_sha256(docs_entries),
        }
        manifest["snapshot_sha256"] = snapshot_manifest_sha256(manifest)
        _write_manifest(stage / "manifest.json", manifest)
        _fsync_tree(stage)
        _verify_sealed_snapshot_at(
            stage,
            allow_preflight=True,
            require_all_sources=True,
            expected_snapshot_id=snapshot_id,
            validate_snapshot_id=False,
        )
        _rename_noreplace(stage, destination)
        published = True
        descriptor = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if not published and os.path.lexists(stage):
            shutil.rmtree(stage)

    print(f"\nSnapshot {snapshot_id} written to {destination}")
    print(
        f"  clean={manifest['totals']['clean']} "  # type: ignore[index]
        f"quarantined={manifest['totals']['quarantined']} "  # type: ignore[index]
        f"malformed={manifest['totals']['malformed']} "  # type: ignore[index]
        f"snapshot_sha256={manifest['snapshot_sha256']}"
    )
    return manifest


__all__ = [
    "DEFAULT_SNAPSHOT_ROOT",
    "FROZEN_V1_ROOT",
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "SNAPSHOT_PIPELINE_VERSION",
    "SOURCE_STATE_EVIDENCE_SCHEMA_VERSION",
    "SOURCES_PRESENT",
    "SnapshotSafetyError",
    "build_snapshot",
    "config_hash",
    "inventory_sha256",
    "snapshot_manifest_sha256",
    "verify_sealed_snapshot",
]
