"""Strict offline validation for the immutable 512-token candidate input bundle.

The release bundle is external, read-only evidence.  This module never downloads a
model, talks to Qdrant, or mutates the bundle.  It validates the exact frozen identity,
rehashes every declared file without following symlinks, and proves that the supplied
v2 baseline contains two pairable, deterministic 337-query traces before a crawler or
paid worker can be started.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .model_policy import LicenseAttestation, ModelRole

SNAPSHOT_ID = "v3_512_attested_20260715_01"
GENERATION_ID = "v3_512_candidate_20260715_01"
PHYSICAL_COLLECTION = "georgian_legal__gen_v3_512_candidate_20260715_01"
CRAWL_START_DATE = "1900-01-01"
CRAWL_END_DATE = "2026-07-15"
SOURCES = (
    "matsne",
    "ecd",
    "constcourt",
    "napr",
    "tbappeal",
    "supremecourt",
    "tas",
)
GOLDEN_SET_SHA256 = "753e2985315be3e408c3db3303f90625d9c66984b5fc9519cf7e32a8d46252c6"
TRANSLATION_SHA256 = "0884870a8fa68527c959de3781c4515458d96784c4f599158427f058ce727fad"
HOLDOUT_SHA256 = "eaee96072f66a0f3f63d6d1cbe61e4566d3b405daedf389211bb351d05b7ad3e"
EXPECTED_QUERY_COUNT = 337
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
RELEASE_BUNDLE_PATH = "/secure/release-inputs/v3_512_candidate_20260715_01"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANONICAL_POSITIVE_INTEGER_RE = re.compile(r"^[1-9][0-9]*$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+\-]{0,255}$")
_PLACEHOLDER_RE = re.compile(
    r"(?:replace[_ -]?me|placeholder|\btodo\b|\btbd\b|changeme|unconfigured|"
    r"example\.(?:com|test)|<{2,}|>{2,})",
    re.IGNORECASE,
)
_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:api_?key|secret|password|passwd|credential|access_?token|"
    r"refresh_?token|private_?key)(?:$|_)",
    re.IGNORECASE,
)
_SECRET_VALUE_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
)

_TOP_KEYS = {
    "schema_version",
    "release",
    "code",
    "models",
    "dependencies",
    "runtime",
    "retrieval",
    "operator",
    "baseline",
}
_RELEASE_KEYS = {
    "snapshot_id",
    "generation_id",
    "physical_collection",
    "crawl_start_date",
    "crawl_end_date",
    "sources",
}
_CODE_KEYS = {
    "repository_revision",
    "code_identity_sha256",
    "retriever_revision",
    "tokenizer_revision",
    "reranker_revision",
}
_MODEL_KEYS = {"embedding", "tokenizer", "reranker"}
_MODEL_ENTRY_KEYS = {
    "name",
    "revision",
    "artifact_path",
    "artifact_sha256",
    "artifact_size_bytes",
    "license_path",
    "license_sha256",
    "license_size_bytes",
}
_DEPENDENCY_KEYS = {"lock_path", "lock_sha256", "lock_size_bytes", "files"}
_DEPENDENCY_FILE_KEYS = {"name", "path", "sha256", "size_bytes"}
_RUNTIME_KEYS = {
    "identity_path",
    "identity_sha256",
    "identity_size_bytes",
    "runtime_revision",
    "qdrant_revision",
    "oci_reference",
    "oci_digest",
    "environment",
}
_ENVIRONMENT_KEYS = {
    "SNAPSHOT_ID",
    "GENERATION_ID",
    "COLLECTION_NAME",
    "EMBED_MODEL",
    "TOKENIZER_MODEL",
    "RERANK_MODEL",
    "EMBED_REVISION",
    "TOKENIZER_REVISION",
    "RERANK_REVISION",
    "DENSE_DIM",
    "CHUNK_TOKENS",
    "CHUNK_OVERLAP",
    "CHUNK_MIN_TOKENS",
    "RERANK_ENABLED",
    "RERANK_CANDIDATES",
    "RERANK_MIN_SCORE",
    "RERANK_BACKEND",
    "RERANK_CONTEXT_ENRICHED",
    "RERANK_MAX_LENGTH",
    "CITATION_ROUTE",
    "EMBED_HEADER_V2",
    "EMBED_DEVICE",
    "EMBED_USE_FP16",
    "EMBED_BATCH_SIZE",
    "PRODUCTION_MODE",
}
_RETRIEVAL_KEYS = {"configuration_hash", "knobs"}
_RETRIEVAL_KNOB_KEYS = {
    "dense_dimension",
    "chunk_tokens",
    "chunk_overlap",
    "chunk_min_tokens",
    "rerank_enabled",
    "rerank_candidates",
    "rerank_min_score",
    "rerank_backend",
    "rerank_context_enriched",
    "rerank_max_length",
    "citation_route",
    "document_header",
}
_OPERATOR_KEYS = {"actor", "run_id"}
_BASELINE_BINDING_KEYS = {
    "root",
    "manifest_path",
    "manifest_sha256",
    "manifest_size_bytes",
}

_BASELINE_KEYS = {
    "schema_version",
    "dataset",
    "collection",
    "models",
    "configuration_hash",
    "repeats",
}
_BASELINE_DATASET_KEYS = {
    "name",
    "query_count",
    "golden_set_sha256",
    "translation_sha256",
    "holdout_sha256",
}
_BASELINE_COLLECTION_KEYS = {
    "physical_collection",
    "generation_id",
    "snapshot_sha256",
    "collection_sha256",
    "configuration_sha256",
}
_BASELINE_MODEL_ENTRY_KEYS = {"name", "revision", "artifact_sha256"}
_REPEAT_KEYS = {
    "repeat",
    "track",
    "trace_path",
    "trace_sha256",
    "trace_size_bytes",
    "ranking_hash",
    "decision_result_hash",
}
_TRACE_KEYS = {
    "query_id",
    "cluster_id",
    "status",
    "configuration_hash",
    "raw_candidates",
    "final_ranking",
    "branch_provenance",
    "entity_matches",
    "document_matches",
    "route_decision",
    "degraded",
    "timings_ms",
    "score",
}
_RANKED_KEYS = {"point_id", "score"}
_BASELINE_METRIC_KEYS = {
    "success1",
    "success5",
    "success10",
    "required_evidence_recall10",
    "candidate_recall50",
    "candidate_recall80",
    "document_identity1",
    "passage_accuracy1",
    "context_duplication10",
    "context_noise10",
    "ndcg10",
    "mrr10",
}
_BASELINE_SCORE_KEYS = _BASELINE_METRIC_KEYS | {
    "id",
    "query_type",
    "language",
    "cluster_id",
    "failed",
    "failure_reason",
    "source",
    "tags",
    "risk_level",
    "expected_outcome",
}
_FIXED_ZERO_SOURCES = {"tas": 25, "tbappeal": 31}
_FAILURE_STATUSES = {
    "zero_score_failure",
    "exception",
    "skipped",
    "unchunkable",
    "abstention",
}

_EXPECTED_RETRIEVAL_KNOBS: dict[str, object] = {
    "dense_dimension": 1024,
    "chunk_tokens": 512,
    "chunk_overlap": 80,
    "chunk_min_tokens": 64,
    "rerank_enabled": True,
    "rerank_candidates": 80,
    "rerank_min_score": 0.3,
    "rerank_backend": "torch",
    "rerank_context_enriched": True,
    "rerank_max_length": 1024,
    "citation_route": "ids",
    "document_header": True,
}
_BRANCH_PROVENANCE_KEYS = {"route", "branches"}
_BRANCH_TRACE_KEYS = {"name", "point_ids"}
_ENTITY_MATCH_KEYS = {"query_entities", "matched_point_ids"}
_DOCUMENT_MATCH_KEYS = {"expected_document_id", "matched_point_ids"}
_ROUTE_DECISION_KEYS = {"original", "translated", "selected"}


class ReleaseInputError(ValueError):
    """The supplied external evidence cannot authorize candidate execution."""


@dataclass(frozen=True, slots=True)
class ValidatedBaseline:
    root: Path
    manifest_sha256: str
    query_ids: tuple[str, ...]
    ranking_hash: str
    decision_result_hash: str
    collection_sha256: str
    configuration_hash: str
    ordered_query_ids: tuple[str, ...]
    queries: tuple[dict[str, object], ...]
    ranking_hashes: tuple[str, str]
    decision_result_hashes: tuple[str, str]
    repeats: tuple[dict[str, str], dict[str, str]]


@dataclass(frozen=True, slots=True)
class ValidatedReleaseInputs:
    root: Path
    manifest_sha256: str
    repository_revision: str
    code_identity_sha256: str
    configuration_hash: str
    oci_digest: str
    actor: str
    run_id: str
    baseline: ValidatedBaseline

    def report(self) -> dict[str, object]:
        """Return a secret-free validation summary suitable for a release report."""

        return {
            "schema_version": SCHEMA_VERSION,
            "status": "valid",
            "release": {
                "snapshot_id": SNAPSHOT_ID,
                "generation_id": GENERATION_ID,
                "physical_collection": PHYSICAL_COLLECTION,
            },
            "manifest_sha256": self.manifest_sha256,
            "repository_revision": self.repository_revision,
            "code_identity_sha256": self.code_identity_sha256,
            "configuration_hash": self.configuration_hash,
            "oci_digest": self.oci_digest,
            "actor": self.actor,
            "run_id": self.run_id,
            "baseline": {
                "manifest_sha256": self.baseline.manifest_sha256,
                "query_count": len(self.baseline.query_ids),
                "ranking_hash": self.baseline.ranking_hash,
                "decision_result_hash": self.baseline.decision_result_hash,
                "collection_sha256": self.baseline.collection_sha256,
                "configuration_hash": self.baseline.configuration_hash,
            },
        }


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ReleaseInputError(
            f"{field} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseInputError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReleaseInputError(f"{field} must be an array")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReleaseInputError(f"{field} must be a lowercase 64-hex SHA-256")
    return value


def _revision(value: object, field: str) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise ReleaseInputError(f"{field} must be an immutable lowercase hex revision")
    return value


def _identity(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ReleaseInputError(f"{field} is missing or unsafe")
    if _PLACEHOLDER_RE.search(value) or set(value) == {"0"}:
        raise ReleaseInputError(f"{field} contains a placeholder")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReleaseInputError(f"{field} must be a non-negative integer")
    return value


def _strict_json_bytes(raw: bytes, field: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ReleaseInputError(f"{field} contains duplicate key {key!r}")
            out[key] = value
        return out

    def no_constants(value: str) -> None:
        raise ReleaseInputError(f"{field} contains non-finite number {value}")

    try:
        return json.loads(
            raw, object_pairs_hook=no_duplicates, parse_constant=no_constants
        )
    except ReleaseInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseInputError(f"{field} is not strict UTF-8 JSON: {exc}") from exc


def _walk_reportable(
    value: object, path: str = "manifest"
) -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = f"{path}.{key}"
            if _SECRET_KEY_RE.search(str(key)):
                raise ReleaseInputError(f"secret-bearing field is forbidden: {field}")
            yield from _walk_reportable(child, field)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk_reportable(child, f"{path}[{index}]")
    else:
        yield path, value


def _reject_placeholders_and_secrets(value: object) -> None:
    for field, child in _walk_reportable(value):
        if not isinstance(child, str):
            continue
        if _PLACEHOLDER_RE.search(child):
            raise ReleaseInputError(f"placeholder value is forbidden at {field}")
        if any(pattern.search(child) for pattern in _SECRET_VALUE_RES):
            raise ReleaseInputError(f"probable secret value is forbidden at {field}")


def _safe_root(root: Path) -> Path:
    absolute = root.expanduser().absolute()
    try:
        info = absolute.lstat()
    except FileNotFoundError as exc:
        raise ReleaseInputError(f"release bundle is absent: {absolute}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ReleaseInputError("release bundle root must be a real directory")
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseInputError(f"release bundle path contains symlink: {cursor}")
    return absolute


def _current_repository_revision(repo_root: Path) -> str:
    """Resolve the checked-out Git object without invoking hooks or the network."""

    git_dir = repo_root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseInputError(f"cannot read repository HEAD: {exc}") from exc
    if _REVISION_RE.fullmatch(head):
        return head
    prefix = "ref: "
    if not head.startswith(prefix):
        raise ReleaseInputError(
            "repository HEAD is not an immutable revision or safe ref"
        )
    reference = head[len(prefix) :]
    if (
        not reference.startswith("refs/")
        or ".." in Path(reference).parts
        or "\\" in reference
    ):
        raise ReleaseInputError("repository HEAD reference is unsafe")
    loose = git_dir / Path(reference)
    try:
        revision = loose.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        revision = ""
        try:
            packed_lines = (
                (git_dir / "packed-refs").read_text(encoding="ascii").splitlines()
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise ReleaseInputError(
                f"cannot resolve repository HEAD reference: {exc}"
            ) from exc
        for line in packed_lines:
            if not line or line.startswith(("#", "^")):
                continue
            candidate, separator, name = line.partition(" ")
            if separator and name == reference:
                revision = candidate
                break
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseInputError(
            f"cannot resolve repository HEAD reference: {exc}"
        ) from exc
    if _REVISION_RE.fullmatch(revision) is None:
        raise ReleaseInputError(
            "repository HEAD reference is not an immutable revision"
        )
    return revision


def _relative_file(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReleaseInputError(f"{field} must be a non-empty POSIX relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ReleaseInputError(f"{field} must be a normalized relative path")
    path = root / relative
    cursor = root
    for component in relative.parts:
        cursor /= component
        try:
            info = cursor.lstat()
        except FileNotFoundError as exc:
            raise ReleaseInputError(
                f"declared release file is absent: {field}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseInputError(
                f"declared release path contains a symlink: {field}"
            )
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseInputError(f"declared release path is not a regular file: {field}")
    return path


def _read_bounded_file(path: Path, *, field: str, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ReleaseInputError(f"cannot inspect {field}: {exc}") from exc
    if size <= 0 or size > max_bytes:
        raise ReleaseInputError(f"{field} must contain 1 byte to {max_bytes} bytes")
    payload = bytearray()
    try:
        with path.open("rb") as handle:
            while block := handle.read(min(1024 * 1024, max_bytes + 1)):
                payload.extend(block)
                if len(payload) > max_bytes:
                    raise ReleaseInputError(f"{field} exceeds {max_bytes} bytes")
    except ReleaseInputError:
        raise
    except OSError as exc:
        raise ReleaseInputError(f"cannot read {field}: {exc}") from exc
    if not payload:
        raise ReleaseInputError(f"{field} must not be empty")
    return bytes(payload)


def _read_bound_file(
    root: Path,
    path_value: object,
    digest_value: object,
    size_value: object,
    field: str,
    *,
    max_bytes: int | None = None,
    capture: bool = True,
) -> tuple[Path, bytes | None, str]:
    """Stream-verify one declared file and optionally retain its bounded bytes."""

    path = _relative_file(root, path_value, f"{field}.path")
    expected = _sha256(digest_value, f"{field}.sha256")
    size = _nonnegative_int(size_value, f"{field}.size_bytes")
    if size == 0:
        raise ReleaseInputError(f"{field} must not be empty")
    if max_bytes is not None and size > max_bytes:
        raise ReleaseInputError(f"{field} exceeds the fixed {max_bytes}-byte limit")
    digest = hashlib.sha256()
    observed_size = 0
    captured = bytearray() if capture else None
    try:
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                observed_size += len(block)
                if observed_size > size:
                    raise ReleaseInputError(f"{field} size changed")
                digest.update(block)
                if captured is not None:
                    captured.extend(block)
    except ReleaseInputError:
        raise
    except OSError as exc:
        raise ReleaseInputError(f"cannot read {field}: {exc}") from exc
    if observed_size != size:
        raise ReleaseInputError(f"{field} size changed")
    observed = digest.hexdigest()
    if observed != expected:
        raise ReleaseInputError(f"{field} SHA-256 mismatch")
    return path, bytes(captured) if captured is not None else None, observed


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def code_identity_sha256(repo_root: Path | str) -> str:
    """Hash HEAD plus every tracked diff and untracked, non-ignored file byte identity.

    State/artifact roots are ignored by Git and therefore do not invalidate the identity
    when a create-only validation report or evidence-crawl run is written.  Source changes
    do invalidate it, including changes to already-untracked code files.
    """

    root = Path(repo_root).expanduser().absolute()

    def git(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReleaseInputError(
                f"cannot collect repository code identity with {' '.join(arguments)}"
            ) from exc

    repository_revision = _current_repository_revision(root)
    patch = git("diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    material = hashlib.sha256()
    has_dirty_material = bool(patch)
    material.update(patch)
    untracked = [
        value
        for value in git("ls-files", "--others", "--exclude-standard", "-z").split(
            b"\0"
        )
        if value
    ]
    for raw_name in sorted(untracked):
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseInputError("untracked repository path is not UTF-8") from exc
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReleaseInputError("untracked repository path is unsafe")
        path = root / relative
        try:
            info = path.lstat()
        except OSError as exc:
            raise ReleaseInputError(
                f"cannot inspect untracked repository file: {name}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ReleaseInputError(
                f"untracked repository input is not a regular file: {name}"
            )
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ReleaseInputError(
                f"cannot hash untracked repository file: {name}"
            ) from exc
        has_dirty_material = True
        material.update(b"\0untracked\0")
        material.update(raw_name)
        material.update(b"\0")
        material.update(bytes.fromhex(digest.hexdigest()))
    identity = {
        "repository_revision": repository_revision,
        "dirty_patch_sha256": material.hexdigest() if has_dirty_material else None,
    }
    return _canonical_hash(identity)


def _ranked_rows(
    value: object,
    field: str,
    *,
    minimum: int,
    exact_count: int | None = None,
) -> list[dict[str, object]]:
    rows = _sequence(value, field)
    if exact_count is not None and len(rows) != exact_count:
        raise ReleaseInputError(
            f"{field} must contain exactly {exact_count} ranked points"
        )
    if len(rows) < minimum:
        raise ReleaseInputError(
            f"{field} must contain at least {minimum} ranked points"
        )
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{field}[{index}]")
        _exact_keys(row, _RANKED_KEYS, f"{field}[{index}]")
        point_id = _identity(row["point_id"], f"{field}[{index}].point_id")
        score = row["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ReleaseInputError(f"{field}[{index}].score must be finite")
        if point_id in seen:
            raise ReleaseInputError(f"{field} contains duplicate point ID")
        seen.add(point_id)
        out.append({"point_id": point_id, "score": float(score)})
    if any(
        float(left["score"]) < float(right["score"])
        for left, right in zip(out, out[1:])
    ):
        raise ReleaseInputError(f"{field} is not ordered by descending score")
    return out


def _load_golden_queries(repo_root: Path) -> tuple[dict[str, str], ...]:
    paths = {
        "golden": repo_root / "ingest/eval/golden_set_v2.jsonl",
        "translation": repo_root / "ingest/eval/query_translations_v2.json",
        "holdout": repo_root / "ingest/eval/holdout_doc_ids_v2.json",
    }
    expected = {
        "golden": GOLDEN_SET_SHA256,
        "translation": TRANSLATION_SHA256,
        "holdout": HOLDOUT_SHA256,
    }
    for name, path in paths.items():
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ReleaseInputError(
                f"cannot read frozen local {name} input: {exc}"
            ) from exc
        if observed != expected[name]:
            raise ReleaseInputError(f"local frozen {name} SHA-256 drifted")
    queries: list[dict[str, str]] = []
    for number, line in enumerate(
        paths["golden"].read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = _strict_json_bytes(line.encode("utf-8"), f"golden line {number}")
        row = _mapping(value, f"golden line {number}")
        queries.append(
            {
                "query_id": _identity(row.get("id"), f"golden line {number}.id"),
                "source": _identity(row.get("source"), f"golden line {number}.source"),
                "query_type": _identity(
                    row.get("query_type"), f"golden line {number}.query_type"
                ),
                "language": _identity(
                    row.get("query_language"),
                    f"golden line {number}.query_language",
                ),
                "document_id": _identity(
                    row.get("document_id"), f"golden line {number}.document_id"
                ),
            }
        )
    query_ids = [row["query_id"] for row in queries]
    if len(query_ids) != EXPECTED_QUERY_COUNT or len(query_ids) != len(set(query_ids)):
        raise ReleaseInputError(
            "local frozen golden set is not exactly 337 unique queries"
        )
    source_counts = {
        source: sum(row["source"] == source for row in queries)
        for source in _FIXED_ZERO_SOURCES
    }
    if source_counts != _FIXED_ZERO_SOURCES:
        raise ReleaseInputError("local frozen incomplete-source slice counts drifted")
    return tuple(queries)


def _validate_baseline_score(
    value: object,
    *,
    query_id: str,
    cluster_id: str,
    expected_query: Mapping[str, str],
) -> dict[str, object]:
    score = _mapping(value, f"baseline trace query {query_id}.score")
    _exact_keys(score, _BASELINE_SCORE_KEYS, f"baseline trace query {query_id}.score")
    if score["id"] != query_id or score["cluster_id"] != cluster_id:
        raise ReleaseInputError(f"baseline trace query {query_id} score identity drift")
    for field in ("source", "query_type", "language"):
        if score[field] != expected_query[field]:
            raise ReleaseInputError(
                f"baseline trace query {query_id} score {field} drifted from frozen v2"
            )
    failed = score["failed"]
    if type(failed) is not bool:
        raise ReleaseInputError(
            f"baseline trace query {query_id}.score.failed must be boolean"
        )
    failure_reason = score["failure_reason"]
    if failure_reason is not None and (
        not isinstance(failure_reason, str) or not failure_reason
    ):
        raise ReleaseInputError(
            f"baseline trace query {query_id}.score.failure_reason is invalid"
        )
    if failed != (failure_reason is not None):
        raise ReleaseInputError(
            f"baseline trace query {query_id} failure fields do not reconcile"
        )
    normalized: dict[str, object] = dict(score)
    for metric in sorted(_BASELINE_METRIC_KEYS):
        observed = score[metric]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not 0.0 <= float(observed) <= 1.0
        ):
            raise ReleaseInputError(
                f"baseline trace query {query_id}.score.{metric} must be finite in [0,1]"
            )
        normalized[metric] = float(observed)
    for field in ("risk_level", "expected_outcome"):
        if not isinstance(score[field], str) or len(score[field]) > 128:
            raise ReleaseInputError(
                f"baseline trace query {query_id}.score.{field} is invalid"
            )
    tags = score["tags"]
    if (
        not isinstance(tags, list)
        or any(not isinstance(tag, str) or not tag for tag in tags)
        or len(tags) != len(set(tags))
    ):
        raise ReleaseInputError(
            f"baseline trace query {query_id}.score.tags is invalid"
        )
    fixed_zero = expected_query["source"] in _FIXED_ZERO_SOURCES
    if fixed_zero and (
        not failed
        or failure_reason != "frozen_incomplete_source_label"
        or any(float(normalized[metric]) != 0.0 for metric in _BASELINE_METRIC_KEYS)
    ):
        raise ReleaseInputError(
            f"baseline trace query {query_id} violates the frozen incomplete-label zero policy"
        )
    return normalized


def _point_id_list(
    value: object,
    field: str,
    *,
    allowed: set[str],
    allow_empty: bool,
) -> list[str]:
    raw_values = _sequence(value, field)
    values = [
        _identity(item, f"{field}[{index}]") for index, item in enumerate(raw_values)
    ]
    if not allow_empty and not values:
        raise ReleaseInputError(f"{field} must not be empty")
    if len(values) != len(set(values)):
        raise ReleaseInputError(f"{field} contains duplicate point IDs")
    if any(value not in allowed for value in values):
        raise ReleaseInputError(
            f"{field} references a point outside the raw candidate pool"
        )
    return values


def _validate_trace_provenance(
    row: Mapping[str, Any],
    *,
    query_id: str,
    expected_document_id: str,
    raw_candidates: Sequence[Mapping[str, object]],
    final_ranking: Sequence[Mapping[str, object]],
    score: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    raw_ids = [str(item["point_id"]) for item in raw_candidates]
    raw_id_set = set(raw_ids)
    final_ids = [str(item["point_id"]) for item in final_ranking]
    if any(point_id not in raw_id_set for point_id in final_ids):
        raise ReleaseInputError(
            f"baseline trace query {query_id} final ranking is not pool-preserving"
        )

    branch_value = _mapping(
        row["branch_provenance"],
        f"baseline trace query {query_id}.branch_provenance",
    )
    _exact_keys(
        branch_value,
        _BRANCH_PROVENANCE_KEYS,
        f"baseline trace query {query_id}.branch_provenance",
    )
    route = _identity(
        branch_value["route"],
        f"baseline trace query {query_id}.branch_provenance.route",
    )
    branch_rows = _sequence(
        branch_value["branches"],
        f"baseline trace query {query_id}.branch_provenance.branches",
    )
    if not branch_rows:
        raise ReleaseInputError(
            f"baseline trace query {query_id} lacks branch provenance"
        )
    normalized_branches: list[dict[str, object]] = []
    covered: set[str] = set()
    branch_names: set[str] = set()
    for index, raw_branch in enumerate(branch_rows):
        branch = _mapping(
            raw_branch,
            f"baseline trace query {query_id}.branch_provenance.branches[{index}]",
        )
        _exact_keys(
            branch,
            _BRANCH_TRACE_KEYS,
            f"baseline trace query {query_id}.branch_provenance.branches[{index}]",
        )
        name = _identity(
            branch["name"],
            f"baseline trace query {query_id}.branch_provenance.branches[{index}].name",
        )
        if name in branch_names:
            raise ReleaseInputError(
                f"baseline trace query {query_id} contains duplicate branch names"
            )
        branch_names.add(name)
        point_ids = _point_id_list(
            branch["point_ids"],
            f"baseline trace query {query_id}.branch_provenance.branches[{index}].point_ids",
            allowed=raw_id_set,
            allow_empty=False,
        )
        covered.update(point_ids)
        normalized_branches.append({"name": name, "point_ids": point_ids})
    if covered != raw_id_set:
        raise ReleaseInputError(
            f"baseline trace query {query_id} branch provenance does not cover the raw pool"
        )

    entity_value = _mapping(
        row["entity_matches"], f"baseline trace query {query_id}.entity_matches"
    )
    _exact_keys(
        entity_value,
        _ENTITY_MATCH_KEYS,
        f"baseline trace query {query_id}.entity_matches",
    )
    raw_entities = _sequence(
        entity_value["query_entities"],
        f"baseline trace query {query_id}.entity_matches.query_entities",
    )
    entities = [
        _identity(
            value,
            f"baseline trace query {query_id}.entity_matches.query_entities[{index}]",
        )
        for index, value in enumerate(raw_entities)
    ]
    if len(entities) != len(set(entities)):
        raise ReleaseInputError(
            f"baseline trace query {query_id} contains duplicate query entities"
        )
    entity_point_ids = _point_id_list(
        entity_value["matched_point_ids"],
        f"baseline trace query {query_id}.entity_matches.matched_point_ids",
        allowed=raw_id_set,
        allow_empty=True,
    )

    document_value = _mapping(
        row["document_matches"],
        f"baseline trace query {query_id}.document_matches",
    )
    _exact_keys(
        document_value,
        _DOCUMENT_MATCH_KEYS,
        f"baseline trace query {query_id}.document_matches",
    )
    if document_value["expected_document_id"] != expected_document_id:
        raise ReleaseInputError(
            f"baseline trace query {query_id} expected document identity drifted"
        )
    document_point_ids = _point_id_list(
        document_value["matched_point_ids"],
        f"baseline trace query {query_id}.document_matches.matched_point_ids",
        allowed=raw_id_set,
        allow_empty=True,
    )
    document_matches = set(document_point_ids)
    expected_document_metrics = {
        "candidate_recall50": float(bool(document_matches & set(raw_ids[:50]))),
        "candidate_recall80": float(bool(document_matches)),
        "document_identity1": float(
            bool(final_ids and final_ids[0] in document_matches)
        ),
    }
    if any(
        float(score[field]) != expected
        for field, expected in expected_document_metrics.items()
    ):
        raise ReleaseInputError(
            f"baseline trace query {query_id} document matches do not reconcile with scores"
        )

    decision_value = _mapping(
        row["route_decision"], f"baseline trace query {query_id}.route_decision"
    )
    _exact_keys(
        decision_value,
        _ROUTE_DECISION_KEYS,
        f"baseline trace query {query_id}.route_decision",
    )
    original = _identity(
        decision_value["original"],
        f"baseline trace query {query_id}.route_decision.original",
    )
    translated_raw = decision_value["translated"]
    translated = (
        None
        if translated_raw is None
        else _identity(
            translated_raw,
            f"baseline trace query {query_id}.route_decision.translated",
        )
    )
    selected = decision_value["selected"]
    if selected not in {"original", "translated"} or (
        selected == "translated" and translated is None
    ):
        raise ReleaseInputError(
            f"baseline trace query {query_id} route selection is invalid"
        )
    selected_route = original if selected == "original" else translated
    if route != selected_route:
        raise ReleaseInputError(
            f"baseline trace query {query_id} route provenance is inconsistent"
        )

    return (
        {"route": route, "branches": normalized_branches},
        {"query_entities": entities, "matched_point_ids": entity_point_ids},
        {
            "expected_document_id": expected_document_id,
            "matched_point_ids": document_point_ids,
        },
        {
            "original": original,
            "translated": translated,
            "selected": selected,
        },
    )


def _validate_trace(
    baseline_root: Path,
    repeat: Mapping[str, Any],
    *,
    expected_queries: tuple[dict[str, str], ...],
    configuration_hash: str,
) -> tuple[str, str, tuple[dict[str, object], ...]]:
    _, raw, _ = _read_bound_file(
        baseline_root,
        repeat["trace_path"],
        repeat["trace_sha256"],
        repeat["trace_size_bytes"],
        f"baseline repeat {repeat['repeat']} trace",
        max_bytes=256 * 1024 * 1024,
    )
    if raw is None:  # pragma: no cover - captured by contract above
        raise ReleaseInputError("baseline trace bytes were not captured")
    rows: list[dict[str, object]] = []
    observed_ids: list[str] = []
    expected_by_id = {row["query_id"]: row for row in expected_queries}
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json_bytes(line, f"baseline trace line {number}")
        row = _mapping(value, f"baseline trace line {number}")
        _reject_placeholders_and_secrets(row)
        _exact_keys(row, _TRACE_KEYS, f"baseline trace line {number}")
        query_id = _identity(row["query_id"], f"baseline trace line {number}.query_id")
        cluster_id = _identity(
            row["cluster_id"], f"baseline trace line {number}.cluster_id"
        )
        if query_id not in expected_by_id:
            raise ReleaseInputError(f"baseline trace contains unknown query {query_id}")
        score = _validate_baseline_score(
            row["score"],
            query_id=query_id,
            cluster_id=cluster_id,
            expected_query=expected_by_id[query_id],
        )
        expected_status = "zero_score_failure" if score["failed"] else "success"
        if row["status"] not in (
            {expected_status} | _FAILURE_STATUSES if score["failed"] else {"success"}
        ):
            raise ReleaseInputError(
                f"baseline trace query {query_id} status/failure state is inconsistent"
            )
        if row["degraded"] is not False:
            raise ReleaseInputError(f"baseline trace query {query_id} is degraded")
        if row["configuration_hash"] != configuration_hash:
            raise ReleaseInputError(
                f"baseline trace query {query_id} configuration drift"
            )
        raw_candidates = _ranked_rows(
            row["raw_candidates"],
            f"baseline trace query {query_id}.raw_candidates",
            minimum=80,
            exact_count=80,
        )
        final_ranking = _ranked_rows(
            row["final_ranking"],
            f"baseline trace query {query_id}.final_ranking",
            minimum=10,
            exact_count=10,
        )
        (
            branch_provenance,
            entity_matches,
            document_matches,
            route_decision,
        ) = _validate_trace_provenance(
            row,
            query_id=query_id,
            expected_document_id=expected_by_id[query_id]["document_id"],
            raw_candidates=raw_candidates,
            final_ranking=final_ranking,
            score=score,
        )
        timings = _mapping(
            row["timings_ms"], f"baseline trace query {query_id}.timings_ms"
        )
        if not timings or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in timings.values()
        ):
            raise ReleaseInputError(
                f"baseline trace query {query_id} timings are invalid"
            )
        observed_ids.append(query_id)
        rows.append(
            {
                "query_id": query_id,
                "raw_candidates": raw_candidates,
                "final_ranking": final_ranking,
                "branch_provenance": branch_provenance,
                "entity_matches": entity_matches,
                "document_matches": document_matches,
                "route_decision": route_decision,
                "score": score,
            }
        )
    expected_query_ids = tuple(row["query_id"] for row in expected_queries)
    if tuple(observed_ids) != expected_query_ids:
        raise ReleaseInputError(
            "baseline trace must contain the frozen 337 queries exactly once in golden-set order"
        )
    ranking_hash = _canonical_hash(
        [
            {
                "query_id": row["query_id"],
                "raw_candidates": row["raw_candidates"],
                "final_ranking": row["final_ranking"],
            }
            for row in rows
        ]
    )
    decision_hash = _canonical_hash(rows)
    if repeat["ranking_hash"] != ranking_hash:
        raise ReleaseInputError(
            "baseline repeat ranking_hash does not match trace bytes"
        )
    if repeat["decision_result_hash"] != decision_hash:
        raise ReleaseInputError(
            "baseline repeat decision_result_hash does not match trace bytes"
        )
    return ranking_hash, decision_hash, tuple(rows)


def _validate_baseline(
    bundle_root: Path,
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    release_models: Mapping[str, Any],
) -> ValidatedBaseline:
    _exact_keys(binding, _BASELINE_BINDING_KEYS, "baseline binding")
    root_value = binding["root"]
    if (
        not isinstance(root_value, str)
        or not root_value
        or ".." in Path(root_value).parts
    ):
        raise ReleaseInputError("baseline.root must be a normalized relative directory")
    baseline_root = bundle_root / Path(root_value)
    try:
        root_info = baseline_root.lstat()
    except FileNotFoundError as exc:
        raise ReleaseInputError("frozen baseline package is absent") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ReleaseInputError("baseline.root must be a real directory")
    manifest_relative = Path(root_value) / str(binding["manifest_path"])
    _, raw, manifest_sha = _read_bound_file(
        bundle_root,
        manifest_relative.as_posix(),
        binding["manifest_sha256"],
        binding["manifest_size_bytes"],
        "baseline manifest",
        max_bytes=4 * 1024 * 1024,
    )
    if raw is None:  # pragma: no cover - captured by contract above
        raise ReleaseInputError("baseline manifest bytes were not captured")
    manifest = _mapping(
        _strict_json_bytes(raw, "baseline manifest"), "baseline manifest"
    )
    _reject_placeholders_and_secrets(manifest)
    _exact_keys(manifest, _BASELINE_KEYS, "baseline manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ReleaseInputError("baseline schema_version mismatch")
    dataset = _mapping(manifest["dataset"], "baseline.dataset")
    _exact_keys(dataset, _BASELINE_DATASET_KEYS, "baseline.dataset")
    expected_dataset = {
        "name": "v2",
        "query_count": EXPECTED_QUERY_COUNT,
        "golden_set_sha256": GOLDEN_SET_SHA256,
        "translation_sha256": TRANSLATION_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
    }
    if dict(dataset) != expected_dataset:
        raise ReleaseInputError("baseline does not bind the frozen 337-query v2 inputs")
    configuration_hash = _sha256(
        manifest["configuration_hash"], "baseline.configuration_hash"
    )
    collection = _mapping(manifest["collection"], "baseline.collection")
    _exact_keys(collection, _BASELINE_COLLECTION_KEYS, "baseline.collection")
    for field in ("snapshot_sha256", "collection_sha256", "configuration_sha256"):
        _sha256(collection[field], f"baseline.collection.{field}")
    if collection["configuration_sha256"] != configuration_hash:
        raise ReleaseInputError("baseline collection/configuration provenance mismatch")
    _identity(
        collection["physical_collection"], "baseline.collection.physical_collection"
    )
    _identity(collection["generation_id"], "baseline.collection.generation_id")
    models = _mapping(manifest["models"], "baseline.models")
    _exact_keys(models, _MODEL_KEYS, "baseline.models")
    for role in sorted(_MODEL_KEYS):
        item = _mapping(models[role], f"baseline.models.{role}")
        _exact_keys(item, _BASELINE_MODEL_ENTRY_KEYS, f"baseline.models.{role}")
        release_item = _mapping(release_models[role], f"models.{role}")
        if (
            item["name"] != release_item["name"]
            or item["revision"] != release_item["revision"]
            or item["artifact_sha256"] != release_item["artifact_sha256"]
        ):
            raise ReleaseInputError(
                f"baseline {role} model artifact is not pairable with release input"
            )
        _sha256(item["artifact_sha256"], f"baseline.models.{role}.artifact_sha256")

    expected_queries = _load_golden_queries(repo_root)
    expected_query_ids = tuple(row["query_id"] for row in expected_queries)
    repeats = _sequence(manifest["repeats"], "baseline.repeats")
    if len(repeats) != 2:
        raise ReleaseInputError(
            "baseline must contain exactly two deterministic repeats"
        )
    validated_repeats: list[tuple[str, str, tuple[dict[str, object], ...]]] = []
    for expected_repeat, raw_repeat in enumerate(repeats, 1):
        repeat = _mapping(raw_repeat, f"baseline.repeats[{expected_repeat - 1}]")
        _exact_keys(repeat, _REPEAT_KEYS, f"baseline.repeats[{expected_repeat - 1}]")
        if repeat["repeat"] != expected_repeat or repeat["track"] != "production":
            raise ReleaseInputError(
                "baseline repeats must be production repeats 1 then 2"
            )
        _sha256(repeat["trace_sha256"], "baseline repeat trace_sha256")
        _sha256(repeat["ranking_hash"], "baseline repeat ranking_hash")
        _sha256(repeat["decision_result_hash"], "baseline repeat decision_result_hash")
        validated_repeats.append(
            _validate_trace(
                baseline_root,
                repeat,
                expected_queries=expected_queries,
                configuration_hash=configuration_hash,
            )
        )
    if validated_repeats[0] != validated_repeats[1]:
        raise ReleaseInputError("baseline repeats are not deterministic and pairable")
    ranking_hash = validated_repeats[0][0]
    decision_hash = validated_repeats[0][1]
    queries = validated_repeats[0][2]
    return ValidatedBaseline(
        root=baseline_root,
        manifest_sha256=manifest_sha,
        query_ids=expected_query_ids,
        ranking_hash=ranking_hash,
        decision_result_hash=decision_hash,
        collection_sha256=str(collection["collection_sha256"]),
        configuration_hash=configuration_hash,
        ordered_query_ids=expected_query_ids,
        queries=queries,
        ranking_hashes=(ranking_hash, ranking_hash),
        decision_result_hashes=(decision_hash, decision_hash),
        repeats=(
            {
                "ranking_hash": ranking_hash,
                "decision_result_hash": decision_hash,
            },
            {
                "ranking_hash": ranking_hash,
                "decision_result_hash": decision_hash,
            },
        ),
    )


def _validate_license_attestation(
    raw: bytes,
    *,
    role: str,
    model: Mapping[str, Any],
) -> None:
    value = _mapping(
        _strict_json_bytes(raw, f"models.{role}.license"),
        f"models.{role}.license",
    )
    _reject_placeholders_and_secrets(value)
    try:
        attestation = LicenseAttestation.from_dict(value)
    except ValueError as exc:
        raise ReleaseInputError(
            f"models.{role}.license is not a valid license attestation: {exc}"
        ) from exc
    expected_role = ModelRole.RERANKER if role == "reranker" else ModelRole.EMBEDDER
    if (
        attestation.model_id != model["name"]
        or attestation.revision != model["revision"]
        or attestation.role is not expected_role
    ):
        raise ReleaseInputError(
            f"models.{role}.license does not bind the exact model identity and role"
        )
    if not attestation.production_eligible:
        raise ReleaseInputError(f"models.{role}.license is not production-eligible")
    try:
        reviewed_at = date.fromisoformat(attestation.reviewed_at)
    except ValueError as exc:
        raise ReleaseInputError(
            f"models.{role}.license reviewed_at must be an ISO date"
        ) from exc
    if reviewed_at > date.fromisoformat(CRAWL_END_DATE):
        raise ReleaseInputError(
            f"models.{role}.license was reviewed after the frozen release date"
        )
    source = urlparse(attestation.authoritative_source_url)
    if (
        source.scheme != "https"
        or not source.hostname
        or source.username is not None
        or source.password is not None
    ):
        raise ReleaseInputError(
            f"models.{role}.license authoritative source must be credential-free HTTPS"
        )


def _load_supply_chain_lock(path: Path, *, expected_sha256: str) -> Any:
    """Use the one established strict production-lock parser."""

    try:
        from scripts import supply_chain

        lock = supply_chain.parse_requirements_lock(path)
    except (ImportError, OSError, UnicodeError, ValueError) as exc:
        raise ReleaseInputError(f"invalid dependency lock: {exc}") from exc
    if lock.sha256 != expected_sha256:
        raise ReleaseInputError(
            "strict dependency-lock parser digest does not match exact file bytes"
        )
    return lock


def _validate_runtime_identity_strict(
    identity: Mapping[str, Any],
    lock: Any,
    *,
    runtime_revision: str,
    qdrant_revision: str,
    oci_reference: str,
) -> None:
    """Apply the established runtime validator and cross-bind manifest identities."""

    if identity.get("runtime_revision") != runtime_revision:
        raise ReleaseInputError(
            "runtime identity does not match runtime.runtime_revision"
        )
    if identity.get("base_image") != oci_reference:
        raise ReleaseInputError(
            "runtime identity does not match the exact OCI reference"
        )
    if identity.get("qdrant_version") != qdrant_revision:
        raise ReleaseInputError(
            "runtime identity does not match runtime.qdrant_revision"
        )
    try:
        from scripts import supply_chain

        supply_chain.validate_runtime_identity(
            dict(identity),
            lock,
            base_image=oci_reference,
            qdrant_version=qdrant_revision,
            qdrant_archive_url_value=str(identity.get("qdrant_archive_url", "")),
            qdrant_archive_sha256=str(identity.get("qdrant_archive_sha256", "")),
            qdrant_checksum_source_url=str(
                identity.get("qdrant_checksum_source_url", "")
            ),
        )
    except (ImportError, OSError, UnicodeError, ValueError) as exc:
        raise ReleaseInputError(f"invalid runtime identity: {exc}") from exc


def validate_release_inputs(
    root: Path | str,
    *,
    repo_root: Path | str,
    environ: Mapping[str, str] | None = None,
) -> ValidatedReleaseInputs:
    """Validate the complete external bundle and frozen baseline without side effects."""

    bundle_root = _safe_root(Path(root))
    manifest_path = _relative_file(bundle_root, MANIFEST_NAME, "manifest")
    raw = _read_bounded_file(
        manifest_path, field="release manifest", max_bytes=4 * 1024 * 1024
    )
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = _mapping(_strict_json_bytes(raw, "release manifest"), "release manifest")
    _reject_placeholders_and_secrets(manifest)
    _exact_keys(manifest, _TOP_KEYS, "release manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ReleaseInputError("release manifest schema_version mismatch")

    release = _mapping(manifest["release"], "release")
    _exact_keys(release, _RELEASE_KEYS, "release")
    expected_release = {
        "snapshot_id": SNAPSHOT_ID,
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "crawl_start_date": CRAWL_START_DATE,
        "crawl_end_date": CRAWL_END_DATE,
        "sources": list(SOURCES),
    }
    if dict(release) != expected_release:
        raise ReleaseInputError(
            "release identity, interval, collection, or source order drifted"
        )

    code = _mapping(manifest["code"], "code")
    _exact_keys(code, _CODE_KEYS, "code")
    repository_revision = _revision(
        code["repository_revision"], "code.repository_revision"
    )
    resolved_repo_root = Path(repo_root).expanduser().absolute()
    observed_repository_revision = _current_repository_revision(resolved_repo_root)
    if repository_revision != observed_repository_revision:
        raise ReleaseInputError(
            "release repository revision does not match checked-out HEAD"
        )
    expected_code_identity = _sha256(
        code["code_identity_sha256"], "code.code_identity_sha256"
    )
    observed_code_identity = code_identity_sha256(resolved_repo_root)
    if expected_code_identity != observed_code_identity:
        raise ReleaseInputError(
            "release code identity does not match tracked and untracked repository bytes"
        )
    for field in ("retriever_revision", "tokenizer_revision", "reranker_revision"):
        _revision(code[field], f"code.{field}")

    models = _mapping(manifest["models"], "models")
    _exact_keys(models, _MODEL_KEYS, "models")
    for role in sorted(_MODEL_KEYS):
        item = _mapping(models[role], f"models.{role}")
        _exact_keys(item, _MODEL_ENTRY_KEYS, f"models.{role}")
        _identity(item["name"], f"models.{role}.name")
        revision = _revision(item["revision"], f"models.{role}.revision")
        if revision != code[f"{role if role != 'embedding' else 'retriever'}_revision"]:
            raise ReleaseInputError(f"models.{role}.revision is not code-bound")
        _read_bound_file(
            bundle_root,
            item["artifact_path"],
            item["artifact_sha256"],
            item["artifact_size_bytes"],
            f"models.{role}.artifact",
            capture=False,
        )
        _, license_raw, _ = _read_bound_file(
            bundle_root,
            item["license_path"],
            item["license_sha256"],
            item["license_size_bytes"],
            f"models.{role}.license",
            max_bytes=4 * 1024 * 1024,
        )
        if license_raw is None:  # pragma: no cover - captured by contract above
            raise ReleaseInputError(f"models.{role}.license bytes were not captured")
        _validate_license_attestation(license_raw, role=role, model=item)

    dependencies = _mapping(manifest["dependencies"], "dependencies")
    _exact_keys(dependencies, _DEPENDENCY_KEYS, "dependencies")
    lock_path, lock_raw, lock_sha256 = _read_bound_file(
        bundle_root,
        dependencies["lock_path"],
        dependencies["lock_sha256"],
        dependencies["lock_size_bytes"],
        "dependencies.lock",
        max_bytes=16 * 1024 * 1024,
    )
    if lock_raw is None:  # pragma: no cover - captured by contract above
        raise ReleaseInputError("dependency lock bytes were not captured")
    try:
        lock_raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ReleaseInputError("dependency lock must be strict UTF-8") from exc
    lock = _load_supply_chain_lock(lock_path, expected_sha256=lock_sha256)
    files = _sequence(dependencies["files"], "dependencies.files")
    if not files:
        raise ReleaseInputError(
            "dependencies.files must bind offline dependency artifacts"
        )
    dependency_names: set[str] = set()
    for index, raw_file in enumerate(files):
        item = _mapping(raw_file, f"dependencies.files[{index}]")
        _exact_keys(item, _DEPENDENCY_FILE_KEYS, f"dependencies.files[{index}]")
        raw_name = _identity(item["name"], f"dependencies.files[{index}].name")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if name in dependency_names:
            raise ReleaseInputError("dependencies.files contains duplicate names")
        dependency_names.add(name)
        if name not in lock.requirements:
            raise ReleaseInputError(
                f"dependencies.files[{index}] is not present in the strict lock"
            )
        _, _, artifact_sha256 = _read_bound_file(
            bundle_root,
            item["path"],
            item["sha256"],
            item["size_bytes"],
            f"dependencies.files[{index}]",
            capture=False,
        )
        if artifact_sha256 not in lock.requirements[name].hashes:
            raise ReleaseInputError(
                f"dependencies.files[{index}] hash is not authorized by the strict lock"
            )
    if dependency_names != set(lock.requirements):
        missing = sorted(set(lock.requirements) - dependency_names)
        raise ReleaseInputError(
            f"dependencies.files does not cover every locked requirement: {missing}"
        )

    runtime = _mapping(manifest["runtime"], "runtime")
    _exact_keys(runtime, _RUNTIME_KEYS, "runtime")
    _, runtime_raw, runtime_sha256 = _read_bound_file(
        bundle_root,
        runtime["identity_path"],
        runtime["identity_sha256"],
        runtime["identity_size_bytes"],
        "runtime.identity",
        max_bytes=4 * 1024 * 1024,
    )
    if runtime_raw is None:  # pragma: no cover - captured by contract above
        raise ReleaseInputError("runtime identity bytes were not captured")
    runtime_identity = _mapping(
        _strict_json_bytes(runtime_raw, "runtime identity"), "runtime identity"
    )
    _reject_placeholders_and_secrets(runtime_identity)
    runtime_revision = _revision(
        runtime["runtime_revision"], "runtime.runtime_revision"
    )
    qdrant_revision = _identity(runtime["qdrant_revision"], "runtime.qdrant_revision")
    oci_digest = runtime["oci_digest"]
    if not isinstance(oci_digest, str) or _OCI_DIGEST_RE.fullmatch(oci_digest) is None:
        raise ReleaseInputError("runtime.oci_digest must be sha256:<64 lowercase hex>")
    oci_reference = runtime["oci_reference"]
    if (
        not isinstance(oci_reference, str)
        or oci_reference.count("@sha256:") != 1
        or not oci_reference.endswith(oci_digest)
    ):
        raise ReleaseInputError(
            "runtime.oci_reference must end in the exact OCI digest"
        )
    _validate_runtime_identity_strict(
        runtime_identity,
        lock,
        runtime_revision=runtime_revision,
        qdrant_revision=qdrant_revision,
        oci_reference=oci_reference,
    )
    environment = _mapping(runtime["environment"], "runtime.environment")
    _exact_keys(environment, _ENVIRONMENT_KEYS, "runtime.environment")
    expected_environment = {
        "SNAPSHOT_ID": SNAPSHOT_ID,
        "GENERATION_ID": GENERATION_ID,
        "COLLECTION_NAME": PHYSICAL_COLLECTION,
        "EMBED_MODEL": str(models["embedding"]["name"]),
        "TOKENIZER_MODEL": str(models["tokenizer"]["name"]),
        "RERANK_MODEL": str(models["reranker"]["name"]),
        "EMBED_REVISION": str(models["embedding"]["revision"]),
        "TOKENIZER_REVISION": str(models["tokenizer"]["revision"]),
        "RERANK_REVISION": str(models["reranker"]["revision"]),
        "DENSE_DIM": "1024",
        "CHUNK_TOKENS": "512",
        "CHUNK_OVERLAP": "80",
        "CHUNK_MIN_TOKENS": "64",
        "RERANK_ENABLED": "true",
        "RERANK_CANDIDATES": "80",
        "RERANK_MIN_SCORE": "0.3",
        "RERANK_BACKEND": "torch",
        "RERANK_CONTEXT_ENRICHED": "true",
        "RERANK_MAX_LENGTH": "1024",
        "CITATION_ROUTE": "ids",
        "EMBED_HEADER_V2": "true",
        "PRODUCTION_MODE": "true",
    }
    embed_device = environment.get("EMBED_DEVICE")
    embed_use_fp16 = environment.get("EMBED_USE_FP16")
    embed_batch_size = environment.get("EMBED_BATCH_SIZE")
    if embed_device != "cuda":
        raise ReleaseInputError(
            "runtime.environment.EMBED_DEVICE must be exactly 'cuda'"
        )
    if embed_use_fp16 not in {"true", "false"}:
        raise ReleaseInputError(
            "runtime.environment.EMBED_USE_FP16 must be exactly 'true' or 'false'"
        )
    if (
        not isinstance(embed_batch_size, str)
        or _CANONICAL_POSITIVE_INTEGER_RE.fullmatch(embed_batch_size) is None
    ):
        raise ReleaseInputError(
            "runtime.environment.EMBED_BATCH_SIZE must be a canonical positive integer"
        )
    expected_environment.update(
        {
            "EMBED_DEVICE": embed_device,
            "EMBED_USE_FP16": embed_use_fp16,
            "EMBED_BATCH_SIZE": embed_batch_size,
        }
    )
    if dict(environment) != expected_environment:
        raise ReleaseInputError(
            "runtime.environment does not bind the exact release tuple"
        )
    actual_environment = os.environ if environ is None else environ
    for name, expected in environment.items():
        if _SECRET_KEY_RE.search(str(name)):
            raise ReleaseInputError(
                "secret environment names are forbidden in the bundle"
            )
        if not isinstance(expected, str) or _PLACEHOLDER_RE.search(expected):
            raise ReleaseInputError(f"runtime.environment.{name} is invalid")
        if actual_environment.get(str(name)) != expected:
            raise ReleaseInputError(f"runtime environment drift for {name}")

    retrieval = _mapping(manifest["retrieval"], "retrieval")
    _exact_keys(retrieval, _RETRIEVAL_KEYS, "retrieval")
    configuration_hash = _sha256(
        retrieval["configuration_hash"], "retrieval.configuration_hash"
    )
    knobs = _mapping(retrieval["knobs"], "retrieval.knobs")
    _exact_keys(knobs, _RETRIEVAL_KNOB_KEYS, "retrieval.knobs")
    if any(
        type(knobs[field]) is not type(expected) or knobs[field] != expected
        for field, expected in _EXPECTED_RETRIEVAL_KNOBS.items()
    ):
        raise ReleaseInputError(
            "retrieval knobs do not match the complete frozen release contract"
        )
    if _canonical_hash(knobs) != configuration_hash:
        raise ReleaseInputError(
            "retrieval.configuration_hash does not match exact knobs"
        )

    operator = _mapping(manifest["operator"], "operator")
    _exact_keys(operator, _OPERATOR_KEYS, "operator")
    actor = _identity(operator["actor"], "operator.actor")
    run_id = _identity(operator["run_id"], "operator.run_id")
    baseline = _validate_baseline(
        bundle_root,
        _mapping(manifest["baseline"], "baseline"),
        repo_root=resolved_repo_root,
        release_models=models,
    )
    if baseline.configuration_hash != configuration_hash:
        raise ReleaseInputError(
            "baseline retrieval configuration is not pairable with the release"
        )
    # Re-read both supply-chain inputs after all dependent evidence was checked so an
    # in-place replacement cannot be paired with stale parsed values.
    _read_bound_file(
        bundle_root,
        dependencies["lock_path"],
        lock_sha256,
        dependencies["lock_size_bytes"],
        "dependencies.lock",
        max_bytes=16 * 1024 * 1024,
        capture=False,
    )
    _read_bound_file(
        bundle_root,
        runtime["identity_path"],
        runtime_sha256,
        runtime["identity_size_bytes"],
        "runtime.identity",
        max_bytes=4 * 1024 * 1024,
        capture=False,
    )
    if code_identity_sha256(resolved_repo_root) != observed_code_identity:
        raise ReleaseInputError("repository code identity changed during validation")
    return ValidatedReleaseInputs(
        root=bundle_root,
        manifest_sha256=manifest_sha,
        repository_revision=repository_revision,
        code_identity_sha256=observed_code_identity,
        configuration_hash=configuration_hash,
        oci_digest=oci_digest,
        actor=actor,
        run_id=run_id,
        baseline=baseline,
    )
