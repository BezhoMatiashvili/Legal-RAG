"""Create-only, exact v2-to-immutable-candidate qrel binding.

The frozen v2 annotations identify a source/document and character span, while immutable
generation points identify a specific canonical version.  This module is the only supported
bridge between those identity schemes.  It never searches by title, URL, similarity, or
nearby text: a query is mapped only when the candidate snapshot contains exactly one admitted
``primary_official`` record with the same source/document identity and a byte-for-byte,
NFC-identical body.  Every other outcome is retained as an explicit zero-score failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import goldset

SCHEMA_VERSION = "v2-candidate-qrels/v1"
EXPECTED_GOLDEN_SHA256 = (
    "753e2985315be3e408c3db3303f90625d9c66984b5fc9519cf7e32a8d46252c6"
)
EXPECTED_TRANSLATIONS_SHA256 = (
    "0884870a8fa68527c959de3781c4515458d96784c4f599158427f058ce727fad"
)
EXPECTED_HOLDOUT_SHA256 = (
    "eaee96072f66a0f3f63d6d1cbe61e4566d3b405daedf389211bb351d05b7ad3e"
)
EXPECTED_QUERY_COUNT = 337
EXPECTED_SNAPSHOT_ID = "v3_512_attested_20260715_01"
FIXED_INCOMPLETE_SOURCE_COUNTS = {"tas": 25, "tbappeal": 31}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateQrelError(ValueError):
    """The frozen annotations cannot be safely bound to the candidate snapshot."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _create_private_json(path: Path, value: object) -> Path:
    """Atomically create ``path`` without ever replacing an existing destination."""

    destination = path.expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise CandidateQrelError("qrel artifact parent must be a real directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"qrel artifact already exists: {destination}")
    payload = _canonical_json_bytes(value)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"qrel artifact already exists: {destination}") from None
        temporary.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CandidateQrelError(f"{path}:{line_number}: expected an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateQrelError(f"cannot read candidate snapshot JSONL {path}: {exc}") from exc
    return rows


def _candidate_inventory(
    snapshot_root: Path, sources: Sequence[str]
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    admitted: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    quarantined: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sorted(set(sources)):
        for row in _iter_jsonl(snapshot_root / "docs" / f"{source}.jsonl"):
            if row.get("source") != source:
                raise CandidateQrelError(f"candidate docs/{source}.jsonl contains wrong source")
            document_id = row.get("document_id")
            version_id = row.get("version_id")
            body = row.get("body_markdown")
            if not all(isinstance(item, str) and item for item in (document_id, version_id)):
                raise CandidateQrelError(
                    f"candidate record lacks document/version identity: {source}:{document_id}"
                )
            if not isinstance(body, str):
                raise CandidateQrelError(
                    f"candidate record lacks canonical body: {source}:{document_id}@{version_id}"
                )
            admitted[(source, document_id)].append(row)
    quarantine_path = snapshot_root / "quarantine.jsonl"
    for row in _iter_jsonl(quarantine_path):
        source = row.get("source")
        document_id = row.get("document_id")
        if isinstance(source, str) and isinstance(document_id, str):
            quarantined[(source, document_id)].append(row)
    return dict(admitted), dict(quarantined)


def _validate_frozen_inputs(
    golden_path: Path, translations_path: Path, holdout_path: Path
) -> tuple[list[goldset.GoldQuery], dict[str, str]]:
    hashes = {
        "golden_set_sha256": _sha256_file(golden_path),
        "translations_sha256": _sha256_file(translations_path),
        "holdout_sha256": _sha256_file(holdout_path),
    }
    expected = {
        "golden_set_sha256": EXPECTED_GOLDEN_SHA256,
        "translations_sha256": EXPECTED_TRANSLATIONS_SHA256,
        "holdout_sha256": EXPECTED_HOLDOUT_SHA256,
    }
    if hashes != expected:
        raise CandidateQrelError(
            f"frozen v2 identity mismatch: expected={expected}, observed={hashes}"
        )
    queries = goldset.load_golden_set(golden_path)
    if len(queries) != EXPECTED_QUERY_COUNT or len({query.id for query in queries}) != len(queries):
        raise CandidateQrelError("frozen v2 must contain exactly 337 unique query ids")
    if any(len(query.relevance) != 1 for query in queries):
        raise CandidateQrelError("frozen v2 adapter requires exactly one relevance span per query")
    source_counts = Counter(query.gold_source for query in queries)
    for source, count in FIXED_INCOMPLETE_SOURCE_COUNTS.items():
        if source_counts[source] != count:
            raise CandidateQrelError(
                f"frozen incomplete-source slice {source} must contain exactly {count} queries"
            )
    return queries, hashes


def _mapping_row(
    query: goldset.GoldQuery,
    frozen_body: str,
    admitted: Mapping[tuple[str, str], list[dict[str, Any]]],
    quarantined: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    relevance = query.relevance[0]
    source = relevance.source or query.gold_source
    document_id = relevance.document_id
    identity = (source, document_id)
    base = {
        "query_id": query.id,
        "source": source,
        "document_id": document_id,
        "frozen_body_sha256": hashlib.sha256(frozen_body.encode("utf-8")).hexdigest(),
        "frozen_body_nfc_sha256": hashlib.sha256(
            unicodedata.normalize("NFC", frozen_body).encode("utf-8")
        ).hexdigest(),
    }

    # These labels are known incomplete summaries in frozen v2.  They remain explicit
    # failures even if a future candidate accidentally admits a similarly named record.
    if source in FIXED_INCOMPLETE_SOURCE_COUNTS:
        return {
            **base,
            "status": "failure",
            "failure_reason": "frozen_incomplete_source_label",
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
        }

    candidates = admitted.get(identity, [])
    if not candidates:
        reason = "quarantined_candidate_document" if quarantined.get(identity) else "missing_candidate_document"
        return {
            **base,
            "status": "failure",
            "failure_reason": reason,
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
        }
    candidate_inventory = [
        {
            "version_id": candidate.get("version_id"),
            "source_authority": candidate.get("source_authority"),
            "content_complete": candidate.get("content_complete"),
            "admissible": candidate.get("admissible"),
            "body_sha256": hashlib.sha256(
                candidate["body_markdown"].encode("utf-8")
            ).hexdigest(),
            "body_nfc_sha256": hashlib.sha256(
                unicodedata.normalize("NFC", candidate["body_markdown"]).encode("utf-8")
            ).hexdigest(),
        }
        for candidate in candidates
    ]
    primary = [
        candidate
        for candidate in candidates
        if candidate.get("source_authority") == "primary_official"
    ]
    if not primary:
        return {
            **base,
            "status": "failure",
            "failure_reason": "candidate_not_primary_official",
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
            "candidate_inventory": candidate_inventory,
        }
    admissible = [
        candidate
        for candidate in primary
        if candidate.get("content_complete") is True
        and candidate.get("admissible") is True
    ]
    if not admissible:
        return {
            **base,
            "status": "failure",
            "failure_reason": "candidate_not_admissible",
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
            "candidate_inventory": candidate_inventory,
        }
    byte_matches = [
        candidate
        for candidate in admissible
        if candidate["body_markdown"].encode("utf-8") == frozen_body.encode("utf-8")
    ]
    if not byte_matches:
        return {
            **base,
            "status": "failure",
            "failure_reason": "candidate_body_bytes_changed",
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
            "candidate_inventory": candidate_inventory,
        }
    nfc_matches = [
        candidate
        for candidate in byte_matches
        if unicodedata.normalize("NFC", candidate["body_markdown"])
        == unicodedata.normalize("NFC", frozen_body)
    ]
    if not nfc_matches:
        return {
            **base,
            "status": "failure",
            "failure_reason": "candidate_body_nfc_changed",
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
            "candidate_inventory": candidate_inventory,
        }
    if len(nfc_matches) != 1:
        return {
            **base,
            "status": "failure",
            "failure_reason": "ambiguous_candidate_document",
            "zero_score_failure": True,
            "version_id": None,
            "candidate_body_sha256": None,
            "candidate_body_nfc_sha256": None,
            "candidate_inventory": candidate_inventory,
        }
    candidate = nfc_matches[0]
    candidate_body = candidate["body_markdown"]
    candidate_sha = hashlib.sha256(candidate_body.encode("utf-8")).hexdigest()
    candidate_nfc_sha = hashlib.sha256(
        unicodedata.normalize("NFC", candidate_body).encode("utf-8")
    ).hexdigest()
    common = {
        **base,
        "version_id": candidate["version_id"],
        "candidate_body_sha256": candidate_sha,
        "candidate_body_nfc_sha256": candidate_nfc_sha,
        "candidate_inventory": candidate_inventory,
    }
    got = candidate_body[relevance.char_start : relevance.char_end]
    if unicodedata.normalize("NFC", got) != unicodedata.normalize(
        "NFC", relevance.evidence_quote
    ):
        return {
            **common,
            "status": "failure",
            "failure_reason": "candidate_evidence_span_changed",
            "zero_score_failure": True,
        }
    return {
        **common,
        "status": "mapped",
        "failure_reason": None,
        "zero_score_failure": False,
    }


def build_v2_candidate_qrel_artifact(
    *,
    candidate_snapshot: Path,
    output: Path,
    golden_path: Path = goldset.DEFAULT_GOLD_V2,
    translations_path: Path | None = None,
    holdout_path: Path = goldset.DEFAULT_HOLDOUT_V2,
    frozen_roots: Sequence[Path] | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Validate and create one immutable qrel-binding artifact.

    The destination is create-only.  Failures in individual mappings are data in the artifact;
    malformed inputs, changed frozen identities, or an unsealed candidate abort the build.
    """

    from ingest.snapshot import verify_sealed_snapshot

    translations_path = translations_path or (goldset.EVAL_DIR / "query_translations_v2.json")
    queries, frozen_hashes = _validate_frozen_inputs(
        Path(golden_path), Path(translations_path), Path(holdout_path)
    )
    snapshot_root = Path(candidate_snapshot).expanduser().absolute()
    try:
        manifest = verify_sealed_snapshot(
            snapshot_root, allow_preflight=False, require_all_sources=True
        )
    except Exception as exc:
        raise CandidateQrelError(f"candidate snapshot is not a sealed production snapshot: {exc}") from exc
    if manifest.get("preflight") is not False:
        raise CandidateQrelError("candidate qrels cannot bind a preflight snapshot")
    if manifest.get("snapshot_id") != EXPECTED_SNAPSHOT_ID:
        raise CandidateQrelError(
            f"candidate qrels require snapshot {EXPECTED_SNAPSHOT_ID!r}"
        )

    roots = tuple(frozen_roots or goldset.EVAL_SETS["v2"].roots)
    bodies = goldset.SnapshotBodies(
        root=roots[0], needed=goldset.gold_docs(queries), extra_roots=roots[1:]
    )
    # Re-ground first so a corrupt frozen corpus cannot be disguised as candidate failures.
    goldset.reground(queries, bodies)
    admitted, quarantined = _candidate_inventory(
        snapshot_root, tuple(sorted({query.gold_source for query in queries}))
    )
    mappings = [
        _mapping_row(
            query,
            goldset.relevance_body(
                bodies, query.relevance[0].source or query.gold_source, query.relevance[0]
            ),
            admitted,
            quarantined,
        )
        for query in queries
    ]
    failures = Counter(
        row["failure_reason"] for row in mappings if row["status"] == "failure"
    )
    source_counts = Counter(query.gold_source for query in queries)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
        "candidate_snapshot": {
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_sha256": manifest["snapshot_sha256"],
            "corpus_sha256": manifest["corpus_sha256"],
        },
        "frozen_inputs": frozen_hashes,
        "query_count": len(queries),
        "ordered_query_ids_sha256": hashlib.sha256(
            _canonical_json_bytes([query.id for query in queries])
        ).hexdigest(),
        "counts": {
            "mapped": sum(row["status"] == "mapped" for row in mappings),
            "failures": sum(row["status"] == "failure" for row in mappings),
            "by_failure_reason": dict(sorted(failures.items())),
            "by_source": dict(sorted(source_counts.items())),
        },
        "mappings": mappings,
        "mapping_policy": {
            "identity": "exact_source_document",
            "admission": "exactly_one_primary_official_complete_admissible_exact_body",
            "body": "utf8_bytes_and_nfc_identical",
            "fuzzy_reanchoring": False,
            "fixed_incomplete_sources": FIXED_INCOMPLETE_SOURCE_COUNTS,
        },
    }
    return _create_private_json(Path(output), artifact)


def load_v2_candidate_qrel_artifact(path: Path) -> dict[str, Any]:
    """Load and structurally validate a previously created binding artifact."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateQrelError(f"cannot read candidate qrel artifact: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise CandidateQrelError("unsupported candidate qrel artifact schema")
    mappings = value.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != EXPECTED_QUERY_COUNT:
        raise CandidateQrelError("candidate qrel artifact must contain 337 mappings")
    query_ids = [row.get("query_id") for row in mappings if isinstance(row, dict)]
    if len(query_ids) != EXPECTED_QUERY_COUNT or len(set(query_ids)) != EXPECTED_QUERY_COUNT:
        raise CandidateQrelError("candidate qrel artifact query ids are invalid")
    if value.get("frozen_inputs") != {
        "golden_set_sha256": EXPECTED_GOLDEN_SHA256,
        "translations_sha256": EXPECTED_TRANSLATIONS_SHA256,
        "holdout_sha256": EXPECTED_HOLDOUT_SHA256,
    }:
        raise CandidateQrelError("candidate qrel artifact frozen identity mismatch")
    frozen_queries, _hashes = _validate_frozen_inputs(
        goldset.DEFAULT_GOLD_V2,
        goldset.EVAL_DIR / "query_translations_v2.json",
        goldset.DEFAULT_HOLDOUT_V2,
    )
    candidate_snapshot = value.get("candidate_snapshot")
    if not isinstance(candidate_snapshot, dict):
        raise CandidateQrelError("candidate qrel artifact lacks snapshot identity")
    for field_name in ("snapshot_sha256", "corpus_sha256"):
        if not _SHA256_RE.fullmatch(str(candidate_snapshot.get(field_name) or "")):
            raise CandidateQrelError(
                f"candidate qrel artifact has invalid {field_name}"
            )
    if candidate_snapshot.get("snapshot_id") != EXPECTED_SNAPSHOT_ID:
        raise CandidateQrelError("candidate qrel artifact binds the wrong snapshot id")
    allowed_failures = {
        "frozen_incomplete_source_label",
        "quarantined_candidate_document",
        "missing_candidate_document",
        "ambiguous_candidate_document",
        "candidate_not_primary_official",
        "candidate_not_admissible",
        "candidate_body_bytes_changed",
        "candidate_body_nfc_changed",
        "candidate_evidence_span_changed",
    }
    for query, row in zip(frozen_queries, mappings):
        if not isinstance(row, dict) or row.get("status") not in {"mapped", "failure"}:
            raise CandidateQrelError("candidate qrel artifact contains an invalid mapping")
        relevance = query.relevance[0]
        expected_source = relevance.source or query.gold_source
        if (
            row.get("query_id") != query.id
            or row.get("source") != expected_source
            or row.get("document_id") != relevance.document_id
        ):
            raise CandidateQrelError("candidate qrel artifact mapping identity drift")
        for field_name in (
            "frozen_body_sha256",
            "frozen_body_nfc_sha256",
        ):
            if not _SHA256_RE.fullmatch(str(row.get(field_name) or "")):
                raise CandidateQrelError(f"candidate qrel mapping has invalid {field_name}")
        if row["status"] == "mapped":
            if not isinstance(row.get("version_id"), str) or not row["version_id"]:
                raise CandidateQrelError("mapped qrel lacks a candidate version id")
            if row.get("failure_reason") is not None or row.get("zero_score_failure") is not False:
                raise CandidateQrelError("mapped qrel has inconsistent failure state")
            if (
                row.get("candidate_body_sha256") != row.get("frozen_body_sha256")
                or row.get("candidate_body_nfc_sha256")
                != row.get("frozen_body_nfc_sha256")
            ):
                raise CandidateQrelError("mapped qrel body identities differ")
            inventory = row.get("candidate_inventory")
            if not isinstance(inventory, list):
                raise CandidateQrelError("mapped qrel lacks candidate inventory")
            exact = [
                item
                for item in inventory
                if isinstance(item, dict)
                and item.get("version_id") == row["version_id"]
                and item.get("source_authority") == "primary_official"
                and item.get("content_complete") is True
                and item.get("admissible") is True
                and item.get("body_sha256") == row["frozen_body_sha256"]
                and item.get("body_nfc_sha256") == row["frozen_body_nfc_sha256"]
            ]
            if len(exact) != 1:
                raise CandidateQrelError(
                    "mapped qrel does not identify exactly one admissible exact body"
                )
        else:
            if (
                row.get("failure_reason") not in allowed_failures
                or row.get("zero_score_failure") is not True
            ):
                raise CandidateQrelError("failed qrel has inconsistent failure state")
            if expected_source in FIXED_INCOMPLETE_SOURCE_COUNTS and row.get(
                "failure_reason"
            ) != "frozen_incomplete_source_label":
                raise CandidateQrelError(
                    "frozen incomplete-source mapping is not a fixed zero failure"
                )
    policy = value.get("mapping_policy")
    if not isinstance(policy, dict) or policy != {
        "identity": "exact_source_document",
        "admission": "exactly_one_primary_official_complete_admissible_exact_body",
        "body": "utf8_bytes_and_nfc_identical",
        "fuzzy_reanchoring": False,
        "fixed_incomplete_sources": FIXED_INCOMPLETE_SOURCE_COUNTS,
    }:
        raise CandidateQrelError("candidate qrel artifact mapping policy mismatch")
    expected_order_hash = hashlib.sha256(
        _canonical_json_bytes([row["query_id"] for row in mappings])
    ).hexdigest()
    if value.get("ordered_query_ids_sha256") != expected_order_hash:
        raise CandidateQrelError("candidate qrel artifact ordered query identity mismatch")
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise CandidateQrelError("candidate qrel artifact lacks mapping counts")
    mapped = sum(row["status"] == "mapped" for row in mappings)
    failed = len(mappings) - mapped
    if counts.get("mapped") != mapped or counts.get("failures") != failed:
        raise CandidateQrelError("candidate qrel artifact mapping counts do not reconcile")
    expected_failure_counts = dict(
        sorted(
            Counter(
                row["failure_reason"]
                for row in mappings
                if row["status"] == "failure"
            ).items()
        )
    )
    expected_source_counts = dict(
        sorted(Counter(query.gold_source for query in frozen_queries).items())
    )
    if counts.get("by_failure_reason") != expected_failure_counts:
        raise CandidateQrelError("candidate qrel failure counts do not reconcile")
    if counts.get("by_source") != expected_source_counts:
        raise CandidateQrelError("candidate qrel source counts do not reconcile")
    return value


def bind_gold_queries_to_candidate(
    queries: Sequence[goldset.GoldQuery], artifact: Mapping[str, Any]
) -> tuple[list[goldset.GoldQuery], dict[str, str]]:
    """Apply exact candidate version IDs and return explicit per-query failures."""

    rows = {row["query_id"]: row for row in artifact["mappings"]}
    if [query.id for query in queries] != [row["query_id"] for row in artifact["mappings"]]:
        raise CandidateQrelError("candidate qrel order differs from frozen v2 query order")
    bound: list[goldset.GoldQuery] = []
    failures: dict[str, str] = {}
    for query in queries:
        row = rows[query.id]
        if row["status"] == "failure":
            failures[query.id] = str(row["failure_reason"])
            bound.append(query)
            continue
        version_id = str(row["version_id"])
        relevance = [
            replace(item, source=item.source or query.gold_source, version_id=version_id)
            for item in query.relevance
        ]
        bound.append(replace(query, relevance=relevance, gold_version_id=version_id))
    return bound, failures


def artifact_sha256(path: Path) -> str:
    """Return the exact full-file identity used by evaluation provenance."""

    return _sha256_file(Path(path))


def main(argv: Sequence[str] | None = None) -> int:
    """Create an immutable binding from the frozen v2 labels to one sealed snapshot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden-path", type=Path, default=goldset.DEFAULT_GOLD_V2)
    parser.add_argument(
        "--translations-path",
        type=Path,
        default=goldset.EVAL_DIR / "query_translations_v2.json",
    )
    parser.add_argument("--holdout-path", type=Path, default=goldset.DEFAULT_HOLDOUT_V2)
    args = parser.parse_args(argv)
    destination = build_v2_candidate_qrel_artifact(
        candidate_snapshot=args.candidate_snapshot,
        output=args.output,
        golden_path=args.golden_path,
        translations_path=args.translations_path,
        holdout_path=args.holdout_path,
    )
    print(f"created {destination} sha256={artifact_sha256(destination)}")
    return 0


__all__ = [
    "CandidateQrelError",
    "EXPECTED_GOLDEN_SHA256",
    "EXPECTED_HOLDOUT_SHA256",
    "EXPECTED_QUERY_COUNT",
    "EXPECTED_SNAPSHOT_ID",
    "EXPECTED_TRANSLATIONS_SHA256",
    "FIXED_INCOMPLETE_SOURCE_COUNTS",
    "SCHEMA_VERSION",
    "artifact_sha256",
    "bind_gold_queries_to_candidate",
    "build_v2_candidate_qrel_artifact",
    "load_v2_candidate_qrel_artifact",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the release command
    raise SystemExit(main())
