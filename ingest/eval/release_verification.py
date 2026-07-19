"""Strict loading for the two direct-physical candidate verification sidecars."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingest.artifacts import parse_utc_datetime
from ingest.generation import GENERATION_SCHEMA_VERSION, load_generation
from ingest.integrity import load_generation_structural_provenance
from ingest.release_inputs import GENERATION_ID, PHYSICAL_COLLECTION

VERIFICATION_IDS = ("physical-v3-512-01", "physical-v3-512-02")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_KEYS = {
    "schema_version",
    "generation_id",
    "manifest_sha256",
    "physical_collection",
    "verified_at",
    "verification_id",
    "ok",
    "covered_runs",
    "stats",
    "expected_collection_sha256",
    "observed_collection_sha256",
    "expected_collection_configuration_sha256",
    "observed_collection_configuration_sha256",
    "vector_checksum_artifact_sha256",
    "vector_probe_sha256",
    "coverage",
    "integrity",
    "freshness",
    "quality",
}


class ReleaseVerificationError(ValueError):
    """A physical verification sidecar is absent, drifted, or replaceable."""


@dataclass(frozen=True, slots=True)
class ValidatedVerificationPair:
    paths: tuple[Path, Path]
    sha256: tuple[str, str]
    reports: tuple[dict[str, Any], dict[str, Any]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReleaseVerificationError(
                f"verification sidecar contains duplicate key {key!r}"
            )
        value[key] = item
    return value


def _load_report(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReleaseVerificationError(f"cannot inspect verification sidecar: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReleaseVerificationError("verification sidecar must be a real regular file")
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
        raise ReleaseVerificationError(
            "verification sidecar must be owner-only and owned by this operator"
        )
    if info.st_size < 2 or info.st_size > 16 * 1024 * 1024:
        raise ReleaseVerificationError("verification sidecar size is invalid")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReleaseVerificationError(
                    f"verification sidecar contains non-finite number {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(
            f"cannot read verification sidecar: {exc}"
        ) from exc
    if not isinstance(value, dict) or set(value) != _REPORT_KEYS:
        raise ReleaseVerificationError("verification sidecar has an invalid shape")
    return value


def validate_physical_verification_pair(
    generation_dir: str | Path,
    paths: tuple[str | Path, str | Path],
) -> ValidatedVerificationPair:
    """Revalidate both create-only reports against one sealed generation."""

    artifacts = load_generation(generation_dir)
    manifest = artifacts.manifest
    digest = artifacts.collection_digest
    if manifest.generation_id != GENERATION_ID or digest is None:
        raise ReleaseVerificationError(
            "physical verification pair is not for the sealed 512 candidate"
        )
    manifest_path = artifacts.root / "manifest.json"
    manifest_sha = _file_sha256(manifest_path)
    generation_structural = load_generation_structural_provenance(artifacts)
    structural_inventory = generation_structural["structural_chunk_inventory"]
    expected_covered_runs = [
        {"source": run.source, "run_id": run.run_id} for run in manifest.covered_runs
    ]
    resolved_paths: list[Path] = []
    hashes: list[str] = []
    reports: list[dict[str, Any]] = []
    structural_proof: dict[str, Any] | None = None
    for verification_id, raw_path in zip(VERIFICATION_IDS, paths, strict=True):
        path = Path(raw_path).expanduser().absolute()
        expected_path = artifacts.root.with_name(
            f"{GENERATION_ID}.{verification_id}.verification.json"
        ).absolute()
        if path != expected_path:
            raise ReleaseVerificationError(
                "verification sidecar path is not the canonical run-specific path"
            )
        report = _load_report(path)
        if (
            report["schema_version"] != GENERATION_SCHEMA_VERSION
            or report["generation_id"] != GENERATION_ID
            or report["physical_collection"] != PHYSICAL_COLLECTION
            or report["verification_id"] != verification_id
            or report["manifest_sha256"] != manifest_sha
            or report["covered_runs"] != expected_covered_runs
            or report["ok"] is not True
            or not isinstance(report["stats"], dict)
        ):
            raise ReleaseVerificationError(
                "verification sidecar generation identity does not reconcile"
            )
        try:
            parse_utc_datetime(report["verified_at"])
        except (TypeError, ValueError) as exc:
            raise ReleaseVerificationError(
                "verification sidecar timestamp is invalid"
            ) from exc
        expected_digests = {
            "expected_collection_sha256": digest.collection_sha256,
            "observed_collection_sha256": digest.collection_sha256,
            "expected_collection_configuration_sha256": (
                digest.collection_configuration_sha256
            ),
            "observed_collection_configuration_sha256": (
                digest.collection_configuration_sha256
            ),
            "vector_checksum_artifact_sha256": (
                digest.vector_checksum_artifact_sha256
            ),
            "vector_probe_sha256": digest.vector_probe_sha256,
        }
        if any(report[name] != expected for name, expected in expected_digests.items()):
            raise ReleaseVerificationError(
                "verification sidecar collection/vector digest does not reconcile"
            )
        for gate in ("coverage", "integrity", "freshness", "quality"):
            outcome = report[gate]
            if outcome != {"ok": True, "issue_count": 0, "examples": []}:
                raise ReleaseVerificationError(
                    f"verification sidecar gate {gate} is not exactly clean"
                )
        proof = report["stats"].get("structural_inventory_proof")
        if not isinstance(proof, dict):
            raise ReleaseVerificationError(
                "verification sidecar lacks the exact structural inventory join proof"
            )
        expected_proof_fields = {
            "snapshot_id": manifest.corpus.name,
            "snapshot_sha256": manifest.corpus.snapshot_sha256,
            "corpus_sha256": generation_structural["corpus_sha256"],
            "structural_inventory_sha256": structural_inventory["sha256"],
            "structural_inventory_size_bytes": structural_inventory["size_bytes"],
            "structural_inventory_identity_sha256": structural_inventory[
                "identity_sha256"
            ],
            "structural_inventory_record_count": structural_inventory["record_count"],
            "expected_document_count": manifest.indexed_document_count,
            "expected_chunk_count": manifest.chunk_count,
            "artifact_exhausted": True,
            "document_rows": manifest.indexed_document_count,
            "chunk_rows": manifest.chunk_count,
            "point_rows_found": manifest.chunk_count,
            "exact_point_matches": manifest.chunk_count,
            "embed_inputs_verified": manifest.chunk_count,
        }
        if any(proof.get(name) != expected for name, expected in expected_proof_fields.items()):
            raise ReleaseVerificationError(
                "verification sidecar structural inventory join is incomplete"
            )
        if structural_proof is None:
            structural_proof = proof
        elif proof != structural_proof:
            raise ReleaseVerificationError(
                "independent verification sidecars bind different structural proofs"
            )
        report_sha = _file_sha256(path)
        if not _SHA256.fullmatch(report_sha):  # pragma: no cover - hashlib invariant
            raise ReleaseVerificationError("verification sidecar hash is invalid")
        resolved_paths.append(path)
        hashes.append(report_sha)
        reports.append(report)
    if hashes[0] == hashes[1]:
        raise ReleaseVerificationError(
            "independent verification sidecars must have distinct artifact hashes"
        )
    return ValidatedVerificationPair(
        paths=(resolved_paths[0], resolved_paths[1]),
        sha256=(hashes[0], hashes[1]),
        reports=(reports[0], reports[1]),
    )


__all__ = [
    "ReleaseVerificationError",
    "ValidatedVerificationPair",
    "VERIFICATION_IDS",
    "validate_physical_verification_pair",
]
