"""Concrete, fail-closed Qdrant adapter for the reversible promotion state machine.

The adapter owns only Qdrant mechanics.  It deliberately requires injected semantic
smoke and readiness checks: exact counts and payload identity cannot prove legal-search
quality, so storage compatibility alone is never sufficient to switch the serving alias.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from qdrant_client import models

from .artifacts import atomic_write_json
from .generation import MANIFEST_FILENAME, GenerationArtifacts, load_generation
from .integrity import verify_generation_artifacts, write_verification_report
from .qdrant_store import collection_configuration, collection_configuration_sha256
from .promotion import (
    CollectionInspection,
    PromotionPlan,
    PromotionPreconditionError,
    _candidate_mismatches,
    promotion_plan_sha256,
    refuse_frozen_candidate_promotion,
)

Check = Callable[[str, Any], bool]
CHECKS_FACTORY_ENV = "PROMOTION_CHECKS_FACTORY"


@dataclass(frozen=True, slots=True)
class CandidateVerificationProof:
    """Durable binding between a restored collection, its plan, and its report."""

    ok: bool
    report_path: Path
    report_sha256: str
    provenance_path: Path
    provenance_sha256: str


IntegrityCheck = Callable[[PromotionPlan, Any], CandidateVerificationProof]


def _value(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _normal(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").rsplit(".", maxsplit=1)[-1].lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_snapshot_location(plan: PromotionPlan) -> str:
    """Validate a secret-free immutable snapshot reference before recovery."""
    parsed = urlparse(plan.snapshot_ref)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise PromotionPreconditionError("snapshot file URL is not local and canonical")
        path = Path(unquote(parsed.path))
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise PromotionPreconditionError(
                f"cannot stat promotion snapshot: {type(exc).__name__}"
            ) from exc
        if not path.is_absolute() or stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise PromotionPreconditionError(
                "promotion snapshot must be an absolute, regular, non-symlink file"
            )
        if _sha256_file(path) != plan.snapshot_sha256:
            raise PromotionPreconditionError("promotion snapshot SHA-256 mismatch")
        return path.as_uri()
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise PromotionPreconditionError(
            "remote snapshot must be a credential-free immutable HTTPS URL"
        )
    return plan.snapshot_ref


def _stream_points(
    client: Any,
    collection: str,
    *,
    page_size: int = 256,
) -> Iterator[Any]:
    """Stream a physical collection without materializing it in memory."""
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=page_size,
            with_payload=True,
            with_vectors=True,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise PromotionPreconditionError("Qdrant scroll returned an invalid page")
        points, next_offset = result
        if points is None:
            raise PromotionPreconditionError("Qdrant scroll returned no point iterable")
        yielded = 0
        for point in points:
            yielded += 1
            yield point
        if next_offset is None:
            return
        if not yielded or next_offset == offset:
            raise PromotionPreconditionError("Qdrant scroll continuation did not advance")
        offset = next_offset


def _generation_integrity_check(artifacts: GenerationArtifacts) -> IntegrityCheck:
    """Bind a streamed post-restore verifier to one immutable generation."""
    manifest_sha256 = artifacts.checksums.files[MANIFEST_FILENAME]

    def check(plan: PromotionPlan, client: Any) -> CandidateVerificationProof:
        if (
            artifacts.manifest.generation_id != plan.generation_id
            or manifest_sha256 != plan.manifest_sha256
        ):
            raise PromotionPreconditionError(
                "GENERATION_DIR does not match the immutable promotion plan"
            )
        report = verify_generation_artifacts(
            artifacts,
            _stream_points(client, plan.physical_collection),
            physical_collection=plan.physical_collection,
            observed_collection_configuration_sha256=collection_configuration_sha256(
                collection_configuration(client.get_collection(plan.physical_collection))
            ),
            verification_id=plan.promotion_id,
        )
        report_path = artifacts.root.parent / (
            f"{plan.generation_id}.{plan.promotion_id}.candidate-verification.json"
        )
        write_verification_report(report_path, report)
        report_sha256 = _sha256_file(report_path)
        report_matches = (
            report.generation_id == plan.generation_id
            and report.manifest_sha256 == plan.manifest_sha256
            and report.physical_collection == plan.physical_collection
        )
        provenance_path = report_path.with_name(f"{report_path.stem}.provenance.json")
        atomic_write_json(
            provenance_path,
            {
                "schema_version": 1,
                "promotion_id": plan.promotion_id,
                "plan_sha256": promotion_plan_sha256(plan),
                "generation_id": plan.generation_id,
                "generation_manifest_sha256": plan.manifest_sha256,
                "physical_collection": plan.physical_collection,
                "snapshot_sha256": plan.snapshot_sha256,
                "verification_report": report_path.name,
                "verification_report_sha256": report_sha256,
                "verification_report_ok": report.ok,
                "verification_report_identity_matches": report_matches,
            },
        )
        return CandidateVerificationProof(
            ok=report.ok and report_matches,
            report_path=report_path,
            report_sha256=report_sha256,
            provenance_path=provenance_path,
            provenance_sha256=_sha256_file(provenance_path),
        )

    return check


class QdrantPromotionBackend:
    """No-delete Qdrant implementation with exact candidate inspection."""

    def __init__(
        self,
        client: Any,
        *,
        integrity_check: IntegrityCheck | None = None,
        smoke_check: Check | None = None,
        readiness_check: Check | None = None,
        poll_interval_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or poll_interval_seconds < 0
        ):
            raise ValueError("poll_interval_seconds must be a non-negative number")
        self.client = client
        self.integrity_check = integrity_check
        self.smoke_check = smoke_check
        self.readiness_check = readiness_check
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.sleep = sleep
        self._plans: dict[str, PromotionPlan] = {}
        self._integrity_verified: set[str] = set()
        self._verification_proofs: dict[str, CandidateVerificationProof] = {}

    def restore_candidate(self, plan: PromotionPlan) -> None:
        if not isinstance(plan, PromotionPlan):
            raise TypeError("plan must be a PromotionPlan")
        # Dataclass construction can bypass PromotionPlan.from_dict; repeat the guard at
        # the final restore boundary before the client is inspected or mutated.
        refuse_frozen_candidate_promotion(
            plan.generation_id, plan.physical_collection
        )
        collection = plan.physical_collection
        self._plans[collection] = plan
        if self.client.collection_exists(collection):
            inspection = self._inspect(plan)
            if inspection.optimizer_status != "green":
                # A process may have exited after Qdrant accepted recovery.  Never submit
                # a second in-place recovery; the bounded green wait will inspect the
                # finished collection exactly before any check or alias operation.
                return
            mismatches = [
                mismatch
                for mismatch in _candidate_mismatches(plan, inspection)
                if not mismatch.startswith(("manifest_sha256:", "integrity_ok:"))
            ]
            if mismatches:
                raise PromotionPreconditionError(
                    "existing candidate is incompatible and will not be overwritten: "
                    + "; ".join(mismatches[:20])
                )
            return

        location = _verified_snapshot_location(plan)
        restored = self.client.recover_snapshot(
            collection_name=collection,
            location=location,
            checksum=plan.snapshot_sha256,
            priority=models.SnapshotPriority.SNAPSHOT,
            wait=True,
        )
        if restored is not True or not self.client.collection_exists(collection):
            raise PromotionPreconditionError("Qdrant did not confirm candidate recovery")

    @staticmethod
    def _expected_payload(plan: PromotionPlan) -> dict[str, Any]:
        expected = plan.expected_collection
        return {
            "schema_version": expected.payload_schema_version,
            "generation_id": plan.generation_id,
            "embedding_model": expected.embedding_model,
            "embedding_revision": expected.embedding_revision,
            "tokenizer_model": expected.tokenizer_model,
            "tokenizer_revision": expected.tokenizer_revision,
            "reranker_model": expected.reranker_model,
            "reranker_revision": expected.reranker_revision,
            "vector_space_id": expected.vector_space_id,
            "chunking_fingerprint": expected.chunking_fingerprint,
            "document_header": expected.document_header,
            "retrieval_fingerprint_revision": expected.retrieval_fingerprint_revision,
            "retrieval_fingerprint": expected.retrieval_fingerprint,
        }

    def _identity_count(self, plan: PromotionPlan) -> int | None:
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in self._expected_payload(plan).items()
        ]
        result = self.client.count(
            collection_name=plan.physical_collection,
            count_filter=models.Filter(must=conditions),
            exact=True,
        )
        count = _value(result, "count")
        return count if isinstance(count, int) and not isinstance(count, bool) else None

    def _inspect(self, plan: PromotionPlan) -> CollectionInspection:
        info = self.client.get_collection(plan.physical_collection)
        expected = plan.expected_collection
        config = _value(info, "config")
        params = _value(config, "params")
        vectors = _value(params, "vectors")
        sparse_vectors = _value(params, "sparse_vectors")
        dense = vectors.get(expected.dense_name) if isinstance(vectors, Mapping) else None
        dense_names = set(vectors) if isinstance(vectors, Mapping) else set()
        sparse_names = set(sparse_vectors) if isinstance(sparse_vectors, Mapping) else set()
        points_count = _value(info, "points_count")
        identity_count = self._identity_count(plan)
        identity_ok = (
            type(points_count) is int
            and points_count == expected.points_count
            and identity_count == expected.points_count
            and dense_names == {expected.dense_name}
            and sparse_names == {expected.sparse_name}
        )
        integrity_ok = (
            identity_ok and plan.physical_collection in self._integrity_verified
        )
        proof = self._verification_proofs.get(plan.physical_collection)
        status = _normal(_value(info, "status"))
        optimizer = _normal(_value(info, "optimizer_status"))
        optimizer_status = "green" if status == "green" and optimizer == "ok" else (
            f"{status or 'unknown'}/{optimizer or 'unknown'}"
        )
        mismatch = "<identity-mismatch>"
        return CollectionInspection(
            name=plan.physical_collection,
            payload_schema_version=(
                expected.payload_schema_version if identity_ok else -1
            ),
            points_count=points_count if type(points_count) is int else -1,
            dense_name=(expected.dense_name if dense_names == {expected.dense_name} else mismatch),
            dense_dimension=_value(dense, "size") if dense is not None else -1,
            distance=_normal(_value(dense, "distance")),
            sparse_name=(
                expected.sparse_name if sparse_names == {expected.sparse_name} else mismatch
            ),
            generation_id=plan.generation_id if identity_ok else mismatch,
            manifest_sha256=plan.manifest_sha256 if integrity_ok else mismatch,
            embedding_model=expected.embedding_model if identity_ok else mismatch,
            embedding_revision=expected.embedding_revision if identity_ok else mismatch,
            tokenizer_model=expected.tokenizer_model if identity_ok else mismatch,
            tokenizer_revision=expected.tokenizer_revision if identity_ok else mismatch,
            reranker_model=expected.reranker_model if identity_ok else mismatch,
            reranker_revision=expected.reranker_revision if identity_ok else mismatch,
            vector_space_id=expected.vector_space_id if identity_ok else mismatch,
            chunking_fingerprint=(
                expected.chunking_fingerprint if identity_ok else mismatch
            ),
            document_header=(expected.document_header if identity_ok else not expected.document_header),
            retrieval_fingerprint_revision=(
                expected.retrieval_fingerprint_revision if identity_ok else -1
            ),
            retrieval_fingerprint=(
                expected.retrieval_fingerprint if identity_ok else mismatch
            ),
            optimizer_status=optimizer_status,
            integrity_ok=integrity_ok,
            verification_provenance_sha256=(
                proof.provenance_sha256 if integrity_ok and proof is not None else None
            ),
        )

    def wait_for_green(
        self, collection: str, timeout_seconds: float
    ) -> CollectionInspection:
        plan = self._plans.get(collection)
        if plan is None:
            raise PromotionPreconditionError("candidate has no bound promotion plan")
        deadline = time.monotonic() + timeout_seconds
        while True:
            inspection = self._inspect(plan)
            if inspection.optimizer_status == "green":
                storage_mismatches = [
                    mismatch
                    for mismatch in _candidate_mismatches(plan, inspection)
                    if not mismatch.startswith(("manifest_sha256:", "integrity_ok:"))
                ]
                if storage_mismatches:
                    raise PromotionPreconditionError(
                        "candidate storage compatibility failed: "
                        + "; ".join(storage_mismatches[:20])
                    )
                if self.integrity_check is None:
                    raise PromotionPreconditionError(
                        "candidate streamed integrity check is not configured"
                    )
                try:
                    proof = self.integrity_check(plan, self.client)
                except Exception as exc:
                    raise PromotionPreconditionError(
                        "candidate streamed integrity check failed: "
                        f"{type(exc).__name__}"
                    ) from exc
                if not isinstance(proof, CandidateVerificationProof) or proof.ok is not True:
                    raise PromotionPreconditionError(
                        "candidate streamed integrity check failed"
                    )
                for path, expected_sha256 in (
                    (proof.report_path, proof.report_sha256),
                    (proof.provenance_path, proof.provenance_sha256),
                ):
                    try:
                        mode = path.lstat().st_mode
                    except OSError as exc:
                        raise PromotionPreconditionError(
                            "candidate verification proof is unavailable"
                        ) from exc
                    if (
                        not stat.S_ISREG(mode)
                        or stat.S_IMODE(mode) != 0o600
                        or _sha256_file(path) != expected_sha256
                    ):
                        raise PromotionPreconditionError(
                            "candidate verification proof is not an intact owner-only file"
                        )
                self._verification_proofs[plan.physical_collection] = proof
                self._integrity_verified.add(plan.physical_collection)
                return self._inspect(plan)
            if time.monotonic() >= deadline:
                raise PromotionPreconditionError("candidate optimizer did not become green")
            self.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))

    def smoke(self, collection: str) -> bool:
        return bool(
            self.smoke_check
            and self.smoke_check(collection, self.client) is True
        )

    def readiness(self, collection: str) -> bool:
        if (
            self.readiness_check is None
            or self.readiness_check(collection, self.client) is not True
        ):
            return False
        plan = self._plans.get(collection)
        if plan is None:
            return True
        return not _candidate_mismatches(plan, self._inspect(plan))

    def alias_target(self, alias: str) -> str | None:
        response = self.client.get_aliases()
        aliases = _value(response, "aliases")
        if not isinstance(aliases, list):
            raise PromotionPreconditionError("Qdrant alias inventory is unavailable")
        targets = [
            _value(item, "collection_name")
            for item in aliases
            if _value(item, "alias_name") == alias
        ]
        if len(targets) > 1:
            raise PromotionPreconditionError("serving alias has multiple targets")
        return targets[0] if targets else None

    def switch_alias(self, alias: str, collection: str) -> None:
        # ``PromotionPlan`` validation protects the state-machine path, but callers can
        # invoke this public adapter method directly.  The frozen candidate is explicitly
        # retrieval-only and must never become reachable through any alias, so refuse its
        # unique physical name before even reading alias state from Qdrant.
        from .release_inputs import PHYSICAL_COLLECTION as FROZEN_PHYSICAL_COLLECTION

        if collection == FROZEN_PHYSICAL_COLLECTION:
            raise PromotionPreconditionError(
                "the frozen 512-token candidate cannot be targeted by an alias operation"
            )
        current = self.alias_target(alias)
        if current is None:
            raise PromotionPreconditionError(
                "serving alias does not exist; maintenance migration is required"
            )
        if current == collection:
            return
        if not self.client.collection_exists(collection):
            raise PromotionPreconditionError("alias target collection does not exist")
        operations = [
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias)
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection,
                    alias_name=alias,
                )
            ),
        ]
        if self.client.update_collection_aliases(
            change_aliases_operations=operations
        ) is not True:
            raise PromotionPreconditionError("Qdrant did not confirm atomic alias switch")


def make_qdrant_promotion_backend() -> QdrantPromotionBackend:
    """CLI factory that refuses to exist without deployment-specific quality checks.

    ``PROMOTION_CHECKS_FACTORY=package.module:callable`` must return a mapping/object
    containing callable ``smoke`` and ``readiness`` fields with signature
    ``(collection_name, qdrant_client) -> bool``.
    """
    specification = os.getenv(CHECKS_FACTORY_ENV, "")
    try:
        module_name, attribute = specification.split(":", 1)
        checks_factory = getattr(importlib.import_module(module_name), attribute)
        checks = checks_factory()
        smoke = _value(checks, "smoke")
        readiness = _value(checks, "readiness")
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        raise PromotionPreconditionError(
            f"{CHECKS_FACTORY_ENV} must name a callable returning smoke/readiness checks"
        ) from exc
    if not callable(smoke) or not callable(readiness):
        raise PromotionPreconditionError(
            f"{CHECKS_FACTORY_ENV} did not provide callable smoke/readiness checks"
        )

    from .config import load_config
    from .qdrant_store import make_client

    config = load_config()
    if config.generation_dir is None:
        raise PromotionPreconditionError(
            "GENERATION_DIR is required for post-restore integrity verification"
        )
    try:
        artifacts = load_generation(config.generation_dir)
    except Exception as exc:
        raise PromotionPreconditionError(
            f"cannot load GENERATION_DIR for promotion: {type(exc).__name__}"
        ) from exc
    client = make_client(config)
    return QdrantPromotionBackend(
        client,
        integrity_check=_generation_integrity_check(artifacts),
        smoke_check=smoke,
        readiness_check=readiness,
    )


__all__ = [
    "CandidateVerificationProof",
    "CHECKS_FACTORY_ENV",
    "QdrantPromotionBackend",
    "make_qdrant_promotion_backend",
]
