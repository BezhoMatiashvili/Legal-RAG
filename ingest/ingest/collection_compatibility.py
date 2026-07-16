"""Read-only collection identity checks shared by health and restore callers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from qdrant_client import models

from .config import (
    Config,
    RETRIEVAL_FINGERPRINT_REVISION,
    retrieval_fingerprint_sha256,
)
from .generation import (
    CANONICAL_PAYLOAD_REQUIRED_NONEMPTY_FIELDS,
    CANONICAL_PAYLOAD_REVISION,
    GenerationManifest,
)
from .qdrant_store import chunking_fingerprint, vector_space_id

CompatibilityGate = Literal["coverage", "integrity"]


@dataclass(frozen=True)
class CompatibilityIssue:
    gate: CompatibilityGate
    code: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "code": self.code, **dict(self.details)}


@dataclass(frozen=True)
class CollectionCompatibility:
    collection_name: str
    generation_id: str
    expected_points: int
    points_count: int | None
    identity_matched_points: int | None
    issues: tuple[CompatibilityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def coverage_issues(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.gate == "coverage")

    @property
    def integrity_issues(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.gate == "integrity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "collection_name": self.collection_name,
            "generation_id": self.generation_id,
            "expected_points": self.expected_points,
            "points_count": self.points_count,
            "identity_matched_points": self.identity_matched_points,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class CollectionIncompatibleError(RuntimeError):
    """A collection cannot safely serve the requested immutable generation."""


def config_manifest_issues(
    cfg: Config, manifest: GenerationManifest
) -> tuple[CompatibilityIssue, ...]:
    """Compare every serving identity with its immutable generation manifest."""
    expected_actual = (
        ("generation_id", manifest.generation_id, cfg.generation_id),
        ("embedding_model", manifest.model.embedding_model, cfg.embed_model),
        (
            "embedding_revision",
            manifest.model.embedding_revision,
            cfg.embedding_revision,
        ),
        (
            "tokenizer_model",
            manifest.model.tokenizer_model,
            cfg.tokenizer_model,
        ),
        (
            "tokenizer_revision",
            manifest.model.tokenizer_revision,
            cfg.tokenizer_revision,
        ),
        ("reranker_model", manifest.model.reranker_model, cfg.rerank_model),
        (
            "reranker_revision",
            manifest.model.reranker_revision,
            cfg.reranker_revision,
        ),
        (
            "dense_dimension",
            manifest.vector_space.dense_dimension,
            cfg.dense_dim,
        ),
        ("vector_space_id", manifest.vector_space.id, vector_space_id(cfg)),
        (
            "chunking_fingerprint",
            manifest.chunking.fingerprint,
            chunking_fingerprint(cfg),
        ),
        (
            "document_header",
            manifest.chunking.document_header,
            cfg.embed_header_v2,
        ),
        (
            "retrieval_fingerprint_revision",
            manifest.retrieval_fingerprint_revision,
            RETRIEVAL_FINGERPRINT_REVISION,
        ),
        (
            "retrieval_fingerprint",
            manifest.retrieval_fingerprint,
            retrieval_fingerprint_sha256(cfg),
        ),
    )
    return tuple(
        CompatibilityIssue(
            "integrity",
            "runtime_manifest_identity_mismatch",
            {"field": field, "expected": expected, "actual": actual},
        )
        for field, expected, actual in expected_actual
        if expected != actual
    )


def require_config_manifest_compatibility(
    cfg: Config, manifest: GenerationManifest
) -> None:
    issues = config_manifest_issues(cfg, manifest)
    if issues:
        fields = ", ".join(str(issue.details["field"]) for issue in issues)
        raise CollectionIncompatibleError(
            f"runtime configuration is incompatible with generation "
            f"{manifest.generation_id!r}: {fields}"
        )


def expected_point_identity(manifest: GenerationManifest) -> dict[str, Any]:
    """Return every payload identity that all points in a generation must share."""
    if not isinstance(manifest, GenerationManifest):
        raise TypeError("manifest must be a GenerationManifest")
    manifest = GenerationManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    return {
        "schema_version": manifest.schema_version,
        "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        "canonical_text_exact": True,
        "content_complete": True,
        "generation_id": manifest.generation_id,
        "embedding_model": manifest.model.embedding_model,
        "embedding_revision": manifest.model.embedding_revision,
        "tokenizer_model": manifest.model.tokenizer_model,
        "tokenizer_revision": manifest.model.tokenizer_revision,
        "reranker_model": manifest.model.reranker_model,
        "reranker_revision": manifest.model.reranker_revision,
        "vector_space_id": manifest.vector_space.id,
        "chunking_fingerprint": manifest.chunking.fingerprint,
        "document_header": manifest.chunking.document_header,
        "retrieval_fingerprint_revision": manifest.retrieval_fingerprint_revision,
        "retrieval_fingerprint": manifest.retrieval_fingerprint,
    }


def _value(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _distance(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).rsplit(".", maxsplit=1)[-1].lower()


def _schema_issues(info: Any, manifest: GenerationManifest) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    config = _value(info, "config")
    params = _value(config, "params")
    vectors = _value(params, "vectors")
    sparse_vectors = _value(params, "sparse_vectors")

    expected_dense = manifest.vector_space.dense_name
    if not isinstance(vectors, Mapping):
        issues.append(
            CompatibilityIssue(
                "integrity",
                "named_dense_schema_missing",
                {"expected": expected_dense},
            )
        )
    else:
        actual_names = {str(name) for name in vectors}
        if actual_names != {expected_dense}:
            issues.append(
                CompatibilityIssue(
                    "integrity",
                    "dense_vector_names_mismatch",
                    {
                        "expected": [expected_dense],
                        "actual": sorted(actual_names),
                    },
                )
            )
        dense = vectors.get(expected_dense)
        if dense is None:
            issues.append(
                CompatibilityIssue(
                    "integrity",
                    "dense_vector_missing",
                    {"expected": expected_dense},
                )
            )
        else:
            actual_size = _value(dense, "size")
            if actual_size != manifest.vector_space.dense_dimension:
                issues.append(
                    CompatibilityIssue(
                        "integrity",
                        "dense_dimension_mismatch",
                        {
                            "expected": manifest.vector_space.dense_dimension,
                            "actual": actual_size,
                        },
                    )
                )
            actual_distance = _distance(_value(dense, "distance"))
            if actual_distance != manifest.vector_space.distance:
                issues.append(
                    CompatibilityIssue(
                        "integrity",
                        "dense_distance_mismatch",
                        {
                            "expected": manifest.vector_space.distance,
                            "actual": actual_distance,
                        },
                    )
                )

    expected_sparse = manifest.vector_space.sparse_name
    if not isinstance(sparse_vectors, Mapping):
        issues.append(
            CompatibilityIssue(
                "integrity",
                "named_sparse_schema_missing",
                {"expected": expected_sparse},
            )
        )
    else:
        actual_names = {str(name) for name in sparse_vectors}
        if actual_names != {expected_sparse}:
            issues.append(
                CompatibilityIssue(
                    "integrity",
                    "sparse_vector_names_mismatch",
                    {
                        "expected": [expected_sparse],
                        "actual": sorted(actual_names),
                    },
                )
            )
    return issues


def _identity_filter(manifest: GenerationManifest) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in expected_point_identity(manifest).items()
        ],
        must_not=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key=field))
            for field in sorted(CANONICAL_PAYLOAD_REQUIRED_NONEMPTY_FIELDS)
        ],
    )


def check_collection_compatibility(
    client: Any,
    collection_name: str,
    manifest: GenerationManifest,
) -> CollectionCompatibility:
    """Check an existing collection using only read-only Qdrant operations.

    The exact identity count proves every point carries the requested immutable
    generation/model/vector/chunk/retrieval identity when the collection count
    also equals the manifest count.
    """
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("collection_name must be a non-empty string")
    if not isinstance(manifest, GenerationManifest):
        raise TypeError("manifest must be a GenerationManifest")
    manifest = GenerationManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    issues: list[CompatibilityIssue] = []
    points_count: int | None = None
    identity_matched_points: int | None = None

    try:
        info = client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001 - health must report unavailable, not pass open
        issues.append(
            CompatibilityIssue(
                "integrity",
                "collection_info_unavailable",
                {"error_type": type(exc).__name__},
            )
        )
        return CollectionCompatibility(
            collection_name=collection_name,
            generation_id=manifest.generation_id,
            expected_points=manifest.chunk_count,
            points_count=None,
            identity_matched_points=None,
            issues=tuple(issues),
        )

    raw_points_count = _value(info, "points_count")
    if (
        isinstance(raw_points_count, bool)
        or not isinstance(raw_points_count, int)
        or raw_points_count < 0
    ):
        issues.append(
            CompatibilityIssue(
                "coverage",
                "invalid_collection_point_count",
                {"actual": raw_points_count},
            )
        )
    else:
        points_count = raw_points_count
        if points_count != manifest.chunk_count:
            issues.append(
                CompatibilityIssue(
                    "coverage",
                    "collection_point_count_mismatch",
                    {
                        "expected": manifest.chunk_count,
                        "actual": points_count,
                    },
                )
            )

    issues.extend(_schema_issues(info, manifest))
    try:
        count_result = client.count(
            collection_name=collection_name,
            count_filter=_identity_filter(manifest),
            exact=True,
        )
        raw_identity_count = _value(count_result, "count")
        if (
            isinstance(raw_identity_count, bool)
            or not isinstance(raw_identity_count, int)
            or raw_identity_count < 0
        ):
            issues.append(
                CompatibilityIssue(
                    "integrity",
                    "invalid_identity_match_count",
                    {"actual": raw_identity_count},
                )
            )
        else:
            identity_matched_points = raw_identity_count
            if identity_matched_points != manifest.chunk_count:
                issues.append(
                    CompatibilityIssue(
                        "integrity",
                        "identity_payload_count_mismatch",
                        {
                            "expected": manifest.chunk_count,
                            "actual": identity_matched_points,
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001 - identity uncertainty is incompatible
        issues.append(
            CompatibilityIssue(
                "integrity",
                "identity_count_unavailable",
                {"error_type": type(exc).__name__},
            )
        )

    return CollectionCompatibility(
        collection_name=collection_name,
        generation_id=manifest.generation_id,
        expected_points=manifest.chunk_count,
        points_count=points_count,
        identity_matched_points=identity_matched_points,
        issues=tuple(issues),
    )


def require_collection_compatibility(
    client: Any,
    collection_name: str,
    manifest: GenerationManifest,
) -> CollectionCompatibility:
    """Return compatibility or raise a concise fail-closed startup error."""
    result = check_collection_compatibility(client, collection_name, manifest)
    if not result.ok:
        codes = ", ".join(issue.code for issue in result.issues)
        raise CollectionIncompatibleError(
            f"collection {collection_name!r} is incompatible with generation "
            f"{manifest.generation_id!r}: {codes}"
        )
    return result
