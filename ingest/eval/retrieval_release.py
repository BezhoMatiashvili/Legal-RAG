"""Create-only repeated retrieval evaluation, paired verdict, and handoff artifacts.

This release path is intentionally stricter than the interactive experiment log.  Each
``production`` and ``accuracy_strict`` repeat is written before another repeat starts,
ordered candidate/final rankings and branch routes are retained, and aggregate claims are
made only from an exactly pairable frozen baseline.  A summary-only baseline is useful
provenance, but it produces an ``inconclusive`` ceiling rather than invented paired scores.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .evaluate import run_mode
from .metrics import METRIC_DIRECTIONS, aggregate
from .stats import DEFAULT_RESAMPLES, DEFAULT_SEED, compare, holm_correct
from ingest.config import retrieval_fingerprint_sha256
from ingest.release_inputs import (
    GENERATION_ID,
    GOLDEN_SET_SHA256,
    HOLDOUT_SHA256,
    PHYSICAL_COLLECTION,
    SNAPSHOT_ID,
    TRANSLATION_SHA256,
)
from ingest.retrieval import EvaluationProvenance

REPEAT_SCHEMA_VERSION = "retrieval-release-repeat/v1"
COMPARISON_SCHEMA_VERSION = "retrieval-release-comparison/v1"
HANDOFF_SCHEMA_VERSION = "retrieval-release-handoff/v1"
TRACKS = ("production", "accuracy_strict")
REPEAT_COUNT = 2
EXPECTED_QUERY_COUNT = 337
FIXED_ZERO_SOURCES = {"tas": 25, "tbappeal": 31}
RETRIEVAL_ONLY_DISCLAIMER = (
    "This verdict concerns retrieval and canonical evidence selection only; it does not "
    "establish legal-answer accuracy."
)
REQUIRED_IDENTITY_HASHES = (
    "snapshot_sha256",
    "generation_sha256",
    "configuration_sha256",
    "collection_sha256",
    "collection_configuration_sha256",
    "vector_checksum_sha256",
    "vector_probe_sha256",
    "qrel_adapter_sha256",
    "baseline_manifest_sha256",
    "verification_1_sha256",
    "verification_2_sha256",
    "dependency_lock_sha256",
    "runtime_identity_sha256",
    "code_identity_sha256",
    "embedding_artifact_sha256",
    "tokenizer_artifact_sha256",
    "reranker_artifact_sha256",
    "corpus_hash",
    "vector_space_id",
    "chunk_config_id",
    "header_config_id",
    "retrieval_fingerprint",
    "dirty_patch_hash",
    "golden_set_sha256",
    "translation_sha256",
    "holdout_sha256",
)
PUBLIC_METRICS = (
    "success1",
    "success5",
    "success10",
    "candidate_recall50",
    "candidate_recall80",
    "required_evidence_recall10",
    "document_identity1",
    "passage_accuracy1",
    "context_duplication10",
    "context_noise10",
)
METRIC_LABELS = {
    "success1": "Success@1",
    "success5": "Success@5",
    "success10": "Success@10",
    "candidate_recall50": "Candidate recall@50",
    "candidate_recall80": "Candidate recall@80",
    "required_evidence_recall10": "Required-evidence recall@10",
    "document_identity1": "Correct document identity@1",
    "passage_accuracy1": "Passage accuracy@1",
    "context_duplication10": "Context duplication@10",
    "context_noise10": "Context noise@10",
}
METRIC_REPORT_ALIASES = {
    "success_at_1": "success1",
    "success_at_5": "success5",
    "success_at_10": "success10",
    "correct_document_identity": "document_identity1",
    "passage_accuracy": "passage_accuracy1",
    # Frozen v2 has one required evidence span per query.  These are aliases, not
    # independent corroborating measurements.
    "evidence_span_coverage10": "required_evidence_recall10",
}
ALIAS_METRIC_LABELS = {
    "success_at_1": "Success@1",
    "success_at_5": "Success@5",
    "success_at_10": "Success@10",
    "correct_document_identity": "Correct document identity@1",
    "passage_accuracy": "Passage accuracy@1",
    "evidence_span_coverage10": "Evidence-span coverage@10",
}
_TRACE_STATUSES = {
    "ok",
    "degraded",
    "failed_abstention",
    "abstain",
    "unscorable",
    "exception",
}
_FAILED_TRACE_STATUSES = {
    "degraded",
    "failed_abstention",
    "unscorable",
    "exception",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RetrievalReleaseError(ValueError):
    """A release artifact is missing identity, is non-pairable, or would be replaced."""


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


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _comparison_record(value: object) -> dict[str, Any]:
    """Expose statistical fields with explicit baseline/candidate release labels."""

    row = dataclasses.asdict(value)
    row.update(
        {
            "metric_label": METRIC_LABELS.get(row["metric"], row["metric"]),
            "baseline": row["mean_a"],
            "candidate": row["mean_b"],
            "paired_delta": row["diff"],
            "paired_delta_direction": "candidate_minus_baseline",
            "confidence_interval_95": [row["diff_lo"], row["diff_hi"]],
            "sample_count": row["n"],
        }
    )
    return row


def _alias_comparison_record(
    alias: str, canonical: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **record,
        "metric": alias,
        "metric_label": ALIAS_METRIC_LABELS[alias],
        "alias_of": canonical,
        "independent_corroboration": False,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_json(path: Path, value: object) -> Path:
    """Durably create one JSON artifact without a replace-capable operation."""

    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise RetrievalReleaseError("release artifact parent must be a real directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"release artifact already exists: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"release artifact already exists: {destination}"
            ) from None
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


def _plain_mapping(value: object, *, where: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if not isinstance(value, Mapping):
        raise RetrievalReleaseError(f"{where} must be a mapping")
    return dict(_json_safe(dict(value)))


def _json_safe(value: object) -> object:
    """Normalize dataclass/Path/tuple identity values into canonical JSON values."""

    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return _json_safe(value.value)
    return value


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    material = dict(identity)
    missing = [name for name in REQUIRED_IDENTITY_HASHES if name not in material]
    invalid = [
        name
        for name in REQUIRED_IDENTITY_HASHES
        if name in material and not _SHA256.fullmatch(str(material[name]))
    ]
    for name in (
        "run_id",
        "snapshot_id",
        "generation_id",
        "physical_collection",
        "embedding_model",
        "embedding_revision",
        "tokenizer_model",
        "tokenizer_revision",
        "reranker_model",
        "reranker_revision",
        "runtime_image_digest",
        "repository_revision",
    ):
        if not isinstance(material.get(name), str) or not material[name]:
            missing.append(name)
    points_count = material.get("points_count")
    if (
        isinstance(points_count, bool)
        or not isinstance(points_count, int)
        or points_count < 1
    ):
        missing.append("points_count")
    if material.get("retrieval_fingerprint_revision") != 2:
        missing.append("retrieval_fingerprint_revision")
    if missing or invalid:
        raise RetrievalReleaseError(
            "incomplete retrieval release identity: "
            f"missing={sorted(set(missing))}, invalid={sorted(set(invalid))}"
        )
    if (
        material["snapshot_id"] != SNAPSHOT_ID
        or material["generation_id"] != GENERATION_ID
        or material["physical_collection"] != PHYSICAL_COLLECTION
    ):
        raise RetrievalReleaseError(
            "retrieval release identity is not the frozen 512 candidate tuple"
        )
    # Any additional reportable SHA field is held to the same full-hash contract.
    for name, value in material.items():
        if name.endswith("_sha256") and not _SHA256.fullmatch(str(value)):
            raise RetrievalReleaseError(f"identity {name} is not a full SHA-256")
    revision = re.compile(r"^[0-9a-f]{7,64}$")
    for name in (
        "embedding_revision",
        "tokenizer_revision",
        "reranker_revision",
        "repository_revision",
    ):
        if not revision.fullmatch(str(material.get(name) or "")):
            raise RetrievalReleaseError(f"identity {name} is not an immutable revision")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", material["runtime_image_digest"]) is None:
        raise RetrievalReleaseError(
            "identity runtime_image_digest is not digest-qualified"
        )
    expected_frozen_hashes = {
        "golden_set_sha256": GOLDEN_SET_SHA256,
        "translation_sha256": TRANSLATION_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
    }
    observed_frozen_hashes = {
        name: material.get(name) for name in expected_frozen_hashes
    }
    if observed_frozen_hashes != expected_frozen_hashes:
        raise RetrievalReleaseError(
            "identity does not bind the exact frozen 337-query v2 inputs"
        )
    return material


def _baseline_identity(value: object) -> dict[str, Any]:
    baseline = _plain_mapping(value, where="baseline")
    # Bind only the identity summary in every repeat; complete per-query traces are consumed
    # by comparison and must not be redundantly copied four times.
    keep = (
        "root",
        "manifest_sha256",
        "collection_sha256",
        "configuration_hash",
        "ordered_query_ids",
        "query_ids",
        "ranking_hashes",
        "decision_result_hashes",
        "repeats",
    )
    out = {name: baseline[name] for name in keep if name in baseline}
    repeats = out.get("repeats")
    if isinstance(repeats, Sequence) and not isinstance(repeats, (str, bytes)):
        out["repeats"] = [
            {
                key: row.get(key)
                for key in ("ranking_hash", "decision_result_hash", "result_hash")
                if isinstance(row, Mapping) and key in row
            }
            for row in repeats
        ]
    return out


def _validated_evaluation_provenance(
    value: object,
    *,
    release_identity: Mapping[str, Any],
    track: str,
) -> dict[str, Any]:
    """Require the complete direct-physical model/runtime quality-claim identity."""

    provenance = _plain_mapping(value, where="evaluation_provenance")
    try:
        validated = EvaluationProvenance(**provenance)
        validated.validate_complete()
        normalized = validated.to_dict()
    except (TypeError, ValueError) as exc:
        raise RetrievalReleaseError(
            f"evaluation provenance is incomplete or invalid: {exc}"
        ) from exc
    if normalized != provenance:
        raise RetrievalReleaseError(
            "evaluation provenance is not an exact canonical identity"
        )
    if (
        normalized["physical_collection"] != release_identity["physical_collection"]
        or normalized["queried_collection"] != release_identity["physical_collection"]
        or normalized["access_kind"] != "direct_physical"
    ):
        raise RetrievalReleaseError(
            "evaluation provenance does not query the exact physical collection"
        )
    if normalized["generation_id"] != release_identity["generation_id"]:
        raise RetrievalReleaseError(
            "evaluation provenance targets a different generation"
        )
    if normalized["snapshot_hash"] != release_identity["snapshot_sha256"]:
        raise RetrievalReleaseError(
            "evaluation provenance targets a different snapshot"
        )
    if normalized["execution_mode"] != track:
        raise RetrievalReleaseError(
            "evaluation provenance execution mode differs from the release track"
        )
    if normalized["dependency_identity"] != release_identity["dependency_lock_sha256"]:
        raise RetrievalReleaseError(
            "evaluation provenance dependency identity differs from the release identity"
        )
    for provenance_field, identity_field in (
        ("points_count", "points_count"),
        ("corpus_hash", "corpus_hash"),
        ("vector_space_id", "vector_space_id"),
        ("chunk_config_id", "chunk_config_id"),
        ("header_config_id", "header_config_id"),
        ("retrieval_fingerprint_revision", "retrieval_fingerprint_revision"),
        ("retrieval_fingerprint", "retrieval_fingerprint"),
        ("dirty_patch_hash", "dirty_patch_hash"),
    ):
        if normalized[provenance_field] != release_identity[identity_field]:
            raise RetrievalReleaseError(
                "evaluation provenance collection/configuration identity differs for "
                f"{provenance_field}"
            )
    for provenance_field, identity_field in (
        ("embedding_model", "embedding_model"),
        ("embedding_revision", "embedding_revision"),
        ("tokenizer_model", "tokenizer_model"),
        ("tokenizer_revision", "tokenizer_revision"),
        ("reranker_model", "reranker_model"),
        ("reranker_revision", "reranker_revision"),
        ("image_identity", "runtime_image_digest"),
        ("git_sha", "repository_revision"),
    ):
        if normalized[provenance_field] != release_identity[identity_field]:
            raise RetrievalReleaseError(
                "evaluation provenance model/runtime identity differs for "
                f"{provenance_field}"
            )
    expected_frozen_hashes = {
        "golden_v2": release_identity["golden_set_sha256"],
        "holdout_v2": release_identity["holdout_sha256"],
        "authored_query_translations": release_identity["translation_sha256"],
        "v2_candidate_qrels": release_identity["qrel_adapter_sha256"],
    }
    if normalized["frozen_set_hashes"] != expected_frozen_hashes:
        raise RetrievalReleaseError(
            "evaluation provenance does not exactly bind the frozen v2 inputs and "
            "candidate qrel adapter"
        )
    return normalized


def _failure_counts_from_queries(
    queries: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Validate run-mode status/score invariants and derive overlapping failure counts."""

    for row in queries:
        status = row.get("status")
        if status not in _TRACE_STATUSES:
            raise RetrievalReleaseError(
                f"release repeat has unknown query status {status!r}"
            )
        score = _score(row)
        failed = score.get("failed")
        if type(failed) is not bool:
            raise RetrievalReleaseError(
                "release repeat query score.failed must be boolean"
            )
        expected_failed = status in _FAILED_TRACE_STATUSES
        if failed is not expected_failed:
            raise RetrievalReleaseError(
                "release repeat query status/failure state does not reconcile"
            )
        failure_reason = score.get("failure_reason")
        if failed != (isinstance(failure_reason, str) and bool(failure_reason)):
            raise RetrievalReleaseError(
                "release repeat query failure reason does not reconcile"
            )
        answerable = row.get("answerable")
        if type(answerable) is not bool:
            raise RetrievalReleaseError(
                "release repeat query answerable must be boolean"
            )
        if status == "failed_abstention" and not answerable:
            raise RetrievalReleaseError(
                "failed abstention must concern an answerable query"
            )
        if status == "abstain" and answerable:
            raise RetrievalReleaseError(
                "answerable abstention must be retained as a failed abstention"
            )
    return {
        # ``failed`` is the total.  The remaining fields are overlapping typed subsets,
        # not additive buckets.
        "failed": sum(bool(_score(row).get("failed")) for row in queries),
        "degraded": sum(row.get("status") == "degraded" for row in queries),
        "skipped_or_unchunkable": sum(
            row.get("status") == "unscorable" for row in queries
        ),
        "exceptions": sum(row.get("status") == "exception" for row in queries),
        "answerable_abstentions": sum(
            row.get("status") == "failed_abstention" for row in queries
        ),
    }


def _event_query_ids(events: object) -> list[str]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise RetrievalReleaseError("release latency failure events must be a sequence")
    rows = list(events)
    if any(not isinstance(row, Mapping) or not row.get("query_id") for row in rows):
        raise RetrievalReleaseError("release latency failure event is malformed")
    return [str(row["query_id"]) for row in rows]


def _validate_query_trace_hashes(queries: Sequence[Mapping[str, Any]]) -> None:
    """Rebuild each run-mode hash so an invalid create-only artifact is never written."""

    for row in queries:
        if not isinstance(row, Mapping):
            raise RetrievalReleaseError("release repeat query trace is not an object")
        for field_name in ("ranking_hash", "decision_result_hash"):
            if not _SHA256.fullmatch(str(row.get(field_name) or "")):
                raise RetrievalReleaseError(
                    f"release repeat query trace has invalid {field_name}"
                )
        if row.get("result_hash") != row["decision_result_hash"]:
            raise RetrievalReleaseError(
                "release repeat query result_hash alias differs from decision_result_hash"
            )

        decision_material = {
            key: item
            for key, item in row.items()
            if key
            not in {
                "timings_ms",
                "outcome_timings_ms",
                "ranking_hash",
                "decision_result_hash",
                "result_hash",
            }
        }
        branches = decision_material.get("branches")
        if isinstance(branches, list):
            decision_material["branches"] = [
                {key: item for key, item in branch.items() if key != "elapsed_ms"}
                if isinstance(branch, Mapping)
                else branch
                for branch in branches
            ]
        if row.get("status") in {"unscorable", "exception"}:
            ranking_material = {
                "ranked_hits": row.get("ranked_hits"),
                "candidate_depth": row.get("candidate_depth"),
            }
        else:
            candidate_depth = row.get("candidate_depth")
            branch_rows = row.get("branches")
            if not isinstance(candidate_depth, Mapping) or not isinstance(
                branch_rows, list
            ):
                raise RetrievalReleaseError(
                    "release repeat query trace lacks ranking provenance"
                )
            ranking_material = {
                "ranked_hits": row.get("ranked_hits"),
                "candidate_ranking": candidate_depth.get("ordered_candidates", []),
                "branch_rankings": [
                    {
                        "name": branch.get("name"),
                        "route": branch.get("route"),
                        "hits": branch.get("hits"),
                    }
                    for branch in branch_rows
                    if isinstance(branch, Mapping)
                ],
            }
            if len(ranking_material["branch_rankings"]) != len(branch_rows):
                raise RetrievalReleaseError(
                    "release repeat query branch ranking is malformed"
                )
        if row["ranking_hash"] != _sha256_value(ranking_material):
            raise RetrievalReleaseError(
                "release repeat query ranking hash does not reconcile"
            )
        if row["decision_result_hash"] != _sha256_value(decision_material):
            raise RetrievalReleaseError(
                "release repeat query decision/result hash does not reconcile"
            )


def persist_repeat(
    *,
    output: Path,
    track: str,
    repeat_index: int,
    scores: Sequence[Any],
    latency: Mapping[str, Any],
    identity: Mapping[str, Any],
    evaluation_provenance: object,
    baseline: object,
    created_at: datetime | None = None,
) -> Path:
    """Persist one completed track repeat as an immutable, self-contained artifact."""

    if track not in TRACKS:
        raise RetrievalReleaseError(f"release track must be one of {TRACKS}")
    if repeat_index not in {1, 2}:
        raise RetrievalReleaseError("release repeat_index must be 1 or 2")
    release_identity = _validate_identity(identity)
    provenance = _validated_evaluation_provenance(
        evaluation_provenance,
        release_identity=release_identity,
        track=track,
    )
    configuration_hash = release_identity["configuration_sha256"]
    if not _SHA256.fullmatch(configuration_hash):
        raise RetrievalReleaseError("configuration identity must be a full SHA-256")

    query_rows = list(latency.get("queries", ()))
    score_rows = [dataclasses.asdict(score) for score in scores]
    if [row.get("query_id") for row in query_rows] != [row["id"] for row in score_rows]:
        raise RetrievalReleaseError("score/trace query order differs")
    if latency.get("ranking_hash") != _sha256_value(
        [
            {"query_id": row["query_id"], "ranking_hash": row["ranking_hash"]}
            for row in query_rows
        ]
    ):
        raise RetrievalReleaseError("repeat ranking hash does not reconcile")
    if latency.get("decision_result_hash") != _sha256_value(
        [
            {
                "query_id": row["query_id"],
                "decision_result_hash": row["decision_result_hash"],
            }
            for row in query_rows
        ]
    ):
        raise RetrievalReleaseError("repeat decision/result hash does not reconcile")
    if any(row.get("score") != score for row, score in zip(query_rows, score_rows)):
        raise RetrievalReleaseError("query trace score differs from the scored row")
    _validate_query_trace_hashes(query_rows)
    failure_counts = _failure_counts_from_queries(query_rows)
    event_statuses = {
        "degraded": "degraded",
        "skipped": "unscorable",
        "exceptions": "exception",
    }
    for event_name, status in event_statuses.items():
        expected_query_ids = [
            str(row["query_id"]) for row in query_rows if row.get("status") == status
        ]
        if _event_query_ids(latency.get(event_name, ())) != expected_query_ids:
            raise RetrievalReleaseError(
                f"release latency {event_name} events do not reconcile with query traces"
            )
    bound_baseline = _baseline_identity(baseline)
    if (
        bound_baseline.get("manifest_sha256")
        != release_identity["baseline_manifest_sha256"]
    ):
        raise RetrievalReleaseError(
            "release identity binds a different baseline manifest"
        )

    artifact = {
        "schema_version": REPEAT_SCHEMA_VERSION,
        "created_at": (created_at or datetime.now(UTC))
        .isoformat()
        .replace("+00:00", "Z"),
        "run_id": release_identity["run_id"],
        "track": track,
        "repeat_index": repeat_index,
        "identity": release_identity,
        "baseline_identity": bound_baseline,
        "evaluation_provenance": provenance,
        "configuration_sha256": configuration_hash,
        "query_count": len(query_rows),
        "ordered_query_ids_sha256": _sha256_value(
            [row["query_id"] for row in query_rows]
        ),
        "ranking_hash": latency["ranking_hash"],
        "decision_result_hash": latency["decision_result_hash"],
        "metrics": aggregate(scores),
        "failure_counts": failure_counts,
        "failure_count_semantics": {
            "failed": "total failed queries",
            "typed_fields": "overlapping subsets of failed; do not sum",
        },
        "queries": query_rows,
    }
    return _create_json(output, artifact)


def _expected_frozen_hashes(identity: Mapping[str, Any]) -> dict[str, str]:
    return {
        "golden_v2": str(identity["golden_set_sha256"]),
        "holdout_v2": str(identity["holdout_sha256"]),
        "authored_query_translations": str(identity["translation_sha256"]),
        "v2_candidate_qrels": str(identity["qrel_adapter_sha256"]),
    }


def _derive_backend_provenance(
    backend: object,
    *,
    track: str,
    release_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive release provenance from the verified real backend binding."""

    from .backend import ProductionBackend
    from .evaluate import build_evaluation_provenance

    if type(backend) is not ProductionBackend:
        raise RetrievalReleaseError(
            "release evaluation requires the exact ProductionBackend implementation"
        )
    cfg = backend.cfg
    if (
        not cfg.production_mode
        or cfg.search_backend != "local"
        or cfg.collection_name != release_identity["physical_collection"]
        or cfg.generation_id != release_identity["generation_id"]
    ):
        raise RetrievalReleaseError(
            "release backend is not direct-physical production configuration"
        )
    expected_cfg_identity = {
        "embedding_model": cfg.embed_model,
        "embedding_revision": cfg.embedding_revision,
        "tokenizer_model": cfg.tokenizer_model,
        "tokenizer_revision": cfg.tokenizer_revision,
        "reranker_model": cfg.rerank_model,
        "reranker_revision": cfg.reranker_revision,
        "retrieval_fingerprint": retrieval_fingerprint_sha256(cfg),
    }
    for name, observed in expected_cfg_identity.items():
        if observed != release_identity[name]:
            raise RetrievalReleaseError(
                f"release backend configuration differs from identity for {name}"
            )
    if not cfg.rerank_enabled or backend.reranker is None:
        raise RetrievalReleaseError("release backend requires the reviewed reranker")
    index_info = backend.release_index_info
    if not isinstance(index_info, Mapping):
        raise RetrievalReleaseError(
            "release backend lacks the verified live collection binding"
        )
    for name in ("verification_1_sha256", "verification_2_sha256"):
        if index_info.get(name) != release_identity[name]:
            raise RetrievalReleaseError(
                f"release backend physical verification binding differs for {name}"
            )
    derived = build_evaluation_provenance(
        dict(index_info), _expected_frozen_hashes(release_identity), track
    ).to_dict()
    return _validated_evaluation_provenance(
        derived,
        release_identity=release_identity,
        track=track,
    )


def _preflight_repeat_destinations(
    output_dir: Path, *, run_id: str
) -> dict[str, tuple[Path, Path]]:
    root = Path(output_dir).expanduser().absolute()
    if os.path.lexists(root):
        if root.is_symlink() or not root.is_dir():
            raise RetrievalReleaseError(
                "release output directory must be a real directory"
            )
    else:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    paths = {
        track: tuple(
            root / f"{run_id}.{track}.repeat-{repeat_index:02d}.json"
            for repeat_index in (1, 2)
        )
        for track in TRACKS
    }
    existing = [
        str(path)
        for pair in paths.values()
        for path in pair
        if os.path.lexists(path)
    ]
    if existing:
        raise FileExistsError(
            "release repeat destination already exists: " + ", ".join(existing)
        )
    return paths


def run_release_repeats(
    *,
    output_dir: Path,
    backends: Mapping[str, Any],
    gold: Sequence[Any],
    relevance: Mapping[str, Any],
    identity: Mapping[str, Any],
    provenances: Mapping[str, object],
    baseline: object,
    level: str = "chunk",
    top_k: int = 10,
) -> dict[str, tuple[Path, Path]]:
    """Run and immediately persist exactly two repeats for both release tracks."""

    release_identity = _validate_identity(identity)
    if set(backends) != set(TRACKS) or set(provenances) != set(TRACKS):
        raise RetrievalReleaseError(
            "release evaluation requires production and accuracy_strict backends/provenance"
        )
    if len(gold) != EXPECTED_QUERY_COUNT:
        raise RetrievalReleaseError(
            "release evaluation requires the frozen 337-query v2 set"
        )
    query_ids = [str(getattr(query, "id", "")) for query in gold]
    if (
        any(not query_id for query_id in query_ids)
        or len(set(query_ids)) != EXPECTED_QUERY_COUNT
        or list(relevance) != query_ids
    ):
        raise RetrievalReleaseError(
            "release golden set and relevance order are not exactly pairable"
        )
    bound_baseline = _baseline_identity(baseline)
    if (
        bound_baseline.get("manifest_sha256")
        != release_identity["baseline_manifest_sha256"]
    ):
        raise RetrievalReleaseError(
            "release identity binds a different baseline manifest"
        )
    derived_provenances: dict[str, dict[str, Any]] = {}
    for track in TRACKS:
        derived = _derive_backend_provenance(
            backends[track], track=track, release_identity=release_identity
        )
        supplied = _validated_evaluation_provenance(
            provenances[track], release_identity=release_identity, track=track
        )
        if supplied != derived:
            raise RetrievalReleaseError(
                f"{track} supplied provenance differs from the verified backend binding"
            )
        derived_provenances[track] = derived
    preflight_paths = _preflight_repeat_destinations(
        output_dir, run_id=release_identity["run_id"]
    )
    paths: dict[str, tuple[Path, Path]] = {}
    for track in TRACKS:
        written: list[Path] = []
        for repeat_index in (1, 2):
            scores, latency = run_mode(
                backends[track], gold, relevance, track, level, top_k
            )
            output = preflight_paths[track][repeat_index - 1]
            written.append(
                persist_repeat(
                    output=output,
                    track=track,
                    repeat_index=repeat_index,
                    scores=scores,
                    latency=latency,
                    identity=release_identity,
                    evaluation_provenance=derived_provenances[track],
                    baseline=baseline,
                )
            )
        paths[track] = (written[0], written[1])
    return paths


def load_repeat(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalReleaseError(
            f"cannot read release repeat {path}: {exc}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != REPEAT_SCHEMA_VERSION
    ):
        raise RetrievalReleaseError("unsupported release repeat schema")
    release_identity = _validate_identity(
        _plain_mapping(value.get("identity"), where="repeat identity")
    )
    if value.get("track") not in TRACKS or value.get("repeat_index") not in {1, 2}:
        raise RetrievalReleaseError("invalid release repeat track/index")
    track = str(value["track"])
    if value.get("run_id") != release_identity["run_id"]:
        raise RetrievalReleaseError("release repeat run_id differs from its identity")
    if value.get("configuration_sha256") != release_identity["configuration_sha256"]:
        raise RetrievalReleaseError(
            "release repeat configuration differs from its identity"
        )
    _validated_evaluation_provenance(
        value.get("evaluation_provenance"),
        release_identity=release_identity,
        track=track,
    )
    baseline_identity = _plain_mapping(
        value.get("baseline_identity"), where="repeat baseline identity"
    )
    if (
        baseline_identity.get("manifest_sha256")
        != release_identity["baseline_manifest_sha256"]
    ):
        raise RetrievalReleaseError(
            "release repeat baseline differs from its identity"
        )
    queries = value.get("queries")
    if not isinstance(queries, list) or value.get("query_count") != len(queries):
        raise RetrievalReleaseError("release repeat query count does not reconcile")
    query_ids = [row.get("query_id") for row in queries if isinstance(row, Mapping)]
    if len(query_ids) != len(queries) or len(query_ids) != len(set(query_ids)):
        raise RetrievalReleaseError("release repeat query ids are invalid")
    if value.get("ordered_query_ids_sha256") != _sha256_value(query_ids):
        raise RetrievalReleaseError("release repeat query order hash mismatch")
    _validate_query_trace_hashes(queries)
    for field_name in ("ranking_hash", "decision_result_hash", "configuration_sha256"):
        if not _SHA256.fullmatch(str(value.get(field_name) or "")):
            raise RetrievalReleaseError(f"release repeat has invalid {field_name}")
    expected_ranking_hash = _sha256_value(
        [
            {"query_id": row["query_id"], "ranking_hash": row["ranking_hash"]}
            for row in queries
        ]
    )
    expected_decision_hash = _sha256_value(
        [
            {
                "query_id": row["query_id"],
                "decision_result_hash": row["decision_result_hash"],
            }
            for row in queries
        ]
    )
    if value["ranking_hash"] != expected_ranking_hash:
        raise RetrievalReleaseError("release repeat ranking hash does not reconcile")
    if value["decision_result_hash"] != expected_decision_hash:
        raise RetrievalReleaseError("release repeat decision hash does not reconcile")
    expected_failures = _failure_counts_from_queries(queries)
    if value.get("failure_counts") != expected_failures:
        raise RetrievalReleaseError("release repeat failure counts do not reconcile")
    if value.get("failure_count_semantics") != {
        "failed": "total failed queries",
        "typed_fields": "overlapping subsets of failed; do not sum",
    }:
        raise RetrievalReleaseError(
            "release repeat failure count semantics are invalid"
        )
    return value


def _repeat_pair(paths: Sequence[Path], expected_track: str) -> tuple[dict, dict]:
    if len(paths) != REPEAT_COUNT:
        raise RetrievalReleaseError(
            f"{expected_track} requires exactly two repeat artifacts"
        )
    pair = tuple(load_repeat(path) for path in paths)
    if [row["track"] for row in pair] != [expected_track, expected_track]:
        raise RetrievalReleaseError("release repeat track mismatch")
    if [row["repeat_index"] for row in pair] != [1, 2]:
        raise RetrievalReleaseError("release repeat indices must be ordered 1,2")
    first, second = pair
    for field_name in (
        "run_id",
        "identity",
        "baseline_identity",
        "evaluation_provenance",
        "configuration_sha256",
        "query_count",
        "ordered_query_ids_sha256",
    ):
        if first[field_name] != second[field_name]:
            raise RetrievalReleaseError(f"repeat {field_name} drifted")
    return first, second


def _extract_baseline_queries(
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    options: list[Any] = [baseline.get("queries"), baseline.get("per_query")]
    production = baseline.get("production")
    if isinstance(production, Mapping):
        options.extend((production.get("queries"), production.get("per_query")))
    repeats = baseline.get("repeats")
    if isinstance(repeats, Sequence) and repeats and isinstance(repeats[0], Mapping):
        options.extend((repeats[0].get("queries"), repeats[0].get("per_query")))
    for value in options:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows = [dict(row) for row in value if isinstance(row, Mapping)]
            if len(rows) == len(value):
                return rows
    return None


def _baseline_repeat_hashes(baseline: Mapping[str, Any], field_name: str) -> list[str]:
    plural = f"{field_name}es" if field_name.endswith("hash") else f"{field_name}s"
    direct = baseline.get(plural)
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [str(value) for value in direct]
    repeats = baseline.get("repeats")
    if isinstance(repeats, Sequence) and not isinstance(repeats, (str, bytes)):
        return [
            str(row.get(field_name) or row.get("result_hash") or "")
            for row in repeats
            if isinstance(row, Mapping)
        ]
    return []


def _score(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("score") or row.get("metrics")
    return value if isinstance(value, Mapping) else row


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id") or row.get("id") or "")


def _metric_value(
    row: Mapping[str, Any], metric: str, *, source: str | None = None
) -> float | None:
    # The frozen TAS/Tbilisi Appeal labels are incomplete summaries.  Both sides of the
    # paired comparison remain zero under the declared scoring policy unless separately
    # authenticated adjudication changes a label.
    if source in FIXED_ZERO_SOURCES:
        return 0.0
    value = _score(row).get(metric)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalReleaseError(f"query metric {metric} is not numeric")
    return float(value)


def _slice_query_ids(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    slices: dict[str, set[str]] = {"all": set()}
    for row in candidate_rows:
        score = _score(row)
        query_id = _query_id(row)
        source = str(score.get("source") or "")
        language = str(score.get("language") or "")
        query_type = str(score.get("query_type") or "")
        slices["all"].add(query_id)
        if source:
            slices.setdefault(f"source:{source}", set()).add(query_id)
        if language == "ka" and query_type == "paraphrase":
            slices.setdefault("georgian_paraphrase", set()).add(query_id)
        if source == "napr":
            slices.setdefault("napr", set()).add(query_id)
        if source == "constcourt":
            slices.setdefault("constitutional_court", set()).add(query_id)
        if query_type in {"keyword", "legal_citation", "exact_identifier"}:
            slices.setdefault("exact_identifier_proxy", set()).add(query_id)
        risk = str(score.get("risk_level") or "")
        if risk:
            slices.setdefault(f"risk:{risk}", set()).add(query_id)
        for tag in score.get("tags", ()) or ():
            slices.setdefault(f"tag:{tag}", set()).add(query_id)
    return slices


def _select_next_experiment(
    accuracy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {
        "document_first_retrieval": 0,
        "structural_384_token_chunking": 0,
        "candidate_pool_preserving_reranker_calibration": 0,
    }
    for row in accuracy_rows:
        if row.get("status") in {"unscorable", "exception"}:
            continue
        score = _score(row)
        if score.get("passage_accuracy1") == 1.0:
            continue
        depth = row.get("candidate_depth")
        if not isinstance(depth, Mapping):
            continue
        document_at_80 = (depth.get("gold_document_present_at_depth") or {}).get("80")
        passage_at_80 = (depth.get("gold_passage_present_at_depth") or {}).get("80")
        if document_at_80 is False:
            counts["document_first_retrieval"] += 1
        elif document_at_80 is True and not passage_at_80:
            counts["structural_384_token_chunking"] += 1
        elif passage_at_80 is True:
            counts["candidate_pool_preserving_reranker_calibration"] += 1
    order = tuple(counts)
    winner = max(order, key=lambda name: (counts[name], -order.index(name)))
    return {
        "experiment": winner,
        "recoverable_query_count": counts[winner],
        "error_attribution_counts": counts,
        "tie_break_order": list(order),
    }


def compare_release_to_baseline(
    *,
    production_paths: Sequence[Path],
    accuracy_strict_paths: Sequence[Path],
    baseline: object,
    output: Path,
    require_frozen_v2: bool = True,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Create the paired comparison and mechanical retrieval-only verdict."""

    production = _repeat_pair(production_paths, "production")
    accuracy = _repeat_pair(accuracy_strict_paths, "accuracy_strict")
    first_production, second_production = production
    first_accuracy, second_accuracy = accuracy
    if first_production["identity"] != first_accuracy["identity"]:
        raise RetrievalReleaseError(
            "production/accuracy_strict release identities differ"
        )
    if first_production["run_id"] != first_accuracy["run_id"]:
        raise RetrievalReleaseError("production/accuracy_strict run ids differ")

    candidate_rows = first_production["queries"]
    strict_rows = first_accuracy["queries"]
    candidate_ids = [_query_id(row) for row in candidate_rows]
    if candidate_ids != [_query_id(row) for row in strict_rows]:
        raise RetrievalReleaseError("release tracks evaluate different query orders")
    unavailable: list[str] = []
    if require_frozen_v2 and len(candidate_ids) != EXPECTED_QUERY_COUNT:
        unavailable.append("candidate_query_count_is_not_frozen_337")

    baseline_map = _plain_mapping(baseline, where="baseline")
    if baseline_map.get("manifest_sha256") != first_production["baseline_identity"].get(
        "manifest_sha256"
    ):
        raise RetrievalReleaseError(
            "comparison baseline differs from the repeated-run binding"
        )
    for field_name in ("ranking_hashes", "decision_result_hashes"):
        repeated_binding = first_production["baseline_identity"].get(field_name)
        if repeated_binding is not None and list(
            baseline_map.get(field_name) or ()
        ) != list(repeated_binding):
            raise RetrievalReleaseError(
                f"comparison baseline {field_name} differs from the repeated-run binding"
            )
    baseline_rows = _extract_baseline_queries(baseline_map)
    baseline_ranking_hashes = _baseline_repeat_hashes(baseline_map, "ranking_hash")
    baseline_decision_hashes = _baseline_repeat_hashes(
        baseline_map, "decision_result_hash"
    )
    if (
        len(baseline_ranking_hashes) != 2
        or any(not _SHA256.fullmatch(value) for value in baseline_ranking_hashes)
        or len(set(baseline_ranking_hashes)) != 1
    ):
        unavailable.append("baseline_two_repeat_ranking_hashes_unavailable_or_drifted")
    if (
        len(baseline_decision_hashes) != 2
        or any(not _SHA256.fullmatch(value) for value in baseline_decision_hashes)
        or len(set(baseline_decision_hashes)) != 1
    ):
        unavailable.append("baseline_two_repeat_decision_hashes_unavailable_or_drifted")
    if baseline_rows is None:
        unavailable.append("baseline_per_query_traces_unavailable")
        baseline_rows = []
    baseline_ids = [_query_id(row) for row in baseline_rows]
    if baseline_rows and baseline_ids != candidate_ids:
        unavailable.append("baseline_query_order_not_pairable")
    if baseline_rows and baseline_ids == candidate_ids:
        identity_fields = ("id", "source", "query_type", "language", "cluster_id")
        if any(
            any(
                _score(left).get(field) != _score(right).get(field)
                for field in identity_fields
            )
            for left, right in zip(baseline_rows, candidate_rows)
        ):
            unavailable.append("baseline_query_score_identity_not_pairable")
        complete_baseline_trace = True
        for row in baseline_rows:
            raw_candidates = row.get("raw_candidates")
            final_ranking = row.get("final_ranking")
            if (
                not isinstance(raw_candidates, list)
                or len(raw_candidates) < 80
                or not isinstance(final_ranking, list)
                or not final_ranking
                or any(
                    not isinstance(item, Mapping)
                    or not item.get("point_id")
                    or isinstance(item.get("score"), bool)
                    or not isinstance(item.get("score"), (int, float))
                    for item in (*raw_candidates, *final_ranking)
                )
                or any(
                    not isinstance(row.get(field), Mapping)
                    for field in (
                        "branch_provenance",
                        "entity_matches",
                        "document_matches",
                        "route_decision",
                    )
                )
            ):
                unavailable.append("baseline_complete_depth_80_trace_unavailable")
                complete_baseline_trace = False
                break
        if complete_baseline_trace:
            expected_baseline_ranking_hash = _sha256_value(
                [
                    {
                        "query_id": row["query_id"],
                        "raw_candidates": row["raw_candidates"],
                        "final_ranking": row["final_ranking"],
                    }
                    for row in baseline_rows
                ]
            )
            expected_baseline_decision_hash = _sha256_value(baseline_rows)
            if baseline_ranking_hashes and any(
                value != expected_baseline_ranking_hash
                for value in baseline_ranking_hashes
            ):
                unavailable.append("baseline_ranking_hash_does_not_reconcile")
            if baseline_decision_hashes and any(
                value != expected_baseline_decision_hash
                for value in baseline_decision_hashes
            ):
                unavailable.append("baseline_decision_hash_does_not_reconcile")

    source_counts = Counter(
        str(_score(row).get("source") or "") for row in candidate_rows
    )
    fixed_zero_checks = {
        source: {
            "expected": count,
            "observed": source_counts[source],
            "all_zero_score_failures": (
                source_counts[source] == count
                and all(
                    bool(_score(row).get("failed"))
                    and all(_score(row).get(metric) == 0.0 for metric in PUBLIC_METRICS)
                    for row in candidate_rows
                    if _score(row).get("source") == source
                )
            ),
        }
        for source, count in FIXED_ZERO_SOURCES.items()
    }
    if require_frozen_v2 and not all(
        check["observed"] == check["expected"] and check["all_zero_score_failures"]
        for check in fixed_zero_checks.values()
    ):
        unavailable.append("fixed_incomplete_source_zero_policy_not_reconciled")

    baseline_by_id = {_query_id(row): row for row in baseline_rows}
    strict_by_id = {_query_id(row): row for row in strict_rows}
    comparisons = {}
    missing_metrics: list[str] = []
    if baseline_rows and baseline_ids == candidate_ids:
        for metric in PUBLIC_METRICS:
            baseline_values: list[float] = []
            candidate_values: list[float] = []
            clusters: list[str] = []
            for candidate_row in candidate_rows:
                query_id = _query_id(candidate_row)
                score = _score(candidate_row)
                source = str(score.get("source") or "")
                candidate_source_row = (
                    strict_by_id[query_id]
                    if metric in {"candidate_recall50", "candidate_recall80"}
                    else candidate_row
                )
                baseline_value = _metric_value(
                    baseline_by_id[query_id], metric, source=source
                )
                candidate_value = _metric_value(
                    candidate_source_row, metric, source=source
                )
                if baseline_value is None or candidate_value is None:
                    missing_metrics.append(f"{query_id}:{metric}")
                    continue
                baseline_values.append(baseline_value)
                candidate_values.append(candidate_value)
                clusters.append(str(score.get("cluster_id") or query_id))
            if len(baseline_values) != len(candidate_ids):
                continue
            comparisons[metric] = compare(
                metric,
                baseline_values,
                candidate_values,
                clusters=clusters,
                higher_is_better=METRIC_DIRECTIONS.get(metric, "higher") == "higher",
                resamples=resamples,
                seed=seed,
            )
    if missing_metrics:
        unavailable.append("paired_metric_values_unavailable")
    corrected = holm_correct(comparisons)

    slices = _slice_query_ids(candidate_rows)
    slice_table: dict[str, Any] = {}
    if baseline_rows and baseline_ids == candidate_ids:
        for slice_name, query_ids in sorted(slices.items()):
            slice_comparisons = {}
            for metric in PUBLIC_METRICS:
                baseline_values = []
                candidate_values = []
                clusters = []
                for query_id in candidate_ids:
                    if query_id not in query_ids:
                        continue
                    candidate_row = next(
                        row for row in candidate_rows if _query_id(row) == query_id
                    )
                    source = str(_score(candidate_row).get("source") or "")
                    candidate_source_row = (
                        strict_by_id[query_id]
                        if metric in {"candidate_recall50", "candidate_recall80"}
                        else candidate_row
                    )
                    left = _metric_value(
                        baseline_by_id[query_id], metric, source=source
                    )
                    right = _metric_value(candidate_source_row, metric, source=source)
                    if left is None or right is None:
                        continue
                    baseline_values.append(left)
                    candidate_values.append(right)
                    clusters.append(
                        str(_score(candidate_row).get("cluster_id") or query_id)
                    )
                if len(baseline_values) == len(query_ids) and baseline_values:
                    slice_comparisons[metric] = compare(
                        metric,
                        baseline_values,
                        candidate_values,
                        clusters=clusters,
                        higher_is_better=(
                            METRIC_DIRECTIONS.get(metric, "higher") == "higher"
                        ),
                        resamples=resamples,
                        seed=seed,
                    )
            corrected_slice = holm_correct(slice_comparisons)
            metrics = {
                metric: _comparison_record(result)
                for metric, result in corrected_slice.items()
            }
            for alias, canonical in METRIC_REPORT_ALIASES.items():
                if canonical in metrics:
                    metrics[alias] = _alias_comparison_record(
                        alias, canonical, metrics[canonical]
                    )
            slice_table[slice_name] = {
                "n": len(query_ids),
                "holm_family": "metrics_within_this_slice",
                "metrics": metrics,
            }

    candidate_repeat_deterministic = all(
        first[field] == second[field]
        for first, second in (production, accuracy)
        for field in ("ranking_hash", "decision_result_hash")
    )
    candidate_failures = sum(
        bool(_score(row).get("failed")) for row in candidate_rows
    ) + sum(bool(_score(row).get("failed")) for row in strict_rows)
    execution_failure_counts = {
        track: {
            key: int(artifact["failure_counts"].get(key, 0))
            for key in (
                "failed",
                "degraded",
                "skipped_or_unchunkable",
                "exceptions",
                "answerable_abstentions",
            )
        }
        for track, artifact in (
            ("production", first_production),
            ("accuracy_strict", first_accuracy),
        )
    }
    execution_failure_present = any(
        count
        for counts in execution_failure_counts.values()
        for count in counts.values()
    )
    regression_failures: list[str] = []
    for slice_name, slice_result in slice_table.items():
        if slice_name == "all":
            continue
        for metric, values in slice_result["metrics"].items():
            if metric not in PUBLIC_METRICS:
                continue
            direction = METRIC_DIRECTIONS.get(metric, "higher")
            regression = -values["diff"] if direction == "higher" else values["diff"]
            if regression > 0.02:
                regression_failures.append(f"{slice_name}:{metric}:{regression:.6f}")

    required_positive = any(
        metric in corrected
        and corrected[metric].diff > 0.0
        and corrected[metric].diff_lo > 0.0
        for metric in ("required_evidence_recall10", "passage_accuracy1")
    )
    if unavailable:
        verdict = "inconclusive"
        verdict_reasons = sorted(set(unavailable))
    else:
        verdict_reasons = []
        if candidate_failures or execution_failure_present:
            verdict_reasons.append(
                "failed_degraded_skipped_or_unchunkable_queries_present"
            )
        if not candidate_repeat_deterministic:
            verdict_reasons.append("candidate_repeat_hashes_drifted")
        if not required_positive:
            verdict_reasons.append("required_positive_paired_improvement_not_proven")
        if regression_failures:
            verdict_reasons.append(
                "source_or_high_risk_slice_regressed_over_two_points"
            )
        verdict = "improved" if not verdict_reasons else "rejected"

    next_experiment = _select_next_experiment(strict_rows)
    comparison_records = {
        metric: _comparison_record(value) for metric, value in corrected.items()
    }
    for alias, canonical in METRIC_REPORT_ALIASES.items():
        if canonical in comparison_records:
            comparison_records[alias] = _alias_comparison_record(
                alias, canonical, comparison_records[canonical]
            )
    artifact = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": first_production["run_id"],
        "identity": first_production["identity"],
        "repeat_artifacts": {
            "production": [
                {"path": str(Path(path).absolute()), "sha256": file_sha256(path)}
                for path in production_paths
            ],
            "accuracy_strict": [
                {"path": str(Path(path).absolute()), "sha256": file_sha256(path)}
                for path in accuracy_strict_paths
            ],
        },
        "reproducibility": {
            "candidate": candidate_repeat_deterministic,
            "production_ranking_hashes": [row["ranking_hash"] for row in production],
            "production_decision_result_hashes": [
                row["decision_result_hash"] for row in production
            ],
            "accuracy_strict_ranking_hashes": [row["ranking_hash"] for row in accuracy],
            "accuracy_strict_decision_result_hashes": [
                row["decision_result_hash"] for row in accuracy
            ],
            "baseline_ranking_hashes": baseline_ranking_hashes,
            "baseline_decision_result_hashes": baseline_decision_hashes,
        },
        "paired_method": {
            "confidence": 0.95,
            "bootstrap": "document/version-family clustered paired bootstrap",
            "test": "cluster sign-flip",
            "multiplicity": "Holm family-wise correction",
            "holm_families": {
                "overall": "all canonical public metrics",
                "slices": "canonical public metrics within each reported slice",
            },
            "resamples": resamples,
            "seed": seed,
            "n": len(candidate_ids) if baseline_ids == candidate_ids else 0,
        },
        "metrics": comparison_records,
        "slices": slice_table,
        "fixed_incomplete_source_policy": fixed_zero_checks,
        "not_labeled": {
            "supremecourt": "not_labeled",
            "current_version": "not_labeled",
            "historical_version": "not_labeled",
            "labels_inferred": False,
        },
        "candidate_failure_count_across_tracks": candidate_failures,
        "execution_failure_counts_by_track": execution_failure_counts,
        "slice_regressions_over_two_points": regression_failures,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "next_experiment": next_experiment,
        "scope_disclaimer": RETRIEVAL_ONLY_DISCLAIMER,
    }
    return _create_json(output, artifact)


def create_handoff_report(
    *,
    output: Path,
    comparison_path: Path,
    source_counts: Mapping[str, Any],
    quarantine_counts: Mapping[str, Any],
    commands_and_checks: Sequence[Mapping[str, Any]],
    external_resource_usage: Mapping[str, Any],
    remaining_limitations: Sequence[str],
) -> Path:
    """Create the final retrieval-only handoff without implying promotion approval."""

    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    if comparison.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise RetrievalReleaseError("handoff requires a retrieval release comparison")
    artifact = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": comparison["run_id"],
        "physical_collection": comparison["identity"]["physical_collection"],
        "exact_artifacts": {
            "comparison": {
                "path": str(Path(comparison_path).absolute()),
                "sha256": file_sha256(comparison_path),
            },
            "repeats": comparison["repeat_artifacts"],
        },
        "identity_hashes": {
            key: value
            for key, value in comparison["identity"].items()
            if key.endswith("_sha256")
        },
        "source_counts": dict(source_counts),
        "quarantine_counts": dict(quarantine_counts),
        "metric_table": comparison["metrics"],
        "slice_table": comparison["slices"],
        "reproducibility": comparison["reproducibility"],
        "commands_and_checks_actually_run": [dict(row) for row in commands_and_checks],
        "paid_and_external_resource_usage": dict(external_resource_usage),
        "remaining_limitations": list(remaining_limitations),
        "retrieval_only_verdict": comparison["verdict"],
        "verdict_reasons": comparison["verdict_reasons"],
        "selected_next_experiment": comparison["next_experiment"],
        "promotion_plan_created": False,
        "alias_operation_invoked": False,
        "scope_disclaimer": RETRIEVAL_ONLY_DISCLAIMER,
    }
    return _create_json(output, artifact)


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "HANDOFF_SCHEMA_VERSION",
    "REPEAT_SCHEMA_VERSION",
    "RETRIEVAL_ONLY_DISCLAIMER",
    "RetrievalReleaseError",
    "compare_release_to_baseline",
    "create_handoff_report",
    "file_sha256",
    "load_repeat",
    "persist_repeat",
    "run_release_repeats",
]
