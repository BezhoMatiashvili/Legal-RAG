"""Executable, fail-closed evaluator for the immutable 512-token candidate.

This is the only authoritative evaluation entry point for the candidate.  It revalidates
the external release bundle and baseline, derives identity from the sealed generation and
two physical-verification reports, constructs one verified direct-physical backend, and
persists each of the four repeats before producing the paired comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ingest.config import load_config
from ingest.generation import load_generation
from ingest.release_inputs import (
    GENERATION_ID,
    GOLDEN_SET_SHA256,
    HOLDOUT_SHA256,
    PHYSICAL_COLLECTION,
    RELEASE_BUNDLE_PATH,
    SNAPSHOT_ID,
    TRANSLATION_SHA256,
    ValidatedReleaseInputs,
    validate_release_inputs,
)

from . import goldset
from .backend import ProductionBackend
from .evaluate import (
    _token_counter,
    build_evaluation_provenance,
    build_query_relevance,
    qdrant_deps,
)
from .release_verification import (
    ValidatedVerificationPair,
    validate_physical_verification_pair,
)
from .retrieval_release import (
    TRACKS,
    RetrievalReleaseError,
    _preflight_repeat_destinations,
    compare_release_to_baseline,
    run_release_repeats,
)
from .translations import load_query_translations
from .v2_candidate_qrels import (
    artifact_sha256,
    bind_gold_queries_to_candidate,
    load_v2_candidate_qrel_artifact,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_identity(document_header: bool) -> str:
    material = json.dumps(
        {"document_header": document_header}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_validated_bundle_manifest(inputs: ValidatedReleaseInputs) -> dict[str, Any]:
    path = inputs.root / "manifest.json"
    if _file_sha256(path) != inputs.manifest_sha256:
        raise RetrievalReleaseError("release manifest changed after validation")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # strict validator already enforces this shape
        raise RetrievalReleaseError("release manifest is not an object")
    return value


def build_release_identity(
    *,
    inputs: ValidatedReleaseInputs,
    generation_dir: Path,
    qrel_path: Path,
    verifications: ValidatedVerificationPair,
) -> dict[str, Any]:
    """Derive and cross-bind the complete repeat identity from sealed artifacts."""

    raw_release = _load_validated_bundle_manifest(inputs)
    artifacts = load_generation(generation_dir)
    manifest = artifacts.manifest
    collection = artifacts.collection_digest
    if collection is None:
        raise RetrievalReleaseError("candidate generation lacks collection_digest.json")
    if (
        manifest.generation_id != GENERATION_ID
        or manifest.corpus.name != SNAPSHOT_ID
        or collection.physical_collection != PHYSICAL_COLLECTION
    ):
        raise RetrievalReleaseError("generation is not the frozen 512 candidate tuple")

    release_models = raw_release["models"]
    generation_models = {
        "embedding": (
            manifest.model.embedding_model,
            manifest.model.embedding_revision,
        ),
        "tokenizer": (
            manifest.model.tokenizer_model,
            manifest.model.tokenizer_revision,
        ),
        "reranker": (
            manifest.model.reranker_model,
            manifest.model.reranker_revision,
        ),
    }
    for role, observed in generation_models.items():
        expected = (release_models[role]["name"], release_models[role]["revision"])
        if observed != expected:
            raise RetrievalReleaseError(
                f"generation {role} identity differs from the validated bundle"
            )
    runtime = raw_release["runtime"]
    code = raw_release["code"]
    dependencies = raw_release["dependencies"]
    operator = raw_release["operator"]
    if (
        manifest.code.git_sha != inputs.repository_revision
        or manifest.code.dirty_patch_sha256 is None
        or manifest.dependency.lock_sha256 != dependencies["lock_sha256"]
        or manifest.dependency.image_digest != inputs.oci_digest
        or manifest.creation.run_id != inputs.run_id
        or manifest.creation.actor != inputs.actor
    ):
        raise RetrievalReleaseError(
            "generation code/runtime/creation identity differs from the validated bundle"
        )
    if operator != {"actor": inputs.actor, "run_id": inputs.run_id}:
        raise RetrievalReleaseError("validated operator identity drifted")

    qrel_artifact = load_v2_candidate_qrel_artifact(qrel_path)
    candidate_snapshot = qrel_artifact["candidate_snapshot"]
    if (
        candidate_snapshot["snapshot_sha256"] != manifest.corpus.snapshot_sha256
        or candidate_snapshot["corpus_sha256"]
        != verifications.reports[0]["stats"]["structural_inventory_proof"][
            "corpus_sha256"
        ]
    ):
        raise RetrievalReleaseError(
            "candidate qrel artifact differs from the physically verified snapshot"
        )

    identity = {
        "run_id": inputs.run_id,
        "snapshot_id": SNAPSHOT_ID,
        "generation_id": GENERATION_ID,
        "physical_collection": PHYSICAL_COLLECTION,
        "snapshot_sha256": manifest.corpus.snapshot_sha256,
        "generation_sha256": _file_sha256(artifacts.root / "manifest.json"),
        "configuration_sha256": inputs.configuration_hash,
        "collection_sha256": collection.collection_sha256,
        "collection_configuration_sha256": (
            collection.collection_configuration_sha256
        ),
        "vector_checksum_sha256": collection.vector_checksum_artifact_sha256,
        "vector_probe_sha256": collection.vector_probe_sha256,
        "qrel_adapter_sha256": artifact_sha256(qrel_path),
        "baseline_manifest_sha256": inputs.baseline.manifest_sha256,
        "verification_1_sha256": verifications.sha256[0],
        "verification_2_sha256": verifications.sha256[1],
        "dependency_lock_sha256": dependencies["lock_sha256"],
        "runtime_identity_sha256": runtime["identity_sha256"],
        "code_identity_sha256": code["code_identity_sha256"],
        "embedding_artifact_sha256": release_models["embedding"]["artifact_sha256"],
        "tokenizer_artifact_sha256": release_models["tokenizer"]["artifact_sha256"],
        "reranker_artifact_sha256": release_models["reranker"]["artifact_sha256"],
        "points_count": collection.point_count,
        "corpus_hash": manifest.source.state_sha256,
        "vector_space_id": manifest.vector_space.id,
        "chunk_config_id": manifest.chunking.fingerprint,
        "header_config_id": _header_identity(manifest.chunking.document_header),
        "retrieval_fingerprint_revision": manifest.retrieval_fingerprint_revision,
        "retrieval_fingerprint": manifest.retrieval_fingerprint,
        "dirty_patch_hash": manifest.code.dirty_patch_sha256,
        "golden_set_sha256": GOLDEN_SET_SHA256,
        "translation_sha256": TRANSLATION_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "embedding_model": manifest.model.embedding_model,
        "embedding_revision": manifest.model.embedding_revision,
        "tokenizer_model": manifest.model.tokenizer_model,
        "tokenizer_revision": manifest.model.tokenizer_revision,
        "reranker_model": manifest.model.reranker_model,
        "reranker_revision": manifest.model.reranker_revision,
        "runtime_image_digest": inputs.oci_digest,
        "repository_revision": inputs.repository_revision,
    }
    # Use the repeat validator as the canonical final schema/cross-binding check without
    # writing anything or starting a client/model.
    from .retrieval_release import _validate_identity

    return _validate_identity(identity)


def _load_frozen_evaluation_inputs(cfg: Any, qrel_path: Path):
    spec = goldset.EVAL_SETS["v2"]
    if (
        _file_sha256(spec.gold) != GOLDEN_SET_SHA256
        or _file_sha256(spec.holdout) != HOLDOUT_SHA256
        or _file_sha256(goldset.EVAL_DIR / "query_translations_v2.json")
        != TRANSLATION_SHA256
    ):
        raise RetrievalReleaseError("checked-in frozen v2 inputs drifted")
    gold = goldset.load_golden_set(spec.gold)
    holdout = goldset.load_holdout(spec.holdout)
    bodies = goldset.SnapshotBodies(
        root=spec.roots[0], needed=goldset.gold_docs(gold), extra_roots=spec.roots[1:]
    )
    goldset.reground(gold, bodies)
    goldset.enforce_holdout(gold, holdout)
    qrel_artifact = load_v2_candidate_qrel_artifact(qrel_path)
    gold, qrel_failures = bind_gold_queries_to_candidate(gold, qrel_artifact)
    count_tokens = _token_counter(
        "bge", cfg.tokenizer_model, cfg.tokenizer_revision
    )
    chunk_cfg = {
        "max_tokens": cfg.chunk_tokens,
        "overlap": cfg.chunk_overlap,
        "min_tokens": cfg.chunk_min_tokens,
    }
    relevance = build_query_relevance(gold, bodies, chunk_cfg, count_tokens)
    for query_id, reason in qrel_failures.items():
        query_relevance = relevance[query_id]
        query_relevance["issues"].append(f"candidate_qrel:{reason}")
        query_relevance["failure_reason"] = (
            reason
            if reason == "frozen_incomplete_source_label"
            else f"candidate_qrel:{reason}"
        )
    translations, _translation_hash = load_query_translations(
        goldset.EVAL_DIR / "query_translations_v2.json", gold
    )
    return gold, relevance, translations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(RELEASE_BUNDLE_PATH))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--candidate-qrels", type=Path, required=True)
    parser.add_argument(
        "--physical-verification", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(args.physical_verification) != 2:
        raise SystemExit("--physical-verification must be supplied exactly twice")
    if args.resamples < 1:
        raise SystemExit("--resamples must be >= 1")
    if os.path.lexists(args.comparison_output):
        raise SystemExit("comparison output already exists")

    inputs = validate_release_inputs(args.bundle, repo_root=args.repo_root)
    verification_paths = (
        args.physical_verification[0],
        args.physical_verification[1],
    )
    verification_pair = validate_physical_verification_pair(
        args.generation_dir, verification_paths
    )
    identity = build_release_identity(
        inputs=inputs,
        generation_dir=args.generation_dir,
        qrel_path=args.candidate_qrels,
        verifications=verification_pair,
    )
    _preflight_repeat_destinations(args.output_dir, run_id=identity["run_id"])

    cfg = load_config()
    gold, relevance, translations = _load_frozen_evaluation_inputs(
        cfg, args.candidate_qrels
    )
    client, embedder, reranker, index_info = qdrant_deps(
        cfg, verification_paths=verification_paths
    )
    backend = ProductionBackend(
        cfg,
        client,
        embedder,
        reranker=reranker,
        translations=translations,
        release_index_info=index_info,
    )
    expected_frozen_hashes = {
        "golden_v2": GOLDEN_SET_SHA256,
        "holdout_v2": HOLDOUT_SHA256,
        "authored_query_translations": TRANSLATION_SHA256,
        "v2_candidate_qrels": identity["qrel_adapter_sha256"],
    }
    provenances = {
        track: build_evaluation_provenance(
            index_info, expected_frozen_hashes, track
        ).to_dict()
        for track in TRACKS
    }
    paths = run_release_repeats(
        output_dir=args.output_dir,
        backends={track: backend for track in TRACKS},
        gold=gold,
        relevance=relevance,
        identity=identity,
        provenances=provenances,
        baseline=inputs.baseline,
        level="chunk",
        top_k=10,
    )
    comparison = compare_release_to_baseline(
        production_paths=paths["production"],
        accuracy_strict_paths=paths["accuracy_strict"],
        baseline=inputs.baseline,
        output=args.comparison_output,
        require_frozen_v2=True,
        resamples=args.resamples,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "physical_collection": PHYSICAL_COLLECTION,
                "repeat_paths": {
                    track: [str(path) for path in paths[track]] for track in TRACKS
                },
                "comparison": str(comparison),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
