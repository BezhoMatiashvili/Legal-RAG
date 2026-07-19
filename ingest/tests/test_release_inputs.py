from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ingest import gpu_workflow
from ingest import release_inputs as ri
from ingest.embed_job import EmbedStateError
from scripts import validate_candidate_release


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(value).hexdigest(),
        "size_bytes": len(value),
    }


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _queries() -> list[dict[str, object]]:
    values = []
    for line in (
        (REPO_ROOT / "ingest/eval/golden_set_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.strip() and not line.startswith("#"):
            values.append(json.loads(line))
    assert len(values) == 337
    return values


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "bundle"
    root.mkdir()
    revisions = {"embedding": "1" * 40, "tokenizer": "2" * 40, "reranker": "3" * 40}
    models = {}
    for role, revision in revisions.items():
        artifact = f"offline-{role}.tar"
        artifact_bytes = f"offline {role} model".encode()
        license_name = f"{role}-license-attestation.json"
        license_bytes = json.dumps(
            {
                "model_id": (
                    "BAAI/bge-m3" if role != "reranker" else "BAAI/bge-reranker-v2-m3"
                ),
                "revision": revision,
                "version": "offline-release-v1",
                "role": "reranker" if role == "reranker" else "embedder",
                "license_id": "Apache-2.0",
                "commercial_use_allowed": True,
                "weights_private_deployment_allowed": True,
                "training_data_use_allowed": True,
                "reviewed_by": "release-legal-reviewer",
                "reviewed_at": "2026-07-15",
                "authoritative_source_url": "https://licenses.example.org/model",
            },
            sort_keys=True,
        ).encode()
        (root / artifact).write_bytes(artifact_bytes)
        (root / license_name).write_bytes(license_bytes)
        models[role] = {
            "name": "BAAI/bge-m3" if role != "reranker" else "BAAI/bge-reranker-v2-m3",
            "revision": revision,
            "artifact_path": artifact,
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "artifact_size_bytes": len(artifact_bytes),
            "license_path": license_name,
            "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
            "license_size_bytes": len(license_bytes),
        }

    dependency_artifacts = {
        "torch": ("torch.whl", b"offline torch CUDA wheel"),
        "qdrant-client": ("qdrant_client.whl", b"offline qdrant client wheel"),
    }
    for _, (path, payload) in dependency_artifacts.items():
        (root / path).write_bytes(payload)
    torch_sha = hashlib.sha256(dependency_artifacts["torch"][1]).hexdigest()
    qdrant_client_sha = hashlib.sha256(
        dependency_artifacts["qdrant-client"][1]
    ).hexdigest()
    lock = (
        "--extra-index-url https://download.pytorch.org/whl/cu128\n"
        f"torch==2.9.0+cu128 --hash=sha256:{torch_sha}\n"
        f"qdrant-client==1.15.0 --hash=sha256:{qdrant_client_sha}\n"
    ).encode()
    (root / "requirements.lock").write_bytes(lock)
    oci_reference = "registry.example.org/legal@sha256:" + "8" * 64
    qdrant_revision = "v1.15.0"
    qdrant_archive_url = (
        "https://github.com/qdrant/qdrant/releases/download/v1.15.0/"
        "qdrant-x86_64-unknown-linux-musl.tar.gz"
    )
    runtime_identity = json.dumps(
        {
            "schema_version": 1,
            "status": "validated",
            "platform": "linux/amd64",
            "runtime_revision": "7" * 40,
            "base_image": oci_reference,
            "python_version": "3.11.9",
            "torch_version": "2.9.0+cu128",
            "torch_wheel_sha256": torch_sha,
            "cuda_build": "cu128",
            "cuda_runtime_version": "12.8",
            "pytorch_index_url": "https://download.pytorch.org/whl/cu128",
            "requirements_lock_sha256": hashlib.sha256(lock).hexdigest(),
            "qdrant_version": qdrant_revision,
            "qdrant_archive_url": qdrant_archive_url,
            "qdrant_archive_sha256": "c" * 64,
            "qdrant_checksum_source_url": (
                "https://github.com/qdrant/qdrant/releases/download/v1.15.0/SHA256SUMS"
            ),
            "qdrant_checksum_evidence_sha256": "d" * 64,
            "known_good_worker_artifact_sha256": "e" * 64,
        },
        sort_keys=True,
    ).encode()
    (root / "runtime.json").write_bytes(runtime_identity)

    knobs = {
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
    configuration_hash = _canonical_hash(knobs)

    baseline_root = root / "baseline"
    baseline_root.mkdir()
    trace_rows = []
    hash_rows = []
    for query_index, query in enumerate(_queries()):
        query_id = query["id"]
        candidates = [
            {
                "point_id": f"00000000-0000-0000-{query_index:04d}-{rank:012d}",
                "score": float(80 - rank),
            }
            for rank in range(80)
        ]
        final = candidates[:10]
        fixed_zero = query["source"] in {"tas", "tbappeal"}
        score = {
            "id": query_id,
            "query_type": query["query_type"],
            "language": query["query_language"],
            "cluster_id": f"cluster:{query_index}",
            "success1": 0.0,
            "success5": 0.0,
            "success10": 0.0,
            "required_evidence_recall10": 0.0,
            "candidate_recall50": 0.0,
            "candidate_recall80": 0.0,
            "document_identity1": 0.0,
            "passage_accuracy1": 0.0,
            "context_duplication10": 0.0,
            "context_noise10": 0.0,
            "ndcg10": 0.0,
            "mrr10": 0.0,
            "failed": fixed_zero,
            "failure_reason": (
                "frozen_incomplete_source_label" if fixed_zero else None
            ),
            "source": query["source"],
            "tags": [],
            "risk_level": "",
            "expected_outcome": "answer",
        }
        trace_rows.append(
            {
                "query_id": query_id,
                "cluster_id": f"cluster:{query_index}",
                "status": "zero_score_failure" if fixed_zero else "success",
                "configuration_hash": configuration_hash,
                "raw_candidates": candidates,
                "final_ranking": final,
                "branch_provenance": {
                    "route": "hybrid",
                    "branches": [
                        {
                            "name": "hybrid",
                            "point_ids": [item["point_id"] for item in candidates],
                        }
                    ],
                },
                "entity_matches": {
                    "query_entities": [],
                    "matched_point_ids": [],
                },
                "document_matches": {
                    "expected_document_id": query["document_id"],
                    "matched_point_ids": [],
                },
                "route_decision": {
                    "original": "hybrid",
                    "translated": None,
                    "selected": "original",
                },
                "degraded": False,
                "timings_ms": {"total": 1.0},
                "score": score,
            }
        )
        hash_rows.append(
            {
                "query_id": query_id,
                "raw_candidates": candidates,
                "final_ranking": final,
                "branch_provenance": {
                    "route": "hybrid",
                    "branches": [
                        {
                            "name": "hybrid",
                            "point_ids": [item["point_id"] for item in candidates],
                        }
                    ],
                },
                "entity_matches": {
                    "query_entities": [],
                    "matched_point_ids": [],
                },
                "document_matches": {
                    "expected_document_id": query["document_id"],
                    "matched_point_ids": [],
                },
                "route_decision": {
                    "original": "hybrid",
                    "translated": None,
                    "selected": "original",
                },
                "score": score,
            }
        )
    ranking_hash = _canonical_hash(
        [
            {
                "query_id": row["query_id"],
                "raw_candidates": row["raw_candidates"],
                "final_ranking": row["final_ranking"],
            }
            for row in hash_rows
        ]
    )
    decision_hash = _canonical_hash(hash_rows)
    repeats = []
    trace_bytes = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in trace_rows
    )
    for repeat in (1, 2):
        name = f"production-repeat-{repeat}.jsonl"
        (baseline_root / name).write_bytes(trace_bytes)
        repeats.append(
            {
                "repeat": repeat,
                "track": "production",
                "trace_path": name,
                "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
                "trace_size_bytes": len(trace_bytes),
                "ranking_hash": ranking_hash,
                "decision_result_hash": decision_hash,
            }
        )
    baseline_manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "v2",
            "query_count": 337,
            "golden_set_sha256": ri.GOLDEN_SET_SHA256,
            "translation_sha256": ri.TRANSLATION_SHA256,
            "holdout_sha256": ri.HOLDOUT_SHA256,
        },
        "collection": {
            "physical_collection": "georgian_legal__gen_v2_frozen_baseline",
            "generation_id": "v2_frozen_baseline",
            "snapshot_sha256": "4" * 64,
            "collection_sha256": "5" * 64,
            "configuration_sha256": configuration_hash,
        },
        "models": {
            role: {
                "name": item["name"],
                "revision": item["revision"],
                "artifact_sha256": item["artifact_sha256"],
            }
            for role, item in models.items()
        },
        "configuration_hash": configuration_hash,
        "repeats": repeats,
    }
    baseline_bytes = json.dumps(
        baseline_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    (baseline_root / "manifest.json").write_bytes(baseline_bytes)

    manifest = {
        "schema_version": 1,
        "release": {
            "snapshot_id": ri.SNAPSHOT_ID,
            "generation_id": ri.GENERATION_ID,
            "physical_collection": ri.PHYSICAL_COLLECTION,
            "crawl_start_date": ri.CRAWL_START_DATE,
            "crawl_end_date": ri.CRAWL_END_DATE,
            "sources": list(ri.SOURCES),
        },
        "code": {
            "repository_revision": ri._current_repository_revision(REPO_ROOT),
            "code_identity_sha256": ri.code_identity_sha256(REPO_ROOT),
            "retriever_revision": revisions["embedding"],
            "tokenizer_revision": revisions["tokenizer"],
            "reranker_revision": revisions["reranker"],
        },
        "models": models,
        "dependencies": {
            "lock_path": "requirements.lock",
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
            "lock_size_bytes": len(lock),
            "files": [
                {
                    "name": name,
                    "path": path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, (path, payload) in dependency_artifacts.items()
            ],
        },
        "runtime": {
            "identity_path": "runtime.json",
            "identity_sha256": hashlib.sha256(runtime_identity).hexdigest(),
            "identity_size_bytes": len(runtime_identity),
            "runtime_revision": "7" * 40,
            "qdrant_revision": qdrant_revision,
            "oci_reference": oci_reference,
            "oci_digest": "sha256:" + "8" * 64,
            "environment": {
                "SNAPSHOT_ID": ri.SNAPSHOT_ID,
                "GENERATION_ID": ri.GENERATION_ID,
                "COLLECTION_NAME": ri.PHYSICAL_COLLECTION,
                "EMBED_MODEL": "BAAI/bge-m3",
                "TOKENIZER_MODEL": "BAAI/bge-m3",
                "RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
                "EMBED_REVISION": revisions["embedding"],
                "TOKENIZER_REVISION": revisions["tokenizer"],
                "RERANK_REVISION": revisions["reranker"],
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
                "EMBED_DEVICE": "cuda",
                "EMBED_USE_FP16": "true",
                "EMBED_BATCH_SIZE": "256",
                "PRODUCTION_MODE": "true",
            },
        },
        "retrieval": {"configuration_hash": configuration_hash, "knobs": knobs},
        "operator": {"actor": "release-operator", "run_id": "candidate-run-01"},
        "baseline": {
            "root": "baseline",
            "manifest_path": "manifest.json",
            "manifest_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "manifest_size_bytes": len(baseline_bytes),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return root, dict(manifest["runtime"]["environment"])


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _rebind_baseline(root: Path, baseline: dict[str, object]) -> None:
    baseline_bytes = json.dumps(
        baseline, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    (root / "baseline/manifest.json").write_bytes(baseline_bytes)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["baseline"]["manifest_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
    manifest["baseline"]["manifest_size_bytes"] = len(baseline_bytes)
    _write_manifest(root, manifest)


def test_absent_bundle_stops_before_execution(tmp_path):
    with pytest.raises(ri.ReleaseInputError, match="bundle is absent"):
        ri.validate_release_inputs(tmp_path / "absent", repo_root=REPO_ROOT, environ={})


def test_validates_exact_release_and_pairable_baseline(tmp_path):
    root, environ = _bundle(tmp_path)
    result = ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)

    assert result.configuration_hash == result.baseline.configuration_hash
    assert result.code_identity_sha256 == ri.code_identity_sha256(REPO_ROOT)
    assert len(result.baseline.query_ids) == 337
    assert result.baseline.ordered_query_ids == result.baseline.query_ids
    assert len(result.baseline.queries) == 337
    assert result.baseline.ranking_hashes == (
        result.baseline.ranking_hash,
        result.baseline.ranking_hash,
    )
    fixed_zero = [
        row
        for row in result.baseline.queries
        if row["score"]["source"] in {"tas", "tbappeal"}
    ]
    assert len(fixed_zero) == 56
    assert all(row["score"]["failed"] is True for row in fixed_zero)
    assert (
        result.baseline.ranking_hash == result.baseline.decision_result_hash
        or len(result.baseline.decision_result_hash) == 64
    )
    assert result.report()["baseline"]["query_count"] == 337
    assert result.report()["code_identity_sha256"] == result.code_identity_sha256


def test_gpu_workflow_plan_uses_real_release_validator_signature(tmp_path, monkeypatch):
    root, environ = _bundle(tmp_path)
    launch_directory = tmp_path / "unrelated-launch-directory"
    launch_directory.mkdir()
    monkeypatch.chdir(launch_directory)

    # The real release validator must accept and validate the workflow's bound
    # repository root.  Stop at the immediately following snapshot check so this
    # regression cannot be hidden by a permissive validator monkeypatch.
    with pytest.raises(EmbedStateError, match="cannot stat snapshot docs"):
        gpu_workflow.create_workflow_plan(
            bundle_root=root,
            snapshot_docs=tmp_path / "missing-snapshot" / "docs",
            cpu_checksum=tmp_path / "missing-checksum.json",
            storage_identity=tmp_path / "missing-volume.identity",
            output=tmp_path / "workflow.json",
            workflow_id="release-signature-regression",
            container_paths={},
            volume_size_gib=120,
            worker_count=1,
            gpu_sku="NVIDIA-L40S",
            gpu_count=1,
            total_hourly_usd=1.0,
            storage_gib_month_usd=0.1,
            max_runtime_hours=4.0,
            max_exposure_usd=5.0,
            auto_teardown=True,
            environ=environ,
        )


def test_rejects_environment_drift_before_baseline_use(tmp_path):
    root, _ = _bundle(tmp_path)
    with pytest.raises(ri.ReleaseInputError, match="environment drift"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ={})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("EMBED_DEVICE", "cpu", "EMBED_DEVICE"),
        ("EMBED_USE_FP16", "1", "EMBED_USE_FP16"),
        ("EMBED_BATCH_SIZE", "0", "EMBED_BATCH_SIZE"),
        ("EMBED_BATCH_SIZE", "0256", "EMBED_BATCH_SIZE"),
    ],
)
def test_rejects_unreviewable_embedding_runtime_environment(
    tmp_path, name, value, message
):
    root, environ = _bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["environment"][name] = value
    _write_manifest(root, manifest)
    environ[name] = value

    with pytest.raises(ri.ReleaseInputError, match=message):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_rejects_secret_bearing_reportable_fields(tmp_path):
    root, environ = _bundle(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["runtime"]["environment"]["API_KEY"] = "not-even-a-real-secret"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ri.ReleaseInputError, match="secret-bearing field"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_rejects_non_pairable_repeat_hash(tmp_path):
    root, environ = _bundle(tmp_path)
    baseline_path = root / "baseline/manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["repeats"][1]["ranking_hash"] = "f" * 64
    baseline_bytes = json.dumps(
        baseline, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    baseline_path.write_bytes(baseline_bytes)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline"]["manifest_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
    manifest["baseline"]["manifest_size_bytes"] = len(baseline_bytes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ri.ReleaseInputError, match="ranking_hash"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_rejects_code_identity_that_only_binds_head(tmp_path):
    root, environ = _bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["code"]["code_identity_sha256"] = "f" * 64
    _write_manifest(root, manifest)

    with pytest.raises(ri.ReleaseInputError, match="tracked and untracked"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_strict_supply_chain_parser_rejects_fake_hash_text(tmp_path):
    root, environ = _bundle(tmp_path)
    malformed = b"torch==2.9.0+cu128 --hash=sha256:not-a-digest\n"
    (root / "requirements.lock").write_bytes(malformed)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["dependencies"]["lock_sha256"] = hashlib.sha256(malformed).hexdigest()
    manifest["dependencies"]["lock_size_bytes"] = len(malformed)
    _write_manifest(root, manifest)

    with pytest.raises(ri.ReleaseInputError, match="exact name==version"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_malformed_utf8_lock_is_a_release_input_error(tmp_path):
    root, environ = _bundle(tmp_path)
    malformed = b"torch==2.9.0 --hash=sha256:" + b"a" * 64 + b"\xff"
    (root / "requirements.lock").write_bytes(malformed)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["dependencies"]["lock_sha256"] = hashlib.sha256(malformed).hexdigest()
    manifest["dependencies"]["lock_size_bytes"] = len(malformed)
    _write_manifest(root, manifest)

    with pytest.raises(ri.ReleaseInputError, match="strict UTF-8"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_runtime_identity_requires_complete_strict_evidence(tmp_path):
    root, environ = _bundle(tmp_path)
    runtime_path = root / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime["known_good_worker_artifact_sha256"]
    runtime_bytes = json.dumps(runtime, sort_keys=True).encode()
    runtime_path.write_bytes(runtime_bytes)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["runtime"]["identity_sha256"] = hashlib.sha256(runtime_bytes).hexdigest()
    manifest["runtime"]["identity_size_bytes"] = len(runtime_bytes)
    _write_manifest(root, manifest)

    with pytest.raises(ri.ReleaseInputError, match="known-good worker evidence"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_rejects_any_retrieval_knob_drift_even_when_self_hashed(tmp_path):
    root, environ = _bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["retrieval"]["knobs"]["rerank_enabled"] = False
    manifest["retrieval"]["configuration_hash"] = _canonical_hash(
        manifest["retrieval"]["knobs"]
    )
    _write_manifest(root, manifest)

    with pytest.raises(ri.ReleaseInputError, match="complete frozen"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_baseline_model_must_bind_exact_release_artifact(tmp_path):
    root, environ = _bundle(tmp_path)
    baseline = json.loads((root / "baseline/manifest.json").read_text(encoding="utf-8"))
    baseline["models"]["embedding"]["artifact_sha256"] = "f" * 64
    _rebind_baseline(root, baseline)

    with pytest.raises(ri.ReleaseInputError, match="model artifact is not pairable"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_baseline_trace_provenance_cannot_be_empty_mapping(tmp_path):
    root, environ = _bundle(tmp_path)
    trace_paths = [
        root / "baseline/production-repeat-1.jsonl",
        root / "baseline/production-repeat-2.jsonl",
    ]
    rows = [json.loads(line) for line in trace_paths[0].read_text().splitlines()]
    rows[0]["branch_provenance"] = {}
    trace_bytes = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in rows
    )
    decision_rows = [
        {
            "query_id": row["query_id"],
            "raw_candidates": row["raw_candidates"],
            "final_ranking": row["final_ranking"],
            "branch_provenance": row["branch_provenance"],
            "entity_matches": row["entity_matches"],
            "document_matches": row["document_matches"],
            "route_decision": row["route_decision"],
            "score": row["score"],
        }
        for row in rows
    ]
    baseline = json.loads((root / "baseline/manifest.json").read_text(encoding="utf-8"))
    for path, repeat in zip(trace_paths, baseline["repeats"]):
        path.write_bytes(trace_bytes)
        repeat["trace_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
        repeat["trace_size_bytes"] = len(trace_bytes)
        repeat["decision_result_hash"] = _canonical_hash(decision_rows)
    _rebind_baseline(root, baseline)

    with pytest.raises(ri.ReleaseInputError, match="branch_provenance keys mismatch"):
        ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_large_model_artifact_is_streamed_not_read_bytes(tmp_path, monkeypatch):
    root, environ = _bundle(tmp_path)
    blocked = root / "offline-embedding.tar"
    original = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == blocked:
            raise AssertionError("model artifact must be streamed")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    ri.validate_release_inputs(root, repo_root=REPO_ROOT, environ=environ)


def test_validator_emits_inconclusive_report_for_unexpected_gate_exception(
    tmp_path, monkeypatch
):
    report_path = tmp_path / "report.json"

    def fail(*args, **kwargs):
        del args, kwargs
        raise TypeError("malformed nested field")

    monkeypatch.setattr(validate_candidate_release, "validate_release_inputs", fail)
    result = validate_candidate_release.main(
        [
            "--bundle",
            ri.RELEASE_BUNDLE_PATH,
            "--repo-root",
            str(REPO_ROOT),
            "--report",
            str(report_path),
        ]
    )

    assert result == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "inconclusive"
    assert report["ceiling"] == "no_network_or_paid_work"
    assert report["reason"] == "malformed nested field"
