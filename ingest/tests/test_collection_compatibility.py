import dataclasses
from types import SimpleNamespace

import pytest

from ingest.collection_compatibility import (
    CollectionIncompatibleError,
    check_collection_compatibility,
    config_manifest_issues,
    expected_point_identity,
    require_collection_compatibility,
)
from ingest.config import load_config
from ingest.generation import GENERATION_SCHEMA_VERSION, GenerationManifest
from ingest.generation import CANONICAL_PAYLOAD_REQUIRED_NONEMPTY_FIELDS

GENERATION_ID = "20260713t120000z_core"
REVISION = "1" * 40
VECTOR_SPACE_ID = "2" * 64
CHUNKING_FINGERPRINT = "3" * 64
RETRIEVAL_FINGERPRINT = "4" * 64


def _manifest():
    return GenerationManifest.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "document_count": 1,
            "indexed_document_count": 1,
            "excluded_document_count": 0,
            "chunk_count": 2,
            "sample_count": 1,
            "corpus": {
                "name": "georgian_legal",
                "snapshot_sha256": "5" * 64,
            },
            "source": {"name": "snapshot-v3", "state_sha256": "6" * 64},
            "model": {
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": REVISION,
                "tokenizer_model": "BAAI/bge-m3",
                "tokenizer_revision": REVISION,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_revision": REVISION,
            },
            "vector_space": {
                "id": VECTOR_SPACE_ID,
                "dense_name": "dense",
                "dense_dimension": 3,
                "distance": "cosine",
                "sparse_name": "sparse",
            },
            "chunking": {
                "fingerprint": CHUNKING_FINGERPRINT,
                "max_tokens": 512,
                "overlap_tokens": 64,
                "document_header": True,
            },
            "covered_runs": [{"source": "matsne", "run_id": "20260713t100000z"}],
            "retrieval_fingerprint_revision": 2,
            "retrieval_fingerprint": RETRIEVAL_FINGERPRINT,
            "code": {"git_sha": "7" * 40, "dirty_patch_sha256": None},
            "dependency": {"lock_sha256": "8" * 64, "image_digest": None},
            "creation": {
                "created_at": "2026-07-13T12:00:00Z",
                "run_id": "build-20260713",
                "actor": "release-worker",
            },
        }
    )


def _info(*, points=2, size=3, distance="Cosine", sparse_names=("sparse",)):
    return SimpleNamespace(
        points_count=points,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    "dense": SimpleNamespace(size=size, distance=distance),
                },
                sparse_vectors={name: SimpleNamespace() for name in sparse_names},
            )
        ),
    )


class FakeClient:
    def __init__(self, info, identity_count=2):
        self.info = info
        self.identity_count = identity_count
        self.calls = []

    def get_collection(self, collection_name):
        self.calls.append(("get_collection", collection_name))
        if isinstance(self.info, Exception):
            raise self.info
        return self.info

    def count(self, **kwargs):
        self.calls.append(("count", kwargs))
        if isinstance(self.identity_count, Exception):
            raise self.identity_count
        return SimpleNamespace(count=self.identity_count)


def _filter_values(client):
    kwargs = client.calls[1][1]
    return {
        condition.key: condition.match.value
        for condition in kwargs["count_filter"].must
    }


def test_exact_compatibility_uses_only_read_operations_and_all_identity_fields():
    client = FakeClient(_info())
    manifest = _manifest()

    result = check_collection_compatibility(client, "candidate", manifest)

    assert result.ok
    assert result.points_count == manifest.chunk_count
    assert result.identity_matched_points == manifest.chunk_count
    assert [call[0] for call in client.calls] == ["get_collection", "count"]
    assert client.calls[1][1]["collection_name"] == "candidate"
    assert client.calls[1][1]["exact"] is True
    assert _filter_values(client) == {
        **expected_point_identity(manifest),
        "admissible": True,
    }
    required_nonempty = {
        condition.is_empty.key
        for condition in client.calls[1][1]["count_filter"].must_not
    }
    assert required_nonempty == set(CANONICAL_PAYLOAD_REQUIRED_NONEMPTY_FIELDS)


def test_schema_count_and_identity_mismatches_are_separate_fail_closed_issues():
    client = FakeClient(
        _info(points=3, size=4, distance="Dot", sparse_names=("sparse", "extra")),
        identity_count=0,
    )

    result = check_collection_compatibility(client, "candidate", _manifest())

    assert not result.ok
    coverage_codes = {issue.code for issue in result.coverage_issues}
    integrity_codes = {issue.code for issue in result.integrity_issues}
    assert coverage_codes == {"collection_point_count_mismatch"}
    assert {
        "dense_dimension_mismatch",
        "dense_distance_mismatch",
        "sparse_vector_names_mismatch",
        "identity_payload_count_mismatch",
    } <= integrity_codes


def test_legacy_collection_without_identity_markers_is_incompatible():
    client = FakeClient(_info(), identity_count=0)

    result = check_collection_compatibility(client, "legacy", _manifest())

    assert not result.ok
    assert "identity_payload_count_mismatch" in {
        issue.code for issue in result.integrity_issues
    }
    with pytest.raises(
        CollectionIncompatibleError, match="identity_payload_count_mismatch"
    ):
        require_collection_compatibility(client, "legacy", _manifest())


def test_unavailable_metadata_or_exact_identity_count_never_passes_open():
    metadata_failure = check_collection_compatibility(
        FakeClient(RuntimeError("offline")),
        "candidate",
        _manifest(),
    )
    assert not metadata_failure.ok
    assert metadata_failure.issues[0].code == "collection_info_unavailable"

    count_failure = check_collection_compatibility(
        FakeClient(_info(), identity_count=RuntimeError("count failed")),
        "candidate",
        _manifest(),
    )
    assert not count_failure.ok
    assert "identity_count_unavailable" in {
        issue.code for issue in count_failure.integrity_issues
    }


def test_runtime_config_must_match_manifest_vector_chunk_and_retrieval_identity():
    manifest = _manifest()
    cfg = dataclasses.replace(
        load_config(),
        generation_id=manifest.generation_id,
        embed_model=manifest.model.embedding_model,
        embedding_revision=manifest.model.embedding_revision,
        tokenizer_model=manifest.model.tokenizer_model,
        tokenizer_revision=manifest.model.tokenizer_revision,
        rerank_model=manifest.model.reranker_model,
        reranker_revision=manifest.model.reranker_revision,
        dense_dim=manifest.vector_space.dense_dimension,
        chunk_tokens=manifest.chunking.max_tokens,
        chunk_overlap=manifest.chunking.overlap_tokens,
        embed_header_v2=manifest.chunking.document_header,
        rerank_enabled=False,
    )
    # The fixture uses arbitrary hashes, so the three derived identities fail closed.
    fields = {issue.details["field"] for issue in config_manifest_issues(cfg, manifest)}
    assert fields == {
        "vector_space_id",
        "chunking_fingerprint",
        "retrieval_fingerprint",
    }

    mismatched = dataclasses.replace(cfg, embedding_revision="9" * 40)
    fields = {
        issue.details["field"]
        for issue in config_manifest_issues(mismatched, manifest)
    }
    assert "embedding_revision" in fields

    for field, config_field in (
        ("tokenizer_model", "tokenizer_model"),
        ("reranker_model", "rerank_model"),
        ("reranker_revision", "reranker_revision"),
    ):
        changed = dataclasses.replace(cfg, **{config_field: "9" * 40})
        mismatch_fields = {
            issue.details["field"]
            for issue in config_manifest_issues(changed, manifest)
        }
        assert field in mismatch_fields
