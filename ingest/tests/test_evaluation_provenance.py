from dataclasses import replace
from types import SimpleNamespace

import pytest

from eval.evaluate import (
    _resolve_physical_collection,
    build_evaluation_provenance,
    qdrant_deps,
)
from ingest.config import load_config


def _index_info(**overrides):
    values = {
        "collection_alias": "georgian_legal",
        "serving_alias": "georgian_legal",
        "physical_collection": "georgian_legal__gen_20260713",
        "queried_collection": "georgian_legal",
        "access_kind": "serving_alias",
        "generation_id": "20260713",
        "n_points": 17,
        "corpus_hash": "1" * 64,
        "snapshot_hash": "2" * 64,
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "a" * 40,
        "tokenizer_model": "BAAI/bge-m3",
        "tokenizer_revision": "b" * 40,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "c" * 40,
        "vector_space_id": "3" * 64,
        "chunk_config_id": "4" * 64,
        "header_config_id": "5" * 64,
        "retrieval_fingerprint_revision": 2,
        "retrieval_fingerprint": "6" * 64,
        "dependency_identity": "7" * 64,
        "image_identity": "sha256:" + "8" * 64,
        "git_sha": "d" * 40,
        "dirty_patch_hash": None,
    }
    values.update(overrides)
    return values


class _Aliases:
    def __init__(self, entries):
        self.entries = entries

    def get_aliases(self):
        return SimpleNamespace(aliases=self.entries)


class _NoAliasLookup:
    def get_aliases(self):  # pragma: no cover - a call is the assertion failure
        raise AssertionError("direct physical evaluation must not inspect aliases")


def test_alias_resolution_requires_exact_generation_target():
    client = _Aliases(
        [
            SimpleNamespace(
                alias_name="georgian_legal",
                collection_name="georgian_legal__gen_20260713",
            )
        ]
    )
    assert (
        _resolve_physical_collection(client, "georgian_legal", "20260713")
        == "georgian_legal__gen_20260713"
    )

    wrong = _Aliases(
        [
            SimpleNamespace(
                alias_name="georgian_legal",
                collection_name="georgian_legal__gen_20260712",
            )
        ]
    )
    with pytest.raises(RuntimeError, match="must resolve exactly"):
        _resolve_physical_collection(wrong, "georgian_legal", "20260713")


def test_direct_physical_resolution_never_reads_alias_inventory():
    physical = "georgian_legal__gen_20260713"
    assert _resolve_physical_collection(_NoAliasLookup(), physical, "20260713") == physical


@pytest.mark.parametrize("entries", ([], [
    SimpleNamespace(
        alias_name="georgian_legal",
        collection_name="georgian_legal__gen_20260713",
    ),
    SimpleNamespace(
        alias_name="georgian_legal",
        collection_name="georgian_legal__gen_20260713",
    ),
]))
def test_alias_resolution_rejects_missing_or_duplicate_mapping(entries):
    with pytest.raises(RuntimeError, match="must resolve exactly"):
        _resolve_physical_collection(_Aliases(entries), "georgian_legal", "20260713")


def test_collection_target_rejects_any_non_serving_alias_name():
    with pytest.raises(RuntimeError, match="evaluation target must be"):
        _resolve_physical_collection(_NoAliasLookup(), "scratch", "20260713")


def test_provenance_builder_binds_frozen_sets_and_clean_patch_identity():
    result = build_evaluation_provenance(
        _index_info(), {"golden_v2": "9" * 64}, "production"
    )

    payload = result.to_dict()
    assert payload["physical_collection"] == "georgian_legal__gen_20260713"
    assert payload["serving_alias"] == "georgian_legal"
    assert payload["queried_collection"] == "georgian_legal"
    assert payload["access_kind"] == "serving_alias"
    assert payload["frozen_set_hashes"] == {"golden_v2": "9" * 64}
    assert payload["dirty_patch_hash"] == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )


def test_provenance_builder_rejects_unpinned_release_image():
    with pytest.raises(ValueError, match="image_identity"):
        build_evaluation_provenance(
            _index_info(image_identity=None),
            {"golden_v2": "9" * 64},
            "production",
        )


def test_qdrant_eval_rejects_missing_generation_before_model_load(monkeypatch):
    import ingest.qdrant_store as qdrant_store

    monkeypatch.setattr(qdrant_store, "make_client", lambda _cfg: object())
    cfg = replace(load_config(), generation_dir=None)

    with pytest.raises(RuntimeError, match="requires GENERATION_DIR"):
        qdrant_deps(cfg)


def test_candidate_qdrant_eval_requires_explicit_verification_pair_before_client(
    monkeypatch, tmp_path
):
    import ingest.generation as generation
    import ingest.qdrant_store as qdrant_store

    generation_id = "v3_512_candidate_20260715_01"
    generation_dir = tmp_path / generation_id
    generation_dir.mkdir()
    cfg = replace(
        load_config(), generation_id=generation_id, generation_dir=generation_dir
    )
    monkeypatch.setattr(
        generation,
        "load_generation",
        lambda _path: SimpleNamespace(
            manifest=SimpleNamespace(generation_id=generation_id)
        ),
    )

    def explode(_cfg):
        raise AssertionError("Qdrant client must not be constructed")

    monkeypatch.setattr(qdrant_store, "make_client", explode)
    with pytest.raises(RuntimeError, match="two explicit run-specific"):
        qdrant_deps(cfg)


def test_qdrant_eval_provenance_uses_verified_manifest_model_identity(
    monkeypatch, tmp_path
):
    import ingest.artifacts as artifacts
    import ingest.collection_compatibility as collection_compatibility
    import ingest.embedding as embedding
    import ingest.generation as generation
    import ingest.qdrant_store as qdrant_store

    generation_id = "gen_20260713_test"
    generation_dir = tmp_path / generation_id
    generation_dir.mkdir()
    cfg = replace(
        load_config(),
        generation_id=generation_id,
        generation_dir=generation_dir,
        rerank_enabled=False,
    )
    manifest = SimpleNamespace(
        generation_id=generation_id,
        source=SimpleNamespace(state_sha256="1" * 64),
        corpus=SimpleNamespace(snapshot_sha256="2" * 64),
        model=SimpleNamespace(
            embedding_model="manifest-embedder",
            embedding_revision="a" * 40,
            tokenizer_model="manifest-tokenizer",
            tokenizer_revision="b" * 40,
            reranker_model="manifest-reranker",
            reranker_revision="c" * 40,
        ),
        vector_space=SimpleNamespace(id="3" * 64),
        chunking=SimpleNamespace(fingerprint="4" * 64, document_header=True),
        retrieval_fingerprint_revision=2,
        retrieval_fingerprint="5" * 64,
        dependency=SimpleNamespace(
            lock_sha256="6" * 64,
            image_digest="sha256:" + "7" * 64,
        ),
        code=SimpleNamespace(git_sha="d" * 40, dirty_patch_sha256=None),
    )
    client = _Aliases(
        [
            SimpleNamespace(
                alias_name=cfg.collection_name,
                collection_name=f"georgian_legal__gen_{generation_id}",
            )
        ]
    )
    checked = []

    monkeypatch.setattr(qdrant_store, "make_client", lambda _cfg: client)
    monkeypatch.setattr(
        generation,
        "load_generation",
        lambda _path: SimpleNamespace(manifest=manifest),
    )
    monkeypatch.setattr(
        artifacts,
        "load_verified_generation_coverage",
        lambda _root: (
            SimpleNamespace(
                generation_id=generation_id,
                manifest_path=(generation_dir / "manifest.json").resolve(),
            ),
        ),
    )
    monkeypatch.setattr(
        collection_compatibility,
        "require_config_manifest_compatibility",
        lambda actual_cfg, actual_manifest: checked.append(
            (actual_cfg, actual_manifest)
        ),
    )
    monkeypatch.setattr(
        collection_compatibility,
        "require_collection_compatibility",
        lambda *_args: SimpleNamespace(points_count=17),
    )
    monkeypatch.setattr(embedding, "BGEM3Embedder", lambda _cfg: "embedder")

    _, _, reranker, index_info = qdrant_deps(cfg)

    assert checked == [(cfg, manifest)]
    assert reranker is None
    assert index_info["tokenizer_model"] == "manifest-tokenizer"
    assert index_info["reranker_model"] == "manifest-reranker"
    assert index_info["reranker_revision"] == "c" * 40
