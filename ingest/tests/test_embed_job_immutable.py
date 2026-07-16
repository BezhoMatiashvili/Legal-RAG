from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from ingest import __main__ as primary_cli
from ingest import embed_job
from ingest.config import ConfigurationError, load_config


def _cfg(tmp_path, **over):
    generation_id = "gen_20260715_embed"
    cfg = dataclasses.replace(
        load_config(),
        generation_id=generation_id,
        collection_name=f"georgian_legal__gen_{generation_id}",
        embedding_revision="a" * 40,
        tokenizer_revision="b" * 40,
        reranker_revision="c" * 40,
        state_dir=tmp_path / "state",
    )
    return dataclasses.replace(cfg, **over)


def _sealed(tmp_path, cfg=None):
    cfg = cfg or _cfg(tmp_path)
    root = tmp_path / "snapshot"
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    return embed_job.SealedSnapshot(
        root=root.resolve(),
        docs=docs.resolve(),
        snapshot_id="v3_512_candidate_20260715",
        snapshot_sha256="d" * 64,
        corpus_sha256="e" * 64,
        sources=embed_job.SOURCES,
        manifest={
            "build": {
                "embed_model": cfg.embed_model,
                "embedding_revision": cfg.embedding_revision,
                "tokenizer": {
                    "model": cfg.tokenizer_model,
                    "revision": cfg.tokenizer_revision,
                },
                "chunk": {
                    "tokens": cfg.chunk_tokens,
                    "overlap": cfg.chunk_overlap,
                    "min_tokens": cfg.chunk_min_tokens,
                },
            }
        },
    )


def _doc(document_id: str, version_id: str):
    return SimpleNamespace(
        source="matsne",
        document_id=document_id,
        version_id=version_id,
    )


def _canonical_doc(document_id="1", version_id="derived:v1"):
    return embed_job.snapshot_doc_to_canonical(
        {
            "source": "matsne",
            "document_id": document_id,
            "body_markdown": "complete official body for deterministic resume proof",
            "language": "ka",
            "source_fingerprint": "f" * 64,
            "normalizer_revision": "canonical-v4",
            "version_id": version_id,
            "version_id_kind": "derived",
            "content_kind": "full_text",
            "content_complete": True,
            "extraction_status": "full_text",
            "version_lineage_status": "derived",
            "version_lineage_complete": False,
            "source_authority": "primary_official",
            "official_url": "https://example.test/law/1",
            "source_url": "https://example.test/law/1",
            "freshness_sla_met": True,
            "supersedes": [],
            "consolidated_dates": [],
        },
        strict=True,
    )


class _Client:
    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def upsert(self, collection_name, points, wait=False):
        self.calls.append((collection_name, [point.id for point in points], wait))
        if self.fail_on_call == len(self.calls):
            raise OSError("acknowledgement unavailable")


def test_binding_is_create_only_and_resume_requires_exact_full_tuple(tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = embed_job.prepare_binding(cfg, sealed, resume=False)

    assert binding.path == tmp_path / "state" / "embed" / cfg.generation_id / "binding.json"
    assert binding.value["retrieval"] == {
        "fingerprint_revision": 2,
        "fingerprint_sha256": binding.value["retrieval"]["fingerprint_sha256"],
    }
    assert binding.value["payload_schema"]["generation_schema_version"] == 2
    assert embed_job.prepare_binding(cfg, sealed, resume=True).value == binding.value

    with pytest.raises(embed_job.EmbedStateError, match="already exists"):
        embed_job.prepare_binding(cfg, sealed, resume=False)
    with pytest.raises(embed_job.EmbedStateError, match="configuration mismatch"):
        embed_job.prepare_binding(
            dataclasses.replace(cfg, chunk_tokens=cfg.chunk_tokens + 1),
            sealed,
            resume=True,
        )


def test_snapshot_build_config_is_exact_and_enforces_512_candidate(tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    embed_job.validate_snapshot_build_config(sealed, cfg)

    bad_manifest = json.loads(json.dumps(sealed.manifest))
    bad_manifest["build"]["tokenizer"]["revision"] = "9" * 40
    with pytest.raises(embed_job.EmbedStateError, match="tokenizer.revision"):
        embed_job.validate_snapshot_build_config(
            dataclasses.replace(sealed, manifest=bad_manifest), cfg
        )

    non_candidate = dataclasses.replace(cfg, chunk_tokens=513)
    with pytest.raises(embed_job.EmbedStateError, match="candidate chunk contract"):
        embed_job.validate_snapshot_build_config(
            _sealed(tmp_path, non_candidate), non_candidate
        )


def test_binding_refuses_symlinked_state_parent(tmp_path):
    cfg = _cfg(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    cfg.state_dir.mkdir()
    (cfg.state_dir / "embed").symlink_to(outside, target_is_directory=True)
    with pytest.raises(embed_job.EmbedStateError, match="symlink"):
        embed_job.prepare_binding(cfg, _sealed(tmp_path, cfg), resume=False)
    assert list(outside.iterdir()) == []


def test_resume_rejects_absent_corrupt_and_mismatched_checkpoint(tmp_path):
    binding = embed_job.prepare_binding(_cfg(tmp_path), _sealed(tmp_path), resume=False)
    with pytest.raises(embed_job.EmbedStateError, match="checkpoint is absent"):
        embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=True)

    path = embed_job.checkpoint_path(binding, "matsne")
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(embed_job.EmbedStateError, match="invalid JSON"):
        embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=True)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": binding.value["generation_id"],
                "variant_id": binding.variant_id,
                "snapshot_sha256": "0" * 64,
                "physical_collection": binding.value["physical_collection"],
                "source": "matsne",
                "shard": {"index": 0, "count": 1},
                "cursor": None,
                "documents_completed": 0,
                "chunks_completed": 0,
                "complete": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(embed_job.EmbedStateError, match="snapshot_sha256 mismatch"):
        embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=True)


def test_checkpoint_uses_global_source_document_version_cursor_and_variant_namespace(
    monkeypatch, tmp_path
):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = embed_job.prepare_binding(cfg, sealed, resume=False)
    prepared = embed_job.preflight_checkpoints(
        binding, ["matsne"], (1, 2), resume=False
    )["matsne"]
    docs = [_doc(str(index), f"v{index}") for index in range(5)]
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter(docs),
    )
    monkeypatch.setattr(
        embed_job,
        "_build_doc_points",
        lambda cfg, embedder, count_tokens, doc: ([SimpleNamespace(id=doc.document_id)], 1),
    )

    client = _Client()
    result = embed_job.embed_source_resumable(
        cfg,
        client,
        None,
        None,
        "matsne",
        binding=binding,
        snapshot_docs=sealed.docs,
        resume=False,
        prepared_checkpoint=prepared,
        batch_size=1,
        shard=(1, 2),
    )

    assert result == (2, 2, 0)
    assert client.calls == [
        (cfg.collection_name, ["1"], True),
        (cfg.collection_name, ["3"], True),
    ]
    path = embed_job.checkpoint_path(binding, "matsne", (1, 2))
    assert binding.variant_id in path.parts
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert checkpoint["cursor"] == {
        "global_index": 3,
        "source": "matsne",
        "document_id": "3",
        "version_id": "v3",
    }
    assert checkpoint["complete"] is True


def test_checkpoint_never_advances_past_failed_acknowledgement(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = embed_job.prepare_binding(cfg, sealed, resume=False)
    prepared = embed_job.preflight_checkpoints(
        binding, ["matsne"], None, resume=False
    )["matsne"]
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter([_doc("0", "v0"), _doc("1", "v1")]),
    )
    monkeypatch.setattr(
        embed_job,
        "_build_doc_points",
        lambda cfg, embedder, count_tokens, doc: ([SimpleNamespace(id=doc.document_id)], 1),
    )

    with pytest.raises(OSError, match="acknowledgement unavailable"):
        embed_job.embed_source_resumable(
            cfg,
            _Client(fail_on_call=2),
            None,
            None,
            "matsne",
            binding=binding,
            snapshot_docs=sealed.docs,
            resume=False,
            prepared_checkpoint=prepared,
            batch_size=1,
        )

    checkpoint = json.loads(
        embed_job.checkpoint_path(binding, "matsne").read_text(encoding="utf-8")
    )
    assert checkpoint["cursor"]["document_id"] == "0"
    assert checkpoint["documents_completed"] == 1
    assert checkpoint["complete"] is False


def test_embedding_failure_is_fatal_and_does_not_advance_checkpoint(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    binding = embed_job.prepare_binding(cfg, sealed, resume=False)
    prepared = embed_job.preflight_checkpoints(
        binding, ["matsne"], None, resume=False
    )["matsne"]
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter([_doc("0", "v0")]),
    )

    def fail(*args, **kwargs):
        raise ValueError("conversion failed")

    monkeypatch.setattr(embed_job, "_build_doc_points", fail)
    with pytest.raises(embed_job.EmbedStateError, match="embedding failed"):
        embed_job.embed_source_resumable(
            cfg,
            _Client(),
            None,
            None,
            "matsne",
            binding=binding,
            snapshot_docs=sealed.docs,
            resume=False,
            prepared_checkpoint=prepared,
        )
    checkpoint = json.loads(
        embed_job.checkpoint_path(binding, "matsne").read_text(encoding="utf-8")
    )
    assert checkpoint["cursor"] is None
    assert checkpoint["documents_completed"] == 0


def test_strict_snapshot_conversion_never_defaults_authority_completeness_or_version(
    tmp_path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    record = {
        "source": "matsne",
        "document_id": "1",
        "body_markdown": "complete official text",
        "language": "ka",
        "source_fingerprint": "a" * 64,
        "normalizer_revision": "canonical-v4",
        "version_id": "derived:abc",
        "version_id_kind": "derived",
        "content_kind": "full_text",
        "content_complete": True,
        "extraction_status": "full_text",
        "version_lineage_status": "derived",
        "version_lineage_complete": False,
        "source_authority": "primary_official",
        "official_url": "https://example.test/1",
        "freshness_sla_met": True,
        "supersedes": [],
        "consolidated_dates": [],
    }
    for missing in ("source_authority", "content_complete", "version_id"):
        candidate = dict(record)
        candidate.pop(missing)
        (docs / "matsne.jsonl").write_text(
            json.dumps(candidate, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(embed_job.EmbedStateError, match=missing):
            list(embed_job.iter_snapshot_docs("matsne", root=docs, strict=True))


def test_checksum_output_is_explicit_create_only_and_outside_snapshot(tmp_path):
    class Embedder:
        def encode_query(self, _sentence):
            return SimpleNamespace(dense=[1.0, 0.5])

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "evidence" / "checksum.json"
    assert embed_job.save_checksum_reference(
        Embedder(), output, snapshot_root=snapshot
    )
    with pytest.raises(embed_job.EmbedStateError, match="already exists"):
        embed_job.save_checksum_reference(Embedder(), output, snapshot_root=snapshot)
    with pytest.raises(embed_job.EmbedStateError, match="outside"):
        embed_job.save_checksum_reference(
            Embedder(), snapshot / "checksum.json", snapshot_root=snapshot
        )


def test_snapshot_verification_rejects_preflight_even_if_delegate_returns_it(
    monkeypatch, tmp_path
):
    root = tmp_path / "snapshot"
    docs = root / "docs"
    docs.mkdir(parents=True)
    manifest = {
        "snapshot_id": "v3_test",
        "snapshot_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "preflight": True,
        "build": {"sources": list(embed_job.SOURCES)},
    }
    from ingest import snapshot

    monkeypatch.setattr(snapshot, "verify_sealed_snapshot", lambda *args, **kwargs: manifest)
    with pytest.raises(embed_job.EmbedStateError, match="preflight"):
        embed_job.verify_snapshot_docs(docs)


def test_resume_proof_rejects_missing_acknowledged_chunk_even_when_other_point_masks_count(
    monkeypatch, tmp_path
):
    from ingest import qdrant_store
    from ingest.pipeline import _document_state_hash

    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    binding = embed_job.prepare_binding(cfg, sealed, resume=False)
    initial = embed_job.preflight_checkpoints(
        binding, ["matsne"], None, resume=False
    )["matsne"]
    doc = _canonical_doc()
    checkpoint = dict(initial)
    checkpoint.update(
        {
            "cursor": {
                "global_index": 0,
                "source": doc.source,
                "document_id": doc.document_id,
                "version_id": doc.version_id,
            },
            "documents_completed": 1,
            "chunks_completed": 2,
        }
    )
    path = embed_job.checkpoint_path(binding, "matsne")
    embed_job._replace_checkpoint(path, checkpoint, expected_previous=initial)
    monkeypatch.setattr(
        embed_job,
        "iter_snapshot_docs",
        lambda source, **kwargs: iter([doc]),
    )

    identity = qdrant_store.validate_generation_identity(cfg).as_payload()
    state_hash = _document_state_hash(cfg, doc=doc)
    body_hash = qdrant_store.content_hash(doc.body_markdown)

    def record(point_id, chunk_index):
        return SimpleNamespace(
            id=point_id,
            payload={
                **identity,
                "source": doc.source,
                "document_id": doc.document_id,
                "version_id": doc.version_id,
                "chunk_index": chunk_index,
                "document_chunk_count": 2,
                "document_state_hash": state_hash,
                "content_hash": body_hash,
                "canonical_content_hash": body_hash,
                "source_fingerprint": doc.source_fingerprint,
                "normalizer_revision": doc.normalizer_revision,
                "content_complete": True,
                "extraction_status": "full_text",
                "source_authority": doc.source_authority,
                "canonical_text_exact": True,
                "passage_hash": "a" * 64,
            },
        )

    zero_id = qdrant_store.point_id(
        doc.source, doc.document_id, 0, version_id=doc.version_id
    )
    missing_id = qdrant_store.point_id(
        doc.source, doc.document_id, 1, version_id=doc.version_id
    )
    masking_id = qdrant_store.point_id("matsne", "unrelated", 0, version_id="v-other")

    class Client:
        # Collection-wide counts can still be two because this unrelated same-identity
        # point masks the deleted acknowledged chunk. Exact ID retrieval must catch it.
        points = {
            zero_id: record(zero_id, 0),
            masking_id: record(masking_id, 0),
        }

        def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
            assert collection_name == cfg.collection_name
            assert with_vectors is False
            return [self.points[point_id] for point_id in ids if point_id in self.points]

    with pytest.raises(embed_job.EmbedStateError, match="deterministic points are missing"):
        embed_job.verify_resume_checkpoint_points(
            cfg,
            Client(),
            sealed,
            {"matsne": checkpoint},
        )
    assert missing_id not in Client.points


def _cli_args(tmp_path, **over):
    value = {
        "collection": None,
        "snapshot_docs": tmp_path / "snapshot" / "docs",
        "source": "all",
        "checksum": False,
        "checksum_output": None,
        "resume": False,
        "recreate": False,
        "apply": True,
        "shard": None,
        "batch_size": 10,
    }
    value.update(over)
    return SimpleNamespace(**value)


def test_cli_rejects_mutable_revision_before_snapshot_client_or_model(
    monkeypatch, tmp_path
):
    cfg = _cfg(tmp_path, embedding_revision="main")
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        embed_job,
        "verify_snapshot_docs",
        lambda path: pytest.fail("snapshot must not be accessed"),
    )
    from ingest import qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    with pytest.raises(ConfigurationError, match="immutable revisions"):
        primary_cli._cmd_embed(_cli_args(tmp_path))


def test_cli_snapshot_failure_happens_before_client_or_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        embed_job,
        "verify_snapshot_docs",
        lambda path: (_ for _ in ()).throw(embed_job.EmbedStateError("tampered snapshot")),
    )
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(embed_job.EmbedStateError, match="tampered snapshot"):
        primary_cli._cmd_embed(_cli_args(tmp_path))


def test_cli_resume_requires_binding_before_client_or_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(embed_job.EmbedStateError, match="binding is absent"):
        primary_cli._cmd_embed(_cli_args(tmp_path, resume=True))


def test_cli_rejects_zero_batch_before_snapshot_state_client_or_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(
        embed_job,
        "verify_snapshot_docs",
        lambda path: pytest.fail("snapshot must not be accessed"),
    )
    from ingest import embedding, qdrant_store

    monkeypatch.setattr(
        qdrant_store,
        "make_client",
        lambda value: pytest.fail("client must not be accessed"),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(SystemExit, match="batch-size must be a positive"):
        primary_cli._cmd_embed(_cli_args(tmp_path, batch_size=0))
    assert not cfg.state_dir.exists()


def test_cli_runs_exact_resume_point_proof_before_model(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    sealed = _sealed(tmp_path, cfg)
    binding = embed_job.prepare_binding(cfg, sealed, resume=False)
    embed_job.preflight_checkpoints(binding, ["matsne"], None, resume=False)
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda args: cfg)
    monkeypatch.setattr(embed_job, "verify_snapshot_docs", lambda path: sealed)
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    from ingest import embedding, qdrant_store

    client = object()
    monkeypatch.setattr(qdrant_store, "make_client", lambda value: client)
    monkeypatch.setattr(qdrant_store, "prepare_embed_collection", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        embed_job,
        "verify_resume_checkpoint_points",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            embed_job.EmbedStateError("missing acknowledged point")
        ),
    )
    monkeypatch.setattr(
        embedding,
        "BGEM3Embedder",
        lambda value: pytest.fail("model must not be loaded"),
    )
    with pytest.raises(embed_job.EmbedStateError, match="missing acknowledged point"):
        primary_cli._cmd_embed(_cli_args(tmp_path, resume=True, source="matsne"))
