"""Sharded embed: N shards must partition a source's docs exactly (each doc once, no dupes)."""

import dataclasses
from types import SimpleNamespace

import pytest

from ingest import embed_job
from ingest.config import load_config


class _Client:
    def __init__(self):
        self.upserted = []

    def upsert(self, collection_name, points, wait=False):
        self.upserted.extend(p.id for p in points)


def _staging_cfg(tmp_path):
    """Shard mechanics write only to a run-scoped test collection."""

    return dataclasses.replace(
        load_config(),
        state_dir=tmp_path,
        collection_name="georgian_legal__test_shards",
    )


def _binding(tmp_path, cfg):
    docs = tmp_path / "snapshot" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    value = {
        "generation_id": "gen_20260715_shards",
        "physical_collection": cfg.collection_name,
        "variant_id": "b" * 64,
        "snapshot": {
            "snapshot_sha256": "a" * 64,
            "docs": str(docs.resolve()),
        },
    }
    path = tmp_path / "embed" / "gen_20260715_shards" / "binding.json"
    return embed_job.EmbedBinding(path=path, value=value), docs


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shards_partition_all_docs_exactly_once(monkeypatch, tmp_path, n):
    docs = [SimpleNamespace(source="matsne", document_id=str(i), version_id=f"v{i}")
            for i in range(23)]
    monkeypatch.setattr(embed_job, "iter_snapshot_docs", lambda source, **kw: iter(list(docs)))
    monkeypatch.setattr(embed_job, "_build_doc_points",
                        lambda cfg, emb, ct, doc: ([SimpleNamespace(id=doc.document_id)], 1))
    cfg = _staging_cfg(tmp_path)
    binding, snapshot_docs = _binding(tmp_path, cfg)

    all_ids = []
    for i in range(n):
        client = _Client()
        embed_job.embed_source_resumable(
            cfg,
            client,
            None,
            None,
            "matsne",
            binding=binding,
            snapshot_docs=snapshot_docs,
            resume=False,
            batch_size=1,
            shard=(i, n),
        )
        all_ids.extend(client.upserted)

    assert sorted(all_ids, key=int) == [str(i) for i in range(23)]  # complete + no duplicates


def test_shard_checkpoints_are_separate_files(monkeypatch, tmp_path):
    docs = [SimpleNamespace(source="matsne", document_id=str(i), version_id=f"v{i}")
            for i in range(8)]
    monkeypatch.setattr(embed_job, "iter_snapshot_docs", lambda source, **kw: iter(list(docs)))
    monkeypatch.setattr(embed_job, "_build_doc_points",
                        lambda cfg, emb, ct, doc: ([SimpleNamespace(id=doc.document_id)], 1))
    cfg = _staging_cfg(tmp_path)
    binding, snapshot_docs = _binding(tmp_path, cfg)
    embed_job.embed_source_resumable(
        cfg,
        _Client(),
        None,
        None,
        "matsne",
        binding=binding,
        snapshot_docs=snapshot_docs,
        resume=False,
        batch_size=1,
        shard=(1, 4),
    )
    assert embed_job.checkpoint_path(binding, "matsne", (1, 4)).exists()
    assert not embed_job.checkpoint_path(binding, "matsne", (0, 1)).exists()


def test_shard_resumes_from_its_checkpoint(monkeypatch, tmp_path):
    docs = [SimpleNamespace(source="matsne", document_id=str(i), version_id=f"v{i}")
            for i in range(20)]
    monkeypatch.setattr(embed_job, "iter_snapshot_docs", lambda source, **kw: iter(list(docs)))
    monkeypatch.setattr(embed_job, "_build_doc_points",
                        lambda cfg, emb, ct, doc: ([SimpleNamespace(id=doc.document_id)], 1))
    cfg = _staging_cfg(tmp_path)
    binding, snapshot_docs = _binding(tmp_path, cfg)
    path = embed_job.checkpoint_path(binding, "matsne", (0, 4))
    initial = embed_job.preflight_checkpoints(
        binding, ["matsne"], (0, 4), resume=False
    )["matsne"]
    checkpoint = dict(initial)
    checkpoint.update(
        {
            "cursor": {
                "global_index": 0,
                "source": "matsne",
                "document_id": "0",
                "version_id": "v0",
            },
            "documents_completed": 1,
            "chunks_completed": 1,
        }
    )
    embed_job._replace_checkpoint(path, checkpoint, expected_previous=initial)
    client = _Client()
    embed_job.embed_source_resumable(
        cfg,
        client,
        None,
        None,
        "matsne",
        binding=binding,
        snapshot_docs=snapshot_docs,
        resume=True,
        batch_size=1,
        shard=(0, 4),
    )
    # shard 0 covers gi 0,4,8,12,16 → ids 0,4,8,12,16; resume after "0" → 4,8,12,16
    assert client.upserted == ["4", "8", "12", "16"]
