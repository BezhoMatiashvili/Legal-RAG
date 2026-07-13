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


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shards_partition_all_docs_exactly_once(monkeypatch, tmp_path, n):
    docs = [SimpleNamespace(source="matsne", document_id=str(i), body_markdown=f"d{i}")
            for i in range(23)]
    monkeypatch.setattr(embed_job, "iter_snapshot_docs", lambda source, **kw: iter(list(docs)))
    monkeypatch.setattr(embed_job, "_build_doc_points",
                        lambda cfg, emb, ct, doc: ([SimpleNamespace(id=doc.document_id)], 1))
    cfg = dataclasses.replace(load_config(), state_dir=tmp_path)

    all_ids = []
    for i in range(n):
        client = _Client()
        embed_job.embed_source_resumable(cfg, client, None, None, "matsne",
                                         batch_size=1, shard=(i, n))
        all_ids.extend(client.upserted)

    assert sorted(all_ids, key=int) == [str(i) for i in range(23)]  # complete + no duplicates


def test_shard_checkpoints_are_separate_files(monkeypatch, tmp_path):
    docs = [SimpleNamespace(source="matsne", document_id=str(i), body_markdown=f"d{i}")
            for i in range(8)]
    monkeypatch.setattr(embed_job, "iter_snapshot_docs", lambda source, **kw: iter(list(docs)))
    monkeypatch.setattr(embed_job, "_build_doc_points",
                        lambda cfg, emb, ct, doc: ([SimpleNamespace(id=doc.document_id)], 1))
    cfg = dataclasses.replace(load_config(), state_dir=tmp_path)
    embed_job.embed_source_resumable(cfg, _Client(), None, None, "matsne", batch_size=1, shard=(1, 4))
    assert (tmp_path / "matsne.shard1of4.embed.json").exists()
    assert not (tmp_path / "matsne.embed.json").exists()  # whole-source ckpt untouched


def test_shard_resumes_from_its_checkpoint(monkeypatch, tmp_path):
    docs = [SimpleNamespace(source="matsne", document_id=str(i), body_markdown=f"d{i}")
            for i in range(20)]
    monkeypatch.setattr(embed_job, "iter_snapshot_docs", lambda source, **kw: iter(list(docs)))
    monkeypatch.setattr(embed_job, "_build_doc_points",
                        lambda cfg, emb, ct, doc: ([SimpleNamespace(id=doc.document_id)], 1))
    cfg = dataclasses.replace(load_config(), state_dir=tmp_path)
    # pre-seed shard 0/4's checkpoint as if doc "0" (first shard-0 doc, gi=0) already done
    (tmp_path / "matsne.shard0of4.embed.json").write_text(
        '{"last_document_id": "0", "docs": 1, "chunks": 1}')
    client = _Client()
    embed_job.embed_source_resumable(cfg, client, None, None, "matsne", batch_size=1, shard=(0, 4))
    # shard 0 covers gi 0,4,8,12,16 → ids 0,4,8,12,16; resume after "0" → 4,8,12,16
    assert client.upserted == ["4", "8", "12", "16"]
