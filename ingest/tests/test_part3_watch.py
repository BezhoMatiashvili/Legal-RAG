"""Part-3 watch hardening: content-hash change detection, schema-drift alerts, ingest report."""

import dataclasses
import json
from types import SimpleNamespace

from ingest import pipeline
from ingest.config import load_config
from ingest.embedding import Embedded, Sparse


class FakeEmbedder:
    def encode_passages(self, texts):
        return [Embedded(dense=[0.1, 0.2, 0.3, 0.4], sparse=Sparse([1, 2], [0.5, 0.6])) for _ in texts]

    def encode_query(self, text):
        return self.encode_passages([text])[0]


class StoringClient:
    """Fake client that remembers upserted payloads so ``retrieve`` can serve content_hash."""

    def __init__(self):
        self.by_id = {}
        self.upserts = []
        self.deletes = []

    def upsert(self, collection_name, points, wait=False):
        pts = list(points)
        self.upserts.append(pts)
        for pt in pts:
            self.by_id[pt.id] = pt.payload

    def delete(self, collection_name, points_selector, wait=False):
        assert wait is True
        self.deletes.append(points_selector)

    def retrieve(self, collection_name, ids, with_payload=None):
        return [SimpleNamespace(payload=self.by_id[i]) for i in ids if i in self.by_id]


def _count_tokens(text):
    return len(text.split())


def _cfg(tmp_path):
    return dataclasses.replace(
        load_config(), collection_name="test",
        artifacts_root=tmp_path / "artifacts", state_dir=tmp_path / "state",
        chunk_tokens=40, chunk_overlap=5, chunk_min_tokens=1,
        dense_dim=4,
    )


def _ecd(doc_id, body="body text here for chunking with enough meaningful legal content", **extra):
    item = {"decision_document_id": doc_id, "case_no": f"case-{doc_id}",
            "decision_type_name": "x", "court_name": "court",
            "decision_date": "2020-04-30", "body_markdown": body}
    item.update(extra)
    return item


def _write_run(cfg, source, run_id, items):
    run_dir = cfg.artifacts_root / source / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")


def _drain(cfg, client, state, **kw):
    return pipeline.watch_drain_source(
        cfg, client, FakeEmbedder(), _count_tokens, "ecd", state, batch_size=8, **kw)


def test_unchanged_doc_is_not_reembedded(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    state = {"run_offsets": {}}
    body = "same body one two three with enough meaningful legal content for indexing"
    _write_run(cfg, "ecd", "001", [_ecd("X", body)])
    _drain(cfg, client, state)
    assert len(client.upserts) == 1 and state.get("added") == 1

    # A later run re-emits the identical doc → must be skipped (no new upsert).
    _write_run(cfg, "ecd", "002", [_ecd("X", body)])
    _drain(cfg, client, state)
    assert len(client.upserts) == 1                 # unchanged: no second upsert
    assert state.get("unchanged") == 1


def test_changed_body_is_reembedded(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    state = {"run_offsets": {}}
    _write_run(cfg, "ecd", "001", [_ecd(
        "X", "original body alpha beta with enough meaningful legal content for indexing")])
    _drain(cfg, client, state)
    _write_run(cfg, "ecd", "002", [_ecd(
        "X", "REVISED body gamma delta with enough meaningful legal content for indexing")])
    _drain(cfg, client, state)
    assert len(client.upserts) == 2 and state.get("updated") == 1


def test_same_body_metadata_change_is_reembedded(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    state = {"run_offsets": {}}
    body = "same body with enough meaningful legal content for metadata transition"
    _write_run(cfg, "ecd", "001", [_ecd("X", body, decision_type_name="draft")])
    _drain(cfg, client, state)
    _write_run(cfg, "ecd", "002", [_ecd("X", body, decision_type_name="final decision")])

    _drain(cfg, client, state)

    assert len(client.upserts) == 2
    assert state["updated"] == 1
    assert state.get("unchanged", 0) == 0


def test_skip_unchanged_false_always_reembeds(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    state = {"run_offsets": {}}
    body = "same body with enough meaningful legal content for repeated indexing"
    _write_run(cfg, "ecd", "001", [_ecd("X", body)])
    _drain(cfg, client, state, skip_unchanged=False)
    _write_run(cfg, "ecd", "002", [_ecd("X", body)])
    _drain(cfg, client, state, skip_unchanged=False)
    assert len(client.upserts) == 2                 # no change-detection → re-embeds


def test_schema_drift_flags_new_field(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    state = {"run_offsets": {}}
    # First item seeds the key baseline; second introduces an unexpected field.
    _write_run(cfg, "ecd", "001", [_ecd("X"), _ecd("Y", surprise_new_field="!")])
    _drain(cfg, client, state)
    assert "surprise_new_field" in state.get("schema_drift", {})


def test_write_ingest_report(tmp_path):
    cfg = _cfg(tmp_path)
    states = {"ecd": {"docs": 3, "chunks": 9, "added": 2, "updated": 1, "unchanged": 5,
                      "skipped": 0, "schema_drift": {"foo": 2}, "updated_at": "2026-07-08T00:00:00+00:00"}}
    path = pipeline.write_ingest_report(cfg, states, kind="watch")
    assert path.exists()
    report = json.loads(path.read_text())
    assert report["totals"]["docs"] == 3 and report["totals"]["unchanged"] == 5
    assert report["schema_drift"] == {"foo": 2}
    assert report["per_source"]["ecd"]["added"] == 2
    assert report["totals"]["pending_retries"] == 0
    assert report["totals"]["dead_letters"] == 0


def test_watch_hashes_and_indexes_cleaned_text(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    state = {"run_offsets": {}}
    raw = "same\x00 body with enough meaningful legal content for clean indexing"
    _write_run(cfg, "ecd", "001", [_ecd("X", raw)])

    _drain(cfg, client, state)

    assert len(client.upserts) == 1
    payload = next(iter(client.by_id.values()))
    assert "\x00" not in payload["text"]
    assert payload["content_hash"] == pipeline.content_hash(raw.replace("\x00", ""))

    # A later raw re-emission compares in the cleaned hash space and remains a no-op.
    _write_run(cfg, "ecd", "002", [_ecd("X", raw)])
    _drain(cfg, client, state)
    assert len(client.upserts) == 1
    assert state["unchanged"] == 1


def test_watch_quarantine_advances_without_embed_retry(tmp_path):
    cfg = _cfg(tmp_path)
    client = StoringClient()
    run_id = "001"
    state = {"run_offsets": {}}
    _write_run(cfg, "ecd", run_id, [_ecd("X", "too short")])

    docs, chunks, skipped = _drain(cfg, client, state)

    assert (docs, chunks, skipped) == (0, 0, 1)
    assert state["run_offsets"][run_id] > 0
    assert state["quarantined"] == {"near_empty": 1}
    assert "embed_retry" not in state
    assert client.upserts == []
