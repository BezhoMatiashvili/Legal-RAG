"""Offline tests for the continuous ``watch`` ingest mode (no Qdrant, no model)."""

import dataclasses
import json

from ingest import pipeline
from ingest.config import load_config
from ingest.embedding import Embedded, Sparse


class FakeEmbedder:
    def encode_passages(self, texts):
        return [Embedded(dense=[0.1, 0.2, 0.3, 0.4], sparse=Sparse([1, 2], [0.5, 0.6])) for _ in texts]

    def encode_query(self, text):
        return self.encode_passages([text])[0]


class FakeClient:
    """Captures upserts/deletes so we can assert watch behaviour offline."""

    def __init__(self):
        self.upserts = []  # (points, wait)
        self.deletes = []

    def upsert(self, collection_name, points, wait=False):
        self.upserts.append((list(points), wait))

    def delete(self, collection_name, points_selector, wait=False):
        assert wait is True
        self.deletes.append(points_selector)


def _count_tokens(text):
    return len(text.split())


def _make_cfg(tmp_path):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        collection_name="test",
        artifacts_root=tmp_path / "artifacts",
        state_dir=tmp_path / "state",
        chunk_tokens=40,
        chunk_overlap=5,
        chunk_min_tokens=1,
        dense_dim=4,
    )


def _ecd_item(doc_id, body="body text here for chunking with enough meaningful legal content"):
    return {
        "decision_document_id": doc_id, "case_no": f"case-{doc_id}",
        "decision_type_name": "x", "court_name": "court",
        "decision_date": "2020-04-30", "body_markdown": body,
    }


def _write_run(cfg, source, run_id, items, *, terminate=True):
    """Write a run's items.jsonl. ``terminate=False`` leaves the last line unterminated."""
    run_dir = cfg.artifacts_root / source / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(i, ensure_ascii=False) for i in items)
    if items and terminate:
        text += "\n"
    path = run_dir / "items.jsonl"
    path.write_bytes(text.encode("utf-8"))
    return path


def _drain(cfg, client, source, state, **kw):
    return pipeline.watch_drain_source(
        cfg, client, FakeEmbedder(), _count_tokens, source, state, batch_size=1, **kw)


def _doc_order(client):
    """document_ids in the order they were first upserted."""
    seen = []
    for batch, _ in client.upserts:
        for p in batch:
            did = p.payload["document_id"]
            if did not in seen:
                seen.append(did)
    return seen


# --- discover_runs ---------------------------------------------------------


def test_discover_runs_sorted_and_excludes_latest(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200102T000000Z_s_e", [_ecd_item(2)])
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1)])
    latest = cfg.artifacts_root / "ecd" / "latest"
    latest.mkdir(parents=True)
    (latest / "items.jsonl").write_text("{}\n", encoding="utf-8")

    runs = pipeline.discover_runs(cfg, "ecd")
    assert [rid for rid, _ in runs] == ["20200101T000000Z_s_e", "20200102T000000Z_s_e"]


def test_discover_runs_missing_dir_is_empty(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert pipeline.discover_runs(cfg, "ecd") == []


# --- _read_complete_lines --------------------------------------------------


def test_read_complete_lines_leaves_partial_trailing(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')  # two complete lines + a partial one
    lines, off = pipeline._read_complete_lines(p, 0)
    assert [t for t, _ in lines] == ['{"a":1}', '{"b":2}']
    assert off == len(b'{"a":1}\n{"b":2}\n')

    with open(p, "ab") as fh:  # finish the partial line
        fh.write(b'3}\n')
    lines2, off2 = pipeline._read_complete_lines(p, off)
    assert [t for t, _ in lines2] == ['{"c":3}']
    assert off2 == p.stat().st_size


def test_read_complete_lines_byte_offsets_unicode(tmp_path):
    p = tmp_path / "f.jsonl"
    line = '{"t":"საქართველო"}'  # multi-byte Georgian text
    p.write_bytes((line + "\n").encode("utf-8"))
    lines, off = pipeline._read_complete_lines(p, 0)
    assert lines[0][0] == line
    expected = len((line + "\n").encode("utf-8"))
    assert lines[0][1] == expected
    assert off == expected


def test_read_complete_lines_self_heals_on_shrink(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n')
    lines, off = pipeline._read_complete_lines(p, 9999)  # offset past EOF -> restart
    assert [t for t, _ in lines] == ['{"a":1}']


# --- watch_drain_source ----------------------------------------------------


def test_backfill_all_runs_oldest_first(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1)])
    _write_run(cfg, "ecd", "20200102T000000Z_s_e", [_ecd_item(2), _ecd_item(3)])
    state = pipeline._load_watch_state(cfg, "ecd")
    client = FakeClient()

    docs, chunks, skipped = _drain(cfg, client, "ecd", state)
    assert docs == 3 and chunks > 0 and skipped == 0
    assert _doc_order(client) == ["1", "2", "3"]          # chronological
    assert all(wait for _, wait in client.upserts)        # durable (wait=True)


def test_second_pass_with_no_new_bytes_is_noop(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1), _ecd_item(2)])
    state = pipeline._load_watch_state(cfg, "ecd")
    assert _drain(cfg, FakeClient(), "ecd", state)[0] == 2

    state2 = pipeline._load_watch_state(cfg, "ecd")        # reload persisted offsets
    client2 = FakeClient()
    docs, _, _ = _drain(cfg, client2, "ecd", state2)
    assert docs == 0 and client2.upserts == []            # nothing re-embedded


def test_appended_lines_are_picked_up(tmp_path):
    cfg = _make_cfg(tmp_path)
    run = "20200101T000000Z_s_e"
    path = _write_run(cfg, "ecd", run, [_ecd_item(1)])
    state = pipeline._load_watch_state(cfg, "ecd")
    assert _drain(cfg, FakeClient(), "ecd", state)[0] == 1

    with open(path, "ab") as fh:  # scraper appends a new doc to the growing run file
        fh.write((json.dumps(_ecd_item(2)) + "\n").encode("utf-8"))
    client = FakeClient()
    docs, _, _ = _drain(cfg, client, "ecd", state)
    assert docs == 1 and _doc_order(client) == ["2"]


def test_new_run_dir_is_picked_up(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1)])
    state = pipeline._load_watch_state(cfg, "ecd")
    assert _drain(cfg, FakeClient(), "ecd", state)[0] == 1

    _write_run(cfg, "ecd", "20200102T000000Z_s_e", [_ecd_item(2)])  # later run appears
    client = FakeClient()
    docs, _, _ = _drain(cfg, client, "ecd", state)
    assert docs == 1 and _doc_order(client) == ["2"]


def test_malformed_trailing_line_skipped_but_offset_advances(tmp_path):
    cfg = _make_cfg(tmp_path)
    run = "20200101T000000Z_s_e"
    path = _write_run(cfg, "ecd", run, [_ecd_item(1)])
    with open(path, "ab") as fh:  # a complete-but-malformed record
        fh.write(b'{bad json}\n')
    state = pipeline._load_watch_state(cfg, "ecd")
    client = FakeClient()

    docs, _, skipped = _drain(cfg, client, "ecd", state)
    assert docs == 1 and skipped == 1
    assert state["run_offsets"][run] == path.stat().st_size  # advanced past the bad line
    assert all(wait for _, wait in client.upserts)


def test_stale_delete_skipped_on_fresh_collection(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1)])
    state = pipeline._load_watch_state(cfg, "ecd")
    client = FakeClient()
    _drain(cfg, client, "ecd", state, skip_stale_delete=True)
    assert client.deletes == []


def test_stop_event_halts_and_keeps_offset_at_last_doc(tmp_path):
    import threading
    cfg = _make_cfg(tmp_path)
    run = "20200101T000000Z_s_e"
    _write_run(cfg, "ecd", run, [_ecd_item(1), _ecd_item(2)])
    state = pipeline._load_watch_state(cfg, "ecd")
    stop = threading.Event()
    stop.set()  # already set -> first line check stops before processing anything
    client = FakeClient()
    docs, _, _ = _drain(cfg, client, "ecd", state, stop_event=stop)
    assert docs == 0 and client.upserts == []


# --- watch state I/O -------------------------------------------------------


def test_delete_watch_state_is_idempotent(tmp_path):
    cfg = _make_cfg(tmp_path)
    state = pipeline._load_watch_state(cfg, "ecd")
    pipeline._save_watch_state(cfg, "ecd", state)
    assert (cfg.state_dir / "ecd.watch.json").exists()
    pipeline.delete_watch_state(cfg, "ecd")
    assert not (cfg.state_dir / "ecd.watch.json").exists()
    pipeline.delete_watch_state(cfg, "ecd")  # no error on second call


def test_watch_state_independent_of_plain_checkpoint(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1)])
    state = pipeline._load_watch_state(cfg, "ecd")
    _drain(cfg, FakeClient(), "ecd", state)
    assert (cfg.state_dir / "ecd.watch.json").exists()
    assert not (cfg.state_dir / "ecd.json").exists()  # plain ingest checkpoint untouched


# --- watch_loop (once mode) ------------------------------------------------


def test_watch_loop_once_backfills_then_exits(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_run(cfg, "ecd", "20200101T000000Z_s_e", [_ecd_item(1)])
    _write_run(cfg, "ecd", "20200102T000000Z_s_e", [_ecd_item(2), _ecd_item(3)])
    client = FakeClient()
    pipeline.watch_loop(
        cfg, client, FakeEmbedder(), _count_tokens, ["ecd"],
        batch_size=1, poll_interval=0.01, once=True,
    )
    assert _doc_order(client) == ["1", "2", "3"]
    state = pipeline._load_watch_state(cfg, "ecd")
    assert state["docs"] == 3
