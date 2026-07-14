"""Bounded external-state contracts for delta input deduplication."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import embed_delta  # noqa: E402


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sqlite_dedup_preserves_last_value_and_first_seen_order(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_rows(
        first,
        [
            {"document_id": "a", "body_markdown": "old a"},
            {"document_id": "b", "body_markdown": "only b"},
        ],
    )
    _write_rows(
        second,
        [{"document_id": "a", "body_markdown": "new a"}],
    )

    documents, failures = embed_delta._load_items(  # noqa: SLF001
        "matsne", [first, second]
    )
    storage_path = documents.storage_path
    storage_directory = storage_path.parent
    try:
        item_iterator = documents.items()
        assert iter(item_iterator) is item_iterator
        loaded = list(item_iterator)

        assert failures == []
        assert [document_id for document_id, _doc in loaded] == ["a", "b"]
        assert [doc.body_markdown for _document_id, doc in loaded] == [
            "new a",
            "only b",
        ]
        assert stat.S_IMODE(storage_directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(storage_path.stat().st_mode) == 0o600
    finally:
        documents.close()

    assert not storage_path.exists()
    assert not storage_directory.exists()


def test_sqlite_dedup_bounds_large_revision_stream_to_unique_rows(tmp_path):
    path = tmp_path / "many-revisions.jsonl"
    identity_count = 256
    revision_count = 32
    rows = [
        {
            "document_id": str(document_id),
            "body_markdown": f"revision {revision} for {document_id}",
        }
        for revision in range(revision_count)
        for document_id in range(identity_count)
    ]
    _write_rows(path, rows)

    documents, failures = embed_delta._load_items("matsne", [path])  # noqa: SLF001
    try:
        assert failures == []
        assert len(documents) == identity_count
        assert documents._connection.execute(  # noqa: SLF001
            "PRAGMA cache_size"
        ).fetchone() == (-embed_delta._DEDUP_CACHE_KIB,)  # noqa: SLF001

        loaded = list(documents.items())
        assert [document_id for document_id, _doc in loaded] == [
            str(document_id) for document_id in range(identity_count)
        ]
        assert all(
            doc.body_markdown == f"revision {revision_count - 1} for {document_id}"
            for document_id, doc in loaded
        )
    finally:
        documents.close()


def test_sqlite_supreme_duplicates_remain_fail_closed_and_first_wins(tmp_path):
    path = tmp_path / "supreme.jsonl"
    chamber = "სამოქალაქო საქმეთა პალატა"
    _write_rows(
        path,
        [
            {
                "case_id": "101",
                "chamber": chamber,
                "body_markdown": "first ruling",
            },
            {
                "case_id": "101",
                "chamber": chamber,
                "body_markdown": "later duplicate",
            },
        ],
    )

    documents, failures = embed_delta._load_items(  # noqa: SLF001
        "supremecourt", [path]
    )
    try:
        loaded = list(documents.items())
    finally:
        documents.close()

    assert len(loaded) == 1
    assert loaded[0][1].body_markdown == "first ruling"
    assert failures == [
        {
            "source": "supremecourt",
            "file": "supreme.jsonl",
            "line": 2,
            "reason": f"duplicate Supreme Court identity: 101:{chamber}",
        }
    ]


def test_analyzed_documents_remain_external_reiterable_and_exclude_bad_chunks(tmp_path):
    path = tmp_path / "items.jsonl"
    _write_rows(
        path,
        [
            {
                "document_id": "good",
                "body_markdown": "usable complete legal body " * 8,
            },
            {"document_id": "short", "body_markdown": "too short"},
        ],
    )
    cfg = SimpleNamespace(chunk_tokens=64, chunk_overlap=8, chunk_min_tokens=1)

    documents, manifest = embed_delta._analyze_docs(  # noqa: SLF001
        cfg,
        [("matsne", [path])],
        lambda text: max(1, len(text.split())),
    )
    storage_paths = [  # noqa: SLF001 - assert external-state lifecycle
        store.storage_path for store in documents._stores  # noqa: SLF001
    ]
    try:
        first_pass = [(doc.document_id, doc.body_markdown) for doc in documents]
        second_pass = [(doc.document_id, doc.body_markdown) for doc in documents]

        assert not isinstance(documents, list)
        assert first_pass == second_pass == [
            ("good", "usable complete legal body " * 8)
        ]
        assert manifest["documents"] == 1
        assert manifest["skipped"] == 1
        assert "near_empty" in manifest["failures"][0]["reason"]
        assert all(storage_path.exists() for storage_path in storage_paths)
    finally:
        documents.close()

    assert all(not storage_path.exists() for storage_path in storage_paths)
