from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ingest import chunk_inventory


BODY = (
    "# თავი I\n"
    "მუხლი 1. პირველი ნორმა მოქმედებს ყველა შესაბამის პირზე.\n\n"
    "1. პირველი პუნქტი შეიცავს დამატებით წესს და განმარტებას.\n\n"
    "2. მეორე პუნქტი შეიცავს საბოლოო წესს."
)


def _write_document(
    root: Path,
    *,
    body: str = BODY,
    page_boundaries: list[dict[str, int]] | None = None,
) -> Path:
    docs = root / "docs"
    docs.mkdir(parents=True)
    pages = page_boundaries or []
    record = {
        "doc_id": "matsne:law-1:version-1",
        "source": "matsne",
        "document_id": "law-1",
        "version_id": "version-1",
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "body_char_len": len(body),
        "body_markdown": body,
        "title": "ტესტის კანონი",
        "document_type": "law",
        "document_number": "1",
        "date": "2026-07-15",
        "date_raw": "2026-07-15",
        "status": "in_force",
        "is_consolidated": True,
        "admissible": True,
        "page_boundaries": pages,
        "page_coordinate_reason": (
            "exact_pdf_text" if pages else "source_not_paginated"
        ),
    }
    (docs / "matsne.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return docs


def _compute(docs: Path, **overrides):
    values = {
        "sources": ("matsne",),
        "tokenizer_model": "pinned-tokenizer",
        "tokenizer_revision": "a" * 40,
        "max_tokens": 12,
        "overlap_tokens": 2,
        "min_tokens": 2,
        "document_header": True,
        "count_tokens": lambda text: len(text.split()),
    }
    values.update(overrides)
    return chunk_inventory.compute_structural_chunk_inventory(docs, **values)


def test_inventory_recomputes_exact_bytes_and_output_is_create_only(tmp_path):
    docs = _write_document(tmp_path / "base")
    output = tmp_path / "inventory.jsonl"
    written = _compute(docs, output=output)
    recomputed = _compute(docs)

    assert written == recomputed
    assert written.artifact_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert written.document_count == 1
    assert written.chunk_count >= 1
    assert written.record_count == 2
    rows = list(
        chunk_inventory.iter_validated_inventory(
            output,
            manifest_entry=written.manifest_entry(),
            expected_identity=written.identity,
        )
    )
    assert len(rows) == 1
    assert all(
        len(chunk["embed_input"]["sha256"]) == 64
        and chunk["embed_input"]["token_count"] >= chunk["token_count"]
        for chunk in rows[0]["chunks"]
    )
    with pytest.raises(chunk_inventory.ChunkInventoryError, match="cannot create"):
        _compute(docs, output=output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == written.artifact_sha256


def test_inventory_digest_changes_with_passage_page_mapping_and_config(tmp_path):
    baseline_docs = _write_document(tmp_path / "baseline")
    changed_passage_docs = _write_document(
        tmp_path / "passage", body=BODY.replace("საბოლოო", "შეცვლილ")
    )
    split = len(BODY) // 2
    changed_pages_docs = _write_document(
        tmp_path / "pages",
        page_boundaries=[
            {"page_number": 1, "char_start": 0, "char_end": split},
            {"page_number": 2, "char_start": split, "char_end": len(BODY)},
        ],
    )

    baseline = _compute(baseline_docs)
    passage = _compute(changed_passage_docs)
    pages = _compute(changed_pages_docs)
    config = _compute(baseline_docs, max_tokens=13)

    assert len(
        {
            baseline.artifact_sha256,
            passage.artifact_sha256,
            pages.artifact_sha256,
            config.artifact_sha256,
        }
    ) == 4
    assert baseline.identity_sha256 != config.identity_sha256
    assert baseline.identity["chunker"]["revision"]
    assert baseline.identity["document_header"]["revision"]
