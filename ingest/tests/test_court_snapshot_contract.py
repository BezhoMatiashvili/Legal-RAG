from types import SimpleNamespace

import pytest

from ingest import snapshot
from ingest.embed_job import (
    EmbedStateError,
    snapshot_doc_to_canonical,
    validate_snapshot_build_config,
)
from ingest.snapshot import _snapshot_record
from ingest.sources import COURT_CANONICAL_FIELDS, normalize


BODY = (
    "მოსამართლეები: ნუგზარ სხირტლაძე\n"
    "დ ა ა დ გ ი ნ ა:\n1. საკასაციო საჩივარი არ დაკმაყოფილდეს."
)


def _legacy_record():
    return {
        "source": "ecd",
        "document_id": "7",
        "body_markdown": BODY,
        "promoted": {},
    }


def _serialized_record():
    doc = snapshot_doc_to_canonical(_legacy_record())
    record = _legacy_record()
    for field in COURT_CANONICAL_FIELDS:
        value = getattr(doc, field)
        record[field] = list(value) if isinstance(value, tuple) else value
    return record


def test_legacy_snapshot_record_recomputes_complete_bundle():
    doc = snapshot_doc_to_canonical(_legacy_record())
    assert doc.judges == ("ნ. სხირტლაძე",)
    assert doc.disposition == "upheld"
    assert doc.disposition_source == "body_operative"
    assert doc.court_extractor_revision == "court-extract-v1"


def test_complete_snapshot_bundle_is_accepted_and_revalidated():
    doc = snapshot_doc_to_canonical(_serialized_record())
    assert doc.judges == ("ნ. სხირტლაძე",)
    assert doc.disposition == "upheld"


def test_partial_snapshot_bundle_fails_loudly():
    record = _legacy_record()
    record["disposition"] = "upheld"
    with pytest.raises(EmbedStateError, match="bundle is partial"):
        snapshot_doc_to_canonical(record)


def test_mismatched_snapshot_bundle_fails_loudly():
    record = _serialized_record()
    record["disposition"] = "overturned"
    with pytest.raises(EmbedStateError, match="does not match"):
        snapshot_doc_to_canonical(record)


def test_snapshot_writer_serializes_the_complete_bundle():
    doc = normalize(
        "ecd",
        {
            "decision_document_id": "7",
            "body_markdown": BODY,
        },
    )
    structure = SimpleNamespace(
        primary_kind="plain",
        has_article=False,
        has_heading=False,
        has_num_clause=True,
        article_count=0,
    )
    record = _snapshot_record("ecd", doc, BODY, "a" * 64, structure, "run")
    assert set(COURT_CANONICAL_FIELDS).issubset(record)
    assert record["judges"] == ["ნ. სხირტლაძე"]
    assert record["disposition"] == "upheld"


def _snapshot_cfg():
    return SimpleNamespace(
        chunk_tokens=512,
        chunk_overlap=80,
        chunk_min_tokens=64,
        embed_header_v2=False,
        embed_model="embed-model",
        embedding_revision="a" * 40,
        tokenizer_model="tokenizer",
        tokenizer_revision="b" * 40,
    )


def test_snapshot_config_hash_pins_court_extractor_revision(monkeypatch):
    cfg = _snapshot_cfg()
    current = snapshot.config_hash(cfg)
    monkeypatch.setattr(snapshot, "EXTRACTOR_REVISION", "court-extract-v2")
    assert snapshot.config_hash(cfg) != current


def test_embed_validation_rejects_present_mismatched_extractor_revision():
    cfg = _snapshot_cfg()
    sealed = SimpleNamespace(
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
                "document_header": cfg.embed_header_v2,
                "court_extractor_revision": "court-extract-stale",
            }
        }
    )
    with pytest.raises(EmbedStateError, match="court_extractor_revision"):
        validate_snapshot_build_config(sealed, cfg)
