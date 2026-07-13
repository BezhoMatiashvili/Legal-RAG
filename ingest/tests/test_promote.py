"""Tests for promoting structured (PII) fields into the canonical doc + Qdrant payload.

Personal data is retained by design — the index is local and confidential (prompt.md:40).
All values here are synthetic, not real personal data.
"""

from ingest.chunking import Chunk
from ingest.sources import PROMOTED_KEYWORD_FIELDS, PROMOTED_TEXT_FIELDS, normalize


def _tas_item():
    return {
        "document_id": "T1",
        "document_no": "AR-42",
        "nomenclature": "ნებართვა",
        "registration_date": "2024-06-03",
        "body_markdown": "ნებართვის ტექსტი საკმარისი სიგრძის დოკუმენტისთვის.",
        "applicant_personal_no": "01001000000",
        "applicant_birth_date": "1990-01-01",
        "applicant_address": "თბილისი, რუსთაველის 1",
        "applicant_phone": "599123456",
        "applicant_passport": "",           # empty → not promoted
        "executor_personal_no": "02002000000",
        "executor_phone": "577000000",
        "address": "ვაკე, ჭავჭავაძის 10",
    }


def test_tas_promotes_present_structured_fields_only():
    doc = normalize("tas", _tas_item())
    assert doc.promoted["applicant_personal_no"] == "01001000000"
    assert doc.promoted["executor_phone"] == "577000000"
    assert doc.promoted["address"] == "ვაკე, ჭავჭავაძის 10"
    assert "applicant_passport" not in doc.promoted  # empty value dropped


def test_non_promoting_source_has_empty_promoted():
    doc = normalize("matsne", {"document_id": "1", "title": "x", "body_markdown": "სამართალი"})
    assert doc.promoted == {}


def test_build_payload_includes_promoted_without_clobbering_canonical():
    store = __import__("ingest.qdrant_store", fromlist=["build_payload"])
    doc = normalize("tas", _tas_item())
    chunk = Chunk(text="ნებართვის ტექსტი", chunk_index=0, heading_path=[], token_count=3)
    payload = store.build_payload(doc, chunk)
    # promoted PII present and filterable-by-key
    assert payload["applicant_personal_no"] == "01001000000"
    assert payload["applicant_address"] == "თბილისი, რუსთაველის 1"
    # canonical fields intact
    assert payload["source"] == "tas" and payload["document_id"] == "T1"
    assert payload["text"] == "ნებართვის ტექსტი"


def test_promoted_index_field_lists_are_disjoint_from_canonical():
    canonical = {"source", "document_id", "document_type", "language", "court",
                 "document_number", "registration_code", "status", "title", "parties",
                 "text", "date", "in_force_date", "expiry_date", "chunk_index"}
    promoted = set(PROMOTED_KEYWORD_FIELDS) | set(PROMOTED_TEXT_FIELDS)
    assert promoted.isdisjoint(canonical)
