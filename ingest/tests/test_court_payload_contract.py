from dataclasses import replace
from types import SimpleNamespace

from ingest.chunking import Chunk
from ingest.court_extract import EXTRACTOR_REVISION
from ingest.pipeline import _document_state_hash
from ingest.qdrant_store import BOOL_FIELDS, KEYWORD_FIELDS, build_payload
from ingest.sources import PROMOTED_KEYWORD_FIELDS, normalize


def _supreme_item(**overrides):
    item = {
        "case_id": "42",
        "chamber": "civil",
        "date": "2026-07-01",
        "case_number": "ას-42-2026",
        "subject": "დავა",
        "result": "დატოვებულია უცვლელად",
        "appeal_type": "საკასაციო საჩივარი",
        "body_markdown": (
            "შემადგენლობა:\n"
            "ლაშა ქოჩიაშვილი (თავმჯდომარე, მომხსენებელი),\n"
            "პაატა სილაგაძე\nსაქმის განხილვის ფორმა\n"
            "გ ა დ ა წ ყ ვ ი ტ ა:\n1. საკასაციო საჩივარი არ დაკმაყოფილდეს."
        ),
    }
    item.update(overrides)
    return item


def _chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        canonical_text=text,
        chunk_index=0,
        heading_path=[],
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


def _hash_cfg():
    return SimpleNamespace(
        embed_model="model",
        embedding_revision=None,
        tokenizer_model="model",
        tokenizer_revision=None,
        dense_dim=8,
        chunk_tokens=512,
        chunk_overlap=80,
        chunk_min_tokens=64,
        embed_header_v2=False,
    )


def test_supreme_result_and_appeal_are_promoted_and_scraped_result_wins():
    doc = normalize("supremecourt", _supreme_item())
    assert doc.promoted == {
        "result": "დატოვებულია უცვლელად",
        "appeal_type": "საკასაციო საჩივარი",
    }
    assert doc.disposition == "upheld"
    assert doc.disposition_source == "scraped_result"
    assert doc.disposition_confidence == "high"
    assert doc.court_extractor_revision == EXTRACTOR_REVISION


def test_payload_replicates_structured_court_fields_on_a_chunk():
    doc = normalize("supremecourt", _supreme_item())
    payload = build_payload(doc, _chunk(doc.body_markdown))
    assert payload["judges"] == ["ლ. ქოჩიაშვილი", "პ. სილაგაძე"]
    assert payload["judges_raw"] == ["ლაშა ქოჩიაშვილი", "პაატა სილაგაძე"]
    assert payload["reporting_judge"] == "ლ. ქოჩიაშვილი"
    assert payload["judge_extraction_confidence"] == "high"
    assert payload["disposition"] == "upheld"
    assert payload["disposition_mixed"] is False
    assert payload["result"] == "დატოვებულია უცვლელად"
    assert payload["appeal_type"] == "საკასაციო საჩივარი"


def test_new_payload_fields_have_typed_indexes():
    for field in (
        "judges",
        "reporting_judge",
        "judge_extraction_confidence",
        "disposition",
        "disposition_source",
        "disposition_confidence",
        "court_extractor_revision",
    ):
        assert field in KEYWORD_FIELDS
    assert "disposition_mixed" in BOOL_FIELDS
    assert "result" in PROMOTED_KEYWORD_FIELDS
    assert "appeal_type" in PROMOTED_KEYWORD_FIELDS


def test_payload_only_court_metadata_does_not_move_document_state_hash():
    doc = normalize("supremecourt", _supreme_item())
    changed = replace(
        doc,
        promoted={"result": "დაკმაყოფილდა", "appeal_type": "კერძო საჩივარი"},
        judges=("ნ. სხირტლაძე",),
        judges_raw=("ნუგზარ სხირტლაძე",),
        reporting_judge="ნ. სხირტლაძე",
        disposition="granted",
        disposition_source="scraped_result",
        disposition_confidence="low",
        disposition_mixed=True,
        court_extractor_revision="future-revision",
    )
    assert _document_state_hash(_hash_cfg(), doc=doc) == _document_state_hash(
        _hash_cfg(), doc=changed
    )


def test_non_court_payload_omits_court_bundle():
    doc = normalize(
        "napr",
        {"document_id": "1", "body_markdown": "registry decision"},
    )
    payload = build_payload(doc, _chunk(doc.body_markdown))
    assert "judges" not in payload
    assert "disposition" not in payload

