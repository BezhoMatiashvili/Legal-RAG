"""Structured court-field filter and document-projection contracts."""

from types import SimpleNamespace

from ingest import mcp_server
from ingest.mcp_server import BrowseInput, SearchInput
from ingest.search import build_filter


def _conditions_by_key(flt):
    return {
        condition.key: condition
        for condition in flt.must
        if hasattr(condition, "key")
    }


def test_court_filters_map_to_exact_payload_keys_and_normalize_judge():
    flt = build_filter(
        judges="ნუგზარ სხირტლაძე",
        disposition="overturned",
        appeal_type="საკასაციო",
    )

    conditions = _conditions_by_key(flt)
    assert conditions["judges"].match.value == "ნ. სხირტლაძე"
    assert conditions["disposition"].match.value == "overturned"
    assert conditions["appeal_type"].match.value == "საკასაციო"


def test_search_and_browse_forward_court_filters():
    search = SearchInput(
        query="გაუქმება",
        judge="ნუგზარ სხირტლაძე",
        disposition="overturned",
        appeal_type="საკასაციო",
    )
    request = mcp_server._retrieval_request(search)
    assert request.filters["judges"] == "ნუგზარ სხირტლაძე"
    assert request.filters["disposition"] == "overturned"
    assert request.filters["appeal_type"] == "საკასაციო"

    browse = BrowseInput(
        judge="ნ. სხირტლაძე",
        disposition="upheld",
        appeal_type="საკასაციო",
    )
    assert browse.judge == "ნ. სხირტლაძე"
    assert browse.disposition == "upheld"
    assert browse.appeal_type == "საკასაციო"


def test_document_projection_carries_structured_court_fields():
    expected = {
        "judges",
        "judges_raw",
        "reporting_judge",
        "judge_extraction_confidence",
        "disposition",
        "disposition_source",
        "disposition_confidence",
        "disposition_mixed",
        "court_extractor_revision",
        "result",
        "appeal_type",
    }
    assert expected <= set(mcp_server._DOCUMENT_PAYLOAD_FIELDS)

    point = SimpleNamespace(
        payload={
            "source": "supremecourt",
            "document_id": "case-1",
            "chunk_index": 0,
            "judges": ["ნ. სხირტლაძე"],
            "disposition": "overturned",
            "disposition_confidence": "high",
            "appeal_type": "საკასაციო",
        }
    )
    document = mcp_server._dedup_documents([point])[0]
    assert document["judges"] == ["ნ. სხირტლაძე"]
    assert document["disposition"] == "overturned"
    assert document["disposition_confidence"] == "high"
    assert document["appeal_type"] == "საკასაციო"
