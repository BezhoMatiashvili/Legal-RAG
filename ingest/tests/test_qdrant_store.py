from ingest.chunking import Chunk
from ingest.qdrant_store import KEYWORD_FIELDS, TEXT_FIELDS, _rfc3339, build_payload
from ingest.sources import CanonicalDoc


def _doc(**over):
    base = dict(
        source="ecd", document_id="5", title="T", date="2020-04-30", date_raw="2020-04-30",
        language="ka", document_type="court_decision", court="court", source_url="u",
        document_number=None, registration_code=None, parties=None,
        status=None, status_raw=None, in_force_date=None, expiry_date=None,
        body_markdown="b", extra={},
    )
    base.update(over)
    return CanonicalDoc(**base)


def test_rfc3339_validates_calendar_dates():
    assert _rfc3339("2020-04-30") == "2020-04-30T00:00:00Z"
    assert _rfc3339("2020-13-45") is None      # calendar-invalid -> dropped
    assert _rfc3339("2020-04-30 00:00:00") is None  # not a bare date
    assert _rfc3339("") is None
    assert _rfc3339(None) is None


def test_build_payload_shape():
    doc = _doc()
    chunk = Chunk(text="hello world", chunk_index=2, heading_path=["A", "B"], token_count=2)
    p = build_payload(doc, chunk)
    assert p["source"] == "ecd"
    assert p["document_id"] == "5"
    assert p["chunk_index"] == 2
    assert p["date"] == "2020-04-30T00:00:00Z"
    assert p["date_raw"] == "2020-04-30"
    assert p["heading"] == "A > B"
    assert p["text"] == "hello world"
    assert p["language"] == "ka"


def test_build_payload_carries_new_fields():
    doc = _doc(
        source="matsne", document_number="55", registration_code="140130000.22.034.017712",
        parties="ს. წიკლაური",
    )
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["document_number"] == "55"
    assert p["registration_code"] == "140130000.22.034.017712"
    assert p["parties"] == "ს. წიკლაური"


def test_build_payload_keeps_raw_date_independent_of_iso():
    # date is the ISO (sortable) value; date_raw preserves the original scraped string.
    doc = _doc(date="2026-04-08", date_raw="08/04/2026")
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["date"] == "2026-04-08T00:00:00Z"
    assert p["date_raw"] == "08/04/2026"


def test_build_payload_handles_bad_date():
    doc = _doc(date=None, date_raw="not-a-date", title=None, document_type=None,
               court=None, source_url=None)
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["date"] is None          # invalid -> not indexed
    assert p["date_raw"] == "not-a-date"
    assert p["heading"] is None


def test_build_payload_carries_status_and_force_dates():
    doc = _doc(source="matsne", status="repealed", status_raw="ძალადაკარგული აქტები",
               in_force_date="2020-01-01", expiry_date="2024-06-03")
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["status"] == "repealed"
    assert p["in_force_date"] == "2020-01-01T00:00:00Z"   # promoted to RFC3339 for the index
    assert p["expiry_date"] == "2024-06-03T00:00:00Z"


def test_new_index_fields_are_declared():
    # The fields the new lookup/browse tools filter on must be indexable.
    assert "document_number" in KEYWORD_FIELDS
    assert "registration_code" in KEYWORD_FIELDS
    assert "status" in KEYWORD_FIELDS
    assert set(TEXT_FIELDS) == {"text", "title", "parties"}
