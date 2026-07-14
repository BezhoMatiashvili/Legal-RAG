import sys
from pathlib import Path
from types import SimpleNamespace

from ingest.chunking import Chunk
from ingest.embed_job import snapshot_doc_to_canonical
from ingest.qdrant_store import BOOL_FIELDS, INTEGER_FIELDS, build_payload
from ingest.snapshot import _snapshot_record
from ingest.sources import CanonicalDoc, normalize

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import verify_matsne_completeness as V  # noqa: E402


def _matsne_item(**over):
    item = {
        "document_id": "18070",
        "title": "ცეცხლსასროლი იარაღის შესახებ",
        "document_type": "საქართველოს კანონი",
        "status": "ძალადაკარგული აქტები",
        "body_markdown": "x",
    }
    item.update(over)
    return item


def test_normalize_maps_consolidation_fields():
    doc = normalize("matsne", _matsne_item(is_consolidated=True, consolidated_count=3))
    assert doc.is_consolidated is True
    assert doc.consolidated_count == 3
    assert doc.status == "repealed"


def test_normalize_consolidation_absent_is_none():
    doc = normalize("matsne", _matsne_item())
    assert doc.is_consolidated is None
    assert doc.consolidated_count is None


def test_normalize_not_consolidated_is_false_zero():
    doc = normalize("matsne", _matsne_item(is_consolidated=False, consolidated_count=0))
    assert doc.is_consolidated is False
    assert doc.consolidated_count == 0


def test_normalize_coerces_stringy_count():
    doc = normalize("matsne", _matsne_item(is_consolidated=True, consolidated_count="5"))
    assert doc.consolidated_count == 5


def _doc(**over):
    base = dict(
        source="matsne", document_id="18070", title="T", date=None, date_raw=None,
        language="ka", document_type="law", court=None, source_url="u",
        document_number=None, registration_code=None, parties=None,
        status="repealed", status_raw=None, in_force_date=None, expiry_date=None,
        body_markdown="b", extra={},
    )
    base.update(over)
    return CanonicalDoc(**base)


def test_build_payload_carries_consolidation():
    doc = _doc(is_consolidated=True, consolidated_count=3)
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["is_consolidated"] is True
    assert p["consolidated_count"] == 3


def test_snapshot_roundtrip_preserves_consolidation_fields():
    doc = _doc(is_consolidated=True, consolidated_count=3)
    structure = SimpleNamespace(
        primary_kind="flat",
        has_article=False,
        has_heading=False,
        has_num_clause=False,
        article_count=0,
    )

    record = _snapshot_record("matsne", doc, doc.body_markdown, "hash", structure, "run")
    restored = snapshot_doc_to_canonical(record)

    assert record["is_consolidated"] is True
    assert record["consolidated_count"] == 3
    assert restored.is_consolidated is True
    assert restored.consolidated_count == 3


def test_consolidation_fields_are_indexed():
    assert "is_consolidated" in BOOL_FIELDS
    assert "consolidated_count" in INTEGER_FIELDS


def test_verify_pure_helpers():
    assert V.parse_last_page("a?page=3 b?page=15 c?page=2") == 15
    assert V.parse_last_page("no pages here") == 1
    assert V.expected_total(1567, 42) == 156642
    assert V.expected_total(0, 5) == 0
    assert V.is_absent_page("<title>Access Denied</title> Oops") is True
    assert V.is_absent_page("x" * 5000) is False
    assert V.residual_ids({"1", "2", "3"}, {"2"}) == {"1", "3"}
