import uuid

from ingest.qdrant_store import point_id


def test_point_id_is_deterministic():
    assert point_id("ecd", "5861405", 0) == point_id("ecd", "5861405", 0)


def test_point_id_is_a_valid_uuid():
    uuid.UUID(point_id("ecd", "5861405", 3))  # raises if malformed


def test_point_id_varies_by_chunk_source_and_doc():
    base = point_id("ecd", "5861405", 0)
    assert base != point_id("ecd", "5861405", 1)      # chunk
    assert base != point_id("ecd", "5861406", 0)      # document
    assert base != point_id("constcourt", "5861405", 0)  # source
