"""Unit tests for Part-3 additions: diversify, query log, schema-drift, overlap stitching."""

from types import SimpleNamespace

import pytest

from ingest.config import load_config, retrieval_fingerprint
from ingest.mcp_server import _stitch_overlap
from ingest.querylog import append_query_log, build_query_record
from ingest.search import diversify
from ingest.sources import SOURCES, schema_drift


def _pt(doc, score, vec=None, ci=0, text="", cs=None, ce=None):
    payload = {"document_id": doc, "chunk_index": ci, "text": text,
               "char_start": cs, "char_end": ce}
    return SimpleNamespace(payload=payload, score=score,
                           vector=({"dense": vec} if vec else None))


# --- diversify (search.py) ----------------------------------------------------

def test_diversify_max_per_doc_caps_and_preserves_order():
    pts = [_pt("A", 0.9, ci=0), _pt("A", 0.8, ci=1), _pt("A", 0.7, ci=2),
           _pt("B", 0.6), _pt("C", 0.5)]
    out = diversify(pts, top_k=3, max_per_doc=1)
    assert [p.payload["document_id"] for p in out] == ["A", "B", "C"]


def test_diversify_max_per_doc_allows_n_per_doc():
    pts = [_pt("A", 0.9), _pt("A", 0.8), _pt("A", 0.7), _pt("B", 0.6)]
    out = diversify(pts, top_k=3, max_per_doc=2)
    assert [p.payload["document_id"] for p in out] == ["A", "A", "B"]


def test_diversify_noop_truncates_to_top_k():
    pts = [_pt("A", 0.9), _pt("B", 0.8), _pt("C", 0.7)]
    out = diversify(pts, top_k=2)
    assert len(out) == 2 and out[0].payload["document_id"] == "A"


def test_diversify_mmr_drops_near_duplicate_for_diverse_relevant():
    q = [1.0, 1.0, 1.0]
    pts = [_pt("A", 0.80, vec=[1, 1, 0]), _pt("B", 0.82, vec=[1, 1, 0.01]),
           _pt("C", 0.60, vec=[0, 0, 1])]
    out = diversify(pts, top_k=2, mmr_lambda=0.5, query_vec=q)
    docs = [p.payload["document_id"] for p in out]
    assert "C" in docs                       # diverse, still-relevant doc is kept
    assert not ("A" in docs and "B" in docs)  # the two near-identical don't both survive


def test_diversify_mmr_without_vectors_falls_back():
    out = diversify([_pt("A", 0.9), _pt("B", 0.8)], top_k=2, mmr_lambda=0.5)
    assert len(out) == 2


# --- querylog -----------------------------------------------------------------

def test_build_query_record_shape_and_no_text_leak():
    hits = [{"document_id": "d1", "source": "matsne", "chunk_index": 0, "score": 0.9,
             "text": "SECRET PII BODY"}]
    rec = build_query_record(query="q", filters={"source": "matsne", "court": None},
                             top_k=5, hits=hits, latency_ms=12.3, fingerprint="fp0", route="ka")
    assert rec["query"] == "q" and rec["top_k"] == 5 and rec["fingerprint"] == "fp0"
    assert rec["filters"] == {"source": "matsne"}          # empty filters dropped
    assert rec["hits"][0] == {"document_id": "d1", "source": "matsne",
                              "chunk_index": 0, "score": 0.9}
    assert "text" not in rec["hits"][0]                    # never log chunk text


def test_append_query_log_roundtrip(tmp_path):
    import json
    p = tmp_path / "q.jsonl"
    append_query_log({"a": 1}, p)
    append_query_log({"b": 2}, p)
    lines = p.read_text().splitlines()
    assert [json.loads(x) for x in lines] == [{"a": 1}, {"b": 2}]


def test_pii_query_written_locally_verbatim(tmp_path):
    import json
    p = tmp_path / "q.jsonl"
    rec = build_query_record(query="ს. წიკლაური 01001012345", filters={}, top_k=1,
                             hits=[], latency_ms=1.0, fingerprint="fp")
    append_query_log(rec, p)
    assert json.loads(p.read_text())["query"] == "ს. წიკლაური 01001012345"
    assert (p.stat().st_mode & 0o777) == 0o600


def test_query_log_refuses_insecure_existing_file_without_chmod(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("owner evidence\n", encoding="utf-8")
    p.chmod(0o644)

    with pytest.raises(PermissionError, match="refusing to chmod"):
        append_query_log({"new": True}, p)

    assert (p.stat().st_mode & 0o777) == 0o644
    assert p.read_text(encoding="utf-8") == "owner evidence\n"
    assert (p.parent.stat().st_mode & 0o777) == 0o700


def test_retrieval_fingerprint_stable_and_sensitive():
    import dataclasses
    cfg = load_config()
    assert retrieval_fingerprint(cfg) == retrieval_fingerprint(cfg)
    other = dataclasses.replace(cfg, rerank_candidates=cfg.rerank_candidates + 7)
    assert retrieval_fingerprint(cfg) != retrieval_fingerprint(other)


# --- schema drift (sources.py) ------------------------------------------------

def test_declared_keys_covers_read_fields():
    dk = SOURCES["matsne"].declared_keys()
    assert {"document_id", "title", "publication_date", "body_markdown",
            "registration_code", "status"} <= dk


def test_schema_drift_flags_new_and_undeclared_key():
    seen = SOURCES["matsne"].declared_keys()
    new, undeclared = schema_drift("matsne", {"document_id": "1", "brand_new_field": "x"}, seen)
    assert new == {"brand_new_field"} and undeclared == {"brand_new_field"}


def test_schema_drift_empty_when_all_known():
    seen = SOURCES["tas"].declared_keys()
    item = {k: "v" for k in ("document_id", "body_markdown", "applicant_phone")}
    new, undeclared = schema_drift("tas", item, seen)
    assert new == set() and undeclared == set()


# --- overlap-aware stitching (mcp_server.py) ----------------------------------

def test_stitch_overlap_dedupes_shared_boundary():
    # chunk1 body ends with "...ARTICLE TWO", chunk2 starts with "ARTICLE TWO ..." and the
    # char ranges overlap → the repeated "ARTICLE TWO" must appear once.
    p1 = _pt("d", 1.0, ci=0, text="Article one text. ARTICLE TWO", cs=0, ce=29)
    p2 = _pt("d", 1.0, ci=1, text="ARTICLE TWO continues here.", cs=18, ce=45)
    body = _stitch_overlap([p1, p2])
    assert body.count("ARTICLE TWO") == 1
    assert body.startswith("Article one text.")
    assert body.endswith("continues here.")


def test_stitch_overlap_paragraph_joins_when_no_overlap():
    p1 = _pt("d", 1.0, ci=0, text="Part A", cs=0, ce=6)
    p2 = _pt("d", 1.0, ci=1, text="Part B", cs=100, ce=106)  # ranges don't overlap
    assert _stitch_overlap([p1, p2]) == "Part A\n\nPart B"
