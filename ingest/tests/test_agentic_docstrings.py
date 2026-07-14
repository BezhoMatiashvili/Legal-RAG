"""Drift guards for the agentic-retrieval contract in the MCP tool docstrings (I4).

The MCP client (Claude) is the answer composer; the docstrings ARE the serving-layer
behavior contract, so these tests pin their load-bearing markers. The retrieval code
path is untouched by I4 — that invariant is covered by the fingerprint test below.
"""

import inspect

from ingest import mcp_server
from ingest.config import load_config, retrieval_fingerprint


def _doc(fn) -> str:
    return " ".join((inspect.getdoc(fn) or "").split())


def test_legal_search_has_retry_playbook():
    doc = _doc(mcp_server.legal_search)
    assert "Retry playbook" in doc
    assert "re-query in Georgian legal terminology" in doc
    assert "legal_get_document_versions" in doc
    assert "never present the nearest semantic match" in doc


def test_legal_search_warns_about_missing_base_laws():
    doc = _doc(mcp_server.legal_search)
    assert "consolidated base-law texts" in doc


def test_legal_search_has_calibrated_abstention_contract():
    doc = _doc(mcp_server.legal_search)
    assert "Abstention contract" in doc
    assert "below ~0.92" in doc  # calibrated 2026-07-10, .state/min_score_calibration.json
    assert "report that the document was not found" in doc


def test_citation_existence_check_in_search_and_get_document():
    for fn in (mcp_server.legal_search, mcp_server.legal_get_document):
        doc = _doc(fn)
        assert "never cite an identifier that does not resolve" in doc, fn


def test_i4_changes_nothing_in_serving_fingerprint():
    # Docstrings are not retrieval config: the fingerprint of each known serving config
    # must still equal its historical value (env-independent via explicit replace).
    import dataclasses

    base = dataclasses.replace(load_config(), rerank_enabled=True, rerank_backend="torch",
                               rerank_min_score=0.3)
    known = {
        ("torch", 80): "06a64f548fcb4d59",   # Phase C shipped config
        ("torch", 50): "81c807b279399098",   # Phase C recommended config
        ("onnx", 50): "471ee93fc0199b03",    # I7 serving flip (2026-07-10)
    }
    for (backend, rc), expect in known.items():
        cfg = dataclasses.replace(base, rerank_backend=backend, rerank_candidates=rc)
        assert retrieval_fingerprint(cfg) == expect, (backend, rc)
