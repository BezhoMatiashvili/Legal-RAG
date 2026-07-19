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


def test_legal_search_has_paraphrase_caution():
    doc = _doc(mcp_server.legal_search)
    assert "Paraphrase caution" in doc
    assert "statute's defined terms and article-style phrasing" in doc


def test_legal_search_does_not_treat_reranker_score_as_answer_confidence():
    doc = _doc(mcp_server.legal_search)
    assert "Search-score caution" in doc
    assert "not an answer probability" in doc
    assert "legal_ask" in doc
    assert "0.92" not in doc


def test_legal_ask_pins_the_strict_validation_contract():
    doc = _doc(mcp_server.legal_ask)
    assert "atomic claims" in doc
    assert "one repair" in doc
    assert "selective-risk calibrator" in doc
    assert "reranker sigmoid score" in doc


def test_citation_existence_check_in_search_and_get_document():
    for fn in (mcp_server.legal_search, mcp_server.legal_get_document):
        doc = _doc(fn)
        assert "never cite an identifier that does not resolve" in doc, fn


def test_i4_changes_nothing_inside_revision_two_serving_fingerprint():
    # Docstrings are not retrieval config. Revision 2 intentionally invalidated the old
    # digests once to remove storage names; these values now pin the revised material.
    import dataclasses

    base = dataclasses.replace(load_config(), rerank_enabled=True, rerank_backend="torch",
                               rerank_min_score=0.3)
    known = {
        ("torch", 80): "57431d11b802560e",
        ("torch", 50): "2fad68ff22b025d7",
        ("onnx", 50): "39358f4ea0787f38",
    }
    for (backend, rc), expect in known.items():
        cfg = dataclasses.replace(base, rerank_backend=backend, rerank_candidates=rc)
        assert retrieval_fingerprint(cfg) == expect, (backend, rc)
