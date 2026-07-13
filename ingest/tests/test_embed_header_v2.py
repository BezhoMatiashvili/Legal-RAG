"""Tests for the I6 v2 embed headers (contextual metadata in the embedded prefix)."""

import dataclasses
from types import SimpleNamespace

from ingest.chunking import build_embed_text
from ingest.config import load_config
from ingest.pipeline import _header_v2_kwargs


def test_v1_output_byte_identical_without_v2_params():
    out = build_embed_text("body", title="კანონი X", document_type="law",
                           heading_path=["თავი I", "მუხლი 3"])
    assert out == "კანონი X > law > თავი I > მუხლი 3\n\nbody"


def test_v2_header_inserts_meta_segment_after_document_type():
    out = build_embed_text(
        "body", title="შრომის კოდექსი", document_type="კანონი",
        heading_path=["მუხლი 31"], document_number="4113-რს",
        date="2010-12-17T00:00:00Z", status="in_force", is_consolidated=True)
    assert out == ("შრომის კოდექსი > კანონი > "
                   "№4113-რს · 2010-12-17 · ძალაშია · კონსოლიდირებული > მუხლი 31\n\nbody")


def test_v2_header_skips_placeholder_number_and_maps_status():
    out = build_embed_text("body", title="t", document_number="0", status="repealed")
    assert "№" not in out and "ძალადაკარგულია" in out


def test_v2_unknown_status_passes_through():
    out = build_embed_text("body", title="t", status="draft")
    assert "draft" in out


def test_header_kwargs_empty_when_knob_off():
    cfg = dataclasses.replace(load_config(), embed_header_v2=False)
    assert _header_v2_kwargs(cfg, SimpleNamespace()) == {}


def test_header_kwargs_populated_when_knob_on():
    cfg = dataclasses.replace(load_config(), embed_header_v2=True)
    doc = SimpleNamespace(document_number="71", date="2019-08-16", date_raw="16/08/2019",
                          status="in_force", is_consolidated=False)
    kw = _header_v2_kwargs(cfg, doc)
    assert kw == {"document_number": "71", "date": "2019-08-16",
                  "status": "in_force", "is_consolidated": False}


def test_config_reads_embed_header_v2_env(monkeypatch):
    monkeypatch.setenv("EMBED_HEADER_V2", "true")
    assert load_config().embed_header_v2 is True
    monkeypatch.delenv("EMBED_HEADER_V2")
    assert load_config().embed_header_v2 is False


def test_serving_fingerprint_changes_when_v2_header_is_enabled():
    from ingest.config import retrieval_fingerprint

    cfg = load_config()
    off = dataclasses.replace(cfg, embed_header_v2=False)
    on = dataclasses.replace(cfg, embed_header_v2=True)
    assert retrieval_fingerprint(off) != retrieval_fingerprint(on)
