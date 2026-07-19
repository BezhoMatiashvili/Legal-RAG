"""Exact document-level judge-panel statistics and fail-closed trust tests."""

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from ingest import mcp_server
from ingest.config import load_config
from ingest.mcp_server import StatsInput, legal_stats


def _field_values(flt) -> dict[str, object]:
    values: dict[str, object] = {}
    for condition in flt.must or []:
        if hasattr(condition, "key") and getattr(condition, "match", None):
            values[condition.key] = getattr(
                condition.match, "value", getattr(condition.match, "any", None)
            )
    return values


class StatsClient:
    def __init__(self, *, revision_count: int = 10):
        self.revision_count = revision_count
        self.count_calls = []
        self.facet_calls = []

    def count(self, **kwargs):
        self.count_calls.append(kwargs)
        fields = _field_values(kwargs["count_filter"])
        if "court_extractor_revision" in fields:
            return SimpleNamespace(count=self.revision_count)
        return SimpleNamespace(count=10)

    def facet(self, **kwargs):
        self.facet_calls.append(kwargs)
        return SimpleNamespace(
            hits=[
                SimpleNamespace(value="overturned", count=2),
                SimpleNamespace(value="overturned_remanded", count=1),
                SimpleNamespace(value="partially_overturned", count=1),
                SimpleNamespace(value="upheld", count=2),
                SimpleNamespace(value="inadmissible", count=1),
                SimpleNamespace(value="not_considered", count=1),
            ]
        )


@pytest.fixture
def stats_runtime(tmp_path):
    old_cfg, old_client = mcp_server._cfg, mcp_server._client
    mcp_server._cfg = dataclasses.replace(
        load_config(), collection_name="court_stats_test", state_dir=tmp_path / "state"
    )
    client = StatsClient()
    mcp_server._client = client
    yield client
    mcp_server._cfg, mcp_server._client = old_cfg, old_client


def _trusted_decision():
    return SimpleNamespace(
        trusted=True,
        reasons=(),
        eval_set_hash="a" * 64,
    )


def test_legal_stats_counts_documents_once_and_separates_procedural(
    stats_runtime, monkeypatch
):
    monkeypatch.setattr(
        mcp_server, "evaluate_court_extraction_trust", lambda **_kwargs: _trusted_decision()
    )

    result = json.loads(
        asyncio.run(legal_stats(StatsInput(judge="ნუგზარ სხირტლაძე")))
    )

    assert result["judge_key"] == "ნ. სხირტლაძე"
    assert result["numerator"] == 4
    assert result["confident_denominator"] == 6
    assert result["total_denominator"] == 10
    assert result["unknown"] == 2
    assert result["coverage"] == 0.8
    assert result["procedural"] == {
        "inadmissible": 1,
        "not_considered": 1,
        "total": 2,
    }
    assert result["trusted"] is True
    assert "4 of 6 confidently-classified decisions" in result["statement"]
    assert "own ruling is final" in result["statement"]

    assert len(stats_runtime.count_calls) == 4
    for call in stats_runtime.count_calls:
        assert call["exact"] is True
        assert _field_values(call["count_filter"])["chunk_index"] == 0
    assert len(stats_runtime.facet_calls) == 1
    facet_call = stats_runtime.facet_calls[0]
    assert facet_call["exact"] is True
    facet_fields = _field_values(facet_call["facet_filter"])
    assert facet_fields["chunk_index"] == 0
    assert facet_fields["disposition_confidence"] == "high"


def test_legal_stats_fails_closed_on_attestation_or_collection_mismatch(
    stats_runtime, monkeypatch
):
    stats_runtime.revision_count = 9
    monkeypatch.setattr(
        mcp_server,
        "evaluate_court_extraction_trust",
        lambda **_kwargs: SimpleNamespace(
            trusted=False,
            reasons=("attestation_missing",),
            eval_set_hash=None,
        ),
    )

    result = json.loads(asyncio.run(legal_stats(StatsInput(judge="ნ. სხირტლაძე"))))

    assert result["trusted"] is False
    assert "attestation_missing" in result["trust_reasons"]
    assert "collection_extractor_revision_incomplete" in result["trust_reasons"]


def test_legal_stats_remote_branch_forwards_stats_operation(monkeypatch):
    captured = {}

    async def remote(op, params):
        captured.update(op=op, params=params)
        return "remote-result"

    monkeypatch.setattr(mcp_server, "_use_remote", lambda: True)
    monkeypatch.setattr(mcp_server, "_remote_op", remote)

    result = asyncio.run(
        legal_stats(
            StatsInput(
                judge="ნ. სხირტლაძე",
                source="supremecourt",
                outcome="overturned",
            )
        )
    )

    assert result == "remote-result"
    assert captured == {
        "op": "stats",
        "params": {
            "judge": "ნ. სხირტლაძე",
            "outcome": "overturned",
            "source": "supremecourt",
        },
    }
