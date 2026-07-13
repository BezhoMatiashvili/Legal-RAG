"""In-process result cache fronting legal_search — quality-neutral repeat-query speedup.

Byte-identical repeats skip the whole retrieval. Exercised in remote mode with a fake worker
so a cache MISS is observable as an extra worker call, and a HIT as the absence of one."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ingest import mcp_server as srv
from ingest.config import load_config


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Keep the module-global cache from leaking across tests."""
    srv._result_cache.clear()
    yield
    srv._result_cache.clear()


class _FakeWorker:
    def __init__(self):
        self.calls: list[tuple] = []

    def call(self, op, params=None, timeout=None):
        self.calls.append((op, params))
        # A valid positive Markdown search response; distinct per call makes a re-run visible.
        return {"result": f"# 1 result\n\nR{len(self.calls)}:{op}"}

    def health(self):
        return {}


def _remote_cfg(**over):
    base = dict(search_backend="remote", runpod_endpoint_id="ep", runpod_api_key="k",
                result_cache_enabled=True, result_cache_ttl=1800, result_cache_size=64,
                query_log_enabled=False)
    base.update(over)
    return replace(load_config(), **base)


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(srv, "_cfg", _remote_cfg())
    fake = _FakeWorker()
    monkeypatch.setattr(srv, "_remote_client", fake)
    return fake


def test_identical_query_is_served_from_cache(worker):
    a = asyncio.run(srv.legal_search(srv.SearchInput(query="იარაღი", top_k=5)))
    b = asyncio.run(srv.legal_search(srv.SearchInput(query="იარაღი", top_k=5)))
    assert a == b                     # byte-identical
    assert len(worker.calls) == 1     # the 2nd call never reached the worker


def test_distinct_requests_are_distinct_keys(worker):
    asyncio.run(srv.legal_search(srv.SearchInput(query="იარაღი", top_k=5)))
    asyncio.run(srv.legal_search(srv.SearchInput(query="იარაღი", top_k=6)))    # top_k differs
    asyncio.run(srv.legal_search(srv.SearchInput(query="სხვა", top_k=5)))      # query differs
    asyncio.run(srv.legal_search(
        srv.SearchInput(query="იარაღი", top_k=5, status="in_force")))          # filter differs
    assert len(worker.calls) == 4


def test_disabled_knob_never_caches(monkeypatch):
    monkeypatch.setattr(srv, "_cfg", _remote_cfg(result_cache_enabled=False))
    fake = _FakeWorker()
    monkeypatch.setattr(srv, "_remote_client", fake)
    asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    assert len(fake.calls) == 2


def test_cache_is_opt_in_by_default(monkeypatch):
    monkeypatch.delenv("RESULT_CACHE_ENABLED", raising=False)
    assert load_config().result_cache_enabled is False


def test_ttl_expiry_refetches(worker, monkeypatch):
    asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    real = srv.time.monotonic
    monkeypatch.setattr(srv.time, "monotonic", lambda: real() + 10_000)  # jump past the TTL
    asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    assert len(worker.calls) == 2


def test_errors_are_not_cached(worker, monkeypatch):
    def boom(op, params=None, timeout=None):
        raise RuntimeError("worker down")

    monkeypatch.setattr(worker, "call", boom)
    a = asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    assert "SEARCH_BACKEND=local" in a          # actionable error, not a result
    assert len(srv._result_cache) == 0          # nothing stored
    # recovery: a later success must reach the worker, not serve the error from cache
    monkeypatch.setattr(worker, "call",
                        lambda op, params=None, timeout=None: {"result": "recovered"})
    b = asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    assert b == "recovered"


@pytest.mark.parametrize(
    "result",
    [
        "No results found. Try a broader query or remove filters.",
        "> WARNING: retrieval degraded — hybrid only\n\n# 1 results",
        '{"count": 1, "hits": [], "degraded": false}',
        '{"count": 1, "hits": [{"document_id": "1"}], "degraded": true}',
    ],
)
def test_empty_or_degraded_results_are_not_cacheable(result):
    assert srv._cacheable_search_result(result) is False


def test_fingerprint_change_invalidates(worker, monkeypatch):
    asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    monkeypatch.setattr(srv, "_cfg", _remote_cfg(rerank_candidates=999))  # moves the fingerprint
    asyncio.run(srv.legal_search(srv.SearchInput(query="q", top_k=5)))
    assert len(worker.calls) == 2
