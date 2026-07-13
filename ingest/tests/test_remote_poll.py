"""Fast-first poll backoff in RunPodQueueClient — transport-only (wire shape unchanged).

The warm path must return in ~1 fast poll instead of waiting the fixed cap, while the interval
still backs off toward the cap for cold starts. These assert the sleep *schedule*; the
value/exception behaviour is covered in test_remote_backend.py (which stays green)."""
from __future__ import annotations

import ingest.remote_search as rs
from ingest.remote_search import RunPodQueueClient


def _client(seq, monkeypatch, poll_interval=1.5):
    slept: list[float] = []
    monkeypatch.setattr(rs.time, "sleep", lambda s: slept.append(s))
    it = iter(seq)
    c = RunPodQueueClient("ep", "k", poll_interval=poll_interval)
    c._request = lambda method, path, body=None, timeout=30: next(it)
    return c, slept


def test_warm_job_waits_only_the_fast_floor(monkeypatch):
    # submit -> one IN_PROGRESS poll -> COMPLETED: exactly one sleep, at the fast floor.
    seq = [{"id": "j", "status": "IN_QUEUE"},
           {"status": "IN_PROGRESS"},
           {"status": "COMPLETED", "output": {"result": "warm"}}]
    c, slept = _client(seq, monkeypatch)
    assert c.call("search", {}) == {"result": "warm"}
    assert slept == [rs._POLL_INITIAL_S]
    assert rs._POLL_INITIAL_S < rs._POLL_INTERVAL_S  # genuinely faster than the old fixed wait


def test_interval_backs_off_geometrically_to_the_cap(monkeypatch):
    seq = ([{"id": "j", "status": "IN_QUEUE"}]
           + [{"status": "IN_PROGRESS"}] * 6
           + [{"status": "COMPLETED", "output": {"result": "ok"}}])
    c, slept = _client(seq, monkeypatch)
    assert c.call("search", {}) == {"result": "ok"}
    assert len(slept) == 6
    assert slept[0] == rs._POLL_INITIAL_S
    assert slept == sorted(slept)                 # monotonic non-decreasing
    assert max(slept) <= rs._POLL_INTERVAL_S      # never exceeds the cap
    assert slept[-1] == rs._POLL_INTERVAL_S       # reaches the cap after enough polls


def test_zero_interval_stays_zero(monkeypatch):
    # The existing suite constructs clients with poll_interval=0.0; the backoff must degenerate
    # to zero so those tests keep their behaviour.
    seq = [{"id": "j", "status": "IN_QUEUE"},
           {"status": "IN_PROGRESS"},
           {"status": "COMPLETED", "output": {"result": "ok"}}]
    c, slept = _client(seq, monkeypatch, poll_interval=0.0)
    assert c.call("search", {}) == {"result": "ok"}
    assert slept == [0.0]
