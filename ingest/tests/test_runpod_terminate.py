"""Regression: runpod_orchestrate.terminate() must set its _terminated backstop flag ONLY
after podTerminate actually succeeds. Setting it before the API call (the old bug) let one
transient Cloudflare/network failure permanently disable the finally + atexit cleanup, leaking
a billing GPU pod. Offline — gql is monkeypatched; no network, no real WORKDIR touched."""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(monkeypatch, tmp_path):
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import runpod_orchestrate as orch

    # Redirect all filesystem + logging side effects away from the real ~/gpu_embed_work.
    monkeypatch.setattr(orch, "WORKDIR", tmp_path)
    monkeypatch.setattr(orch, "log", lambda *a, **k: None)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)  # no real backoff delay
    orch._terminated = False
    orch._pod_id = None
    return orch


def test_terminate_keeps_backstop_armed_on_api_failure(monkeypatch, tmp_path):
    orch = _load(monkeypatch, tmp_path)
    calls = []

    def _boom(*_a, **_k):
        calls.append(1)
        raise RuntimeError("cloudflare 403 error 1010")

    monkeypatch.setattr(orch, "gql", _boom)
    (tmp_path / "pod.id").write_text("pod-abc")

    orch.terminate("pod-abc")

    assert orch._terminated is False, \
        "one podTerminate failure must NOT disable the finally/atexit backstop"
    assert (tmp_path / "pod.id").exists(), \
        "pod.id must be retained so the emergency `terminate` subcommand can still find the pod"
    assert len(calls) == 3, "should retry the termination a few times before giving up"


def test_terminate_disarms_and_cleans_on_success(monkeypatch, tmp_path):
    orch = _load(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "gql", lambda *_a, **_k: {})  # success
    (tmp_path / "pod.id").write_text("pod-abc")

    orch.terminate("pod-abc")

    assert orch._terminated is True
    assert not (tmp_path / "pod.id").exists(), "pod.id is unlinked only after a successful terminate"


def test_terminate_is_idempotent_once_disarmed(monkeypatch, tmp_path):
    orch = _load(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(orch, "gql", lambda *_a, **_k: calls.append(1) or {})
    orch._terminated = True  # already terminated
    orch.terminate("pod-abc")
    assert calls == [], "a second terminate after success must be a no-op"
