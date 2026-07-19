"""RemoteBGEReranker server-config parity checks (memory-bank reranker-score-parity).

Local and pod read RERANK_MODEL / RERANK_MAX_LENGTH from independent process envs;
these tests lock in that a *reported* mismatch fails loudly before any scoring
request, while an unreachable or older (non-reporting) server only warns.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ingest.rerank import RemoteBGEReranker, RerankerParityError


def _serve(info: dict, scores: list[float]):
    """Tiny stub of runpod_rerank_server: GET / -> info, POST /score -> scores."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._send(info)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            self._send({"scores": scores})

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_matching_config_scores_normally():
    server, url = _serve(
        {"ok": True, "model": "BAAI/bge-reranker-v2-m3", "max_length": 512}, [0.9, 0.1]
    )
    try:
        reranker = RemoteBGEReranker(
            url, expected_model="BAAI/bge-reranker-v2-m3", expected_max_length=512
        )
        assert reranker.score("q", ["a", "b"]) == [0.9, 0.1]
        assert reranker._parity_checked
    finally:
        server.shutdown()


def test_max_length_mismatch_raises_before_scoring():
    server, url = _serve(
        {"ok": True, "model": "BAAI/bge-reranker-v2-m3", "max_length": 1024}, [0.9]
    )
    try:
        reranker = RemoteBGEReranker(
            url, expected_model="BAAI/bge-reranker-v2-m3", expected_max_length=512
        )
        with pytest.raises(RerankerParityError, match="max_length mismatch"):
            reranker.score("q", ["a"])
    finally:
        server.shutdown()


def test_model_mismatch_raises_before_scoring():
    server, url = _serve(
        {"ok": True, "model": "some/other-model", "max_length": 512}, [0.9]
    )
    try:
        reranker = RemoteBGEReranker(
            url, expected_model="BAAI/bge-reranker-v2-m3", expected_max_length=512
        )
        with pytest.raises(RerankerParityError, match="model mismatch"):
            reranker.score("q", ["a"])
    finally:
        server.shutdown()


def test_older_server_without_max_length_warns_and_proceeds(caplog):
    server, url = _serve({"ok": True, "model": "BAAI/bge-reranker-v2-m3"}, [0.5])
    try:
        reranker = RemoteBGEReranker(
            url, expected_model="BAAI/bge-reranker-v2-m3", expected_max_length=512
        )
        with caplog.at_level("WARNING", logger="ingest.rerank"):
            assert reranker.score("q", ["a"]) == [0.5]
        assert any("UNVERIFIED" in r.message for r in caplog.records)
    finally:
        server.shutdown()


def test_unreachable_server_warns_then_fails_in_score(caplog):
    reranker = RemoteBGEReranker(
        "http://127.0.0.1:9",  # discard port — connection refused
        timeout=2,
        expected_model="BAAI/bge-reranker-v2-m3",
        expected_max_length=512,
    )
    with caplog.at_level("WARNING", logger="ingest.rerank"), pytest.raises(Exception):
        reranker.score("q", ["a"])
    assert any("parity check skipped" in r.message for r in caplog.records)


def test_empty_texts_short_circuits_without_network():
    reranker = RemoteBGEReranker("http://127.0.0.1:9")
    assert reranker.score("q", []) == []
    assert not reranker._parity_checked


def test_expectation_defaults_track_env(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "env/model")
    monkeypatch.setenv("RERANK_MAX_LENGTH", "384")
    reranker = RemoteBGEReranker("http://127.0.0.1:9")
    assert reranker.expected_model == "env/model"
    assert reranker.expected_max_length == 384
