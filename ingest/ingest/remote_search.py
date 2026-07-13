"""Thin stdlib-only client for the RunPod Serverless queue API (the "legal-search worker").

Used by ``mcp_server`` when ``SEARCH_BACKEND=remote``: every MCP tool becomes one
``call(op, params)`` that returns the worker's pre-formatted output string — the worker
runs the very same tool coroutines, so local and remote responses are identical.

Uses ``/run`` + ``/status`` polling instead of ``/runsync`` so the client controls its own
budget: a cold start (1–3 min; ~10–20 min on the first boot after a publish) would blow
past any sane synchronous timeout. On budget expiry the job is deliberately NOT cancelled —
the worker keeps warming up server-side and the caller gets an actionable "retry shortly"
error instead of a hang.

No third-party deps (mirrors ``RemoteBGEReranker``): this module must be importable in the
lean MCP runtime env.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

_BASE = "https://api.runpod.ai/v2"
_POLL_INTERVAL_S = 1.5   # cap on the poll cadence (cold-start steady state)
_POLL_INITIAL_S = 0.15   # first poll fires fast so a WARM job returns in ~1 poll instead of
# waiting a fixed 1.5 s; the interval then backs off geometrically toward the cap. Purely a
# transport-latency change — it does not affect any result the worker returns.
# Same browser UA the rest of the RunPod tooling sends (Cloudflare 403s the default
# python-urllib agent on api.runpod.io; harmless on api.runpod.ai).
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/125.0.0.0 Safari/537.36")

_TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMED_OUT"}


class RemoteSearchError(RuntimeError):
    """Base error for the remote backend; the message is agent-facing and actionable."""


class EndpointWarmingUp(RemoteSearchError):
    """The job did not finish within the client budget — almost always a cold start."""


class RemoteOpError(RemoteSearchError):
    """The worker returned an error, or the job failed platform-side."""


class RunPodQueueClient:
    """``call(op, params) -> output dict`` against one serverless endpoint."""

    def __init__(self, endpoint_id: str, api_key: str, timeout: int = 240,
                 poll_interval: float = _POLL_INTERVAL_S):
        if not endpoint_id or not api_key:
            raise RemoteSearchError(
                "SEARCH_BACKEND=remote requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in "
                "ingest/.env (or flip back to SEARCH_BACKEND=local).")
        self.endpoint_id = endpoint_id
        self._api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _request(self, method: str, path: str, body: dict | None = None,
                 timeout: float = 30) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{_BASE}/{self.endpoint_id}{path}", data=data, method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": _UA,
            })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:300]
            except Exception:  # noqa: BLE001
                pass
            raise RemoteSearchError(
                f"RunPod API {method} {path} → HTTP {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise RemoteSearchError(f"RunPod API unreachable ({e.reason})") from e

    def health(self) -> dict:
        """Endpoint health (worker/job counters). Instant; never wakes a worker."""
        return self._request("GET", "/health")

    def call(self, op: str, params: dict | None = None, timeout: float | None = None) -> dict:
        """Submit one op and poll to completion within the budget. Returns the job output."""
        budget = timeout if timeout is not None else self.timeout
        submitted = self._request("POST", "/run", {"input": {"op": op, "params": params or {}}})
        job_id = submitted.get("id")
        if not job_id:
            raise RemoteOpError(f"RunPod /run returned no job id: {submitted}")

        deadline = time.monotonic() + budget
        status = submitted.get("status") or "IN_QUEUE"
        poll_errors = 0
        # Fast-first poll: a warm job finishes in ~1 poll, so start well below the cap and
        # back off geometrically. self.poll_interval still governs the cold-start cadence.
        interval = min(_POLL_INITIAL_S, self.poll_interval)
        while time.monotonic() < deadline:
            # One blip (Cloudflare 502, connection reset) must not abandon a job that is
            # mid-cold-start on the worker; only persistent failure aborts the poll.
            try:
                st = self._request("GET", f"/status/{job_id}")
                poll_errors = 0
            except RemoteSearchError:
                poll_errors += 1
                if poll_errors >= 5:
                    raise
                time.sleep(max(self.poll_interval, 1.0) * poll_errors)
                continue
            status = st.get("status") or status
            if status == "COMPLETED":
                out = st.get("output")
                if not isinstance(out, dict):
                    raise RemoteOpError(f"worker returned unexpected output: {out!r}")
                if out.get("error"):
                    raise RemoteOpError(f"worker op {op!r} failed: {out['error']}")
                return out
            if status in _TERMINAL_FAILURES:
                raise RemoteOpError(
                    f"job {job_id} ({op}) ended {status}: {st.get('error') or 'no detail'}")
            time.sleep(interval)
            interval = min(interval * 1.6, self.poll_interval)  # geometric backoff to the cap

        # Not cancelled on purpose: let the worker finish booting so the retry lands warm.
        raise EndpointWarmingUp(
            f"The serverless endpoint did not answer op {op!r} within {budget:.0f}s "
            f"(job {job_id} still {status}). It is most likely cold-starting — normally "
            "1–3 minutes, or ~10–20 minutes on the first boot after a data publish. "
            "Retry the same call shortly; the worker keeps warming up in the background.")
