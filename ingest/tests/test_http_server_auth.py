"""OAuth front-door hardening: /login brute-force throttle, secret entropy floor,
DNS-rebinding defaults (ingest/ingest/http_server.py).

These test the pure pieces (LoginThrottle, validate_secret_strength,
_transport_security, redeem_pending) without starting uvicorn or loading models.
"""

import os

import pytest

os.environ.setdefault("RERANK_ENABLED", "false")

from ingest.http_server import (  # noqa: E402
    InMemoryOAuthProvider,
    LoginThrottle,
    _transport_security,
    validate_secret_strength,
)


class TestLoginThrottle:
    def test_allows_initial_attempts(self):
        throttle = LoginThrottle()
        assert throttle.retry_after("1.2.3.4", now=1000.0) == 0

    def test_ip_lockout_after_limit(self):
        throttle = LoginThrottle(ip_limit=5, base_lockout=30.0)
        for _ in range(4):
            throttle.record_failure("1.2.3.4", now=1000.0)
        assert throttle.retry_after("1.2.3.4", now=1000.0) == 0
        throttle.record_failure("1.2.3.4", now=1000.0)  # 5th failure -> lock
        assert throttle.retry_after("1.2.3.4", now=1000.0) == 30
        assert throttle.retry_after("1.2.3.4", now=1031.0) == 0

    def test_lockout_backs_off_exponentially(self):
        throttle = LoginThrottle(ip_limit=5, base_lockout=30.0, max_lockout=3600.0)
        for _ in range(5):
            throttle.record_failure("1.2.3.4", now=1000.0)
        throttle.record_failure("1.2.3.4", now=2000.0)  # 6th -> 60s
        assert throttle.retry_after("1.2.3.4", now=2000.0) == 60
        throttle.record_failure("1.2.3.4", now=3000.0)  # 7th -> 120s
        assert throttle.retry_after("1.2.3.4", now=3000.0) == 120

    def test_lockout_capped(self):
        throttle = LoginThrottle(ip_limit=1, base_lockout=30.0, max_lockout=100.0)
        for _ in range(30):
            throttle.record_failure("1.2.3.4", now=1000.0)
        assert throttle.retry_after("1.2.3.4", now=1000.0) == 100

    def test_success_resets_ip(self):
        throttle = LoginThrottle(ip_limit=5, base_lockout=30.0)
        for _ in range(5):
            throttle.record_failure("1.2.3.4", now=1000.0)
        throttle.record_success("1.2.3.4")
        assert throttle.retry_after("1.2.3.4", now=1000.0) == 0

    def test_other_ips_unaffected_by_ip_lockout(self):
        throttle = LoginThrottle(ip_limit=5, base_lockout=30.0)
        for _ in range(5):
            throttle.record_failure("1.2.3.4", now=1000.0)
        assert throttle.retry_after("5.6.7.8", now=1000.0) == 0

    def test_global_lockout_blocks_everyone(self):
        throttle = LoginThrottle(
            ip_limit=100, global_limit=10, global_window=600.0, global_lockout=900.0
        )
        for i in range(10):
            throttle.record_failure(f"10.0.0.{i}", now=1000.0)
        assert throttle.retry_after("99.99.99.99", now=1000.0) == 900
        assert throttle.retry_after("99.99.99.99", now=1901.0) == 0

    def test_global_window_expires_old_failures(self):
        throttle = LoginThrottle(
            ip_limit=100, global_limit=10, global_window=600.0, global_lockout=900.0
        )
        for i in range(9):
            throttle.record_failure(f"10.0.0.{i}", now=1000.0)
        # 10th failure arrives after the window: the 9 old ones no longer count.
        throttle.record_failure("10.0.0.9", now=1700.0)
        assert throttle.retry_after("99.99.99.99", now=1700.0) == 0


class TestSecretStrength:
    def test_rejects_short_secret(self):
        error = validate_secret_strength("hunter2")
        assert error is not None and "at least 16" in error

    def test_accepts_long_secret(self):
        assert validate_secret_strength("x" * 16) is None
        assert validate_secret_strength("a-long-random-secret-value") is None


class TestTransportSecurity:
    def test_explicit_allowlist_used_verbatim(self):
        settings = _transport_security(
            allowed_hosts_env="tunnel.example.com", public_url="https://x.example.com"
        )
        assert settings.enable_dns_rebinding_protection is True
        assert settings.allowed_hosts == ["tunnel.example.com"]

    def test_default_derives_from_public_url(self):
        settings = _transport_security(
            allowed_hosts_env="", public_url="https://abc.trycloudflare.com"
        )
        assert settings.enable_dns_rebinding_protection is True
        assert "abc.trycloudflare.com" in settings.allowed_hosts
        assert any(h.startswith("localhost") for h in settings.allowed_hosts)
        assert "https://abc.trycloudflare.com" in settings.allowed_origins

    def test_explicit_opt_out(self, monkeypatch):
        monkeypatch.setenv("MCP_DISABLE_DNS_REBIND_PROTECTION", "true")
        settings = _transport_security(
            allowed_hosts_env="", public_url="https://abc.trycloudflare.com"
        )
        assert settings.enable_dns_rebinding_protection is False


class TestRedeemPending:
    @pytest.fixture()
    def provider_with_secret(self, monkeypatch):
        monkeypatch.setattr("ingest.http_server._OAUTH_SECRET", "the-correct-secret!!")
        return InMemoryOAuthProvider()

    def test_wrong_secret_returns_none_and_keeps_pending(self, provider_with_secret):
        provider = provider_with_secret
        provider._pending["rid1"] = ("client1", _params(), 10**12)
        assert provider.redeem_pending("rid1", "wrong") is None
        assert "rid1" in provider._pending  # operator may retry

    def test_unknown_rid_returns_none(self, provider_with_secret):
        assert provider_with_secret.redeem_pending("nope", "the-correct-secret!!") is None


def _params():
    from pydantic import AnyUrl

    from mcp.server.auth.provider import AuthorizationParams

    return AuthorizationParams(
        state="s",
        scopes=[],
        code_challenge="c",
        redirect_uri=AnyUrl("https://claude.ai/cb"),
        redirect_uri_provided_explicitly=True,
        resource=None,
    )
