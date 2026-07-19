"""Streamable-HTTP front door for the ``legal_rag`` MCP server, with OAuth 2.1
(Dynamic Client Registration + PKCE) so the **claude.ai website** can connect to it as a
remote custom connector.

The stdio server (``ingest.mcp_server:main``) is left completely untouched: this module
*imports* the already-constructed ``mcp`` object (all tools registered on it at import
time) and attaches auth + an HTTP transport to it. Run it INSTEAD of the stdio server,
not alongside — both share the ``mcp_server.pid`` singleton and would otherwise double-load
~4.6 GB of models on this box (see the ``ram-discipline`` contract).

Auth model (minimal, single resource owner):
  * Open DCR — claude.ai self-registers a client (the SDK's ``/register`` handler does the
    work; our provider just persists the client).
  * The **authorize** step is gated by a shared secret (``MCP_OAUTH_SECRET``): ``authorize``
    does NOT mint a code, it redirects the browser to our own ``/login`` page, which only
    issues the authorization code after the operator types the secret. So merely reaching
    the public URL is not enough to obtain a token.
  * PKCE, redirect-uri, and code-expiry are verified by the SDK's ``/token`` handler before
    ``exchange_authorization_code`` runs — the provider only mints/stores/reads tokens.

Run:
    MCP_PUBLIC_URL=https://<your-tunnel-host>         \\
    MCP_OAUTH_SECRET=<a strong shared secret>         \\
    SEARCH_BACKEND=local                              \\
    uv run --directory ingest python -m ingest.http_server

Then expose port 8000 publicly (e.g. ``cloudflared tunnel --url http://localhost:8000``)
and add ``https://<your-tunnel-host>/mcp`` as a custom connector on claude.ai.
"""

from __future__ import annotations

import hmac
import html
import logging
import os
import secrets
import time

import uvicorn
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    ProviderTokenVerifier,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from ingest.mcp_server import _enforce_singleton, mcp

logger = logging.getLogger("legal_rag.http")

# --- configuration (env, read once at process start) ---------------------------------
_PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "").rstrip("/")
_OAUTH_SECRET = os.environ.get("MCP_OAUTH_SECRET", "")
_HOST = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
_PORT = int(os.environ.get("MCP_HTTP_PORT", "8000"))
# Access-token lifetime (claude.ai transparently refreshes with the refresh token).
_TOKEN_TTL = int(os.environ.get("MCP_TOKEN_TTL_SECONDS", str(30 * 24 * 3600)))
_CODE_TTL = 300  # authorization codes are single-use and short-lived
_PENDING_TTL = 600  # abandoned authorize flows are pruned after this many seconds

# The shared secret is the only thing between the public URL and a PII-bearing index,
# so it gets a hard entropy floor and the /login gate gets a brute-force throttle.
_MIN_SECRET_LENGTH = 16


class LoginThrottle:
    """Brute-force throttle for the /login secret gate.

    Per-IP: after ``ip_limit`` consecutive failures the IP is locked out with
    exponential backoff (``base_lockout`` doubling per further failure, capped).
    Global: after ``global_limit`` failures within ``global_window`` seconds, ALL
    logins are locked for ``global_lockout`` seconds — the backstop against
    distributed guessing and against spoofed forwarded-IP headers.
    A successful login clears that IP's failure history.
    """

    def __init__(
        self,
        ip_limit: int = 5,
        base_lockout: float = 30.0,
        max_lockout: float = 3600.0,
        global_limit: int = 50,
        global_window: float = 600.0,
        global_lockout: float = 900.0,
    ) -> None:
        self.ip_limit = ip_limit
        self.base_lockout = base_lockout
        self.max_lockout = max_lockout
        self.global_limit = global_limit
        self.global_window = global_window
        self.global_lockout = global_lockout
        self._failures: dict[str, tuple[int, float]] = {}  # ip -> (count, locked_until)
        self._global_failures: list[float] = []
        self._global_locked_until = 0.0

    def retry_after(self, ip: str, now: float | None = None) -> int:
        """Seconds the caller must wait before another attempt (0 = allowed)."""
        now = time.time() if now is None else now
        wait = self._global_locked_until - now
        count, locked_until = self._failures.get(ip, (0, 0.0))
        wait = max(wait, locked_until - now)
        return max(0, int(wait) + (1 if wait > int(wait) else 0))

    def record_failure(self, ip: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        count, _ = self._failures.get(ip, (0, 0.0))
        count += 1
        locked_until = 0.0
        if count >= self.ip_limit:
            lockout = min(
                self.base_lockout * (2 ** (count - self.ip_limit)), self.max_lockout
            )
            locked_until = now + lockout
        self._failures[ip] = (count, locked_until)
        self._global_failures = [
            ts for ts in self._global_failures if ts > now - self.global_window
        ]
        self._global_failures.append(now)
        if len(self._global_failures) >= self.global_limit:
            self._global_locked_until = now + self.global_lockout
            logger.warning(
                "login throttle: GLOBAL lockout engaged for %ss (%d failures in %ss)",
                self.global_lockout, len(self._global_failures), self.global_window,
            )

    def record_success(self, ip: str) -> None:
        self._failures.pop(ip, None)


def client_ip(request: Request) -> str:
    """Best-effort client IP behind the cloudflared tunnel.

    Forwarded headers are spoofable on direct (non-tunnel) connections, but the
    per-IP throttle only needs to be honest for well-behaved paths — the global
    lockout backstops anything that lies about its address.
    """
    for header in ("cf-connecting-ip", "x-forwarded-for"):
        value = request.headers.get(header, "")
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def validate_secret_strength(secret: str) -> str | None:
    """Startup gate: return an error message for a too-weak shared secret."""
    if len(secret) < _MIN_SECRET_LENGTH:
        return (
            f"MCP_OAUTH_SECRET must be at least {_MIN_SECRET_LENGTH} characters "
            f"(got {len(secret)}) — it is the only guard on a public PII-bearing "
            "index. Generate one with: python3 -c 'import secrets; "
            "print(secrets.token_urlsafe(24))'"
        )
    return None


class InMemoryOAuthProvider(OAuthAuthorizationServerProvider):
    """A minimal, single-owner OAuth 2.1 AS backed by process-memory dicts.

    Tokens do not survive a restart — claude.ai simply re-runs the OAuth flow (which
    costs the operator one shared-secret entry). That is an acceptable trade for a
    personal, single-connector deployment.
    """

    def __init__(self) -> None:
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}
        # authorize params parked while the browser passes the shared-secret gate,
        # keyed by a one-time id: rid -> (client_id, params, created_at)
        self._pending: dict[str, tuple[str, AuthorizationParams, float]] = {}

    # -- client registration (open DCR) --
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        # The SDK's /register handler already minted client_id/secret and validated
        # metadata; we only persist it.
        self._clients[client_info.client_id] = client_info

    # -- authorization: defer to the shared-secret login gate, mint no code here --
    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        self._prune_pending()
        rid = secrets.token_urlsafe(32)
        self._pending[rid] = (client.client_id, params, time.time())
        return f"{_PUBLIC_URL}/login?rid={rid}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._auth_codes.get(authorization_code)
        if code and code.client_id == client.client_id and code.expires_at >= time.time():
            return code
        return None

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # PKCE / redirect_uri / expiry already checked by the SDK's TokenHandler.
        self._auth_codes.pop(authorization_code.code, None)
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource)

    # -- refresh --
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        rt = self._refresh_tokens.get(refresh_token)
        if rt and rt.client_id == client.client_id:
            return rt
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        self._refresh_tokens.pop(refresh_token.token, None)
        return self._issue(client.client_id, scopes or refresh_token.scopes, None)

    # -- token verification path (ProviderTokenVerifier -> load_access_token) --
    async def load_access_token(self, token: str) -> AccessToken | None:
        at = self._access_tokens.get(token)
        if not at:
            return None
        if at.expires_at is not None and at.expires_at < time.time():
            self._access_tokens.pop(token, None)
            return None
        return at

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._access_tokens.pop(token.token, None)
        self._refresh_tokens.pop(token.token, None)

    # -- internals --
    def _issue(self, client_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        now = int(time.time())
        scopes = scopes or []
        self._access_tokens[access] = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + _TOKEN_TTL,
            resource=resource,
            subject="owner",
        )
        self._refresh_tokens[refresh] = RefreshToken(
            token=refresh, client_id=client_id, scopes=scopes, subject="owner"
        )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=_TOKEN_TTL,
            scope=" ".join(scopes) if scopes else None,
            refresh_token=refresh,
        )

    def _prune_pending(self) -> None:
        cutoff = time.time() - _PENDING_TTL
        for rid in [k for k, (_, _, ts) in self._pending.items() if ts < cutoff]:
            self._pending.pop(rid, None)

    def redeem_pending(self, rid: str, secret: str) -> str | None:
        """Validate the shared secret for a parked authorize flow. On success, mint the
        authorization code and return the redirect URL back to the client; on failure
        (bad/expired ``rid`` or wrong secret) return ``None``."""
        self._prune_pending()
        entry = self._pending.get(rid)
        if not entry:
            return None
        if not hmac.compare_digest(secret, _OAUTH_SECRET):
            return None  # keep the pending entry so the operator can retry
        self._pending.pop(rid, None)
        client_id, params, _ = entry
        code = secrets.token_urlsafe(32)
        self._auth_codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + _CODE_TTL,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject="owner",
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)


_provider = InMemoryOAuthProvider()
_throttle = LoginThrottle()


def _login_form(rid: str, error: str = "") -> HTMLResponse:
    err_html = f'<p style="color:#c00">{html.escape(error)}</p>' if error else ""
    body = f"""<!doctype html><meta charset="utf-8">
<title>Authorize legal_rag</title>
<div style="max-width:22rem;margin:6rem auto;font-family:system-ui">
  <h2>Authorize access to legal_rag</h2>
  <p>Enter the access secret to let this Claude connector query the legal index.</p>
  {err_html}
  <form method="post" action="/login">
    <input type="hidden" name="rid" value="{html.escape(rid)}">
    <input type="password" name="secret" placeholder="Access secret" autofocus
           style="width:100%;padding:.6rem;margin:.4rem 0;box-sizing:border-box">
    <button type="submit" style="width:100%;padding:.6rem">Authorize</button>
  </form>
</div>"""
    status = 200 if not error else 401
    return HTMLResponse(body, status_code=status)


async def _login_get(request: Request) -> HTMLResponse:
    rid = request.query_params.get("rid", "")
    if not rid:
        return HTMLResponse("<p>Missing authorization request.</p>", status_code=400)
    return _login_form(rid)


async def _login_post(request: Request):
    form = await request.form()
    rid = str(form.get("rid", ""))
    secret = str(form.get("secret", ""))
    if not rid:
        return HTMLResponse("<p>Missing authorization request.</p>", status_code=400)
    ip = client_ip(request)
    wait = _throttle.retry_after(ip)
    if wait:
        return HTMLResponse(
            f"<p>Too many failed attempts. Try again in {wait} seconds.</p>",
            status_code=429,
            headers={"Retry-After": str(wait)},
        )
    redirect = _provider.redeem_pending(rid, secret)
    if redirect is None:
        _throttle.record_failure(ip)
        logger.warning("login: failed secret attempt from %s", ip)
        return _login_form(rid, error="Incorrect secret — try again.")
    _throttle.record_success(ip)
    return RedirectResponse(redirect, status_code=302)


def _transport_security(
    allowed_hosts_env: str | None = None, public_url: str | None = None
) -> TransportSecuritySettings:
    """DNS-rebinding protection ON by default.

    Allowed hosts come from ``MCP_ALLOWED_HOSTS`` when set; otherwise they are
    derived from ``MCP_PUBLIC_URL``'s hostname plus the localhost variants the
    tunnel dials — the operator already re-exports MCP_PUBLIC_URL for each new
    tunnel hostname, so the derivation tracks the deployment automatically.
    ``MCP_DISABLE_DNS_REBIND_PROTECTION=true`` is the explicit opt-out.
    """
    if os.environ.get("MCP_DISABLE_DNS_REBIND_PROTECTION", "").lower() == "true":
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    raw = (
        allowed_hosts_env
        if allowed_hosts_env is not None
        else os.environ.get("MCP_ALLOWED_HOSTS", "")
    )
    allowed = [h.strip() for h in raw.split(",") if h.strip()]
    if not allowed:
        from urllib.parse import urlparse

        host = urlparse(public_url if public_url is not None else _PUBLIC_URL).hostname
        allowed = [h for h in (host,) if h]
        # The tunnel and local health checks dial the loopback listener directly.
        allowed += [f"localhost:{_PORT}", f"127.0.0.1:{_PORT}", "localhost", "127.0.0.1"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed,
        allowed_origins=[f"https://{h}" for h in allowed]
        + [f"http://{h}" for h in allowed if h.startswith(("localhost", "127.0.0.1"))],
    )


def _configure_auth() -> None:
    """Attach the OAuth provider + HTTP transport settings to the imported ``mcp``.

    All of these are plain instance attributes read by ``streamable_http_app()`` at call
    time, so setting them post-import is equivalent to passing them to ``FastMCP(...)``.
    """
    resource_url = f"{_PUBLIC_URL}/mcp"
    mcp.settings.host = _HOST
    mcp.settings.port = _PORT
    mcp.settings.auth = AuthSettings(
        issuer_url=_PUBLIC_URL,  # public HTTPS base; claude.ai discovers the AS from here
        resource_server_url=resource_url,
        client_registration_options=ClientRegistrationOptions(enabled=True),
        revocation_options=RevocationOptions(enabled=True),
        required_scopes=None,
    )
    mcp._auth_server_provider = _provider
    mcp._token_verifier = ProviderTokenVerifier(_provider)

    mcp.settings.transport_security = _transport_security()

    mcp.custom_route("/login", methods=["GET"])(_login_get)
    mcp.custom_route("/login", methods=["POST"])(_login_post)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    missing = [k for k, v in {"MCP_PUBLIC_URL": _PUBLIC_URL, "MCP_OAUTH_SECRET": _OAUTH_SECRET}.items() if not v]
    if missing:
        raise SystemExit(f"http_server: missing required env {missing}. See the module docstring.")
    weak = validate_secret_strength(_OAUTH_SECRET)
    if weak:
        raise SystemExit(f"http_server: {weak}")
    _is_local = _PUBLIC_URL.startswith(("http://localhost", "http://127.0.0.1"))
    if not _PUBLIC_URL.startswith("https://") and not _is_local:
        raise SystemExit(
            "http_server: MCP_PUBLIC_URL must be an https:// URL (claude.ai requires HTTPS); "
            "http://localhost is allowed for local testing only."
        )

    _enforce_singleton()  # reap any running stdio ingest.mcp_server so models aren't double-loaded
    _configure_auth()
    app = mcp.streamable_http_app()
    logger.info("legal_rag HTTP front door on %s:%s — public %s/mcp", _HOST, _PORT, _PUBLIC_URL)
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")


if __name__ == "__main__":
    main()
