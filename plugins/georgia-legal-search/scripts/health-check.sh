#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)"
INGEST_DIR="$REPO_ROOT/ingest"
failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }

if command -v uv >/dev/null 2>&1; then
  pass "uv is available"
else
  fail "uv is not available on PATH"
fi

if command -v docker >/dev/null 2>&1; then
  pass "Docker CLI is available"
  if docker info >/dev/null 2>&1; then
    pass "Docker daemon is reachable"
  else
    warn "Docker daemon is not reachable"
  fi
else
  warn "Docker CLI is not available"
fi

if [ -d "$INGEST_DIR" ] && [ -f "$INGEST_DIR/pyproject.toml" ]; then
  pass "ingest project is present"
else
  fail "ingest project is missing at $INGEST_DIR"
fi

if [ -f "$INGEST_DIR/.env" ]; then
  pass "ingest/.env is present"
else
  warn "ingest/.env is missing; copy ingest/.env.example and configure it before serving"
fi

if [ -f "$INGEST_DIR/ingest/mcp_server.py" ]; then
  pass "MCP server module is present"
else
  fail "ingest/ingest/mcp_server.py is missing"
fi

if [ -x "$INGEST_DIR/.venv/bin/python" ]; then
  if (cd "$INGEST_DIR" && .venv/bin/python -c 'import ingest.mcp_server') >/dev/null 2>&1; then
    pass "MCP server imports in the ingest environment"
  else
    fail "MCP server import failed in ingest/.venv; run uv sync in ingest and inspect its Python environment"
  fi
else
  warn "ingest/.venv is missing; run uv sync in ingest before serving"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && [ -f "$INGEST_DIR/docker-compose.yml" ]; then
  if docker compose --project-directory "$INGEST_DIR" ps --status running --services 2>/dev/null | grep -q .; then
    pass "at least one ingest Compose service is running"
  else
    warn "no ingest Compose service is currently reported as running"
  fi
fi

printf '\nSummary: %d failure(s), %d warning(s)\n' "$failures" "$warnings"
if [ "$failures" -gt 0 ]; then
  exit 1
fi
