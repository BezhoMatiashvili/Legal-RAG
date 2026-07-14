#!/usr/bin/env bash
# Daily incremental ingestion: scrape new docs → CPU-embed the delta → verify coverage.
#
#   stage 1  scrape   python -m legal_scrapers.run  (seen.sqlite keeps it delta-only)
#   stage 2  embed    python -m ingest watch --source <src> --once  per corpus source
#   stage 3  verify   scripts/verify_all_embedded.py  (ID coverage only)
#
# Run manually or via the systemd timer in ingest/systemd/. Safe by construction:
#   * flock self-exclusion — overlapping runs cannot double-ingest
#   * skips (exit 0) when a coordination lock is held (an eval or another writer is
#     in flight — writes turn the index yellow and break hybrid queries for everyone)
#   * every stage is idempotent (seen.sqlite, byte-offset watch state, UUIDv5 point ids)
#
# Flags:  --dry-run   preflight + plan only; no locks taken, nothing scraped/embedded.
# Env:    DAILY_INGEST_SOURCES        spiders to scrape (default: the 6 corpus sources —
#                                     supremecourt is excluded from the corpus by design)
#         DAILY_INGEST_LOOKBACK_DAYS  scrape window (default 14; dedup makes wider safe —
#                                     run a wide sweep monthly to catch late-published docs)
#         DAILY_INGEST_{SCRAPE,EMBED,VERIFY}_TIMEOUT  per-stage timeout(1) values
#         DAILY_INGEST_APPROVED=1 is required for every non-dry run. Set it only after
#                                     completing the deployment runbook gate.

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INGEST_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$INGEST_DIR")"
STATE_DIR="$INGEST_DIR/.state"
LOG_FILE="$STATE_DIR/daily_ingest.log"
ROTATING_TEE=("$INGEST_DIR/.venv/bin/python" "$SCRIPT_DIR/rotating_tee.py" "$LOG_FILE")
LOCK_DIR="$REPO_ROOT/coordination/locks"
OWNER="daily-ingest[$$]"

SOURCES="${DAILY_INGEST_SOURCES:-matsne ecd constcourt napr tas tbappeal}"
LOOKBACK_DAYS="${DAILY_INGEST_LOOKBACK_DAYS:-14}"
SCRAPE_TIMEOUT="${DAILY_INGEST_SCRAPE_TIMEOUT:-2h}"
EMBED_TIMEOUT="${DAILY_INGEST_EMBED_TIMEOUT:-3h}"
VERIFY_TIMEOUT="${DAILY_INGEST_VERIFY_TIMEOUT:-30m}"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown arg: $arg (only --dry-run is supported)" >&2; exit 2 ;;
    esac
done

if [ "$DRY_RUN" -eq 0 ]; then
    if [ "${DAILY_INGEST_APPROVED:-0}" != "1" ]; then
        echo "daily ingest refused: integrity verification and corpus re-baseline approval are pending" >&2
        echo "set DAILY_INGEST_APPROVED=1 only after completing the deployment runbook gate" >&2
        exit 78
    fi
    echo "daily ingest refused: direct writes to a serving corpus remain disabled; build a new immutable generation" >&2
    exit 78
fi

mkdir -p "$STATE_DIR"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | "${ROTATING_TEE[@]}"; }

# --- self-exclusion -----------------------------------------------------------------
exec 9>"$STATE_DIR/daily_ingest.flock"
if ! flock -n 9; then
    log "SKIP: another daily_ingest run holds the flock"
    exit 0
fi

# --- coordination locks (see coordination/README.md §4) -------------------------------
foreign_locks() {  # prints any lock we don't own; rc 0 = none found
    local found=0 f
    if [ -d "$LOCK_DIR" ]; then
        for f in "$LOCK_DIR"/*.lock; do
            [ -e "$f" ] || continue
            # -F: OWNER contains [$$] which is a bracket expression under BRE
            grep -qsF -- "owner: $OWNER" "$f" && continue
            log "foreign lock: $f — $(tr '\n' ' ' <"$f")"
            found=1
        done
    fi
    return $found
}

acquire_locks() {
    mkdir -p "$LOCK_DIR"
    local name
    for name in qdrant-write reranker-ram; do
        printf 'owner: %s\nwhy: daily incremental scrape+embed\nsince: %s\nexpected: <2h\n' \
            "$OWNER" "$(date -u +%FT%TZ)" > "$LOCK_DIR/$name.lock"
    done
}

release_locks() {
    local name
    for name in qdrant-write reranker-ram; do
        # if-guard: grep exits 2 on a missing file, which would errexit inside the trap;
        # -F: OWNER contains [$$] which is a bracket expression under BRE
        if grep -qsF -- "owner: $OWNER" "$LOCK_DIR/$name.lock" 2>/dev/null; then
            rm -f "$LOCK_DIR/$name.lock"
        fi
    done
}

STAGE="preflight"
on_exit() {
    local code=$?
    release_locks
    if [ "$code" -ne 0 ]; then
        log "FAILED at stage '$STAGE' (exit $code)"
    fi
    exit "$code"
}
trap on_exit EXIT

# --- Qdrant preflight -----------------------------------------------------------------
QDRANT_API_KEY="$(grep '^QDRANT_API_KEY=' "$INGEST_DIR/.env" | cut -d= -f2- || true)"
COLLECTION="$(grep '^COLLECTION_NAME=' "$INGEST_DIR/.env" | cut -d= -f2- || true)"
COLLECTION="${COLLECTION:-georgian_legal}"

qdrant_status() {
    curl -sf --max-time 10 -H "api-key: $QDRANT_API_KEY" \
        "localhost:6333/collections/$COLLECTION" 2>/dev/null \
        | python3 -c 'import sys, json; print(json.load(sys.stdin)["result"]["status"])' \
        2>/dev/null || echo down
}

wait_green() {  # arg: seconds to wait
    local deadline=$((SECONDS + $1)) s
    while (( SECONDS < deadline )); do
        s="$(qdrant_status)"
        [ "$s" = "green" ] && return 0
        log "waiting for Qdrant green (status=$s)"
        sleep 20
    done
    return 1
}

START_DATE="$(date -d "-${LOOKBACK_DAYS} days" +%F)"

if [ "$DRY_RUN" -eq 1 ]; then
    log "DRY RUN — plan:"
    log "  1) scrape:  (cd scraper && $REPO_ROOT/.venv/bin/python -m legal_scrapers.run --only $SOURCES --start-date $START_DATE --no-progress)"
    log "  2) embed:   (cd ingest && for src in $SOURCES: .venv/bin/python -m ingest watch --source \$src --once)"
    log "  3) verify:  (cd ingest && .venv/bin/python scripts/verify_all_embedded.py)"
    if foreign_locks; then log "  locks: clear"; else log "  locks: WOULD SKIP (foreign lock present)"; fi
    log "  qdrant: status=$(qdrant_status) collection=$COLLECTION"
    exit 0
fi

if ! foreign_locks; then
    log "SKIP: coordination lock held by another session — not touching the index"
    exit 0
fi
acquire_locks
log "=== daily_ingest start (sources: $SOURCES; window: $START_DATE..today) ==="

STAGE="qdrant-up"
(cd "$INGEST_DIR" && docker compose up -d) 2>&1 | "${ROTATING_TEE[@]}"
wait_green 900 || { log "Qdrant never reached green"; exit 1; }

STAGE="scrape"
log "--- stage 1/3: scrape ---"
# shellcheck disable=SC2086  # SOURCES is intentionally word-split
(cd "$REPO_ROOT/scraper" && timeout "$SCRAPE_TIMEOUT" \
    "$REPO_ROOT/.venv/bin/python" -m legal_scrapers.run \
    --only $SOURCES --start-date "$START_DATE" --no-progress) 2>&1 | "${ROTATING_TEE[@]}"

STAGE="embed"
log "--- stage 2/3: embed delta (CPU) ---"
# Per-source (not --source all): 'all' would also pick up artifacts/supremecourt if it
# ever appeared, and supremecourt is excluded from the corpus by design.
for src in $SOURCES; do
    (cd "$INGEST_DIR" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
        timeout "$EMBED_TIMEOUT" .venv/bin/python -m ingest watch --source "$src" --once) 2>&1 | "${ROTATING_TEE[@]}"
done

STAGE="verify"
log "--- stage 3/3: verify coverage ---"
(cd "$INGEST_DIR" && timeout "$VERIFY_TIMEOUT" \
    .venv/bin/python scripts/verify_all_embedded.py) 2>&1 | "${ROTATING_TEE[@]}"

STAGE="done"
log "=== daily_ingest OK ==="
