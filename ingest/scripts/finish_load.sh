#!/usr/bin/env bash
# Finish loading the Georgian-legal corpus into local Qdrant.
# Resumes the snapshot download from the pod (if parts are missing), reassembles,
# verifies md5, restores into local Qdrant as `georgian_legal`, and confirms the count.
# Safe to re-run: completed parts are skipped; a bad md5 aborts before restoring.
set -uo pipefail

echo "Legacy direct restore into georgian_legal is disabled; use an immutable generation and guarded promotion." >&2
exit 78

: "${POD_IP:?set POD_IP to the source pod address}"
: "${POD_PORT:?set POD_PORT to the source pod SSH port}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INGEST_DIR="$(dirname "$SCRIPT_DIR")"
WORKDIR="${GPU_WORKDIR:-$INGEST_DIR/.state/gpu-work}"
KEY="${GPU_SSH_KEY:-$WORKDIR/id_ed25519}"
KH="${GPU_KNOWN_HOSTS:-$WORKDIR/known_hosts}"
PD="$WORKDIR/out_multi/parts"
SNAP="$WORKDIR/out_multi/georgian_legal.snapshot"
SRC_MD5=378985d49eac43f681e90a4fbdeb47ea
SIZE=24166869504
SSHOPTS="-p $POD_PORT -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH -o ConnectTimeout=20 -o ServerAliveInterval=20"
mkdir -p "$PD"

have=$(ls "$PD"/snap_part_* 2>/dev/null | wc -l)
echo "[$(date +%T)] start — have $have/58 parts locally"

if [ "$have" -lt 58 ]; then
  echo "[$(date +%T)] parts missing; checking pod is reachable..."
  if ! ssh $SSHOPTS root@$POD_IP true 2>/dev/null; then
    echo "POD UNREACHABLE ($POD_IP:$POD_PORT) — it may be terminated/out of balance."
    echo "If the pod is gone, the snapshot is lost and the corpus must be re-embedded."
    exit 2
  fi
  echo "[$(date +%T)] pod up; resuming download (skips finished parts, retries on drops)..."
  n=0
  until rsync -t --partial --inplace -e "ssh $SSHOPTS" "root@$POD_IP:/workspace/out/snap_part_*" "$PD/"; do
    n=$((n+1)); echo "  [$(date +%T)] retry $n — have $(ls "$PD"/snap_part_* 2>/dev/null | wc -l)/58; sleep 15"; sleep 15
    [ "$n" -ge 300 ] && { echo "aborting after 300 retries"; exit 1; }
  done
fi

have=$(ls "$PD"/snap_part_* 2>/dev/null | wc -l)
[ "$have" -eq 58 ] || { echo "ERROR: only $have/58 parts present"; exit 1; }

echo "[$(date +%T)] all 58 parts present; reassembling 24GB..."
cat "$PD"/snap_part_* > "$SNAP"
sz=$(stat -c%s "$SNAP"); md=$(md5sum "$SNAP" | cut -d' ' -f1)
echo "[$(date +%T)] size=$sz (want $SIZE) ; md5=$md"
[ "$md" = "$SRC_MD5" ] || { echo "ERROR: md5 mismatch (want $SRC_MD5) — NOT restoring"; exit 1; }
echo "[$(date +%T)] snapshot INTACT ✓"

echo "[$(date +%T)] restoring into local Qdrant as 'georgian_legal' (local, a few min)..."
curl -s -X POST "http://localhost:6333/collections/georgian_legal/snapshots/upload?priority=snapshot" \
  -F "snapshot=@$SNAP"; echo

echo "[$(date +%T)] verifying..."
curl -s http://localhost:6333/collections/georgian_legal \
  | python3 -c "import sys,json;d=json.load(sys.stdin)['result'];print('LOADED: points_count',d['points_count'],'status',d['status'])" \
  2>/dev/null || { echo "verify failed — inspect the curl output above"; exit 1; }

echo "[$(date +%T)] DONE ✓  If points_count is ~2453915, the corpus is loaded and permanent."
echo "Next: stop the pod billing with:"
echo "   cd \"$INGEST_DIR\" && .venv/bin/python scripts/runpod_orchestrate_multi.py terminate"
