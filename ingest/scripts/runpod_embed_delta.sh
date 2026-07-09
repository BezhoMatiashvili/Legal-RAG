#!/usr/bin/env bash
# Self-contained on-pod BGE-M3 DELTA embed (RunPod GPU, CUDA FP16) — the incremental sibling
# of runpod_embed.sh. Embeds only the sweep's newly-scraped matsne docs (shipped as
# delta_items/*.jsonl inside the encrypted payload) into a small `georgian_legal_delta`
# collection and snapshots it for transfer back (~8× smaller than the full-corpus snapshot).
#
# Expects under $WORK (default /workspace):
#   payload.tar.gz.enc  — encrypted tarball of {ingest/ package+scripts, delta_items/*.jsonl}
#   PASSPHRASE env var  — symmetric key to decrypt (only needed if not yet unpacked)
#
# Produces (for transfer back):
#   $WORK/out/${COLLECTION}.snapshot   — Qdrant snapshot of the delta collection
#   $WORK/out/checksum_gpu.json        — GPU vector-space checksum (G2 gate vs CPU ref)
#   $WORK/out/embed.log, $WORK/out/DONE
#
# Idempotent: skips untar/venv/qdrant if present; embed upserts are deterministic-id.
set -euo pipefail

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:-georgian_legal_delta}"
QDRANT_VER="${QDRANT_VER:-v1.18.2}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/embed.log") 2>&1
echo "[$(date -u +%FT%TZ)] pod DELTA embed start  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"

# --- 1. decrypt + unpack the delta payload -------------------------------------
cd "$WORK"
if [ ! -d "$WORK/ingest" ]; then
  test -n "${PASSPHRASE:-}" || { echo "PASSPHRASE not set"; exit 1; }
  openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PASSPHRASE -in payload.tar.gz.enc \
    | tar xz --no-same-owner --no-same-permissions
fi
ls "$WORK"/delta_items/*.jsonl >/dev/null 2>&1 || { echo "no delta_items/*.jsonl in payload"; exit 1; }

# --- 2. python deps in a CLEAN venv (same pinned stack as the full embed = G2-safe) ---
PY="$WORK/venv/bin/python"
if [ ! -x "$PY" ]; then python -m venv "$WORK/venv"; fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q "FlagEmbedding==1.4.0" "transformers==5.12.1" "tokenizers==0.22.2" \
  "accelerate==1.14.0" "datasets==5.0.0" "sentencepiece==0.2.1" "safetensors==0.8.0" \
  "qdrant-client>=1.12" "python-dotenv>=1.0" "tqdm>=4.66" "rich>=13"
"$PY" -c "import torch, transformers; from FlagEmbedding import BGEM3FlagModel; \
  assert torch.cuda.is_available(), 'no CUDA'; \
  print('deps OK: torch', torch.__version__, 'cuda', torch.version.cuda, \
  torch.cuda.get_device_name(0), 'transformers', transformers.__version__)"

# --- 3. Qdrant as a local static binary (no docker on the pod) -----------------
if ! curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1; then
  cd "$WORK"
  curl -sSL -o qdrant.tar.gz \
    "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  tar xzf qdrant.tar.gz --no-same-owner --no-same-permissions
  ( QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant_storage" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant_snapshots" \
    QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=1024 \
    ./qdrant >"$OUT/qdrant.log" 2>&1 & )
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1 && break; sleep 1; done
fi

# --- 4. embed the delta items, GPU FP16 -----------------------------------------
cd "$WORK/ingest"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
"$PY" scripts/embed_delta.py --items "$WORK"/delta_items/*.jsonl \
  --collection "$COLLECTION" --batch-size 256

# --- 5. persist GPU checksum + Qdrant snapshot for transfer back ---------------
"$PY" - <<'PYEOF'
import json
from ingest.config import load_config
from ingest.embedding import BGEM3Embedder
from ingest import embed_job
cfg = load_config()
sha, vec = embed_job.dense_checksum(BGEM3Embedder(cfg))
json.dump({"sha": sha, "sentence": embed_job.CHECKSUM_SENTENCE, "dense": vec},
          open("/workspace/out/checksum_gpu.json", "w"))
print("GPU checksum sha", sha)
PYEOF
SNAP=$(curl -sf -X POST "http://127.0.0.1:6333/collections/${COLLECTION}/snapshots" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
cp "$WORK/qdrant_snapshots/${COLLECTION}/${SNAP}" "$OUT/${COLLECTION}.snapshot"
POINTS=$(curl -sf "http://127.0.0.1:6333/collections/${COLLECTION}" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
echo "[$(date -u +%FT%TZ)] delta embed DONE  points=$POINTS  snapshot=$SNAP"
echo "$POINTS" > "$OUT/DONE"
