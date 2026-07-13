#!/usr/bin/env bash
# Self-contained on-pod BGE-M3 full-corpus embed (RunPod Secure Cloud GPU, CUDA FP16).
#
# Runs detached (setsid) on the pod. Expects, under $WORK (default /workspace):
#   payload.tar.gz.enc   — encrypted tarball of {ingest/ package, snapshots/v1/docs/*.jsonl}
#   PASSPHRASE env var    — symmetric key to decrypt (only needed if ingest/ not yet unpacked)
#
# Produces (for transfer back):
#   $WORK/out/georgian_legal.snapshot     — Qdrant collection snapshot (single file)
#   $WORK/out/checksum_gpu.json           — GPU vector-space checksum (verify vs CPU ref)
#   $WORK/out/embed.log, $WORK/out/DONE   — log + success marker
#
# Guardrails: FP16 CUDA; resumable per-source. Idempotent: safe to re-run (skips the untar if
# ingest/ exists, skips the venv build if present, resumes the embed from per-source checkpoints).
set -euo pipefail

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:-georgian_legal}"
QDRANT_VER="${QDRANT_VER:-v1.12.4}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/embed.log") 2>&1
echo "[$(date -u +%FT%TZ)] pod embed start  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"

# --- 1. decrypt + unpack the corpus payload -----------------------------------
# --no-same-owner/-permissions: the RunPod volume rejects chown (uid 1000), which would make
# tar exit non-zero and abort under `set -e`; we only need the file content.
cd "$WORK"
if [ ! -d "$WORK/ingest" ]; then
  test -n "${PASSPHRASE:-}" || { echo "PASSPHRASE not set"; exit 1; }
  openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PASSPHRASE -in payload.tar.gz.enc \
    | tar xz --no-same-owner --no-same-permissions
fi

# --- 2. python deps in a CLEAN venv --------------------------------------------
# The RunPod image ships pre-installed ML packages (peft, reporting integrations, ...) whose
# versions break transformers 5.12.1's lazy imports. A fresh venv with ONLY the local-matched
# stack (NO peft) avoids all of it. torch cu124 (>=2.6) is required by transformers 5.x for
# torch.load (CVE-2025-32434). Matching the local versions keeps GPU vectors == the CPU ref (G2).
PY="$WORK/venv/bin/python"
if [ ! -x "$PY" ]; then python -m venv "$WORK/venv"; fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q "FlagEmbedding==1.4.0" "transformers==5.12.1" "tokenizers==0.22.2" \
  "accelerate==1.14.0" "datasets==5.0.0" "sentencepiece==0.2.1" "safetensors==0.8.0" \
  "qdrant-client>=1.12" "python-dotenv>=1.0" "tqdm>=4.66" "rich>=13"
# Fail fast if the stack can't import BEFORE the long embed.
"$PY" -c "import torch, transformers; from FlagEmbedding import BGEM3FlagModel; \
  assert torch.cuda.is_available(), 'no CUDA'; \
  print('deps OK: torch', torch.__version__, 'cuda', torch.version.cuda, \
  torch.cuda.get_device_name(0), 'transformers', transformers.__version__)"

# --- 3. Qdrant as a local static binary (no docker needed on the pod) ----------
if ! curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1; then
  cd "$WORK"
  curl -sSL -o qdrant.tar.gz \
    "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  tar xzf qdrant.tar.gz --no-same-owner --no-same-permissions
  # Raise the JSON request-size ceiling well above a full upsert batch (default 32MB is too
  # small: a 256-point batch of 1024-d dense + sparse + Georgian text exceeds it).
  ( QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant_storage" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant_snapshots" \
    QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=1024 \
    ./qdrant >"$OUT/qdrant.log" 2>&1 & )
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1 && break; sleep 1; done
fi

# --- 4. embed the whole clean snapshot, GPU FP16, resumable --------------------
cd "$WORK/ingest"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
"$PY" -m ingest embed --checksum >"$OUT/checksum_stdout.txt" 2>&1 || true
"$PY" -m ingest embed --source all --batch-size 256   # resumes if re-run (smaller upsert batch)

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
echo "[$(date -u +%FT%TZ)] embed DONE  points=$POINTS  snapshot=$SNAP"
echo "$POINTS" > "$OUT/DONE"
