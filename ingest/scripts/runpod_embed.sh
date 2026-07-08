#!/usr/bin/env bash
# Self-contained on-pod BGE-M3 full-corpus embed (RunPod Secure Cloud GPU, CUDA FP16).
#
# Runs INSIDE a tmux session on the pod. Expects, under $WORK (default /workspace):
#   payload.tar.gz.enc   — encrypted tarball of {ingest/ package, snapshots/v1/docs/*.jsonl}
#   PASSPHRASE env var    — symmetric key to decrypt (never written to disk/logs)
#
# Produces (for transfer back):
#   $WORK/out/georgian_legal.snapshot     — Qdrant collection snapshot (single file)
#   $WORK/out/checksum_gpu.json           — GPU vector-space checksum (verify vs CPU ref)
#   $WORK/out/embed.log, $WORK/out/DONE   — log + success marker
#
# Guardrails: FP16 CUDA (corpus side of "one vector space"); resumable per-source; wipe the
# volume before terminate (done by the local orchestrator after download).
set -euo pipefail

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:-georgian_legal}"
QDRANT_VER="${QDRANT_VER:-v1.12.4}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/embed.log") 2>&1
echo "[$(date -u +%FT%TZ)] pod embed start  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"

# --- 1. decrypt + unpack the corpus payload -----------------------------------
cd "$WORK"
if [ ! -d "$WORK/ingest" ]; then
  test -n "${PASSPHRASE:-}" || { echo "PASSPHRASE not set"; exit 1; }
  openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PASSPHRASE -in payload.tar.gz.enc \
    | tar xz
fi

# --- 2. python deps (torch ships in the RunPod PyTorch image) ------------------
python -m pip install -q --no-input "FlagEmbedding>=1.2.10" "qdrant-client>=1.12" \
  "python-dotenv>=1.0" "tqdm>=4.66" "rich>=13" "transformers" "tokenizers" || true
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
  print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# --- 3. Qdrant as a local static binary (no docker needed on the pod) ----------
if ! curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1; then
  cd "$WORK"
  curl -sSL -o qdrant.tar.gz \
    "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  tar xzf qdrant.tar.gz
  ( QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant_storage" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant_snapshots" \
    ./qdrant >"$OUT/qdrant.log" 2>&1 & )
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1 && break; sleep 1; done
fi

# --- 4. embed the whole clean snapshot, GPU FP16, resumable --------------------
cd "$WORK/ingest"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
python -m ingest embed --checksum >"$OUT/checksum_stdout.txt" 2>&1 || true
python -m ingest embed --source all --batch-size 512   # resumes if re-run

# --- 5. persist GPU checksum + Qdrant snapshot for transfer back ---------------
python - <<'PY'
import json
from ingest.config import load_config
from ingest.embedding import BGEM3Embedder
from ingest import embed_job
cfg = load_config()
sha, vec = embed_job.dense_checksum(BGEM3Embedder(cfg))
json.dump({"sha": sha, "sentence": embed_job.CHECKSUM_SENTENCE, "dense": vec},
          open("/workspace/out/checksum_gpu.json", "w"))
print("GPU checksum sha", sha)
PY
SNAP=$(curl -sf -X POST "http://127.0.0.1:6333/collections/${COLLECTION}/snapshots" \
  | python -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
cp "$WORK/qdrant_snapshots/${COLLECTION}/${SNAP}" "$OUT/${COLLECTION}.snapshot"
POINTS=$(curl -sf "http://127.0.0.1:6333/collections/${COLLECTION}" \
  | python -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
echo "[$(date -u +%FT%TZ)] embed DONE  points=$POINTS  snapshot=$SNAP"
echo "$POINTS" > "$OUT/DONE"
