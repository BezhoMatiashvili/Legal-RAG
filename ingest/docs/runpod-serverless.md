# RunPod Serverless hosting runbook (legal-search worker)

The whole read path (Qdrant + BGE-M3 + reranker) runs inside ONE RunPod Serverless GPU
endpoint that scales to zero — $0 compute when idle, per-second billing while a request
runs. The local MCP server becomes a thin client (`SEARCH_BACKEND=remote`); the local
Qdrant stays the write master for scraping/ingest and publishes data via
`scripts/publish_snapshot.py`. Code: `ingest/serverless/` (worker),
`ingest/ingest/remote_search.py` (client).

Provisioning is deliberately console clicks, not code: it's a one-time action and the
serverless GraphQL mutations are under-documented and Cloudflare-fronted.

## 1. One-time provisioning (console)

1. **Network volume** — Storage → New network volume:
   - Datacenter: **EU-CZ-1** (has the S3-compatible gateway and is close; fallback: US-KS-2).
   - Size: **70 GB** (35 GB restored storage + ~24 GB snapshot + ~5 GB HF model cache
     + headroom) ≈ **$4.90/mo** at $0.07/GB/mo. Volumes can grow but not shrink.
   - Note the volume ID — it is also the S3 **bucket name**.
2. **S3 API key** — Settings → S3 API Keys → create. Record access + secret key.
3. **Build & push the image** (Docker Hub free account):
   ```bash
   cd ingest
   docker build -f serverless/Dockerfile -t <user>/legal-search-worker:v1 .
   docker push <user>/legal-search-worker:v1
   ```
4. **Serverless endpoint** — Serverless → New Endpoint:
   - Image: `<user>/legal-search-worker:v1`.
   - GPUs: 16 GB tier (A4000 / RTX 4000 Ada / RTX 2000 Ada, $0.58/hr) with the 24 GB tier
     (L4 / A5000 / 3090, $0.69/hr) as fallback. If the probe (below) shows < ~14 GB system
     RAM on the 16 GB tier, restrict to the 24 GB tier.
   - **Max workers: 1** — HARD requirement. Qdrant holds an exclusive lock on its storage
     dir; a second worker on the same volume cannot serve and risks fighting the lock.
   - **Execution timeout: 1800 s** (the first boot after a publish sha-checks + restores
     ~24 GB — 10–20 min). Idle timeout: 60 s. FlashBoot: on.
   - Attach the network volume (mounts at `/runpod-volume`).
   - Container disk: ≥ 20 GB (the image is ~8 GB unpacked).
   - Env vars (no secrets needed — Qdrant is loopback-only inside the worker, and the
     queue API is authenticated by RunPod itself):
     ```
     COLLECTION_NAME=georgian_legal
     EMBED_DEVICE=cuda          EMBED_USE_FP16=true    EMBED_BATCH_SIZE=16
     RERANK_ENABLED=true        RERANK_DEVICE=cuda     RERANK_USE_FP16=true
     RERANK_CANDIDATES=80       RERANK_MIN_SCORE=0.3
     QUERY_LOG_ENABLED=false    HF_HOME=/runpod-volume/hf
     ```
     Keep `RERANK_CANDIDATES`/`RERANK_MIN_SCORE` in lockstep with ingest/.env — a drift
     silently changes scores (compare via `legal_collection_info`, which reports them).
5. **ingest/.env additions** (names only — values stay out of git):
   ```
   RUNPOD_ENDPOINT_ID=...      # from the endpoint page
   RUNPOD_VOLUME_ID=...        # the volume id = S3 bucket
   RUNPOD_S3_ENDPOINT=https://s3api-eu-cz-1.runpod.io
   RUNPOD_S3_REGION=eu-cz-1
   RUNPOD_S3_ACCESS_KEY=...    RUNPOD_S3_SECRET_KEY=...
   # SEARCH_BACKEND=remote     # flip AFTER --verify passes
   ```
   (`RUNPOD_API_KEY` is already there.)

## 2. Probe before committing the 24 GB upload (~$0.05)

The pattern (Qdrant co-process on serverless + network volume) is undocumented, and RunPod
doesn't publish system RAM per GPU tier. Verify for pennies, before uploading anything:

```bash
cd ingest && uv run python - <<'EOF'
from ingest.config import load_config
from ingest.remote_search import RunPodQueueClient
import json
cfg = load_config()
out = RunPodQueueClient(cfg.runpod_endpoint_id, cfg.runpod_api_key, timeout=900).call("health")
print(json.dumps(out, indent=2))
EOF
```

This cold-starts a worker against the empty volume (models download once into
`/runpod-volume/hf`, ~4.5 GB). Check in `sysinfo`: `ram_total_gb` **≥ ~14** (else restrict
the endpoint to the 24 GB tier), `volume_free_gb` sane, GPU present. `health.ok` will be
false (no collection yet) — expected.

## 3. Seeding / publishing data

```bash
cd ingest
uv run --group publish python scripts/publish_snapshot.py --create   # ~5–20 min, 24 GB local
uv run --group publish python scripts/publish_snapshot.py --upload   # hours; resumable — rerun anytime
uv run --group publish python scripts/publish_snapshot.py --verify   # wakes worker; restore 10–20 min
uv run --group publish python scripts/publish_snapshot.py --cleanup  # reclaim local disk
```

- The upload prints observed Mbps after the first parts — abort early if the projection is
  unacceptable and rerun overnight (it resumes from the last completed 256 MB part).
- `manifest.json` is uploaded **last**: it is the atomic publish signal the worker's
  restore keys on. Never place a manifest for a snapshot that isn't fully uploaded.
- `--verify` uses the worker's `refresh` op, so it *applies* the publish even on a warm
  worker (which never re-runs its boot-time restore check), then asserts restore status +
  point parity. Its budget is `PUBLISH_VERIFY_TIMEOUT` (default 1800 s) — deliberately not
  the day-to-day `RUNPOD_API_TIMEOUT`.
- Belt and braces: the worker also re-checks the manifest cheaply before *every* job, so a
  publish is picked up even if `--verify` is never run.
- A failed restore does not take the worker down — it keeps serving the previous data but
  prepends a loud `WARNING: worker data may be STALE` line to every response, and a failed
  *boot* makes every op return an error instead of crash-looping (retry with the `refresh`
  op after fixing the cause).
- Republishing later: same commands.
- Steady state should move to small delta publishes (the 2.4 GB delta collection) instead
  of 24 GB fulls — not implemented yet; see plan notes.

## 4. Flip the MCP server to remote

```
ingest/.env:  SEARCH_BACKEND=remote
```
Reconnect via `/mcp`. Then smoke test: `legal_health` (endpoint health, no wake) →
`legal_collection_info` → one Georgian + one English `legal_search`; results should match
local mode exactly (same tool code runs worker-side). The local server now loads **no**
models and needs no local Qdrant — `docker compose stop` it day-to-day if you want the RAM;
start it again for ingest/publish.

Flip back anytime: `SEARCH_BACKEND=local` + `/mcp` reconnect. No silent auto-fallback —
remote errors tell the agent to retry (cold start) or suggest the env flip.

## 5. Costs & spend model

- Volume: ~$4.90/mo (70 GB × $0.07). The only recurring cost while idle.
- GPU: per-second while a worker is up: cold start 1–3 min + 60 s idle tail per burst,
  ~$0.58–0.69/hr → casual daily use ≈ $1–3/mo.
- Watch balance + endpoint state on the session monitor (`scripts/session_monitor.py`,
  http://localhost:8770) — the "Serverless GPU" tile never wakes a worker.

## 6. Teardown / emergency stop

Serverless has no orphaned-pod risk (nothing bills at idle). Three levels:
1. **Pause**: endpoint → max workers 0. $0 compute, config preserved.
2. **Delete endpoint**: console → endpoint → delete. Remote mode starts returning the
   actionable error; flip `SEARCH_BACKEND=local` to keep working.
3. **Delete the volume**: console → Storage. Stops the $4.90/mo. Safe: the local Qdrant is
   authoritative and `publish_snapshot.py --create` reproduces the snapshot.

Key rotation: revoke the S3 key (Settings → S3 API Keys) and/or the restricted
`RUNPOD_API_KEY`; update ingest/.env.
