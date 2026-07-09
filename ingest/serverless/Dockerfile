# RunPod Serverless "legal-search worker": Qdrant (static binary, storage on the network
# volume) + BGE-M3 + bge-reranker-v2-m3 + an op-dispatch queue handler that reuses the
# MCP tool coroutines from ingest/ingest/mcp_server.py unchanged.
#
# Build from the ingest/ directory (the build context must contain ingest/ and serverless/):
#   cd ingest
#   docker build -f serverless/Dockerfile -t <dockerhub-user>/legal-search-worker:v1 .
#   docker push <dockerhub-user>/legal-search-worker:v1
#
# The image bakes code + binaries only. Data (qdrant storage, published snapshots) and the
# HF model cache live on the network volume mounted at /runpod-volume. Secrets never enter
# the image (ingest/.dockerignore excludes .env; the queue API is authed by RunPod itself).
FROM python:3.11-slim-bookworm

# Must match the LOCAL Qdrant (ingest/docker-compose.yml) — snapshots are only guaranteed
# compatible within the same minor version.
ARG QDRANT_VER=v1.18.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# Same static-musl-binary mechanism scripts/runpod_embed_delta.sh already uses on pods.
RUN mkdir -p /opt/qdrant \
    && curl -sSL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz" \
       | tar xz -C /opt/qdrant --no-same-owner --no-same-permissions \
    && /opt/qdrant/qdrant --version

# Deps first (the big, rarely-changing layer), code after (the small, hot layer).
COPY serverless/requirements.txt /app/serverless/requirements.txt
RUN pip install --no-cache-dir -r /app/serverless/requirements.txt

COPY ingest/ /app/ingest/
COPY serverless/ /app/serverless/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app
CMD ["python", "-u", "/app/serverless/handler.py"]
