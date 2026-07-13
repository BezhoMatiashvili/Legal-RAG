#!/usr/bin/env python3
"""RunPod Serverless queue handler for the Georgian legal RAG ("legal-search worker").

One worker = the whole read path: a local Qdrant (restored onto ephemeral container disk), BGE-M3
and the cross-encoder reranker on GPU, and an op dispatcher that calls the MCP tool
coroutines from ``ingest.mcp_server`` directly. ``FastMCP.tool()`` returns the original
function, so ``legal_search`` etc. are plain awaitables here and every op returns the
same formatted string the local MCP server would produce — the local server in
``SEARCH_BACKEND=remote`` mode just passes it through.

Request shape (queue API ``input``):   {"op": "search", "params": {...}}
Response shape (job ``output``):       {"result": "<tool output string>"}   or {"error": ...}

Ops: search | get_document | lookup | browse | versions | collection_info | health | refresh.
``ingest_status`` is intentionally absent — watcher state lives on the local box only.

Boot is FAIL-SOFT: an exception during qdrant boot/restore must not exit the process —
RunPod would restart the container in a billed crash-loop. Instead the error is stored and
every op reports it; the ``refresh`` op retries the boot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("serverless.handler")

# The worker IS the backend: never let a stray endpoint env var route it to itself, and
# never write query logs off-box (hard assign — privacy guardrail, not a default).
os.environ["SEARCH_BACKEND"] = "local"
os.environ["QUERY_LOG_ENABLED"] = "false"
os.environ.setdefault("HF_HOME", "/runpod-volume/hf")

import qdrant_boot  # noqa: E402  (sibling module; script dir is on sys.path)

_BOOT_ERROR: str | None = None
_RESTORE: dict = {"status": "not_checked"}


def _boot() -> None:
    """Cold-start boot: qdrant up + apply any pending publish. Sets globals, never raises."""
    global _BOOT_ERROR, _RESTORE
    try:
        qdrant_boot.ensure_running()
        _RESTORE = qdrant_boot.maybe_restore()
        logger.info("boot restore check: %s", _RESTORE)
        _BOOT_ERROR = None
    except Exception as e:  # noqa: BLE001 - see module docstring: no billed crash-loops
        _BOOT_ERROR = f"{type(e).__name__}: {e}"
        logger.error("worker boot failed: %s\n%s", _BOOT_ERROR, traceback.format_exc())


_boot()

from ingest import mcp_server as srv  # noqa: E402  (import is side-effect-free)

OPS: dict[str, tuple] = {
    "search": (srv.legal_search, srv.SearchInput),
    "get_document": (srv.legal_get_document, srv.GetDocumentInput),
    "lookup": (srv.legal_lookup, srv.LookupInput),
    "browse": (srv.legal_browse, srv.BrowseInput),
    "versions": (srv.legal_get_document_versions, srv.GetVersionsInput),
    "collection_info": (srv.legal_collection_info, None),
    "health": (srv.legal_health, None),
}


def _sysinfo() -> dict:
    """RAM/CPU/disk/GPU facts for the probe step — the 16GB-VRAM tier's system RAM is
    undocumented, and Qdrant + both models need ~14GB; this makes the fit measurable."""
    info: dict = {"cpus": os.cpu_count()}
    try:
        mem = {}
        for ln in open("/proc/meminfo"):
            k, _, v = ln.partition(":")
            mem[k.strip()] = int(v.strip().split()[0])
        info["ram_total_gb"] = round(mem.get("MemTotal", 0) / 1024 / 1024, 1)
        info["ram_available_gb"] = round(mem.get("MemAvailable", 0) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        du = shutil.disk_usage(qdrant_boot.VOLUME_ROOT)
        info["volume_total_gb"] = round(du.total / 1e9, 1)
        info["volume_free_gb"] = round(du.free / 1e9, 1)
    except OSError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:  # noqa: BLE001 - sysinfo must never fail an op
        pass
    return info


async def _warmup() -> None:
    """Load both models eagerly. Init time is billed either way, and FlashBoot then
    snapshots a worker with everything resident, so warm revivals answer instantly and
    the first search never pays the model load inside its own request."""
    await srv._get_embedder()
    await srv._get_reranker()


async def handler(job: dict) -> dict:
    global _RESTORE
    inp = job.get("input") or {}
    op = inp.get("op") or "health"
    params = inp.get("params") or {}
    try:
        if op == "refresh":
            # Retry a failed boot, else re-check the published manifest — both without
            # waiting for the next cold start (FlashBoot revivals skip module import).
            if _BOOT_ERROR is not None:
                await asyncio.to_thread(_boot)
                if _BOOT_ERROR is not None:
                    return {"error": f"worker boot failed: {_BOOT_ERROR}"}
            else:
                _RESTORE = await asyncio.to_thread(
                    qdrant_boot.maybe_restore, bool(params.get("force")))
            if _RESTORE.get("status") == "restored":
                srv._result_cache.clear()
            return {"result": json.dumps(_RESTORE, ensure_ascii=False), "restore": _RESTORE}

        if _BOOT_ERROR is not None:
            return {"error": f"worker boot failed: {_BOOT_ERROR}. "
                             "Send op 'refresh' to retry the boot."}

        # A publish that landed after module import is caught by a cheap two-file check.
        # maybe_restore refuses destructive warm replacement, so this records a visible
        # stale-data warning until an empty cold worker (or blue/green promotion) is used.
        if await asyncio.to_thread(qdrant_boot.restore_pending):
            logger.info("new publish detected on a warm worker; checking safe restore policy")
            _RESTORE = await asyncio.to_thread(qdrant_boot.maybe_restore)
            if _RESTORE.get("status") == "restored":
                srv._result_cache.clear()

        entry = OPS.get(op)
        if entry is None:
            return {"error": f"unknown op {op!r}; expected one of "
                             f"{sorted([*OPS, 'refresh'])}"}
        fn, model = entry
        result = await fn(model(**params)) if model is not None else await fn()
        if _RESTORE.get("status") == "error":
            # Serving beats an outage, but stale data must be visible to the caller.
            result = ("> WARNING: worker data may be STALE — the last publish failed to "
                      f"restore: {_RESTORE.get('detail')}\n\n") + result
        out = {"result": result}
        if op == "health":
            out["sysinfo"] = _sysinfo()
            out["restore"] = _RESTORE
        return out
    except Exception as e:  # noqa: BLE001 - the job output is the only error channel
        tb = traceback.format_exc().splitlines()[-12:]
        logger.error("op %s failed: %s\n%s", op, e, "\n".join(tb))
        return {"error": f"{type(e).__name__}: {e}", "traceback": tb}


def main() -> None:
    if os.getenv("EAGER_LOAD", "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            asyncio.run(_warmup())
            logger.info("models warm; sysinfo=%s", _sysinfo())
        except Exception as e:  # noqa: BLE001 - lazy load retries per search; don't crash-loop
            logger.error("eager model load failed (will retry lazily): %s", e)

    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
