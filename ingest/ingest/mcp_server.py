#!/usr/bin/env python3
"""MCP server exposing the Georgian legal Qdrant corpus as RAG tools (stdio).

A thin wrapper over the ``ingest`` package: it reuses ``load_config()``,
``make_client()``, ``BGEM3Embedder`` and ``hybrid_search()`` — no embedding or
search logic is reimplemented here. The heavy BGE-M3 model (~2GB) is loaded
lazily on the first search, so the server handshakes instantly and the tools
that don't need embeddings (``legal_get_document``, ``legal_collection_info``)
never trigger a model load.

Run it inside the ingest uv environment:

    uv run --directory ingest python -m ingest.mcp_server
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models

from . import qdrant_store as store
from .config import Config, load_config
from .search import build_filter, hybrid_search
from .sources import SOURCES

mcp = FastMCP("legal_rag")

SNIPPET_CHARS = 700  # markdown preview length per hit (enough to judge legal relevance)

# --- Lazy runtime singleton ---------------------------------------------------
# Config + Qdrant client are cheap; the embedder and reranker are not, so they are
# built off the event loop on first use and guarded against a concurrent double-load.

_cfg: Config | None = None
_client = None
_embedder = None
_embedder_lock = asyncio.Lock()
_reranker = None
_reranker_lock = asyncio.Lock()


def _get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def _get_client():
    global _client
    if _client is None:
        _client = store.make_client(_get_cfg())
    return _client


async def _get_embedder():
    """Load BGE-M3 once, off the event loop (the import + model load is blocking)."""
    global _embedder
    if _embedder is None:
        async with _embedder_lock:
            if _embedder is None:
                from .embedding import BGEM3Embedder

                _embedder = await asyncio.to_thread(BGEM3Embedder, _get_cfg())
    return _embedder


async def _get_reranker():
    """Load the cross-encoder reranker once, off the event loop. None if disabled."""
    global _reranker
    if not _get_cfg().rerank_enabled:
        return None
    if _reranker is None:
        async with _reranker_lock:
            if _reranker is None:
                from .rerank import BGEReranker

                _reranker = await asyncio.to_thread(BGEReranker, _get_cfg())
    return _reranker


def _handle_error(e: Exception) -> str:
    """Actionable error message — almost always a Qdrant connectivity / config issue."""
    return (
        f"Error: {type(e).__name__}: {e}\n\n"
        "Check that Qdrant is running (cd ingest && docker compose up -d) and that "
        "QDRANT_URL / COLLECTION_NAME in ingest/.env are correct."
    )


# --- Formatting helpers -------------------------------------------------------


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


def _hit_dict(hit) -> dict:
    """Full payload of a search hit, plus its fusion score, for JSON output."""
    p = hit.payload or {}
    return {
        "score": round(hit.score, 4),
        "source": p.get("source"),
        "document_id": p.get("document_id"),
        "document_number": p.get("document_number"),
        "registration_code": p.get("registration_code"),
        "document_type": p.get("document_type"),
        "status": p.get("status"),
        "title": p.get("title"),
        "parties": p.get("parties"),
        "date": p.get("date_raw"),
        "language": p.get("language"),
        "court": p.get("court"),
        "source_url": p.get("source_url"),
        "chunk_index": p.get("chunk_index"),
        "heading": p.get("heading"),
        "text": p.get("text"),
    }


def _format_hit_md(rank: int, hit) -> str:
    """Readable markdown block for one hit (mirrors the ingest CLI's search output)."""
    p = hit.payload or {}
    snippet = " ".join((p.get("text") or "").split())[:SNIPPET_CHARS]
    status = p.get("status")
    lines = [
        f"### {rank}. {p.get('title') or '(untitled)'}",
        f"- score: {hit.score:.4f} · {p.get('source')}/{p.get('document_type') or '—'} "
        f"· {p.get('date_raw') or '—'}" + (f" · status: {status}" if status else ""),
        f"- document_id: `{p.get('document_id')}` · chunk #{p.get('chunk_index')}",
    ]
    if p.get("document_number"):
        lines.append(f"- number: {p.get('document_number')}")
    if p.get("parties"):
        lines.append(f"- parties: {p.get('parties')}")
    if p.get("source_url"):
        lines.append(f"- url: {p.get('source_url')}")
    if p.get("heading"):
        lines.append(f"- section: {p.get('heading')}")
    lines.append(f"\n{snippet}\n")
    return "\n".join(lines)


# --- Tools --------------------------------------------------------------------


class SearchInput(BaseModel):
    """Input for hybrid retrieval over the legal corpus."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Natural-language search query (Georgian or English).",
        min_length=1,
        max_length=2000,
    )
    top_k: int = Field(10, description="Number of chunks to return.", ge=1, le=50)
    source: str | None = Field(
        None,
        description="Restrict to one source key, e.g. 'matsne', 'ecd', 'constcourt', "
        "'napr', 'tbappeal', 'supremecourt', 'tas'.",
    )
    language: str | None = Field(
        None, description="Filter by language code stored in the payload, e.g. 'ka'."
    )
    document_type: str | None = Field(
        None, description="Filter by the document_type payload value."
    )
    court: str | None = Field(
        None,
        description="Exact court key, e.g. 'supremecourt', 'constcourt', 'tbappeal', or "
        "the ecd court name.",
    )
    status: str | None = Field(
        None,
        description="Legal status (matsne acts only): 'in_force', 'repealed', or "
        "'pending'. Use 'in_force' to restrict to law currently in effect.",
    )
    document_number: str | None = Field(
        None,
        description="Exact official document number (e.g. matsne '55', supremecourt "
        "'ბს-174(კს-26)'). Exact match, source-dependent meaning.",
    )
    registration_code: str | None = Field(
        None,
        description="Exact registry code (unique), e.g. matsne "
        "'140130000.22.034.017712'.",
    )
    parties: str | None = Field(
        None,
        description="Full-text match against party/person names (mainly constcourt "
        "litigants and napr senders).",
    )
    contains: str | None = Field(
        None,
        description="Full-text keyword/phrase that must appear in the document text "
        "(exact lexical match, narrows the semantic results).",
    )
    date_from: str | None = Field(
        None, description="Earliest document date, inclusive, as YYYY-MM-DD."
    )
    date_to: str | None = Field(
        None, description="Latest document date, inclusive, as YYYY-MM-DD."
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.MARKDOWN,
        description="'markdown' for readable results, 'json' for full payloads.",
    )


@mcp.tool(
    name="legal_search",
    annotations={
        "title": "Search Georgian Legal Corpus",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def legal_search(params: SearchInput) -> str:
    """Hybrid (dense + sparse) semantic search with cross-encoder reranking over the corpus.

    This is the primary RAG retrieval tool. It embeds the query with BGE-M3, fuses dense
    and sparse matches server-side (RRF) in Qdrant to build a candidate pool, then reranks
    that pool with a cross-encoder (bge-reranker-v2-m3) and drops low-relevance hits, so the
    returned chunks are ordered by true query↔text relevance. The ``score`` field is the
    reranker's calibrated relevance (0..1), not a cosine/RRF score.

    Use the optional filters to scope by source, court, legal ``status`` (in_force /
    repealed / pending — matsne acts), language, document type, document/registration
    number, party names, an exact keyword that must appear (``contains``), or a date range.
    For exact identifier lookups use ``legal_lookup``; for non-semantic browsing/filtering
    use ``legal_browse``. To read a full document after a chunk looks relevant, pass its
    ``source`` + ``document_id`` to ``legal_get_document``.

    Args:
        params (SearchInput): query, top_k (1-50), optional filters (source, court,
            status, language, document_type, document_number, registration_code, parties,
            contains, date_from, date_to), and response_format.

    Returns:
        str: Markdown list of ranked hits (title, score, source/type/status/date,
        document_id, chunk index, url, section, text snippet), or — when
        response_format='json' — a JSON object:
        {"count": int, "hits": [{"score", "source", "document_id", "status",
        "document_type", "title", "date", "language", "court", "source_url",
        "chunk_index", "heading", "text"}, ...]}.
        Returns a "No results found" message when nothing matches.
    """
    try:
        cfg = _get_cfg()
        client = _get_client()
        embedder = await _get_embedder()
        reranker = await _get_reranker()
        hits = await asyncio.to_thread(
            hybrid_search,
            cfg,
            client,
            embedder,
            params.query,
            top_k=params.top_k,
            reranker=reranker,
            rerank_candidates=cfg.rerank_candidates,
            rerank_min_score=cfg.rerank_min_score,
            source=params.source,
            court=params.court,
            status=params.status,
            language=params.language,
            document_type=params.document_type,
            document_number=params.document_number,
            registration_code=params.registration_code,
            parties=params.parties,
            contains=params.contains,
            date_from=params.date_from,
            date_to=params.date_to,
        )
    except Exception as e:  # noqa: BLE001 - surface an actionable message to the agent
        return _handle_error(e)

    if not hits:
        return "No results found. Try a broader query or remove filters."

    if params.response_format is ResponseFormat.JSON:
        payload = {"count": len(hits), "hits": [_hit_dict(h) for h in hits]}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    blocks = [f"# {len(hits)} results for: {params.query}", ""]
    blocks.extend(_format_hit_md(rank, hit) for rank, hit in enumerate(hits, 1))
    return "\n".join(blocks)


class GetDocumentInput(BaseModel):
    """Input for reassembling a full document from its chunks."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    source: str = Field(
        ..., description="Source key, e.g. 'matsne' (as returned in a search hit).", min_length=1
    )
    document_id: str = Field(
        ..., description="The document_id from a search hit's payload.", min_length=1
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.MARKDOWN,
        description="'markdown' for readable text, 'json' for structured fields.",
    )


def _scroll_all(client, collection: str, flt: models.Filter, page: int = 256) -> list:
    """Page through every point matching ``flt`` (a document's chunks are few)."""
    out: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=page,
            offset=offset,
        )
        out.extend(batch)
        if offset is None:
            break
    return out


@mcp.tool(
    name="legal_get_document",
    annotations={
        "title": "Get Full Legal Document",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def legal_get_document(params: GetDocumentInput) -> str:
    """Reassemble a complete document from its stored chunks, ordered by chunk index.

    Use this after ``legal_search`` surfaces a relevant chunk and you need the full
    text. It fetches every chunk for the given source + document_id (no embedding /
    model load) and joins them in order. Note: BGE-M3 chunks carry ~80 tokens of
    overlap, so a little text may repeat across chunk boundaries.

    Args:
        params (GetDocumentInput): source, document_id, and response_format.

    Returns:
        str: Markdown with a metadata header (title, source, type, date,
        document_id, chunk count, url) followed by the joined body, or — when
        response_format='json' — {"source", "document_id", "title", "date",
        "document_type", "court", "language", "source_url", "chunk_count",
        "text"}. Returns a "No document found" message when the id is unknown.
    """
    try:
        cfg = _get_cfg()
        client = _get_client()
        flt = models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=params.source)),
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=params.document_id)
                ),
            ]
        )
        points = await asyncio.to_thread(_scroll_all, client, cfg.collection_name, flt)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)

    if not points:
        return (
            f"No document found for source={params.source!r} "
            f"document_id={params.document_id!r}. Check the values against a search hit."
        )

    points.sort(key=lambda pt: (pt.payload or {}).get("chunk_index", 0))
    first = points[0].payload or {}
    body = "\n\n".join((pt.payload or {}).get("text") or "" for pt in points)

    if params.response_format is ResponseFormat.JSON:
        payload = {
            "source": first.get("source"),
            "document_id": first.get("document_id"),
            "title": first.get("title"),
            "date": first.get("date_raw"),
            "document_type": first.get("document_type"),
            "court": first.get("court"),
            "language": first.get("language"),
            "source_url": first.get("source_url"),
            "chunk_count": len(points),
            "text": body,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    header = [
        f"# {first.get('title') or '(untitled)'}",
        "",
        f"- source: {first.get('source')} · type: {first.get('document_type') or '—'} "
        f"· date: {first.get('date_raw') or '—'}",
        f"- document_id: `{first.get('document_id')}` · {len(points)} chunks",
    ]
    if first.get("source_url"):
        header.append(f"- url: {first.get('source_url')}")
    header.append(
        "\n> Note: chunks carry ~80-token overlap, so some text may repeat across boundaries.\n"
    )
    return "\n".join(header) + "\n" + body


def _dedup_documents(points) -> list[dict]:
    """Collapse scrolled chunks into one record per (source, document_id).

    Picks the lowest chunk_index as the representative payload, attaches ``chunk_count``,
    and sorts newest-first by the sortable ISO ``date`` (falling back to ``date_raw``).
    """
    by_doc: dict[tuple, dict] = {}
    for pt in points:
        p = pt.payload or {}
        key = (p.get("source"), p.get("document_id"))
        idx = p.get("chunk_index", 0)
        entry = by_doc.get(key)
        if entry is None:
            by_doc[key] = {"payload": p, "min_index": idx, "chunk_count": 1}
        else:
            entry["chunk_count"] += 1
            if idx < entry["min_index"]:
                entry["min_index"] = idx
                entry["payload"] = p

    docs = []
    for entry in by_doc.values():
        p = entry["payload"]
        docs.append(
            {
                "source": p.get("source"),
                "document_id": p.get("document_id"),
                "document_number": p.get("document_number"),
                "registration_code": p.get("registration_code"),
                "document_type": p.get("document_type"),
                "title": p.get("title"),
                "parties": p.get("parties"),
                "date": p.get("date_raw"),
                "date_sort": p.get("date") or "",
                "court": p.get("court"),
                "language": p.get("language"),
                "source_url": p.get("source_url"),
                "chunk_count": entry["chunk_count"],
            }
        )
    docs.sort(key=lambda d: d["date_sort"], reverse=True)
    return docs


def _format_doc_line(rank: int, d: dict) -> str:
    """One markdown block per distinct document (no chunk text)."""
    lines = [
        f"### {rank}. {d.get('title') or '(untitled)'}",
        f"- {d.get('source')}/{d.get('document_type') or '—'} · {d.get('date') or '—'}",
        f"- document_id: `{d.get('document_id')}`"
        + (f" · number: {d['document_number']}" if d.get("document_number") else "")
        + (f" · reg: {d['registration_code']}" if d.get("registration_code") else ""),
    ]
    if d.get("parties"):
        lines.append(f"- parties: {d['parties']}")
    if d.get("source_url"):
        lines.append(f"- url: {d['source_url']}")
    return "\n".join(lines)


class LookupInput(BaseModel):
    """Input for exact, non-semantic retrieval by identifier."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_number: str | None = Field(
        None,
        description="Exact official document number, e.g. matsne '55'. Non-unique: many "
        "documents share a number across years/agencies — pass 'source' to narrow.",
    )
    registration_code: str | None = Field(
        None, description="Exact registry code (unique), e.g. matsne '140130000.22.034.017712'."
    )
    document_id: str | None = Field(
        None, description="Exact internal document_id (unique within a source)."
    )
    source: str | None = Field(
        None, description="Optional source key to disambiguate, e.g. 'matsne'."
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.MARKDOWN,
        description="'markdown' for a readable list, 'json' for structured records.",
    )


@mcp.tool(
    name="legal_lookup",
    annotations={
        "title": "Look Up Legal Document by Identifier",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def legal_lookup(params: LookupInput) -> str:
    """Find documents by exact identifier — number, registry code, or document_id.

    Non-semantic and no model load: it filters the Qdrant payload directly, so it is the
    reliable way to pull a specific act a lawyer already knows the number of (e.g. matsne
    Order №55). Provide at least one of document_number / registration_code / document_id;
    add ``source`` to disambiguate a non-unique document_number. Returns one entry per
    matching document (deduplicated across chunks). To read the full body, pass a result's
    source + document_id to ``legal_get_document``.

    Returns:
        str: Markdown list (title, source/type/date, document_id, number, parties, url)
        of matching documents, or — when response_format='json' — {"count", "documents":
        [...]}. Returns a "No documents found" / "provide an identifier" message otherwise.
    """
    if not (params.document_number or params.registration_code or params.document_id):
        return (
            "Provide at least one identifier: document_number, registration_code, or "
            "document_id."
        )
    try:
        cfg = _get_cfg()
        client = _get_client()
        flt = build_filter(
            source=params.source,
            document_number=params.document_number,
            registration_code=params.registration_code,
            document_id=params.document_id,
        )
        points = await asyncio.to_thread(_scroll_all, client, cfg.collection_name, flt)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)

    docs = _dedup_documents(points)
    if not docs:
        return "No documents found for that identifier. Check the value or drop the source filter."

    if params.response_format is ResponseFormat.JSON:
        return json.dumps({"count": len(docs), "documents": docs}, ensure_ascii=False, indent=2)

    blocks = [f"# {len(docs)} document(s) found", ""]
    blocks.extend(_format_doc_line(rank, d) for rank, d in enumerate(docs, 1))
    return "\n\n".join(blocks)


class BrowseInput(BaseModel):
    """Input for non-semantic, document-level filtering / browsing."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    source: str | None = Field(None, description="Source key, e.g. 'matsne', 'constcourt'.")
    document_type: str | None = Field(None, description="Exact document_type value.")
    court: str | None = Field(None, description="Exact court value, e.g. 'supremecourt'.")
    language: str | None = Field(None, description="Language code, e.g. 'ka'.")
    document_number: str | None = Field(None, description="Exact official document number.")
    parties: str | None = Field(None, description="Full-text match on party/person names.")
    contains: str | None = Field(
        None, description="Full-text keyword/phrase that must appear in the document body."
    )
    date_from: str | None = Field(None, description="Earliest date, inclusive, YYYY-MM-DD.")
    date_to: str | None = Field(None, description="Latest date, inclusive, YYYY-MM-DD.")
    limit: int = Field(20, description="Max documents to return.", ge=1, le=200)
    offset: int = Field(0, description="Documents to skip (pagination).", ge=0)
    sort: str = Field("date_desc", description="'date_desc' (newest first) or 'date_asc'.")
    response_format: ResponseFormat = Field(
        ResponseFormat.MARKDOWN,
        description="'markdown' for a readable list, 'json' for structured records.",
    )


@mcp.tool(
    name="legal_browse",
    annotations={
        "title": "Browse / Filter Legal Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def legal_browse(params: BrowseInput) -> str:
    """List documents by structured filters — date range, type, court, number, party, keyword.

    Non-semantic and no model load: this is the "browse the corpus" tool, complementary to
    the meaning-based ``legal_search``. Use it to answer "all decisions from this court in
    March 2026", "everything containing this phrase", "documents with this number", etc.
    Results are deduplicated to one entry per document and sorted by date. At least one
    filter is recommended (an unfiltered call returns the most recent documents overall).

    Returns:
        str: Markdown list of matching documents (title, source/type/date, document_id,
        number, parties, url), or — when response_format='json' — {"count", "offset",
        "limit", "documents": [...]}. ``count`` is the page size, not the global total.
    """
    try:
        cfg = _get_cfg()
        client = _get_client()
        flt = build_filter(
            source=params.source,
            document_type=params.document_type,
            court=params.court,
            language=params.language,
            document_number=params.document_number,
            parties=params.parties,
            contains=params.contains,
            date_from=params.date_from,
            date_to=params.date_to,
        )
        points = await asyncio.to_thread(_scroll_all, client, cfg.collection_name, flt)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)

    docs = _dedup_documents(points)
    if params.sort == "date_asc":
        docs.reverse()
    page = docs[params.offset : params.offset + params.limit]

    if not page:
        return "No documents match those filters. Broaden the date range or remove a filter."

    if params.response_format is ResponseFormat.JSON:
        return json.dumps(
            {"count": len(page), "offset": params.offset, "limit": params.limit, "documents": page},
            ensure_ascii=False,
            indent=2,
        )

    header = f"# {len(page)} document(s) (of {len(docs)} matched), offset {params.offset}"
    blocks = [header, ""]
    blocks.extend(_format_doc_line(params.offset + rank, d) for rank, d in enumerate(page, 1))
    return "\n\n".join(blocks)


def _source_counts(client, collection: str) -> dict | None:
    """Per-source point counts via a facet, or None if the server doesn't support it."""
    try:
        resp = client.facet(collection_name=collection, key="source", limit=50)
        return {hit.value: hit.count for hit in resp.hits}
    except Exception:  # noqa: BLE001 - facet is a nicety, never fail the tool over it
        return None


@mcp.tool(
    name="legal_collection_info",
    annotations={
        "title": "Legal Corpus Collection Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def legal_collection_info() -> str:
    """Report what's available in the Qdrant collection (no embedding / model load).

    Use this first to see how much data is indexed, the embedding model/dimension,
    and which sources are queryable — so you can pick sensible ``source`` filters
    for ``legal_search``.

    Returns:
        str: Markdown summary with collection name, Qdrant URL, total points,
        dense vector dimension, embedding model, the known source keys, and (when
        the server supports faceting) per-source point counts.
    """
    try:
        cfg = _get_cfg()
        client = _get_client()
        info = await asyncio.to_thread(client.get_collection, cfg.collection_name)
        counts = await asyncio.to_thread(_source_counts, client, cfg.collection_name)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)

    vectors = info.config.params.vectors
    dense_dim = (
        vectors["dense"].size if isinstance(vectors, dict) else getattr(vectors, "size", None)
    )

    lines = [
        f"# Collection: {cfg.collection_name}",
        "",
        f"- qdrant_url: {cfg.qdrant_url}",
        f"- points: {getattr(info, 'points_count', None)}",
        f"- dense_dim: {dense_dim} (cosine, hybrid dense + sparse)",
        f"- embed_model: {cfg.embed_model}",
        f"- reranker: {cfg.rerank_model if cfg.rerank_enabled else 'disabled'}"
        + (f" (gate ≥ {cfg.rerank_min_score})" if cfg.rerank_enabled and cfg.rerank_min_score is not None else ""),
        f"- known sources: {', '.join(sorted(SOURCES))}",
    ]
    if counts:
        lines.append("")
        lines.append("## Points per source")
        for src, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {src}: {count}")

    lines += [
        "",
        "## Retrieval tools",
        "- legal_search: hybrid (dense+sparse) recall + cross-encoder rerank; optional filters "
        "source, court, status (in_force/repealed/pending), language, document_type, "
        "document_number, registration_code, parties, contains, date_from/date_to.",
        "- legal_lookup: exact, no embedding — by document_number, registration_code, or document_id.",
        "- legal_browse: non-semantic document-level listing by source/type/court/date range/"
        "number/parties/contains, paginated and date-sorted.",
        "- legal_get_document: full text by source + document_id.",
        "",
        "## Per-source 'document_number' meaning",
        "- matsne: minister/act number (e.g. 55) · registration_code holds the registry code",
        "- ecd: case_no · constcourt: act number(s) · napr: decision_no (app_no in registration_code)",
        "- supremecourt: case_number · tas: document_no · tbappeal: none",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the MCP server over stdio (the default transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
