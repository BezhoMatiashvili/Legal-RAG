"""Map gold evidence spans to the chunks that cover them under the current chunk config.

A gold judgment is a half-open char span ``[char_start, char_end)`` into a document's
cleaned ``body_markdown``. Retrieval is chunk-level, so a span is *covered* by every chunk
whose body region overlaps it, **plus** — when the span sits inside a Markdown heading line,
which :func:`ingest.chunking._split_sections` lifts into each chunk's context prefix rather
than a body region — the chunks that heading governs (their embedded ``text`` carries the
heading, so the cited text is genuinely retrievable there; see the ``legal_citation`` pairs
whose evidence is the amended-act title in the doc heading).

Because :func:`ingest.chunking.chunk_document` is deterministic and offset-aware, this
survives re-chunking: re-run it with the current config and the mapping updates itself.
It must be called with the **same tokenizer + config as the evaluated index** — chunk
boundaries depend on ``count_tokens`` via the token budget.
"""

from collections.abc import Callable, Sequence

from ingest.chunking import Chunk, chunk_document, heading_spans


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open overlap of ``[a_start,a_end)`` and ``[b_start,b_end)``.

    A zero-length (point) gold span ``[p,p)`` is matched inclusively (``a_start <= p <=
    a_end``) so an annotation that collapsed to a caret still maps to its chunk(s).
    """
    if b_start == b_end:
        return a_start <= b_start <= a_end
    return a_start < b_end and b_start < a_end


def chunks_covering_span(
    chunks: Sequence[Chunk],
    hspans: Sequence[tuple[int, int, str]],
    char_start: int,
    char_end: int,
) -> set[int]:
    """Chunk indices covering ``[char_start,char_end)`` via body overlap ∪ heading governance."""
    hit = {
        c.chunk_index
        for c in chunks
        if _overlaps(c.char_start, c.char_end, char_start, char_end)
    }
    head_texts = [ht for (hs, he, ht) in hspans if _overlaps(hs, he, char_start, char_end)]
    if head_texts:
        hit |= {
            c.chunk_index
            for c in chunks
            if any(ht in c.heading_path for ht in head_texts)
        }
    return hit


def map_spans_to_chunks(
    body: str,
    spans: Sequence[tuple[int, int]],
    *,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    count_tokens: Callable[[str], int],
) -> dict[int, set[int]]:
    """Map each span (by list index) to the set of covering ``chunk_index`` values."""
    chunks = chunk_document(
        body, max_tokens=max_tokens, overlap=overlap, min_tokens=min_tokens,
        count_tokens=count_tokens,
    )
    hspans = heading_spans(body)
    return {
        i: chunks_covering_span(chunks, hspans, cs, ce)
        for i, (cs, ce) in enumerate(spans)
    }


def graded_relevant_chunks(
    body: str,
    relevance: Sequence,
    *,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    count_tokens: Callable[[str], int],
) -> dict[int, int]:
    """For one document, return ``chunk_index -> max grade`` over the covering spans.

    ``relevance`` items expose ``.char_start``, ``.char_end`` and ``.grade``. A chunk covered
    by several spans takes the highest grade.
    """
    chunks = chunk_document(
        body, max_tokens=max_tokens, overlap=overlap, min_tokens=min_tokens,
        count_tokens=count_tokens,
    )
    hspans = heading_spans(body)
    graded: dict[int, int] = {}
    for rel in relevance:
        for ci in chunks_covering_span(chunks, hspans, rel.char_start, rel.char_end):
            graded[ci] = max(graded.get(ci, 0), rel.grade)
    return graded
