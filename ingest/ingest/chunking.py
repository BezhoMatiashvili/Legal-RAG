"""Structure-aware Markdown chunking, token-bounded with overlap.

Splits on Markdown headings and legal article markers (``მუხლი N``), packs each section to
a token budget with overlap, and merges tiny tail chunks. The stored ``chunk.text`` is the
clean body slice; the heading/section context is prepended to the *embedded* text separately
via :func:`build_embed_text`, keeping display text clean while embeddings stay context-rich.
The token counter is injectable so production uses the BGE-M3 tokenizer while tests run
offline with a word counter.

Each chunk also records the half-open ``[char_start, char_end)`` range of the *original*
body it was packed from (offsets into the ``text`` passed to :func:`chunk_document`, i.e.
the cleaned ``body_markdown``). This lets the eval harness map a gold evidence span
``(char_start, char_end)`` to whichever chunks overlap it under the current chunking
config, so span→chunk relevance survives re-chunking. Offsets are tracked in the input's
own coordinate space (never by re-locating a joined/stripped/normalized string), so they
stay exact regardless of NFC form or stray ``\\r``. The synthetic heading-context prefix
is not part of the span.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
# A legal article marker (`მუხლი N …`) at the start of a line. Georgian legislation
# (esp. matsne) structures acts by article, so an article line starts a new chunk section
# — the marker stays in the body (kept visible for citations, unlike consumed `#` headings).
# Same-line whitespace only ([ \t], never \s) so a bare line-final "მუხლი" can't bind to a
# digit on the next line (mirrors structure.py's regression-tested rule).
_ARTICLE_LINE_RE = re.compile(r"^[ \t]*მუხლი[ \t]*\d")


def default_token_counter(text: str) -> int:
    """Whitespace word count — a cheap proxy used by tests and as a fallback."""
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    heading_path: list[str]
    token_count: int
    # Half-open offsets into the original body this chunk was packed from. Sentinel -1
    # means "unset" (e.g. a Chunk hand-constructed in a test); chunk_document always sets
    # real values. See module docstring.
    char_start: int = -1
    char_end: int = -1


def _split_keep_pos(s: str, pattern: re.Pattern) -> list[tuple[str, int, int]]:
    """Like ``pattern.split(s)`` but each segment carries its ``(start, end)`` in ``s``.

    Mirrors ``re.split`` exactly for a group-less pattern: the pieces are the spans
    between successive matches, with the matched delimiters dropped (but recovered here
    as gaps between segment ends and the next segment start).
    """
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for m in pattern.finditer(s):
        out.append((s[cursor : m.start()], cursor, m.start()))
        cursor = m.end()
    out.append((s[cursor:], cursor, len(s)))
    return out


def _strip_span(seg: str, base: int) -> tuple[str, int, int]:
    """Strip ``seg`` and return ``(core, abs_start, abs_end)`` shifting both ends."""
    lead = len(seg) - len(seg.lstrip())
    core = seg.strip()
    return core, base + lead, base + lead + len(core)


def _split_sections(text: str) -> list[tuple[list[str], str, int]]:
    """Break Markdown into ``(heading_path, body, body_start)`` sections by ATX headings.

    ``body`` is a contiguous substring of ``text`` (headings always flush the buffer, so a
    section body is a maximal run of consecutive non-heading lines); ``body_start`` is where
    it begins in ``text``. For ``\\n``-only input this yields byte-identical bodies to a
    plain ``"\\n".join(lines).strip()``.
    """
    sections: list[tuple[list[str], str, int]] = []
    stack: list[tuple[int, str]] = []
    cur_path: list[str] = []
    buf: list[tuple[int, int]] = []  # (line_start, content_end) per buffered line

    def flush():
        if not buf:
            return
        raw_start, raw_end = buf[0][0], buf[-1][1]
        raw = text[raw_start:raw_end]
        body = raw.strip()
        if body:
            lead = len(raw) - len(raw.lstrip())
            sections.append((list(cur_path), body, raw_start + lead))

    cursor = 0
    contents = text.splitlines()
    keepends = text.splitlines(keepends=True)
    for content, ke in zip(contents, keepends):
        start = cursor
        cursor += len(ke)
        content_end = start + len(content)
        m = _HEADING_RE.match(content.strip())
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            stack[:] = [(lv, t) for (lv, t) in stack if lv < level]
            stack.append((level, m.group(2).strip()))
            cur_path = [t for (_, t) in stack]
        elif _ARTICLE_LINE_RE.match(content) and buf:
            # Article boundary: start a fresh section but keep the marker line in the body
            # (so "მუხლი 5 …" stays visible for citations). ``buf`` guard avoids a spurious
            # empty flush when a section already begins with an article line.
            flush()
            buf[:] = [(start, content_end)]
        else:
            buf.append((start, content_end))
    flush()

    if not sections and text.strip():
        lead = len(text) - len(text.lstrip())
        sections = [([], text.strip(), lead)]
    return sections


def heading_spans(text: str) -> list[tuple[int, int, str]]:
    """Return ``(char_start, char_end, heading_text)`` for each ATX heading line.

    Heading lines are consumed by :func:`_split_sections` (they become ``heading_path``
    context, not body), so a span pointing at heading characters overlaps no chunk body.
    The eval span→chunk mapper uses these ranges to map such a span to the chunks the
    heading governs — whose embedded ``text`` carries the heading as its context prefix,
    so the heading text is genuinely retrievable there. Uses the same scan and regex as
    :func:`_split_sections`, so the two never disagree on what a heading is.
    """
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for content, ke in zip(text.splitlines(), text.splitlines(keepends=True)):
        start = cursor
        cursor += len(ke)
        content_end = start + len(content)
        m = _HEADING_RE.match(content.strip())
        if m:
            out.append((start, content_end, m.group(2).strip()))
    return out


def _atoms(
    body: str, max_tokens: int, count: Callable[[str], int]
) -> list[tuple[str, int, int, int]]:
    """Break a section body into atoms ``(text, tok, start, end)`` (offsets into ``body``).

    Paragraphs first; oversized ones split into sentences; pathologically long sentences
    packed into word windows. Offsets are recovered from match positions so they stay
    exact through every ``.strip()``/``split`` (which otherwise drop that information).
    """
    atoms: list[tuple[str, int, int, int]] = []
    for para_raw, p_start, _ in _split_keep_pos(body, _PARA_RE):
        para, para_start, _ = _strip_span(para_raw, p_start)
        if not para:
            continue
        tok = count(para)
        if tok <= max_tokens:
            atoms.append((para, tok, para_start, para_start + len(para)))
            continue
        for sent_raw, s_start, _ in _split_keep_pos(para, _SENT_RE):
            if not sent_raw.strip():
                continue
            sent, sent_start, sent_end = _strip_span(sent_raw, para_start + s_start)
            stok = count(sent_raw)  # match legacy: count the unstripped sentence
            if stok <= max_tokens:
                atoms.append((sent, stok, sent_start, sent_end))
                continue
            # Pathologically long sentence: pack words into windows.
            words = [
                (m.group(), sent_start + m.start(), sent_start + m.end())
                for m in re.finditer(r"\S+", sent)
            ]
            window: list[tuple[str, int, int]] = []
            wtok = 0
            for w, ws, we in words:
                wt = count(w) or 1
                if window and wtok + wt > max_tokens:
                    atoms.append(
                        (" ".join(x[0] for x in window), wtok, window[0][1], window[-1][2])
                    )
                    window, wtok = [], 0
                window.append((w, ws, we))
                wtok += wt
            if window:
                atoms.append(
                    (" ".join(x[0] for x in window), wtok, window[0][1], window[-1][2])
                )
    return atoms


def _pack(
    atoms,
    max_tokens,
    overlap,
    min_tokens,
    count: Callable[[str], int] | None = None,
) -> list[tuple[str, int, int, int]]:
    """Greedily pack atoms to the token budget, seeding each new chunk with overlap.

    Returns ``(text, tok, char_start, char_end)`` per chunk; the char span is the
    ``min``/``max`` over the packed atoms' offsets — provably correct even through the
    tail-fold's non-monotonic atom order.
    """
    packed: list[tuple[list[tuple[str, int, int, int]], int]] = []
    cur: list[tuple[str, int, int, int]] = []
    cur_tok = 0

    def token_count(subset) -> int:
        if count is None:
            return sum(atom[1] for atom in subset)
        return count("\n\n".join(atom[0] for atom in subset))

    for atom in atoms:
        atom_tok = token_count([atom])
        if atom_tok > max_tokens:
            raise ValueError(
                f"atom exceeds chunk budget ({atom_tok} > {max_tokens}); "
                "the tokenizer could not split a single non-whitespace token safely"
            )
        if cur and token_count([*cur, atom]) > max_tokens:
            packed.append((cur, cur_tok))
            seed: list[tuple[str, int, int, int]] = []
            seed_tok = 0
            for a in reversed(cur):
                if seed_tok + a[1] > overlap:
                    break
                seed.insert(0, a)
                seed_tok += a[1]
            # An atom may itself consume most/all of the budget. Retain only as much overlap
            # as fits beside it; overlap is a recall aid, never permission to exceed the model
            # input contract.
            while seed and token_count([*seed, atom]) > max_tokens:
                seed_tok -= seed.pop(0)[1]
            cur = list(seed)
            cur_tok = token_count(cur)
        cur.append(atom)
        cur_tok = token_count(cur)
    if cur:
        packed.append((cur, cur_tok))

    # Fold a too-small trailing chunk back into its predecessor.
    if len(packed) >= 2 and packed[-1][1] < min_tokens:
        tail_atoms, tail_tok = packed.pop()
        prev_atoms, prev_tok = packed[-1]
        merged = prev_atoms + tail_atoms
        merged_tok = token_count(merged)
        if merged_tok <= max_tokens:
            packed[-1] = (merged, merged_tok)
        else:
            packed.append((tail_atoms, tail_tok))

    if any(tok > max_tokens for _, tok in packed):
        raise AssertionError("chunk packer emitted a token-budget overflow")

    return [
        (
            "\n\n".join(a[0] for a in subset),
            tok,
            min(a[2] for a in subset),
            max(a[3] for a in subset),
        )
        for subset, tok in packed
    ]


def chunk_document(
    text: str,
    *,
    max_tokens: int = 512,
    overlap: int = 80,
    min_tokens: int = 64,
    count_tokens: Callable[[str], int] = default_token_counter,
) -> list[Chunk]:
    """Chunk a Markdown document into overlapping, structure-aware chunks.

    ``chunk.text`` is the **clean** body slice (what gets stored/displayed); the
    heading/section context lives in ``chunk.heading_path`` and is folded into the
    *embedded* text separately by :func:`build_embed_text`. Sections break on ATX headings
    and on legal article markers (``მუხლი N``).
    """
    chunks: list[Chunk] = []
    idx = 0
    for path, body, body_start in _split_sections(text or ""):
        for piece, tok, cstart, cend in _pack(
            _atoms(body, max_tokens, count_tokens),
            max_tokens,
            overlap,
            min_tokens,
            count_tokens,
        ):
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_index=idx,
                    heading_path=path,
                    token_count=count_tokens(piece),
                    char_start=body_start + cstart,
                    char_end=body_start + cend,
                )
            )
            idx += 1
    return chunks


# Georgian status words for the v2 embed header — queries are Georgian, so the legal-force
# marker embeds in the vocabulary users actually search with.
_STATUS_KA = {"in_force": "ძალაშია", "repealed": "ძალადაკარგულია", "pending": "ძალაში შესვლამდე"}


def build_embed_text(
    text: str,
    *,
    title: str | None = None,
    document_type: str | None = None,
    heading_path: list[str] | None = None,
    document_number: str | None = None,
    date: str | None = None,
    status: str | None = None,
    is_consolidated: bool | None = None,
) -> str:
    """Context-enriched text to **embed** (the stored/displayed ``text`` stays clean).

    Prepends ``title > document_type > section/article path`` so a short clause is embedded
    with the context that disambiguates it — a Georgian legal query often matches the act
    title or article heading, not the bare clause body. Prepended to the embedded text only.

    v2 header (improvement I6; callers gate on ``cfg.embed_header_v2``): passing any of
    ``document_number`` / ``date`` / ``status`` / ``is_consolidated`` inserts one
    ``№N · YYYY-MM-DD · ძალაშია · კონსოლიდირებული`` segment after ``document_type`` —
    identifying metadata in the embedded prefix cuts retrieval failures on
    boilerplate-heavy corpora (Anthropic Contextual Retrieval; amendment acts are exactly
    that). With the v2 params left ``None`` the output is byte-identical to the v1 header.
    """
    parts: list[str] = []
    if title:
        parts.append(title)
    if document_type:
        parts.append(document_type)
    meta: list[str] = []
    if document_number and document_number not in ("0", "-"):
        meta.append(f"№{document_number}")
    if date:
        meta.append(str(date)[:10])
    if status:
        meta.append(_STATUS_KA.get(status, status))
    if is_consolidated:
        meta.append("კონსოლიდირებული")
    if meta:
        parts.append(" · ".join(meta))
    if heading_path:
        parts.extend(heading_path)
    ctx = " > ".join(p for p in parts if p)
    return f"{ctx}\n\n{text}" if ctx else text
