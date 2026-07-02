"""Structure-aware Markdown chunking, token-bounded with overlap.

Splits on Markdown headings (our bodies already carry them), packs each section to a
token budget with overlap, merges tiny tail chunks, and prepends the heading path to
each chunk for lightweight context. The token counter is injectable so production uses
the BGE-M3 tokenizer while tests run offline with a word counter.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def default_token_counter(text: str) -> int:
    """Whitespace word count — a cheap proxy used by tests and as a fallback."""
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    heading_path: list[str]
    token_count: int


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    """Break Markdown into (heading_path, body) sections by ATX headings."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    cur_path: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(cur_path), body))

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            stack[:] = [(lv, t) for (lv, t) in stack if lv < level]
            stack.append((level, m.group(2).strip()))
            cur_path = [t for (_, t) in stack]
        else:
            buf.append(line)
    flush()

    if not sections and text.strip():
        sections = [([], text.strip())]
    return sections


def _atoms(body: str, max_tokens: int, count: Callable[[str], int]) -> list[tuple[str, int]]:
    """Break a section body into atoms (paragraphs, hard-splitting oversized ones)."""
    atoms: list[tuple[str, int]] = []
    for para in _PARA_RE.split(body):
        para = para.strip()
        if not para:
            continue
        tok = count(para)
        if tok <= max_tokens:
            atoms.append((para, tok))
            continue
        for sent in (s for s in _SENT_RE.split(para) if s.strip()):
            stok = count(sent)
            if stok <= max_tokens:
                atoms.append((sent.strip(), stok))
                continue
            # Pathologically long sentence: pack words into windows.
            window: list[str] = []
            wtok = 0
            for word in sent.split():
                w = count(word) or 1
                if window and wtok + w > max_tokens:
                    atoms.append((" ".join(window), wtok))
                    window, wtok = [], 0
                window.append(word)
                wtok += w
            if window:
                atoms.append((" ".join(window), wtok))
    return atoms


def _pack(atoms, max_tokens, overlap, min_tokens) -> list[tuple[str, int]]:
    """Greedily pack atoms to the token budget, seeding each new chunk with overlap."""
    packed: list[tuple[list[tuple[str, int]], int]] = []
    cur: list[tuple[str, int]] = []
    cur_tok = 0
    for atom in atoms:
        if cur and cur_tok + atom[1] > max_tokens:
            packed.append((cur, cur_tok))
            seed: list[tuple[str, int]] = []
            seed_tok = 0
            for a in reversed(cur):
                if seed_tok + a[1] > overlap:
                    break
                seed.insert(0, a)
                seed_tok += a[1]
            cur, cur_tok = list(seed), seed_tok
        cur.append(atom)
        cur_tok += atom[1]
    if cur:
        packed.append((cur, cur_tok))

    # Fold a too-small trailing chunk back into its predecessor.
    if len(packed) >= 2 and packed[-1][1] < min_tokens:
        tail_atoms, tail_tok = packed.pop()
        prev_atoms, prev_tok = packed[-1]
        packed[-1] = (prev_atoms + tail_atoms, prev_tok + tail_tok)

    return [("\n\n".join(a[0] for a in subset), tok) for subset, tok in packed]


def chunk_document(
    text: str,
    *,
    max_tokens: int = 512,
    overlap: int = 80,
    min_tokens: int = 64,
    count_tokens: Callable[[str], int] = default_token_counter,
) -> list[Chunk]:
    """Chunk a Markdown document into overlapping, heading-aware chunks."""
    chunks: list[Chunk] = []
    idx = 0
    for path, body in _split_sections(text or ""):
        ctx = " > ".join(path)
        # Leave room for the prepended heading context, but never collapse the budget
        # to near-zero when the heading is long (keep at least a quarter of max_tokens).
        ctx_tokens = count_tokens(ctx) if ctx else 0
        budget = max(max_tokens // 4, max_tokens - ctx_tokens)
        for piece, _ in _pack(_atoms(body, budget, count_tokens), budget, overlap, min_tokens):
            chunk_text = f"{ctx}\n\n{piece}" if ctx else piece
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=idx,
                    heading_path=path,
                    token_count=count_tokens(chunk_text),
                )
            )
            idx += 1
    return chunks
