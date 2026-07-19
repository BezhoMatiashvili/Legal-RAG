"""Detect legal structure in Georgian legal bodies (articles / clauses / headings).

Structure is uneven across sources (measured 2026-07-07 over the full corpus): matsne
carries `მუხლი N` articles in ~64% of docs, while court decisions (ecd) are numbered-
paragraph prose (~79% numbered clauses, ~3% articles, no Markdown headings). This module
reports which structural markers a document has and extracts article anchors, so
downstream chunking (Part 3) can split on real legal boundaries and citations can point at
an article rather than a chunk index. Pure stdlib.
"""

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

# Article identifiers in the corpus are not only Arabic integers.  Older acts use Roman
# numerals and amended acts can use decimal/superscript suffixes (for example ``12¹``).
# Same-line whitespace only ([ \t], never \s): \s matches \n, which would let a bare
# line-final "მუხლი" bind to an identifier on the *next* line.
ARTICLE_ID_PATTERN = r"(?:\d+(?:(?:[._/-]\d+)+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?|[IVXLCDM]+)"
_ARTICLE = re.compile(rf"მუხლი[ \t]*{ARTICLE_ID_PATTERN}", re.IGNORECASE)
# Markdown conversion often leaves an article/chapter wrapped in ``**``.  The captured
# group deliberately begins at the legal marker, so public offsets point at ``მუხლი`` and
# not at presentation punctuation.
_ARTICLE_LINE = re.compile(
    rf"(?mi)^[ \t]*(?:[#*_]+[ \t]*)?(მუხლი[ \t]*(?P<id>{ARTICLE_ID_PATTERN})[^\n]*)"
)
_HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
_NUM_CLAUSE = re.compile(
    r"(?m)^[ \t]*(?:[*_]+[ \t]*)?(?P<id>\d+(?:\.\d+)*)(?:[.)])(?:[ \t]+|$)"
)  # "1. " / "2) " / "3.1. "
_GEO_ENUM = re.compile(
    r"(?m)^[ \t]*(?:[*_]+[ \t]*)?(?P<id>[ა-ჰ])(?:[.)])(?:[ \t]+|$)"
)  # "ა) " / "ბ) "
# Chapter/part heading at line start followed by a numeral (arabic/roman/georgian) so the
# common prose word "ნაწილი" ("part") does not over-match.
_CHAPTER = re.compile(
    r"(?mi)^[ \t]*(?:[#*_]+[ \t]*)?"
    r"(?P<label>(?:თავი|კარი|ნაწილი)[ \t]+(?P<id>[IVXLCDMა-ჰ0-9]+)[^\n]*)"
)

KIND_ARTICLE = "article"
KIND_HEADING = "heading"
KIND_CLAUSE = "clause"
KIND_FLAT = "flat"


@dataclass(frozen=True)
class StructureInfo:
    has_article: bool
    has_heading: bool
    has_num_clause: bool
    has_geo_enum: bool
    has_chapter: bool
    article_count: int
    primary_kind: str  # article | heading | clause | flat


@dataclass(frozen=True)
class StructuralContext:
    """The legal hierarchy governing one half-open passage span.

    ``clause_ids``/``subarticle_ids`` retain every marker that begins inside a passage;
    the scalar ``clause``/``subarticle`` values are the marker governing its beginning.
    This distinction avoids pretending a multi-clause passage belongs to only one clause.
    """

    article_id: str | None = None
    article_label: str | None = None
    article_start: int | None = None
    clause: str | None = None
    subarticle: str | None = None
    chapter: str | None = None
    parent_id: str | None = None
    clause_ids: tuple[str, ...] = ()
    subarticle_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuralMarker:
    start: int
    end: int
    identifier: str | None
    label: str


@dataclass(frozen=True)
class StructuralIndex:
    """One-pass positional index reused for every chunk in a document."""

    body_length: int
    articles: tuple[StructuralMarker, ...]
    article_starts: tuple[int, ...]
    chapters: tuple[StructuralMarker, ...]
    chapter_starts: tuple[int, ...]
    clauses: tuple[StructuralMarker, ...]
    clause_starts: tuple[int, ...]
    subarticles: tuple[StructuralMarker, ...]
    subarticle_starts: tuple[int, ...]


def _clean_marker(value: str) -> str:
    """Remove Markdown emphasis left around a structural marker, preserving its words."""
    return value.strip().rstrip("#*_ ").strip()


def _governing_marker(
    markers: tuple[StructuralMarker, ...],
    starts: tuple[int, ...],
    start: int,
    end: int,
    *,
    floor: int = -1,
) -> StructuralMarker | None:
    """Find a governing marker in O(log n), respecting a parent-boundary floor."""
    preceding = bisect_right(starts, start) - 1
    if preceding >= 0 and markers[preceding].start >= floor:
        return markers[preceding]
    inside = bisect_left(starts, max(start, floor))
    if inside < len(markers) and markers[inside].start < end:
        return markers[inside]
    return None


def _markers_inside(
    markers: tuple[StructuralMarker, ...],
    starts: tuple[int, ...],
    start: int,
    end: int,
    *,
    floor: int = -1,
) -> tuple[str, ...]:
    lower = bisect_left(starts, max(start, floor))
    upper = bisect_left(starts, end)
    return tuple(
        dict.fromkeys(
            marker.identifier
            for marker in markers[lower:upper]
            if marker.identifier is not None
        )
    )


def build_index(body: str) -> StructuralIndex:
    """Scan a canonical body once and build its reusable legal-structure index."""
    text = body or ""
    articles = tuple(
        StructuralMarker(
            start=match.start(1),
            end=match.end(1),
            identifier=match.group("id").strip(),
            label=_clean_marker(match.group(1)),
        )
        for match in _ARTICLE_LINE.finditer(text)
    )
    chapters = tuple(
        StructuralMarker(
            start=match.start("label"),
            end=match.end("label"),
            identifier=match.group("id").strip(),
            label=_clean_marker(match.group("label")),
        )
        for match in _CHAPTER.finditer(text)
    )
    clauses = tuple(
        StructuralMarker(
            start=match.start(),
            end=match.end(),
            identifier=match.group("id").strip(),
            label=match.group(0).strip(),
        )
        for match in _NUM_CLAUSE.finditer(text)
    )
    subarticles = tuple(
        StructuralMarker(
            start=match.start(),
            end=match.end(),
            identifier=match.group("id").strip(),
            label=match.group(0).strip(),
        )
        for match in _GEO_ENUM.finditer(text)
    )
    return StructuralIndex(
        body_length=len(text),
        articles=articles,
        article_starts=tuple(marker.start for marker in articles),
        chapters=chapters,
        chapter_starts=tuple(marker.start for marker in chapters),
        clauses=clauses,
        clause_starts=tuple(marker.start for marker in clauses),
        subarticles=subarticles,
        subarticle_starts=tuple(marker.start for marker in subarticles),
    )


def detect(body: str) -> StructureInfo:
    """Summarise the structural markers present in a document body."""
    b = body or ""
    article_count = len(_ARTICLE_LINE.findall(b))
    has_article = article_count > 0 or bool(_ARTICLE.search(b))
    has_heading = bool(_HEADING.search(b))
    has_num_clause = bool(_NUM_CLAUSE.search(b))
    has_geo_enum = bool(_GEO_ENUM.search(b))
    has_chapter = bool(_CHAPTER.search(b))

    if has_article:
        kind = KIND_ARTICLE
    elif has_heading:
        kind = KIND_HEADING
    elif has_num_clause or has_geo_enum:
        kind = KIND_CLAUSE
    else:
        kind = KIND_FLAT

    return StructureInfo(
        has_article=has_article,
        has_heading=has_heading,
        has_num_clause=has_num_clause,
        has_geo_enum=has_geo_enum,
        has_chapter=has_chapter,
        article_count=article_count,
        primary_kind=kind,
    )


def article_spans(body: str) -> list[tuple[int, str]]:
    """Return ``(char_offset, article_label)`` for each line-anchored ``მუხლი N`` marker.

    These are citation/chunk anchors: the offset is where the article heading begins in the
    body, the label is the heading line (e.g. ``"მუხლი 5. ..."``).
    """
    return [
        (match.start(1), _clean_marker(match.group(1)))
        for match in _ARTICLE_LINE.finditer(body or "")
    ]


def article_anchors(body: str) -> list[tuple[int, int, str, str]]:
    """Return ``(start, end, article_id, label)`` anchors for structural chunking."""
    return [
        (
            match.start(1),
            match.end(1),
            match.group("id").strip(),
            _clean_marker(match.group(1)),
        )
        for match in _ARTICLE_LINE.finditer(body or "")
    ]


def chapter_spans(body: str) -> list[tuple[int, int, str]]:
    """Return exact chapter/part heading spans and cleaned labels."""
    return [
        (match.start("label"), match.end("label"), _clean_marker(match.group("label")))
        for match in _CHAPTER.finditer(body or "")
    ]


def context_for_span(
    body: str,
    start: int,
    end: int,
    *,
    index: StructuralIndex | None = None,
) -> StructuralContext:
    """Resolve the legal hierarchy governing ``body[start:end]``.

    The lookup is positional, so continuation chunks inherit the preceding article,
    chapter and clause even when their own text no longer repeats those markers.
    """
    text = body or ""
    structural_index = index or build_index(text)
    if structural_index.body_length != len(text):
        raise ValueError("structural index belongs to a different document body")
    bounded_start = max(0, min(int(start), structural_index.body_length))
    bounded_end = max(bounded_start, min(int(end), structural_index.body_length))

    article = _governing_marker(
        structural_index.articles,
        structural_index.article_starts,
        bounded_start,
        bounded_end,
    )
    chapter = _governing_marker(
        structural_index.chapters,
        structural_index.chapter_starts,
        bounded_start,
        bounded_end,
    )
    article_floor = article.start if article is not None else -1

    # A clause from the preceding article must never leak into a new article whose first
    # clause starts later.  The same reset applies to Georgian lettered subclauses.
    clause = _governing_marker(
        structural_index.clauses,
        structural_index.clause_starts,
        bounded_start,
        bounded_end,
        floor=article_floor,
    )
    clause_floor = clause.start if clause is not None else article_floor
    subarticle = _governing_marker(
        structural_index.subarticles,
        structural_index.subarticle_starts,
        bounded_start,
        bounded_end,
        floor=clause_floor,
    )

    article_id = article.identifier if article is not None else None
    article_label = article.label if article is not None else None
    chapter_label = chapter.label if chapter is not None else None
    clause_id = clause.identifier if clause is not None else None
    subarticle_id = subarticle.identifier if subarticle is not None else None

    clause_ids = _markers_inside(
        structural_index.clauses,
        structural_index.clause_starts,
        bounded_start,
        bounded_end,
        floor=article_floor,
    )
    subarticle_ids = _markers_inside(
        structural_index.subarticles,
        structural_index.subarticle_starts,
        bounded_start,
        bounded_end,
        floor=clause_floor,
    )
    if article_id:
        parent_id = f"article:{article_id}"
    elif chapter_label:
        parent_id = f"chapter:{chapter_label}"
    else:
        parent_id = None
    return StructuralContext(
        article_id=article_id,
        article_label=article_label,
        article_start=article.start if article is not None else None,
        clause=clause_id,
        subarticle=subarticle_id,
        chapter=chapter_label,
        parent_id=parent_id,
        clause_ids=clause_ids,
        subarticle_ids=subarticle_ids,
    )
