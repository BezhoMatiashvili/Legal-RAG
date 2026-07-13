"""Detect legal structure in Georgian legal bodies (articles / clauses / headings).

Structure is uneven across sources (measured 2026-07-07 over the full corpus): matsne
carries `მუხლი N` articles in ~64% of docs, while court decisions (ecd) are numbered-
paragraph prose (~79% numbered clauses, ~3% articles, no Markdown headings). This module
reports which structural markers a document has and extracts article anchors, so
downstream chunking (Part 3) can split on real legal boundaries and citations can point at
an article rather than a chunk index. Pure stdlib.
"""

import re
from dataclasses import dataclass

# "მუხლი 12" / "მუხლი12" anywhere; the line-anchored form is used for span extraction.
# Same-line whitespace only ([ \t], never \s): \s matches \n, which would let a bare
# line-final "მუხლი" bind to a digit on the *next* line — a false article + a span label
# that crosses the newline. See the regression test in test_structure.py.
_ARTICLE = re.compile(r"მუხლი[ \t]*\d")
_ARTICLE_LINE = re.compile(r"(?m)^[ \t]*(მუხლი[ \t]*\d+[^\n]*)")
_HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
_NUM_CLAUSE = re.compile(r"(?m)^\s*\d+[.)]\s")           # "1. " / "2) "
_GEO_ENUM = re.compile(r"(?m)^\s*[ა-ჰ][.)]\s")           # "ა) " / "ბ) "
# Chapter/part heading at line start followed by a numeral (arabic/roman/georgian) so the
# common prose word "ნაწილი" ("part") does not over-match.
_CHAPTER = re.compile(r"(?m)^\s*(თავი|კარი|ნაწილი)\s+[IVXLCDMა-ჰ0-9]")

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
    return [(m.start(1), m.group(1).strip()) for m in _ARTICLE_LINE.finditer(body or "")]
