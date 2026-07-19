"""Deterministic legal article cross-reference extraction."""

from __future__ import annotations

import re


_REFERENCE_PATTERNS = (
    # "ამ კანონის 5-ე მუხლით" / "მე-5 მუხლის"
    re.compile(
        r"(?:ამ\s+(?:კანონის|კოდექსის)|აღნიშნული\s+(?:კანონის|კოდექსის)|მე-)\s*"
        r"(?P<article>\d+(?:\.\d+)*)(?:[-–]?ე)?\s*მუხლ"
    ),
    # "მე-5 მუხლით განსაზღვრული" without an explicit law noun.
    re.compile(r"მე-(?P<article>\d+(?:\.\d+)*)\s*მუხლ"),
    # English evidence occasionally cross-references an article directly.
    re.compile(r"\b(?:this\s+(?:law|code)'?s\s+)?article\s+(?P<article>\d+(?:\.\d+)*)",
               re.IGNORECASE),
)


def extract_article_references(
    text: str,
    *,
    own_article_id: str | None = None,
    limit: int = 4,
) -> tuple[str, ...]:
    """Return stable unique referenced article IDs, excluding the passage's own article."""

    found: list[tuple[int, str]] = []
    for pattern in _REFERENCE_PATTERNS:
        for match in pattern.finditer(text or ""):
            found.append((match.start(), match.group("article")))
    out: list[str] = []
    for _, article in sorted(found):
        if article == own_article_id or article in out:
            continue
        out.append(article)
        if len(out) >= limit:
            break
    return tuple(out)
