"""Citation & alias exact-match routing (improvement I1).

Embeddings structurally fail referential queries ("law №432", "შრომის კოდექსი მუხლი 31"):
the identifier tokens carry no semantic signal, so the cited document rarely surfaces.
This module detects an explicit citation in a query, resolves it through the exact
keyword-indexed payload fields (``document_number`` / ``registration_code`` /
``document_id``), and lets the caller give resolved identity candidates a separate
ordering priority without pretending that identity confidence is semantic relevance.

Everything is behind the ``CITATION_ROUTE`` knob (off by default):

* ``ids``  — only unambiguous document-identifier patterns fire (registration codes,
  №-numbers next to an act word, case numbers). The safe mode.
* ``full`` — additionally resolves known law titles/aliases from the checked-in alias
  table, but only when the query is *dominated* by the reference (a known-item lookup),
  not merely mentioning a law inside a broader question.

Resolution is a dense query restricted by the exact filter — the query's own text picks
the best chunks *within* the cited document(s), which also disambiguates non-unique
``document_number`` values. A citation that resolves to nothing falls through cleanly.
"""

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

PIN_LIMIT = 3      # max exact-match chunks prepended ahead of the semantic results
# Kept as an import-compatible legacy constant only.  Exact resolution no longer writes
# this value into ``point.score``; downstream code reads the identity metadata instead.
PIN_SCORE = 1.0
RESOLUTION_SCAN_LIMIT = 20
ALIAS_DOMINANCE = 0.6  # min fraction of query chars the reference must cover (mode=full)

_ALIAS_PATH = Path(__file__).parent / "data" / "law_aliases.json"


@dataclass(frozen=True)
class CitationRef:
    kind: str            # registration_code | document_number | case_number | law_alias
    filters: dict = field(default_factory=dict)  # kwargs for search.build_filter
    matched: str = ""    # raw matched span (logging / tests)
    # Fallback filter dicts tried in order when ``filters`` resolves to nothing — e.g.
    # constcourt stores its number WITH the prefix (`N1603`, `№1/8/594`), matsne without.
    alternates: tuple = ()


# --- detection --------------------------------------------------------------------

# Identifier shapes seen in real payloads (most specific first — order matters):
#   matsne registration_code  240140000.05.001.102038
#   ecd document_number       330100116001532776 (18 digits; before the 15-digit napr code)
#   napr registration_code    011746498351723 (15 digits)
#   tas document_number       AR111390
_RE_MATSNE_REG = re.compile(r"\b\d{9}\.\d{2}\.\d{3}\.\d{6}\b")
_RE_ECD_NUM = re.compile(r"\b\d{18}\b")
_RE_NAPR_REG = re.compile(r"\b\d{15}\b")
_RE_TAS_NUM = re.compile(r"\b[A-Z]{2}\d{5,}\b")

# №-prefixed act numbers (`№71`, `№ 124`, `N01/902`). A bare number is NOT a citation —
# an act word must appear somewhere in the query (stems, so `ბრძანებაში` matches too).
_RE_ACT_NUM = re.compile(r"(?:№\s*|\bN(?=[0-9]))([0-9][0-9A-Za-zა-ჿ/()\-]*)")
_ACT_CONTEXT = re.compile(r"ბრძანებ|დადგენილებ|კანონ|განკარგულებ|წესდებ|რეზოლუცი")

# Court case numbers (`1გ/620-17`, `1/ბ-167-17`, `ბს-729-721(კ-16)`), gated on a case word.
# `მე-` is excluded up front: `მე-17` is an article ordinal, never a case number.
_RE_CASE_NUM = re.compile(
    r"\b(?:(?!მე-)[ა-ჿ]{1,3}-|\d+[ა-ჿ]*/[ა-ჿ]*-?)[0-9][0-9ა-ჿ/\-]*(?:\([ა-ჿ]+-?\d+\))?"
)
_CASE_CONTEXT = re.compile(r"საქმ|განჩინებ|განაჩენ|გადაწყვეტილებ|სარჩელ")
# Conservative Latin/English court identifier: the mandatory letter prefix prevents a
# date or article number from matching, and an explicit English case/decision word gates it.
_RE_LATIN_CASE_NUM = re.compile(r"\b[A-Z]{1,3}-\d{1,7}/\d{2,4}\b", re.IGNORECASE)
_EN_CASE_CONTEXT = re.compile(r"\b(?:case|judg(?:e)?ment|decision|ruling)\b", re.IGNORECASE)
_CONSTCOURT_CONTEXT = re.compile(r"კონსტიტუციურ[^\n]{0,40}(?:სარჩელ|წარდგინებ)|"
                                 r"(?:constitutional\s+(?:claim|complaint|submission))",
                                 re.IGNORECASE)
_RE_CONSTCOURT_NUM = re.compile(r"(?:№\s*|\bN(?=\d))(\d{2,6})\b")

# Guards: an article reference is never a document identifier.
_ARTICLE_AFTER = re.compile(r"^\s*[-–]?ე?\s*მუხლ")   # right after the number: `-ე მუხლი`
_ARTICLE_BEFORE = re.compile(r"მე-$")                  # right before it: `მე-17`


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _is_article_ref(query: str, start: int, end: int) -> bool:
    return bool(_ARTICLE_AFTER.match(query[end:]) or _ARTICLE_BEFORE.search(query[:start]))


def _prefixed_alternates(value: str) -> tuple:
    if value.startswith(("N", "№")):
        return ()
    return ({"document_number": f"N{value}"}, {"document_number": f"№{value}"})


def extract_citation(query: str, *, mode: str = "ids", aliases: list | None = None):
    """Detect an explicit citation in ``query`` → :class:`CitationRef`, or ``None``.

    ``mode="ids"`` fires only on document-identifier patterns; ``mode="full"`` adds
    alias-table resolution for reference-dominated queries. Plain questions, bare
    numbers, and article references all return ``None`` (the caller falls through to
    normal semantic search, byte-identical to knob-off).
    """
    q = _nfc(query)

    m = _RE_MATSNE_REG.search(q)
    if m:
        return CitationRef("registration_code", {"registration_code": m.group(0)}, m.group(0))
    m = _RE_ECD_NUM.search(q)
    if m:
        return CitationRef("document_number", {"document_number": m.group(0)}, m.group(0))
    m = _RE_NAPR_REG.search(q)
    if m:
        return CitationRef("registration_code", {"registration_code": m.group(0)}, m.group(0))
    m = _RE_TAS_NUM.search(q)
    if m:
        return CitationRef("document_number", {"document_number": m.group(0)}, m.group(0))

    # Constitutional Court identifiers are commonly bare ``№1848``/``N1848`` values,
    # which are too ambiguous globally but safe under an explicit source/context template.
    m = _RE_CONSTCOURT_NUM.search(q)
    if m and _CONSTCOURT_CONTEXT.search(q):
        value = m.group(1)
        return CitationRef(
            "case_number",
            {"source": "constcourt", "document_number": value},
            m.group(0),
            alternates=(
                {"source": "constcourt", "document_number": f"N{value}"},
                {"source": "constcourt", "document_number": f"№{value}"},
            ),
        )

    m = _RE_ACT_NUM.search(q)
    if m and _ACT_CONTEXT.search(q) and not _is_article_ref(q, m.start(1), m.end(1)):
        return CitationRef("document_number", {"document_number": m.group(1)}, m.group(0),
                           alternates=_prefixed_alternates(m.group(1)))

    m = _RE_CASE_NUM.search(q)
    if m and _CASE_CONTEXT.search(q) and not _is_article_ref(q, m.start(), m.end()):
        return CitationRef("case_number", {"document_number": m.group(0)}, m.group(0),
                           alternates=_prefixed_alternates(m.group(0)))

    m = _RE_LATIN_CASE_NUM.search(q)
    if m and _EN_CASE_CONTEXT.search(q) and not _is_article_ref(q, m.start(), m.end()):
        return CitationRef(
            "case_number", {"document_number": m.group(0)}, m.group(0)
        )

    if mode == "full":
        return _match_alias(q, load_aliases() if aliases is None else aliases)
    return None


# --- alias table (mode=full) ------------------------------------------------------

_QUOTE_TRANS = str.maketrans("", "", "„“”«»\"'")


def _norm_alias_text(text: str) -> str:
    return " ".join(_nfc(text).lower().translate(_QUOTE_TRANS).split())


@lru_cache(maxsize=1)
def _cached_aliases() -> tuple:
    if not _ALIAS_PATH.exists():
        return ()
    data = json.loads(_ALIAS_PATH.read_text(encoding="utf-8"))
    return tuple(data.get("laws", ()))


def load_aliases() -> list:
    """Entries of ``data/law_aliases.json`` (built from corpus titles), or ``[]``."""
    return list(_cached_aliases())


# Tokens that are reference scaffolding, not question content — they count toward the
# dominance coverage so `„სამოქალაქო კოდექსი" მუხლი 829` reads as a pure reference.
_SCAFFOLD = re.compile(
    r"საქართველოს|შესახებ|კანონი[სთ]?|კოდექსი[სთ]?|მუხლი[სთ]?|ნაწილი[სთ]?|"
    r"პუნქტი[სთ]?|ქვეპუნქტი[სთ]?|მე-\d+|\d+[-–]ე|\d+(\.\d+)*(\.[ა-ჿ])?|№\s*\S+"
)


def _match_alias(q: str, aliases) -> CitationRef | None:
    qn = _norm_alias_text(q)
    best = None  # (alias_len, entry, alias)
    for entry in aliases:
        for alias in entry.get("aliases", []) or []:
            a = _norm_alias_text(alias)
            if a and a in qn and (best is None or len(a) > len(best[0])):
                best = (a, entry)
    if best is None:
        return None
    a, entry = best
    covered = len(a) + sum(len(m.group(0)) for m in _SCAFFOLD.finditer(qn.replace(a, "")))
    if covered / max(len(qn), 1) < ALIAS_DOMINANCE:
        return None  # the query mentions the law but asks a broader question
    filters = {}
    if entry.get("registration_code"):
        filters = {"registration_code": entry["registration_code"]}
    elif entry.get("document_id") and entry.get("source"):
        filters = {"source": entry["source"], "document_id": entry["document_id"]}
    if not filters:
        return None  # entry's base act is absent from the corpus (consolidation pending)
    return CitationRef("law_alias", filters, a)


# --- resolution & pinning ---------------------------------------------------------


def _normative_payload(payload: Mapping[str, object]) -> bool:
    return payload.get("source") == "matsne" or payload.get("document_type") in {
        "legislation",
        "normative_act",
    }


def _effective_identity_payload(
    payload: Mapping[str, object],
    *,
    effective_on: str,
    require_in_force: bool,
) -> bool:
    """Return whether a representative proves the requested legal version.

    Courts and cases are not subjected to legislative status/effective-interval rules.
    Canonical normative versions, however, must prove complete lineage and a half-open
    effective interval; a present-law lookup additionally requires ``status=in_force``.
    """

    if not _normative_payload(payload):
        return True
    if (
        not payload.get("version_id")
        or payload.get("version_lineage_complete") is not True
        or payload.get("version_ambiguous") is True
        or (require_in_force and payload.get("status") != "in_force")
    ):
        return False
    try:
        when = date.fromisoformat(effective_on)
        start = date.fromisoformat(str(payload.get("effective_from") or "")[:10])
        raw_end = payload.get("effective_to")
        end = date.fromisoformat(str(raw_end)[:10]) if raw_end else None
        raw_repeal = payload.get("repeal_date")
        repeal = date.fromisoformat(str(raw_repeal)[:10]) if raw_repeal else None
    except ValueError:
        return False
    return start <= when and (end is None or when < end) and (repeal is None or when < repeal)


def _identity(payload: Mapping[str, object]) -> tuple[object, object, object]:
    return payload.get("source"), payload.get("document_id"), payload.get("version_id")


def citation_lookup(
    client,
    collection: str,
    dense_vec,
    ref: CitationRef,
    *,
    limit: int = PIN_LIMIT,
    effective_on: str | None = None,
    require_in_force: bool = False,
):
    """Best chunks *within* the cited document(s): dense query under the exact filter.

    Returns ``[]`` when the identifier resolves to nothing (e.g. tbappeal case numbers,
    which have NULL ``document_number`` payloads) — the caller falls through unchanged.
    Semantic scores are preserved.  Identity priority, match type, ambiguity, and identity
    confidence are carried as separate payload metadata; :func:`pin_points` applies the
    priority when merging lists.  Strict callers pass ``effective_on`` so historical
    versions do not make an otherwise unique law identity ambiguous.  The scan first
    resolves every version representative, then re-queries the unique effective identity.
    """
    from qdrant_client import models

    from .search import build_filter  # function-level: search.py imports this module

    for filters in (ref.filters, *ref.alternates):
        exact_filter = build_filter(**filters)
        res = client.query_points(
            collection_name=collection, query=dense_vec, using="dense",
            query_filter=exact_filter, limit=max(limit, RESOLUTION_SCAN_LIMIT),
            with_payload=True,
        )
        points = list(res.points)
        if points:
            identities: set[tuple[object, object, object]] = set()
            identity_proven = False
            representatives = []
            if hasattr(client, "scroll"):
                try:
                    chunk_zero = models.FieldCondition(
                        key="chunk_index", match=models.MatchValue(value=0)
                    )
                    identity_filter = models.Filter(
                        must=[*(exact_filter.must or []), chunk_zero]
                    )
                    representatives, next_offset = client.scroll(
                        collection_name=collection,
                        scroll_filter=identity_filter,
                        with_payload=[
                            "source", "document_id", "version_id", "document_type",
                            "status", "effective_from", "effective_to", "repeal_date",
                            "version_lineage_complete", "version_ambiguous",
                        ],
                        with_vectors=False,
                        limit=101,
                    )
                    eligible_representatives = representatives
                    if effective_on is not None:
                        eligible_representatives = [
                            pt for pt in representatives
                            if _effective_identity_payload(
                                pt.payload or {},
                                effective_on=effective_on,
                                require_in_force=require_in_force,
                            )
                        ]
                    identities = {
                        _identity(pt.payload or {}) for pt in eligible_representatives
                    }
                    identities.discard((None, None, None))
                    identity_proven = next_offset is None and len(representatives) <= 100
                except Exception:  # uncertainty fails closed in identity metadata
                    identities = set()
            if not identities:
                eligible_points = points
                if effective_on is not None:
                    eligible_points = [
                        pt for pt in points
                        if _effective_identity_payload(
                            pt.payload or {},
                            effective_on=effective_on,
                            require_in_force=require_in_force,
                        )
                    ]
                identities = {_identity(pt.payload or {}) for pt in eligible_points}
                identities.discard((None, None, None))
            if effective_on is not None and identity_proven and not identities:
                # The identifier exists, but no canonical version is effective at the
                # requested instant. Try a storage-prefix alternate before failing closed.
                continue
            if effective_on is not None:
                if identity_proven and len(identities) == 1:
                    source, document_id, version_id = next(iter(identities))
                    selected_must = list(exact_filter.must or [])
                    for key, value in (
                        ("source", source),
                        ("document_id", document_id),
                        ("version_id", version_id),
                    ):
                        if value is not None:
                            selected_must.append(models.FieldCondition(
                                key=key, match=models.MatchValue(value=value)
                            ))
                    selected = client.query_points(
                        collection_name=collection,
                        query=dense_vec,
                        using="dense",
                        query_filter=models.Filter(must=selected_must),
                        limit=max(limit, RESOLUTION_SCAN_LIMIT),
                        with_payload=True,
                    )
                    points = [
                        pt for pt in selected.points
                        if _identity(pt.payload or {}) == (source, document_id, version_id)
                        and _effective_identity_payload(
                            pt.payload or {},
                            effective_on=effective_on,
                            require_in_force=require_in_force,
                        )
                    ]
                else:
                    points = [
                        pt for pt in points
                        if _identity(pt.payload or {}) in identities
                        and _effective_identity_payload(
                            pt.payload or {},
                            effective_on=effective_on,
                            require_in_force=require_in_force,
                        )
                    ]
                    if not points and identities:
                        # Enough to communicate ambiguity to the strict caller. These
                        # representatives are not admitted as canonical answer passages.
                        points = [
                            pt for pt in representatives
                            if _identity(pt.payload or {}) in identities
                        ]
            if not points:
                continue
            ambiguous = not identity_proven or len(identities) != 1
            confidence = 1.0 if not ambiguous else 1.0 / max(len(identities), 2)
            for pt in points:
                payload = pt.payload
                if payload is None:
                    payload = {}
                    pt.payload = payload
                payload.update(
                    {
                        "match_type": ref.kind,
                        "identity_priority": True,
                        "identity_ambiguous": ambiguous,
                        "identity_confidence": confidence,
                        "identity_resolution_complete": identity_proven,
                    }
                )
            return points[:limit]
    return []


def _point_key(pt):
    pid = getattr(pt, "id", None)
    if pid is not None:
        return pid
    pl = getattr(pt, "payload", None) or {}
    return (
        pl.get("source"),
        pl.get("document_id"),
        pl.get("version_id"),
        pl.get("chunk_index"),
    )


def pin_points(pinned, semantic, top_k: int):
    """Exact-match hits first, then the semantic tail (deduped), truncated to ``top_k``."""
    if not pinned:
        return list(semantic)[:top_k]
    seen = {_point_key(pt) for pt in pinned}
    merged = list(pinned) + [pt for pt in semantic if _point_key(pt) not in seen]
    return merged[:top_k]
