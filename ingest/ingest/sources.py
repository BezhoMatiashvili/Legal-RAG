"""Map each spider's raw item shape onto a single canonical document record.

The scraped items use different field names per source (id is ``document_id`` /
``legal_id`` / ``case_id`` / ``slug``; date is ``date`` / ``decision_date`` /
``registration_date`` / ``publication_date``; the official document number is
``document_number`` / ``case_no`` / ``number`` / ``decision_no`` / ``case_number`` /
``document_no``; etc.). A ``SourceSpec`` declares how to pull the common fields so the
rest of the pipeline is source-agnostic.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date as _date

NORMALIZER_REVISION = "canonical-source-normalizer-v2"

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DMY_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
# Georgian text dates like "26 მარტი 2026" or "26 მარტის 2026 17:55".
_GEO_DATE_RE = re.compile(r"(\d{1,2})\s+([ა-ჿ]+)\s+(\d{4})")

# Morphology-stable prefixes shared by Georgian nominative and genitive month names
# (for example ``თებერვალი`` / ``თებერვლის`` and ``სექტემბერი`` / ``სექტემბრის``).
GEORGIAN_MONTHS = {
    "იანვ": "01",
    "თებერვ": "02",
    "მარტ": "03",
    "აპრილ": "04",
    "მაის": "05",
    "ივნის": "06",
    "ივლის": "07",
    "აგვისტ": "08",
    "სექტემბ": "09",
    "ოქტომბ": "10",
    "ნოემბ": "11",
    "დეკემბ": "12",
}


# Legal status: the scraper stores Georgian status strings (matsne, from the act's
# colour-coded panel). We normalise to a stable enum so it can be a keyword filter and
# so the model can tell a *repealed* act from one that is still in force. Matched by
# stem to tolerate the "აქტები"/declension tails.
STATUS_STEMS = (
    ("ძალადაკარგ", "repealed"),    # ძალადაკარგული — no longer in force
    ("ასამოქმედებ", "pending"),     # ასამოქმედებელი — adopted, not yet in force
    ("ძალაში", "in_force"),         # ძალაში მყოფი — currently in force
)

_REGISTRATION_PLACEHOLDERS = frozenset(
    {
        "-",
        "--",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "არ არის",
        "არ აქვს",
        "უცნობია",
    }
)


def normalize_registration_code(value) -> str | None:
    """Return a real registry code, mapping shared placeholders to ``None``.

    Matsne uses ``000000000.00.00.000000`` for many unrelated historical acts.  Treating
    it as a real value creates a false shared version lineage, so any punctuation-only or
    all-zero identifier fails closed to null.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or text.lower() in _REGISTRATION_PLACEHOLDERS:
        return None
    alnum = "".join(char for char in text if char.isalnum())
    if not alnum or not alnum.strip("0"):
        return None
    return text


def _source_fingerprint(item: dict) -> str:
    """SHA-256 fingerprint of the complete raw source record before normalization."""
    material = json.dumps(
        item,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _string_tuple(value) -> tuple[str, ...]:
    if value in (None, "", []):
        return ()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _optional_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    return None


def normalize_status(value) -> str | None:
    """Map a raw Georgian status string to ``in_force`` / ``repealed`` / ``pending``.

    Returns None for empty/unknown values so the status payload field stays absent
    rather than carrying noise (and so the keyword index isn't polluted).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for stem, canonical in STATUS_STEMS:
        if stem in text:
            return canonical
    return None


def _georgian_month(word: str) -> str | None:
    """Resolve a Georgian month word (nominative or genitive) to a ``MM`` string."""
    w = word.strip()
    for stem, mm in GEORGIAN_MONTHS.items():
        if w.startswith(stem):
            return mm
    return None


def _valid_iso(year: str, month: str, day: str) -> str | None:
    """Return ``YYYY-MM-DD`` if the parts form a real calendar date, else None."""
    try:
        y, m, d = int(year), int(month), int(day)
        return _date(y, m, d).isoformat()
    except (ValueError, OverflowError):
        return None


def _parse_date(value):
    """Normalize any of the sources' date formats to ISO ``YYYY-MM-DD`` (or None).

    Handles ISO (``2020-04-27``), ISO datetime (``2024-06-03 08:00:07``), day-first
    numeric (``08/04/2026`` and ``19-05-2017`` — Georgian gov convention is DD/MM/YYYY),
    and Georgian text dates (``26 მარტი 2026``). Anything else returns None; the original
    string is preserved separately in ``date_raw``.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()

    m = _DATE_RE.search(value)
    if m:
        return _valid_iso(*m.groups())

    m = _DMY_RE.search(value)
    if m:
        day, month, year = m.groups()
        return _valid_iso(year, month, day)

    m = _GEO_DATE_RE.search(value)
    if m:
        day, month_word, year = m.groups()
        mm = _georgian_month(month_word)
        if mm:
            return _valid_iso(year, mm, day)

    return None


@dataclass(frozen=True)
class CanonicalDoc:
    source: str
    document_id: str
    title: str | None
    date: str | None            # ISO YYYY-MM-DD (sortable), or None if unparseable
    date_raw: str | None        # the original scraped date string, for display
    language: str
    document_type: str | None
    court: str | None
    source_url: str | None
    document_number: str | None  # official human-facing number (per-source field)
    registration_code: str | None  # secondary/registry identifier (matsne, napr)
    parties: str | None          # party/person names, when structured at the source
    status: str | None           # normalised legal status: in_force / repealed / pending
    status_raw: str | None       # the original scraped status string (display/debug)
    in_force_date: str | None    # ISO date the act enters into force (matsne)
    expiry_date: str | None      # ISO date the act loses force (matsne)
    body_markdown: str
    extra: dict
    # Selected structured fields promoted from the raw item into the Qdrant payload so they
    # are filterable/returnable (e.g. tas applicant IDs/phones). Personal data is retained
    # by design — the index is local and confidential (see prompt.md:40). Empty for sources
    # with no promoted fields.
    promoted: dict = field(default_factory=dict)
    # Consolidation (matsne): is_consolidated = the act is a matsne "main (consolidated)"
    # document (site filter type=main — the base act carrying current consolidated text,
    # vs amendment/informational acts); consolidated_count = how many consolidated
    # versions the publication switcher lists (0 = never amended). Spider-derived from
    # switcher presence (lower bound) or type=main crawl provenance; reconciled in Qdrant
    # by scripts/reconcile_consolidated.py. None for sources/documents without the concept.
    is_consolidated: bool | None = None
    consolidated_count: int | None = None
    # Source completeness lineage. New binary-backed items set these explicitly;
    # unlabeled legacy Tbilisi articles fail closed because they are known summaries.
    content_kind: str = "full_text"
    content_complete: bool = True
    extraction_status: str = "full_text"
    source_binary_url: str | None = None
    article_summary: str | None = None
    # Reproducible source and temporal identity.  A derived ``version_id`` identifies the
    # exact canonical body but does not claim a complete amendment lineage; callers must
    # inspect ``version_lineage_complete`` before answering historical-law questions.
    source_fingerprint: str | None = None
    normalizer_revision: str = NORMALIZER_REVISION
    version_id: str | None = None
    version_id_kind: str = "derived"
    supersedes: tuple[str, ...] = ()
    effective_from: str | None = None
    effective_to: str | None = None
    repeal_date: str | None = None
    consolidation_status: str | None = None
    version_lineage_status: str = "unknown"
    version_lineage_complete: bool = False
    consolidated_dates: tuple[str, ...] = ()
    official_url: str | None = None
    official_binary_url: str | None = None
    official_html_url: str | None = None
    official_pdf_url: str | None = None
    source_authority: str = "primary_official"
    freshness_sla_met: bool | None = None


def derived_version_id(doc: CanonicalDoc, canonical_text: str | None = None) -> str:
    """Build an immutable version identity from source identity and exact canonical text."""
    body = doc.body_markdown if canonical_text is None else canonical_text
    material = {
        "source": doc.source,
        "document_id": doc.document_id,
        "effective_from": doc.effective_from,
        "effective_to": doc.effective_to,
        "canonical_content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return "derived:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def finalize_canonical_text(doc: CanonicalDoc, canonical_text: str) -> CanonicalDoc:
    """Bind a normalized document to the exact cleaned text used for hashes and offsets."""
    version_id = doc.version_id
    if doc.version_id_kind == "derived" or not version_id:
        version_id = derived_version_id(doc, canonical_text)
    return replace(doc, body_markdown=canonical_text, version_id=version_id)


@dataclass(frozen=True)
class SourceSpec:
    source: str
    id_fields: tuple[str, ...]
    document_type: str                       # static fallback
    date_fields: tuple[str, ...] = ()
    title_fields: tuple[str, ...] = ()       # joined with " — ", empties skipped
    number_fields: tuple[str, ...] = ()      # official document number
    registration_fields: tuple[str, ...] = ()  # secondary/registry number
    parties_fields: tuple[str, ...] = ()     # joined with " / ", empties skipped
    parties_from_title: bool = False         # constcourt: split title on "წინააღმდეგ"
    doc_type_field: str | None = None        # dynamic doc type (matsne)
    court: str | None = None                 # static court
    court_field: str | None = None           # dynamic court (ecd)
    language_field: str | None = None        # dynamic language (matsne)
    status_field: str | None = None          # raw legal status string (matsne)
    in_force_fields: tuple[str, ...] = ()    # date the act enters into force
    expiry_fields: tuple[str, ...] = ()      # date the act loses force
    url_fields: tuple[str, ...] = field(default=("source_url", "document_url"))
    promote_fields: tuple[str, ...] = ()     # raw item keys copied verbatim into payload
    consolidated_field: str | None = None        # bool item key: has consolidated versions
    consolidated_count_field: str | None = None  # int item key: number of consolidated versions
    source_authority: str = "primary_official"

    def declared_keys(self) -> set[str]:
        """Every raw item key this spec reads — the schema-drift baseline of handled fields."""
        keys: set[str] = {
            "body_markdown",
            "content_kind",
            "content_complete",
            "extraction_status",
            "source_binary_url",
            "article_summary",
            "pdf_url",
            "docx_url",
            "official_url",
            "official_binary_url",
            "official_html_url",
            "official_pdf_url",
            "version_id",
            "supersedes",
            "effective_from",
            "effective_to",
            "repeal_date",
            "consolidation_status",
            "version_lineage_complete",
            "freshness_sla_met",
            "consolidated_dates",
        }
        for group in (self.id_fields, self.date_fields, self.title_fields, self.number_fields,
                      self.registration_fields, self.parties_fields, self.in_force_fields,
                      self.expiry_fields, self.url_fields, self.promote_fields):
            keys.update(group)
        for single in (self.doc_type_field, self.court_field, self.language_field,
                       self.status_field, self.consolidated_field,
                       self.consolidated_count_field):
            if single:
                keys.add(single)
        return keys

    def _first(self, item, fields):
        for f in fields:
            val = item.get(f)
            if val not in (None, ""):
                return val
        return None

    def _parties(self, item, title: str | None) -> str | None:
        if self.parties_from_title and title:
            # constcourt titles read "<plaintiff> ... <defendant> წინააღმდეგ" (= "against").
            head = title.split("წინააღმდეგ")[0].strip(" .,-")
            return head or None
        parts = [str(item[f]).strip() for f in self.parties_fields if item.get(f) not in (None, "")]
        return " / ".join(parts) or None

    def build(self, item: dict) -> CanonicalDoc:
        id_parts = [str(item[f]) for f in self.id_fields if item.get(f) not in (None, "")]
        if not id_parts:
            raise ValueError(f"{self.source}: item missing id field(s) {self.id_fields}")
        document_id = ":".join(id_parts)

        title_parts = [str(item[f]).strip() for f in self.title_fields if item.get(f) not in (None, "")]
        title = " — ".join(title_parts) or None

        doc_type = self.document_type
        if self.doc_type_field and item.get(self.doc_type_field):
            doc_type = str(item[self.doc_type_field]).strip()

        court = item[self.court_field].strip() if (self.court_field and item.get(self.court_field)) else self.court

        language = "ka"
        if self.language_field and item.get(self.language_field):
            language = str(item[self.language_field]).strip().lower()

        number_raw = self._first(item, self.number_fields)
        registration_raw = self._first(item, self.registration_fields)
        date_raw = self._first(item, self.date_fields)

        status_raw = item.get(self.status_field) if self.status_field else None
        status_raw = str(status_raw).strip() if status_raw not in (None, "") else None

        promoted = {f: item[f] for f in self.promote_fields if item.get(f) not in (None, "", [])}

        is_consolidated = None
        if self.consolidated_field is not None:
            raw = item.get(self.consolidated_field)
            if isinstance(raw, bool):
                is_consolidated = raw
            elif raw not in (None, ""):
                is_consolidated = bool(raw)
        consolidated_count = None
        if self.consolidated_count_field is not None:
            raw = item.get(self.consolidated_count_field)
            if raw not in (None, ""):
                try:
                    consolidated_count = int(raw)
                except (TypeError, ValueError):
                    consolidated_count = None

        consolidated_dates = tuple(
            sorted(
                {
                    parsed
                    for parsed in (_parse_date(value) for value in _string_tuple(item.get("consolidated_dates")))
                    if parsed
                }
            )
        )

        body_markdown = item.get("body_markdown") or ""
        raw_complete = item.get("content_complete")
        if isinstance(raw_complete, bool):
            content_complete = raw_complete
        elif isinstance(raw_complete, (int, float)):
            content_complete = bool(raw_complete)
        elif isinstance(raw_complete, str):
            content_complete = raw_complete.strip().lower() in {"1", "true", "yes"}
        elif self.source in {"tas", "tbappeal"}:
            # Pre-hardening TB Appeals records contain only article text. Pre-hardening
            # TAS records do not distinguish a full decision response from list/detail
            # metadata. Neither can be promoted as complete without an explicit recrawl.
            content_complete = False
        else:
            content_complete = bool(body_markdown.strip())
        if self.source == "tbappeal" and raw_complete is None:
            default_kind = "article_summary"
        elif self.source == "tas" and raw_complete is None:
            default_kind = "legacy_unlabeled"
        else:
            default_kind = "full_text" if content_complete else "metadata_only"
        content_kind = str(item.get("content_kind") or default_kind).strip()
        extraction_status = str(
            item.get("extraction_status")
            or ("full_text" if content_complete else "malformed")
        ).strip()

        source_url = self._first(item, self.url_fields)
        source_url = str(source_url).strip() if source_url not in (None, "") else None
        source_binary_url = self._first(
            item, ("official_binary_url", "source_binary_url", "pdf_url", "docx_url")
        )
        source_binary_url = (
            str(source_binary_url).strip()
            if source_binary_url not in (None, "")
            else None
        )
        official_url = self._first(item, ("official_url", "official_html_url")) or source_url
        official_pdf_url = self._first(item, ("official_pdf_url", "pdf_url"))
        if official_pdf_url in (None, "") and source_binary_url:
            if source_binary_url.lower().split("?", 1)[0].endswith(".pdf"):
                official_pdf_url = source_binary_url

        in_force_date = _parse_date(self._first(item, self.in_force_fields))
        expiry_date = _parse_date(self._first(item, self.expiry_fields))
        explicit_effective_from = _parse_date(item.get("effective_from"))
        explicit_effective_to = _parse_date(item.get("effective_to"))
        if explicit_effective_from:
            effective_from = explicit_effective_from
        elif self.source == "matsne":
            # The rendered consolidated body is the newest switcher version.  This gives
            # its best-known lower bound, while ``version_lineage_complete=False`` makes
            # clear that prior canonical texts were not scraped.
            effective_from = max(consolidated_dates, default=in_force_date)
        else:
            effective_from = _parse_date(date_raw)
        effective_to = explicit_effective_to or expiry_date
        repeal_date = _parse_date(item.get("repeal_date")) or expiry_date
        consolidation_status = item.get("consolidation_status")
        if consolidation_status in (None, ""):
            if is_consolidated is True:
                consolidation_status = "consolidated"
            elif is_consolidated is False:
                consolidation_status = "unconsolidated"
            else:
                consolidation_status = None
        explicit_lineage_complete = _optional_bool(item.get("version_lineage_complete"))
        if explicit_lineage_complete is None:
            version_lineage_complete = self.source != "matsne"
        else:
            version_lineage_complete = explicit_lineage_complete
        if version_lineage_complete:
            lineage_status = "complete" if self.source == "matsne" else "not_applicable"
        else:
            lineage_status = "partial" if self.source == "matsne" else "unknown"

        raw_version_id = item.get("version_id")
        doc = CanonicalDoc(
            source=self.source,
            document_id=document_id,
            title=title,
            date=_parse_date(date_raw),
            date_raw=str(date_raw).strip() if date_raw not in (None, "") else None,
            language=language,
            document_type=doc_type,
            court=court,
            source_url=source_url,
            document_number=str(number_raw).strip() if number_raw not in (None, "") else None,
            registration_code=normalize_registration_code(registration_raw),
            parties=self._parties(item, title),
            status=normalize_status(status_raw),
            status_raw=status_raw,
            in_force_date=in_force_date,
            expiry_date=expiry_date,
            body_markdown=body_markdown,
            extra=item,
            promoted=promoted,
            is_consolidated=is_consolidated,
            consolidated_count=consolidated_count,
            content_kind=content_kind,
            content_complete=content_complete,
            extraction_status=extraction_status,
            source_binary_url=source_binary_url,
            article_summary=(
                str(item["article_summary"])
                if item.get("article_summary") not in (None, "")
                else None
            ),
            source_fingerprint=_source_fingerprint(item),
            version_id=(
                str(raw_version_id).strip()
                if raw_version_id not in (None, "")
                else None
            ),
            version_id_kind="official" if raw_version_id not in (None, "") else "derived",
            supersedes=_string_tuple(item.get("supersedes")),
            effective_from=effective_from,
            effective_to=effective_to,
            repeal_date=repeal_date,
            consolidation_status=(
                str(consolidation_status).strip() if consolidation_status else None
            ),
            version_lineage_status=lineage_status,
            version_lineage_complete=version_lineage_complete,
            consolidated_dates=consolidated_dates,
            official_url=str(official_url).strip() if official_url else None,
            official_binary_url=source_binary_url,
            official_html_url=str(official_url).strip() if official_url else None,
            official_pdf_url=(
                str(official_pdf_url).strip() if official_pdf_url else None
            ),
            source_authority=self.source_authority,
            freshness_sla_met=_optional_bool(item.get("freshness_sla_met")),
        )
        if doc.version_id is None:
            doc = replace(doc, version_id=derived_version_id(doc))
        return doc


SOURCES: dict[str, SourceSpec] = {
    "matsne": SourceSpec(
        source="matsne",
        id_fields=("document_id",),
        document_type="legislation",
        doc_type_field="document_type",
        date_fields=("publication_date", "adoption_date"),
        title_fields=("title",),
        number_fields=("document_number",),
        registration_fields=("registration_code",),
        parties_fields=("document_recipient",),
        language_field="language",
        status_field="status",
        in_force_fields=("entry_into_force_date",),
        expiry_fields=("expiry_date",),
        url_fields=("document_url",),
        consolidated_field="is_consolidated",
        consolidated_count_field="consolidated_count",
    ),
    "ecd": SourceSpec(
        source="ecd",
        id_fields=("decision_document_id",),
        document_type="court_decision",
        date_fields=("decision_date",),
        title_fields=("case_no", "decision_type_name"),
        number_fields=("case_no",),
        court_field="court_name",
    ),
    "constcourt": SourceSpec(
        source="constcourt",
        id_fields=("legal_id",),
        document_type="constitutional_act",
        doc_type_field="doc_type",
        date_fields=("date",),
        title_fields=("title",),
        number_fields=("number",),
        parties_from_title=True,
        court="constcourt",
    ),
    "napr": SourceSpec(
        source="napr",
        id_fields=("document_id",),
        document_type="registry_decision",
        date_fields=("date", "decision_date"),
        title_fields=("title",),
        number_fields=("decision_no",),
        registration_fields=("app_no",),
        parties_fields=("sender",),
        court="napr",
    ),
    "tbappeal": SourceSpec(
        source="tbappeal",
        id_fields=("slug",),
        document_type="court_decision",
        date_fields=("date",),
        title_fields=("title",),
        court="tbappeal",
    ),
    "supremecourt": SourceSpec(
        source="supremecourt",
        id_fields=("case_id", "chamber"),
        document_type="court_decision",
        date_fields=("date",),
        title_fields=("subject", "case_number"),
        number_fields=("case_number",),
        court="supremecourt",
    ),
    "tas": SourceSpec(
        source="tas",
        id_fields=("document_id",),
        document_type="architecture_permit",
        date_fields=("registration_date", "create_date"),
        title_fields=("document_no", "nomenclature"),
        number_fields=("document_no",),
        court="tas",
        # Structured personal/party data promoted into the payload so staff can filter on
        # it (retained by design — the index is local & confidential; see prompt.md:40).
        # ``status``/``nomenclature`` are intentionally NOT promoted: the former collides
        # with the canonical legal-status key, the latter is already in the title.
        promote_fields=(
            "applicant_personal_no", "applicant_birth_date", "applicant_address",
            "applicant_phone", "applicant_passport",
            "executor_personal_no", "executor_phone",
            "address",
        ),
    ),
}

# Promoted payload keys that get a Qdrant index, with the index kind: exact-match keyword
# for identifiers/phones/dates, full-text for addresses. Keeps promoted PII filterable.
PROMOTED_KEYWORD_FIELDS = (
    "applicant_personal_no", "applicant_passport", "applicant_phone",
    "executor_personal_no", "executor_phone", "applicant_birth_date",
)
PROMOTED_TEXT_FIELDS = ("applicant_address", "address")


def normalize(source: str, item: dict) -> CanonicalDoc:
    try:
        spec = SOURCES[source]
    except KeyError:
        raise ValueError(f"Unknown source {source!r}; known: {sorted(SOURCES)}") from None
    return spec.build(item)


def schema_drift(source: str, item: dict, seen: set[str]) -> tuple[set[str], set[str]]:
    """Detect scraped-JSON schema drift for ``source`` against a running key baseline.

    Returns ``(new_keys, undeclared_keys)``: ``new_keys`` are raw item keys not yet in
    ``seen`` (a field appeared in the crawler output); ``undeclared_keys`` is the subset the
    :class:`SourceSpec` doesn't read at all (definitely unhandled — a likely field rename or
    a new field worth wiring in). Both empty when nothing changed. The caller seeds ``seen``
    from the first item (no alert) and unions ``new_keys`` back into it after each check, so
    drift is reported once per newly-appearing key, not per document.
    """
    new_keys = set(item) - seen
    undeclared = new_keys - SOURCES[source].declared_keys()
    return new_keys, undeclared
