"""Map each spider's raw item shape onto a single canonical document record.

The scraped items use different field names per source (id is ``document_id`` /
``legal_id`` / ``case_id`` / ``slug``; date is ``date`` / ``decision_date`` /
``registration_date`` / ``publication_date``; the official document number is
``document_number`` / ``case_no`` / ``number`` / ``decision_no`` / ``case_number`` /
``document_no``; etc.). A ``SourceSpec`` declares how to pull the common fields so the
rest of the pipeline is source-agnostic.
"""

import re
from dataclasses import dataclass, field

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DMY_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
# Georgian text dates like "26 მარტი 2026" or "26 მარტის 2026 17:55".
_GEO_DATE_RE = re.compile(r"(\d{1,2})\s+([ა-ჿ]+)\s+(\d{4})")

# Nominative month names; we look up by a stem so the genitive ("მარტის") also resolves.
GEORGIAN_MONTHS = {
    "იანვარ": "01",
    "თებერვალ": "02",
    "მარტ": "03",
    "აპრილ": "04",
    "მაის": "05",
    "ივნის": "06",
    "ივლის": "07",
    "აგვისტ": "08",
    "სექტემბერ": "09",
    "ოქტომბერ": "10",
    "ნოემბერ": "11",
    "დეკემბერ": "12",
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
    w = word.strip().rstrip("ის").rstrip("ი")  # strip genitive -ის / nominative -ი tail
    for stem, mm in GEORGIAN_MONTHS.items():
        if w.startswith(stem) or stem.startswith(w):
            return mm
    return None


def _valid_iso(year: str, month: str, day: str) -> str | None:
    """Return ``YYYY-MM-DD`` if the parts form a calendar-plausible date, else None."""
    try:
        y, m, d = int(year), int(month), int(day)
    except ValueError:
        return None
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


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

        return CanonicalDoc(
            source=self.source,
            document_id=document_id,
            title=title,
            date=_parse_date(date_raw),
            date_raw=str(date_raw).strip() if date_raw not in (None, "") else None,
            language=language,
            document_type=doc_type,
            court=court,
            source_url=self._first(item, self.url_fields),
            document_number=str(number_raw).strip() if number_raw not in (None, "") else None,
            registration_code=str(registration_raw).strip() if registration_raw not in (None, "") else None,
            parties=self._parties(item, title),
            status=normalize_status(status_raw),
            status_raw=status_raw,
            in_force_date=_parse_date(self._first(item, self.in_force_fields)),
            expiry_date=_parse_date(self._first(item, self.expiry_fields)),
            body_markdown=item.get("body_markdown") or "",
            extra=item,
        )


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
    ),
}


def normalize(source: str, item: dict) -> CanonicalDoc:
    try:
        spec = SOURCES[source]
    except KeyError:
        raise ValueError(f"Unknown source {source!r}; known: {sorted(SOURCES)}") from None
    return spec.build(item)
