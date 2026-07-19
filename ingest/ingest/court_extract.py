"""Deterministic judge-panel and operative-disposition extraction.

The extractor deliberately has no I/O and no model dependency.  Its outputs are
payload metadata, not evidence that changes the text embedded for retrieval.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

EXTRACTOR_REVISION = "court-extract-v1"

DISPOSITION_VALUES = (
    "inadmissible",
    "not_considered",
    "upheld",
    "overturned",
    "overturned_remanded",
    "partially_overturned",
    "modified",
    "granted",
    "remanded",
    "terminated",
    "settled",
    "unknown",
)

# This order is intentionally a legal-impact order rather than alphabetical order.
# When different appellants receive different outcomes, the strongest change to the
# reviewed judgment is the document-level disposition.
DISPOSITION_PRIORITY = (
    "overturned",
    "overturned_remanded",
    "modified",
    "partially_overturned",
    "remanded",
    "terminated",
    "settled",
    "upheld",
    "not_considered",
    "inadmissible",
    "granted",
)


def _spaced_glyphs(value: str) -> str:
    return r"[ \t\u00a0]*".join(re.escape(char) for char in value)


# ECD contains old records whose Georgian headers were transliterated with Latin
# glyphs.  ``ბრძანებს`` and ``სარეზოლუციო ნაწილი`` cover the small observed residue.
PRIMARY_OPERATIVE_HEADER_PATTERNS = (
    _spaced_glyphs("დაადგინა"),
    _spaced_glyphs("გადაწყვიტა"),
    r"d[ \t\u00a0]*a[ \t\u00a0]*a[ \t\u00a0]*d[ \t\u00a0]*g[ \t\u00a0]*i[ \t\u00a0]*n[ \t\u00a0]*a",
    r"g[ \t\u00a0]*a[ \t\u00a0]*d[ \t\u00a0]*a[ \t\u00a0]*w[ \t\u00a0]*y[ \t\u00a0]*v[ \t\u00a0]*i[ \t\u00a0]*t[ \t\u00a0]*a",
)
FALLBACK_OPERATIVE_HEADER_PATTERNS = (
    _spaced_glyphs("ბრძანებს"),
    r"სარეზოლუციო[ \t\u00a0]+ნაწილი",
)
OPERATIVE_HEADER_PATTERNS = (
    *PRIMARY_OPERATIVE_HEADER_PATTERNS,
    *FALLBACK_OPERATIVE_HEADER_PATTERNS,
)


def _header_re(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(
        r"(?im)(?<![ა-ჿA-Za-z])(?P<header>(?:"
        + "|".join(patterns)
        + r"))(?![ა-ჿA-Za-z])[ \t\u00a0]*:?"
    )


PRIMARY_OPERATIVE_HEADER_RE = _header_re(PRIMARY_OPERATIVE_HEADER_PATTERNS)
FALLBACK_OPERATIVE_HEADER_RE = _header_re(FALLBACK_OPERATIVE_HEADER_PATTERNS)
OPERATIVE_HEADER_RE = re.compile(
    r"(?im)(?<![ა-ჿA-Za-z])(?P<header>(?:"
    + "|".join(OPERATIVE_HEADER_PATTERNS)
    + r"))(?![ა-ჿA-Za-z])[ \t\u00a0]*:?"
)

# Exported morphology tables are part of the revisioned extractor contract.  The
# classifier below combines them with object guards; bare stems are never sufficient
# for ambiguous legal verbs such as "return", "cancel", and "terminate".
DISPOSITION_STEMS = {
    "inadmissible": (r"დაუშვებ", r"არ.{0,40}დაშვებ"),
    "not_considered": (r"განუხილველად",),
    "upheld": (r"უცვლელ", r"არ.{0,40}დაკმაყოფილ", r"უარი.{0,40}დაკმაყოფილ"),
    "cancel": (r"გააუქმ", r"გაუქმდ"),
    "new_decision": (
        r"ახალ[ი]?\s+გადაწყვეტილებ",
        r"გამოტანილ[ი]?[ \t]+იქნა.{0,80}(?:განაჩენ|გადაწყვეტილებ)",
        r"გამოტანილ[ი]?[ \t]+იქნა[ \t]+გამამტყუნებელი",
        r"მიღებულ[ი]?[ \t]+იქნეს[ \t]+ახალ[ი]?[ \t]+გადაწყვეტილებ",
        r"ცნობილ[ი]?[ \t]+იქნეს.{0,80}დამნაშავედ",
    ),
    "return": (r"დაუბრუნ", r"ხელახლა\s+განსახილველად", r"გადაეგზავნ"),
    "partial": (r"ნაწილობრივ",),
    "modified": (
        r"შეიცვალ",
        r"ცვლილება.{0,40}შევიდ",
        r"შევიდ.{0,40}ცვლილება",
        r"შეტანილ[ი]?[ \t]+იქნეს[ \t]+ცვლილება",
    ),
    "granted": (r"დაკმაყოფილდ", r"დაკმაყოფილდეს"),
    "terminated": (r"შეწყ",),
    "settled": (r"მორიგებ",),
}

JUDGE_MARKER_PATTERNS = (
    r"თავმჯდომარე(?:[ \t]*,[ \t]*მომხსენებელი)?",
    r"მომხსენებელი",
    r"მოსამართლ[ეა-ჿ]*",
)

_JUDGE_MARKER_RE = re.compile(
    r"(?im)(?P<label>თავმჯდომარე(?:[ \t]*,[ \t]*მომხსენებელი)?|"
    r"მომხსენებელი|მოსამართლ[ეა-ჿ]*)"
    r"(?P<delimiter>[ \t]*[-\u2013:][ \t]*|[ \t]+)"
)
_ANY_JUDGE_MARKER_RE = re.compile(
    r"(?i)თავმჯდომარე|მომხსენებელი|მოსამართლ[ეა-ჿ]*"
)
_REPORTER_PAREN_RE = re.compile(
    r"\([ \t]*(?:თავმჯდომარე[ \t]*,?[ \t]*)?მომხსენებელი[ \t]*\)", re.I
)
_ROLE_PAREN_RE = re.compile(
    r"\([ \t]*(?:თავმჯდომარე(?:[ \t]*,?[ \t]*მომხსენებელი)?|მომხსენებელი)"
    r"[ \t]*\)",
    re.I,
)
_COMPOSITION_RE = re.compile(
    r"(?:შემდეგი[ \t]+შემადგენლობით|შემადგენლობა(?:/მოსამართლეები)?)[ \t]*:",
    re.I,
)
_COMPOSITION_STOP_RE = re.compile(
    r"^(?:ზეპირ|განიხილ|შეამოწმ|მოისმინ|საქმ|საქართველოს|სასამართლო)", re.I
)
_NAME_TOKEN = r"(?:[ა-ჿ]\.?|[ა-ჿ]{2,})"
_REDACTED_FIRST_TOKEN = r"[ა-ჿ]{1,3}[-\u2013\u2014]{2,}"
_NAME_RE = re.compile(
    rf"^(?:{_NAME_TOKEN}|{_REDACTED_FIRST_TOKEN})"
    rf"(?:[ \t]+{_NAME_TOKEN}){{0,2}}$"
)
_BACKWARD_REPORTER_RE = re.compile(
    rf"(?im)^[ \t]*(?P<name>{_NAME_TOKEN}[ \t]+{_NAME_TOKEN})[ \t]*"
    r"\([ \t]*(?:თავმჯდომარე[ \t]*,?[ \t]*)?მომხსენებელი[ \t]*\)"
)
_NUMBER_OR_BOUNDARY_RE = re.compile(r"\d|განჩინება|საბოლოო", re.I)
_NARRATIVE_NAME_TOKEN_RE = re.compile(
    r"\b(?:მდივანი|მდივნობით|სასამართლოს|მიერ|წესით|გამოც|არის|ზეპირი|"
    r"მოსმენის|გარეშე|თანდასწრებით|მონაწილეობით)\b",
    re.I,
)
_ENUMERATED_POINT_RE = re.compile(
    r"(?m)^[ \t]*(?=(?:\d{1,3}|[ა-ჰ])[.)][ \t]*)"
)

_APPEAL_OBJECT_RE = re.compile(
    r"საკასაციო|სააპელაციო|კერძო\s+საჩივარ|საჩივარ|სარჩელ|შუამდგომლობ",
    re.I,
)
_REVIEWED_ACT_RE = re.compile(
    r"გასაჩივრებულ|ქვემდგომ|სააპელაციო|საქალაქო|რაიონულ|"
    r"გადაწყვეტილებ|განაჩენ|განჩინებ|დადგენილებ",
    re.I,
)
_CASE_RE = re.compile(r"საქმ(?:ე|ის|ეს|ეთა|ეში|ეზე)", re.I)
_PROCEEDINGS_RE = re.compile(r"საქმის[ \t\n]+წარმოებ", re.I)
_INADMISSIBLE_APPEAL_RE = re.compile(
    r"(?:საკასაციო|სააპელაციო|კერძო\s+საჩივარ|საჩივარ|სარჩელ|შუამდგომლობ)"
    r".{0,120}(?:დაუშვებ|არ.{0,40}დაშვებ)|"
    r"(?:დაუშვებ|არ.{0,40}დაშვებ).{0,120}"
    r"(?:საკასაციო|სააპელაციო|კერძო\s+საჩივარ|საჩივარ|სარჩელ|შუამდგომლობ)",
    re.I | re.S,
)
_NOT_CONSIDERED_APPEAL_RE = re.compile(
    r"(?:საკასაციო|სააპელაციო|კერძო\s+საჩივარ|საჩივარ|სარჩელ|შუამდგომლობ)"
    r"[^;.\n]{0,400}განუხილველ(?:ად|ი)|"
    r"განუხილველ(?:ად|ი)[^;.\n]{0,400}"
    r"(?:საკასაციო|სააპელაციო|კერძო\s+საჩივარ|საჩივარ|სარჩელ|შუამდგომლობ)",
    re.I | re.S,
)
_ACT_WORD = r"(?:გადაწყვეტილებ|განაჩენ|განჩინებ|დადგენილებ)[ა-ჿ]*"
_CANCEL_WORD = r"(?:გააუქმ[ა-ჿ]*|გაუქმდ[ა-ჿ]*)(?![ა-ჿ])"
_LOCAL_CANCEL_RE = re.compile(
    rf"{_ACT_WORD}[^;.]{{0,60}}{_CANCEL_WORD}|"
    rf"{_CANCEL_WORD}(?![ \t]*,?[ \t]*და\b)[ \t]*,?[ \t]*"
    rf"[^;.]{{0,140}}{_ACT_WORD}",
    re.I,
)
_LOCAL_MODIFIED_RE = re.compile(
    rf"{_ACT_WORD}[^;.\n]{{0,50}}შეიცვალ[ა-ჿ]*|"
    rf"შეიცვალ[ა-ჿ]*[^;.\n]{{0,80}}{_ACT_WORD}|"
    rf"{_ACT_WORD}[^;.\n]{{0,80}}შევიდ[ა-ჿ]*[^;.\n]{{0,30}}ცვლილება|"
    rf"ცვლილება[^;.\n]{{0,40}}შევიდ[ა-ჿ]*[^;.\n]{{0,100}}{_ACT_WORD}|"
    rf"{_ACT_WORD}[^;.\n]{{0,80}}შეტანილ[ი]?[ \t]+იქნეს[ \t]+ცვლილება",
    re.I,
)
_LOCAL_REMAND_RE = re.compile(
    r"საქმ(?:ე|ის|ეს|ეში|ეზე)[^;.\n]{0,120}(?:დაუბრუნ[ა-ჿ]*|"
    r"ხელახლა[ \t]+განსახილველად)[^;.\n]{0,120}(?:სასამართლო|ხელახლა)|"
    r"(?:დაუბრუნ[ა-ჿ]*|ხელახლა[ \t]+განსახილველად)[^;.\n]{0,120}"
    r"სასამართლო|"
    r"საქმ(?:ე|ის|ეს|ეში|ეზე)[^;.\n]{0,100}გადაეგზავნ[ა-ჿ]*"
    r"[^;.\n]{0,100}სასამართლო",
    re.I,
)


@dataclass(frozen=True, slots=True)
class JudgePanel:
    judges: tuple[str, ...]
    judges_raw: tuple[str, ...]
    reporting_judge: str | None
    confidence: str

    @property
    def judge_extraction_confidence(self) -> str:
        return self.confidence


@dataclass(frozen=True, slots=True)
class DispositionResult:
    disposition: str
    disposition_source: str
    disposition_confidence: str
    disposition_mixed: bool
    operative_start: int | None = None
    operative_end: int | None = None
    operative_header: str | None = None

    @property
    def source(self) -> str:
        return self.disposition_source

    @property
    def confidence(self) -> str:
        return self.disposition_confidence

    @property
    def mixed(self) -> bool:
        return self.disposition_mixed


def _nfc_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFC", value)


def _normalize_surname_case(value: str) -> str:
    """Undo only unambiguous case suffixes observed in judge labels."""

    # -ძე -> -ძემ; -ია/-ა -> -იამ/-ამ; -შვილი -> -შვილმა.
    if value.endswith("ძემ"):
        return value[:-1]
    if value.endswith("შვილმა"):
        return value[:-2] + "ი"
    if value.endswith("იამ") or value.endswith("უამ"):
        return value[:-1]
    return value


def normalize_judge_key(value: str) -> str:
    """Return ``first initial + surname`` as the stable judge facet key."""

    text = _nfc_text(value)
    text = re.sub(r"[()\[\]{},;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-\u2013")
    if not text:
        return ""
    tokens = [token.strip(". ") for token in text.split() if token.strip(". ")]
    if not tokens:
        return ""
    first_match = re.match(r"[ა-ჿ]", tokens[0])
    if first_match is None or any(
        not re.fullmatch(r"[ა-ჿ]+", token) for token in tokens[1:]
    ):
        return ""
    if len(tokens) == 1:
        return (
            _normalize_surname_case(tokens[0])
            if re.fullmatch(r"[ა-ჿ]+", tokens[0])
            else ""
        )
    first = first_match.group(0)
    surname = " ".join((*tokens[1:-1], _normalize_surname_case(tokens[-1])))
    return f"{first}. {surname}"


def _candidate_names(segment: str, *, allow_flat_pairs: bool = False) -> list[str]:
    segment = _ROLE_PAREN_RE.sub("", segment)
    names: list[str] = []
    for raw in re.split(r"[,\n]", segment):
        candidate = re.sub(r"\s+", " ", raw).strip(" \t:;-\u2013.")
        if not candidate:
            continue
        if (
            _NUMBER_OR_BOUNDARY_RE.search(candidate)
            or _NARRATIVE_NAME_TOKEN_RE.search(candidate)
        ):
            return []
        if _NAME_RE.fullmatch(candidate):
            names.append(candidate)
            continue
        tokens = candidate.split()
        if (
            allow_flat_pairs
            and len(tokens) >= 4
            and len(tokens) % 2 == 0
            and all(re.fullmatch(_NAME_TOKEN, token) for token in tokens)
        ):
            names.extend(
                " ".join(tokens[index : index + 2])
                for index in range(0, len(tokens), 2)
            )
            continue
        return []
    return names


def _marker_segment(body: str, match: re.Match[str]) -> str:
    """Capture one marker's line plus name-only continuation lines."""

    line_end = body.find("\n", match.end())
    if line_end < 0:
        line_end = len(body)
    first_line = body[match.end():line_end]
    if first_line.strip() and not _candidate_names(
        first_line,
        allow_flat_pairs=match.group("label").startswith("მოსამართლ"),
    ):
        return first_line
    end = line_end
    cursor = line_end + 1
    continuation_count = 0
    while cursor < len(body) and continuation_count < 12:
        next_end = body.find("\n", cursor)
        if next_end < 0:
            next_end = len(body)
        line = body[cursor:next_end].strip()
        if not line:
            cursor = next_end + 1
            continuation_count += 1
            continue
        if (
            _ANY_JUDGE_MARKER_RE.search(line)
            or _NUMBER_OR_BOUNDARY_RE.search(line)
            or not _candidate_names(line)
        ):
            break
        end = next_end
        cursor = next_end + 1
        continuation_count += 1
    segment = body[match.end():end]
    boundary = _NUMBER_OR_BOUNDARY_RE.search(segment)
    marker_probe = _ROLE_PAREN_RE.sub(
        lambda found: " " * len(found.group(0)), segment
    )
    next_marker = _ANY_JUDGE_MARKER_RE.search(marker_probe)
    cut_candidates = [
        found.start() for found in (boundary, next_marker) if found is not None
    ]
    if cut_candidates:
        segment = segment[: min(cut_candidates)]
    return segment


def _composition_panel(
    body: str,
) -> tuple[list[str], str | None]:
    """Read the explicit Supreme Court ``შემდეგი შემადგენლობით`` roster."""

    for match in _COMPOSITION_RE.finditer(body):
        names: list[str] = []
        reporting: str | None = None
        cursor = match.end()
        # Supreme rosters are two or three short lines separated by optional blank lines.
        # Stop at the first non-name content after the roster has started.
        for raw_line in body[cursor:].splitlines()[:30]:
            line = raw_line.strip()
            if not line:
                continue
            if _COMPOSITION_STOP_RE.search(line):
                if names:
                    break
                continue
            marker = _JUDGE_MARKER_RE.search(line)
            name_segment = line[marker.end():] if marker is not None else line
            line_names = _candidate_names(name_segment, allow_flat_pairs=True)
            if not line_names:
                if marker is not None:
                    continue
                if names:
                    break
                continue
            reporter_line = bool(_REPORTER_PAREN_RE.search(line))
            names.extend(line_names)
            if reporter_line and reporting is None:
                reporting = normalize_judge_key(line_names[0])
        if names:
            return names, reporting
    return [], None


def extract_judges(body: str) -> JudgePanel:
    """Extract and normalize the deciding panel from explicit role markers."""

    text = _nfc_text(body)
    raw_seen: dict[str, None] = {}
    key_seen: set[str] = set()
    reporting_key: str | None = None
    valid_marker = False

    composition_names, composition_reporter = _composition_panel(text)
    if composition_names:
        for raw_name in composition_names:
            key = normalize_judge_key(raw_name)
            if key:
                raw_seen.setdefault(raw_name, None)
                key_seen.add(key)
        if composition_reporter in key_seen:
            reporting_key = composition_reporter
        tail_start = max(0, len(text) - 3000)
        tail_chair: str | None = None
        tail_matches = (
            _JUDGE_MARKER_RE.finditer(text, tail_start)
            if tail_start > 0
            else ()
        )
        for match in tail_matches:
            segment = _marker_segment(text, match)
            label = match.group("label")
            names = _candidate_names(
                segment,
                allow_flat_pairs=label.startswith("მოსამართლ"),
            )
            for raw_name in names:
                key = normalize_judge_key(raw_name)
                if not key:
                    continue
                raw_seen.setdefault(raw_name, None)
                key_seen.add(key)
                if label.startswith("თავმჯდომარე") and tail_chair is None:
                    tail_chair = key
                if (
                    ("მომხსენებელი" in label or _REPORTER_PAREN_RE.search(segment))
                    and reporting_key is None
                ):
                    reporting_key = key
        if (
            reporting_key is None
            and tail_chair is not None
            and _REPORTER_PAREN_RE.search(text[: min(len(text), 5000)])
        ):
            reporting_key = tail_chair
        return JudgePanel(
            judges=tuple(sorted(key_seen)),
            judges_raw=tuple(raw_seen),
            reporting_judge=reporting_key,
            confidence="high" if key_seen else "low",
        )

    groups: list[tuple[int, str, list[str], bool]] = [
        (match.start(), "მომხსენებელი", [match.group("name")], True)
        for match in _BACKWARD_REPORTER_RE.finditer(text)
    ]
    for match in _JUDGE_MARKER_RE.finditer(text):
        segment = _marker_segment(text, match)
        label = match.group("label")
        names = _candidate_names(
            segment,
            allow_flat_pairs=label.startswith("მოსამართლ"),
        )
        if not names and label.startswith("მოსამართლე"):
            names = _candidate_names(segment.split(",", 1)[0])
        if not names:
            continue
        reporter_role = "მომხსენებელი" in label or bool(
            _REPORTER_PAREN_RE.search(segment)
        )
        groups.append((match.start(), label, names, reporter_role))

    if groups:
        reporter_groups = [group for group in groups if group[3]]
        plural_groups = [
            group for group in groups
            if group[1].startswith("მოსამართლ") and len(group[2]) > 1
        ]
        candidates = reporter_groups or plural_groups or groups
        anchor = max(candidates, key=lambda group: (len(group[2]), group[0]))
        # Chair/reporter and the remaining panel commonly occupy adjacent trailer lines.
        # Restricting extraction to that explicit roster zone prevents narrative uses of
        # "მოსამართლემ" elsewhere in the decision from becoming facet values.
        selected = [
            group for group in groups if abs(group[0] - anchor[0]) <= 800
        ]
        valid_marker = True
    else:
        selected = []

    for _offset, _label, names, reporter_role in selected:
        for raw_name in names:
            key = normalize_judge_key(raw_name)
            if not key:
                continue
            raw_seen.setdefault(raw_name, None)
            key_seen.add(key)
            if reporter_role and reporting_key is None:
                reporting_key = key

    judges = tuple(sorted(key_seen))
    if reporting_key not in key_seen:
        reporting_key = None
    return JudgePanel(
        judges=judges,
        judges_raw=tuple(raw_seen),
        reporting_judge=reporting_key,
        confidence="high" if valid_marker and judges else "low",
    )


def _has(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.I | re.S) for pattern in patterns)


def _has_local_judgment_cancellation(text: str) -> bool:
    non_judgment_objects = re.compile(
        r"გირაო|აღკვეთის[ \t]+ღონისძიებ|ყადაღ|პატიმრობ", re.I
    )
    for match in _LOCAL_CANCEL_RE.finditer(text):
        clause_start = max(text.rfind(";", 0, match.start()), text.rfind(".", 0, match.start())) + 1
        clause_ends = [
            position
            for position in (text.find(";", match.end()), text.find(".", match.end()))
            if position >= 0
        ]
        clause_end = min(clause_ends, default=len(text))
        if not non_judgment_objects.search(text[clause_start:clause_end]):
            return True
    return False


def _classify_text(text: str, *, scraped_result: bool) -> set[str]:
    outcomes: set[str] = set()
    appeal_context = scraped_result or bool(_APPEAL_OBJECT_RE.search(text))

    inadmissible = (
        _has(DISPOSITION_STEMS["inadmissible"], text)
        if scraped_result
        else bool(_INADMISSIBLE_APPEAL_RE.search(text))
    )
    not_considered = (
        bool(re.search(r"განუხილველ(?:ად|ი)", text, re.I))
        if scraped_result
        else bool(_NOT_CONSIDERED_APPEAL_RE.search(text))
    )
    unchanged = bool(
        re.search(
            r"უცვლელ(?:ად|ი)?.{0,50}(?:დატოვ|დარჩ)|"
            r"(?:დატოვ|დარჩ).{0,50}უცვლელ(?:ად|ი)?",
            text,
            re.I | re.S,
        )
    )
    denied = bool(
        re.search(
            r"(?:საჩივარ|სარჩელ|შუამდგომლობ).{0,100}არ[ \t\n]+დაკმაყოფილ|"
            r"დაკმაყოფილ(?:ებ)?აზე.{0,60}უარი|"
            r"უარი.{0,60}დაკმაყოფილ",
            text,
            re.I | re.S,
        )
    )

    if inadmissible and appeal_context:
        outcomes.add("inadmissible")
    if not_considered and appeal_context:
        outcomes.add("not_considered")
    if unchanged or (denied and appeal_context):
        outcomes.add("upheld")
    if _has(DISPOSITION_STEMS["settled"], text):
        outcomes.add("settled")
    if _has(DISPOSITION_STEMS["terminated"], text) and (
        scraped_result or _PROCEEDINGS_RE.search(text)
    ):
        outcomes.add("terminated")

    cancelled = (
        _has(DISPOSITION_STEMS["cancel"], text)
        if scraped_result
        else _has_local_judgment_cancellation(text)
    )
    remand = (
        _has(DISPOSITION_STEMS["return"], text)
        if scraped_result
        else bool(_LOCAL_REMAND_RE.search(text))
    )
    partial = _has(DISPOSITION_STEMS["partial"], text)
    modified = (
        _has(DISPOSITION_STEMS["modified"], text)
        if scraped_result
        else bool(_LOCAL_MODIFIED_RE.search(text))
    )
    new_decision = _has(DISPOSITION_STEMS["new_decision"], text)

    # Grant/deny language describes the appeal, not necessarily what happened to the
    # reviewed judgment.  A change verb therefore owns the classification.
    partial_change = partial and (
        cancelled or _has(DISPOSITION_STEMS["granted"], text)
    )
    if partial_change:
        outcomes.add("partially_overturned")
    if cancelled and not partial_change:
        if remand:
            outcomes.add("overturned_remanded")
        elif new_decision:
            outcomes.add("overturned")
    if modified:
        outcomes.add("modified")
    elif remand and not cancelled:
        outcomes.add("remanded")
    if (
        _has(DISPOSITION_STEMS["granted"], text)
        and not denied
        and not (cancelled or modified or remand)
        and not partial_change
    ):
        outcomes.add("granted")
    return outcomes


def _select_outcome(outcomes: set[str]) -> tuple[str, str, bool]:
    if not outcomes:
        return "unknown", "low", False
    disposition = next(value for value in DISPOSITION_PRIORITY if value in outcomes)
    mixed = len(outcomes) > 1
    confidence = "low" if disposition == "granted" else "high"
    return disposition, confidence, mixed


def operative_header_matches(body: str) -> tuple[re.Match[str], ...]:
    """Return primary headers, or fallback headers only when no primary exists."""

    text = _nfc_text(body)
    primary = tuple(PRIMARY_OPERATIVE_HEADER_RE.finditer(text))
    return primary or tuple(FALLBACK_OPERATIVE_HEADER_RE.finditer(text))


def extract_disposition_from_body(body: str) -> DispositionResult:
    """Classify the last operative block of a decision body."""

    text = _nfc_text(body)
    # Fallback labels are considered only when no real operative verb exists.  Some
    # Supreme decisions contain a footer labelled ``სარეზოლუციო ნაწილი`` after their
    # actual ``დაადგინა`` block.
    headers = operative_header_matches(text)
    if not headers:
        return DispositionResult("unknown", "none", "low", False)
    header = headers[-1]
    block = text[header.start():]
    starts = [match.start() for match in _ENUMERATED_POINT_RE.finditer(block)]
    if starts:
        points = [
            block[start:end]
            for start, end in zip(starts, (*starts[1:], len(block)), strict=True)
        ]
    else:
        points = [block]
    outcomes: set[str] = set()
    for point in points:
        outcomes.update(_classify_text(point, scraped_result=False))
    disposition, confidence, mixed = _select_outcome(outcomes)
    return DispositionResult(
        disposition=disposition,
        disposition_source="body_operative",
        disposition_confidence=confidence,
        disposition_mixed=mixed,
        operative_start=header.start(),
        operative_end=len(text),
        operative_header=header.group("header"),
    )


def disposition_from_scraped_result(result_str: str) -> DispositionResult:
    """Map Supreme Court's structured result string through the same outcome table."""

    text = _nfc_text(result_str).strip()
    if not text:
        return DispositionResult("unknown", "scraped_result", "low", False)
    outcomes = _classify_text(text, scraped_result=True)
    disposition, confidence, mixed = _select_outcome(outcomes)
    return DispositionResult(
        disposition=disposition,
        disposition_source="scraped_result",
        disposition_confidence=confidence,
        disposition_mixed=mixed,
    )
