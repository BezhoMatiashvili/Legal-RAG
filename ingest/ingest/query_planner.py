"""Deterministic intent, language, and entity planning for strict legal retrieval.

The planner is deliberately model-free.  It does not decide legal meaning; it decides
which retrieval branches are safe to run and which identifiers must survive translation
byte-for-byte.  Unrecognised languages fail closed instead of taking the old
"non-Georgian means English" route.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol
from collections.abc import Mapping

from .citations import CitationRef, extract_citation


class QueryLanguage(str, Enum):
    GEORGIAN = "ka"
    ENGLISH = "en"
    UNSUPPORTED = "unsupported"


class QueryIntent(str, Enum):
    EXACT_ARTICLE = "exact_article"
    EXACT_DOCUMENT = "exact_document"
    CASE_LOOKUP = "case_lookup"
    CASES_APPLYING_LAW = "cases_applying_law"
    CURRENT_LAW = "current_law"
    HISTORICAL = "historical"
    GENERAL_RESEARCH = "general_research"


class TranslationIntegrityError(ValueError):
    """Translation changed, removed, or duplicated a protected legal token."""


class Translator(Protocol):
    """Private translator contract used by the accuracy profile."""

    version: str

    def translate(self, text: str, *, source_language: str, target_language: str) -> str:
        """Translate only ``text`` and return plain target-language text."""


@dataclass(frozen=True)
class LegalEntityPlan:
    citation: CitationRef | None = None
    article_id: str | None = None
    article_marker: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    question: str
    language: QueryLanguage
    intent: QueryIntent
    entities: LegalEntityPlan
    as_of: str | None
    needs_translation: bool
    clarification_reason: str | None = None

    @property
    def answerable_language(self) -> bool:
        return self.language in {QueryLanguage.GEORGIAN, QueryLanguage.ENGLISH}


@dataclass(frozen=True)
class MaskedQuery:
    text: str
    replacements: tuple[tuple[str, str], ...]

    def restore(self, translated: str) -> str:
        restored = translated
        for placeholder, original in self.replacements:
            if restored.count(placeholder) != 1:
                raise TranslationIntegrityError(
                    f"protected token {placeholder} was changed or duplicated"
                )
            restored = restored.replace(placeholder, original)
        return restored


@dataclass(frozen=True)
class QueryVariant:
    text: str
    language: QueryLanguage
    track: str
    translator_version: str | None = None


class StaticMappingTranslator:
    """Deterministic private/eval translator backed by an exact checked-in mapping."""

    def __init__(self, translations: Mapping[str, str], *, version: str | None = None):
        masked: dict[str, str] = {}
        for source, target in translations.items():
            protected = mask_protected_tokens(source)
            target_masked = target
            for placeholder, original in protected.replacements:
                if original not in target_masked:
                    raise TranslationIntegrityError(
                        f"static translation did not preserve protected token {original!r}"
                    )
                target_masked = target_masked.replace(original, placeholder, 1)
            masked[protected.text] = target_masked
        self._translations = masked
        if version is None:
            material = json.dumps(
                dict(sorted(translations.items())),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            version = "static-map:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        self.version = version

    def translate(self, text: str, *, source_language: str, target_language: str) -> str:
        if (source_language, target_language) != ("en", "ka"):
            raise TranslationIntegrityError("static translator only supports en→ka")
        try:
            return self._translations[text]
        except KeyError as exc:
            raise TranslationIntegrityError("query is missing from static translation map") from exc


_GEORGIAN_RE = re.compile(r"[ა-ჿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_NON_ASCII_LETTER_RE = re.compile(r"[^\x00-\x7f]")

# A conservative discriminator for auto-detected ASCII input.  It is intentionally
# weighted toward legal/research vocabulary; callers can always provide language="en".
_ENGLISH_TERMS = {
    "a", "an", "and", "application", "are", "article", "case", "cases", "civil",
    "code", "complaint", "constitution", "contract", "court", "current", "damages",
    "decision", "decree", "does",
    "effective", "for", "from", "how", "in", "is", "law", "legal", "may", "of",
    "number", "on", "order", "permit", "registry", "right", "rule", "statute",
    "the", "this", "to", "under", "version",
    "was", "what", "when", "which", "who", "with",
}

_ARTICLE_RE = re.compile(
    r"(?:მუხლ(?:ი|ის|ში)?\s*(?:№\s*)?|\barticle\s*(?:no\.?\s*|§\s*)?)"
    r"(?P<article>\d+(?:\.\d+)*(?:\.[ა-ჿa-z])?)",
    re.IGNORECASE,
)
_CASES_APPLYING_RE = re.compile(
    r"(?:საქმე(?:ები|ებს)?[^\n]{0,80}(?:გამოიყენ|შეეხ|განმარტ)|"
    r"(?:cases?|decisions?)\s+(?:applying|interpreting|under|about))",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"(?:ამჟამად|დღეს|მოქმედი\s+(?:კანონ|რედაქცი)|ძალაშია|current(?:ly)?|in[- ]force|"
    r"law\s+now|as\s+of\s+today)",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"(?:იმ\s+დროისთვის|ისტორიულ|ყოფილ\s+რედაქცი|ძალაში\s+იყო|historical|formerly|"
    r"at\s+the\s+time|was\s+in\s+force|as\s+of\s+\d{4})",
    re.IGNORECASE,
)

# Ordered longest/most structured first.  Quoted names/titles and English proper-name
# sequences are included so MT cannot transliterate them and break entity lookup.
_PROTECTED_RE = re.compile(
    r"\b\d{9}\.\d{2}\.\d{3}\.\d{6}\b|"
    r"\b\d{18}\b|\b\d{15}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b|"
    r"(?i:\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b)|"
    r"(?i:\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b)|"
    r"(?:№\s*|\bN(?=\d))[0-9A-Za-zა-ჿ/()\-.]+|"
    r"\b[A-Z]{1,5}-?\d[0-9A-Za-z/()\-.]*\b|"
    r"\b(?:[ა-ჿ]{1,3}-|\d+[ა-ჿ]*/[ა-ჿ]*-?)\d[0-9ა-ჿ/\-]*(?:\([ა-ჿ]+-?\d+\))?|"
    r"(?:მუხლ(?:ი|ის|ში)?\s*|(?i:\barticle\s+))(?:№\s*)?\d+(?:\.\d+)*(?:\.[ა-ჿa-z])?|"
    r"[„“\"'][^„“\"'\n]{2,120}[„“\"']|"
    r"\b[A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){1,3}\b",
)


def detect_supported_language(text: str, declared: str | None = None) -> QueryLanguage:
    """Detect the two benchmarked languages, rejecting everything else.

    ASCII-only input cannot be identified perfectly without a language model.  The
    conservative lexical check avoids silently treating ordinary Spanish/French/etc. as
    English, while identifier-only queries remain usable for exact lookup.
    """

    if declared is not None:
        value = declared.strip().lower()
        if value in {"ka", "geo", "kat", "georgian"}:
            return QueryLanguage.GEORGIAN
        if value in {"en", "eng", "english"}:
            return QueryLanguage.ENGLISH
        return QueryLanguage.UNSUPPORTED

    value = (text or "").strip()
    if _GEORGIAN_RE.search(value):
        return QueryLanguage.GEORGIAN
    if _NON_ASCII_LETTER_RE.search(value):
        return QueryLanguage.UNSUPPORTED
    if not _LATIN_RE.search(value):
        # Exact identifier/date queries need no natural-language translation.
        return QueryLanguage.ENGLISH if re.search(r"\d", value) else QueryLanguage.UNSUPPORTED
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]+", value)}
    if tokens & _ENGLISH_TERMS:
        return QueryLanguage.ENGLISH
    return QueryLanguage.UNSUPPORTED


def _validated_as_of(as_of: str | None) -> str | None:
    if as_of is None:
        return None
    try:
        return date.fromisoformat(as_of).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of must be an ISO date (YYYY-MM-DD)") from exc


def plan_query(
    question: str,
    *,
    language: str | None = None,
    as_of: str | None = None,
) -> QueryPlan:
    """Return a deterministic plan for entity-first and semantic retrieval branches."""

    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    citation = extract_citation(text, mode="full")
    detected = detect_supported_language(text, language)
    # Identifier-only lookups need no language model.  Preserve them even when the
    # identifier contains Latin letters (for example TAS ``AR111390``).
    identifier_only = False
    if citation is not None:
        remainder = text.replace(citation.matched, "", 1)
        identifier_only = not re.search(r"[A-Za-zა-ჿ]", remainder)
        if language is None and identifier_only and detected is QueryLanguage.UNSUPPORTED:
            detected = QueryLanguage.ENGLISH
    effective_as_of = _validated_as_of(as_of)
    article_match = _ARTICLE_RE.search(text)
    article = article_match.group("article") if article_match else None

    if detected is QueryLanguage.UNSUPPORTED:
        return QueryPlan(
            question=text,
            language=detected,
            intent=QueryIntent.GENERAL_RESEARCH,
            entities=LegalEntityPlan(citation=citation, article_id=article),
            as_of=effective_as_of,
            needs_translation=False,
            clarification_reason="unsupported_language",
        )

    if _CASES_APPLYING_RE.search(text):
        intent = QueryIntent.CASES_APPLYING_LAW
    elif citation is not None and citation.kind == "case_number":
        intent = QueryIntent.CASE_LOOKUP
    elif article is not None:
        intent = QueryIntent.EXACT_ARTICLE
    elif citation is not None:
        intent = QueryIntent.EXACT_DOCUMENT
    elif effective_as_of is not None or _HISTORICAL_RE.search(text):
        intent = QueryIntent.HISTORICAL
    elif _CURRENT_RE.search(text):
        intent = QueryIntent.CURRENT_LAW
    else:
        intent = QueryIntent.GENERAL_RESEARCH

    return QueryPlan(
        question=text,
        language=detected,
        intent=intent,
        entities=LegalEntityPlan(
            citation=citation,
            article_id=article,
            article_marker=article_match.group(0) if article_match else None,
        ),
        as_of=effective_as_of,
        needs_translation=(
            detected is QueryLanguage.ENGLISH
            and not identifier_only
            and bool(_LATIN_RE.search(text))
        ),
    )


def mask_protected_tokens(text: str) -> MaskedQuery:
    """Replace identifiers, dates, article references, and names with stable tokens."""

    replacements: list[tuple[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        # Translate the article word but preserve only its legal identifier.
        if re.match(r"(?:მუხლ|(?i:article))", original):
            identifier = re.search(r"\d+(?:\.\d+)*(?:\.[ა-ჿa-z])?", original)
            if identifier is not None:
                placeholder = f"__LEGAL_{len(replacements)}__"
                replacements.append((placeholder, identifier.group(0)))
                return (
                    original[: identifier.start()]
                    + placeholder
                    + original[identifier.end() :]
                )
        # Capitalized English legal titles need translation; treating "Civil Code" as a
        # person's name would preserve the very English tokens the Georgian branch is
        # intended to replace.
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", original):
            words = {word.lower() for word in original.split()}
            if words & {"act", "code", "constitution", "court", "law", "statute"}:
                return original
        placeholder = f"__LEGAL_{len(replacements)}__"
        replacements.append((placeholder, original))
        return placeholder

    return MaskedQuery(_PROTECTED_RE.sub(replace, text), tuple(replacements))


def build_query_variants(plan: QueryPlan, translator: Translator | None) -> tuple[QueryVariant, ...]:
    """Build the original+Georgian paths, enforcing translation token integrity."""

    if not plan.answerable_language:
        return ()
    original = QueryVariant(plan.question, plan.language, "original")
    if not plan.needs_translation:
        return (original,)
    if translator is None:
        raise TranslationIntegrityError("English strict retrieval requires a private translator")
    masked = mask_protected_tokens(plan.question)
    translated = translator.translate(
        masked.text,
        source_language=QueryLanguage.ENGLISH.value,
        target_language=QueryLanguage.GEORGIAN.value,
    )
    translated = masked.restore((translated or "").strip())
    if not _GEORGIAN_RE.search(translated):
        raise TranslationIntegrityError("translator did not return Georgian text")
    return (
        original,
        QueryVariant(
            translated,
            QueryLanguage.GEORGIAN,
            "translated_ka",
            getattr(translator, "version", None),
        ),
    )
