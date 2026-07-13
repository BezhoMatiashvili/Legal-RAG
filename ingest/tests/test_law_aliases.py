"""Static precision checks for the corpus-derived citation alias table."""

import json
import unicodedata
from pathlib import Path

from ingest.citations import extract_citation, load_aliases


ALIAS_PATH = Path(__file__).parents[1] / "ingest" / "data" / "law_aliases.json"


def _laws() -> list[dict]:
    return json.loads(ALIAS_PATH.read_text(encoding="utf-8"))["laws"]


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFC", value).lower()
    return " ".join(value.translate(str.maketrans("", "", "„“”«»\"'")).split())


def test_alias_table_uses_the_loader_schema_and_effective_targets():
    laws = _laws()
    assert len(laws) >= 40
    assert load_aliases() == laws

    titles = set()
    document_targets = set()
    registration_codes = set()
    for entry in laws:
        assert set(entry) == {
            "canonical_title",
            "source",
            "document_id",
            "registration_code",
            "aliases",
        }
        assert entry["source"] == "matsne"
        assert entry["document_id"].strip()
        assert entry["registration_code"].strip()
        assert entry["aliases"]
        assert "filters" not in entry and "document_number" not in entry

        title = _norm(entry["canonical_title"])
        document_target = (entry["source"], entry["document_id"])
        registration_code = entry["registration_code"]
        assert title not in titles
        assert document_target not in document_targets
        assert registration_code not in registration_codes
        titles.add(title)
        document_targets.add(document_target)
        registration_codes.add(registration_code)


def test_aliases_are_trimmed_nfc_and_unique_across_targets():
    seen: dict[str, str] = {}
    for entry in _laws():
        for alias in entry["aliases"]:
            assert alias == alias.strip()
            assert alias == unicodedata.normalize("NFC", alias)
            normalized = _norm(alias)
            assert len(normalized) >= 8
            assert normalized not in seen, (
                f"alias {alias!r} targets both {seen[normalized]!r} "
                f"and {entry['canonical_title']!r}"
            )
            seen[normalized] = entry["canonical_title"]


def test_every_alias_resolves_to_its_declared_registration_code():
    laws = _laws()
    for entry in laws:
        expected = {"registration_code": entry["registration_code"]}
        for alias in entry["aliases"]:
            ref = extract_citation(alias, mode="full", aliases=laws)
            assert ref is not None, (entry["canonical_title"], alias)
            assert ref.kind == "law_alias"
            assert ref.filters == expected


def test_handoff_named_law_examples_resolve_to_current_acts():
    laws = _laws()
    civil = extract_citation("Civil Code art. 829", mode="full", aliases=laws)
    entrepreneurs = extract_citation(
        "Law on Entrepreneurs 189.5.d", mode="full", aliases=laws
    )

    assert civil is not None
    assert civil.filters == {"registration_code": "040.000.000.05.001.000.223"}
    assert entrepreneurs is not None
    assert entrepreneurs.filters == {"registration_code": "240000000.05.001.020373"}


def test_common_georgian_genitive_code_references_resolve():
    laws = _laws()
    cases = {
        "სამოქალაქო კოდექსის 829-ე მუხლი": "040.000.000.05.001.000.223",
        "სისხლის სამართლის კოდექსის 126-ე მუხლი": "080.000.000.05.001.000.648",
        "შრომის კოდექსის 31-ე მუხლი": "270000000.04.001.016012",
    }
    for query, registration_code in cases.items():
        ref = extract_citation(query, mode="full", aliases=laws)
        assert ref is not None
        assert ref.filters == {"registration_code": registration_code}


def test_known_ambiguous_or_repealed_shortcuts_are_not_present():
    aliases = {_norm(alias) for entry in _laws() for alias in entry["aliases"]}
    assert "კონსტიტუცია" not in aliases
    assert "ნორმატიული აქტების შესახებ" not in aliases
    assert "დაზღვევის შესახებ" not in aliases
    assert "სსკ" not in aliases
    assert "ასკ" not in aliases
