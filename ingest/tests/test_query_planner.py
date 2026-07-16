import pytest

from ingest.query_planner import (
    QueryIntent,
    QueryLanguage,
    TranslationIntegrityError,
    StaticMappingTranslator,
    build_query_variants,
    detect_supported_language,
    mask_protected_tokens,
    plan_query,
)


class _Translator:
    version = "translator-rev-1"

    def translate(self, text, *, source_language, target_language):
        assert (source_language, target_language) == ("en", "ka")
        return f"რას ამბობს {text}"


def test_language_detection_is_fail_closed_for_unsupported_text():
    assert detect_supported_language("რა ამბობს კანონი") is QueryLanguage.GEORGIAN
    assert detect_supported_language("What does the current law say?") is QueryLanguage.ENGLISH
    assert detect_supported_language("cual es la ley") is QueryLanguage.UNSUPPORTED
    assert detect_supported_language("что говорит закон") is QueryLanguage.UNSUPPORTED
    assert detect_supported_language("12345") is QueryLanguage.ENGLISH
    assert detect_supported_language("bonjour", declared="fr") is QueryLanguage.UNSUPPORTED


def test_planner_distinguishes_article_case_application_and_temporal_intent():
    exact = plan_query("სამოქალაქო კოდექსის მუხლი 829")
    assert exact.intent is QueryIntent.EXACT_ARTICLE
    assert exact.entities.article_id == "829"

    applying = plan_query("cases applying Civil Code article 829", language="en")
    assert applying.intent is QueryIntent.CASES_APPLYING_LAW

    historical = plan_query("What was the rule?", language="en", as_of="2020-05-01")
    assert historical.intent is QueryIntent.HISTORICAL
    assert historical.as_of == "2020-05-01"


def test_unsupported_language_produces_clarification_plan():
    plan = plan_query("cual es la ley")
    assert not plan.answerable_language
    assert plan.clarification_reason == "unsupported_language"
    assert build_query_variants(plan, None) == ()


def test_identifier_only_lookup_does_not_require_translation():
    plan = plan_query("AR111390")
    assert plan.intent is QueryIntent.EXACT_DOCUMENT
    assert plan.answerable_language
    assert not plan.needs_translation
    assert [variant.track for variant in build_query_variants(plan, None)] == ["original"]


def test_translation_masks_and_restores_legal_identifiers_dates_and_names():
    query = "Civil Code article 829 in John Smith case N01/902 on 2024-01-02"
    masked = mask_protected_tokens(query)
    assert "article 829" not in masked.text
    assert "John Smith" not in masked.text
    assert "Civil Code" in masked.text
    assert "N01/902" not in masked.text
    plan = plan_query(query, language="en")
    variants = build_query_variants(plan, _Translator())
    assert [v.track for v in variants] == ["original", "translated_ka"]
    assert "article 829" in variants[1].text
    assert "John Smith" in variants[1].text
    assert "N01/902" in variants[1].text
    assert variants[1].translator_version == "translator-rev-1"


def test_translation_protects_word_dates_and_latin_case_numbers():
    plan = plan_query(
        "What did case C-123/2020 decide on July 15, 2020?",
        language="en",
    )
    assert plan.intent is QueryIntent.CASE_LOOKUP
    assert plan.entities.citation is not None
    assert plan.entities.citation.filters == {"document_number": "C-123/2020"}
    masked = mask_protected_tokens(plan.question)
    originals = {original for _placeholder, original in masked.replacements}
    assert {"C-123/2020", "July 15, 2020"} <= originals


def test_translation_fails_if_a_protected_token_is_changed():
    class Bad(_Translator):
        def translate(self, text, **kwargs):
            return "ქართული ტექსტი " + text.replace("__LEGAL_0__", "")

    with pytest.raises(TranslationIntegrityError, match="changed or duplicated"):
        build_query_variants(plan_query("Civil Code article 829", language="en"), Bad())


def test_invalid_as_of_is_rejected_before_retrieval():
    with pytest.raises(ValueError, match="ISO date"):
        plan_query("რა ამბობს კანონი", as_of="2024/01/02")


def test_static_mapping_translator_is_deterministic_and_preserves_identifiers():
    source = "What does Civil Code article 829 say in John Smith case N01/902?"
    target = "რას ამბობს სამოქალაქო კოდექსის მუხლი 829 John Smith-ის საქმეში N01/902?"
    translator = StaticMappingTranslator({source: target}, version="authored-v2")
    variants = build_query_variants(plan_query(source, language="en"), translator)
    assert variants[-1].text == target
    assert variants[-1].translator_version == "authored-v2"
