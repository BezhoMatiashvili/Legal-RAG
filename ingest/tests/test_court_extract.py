from ingest.court_extract import (
    EXTRACTOR_REVISION,
    disposition_from_scraped_result,
    extract_disposition_from_body,
    extract_judges,
    normalize_judge_key,
)


def test_judge_key_normalizes_names_initials_and_observed_case_suffix():
    assert EXTRACTOR_REVISION == "court-extract-v1"
    assert normalize_judge_key(" ლაშა   ქოჩიაშვილი ") == "ლ. ქოჩიაშვილი"
    assert normalize_judge_key("პ. სილაგაძე") == "პ. სილაგაძე"
    assert normalize_judge_key("შალვა კაკაურიძემ") == "შ. კაკაურიძე"


def test_extracts_observed_trailer_forms_and_deduplicates_sorted_keys():
    body = """
    თავმჯდომარე - მარინა სირაძე
    მოსამართლეეები: ლაშა ქოჩიაშვილი (თავმჯდომარე, მომხსენებელი),
    პ. სილაგაძე
    საბოლოოა
    """
    panel = extract_judges(body)
    assert panel.judges == ("ლ. ქოჩიაშვილი", "მ. სირაძე", "პ. სილაგაძე")
    assert panel.reporting_judge == "ლ. ქოჩიაშვილი"
    assert panel.confidence == "high"


def test_extracts_undelimited_singular_and_compound_reporter_label():
    body = """
    მოსამართლე შალვა კაკაურიძემ
    თავმჯდომარე, მომხსენებელი - გენადი მაკარიძე
    """
    panel = extract_judges(body)
    assert panel.judges == ("გ. მაკარიძე", "შ. კაკაურიძე")
    assert panel.reporting_judge == "გ. მაკარიძე"


def test_supreme_composition_roster_is_authoritative_over_narrative_markers():
    body = """
    სისხლის სამართლის პალატამ შემდეგი შემადგენლობით:
    ლევან თევზაძე (თავმჯდომარე),
    მერაბ გაბინაშვილი, ნინო სანდოძე
    განიხილა საქმე, რომელშიც მოსამართლემ კანონს თუმცა არასწორად მიუთითა.
    """
    panel = extract_judges(body)
    assert panel.judges == ("ლ. თევზაძე", "მ. გაბინაშვილი", "ნ. სანდოძე")
    assert panel.reporting_judge is None


def test_supreme_composition_supports_marker_line_and_reporter_parenthetical():
    body = """
    შემადგენლობა:
    ამირან ძაბუნიძე (თავმჯდომარე, მომხსენებელი),
    მოსამართლეები: ლაშა ქოჩიაშვილი,
    გოჩა ჯეირანაშვილი
    საქმის განხილვის ფორმა – ზეპირი მოსმენის გარეშე
    """
    panel = extract_judges(body)
    assert panel.judges == ("ა. ძაბუნიძე", "გ. ჯეირანაშვილი", "ლ. ქოჩიაშვილი")
    assert panel.reporting_judge == "ა. ძაბუნიძე"


def test_no_marker_is_an_explicit_low_confidence_empty_panel():
    panel = extract_judges("ლაშა ქოჩიაშვილი მონაწილეობდა სხდომაში")
    assert panel.judges == ()
    assert panel.reporting_judge is None
    assert panel.confidence == "low"


def test_last_spaced_header_wins_and_optional_colon_is_supported():
    body = """
    დ ა ა დ გ ი ნ ა: საკასაციო საჩივარი არ დაკმაყოფილდეს.
    მსჯელობის ტექსტი
    გ ა დ ა წ ყ ვ ი ტ ა
    1. გასაჩივრებული გადაწყვეტილება გაუქმდეს და მიღებულ იქნეს ახალი გადაწყვეტილება.
    """
    result = extract_disposition_from_body(body)
    assert result.disposition == "overturned"
    assert result.disposition_source == "body_operative"
    assert result.disposition_confidence == "high"
    assert body[result.operative_start :].startswith("გ ა დ ა წ ყ ვ ი ტ ა")


def test_legacy_latin_and_resolution_headers_are_supported():
    latin = extract_disposition_from_body(
        "d a a d g i n a\n1. საკასაციო საჩივარი არ დაკმაყოფილდეს."
    )
    resolution = extract_disposition_from_body(
        "სარეზოლუციო ნაწილი\nსაქმის წარმოება შეწყდეს."
    )
    assert latin.disposition == "upheld"
    assert resolution.disposition == "terminated"


def test_multi_appellant_priority_and_mixed_flag():
    body = """
    დ ა ა დ გ ი ნ ა:
    1. პირველი საკასაციო საჩივარი არ დაკმაყოფილდეს.
    2. მეორე საკასაციო საჩივარი დაკმაყოფილდეს ნაწილობრივ და გასაჩივრებული
       გადაწყვეტილება ნაწილობრივ გაუქმდეს.
    """
    result = extract_disposition_from_body(body)
    assert result.disposition == "partially_overturned"
    assert result.disposition_mixed is True


def test_polarity_trap_denial_means_lower_decision_upheld():
    result = extract_disposition_from_body(
        "გ ა დ ა წ ყ ვ ი ტ ა:\n1. საკასაციო საჩივარი არ დაკმაყოფილდეს."
    )
    assert result.disposition == "upheld"
    assert result.disposition_confidence == "high"


def test_bare_grant_is_low_confidence_but_change_verbs_are_specific():
    grant = disposition_from_scraped_result("საკასაციო საჩივარი დაკმაყოფილდა")
    remand = extract_disposition_from_body(
        "დ ა ა დ გ ი ნ ა:\n1. გასაჩივრებული გადაწყვეტილება გაუქმდეს და საქმე "
        "დაუბრუნდეს სააპელაციო სასამართლოს ხელახლა განსახილველად."
    )
    assert grant.disposition == "granted"
    assert grant.disposition_confidence == "low"
    assert remand.disposition == "overturned_remanded"


def test_ambiguous_verbs_require_legal_object_context():
    result = extract_disposition_from_body(
        "დ ა ა დ გ ი ნ ა:\n1. თანხა დაუბრუნდეს მოქალაქეს; რეგისტრაცია შეწყდეს; "
        "საცხოვრებელი შეზღუდვა შეიცვალოს."
    )
    assert result.disposition == "unknown"
    assert result.disposition_confidence == "low"


def test_scraped_result_maps_observed_reverse_word_orders():
    assert disposition_from_scraped_result("დატოვებულია განუხილველად").disposition == "not_considered"
    assert disposition_from_scraped_result("დატოვებულია უცვლელად").disposition == "upheld"
    assert (
        disposition_from_scraped_result(
            "საკასაციო საჩივარი ცნობილია დაუშვებლად"
        ).disposition
        == "inadmissible"
    )
    assert disposition_from_scraped_result("დაკმაყოფილდა ნაწილობრივ").disposition == "partially_overturned"

