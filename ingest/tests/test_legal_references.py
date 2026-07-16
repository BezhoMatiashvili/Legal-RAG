from ingest.legal_references import extract_article_references


def test_extracts_unique_cross_references_and_excludes_own_article():
    text = (
        "მუხლი 5. ამ კანონის 2-ე მუხლით განსაზღვრული პირი მოქმედებს "
        "მე-7 მუხლის შესაბამისად. ამ კანონის 2-ე მუხლი მეორდება."
    )
    assert extract_article_references(text, own_article_id="5") == ("2", "7")


def test_plain_article_heading_is_not_mistaken_for_georgian_reference():
    assert extract_article_references("მუხლი 12. ზოგადი დებულება", own_article_id="12") == ()


def test_english_reference_is_supported():
    assert extract_article_references("Subject to article 9 of this Code") == ("9",)
