"""Tests for legal-structure detection on Georgian legal bodies."""

from ingest import structure


def test_detects_articles_and_extracts_spans():
    body = "შესავალი ტექსტი.\nმუხლი 1. პირველი მუხლი.\nდებულება.\nმუხლი 2. მეორე მუხლი."
    info = structure.detect(body)
    assert info.has_article
    assert info.article_count == 2
    assert info.primary_kind == structure.KIND_ARTICLE

    spans = structure.article_spans(body)
    assert [label.split(".")[0] for _, label in spans] == ["მუხლი 1", "მუხლი 2"]
    # Offsets point at the start of each article heading.
    for offset, label in spans:
        assert body[offset:].startswith(label.split(".")[0])


def test_numbered_clause_body_is_clause_kind():
    body = "1. პირველი პუნქტი.\n2. მეორე პუნქტი.\n3) მესამე პუნქტი."
    info = structure.detect(body)
    assert info.has_num_clause
    assert not info.has_article
    assert info.primary_kind == structure.KIND_CLAUSE


def test_markdown_heading_body_is_heading_kind():
    body = "# სათაური\n\nპარაგრაფი ტექსტით.\n\n## ქვესათაური\n\nკიდევ ტექსტი."
    info = structure.detect(body)
    assert info.has_heading
    assert not info.has_article
    assert info.primary_kind == structure.KIND_HEADING


def test_flat_prose_has_no_structure():
    body = "ეს არის უბრალო ტექსტი ყოველგვარი სტრუქტურის გარეშე, მხოლოდ წინადადებები."
    info = structure.detect(body)
    assert info.primary_kind == structure.KIND_FLAT
    assert not (info.has_article or info.has_heading or info.has_num_clause)


def test_chapter_marker_requires_a_numeral():
    # Bare "ნაწილი" ("part") in prose must NOT count as a chapter heading.
    assert not structure.detect("ეს ხელშეკრულების ნაწილი მოქმედებს.").has_chapter
    assert structure.detect("თავი 1\nდებულებანი").has_chapter


def test_article_marker_does_not_cross_newline():
    # Regression: a bare line-final "მუხლი" followed by a digit-initial next line must NOT
    # be read as "მუხლი <N>" (the article regex must use same-line whitespace, not \s).
    body = "ტექსტი აქ.\nმუხლი\n2019 წელს დამტკიცდა."
    info = structure.detect(body)
    assert not info.has_article
    assert info.article_count == 0
    assert structure.article_spans(body) == []
    # A real same-line article still matches.
    assert structure.detect("მუხლი 7. დებულება.").has_article


def test_article_kind_wins_over_clause_and_heading():
    body = "# სათაური\nმუხლი 1. დებულება.\n1. პუნქტი."
    assert structure.detect(body).primary_kind == structure.KIND_ARTICLE
