"""Language detection for cross-lingual query routing (KA → hybrid, EN → dense-only)."""

from ingest.search import detect_language


def test_georgian_query_is_ka():
    assert detect_language("რა არის ხელშეკრულების ვადა?") == "ka"


def test_english_query_is_en():
    assert detect_language("What is the term of the contract?") == "en"


def test_mixed_query_counts_as_ka():
    # any Mkhedruli → treat as Georgian (full hybrid); only pure-EN drops the sparse branch
    assert detect_language("define ხელშეკრულება term") == "ka"


def test_empty_and_symbols_default_to_en():
    assert detect_language("") == "en"
    assert detect_language("§ 123 / 2024") == "en"
