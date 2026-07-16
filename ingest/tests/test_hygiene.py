"""Tests for corpus hygiene: control-char stripping, NFC, damage classification.

Georgian text is used throughout because the cleaning must be safe on Mkhedruli and must
never remove personal data — only structural junk.
"""

import unicodedata

from ingest import hygiene


def test_strip_control_removes_nul_and_controls_keeps_whitespace():
    raw = "მუხლი\x001\x07 ტექსტი\tსვეტი\nხაზი\rდაბრუნება"
    cleaned, removed = hygiene.strip_control(raw)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert removed == 2
    # tab / newline / carriage return are preserved
    assert "\t" in cleaned and "\n" in cleaned and "\r" in cleaned
    assert "მუხლი" in cleaned and "ტექსტი" in cleaned


def test_clean_text_is_idempotent():
    raw = "საქართველოს\x00 კანონი\x1f — ნაწილი 2"
    once = hygiene.clean_text(raw)
    assert hygiene.clean_text(once) == once
    assert "\x00" not in once


def test_nfc_normalisation_idempotent_and_round_trip():
    # A decomposed sequence NFC-composes; applying NFC twice is a no-op.
    decomposed = unicodedata.normalize("NFD", "café ქართული")
    nfc = hygiene.to_nfc(decomposed)
    assert nfc == unicodedata.normalize("NFC", decomposed)
    assert hygiene.to_nfc(nfc) == nfc


def test_clean_text_preserves_personal_data():
    # Synthetic (not real) PII: a national-id-like number, phone, and a Georgian name.
    body = "განმცხადებელი: გიორგი მაისურაძე, პ/ნ 01001000000, ტელ 599123456"
    cleaned = hygiene.clean_text(body)
    assert "გიორგი მაისურაძე" in cleaned
    assert "01001000000" in cleaned
    assert "599123456" in cleaned


def test_clean_text_preserves_georgian_punctuation_and_mixed_scripts():
    body = "მუხლი 1. Term “X” — see § 2; ე.წ. „ბრჭყალები“ და ASCII text."
    cleaned = hygiene.clean_text(body)
    assert cleaned == hygiene.to_nfc(body)  # nothing removed, only normalised
    assert "„ბრჭყალები“" in cleaned and "ASCII text" in cleaned


def test_assess_flags_empty_and_near_empty():
    assert hygiene.assess("").quarantine_reason == hygiene.Q_EMPTY
    assert hygiene.assess("   \n\t ").quarantine_reason == hygiene.Q_EMPTY
    assert hygiene.assess("მუხლი 1").quarantine_reason == hygiene.Q_NEAR_EMPTY


def test_assess_flags_mojibake_by_ratio():
    corrupt = "ტექსტი " + ("�" * 50)  # >1% replacement chars
    r = hygiene.assess(corrupt)
    assert r.quarantine_reason == hygiene.Q_MOJIBAKE
    assert r.replacement_chars == 50


def test_assess_keeps_nul_heavy_but_otherwise_good_body():
    # A napr-style body: full of NUL bytes but real content — clean it, do NOT quarantine.
    body = "დადგენილება\x00 № 123.\x00 " + "საქმის განხილვის შედეგად დადგინდა შემდეგი. " * 5
    r = hygiene.assess(body)
    assert r.nul_chars >= 2
    assert r.control_chars >= 2
    assert r.is_usable  # cleaned, not quarantined
    assert "\x00" not in hygiene.clean_text(body)


def test_assess_gate_measured_on_cleaned_text_not_raw():
    # Regression: control padding must not count as "meaningful". A body of mostly NUL
    # with < NEAR_EMPTY_CHARS of real content must quarantine as near_empty, not sneak in.
    r = hygiene.assess("\x00" * 60 + "მუხლი")  # 5 real chars after stripping
    assert r.quarantine_reason == hygiene.Q_NEAR_EMPTY
    # All-control body → empty after cleaning → Q_EMPTY (not "usable").
    assert hygiene.assess("\x00" * 100).quarantine_reason == hygiene.Q_EMPTY
    # Stripped down to a few real chars + mojibake → near_empty (still quarantined).
    assert hygiene.assess("ტექსტი " + "�" * 4 + "\x00" * 400).quarantine_reason == hygiene.Q_NEAR_EMPTY
    # Mojibake ratio is measured on surviving text: enough real content to pass near_empty,
    # but >1% U+FFFD after stripping NUL padding → still quarantined as mojibake (not diluted).
    long_ka = "ქართული სამართლებრივი ტექსტი საკმაოდ გრძელია აქ."  # 42 non-space chars
    assert hygiene.assess(long_ka + "�" * 3 + "\x00" * 400).quarantine_reason == hygiene.Q_MOJIBAKE


def test_strip_data_uris_replaces_image_and_counts():
    blob = "data:image/jpeg;base64," + "A" * 500 + "=="
    body = f"მუხლი 1. იხილეთ ილუსტრაცია: {blob} შემდეგი ტექსტი."
    cleaned, count = hygiene.strip_data_uris(body)
    assert count == 1
    assert "[image]" in cleaned
    assert blob not in cleaned
    assert "A" * 20 not in cleaned  # payload itself is gone, not just truncated
    assert "მუხლი 1" in cleaned and "შემდეგი ტექსტი" in cleaned


def test_strip_data_uris_leaves_non_data_uri_text_untouched():
    body = "მუხლი 1. ჩვეულებრივი ტექსტი base64 სიტყვის ხსენებით, არა URI."
    cleaned, count = hygiene.strip_data_uris(body)
    assert count == 0
    assert cleaned == body


def test_clean_text_strips_data_uri_and_preserves_surrounding_text():
    blob = "data:image/png;base64," + "QUJDREVGRw==" * 40
    body = f"დადგენილება № 5.\x00 {blob} დასკვნა."
    cleaned = hygiene.clean_text(body)
    assert "[image]" in cleaned
    assert blob not in cleaned
    assert "\x00" not in cleaned
    assert "დადგენილება № 5." in cleaned and "დასკვნა." in cleaned


def test_clean_text_data_uri_stripping_is_idempotent():
    blob = "data:image/jpeg;base64," + "Zm9vYmFy" * 30
    body = f"ტექსტი {blob} დასასრული"
    once = hygiene.clean_text(body)
    assert hygiene.clean_text(once) == once


def test_assess_measures_meaningful_chars_after_data_uri_strip():
    # A body that is almost entirely a base64 blob with only a few real chars must not
    # be counted as "meaningful" via the blob's characters — assess() must measure on
    # what clean_text actually leaves in the index (a short [image] marker), matching
    # the module's own "measured on cleaned text" invariant for control/mojibake gates.
    blob = "data:image/png;base64," + "Q" * 2000
    r = hygiene.assess(blob + " მუხლი")
    assert r.meaningful_chars < 30  # "[image] მუხლი" — well under NEAR_EMPTY_CHARS
    assert r.quarantine_reason == hygiene.Q_NEAR_EMPTY


def test_assess_reports_nfc_change():
    decomposed = unicodedata.normalize("NFD", "ჩ") + "test text long enough to be usable content here"
    r = hygiene.assess(decomposed)
    # Georgian has no canonical decomposition, so use a Latin accent to force the change.
    accented = unicodedata.normalize("NFD", "café") + " some usable body text here please"
    assert hygiene.assess(accented).changed_by_nfc is True
    assert r.is_usable
