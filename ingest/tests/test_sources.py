import pytest

from ingest.sources import SOURCES, _parse_date, normalize, normalize_status


@pytest.mark.parametrize(
    "raw,iso",
    [
        ("2020-04-27", "2020-04-27"),          # ISO (ecd/napr/supremecourt)
        ("2024-06-03 08:00:07", "2024-06-03"),  # ISO datetime (tas registration_date)
        ("08/04/2026", "2026-04-08"),          # DD/MM/YYYY day-first (matsne, order №55)
        ("03/06/2024", "2024-06-03"),          # DD/MM/YYYY (tas create_date)
        ("19-05-2017", "2017-05-19"),          # DD-MM-YYYY (tbappeal)
        ("26 მარტი 2026", "2026-03-26"),       # Georgian month (constcourt)
        ("26 მარტის 2026 17:55", "2026-03-26"),  # Georgian genitive + time
        ("1 იანვარი 2020", "2020-01-01"),
        ("31 დეკემბერი 2025", "2025-12-31"),
        ("garbage", None),
        ("", None),
        (None, None),
        ("32/01/2020", None),                  # invalid day -> None
    ],
)
def test_parse_date_formats(raw, iso):
    assert _parse_date(raw) == iso


def test_matsne_number_and_registration_and_date():
    doc = normalize("matsne", {
        "document_id": "6835981", "document_number": "55",
        "registration_code": "140130000.22.034.017712",
        "document_recipient": "საქართველოს შინაგან საქმეთა მინისტრი",
        "document_type": "საქართველოს მინისტრის ბრძანება",
        "publication_date": "08/04/2026", "document_url": "u", "body_markdown": "x",
    })
    assert doc.document_number == "55"
    assert doc.registration_code == "140130000.22.034.017712"
    assert doc.parties == "საქართველოს შინაგან საქმეთა მინისტრი"
    assert doc.date == "2026-04-08"          # ISO (sortable)
    assert doc.date_raw == "08/04/2026"      # original preserved


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("ძალაში მყოფი აქტები", "in_force"),
        ("ძალადაკარგული აქტები", "repealed"),
        ("ასამოქმედებელი აქტები", "pending"),
        ("", None),
        (None, None),
        ("რაღაც უცნობი", None),  # unknown -> None (don't pollute the keyword index)
    ],
)
def test_normalize_status(raw, canonical):
    assert normalize_status(raw) == canonical


def test_matsne_status_and_force_dates_indexed():
    doc = normalize("matsne", {
        "document_id": "111", "title": "L", "document_url": "u", "body_markdown": "x",
        "status": "ძალადაკარგული აქტები",
        "entry_into_force_date": "01/01/2020", "expiry_date": "2024-06-03",
    })
    assert doc.status == "repealed"
    assert doc.status_raw == "ძალადაკარგული აქტები"
    assert doc.in_force_date == "2020-01-01"
    assert doc.expiry_date == "2024-06-03"


def test_non_matsne_sources_have_no_status():
    doc = normalize("ecd", {
        "decision_document_id": 5, "case_no": "X", "decision_date": "2020-04-30",
        "body_markdown": "x",
    })
    assert doc.status is None
    assert doc.in_force_date is None


def test_constcourt_parties_split_from_title():
    doc = normalize("constcourt", {
        "legal_id": "19474", "number": "N3/1/1914",
        "title": "შორენა წიკლაური საქართველოს იუსტიციის მინისტრის წინააღმდეგ.",
        "doc_type": "გადაწყვეტილება", "date": "26 მარტი 2026", "body_markdown": "x",
    })
    assert doc.document_number == "N3/1/1914"
    assert doc.parties == "შორენა წიკლაური საქართველოს იუსტიციის მინისტრის"
    assert doc.date == "2026-03-26"


def test_napr_number_app_no_and_sender():
    doc = normalize("napr", {
        "document_id": "95238", "decision_no": "131383", "app_no": "011791088432826",
        "sender": ": 1. ***** ს", "title": "გადაწყვეტილება", "date": "2026-06-29",
        "body_markdown": "x",
    })
    assert doc.document_number == "131383"
    assert doc.registration_code == "011791088432826"
    assert doc.parties == ": 1. ***** ს"


def test_supremecourt_number_and_title_fallback():
    # subject present -> title combines subject + case_number
    doc = normalize("supremecourt", {
        "case_id": "35513", "chamber": "0", "case_number": "ბს-174(კს-26)",
        "subject": "ადმინისტრაციული აქტი", "date": "2026-03-05", "body_markdown": "x",
    })
    assert doc.document_number == "ბს-174(კს-26)"
    assert "ბს-174(კს-26)" in doc.title
    # subject missing -> falls back to case_number alone (no empty title)
    doc2 = normalize("supremecourt", {
        "case_id": "1", "chamber": "0", "case_number": "ბს-1", "date": "2026-03-05",
        "body_markdown": "x",
    })
    assert doc2.title == "ბს-1"


def test_tas_number():
    doc = normalize("tas", {
        "document_id": "1039800", "document_no": "AR11039800",
        "nomenclature": "ცვლილება", "registration_date": "2024-06-03 08:00:07",
        "body_markdown": "x",
    })
    assert doc.document_number == "AR11039800"
    assert doc.date == "2024-06-03"


def test_tbappeal_has_no_number():
    doc = normalize("tbappeal", {
        "slug": "abc", "title": "T", "date": "19-05-2017", "body_markdown": "x",
    })
    assert doc.document_number is None
    assert doc.date == "2017-05-19"


def test_ecd_mapping_joins_title_and_reads_dynamic_court():
    item = {
        "decision_document_id": 5861405,
        "case_no": "330100119003015732",
        "decision_type_name": "განაჩენი",
        "court_name": "თბილისის საქალაქო სასამართლო",
        "decision_date": "2020-04-30",
        "body_markdown": "ტექსტი",
    }
    doc = normalize("ecd", item)
    assert doc.source == "ecd"
    assert doc.document_id == "5861405"
    assert doc.title == "330100119003015732 — განაჩენი"
    assert doc.court == "თბილისის საქალაქო სასამართლო"
    assert doc.document_type == "court_decision"
    assert doc.language == "ka"
    assert doc.date == "2020-04-30"


def test_supremecourt_composite_id():
    doc = normalize("supremecourt", {
        "case_id": "73901", "chamber": "1", "subject": "დავა", "date": "2024-12-26",
        "body_markdown": "x",
    })
    assert doc.document_id == "73901:1"
    assert doc.title == "დავა"
    assert doc.court == "supremecourt"


def test_matsne_uses_language_field_and_dynamic_doc_type_and_trims_datetime():
    doc = normalize("matsne", {
        "document_id": "1234", "language": "en", "document_type": "კანონი",
        "title": "Law", "publication_date": "2026-06-22 00:00:00", "document_url": "u",
        "body_markdown": "x",
    })
    assert doc.language == "en"
    assert doc.document_type == "კანონი"
    assert doc.date == "2026-06-22"
    assert doc.source_url == "u"


def test_default_language_is_georgian():
    doc = normalize("constcourt", {"legal_id": "20024", "title": "T", "date": "2026-06-24", "body_markdown": "x"})
    assert doc.language == "ka"
    assert doc.document_id == "20024"


def test_missing_id_raises():
    with pytest.raises(ValueError):
        normalize("ecd", {"body_markdown": "x"})


def test_all_spiders_have_specs():
    assert set(SOURCES) == {"matsne", "ecd", "constcourt", "napr", "tbappeal", "supremecourt", "tas"}
