"""Unit tests for citation detection/extraction (improvement I1) — pure functions."""

from types import SimpleNamespace

from ingest.citations import (
    CitationRef,
    citation_lookup,
    extract_citation,
    pin_points,
)


# --- extraction: identifier classes -------------------------------------------------


def test_extract_matsne_registration_code():
    ref = extract_citation("240140000.05.001.102038 რეგისტრირებული აქტი")
    assert ref == CitationRef(
        "registration_code", {"registration_code": "240140000.05.001.102038"},
        "240140000.05.001.102038")


def test_extract_ecd_18_digit_case_number_before_napr_15():
    ref = extract_citation("საქმე 330100116001532776 განაჩენი")
    assert ref.kind == "document_number"
    assert ref.filters == {"document_number": "330100116001532776"}


def test_extract_napr_15_digit_registration_code():
    ref = extract_citation("გადაწყვეტილება 011746498351723")
    assert ref.kind == "registration_code"
    assert ref.filters == {"registration_code": "011746498351723"}


def test_extract_tas_permit_id():
    ref = extract_citation("ნებართვა AR111390 მშენებლობა")
    assert ref.filters == {"document_number": "AR111390"}


def test_extract_act_number_with_context():
    ref = extract_citation("შსს მინისტრის 2019 წლის 16 აგვისტოს №71 ბრძანებაში ცვლილება")
    assert ref.kind == "document_number"
    assert ref.filters == {"document_number": "71"}


def test_extract_government_decree_number():
    ref = extract_citation("მთავრობის №124 დადგენილება სამხედრო სადისციპლინო წესდების ცვლილება")
    assert ref.filters == {"document_number": "124"}


def test_extract_latin_n_slash_number():
    ref = extract_citation("N01/902 ბრძანება ორი შენობის კომპლექსური შეთანხმება")
    assert ref.filters == {"document_number": "01/902"}


def test_extract_case_number_with_georgian_letters():
    ref = extract_citation("განჩინება №1გ/620-17 ქონებაზე ყადაღის დადება")
    assert ref.kind == "case_number"
    assert ref.filters == {"document_number": "1გ/620-17"}


def test_extract_case_number_slash_letter_form():
    ref = extract_citation("თბილისის სააპელაციო სასამართლოს განაჩენი 1/ბ-167-17 ნარკოტიკის საქმეზე")
    assert ref.kind == "case_number"
    assert ref.filters == {"document_number": "1/ბ-167-17"}


def test_extract_supreme_court_case_number():
    ref = extract_citation("უზენაესი სასამართლოს საქმე № ბს-729-721(კ-16) რას ადგენს?")
    assert ref.kind == "case_number"
    assert ref.filters == {"document_number": "ბს-729-721(კ-16)"}


def test_extract_context_gated_latin_case_number():
    ref = extract_citation("What did case C-123/2020 decide?")
    assert ref.kind == "case_number"
    assert ref.filters == {"document_number": "C-123/2020"}


# --- extraction: false-positive guards ----------------------------------------------


def test_bare_number_without_act_context_is_none():
    assert extract_citation("2019 წლის 16 აგვისტოს შეხვედრა №5 კორპუსში") is None


def test_article_ordinal_is_not_a_document_number():
    assert extract_citation("სისხლის სამართლის საპროცესო კოდექსის 200-ე მუხლი გირაო") is None
    assert extract_citation("კანონის მე-3 მუხლი სამხედრო ფიცი") is None


def test_dotted_article_reference_is_none():
    assert extract_citation("საჯარო რეესტრის შესახებ კანონის 19.6 მუხლი შეზღუდვა") is None
    assert extract_citation("„მეწარმეთა შესახებ“ კანონის 189.5.დ მუხლი კრების მოწვევა") is None


def test_plain_question_is_none():
    assert extract_citation("რა ხანგრძლივობისაა ყოველწლიური ანაზღაურებადი შვებულება?") is None
    assert extract_citation("annual paid leave duration for employees") is None
    assert extract_citation("article C-123/2020") is None


def test_constitutional_complaint_numbers_dont_match_case_class():
    ref = extract_citation("რას შეეხებოდა №1848 და №1849 კონსტიტუციური სარჩელები?")
    assert ref.kind == "case_number"
    assert ref.filters == {"source": "constcourt", "document_number": "1848"}


# --- alias matching (mode=full) ------------------------------------------------------

_ALIASES = [
    {
        "canonical_title": "საქართველოს სამოქალაქო კოდექსი",
        "source": "matsne", "document_id": "31702",
        "registration_code": "040000000.05.001.000223",
        "aliases": ["სამოქალაქო კოდექსი", "civil code"],
    },
    {
        "canonical_title": "მეწარმეთა შესახებ",
        "source": "matsne", "document_id": None, "registration_code": None,
        "aliases": ["მეწარმეთა შესახებ"],
    },
]


def test_alias_fires_only_in_full_mode():
    q = "სამოქალაქო კოდექსი მუხლი 829"
    assert extract_citation(q, mode="ids", aliases=_ALIASES) is None
    ref = extract_citation(q, mode="full", aliases=_ALIASES)
    assert ref is not None and ref.kind == "law_alias"
    assert ref.filters == {"registration_code": "040000000.05.001.000223"}


def test_alias_dominance_rejects_broader_question():
    # The law is mentioned, but the query asks a substantive question → no pin.
    q = ("სამოქალაქო კოდექსი როგორ არეგულირებს დამზღვევის ბრალეულობას სადაზღვევო "
         "შემთხვევის დადგომისას და რა გავლენა აქვს ჩვეულებრივ გაუფრთხილებლობას ანაზღაურებაზე")
    assert extract_citation(q, mode="full", aliases=_ALIASES) is None


def test_alias_without_resolvable_target_is_skipped():
    ref = extract_citation("„მეწარმეთა შესახებ“ კანონი", mode="full", aliases=_ALIASES)
    assert ref is None  # entry has no registration_code / document_id (base act absent)


def test_alias_quote_normalization():
    ref = extract_citation("„სამოქალაქო კოდექსი“ მუხლი 829", mode="full", aliases=_ALIASES)
    assert ref is not None and ref.kind == "law_alias"


# --- lookup & pinning ----------------------------------------------------------------


class _FilterRecordingClient:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def query_points(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(points=list(self.points))

    def scroll(self, **kw):
        representatives = {}
        for point in self.points:
            key = (
                point.payload["source"],
                point.payload["document_id"],
                point.payload.get("version_id"),
            )
            representatives.setdefault(key, point)
        return list(representatives.values()), None


def _pt(pid, doc, ci, score, *, version_id="v1", **overrides):
    payload = {
        "source": "matsne", "document_id": doc, "version_id": version_id,
        "document_type": "legislation", "status": "in_force",
        "effective_from": "2020-01-01", "effective_to": None, "repeal_date": None,
        "version_lineage_complete": True, "version_ambiguous": False,
        "chunk_index": ci, "text": f"{doc}-{ci}",
    }
    payload.update(overrides)
    return SimpleNamespace(
        id=pid, score=score,
        payload=payload,
    )


def test_citation_lookup_preserves_semantic_score_and_records_identity_priority():
    client = _FilterRecordingClient([_pt("a", "d1", 0, 0.02), _pt("b", "d1", 1, 0.01)])
    ref = CitationRef("document_number", {"document_number": "71"}, "№71")
    out = citation_lookup(client, "georgian_legal", [0.1, 0.2], ref)
    kw = client.calls[-1]
    assert kw["using"] == "dense" and kw["query_filter"] is not None
    conds = {c.key for c in kw["query_filter"].must}
    assert conds == {"document_number"}
    assert [pt.score for pt in out] == [0.02, 0.01]
    assert all(pt.payload["identity_priority"] for pt in out)
    assert all(pt.payload["match_type"] == "document_number" for pt in out)
    assert all(pt.payload["identity_confidence"] == 1.0 for pt in out)
    assert not any(pt.payload["identity_ambiguous"] for pt in out)
    assert all(pt.payload["identity_resolution_complete"] for pt in out)


def test_citation_identity_treats_two_versions_as_ambiguous():
    client = _FilterRecordingClient([
        _pt("current", "d1", 0, 0.02, version_id="current"),
        _pt("repealed", "d1", 0, 0.01, version_id="repealed"),
    ])
    ref = CitationRef("document_number", {"document_number": "71"}, "№71")

    out = citation_lookup(client, "georgian_legal", [0.1, 0.2], ref)

    assert all(point.payload["identity_ambiguous"] for point in out)
    assert all(point.payload["identity_confidence"] == 0.5 for point in out)


def test_citation_current_resolution_selects_unique_operative_version():
    client = _FilterRecordingClient([
        _pt(
            "old", "d1", 0, 0.03, version_id="v1", status="repealed",
            effective_to="2025-01-01", repeal_date="2025-01-01",
        ),
        _pt("current", "d1", 0, 0.02, version_id="v2", effective_from="2025-01-01"),
    ])
    ref = CitationRef("document_number", {"document_number": "71"}, "№71")

    out = citation_lookup(
        client, "georgian_legal", [0.1, 0.2], ref,
        effective_on="2026-07-15", require_in_force=True,
    )

    assert [point.id for point in out] == ["current"]
    assert out[0].payload["identity_ambiguous"] is False
    assert out[0].payload["identity_resolution_complete"] is True


def test_citation_as_of_resolution_selects_historical_effective_version():
    client = _FilterRecordingClient([
        _pt(
            "old", "d1", 0, 0.03, version_id="v1", status="repealed",
            effective_from="2020-01-01", effective_to="2025-01-01",
            repeal_date="2025-01-01",
        ),
        _pt("current", "d1", 0, 0.02, version_id="v2", effective_from="2025-01-01"),
    ])
    ref = CitationRef("document_number", {"document_number": "71"}, "№71")

    out = citation_lookup(
        client, "georgian_legal", [0.1, 0.2], ref, effective_on="2024-01-01"
    )

    assert [point.id for point in out] == ["old"]
    assert out[0].payload["identity_ambiguous"] is False


def test_citation_overlapping_effective_versions_remain_ambiguous():
    client = _FilterRecordingClient([
        _pt("one", "d1", 0, 0.03, version_id="v1"),
        _pt("two", "d1", 0, 0.02, version_id="v2", effective_from="2025-01-01"),
    ])
    ref = CitationRef("document_number", {"document_number": "71"}, "№71")

    out = citation_lookup(
        client, "georgian_legal", [0.1, 0.2], ref,
        effective_on="2026-07-15", require_in_force=True,
    )

    assert {point.id for point in out} == {"one", "two"}
    assert all(point.payload["identity_ambiguous"] for point in out)


def test_citation_case_identity_does_not_require_legislative_status():
    client = _FilterRecordingClient([
        _pt(
            "case", "case-1", 0, 0.04, version_id="decision-v1",
            source="supremecourt", document_type="court_decision", status="final",
            effective_from=None, version_lineage_complete=False,
        ),
    ])
    ref = CitationRef("case_number", {"document_number": "ბს-1"}, "ბს-1")

    out = citation_lookup(
        client, "georgian_legal", [0.1, 0.2], ref,
        effective_on="2026-07-15", require_in_force=True,
    )

    assert [point.id for point in out] == ["case"]
    assert out[0].payload["identity_ambiguous"] is False


def test_pin_points_prepends_dedupes_and_truncates():
    pinned = [_pt("a", "d1", 0, 0.2)]
    semantic = [_pt("a", "d1", 0, 0.9), _pt("b", "d2", 0, 0.8), _pt("c", "d3", 0, 0.7)]
    out = pin_points(pinned, semantic, top_k=2)
    assert [pt.id for pt in out] == ["a", "b"]  # dup dropped, truncated to k


def test_pin_points_empty_pinned_is_identity():
    semantic = [_pt("a", "d1", 0, 0.9), _pt("b", "d2", 0, 0.8)]
    assert pin_points([], semantic, top_k=5) == semantic


def test_lookup_tries_prefixed_alternates_until_hit():
    class AltClient:
        def __init__(self):
            self.calls = []

        def query_points(self, **kw):
            self.calls.append(kw)
            val = kw["query_filter"].must[0].match.value
            pts = [_pt("cc", "17546", 0, 0.02)] if val == "N1848" else []
            return SimpleNamespace(points=pts)

    client = AltClient()
    ref = CitationRef("document_number", {"document_number": "1848"}, "№1848",
                      alternates=({"document_number": "N1848"}, {"document_number": "№1848"}))
    out = citation_lookup(client, "georgian_legal", [0.1], ref)
    assert [pt.id for pt in out] == ["cc"] and len(client.calls) == 2
