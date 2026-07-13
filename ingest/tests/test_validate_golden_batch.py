"""I5: deterministic checks of scripts/validate_golden_batch.py on synthetic fixtures.

Each check must fire on the defect it exists for and stay silent on a clean pair —
these are the machine gates every authored batch passes before adversarial review.
"""

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from validate_golden_batch import (  # noqa: E402
    check_citations,
    check_holdout,
    check_near_dup,
    check_paraphrase,
    check_reground,
    check_schema,
    check_span_coverage,
    citation_tokens,
    jaccard,
    tokens,
)

from eval import goldset  # noqa: E402
from eval.goldset import EvalSetSpec, GoldQuery, Relevance  # noqa: E402
from ingest.chunking import default_token_counter  # noqa: E402

CHUNK_CFG = dict(max_tokens=512, overlap=80, min_tokens=64)
BODY = "მუხლი 1. იჯარის ხელშეკრულება იდება წერილობით. " * 20


def gq(id="q900", query="how is a lease contract concluded", qtype="natural_question",
       lang="en", doc="d1", quote=None, start=0, end=None, grade=2):
    quote = quote if quote is not None else BODY[:44]
    end = end if end is not None else start + len(quote)
    return GoldQuery(id=id, query=query, query_type=qtype, query_language=lang,
                     source="s", document_id=doc, gold_source="s", gold_document_id=doc,
                     relevance=[Relevance(doc, quote, start, end, grade)],
                     answer="answer", doc_title="title")


class StubBodies:
    def __init__(self, mapping):
        self.mapping = mapping

    def body(self, source, document_id):
        return self.mapping[(source, document_id)]


def test_tokens_and_jaccard():
    assert tokens("იჯარის ხელშეკრულება, იჯარის!") == {"იჯარის", "ხელშეკრულება"}
    assert "და" not in tokens("იჯარა და ქირა", content_only=True)
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_near_dup_fires_above_threshold_only():
    existing = [("q001", "alpha beta gamma delta epsilon zeta")]
    hot = gq(id="q901", query="alpha beta gamma delta epsilon")        # jaccard 5/6 ≈ 0.83
    cold = gq(id="q902", query="alpha beta gamma fresh words")          # jaccard 3/8 ≈ 0.38
    issues = check_near_dup([hot, cold], existing, 0.8)
    assert len(issues) == 1 and "q901" in issues[0] and "q001" in issues[0]


def test_near_dup_checks_within_batch_too():
    a = gq(id="q903", query="unique query one two three")
    b = gq(id="q904", query="unique query one two three")
    issues = check_near_dup([a, b], [], 0.8)
    assert len(issues) == 1 and "q904" in issues[0] and "q903" in issues[0]


def test_paraphrase_zero_content_overlap():
    fires = gq(id="q905", qtype="paraphrase", lang="ka",
               query="იჯარის შეწყვეტის წესი", quote="იჯარის ხელშეკრულება წყდება")
    stop_only = gq(id="q906", qtype="paraphrase", lang="ka",
                   query="ქირავნობის დასრულება და გაუქმება", quote="იჯარის შეწყვეტა არის წესი")
    non_paraphrase = gq(id="q907", qtype="keyword", lang="ka",
                        query="იჯარის შეწყვეტა", quote="იჯარის შეწყვეტა")
    issues = check_paraphrase([fires, stop_only, non_paraphrase])
    assert len(issues) == 1 and "q905" in issues[0] and "იჯარის" in issues[0]


def test_citation_tokens_extraction():
    assert citation_tokens("განჩინება №1გ/620-17") == ["1გ/620-17"]
    assert "010.090.000.05.001.016.312" in citation_tokens("კოდი 010.090.000.05.001.016.312")
    assert "330210015800735" in citation_tokens("საქმე 330210015800735")
    assert citation_tokens("ზოგადი შეკითხვა კანონზე") == []


def test_citation_check_pass_and_fail():
    meta = {("s", "d1"): {"document_number": "1გ/620-17", "registration_code": "",
                          "title": "განჩინება", "doc_id": "s:d1", "document_id": "d1",
                          "date": "2017-05-01", "date_raw": "2017", "body_markdown": BODY}}
    ok = gq(id="q908", qtype="legal_citation", lang="ka", query="განჩინება №1გ/620-17 ყადაღა")
    wrong = gq(id="q909", qtype="legal_citation", lang="ka", query="განჩინება №9999-99")
    no_token = gq(id="q910", qtype="legal_citation", lang="ka", query="განჩინება ყადაღის შესახებ")
    unknown_doc = gq(id="q911", qtype="legal_citation", lang="ka", query="განჩინება №1გ/620-17",
                     doc="other")
    issues = check_citations([ok, wrong, no_token, unknown_doc], meta)
    assert not any("q908" in m for m in issues)
    assert any("q909" in m for m in issues)
    assert any("q910" in m and "no extractable" in m for m in issues)
    assert any("q911" in m and "not found" in m for m in issues)


def test_schema_checks():
    ok = gq(id="q912")
    bad_grade = gq(id="q913", grade=5)
    ka_mislabeled = gq(id="q914", query="იჯარის ხელშეკრულების წესი", lang="en")
    dup = gq(id="q912", query="a different query about tax law")
    issues = check_schema([ok, bad_grade, ka_mislabeled, dup], existing_ids={"q001"})
    assert not any(m.startswith("q913:") and "grade" not in m for m in issues)
    assert any("q913" in m and "grade 5" in m for m in issues)
    assert any("q914" in m and "detect_language" in m for m in issues)
    assert any("duplicate id" in m for m in issues)
    assert check_schema([gq(id="q001")], existing_ids={"q001"})  # collision with existing


def test_reground_and_span_coverage():
    bodies = StubBodies({("s", "d1"): BODY})
    good = gq(id="q915")
    drift = gq(id="q916", quote="სხვა ტექსტი სულ", start=0, end=15)
    out_of_range = gq(id="q917", quote="x", start=10 ** 6, end=10 ** 6 + 1)
    assert check_reground([good], bodies) == []
    assert any("q916" in m for m in check_reground([drift], bodies))
    assert check_span_coverage([good], bodies, CHUNK_CFG, default_token_counter) == []
    assert any("q917" in m for m in
               check_span_coverage([out_of_range], bodies, CHUNK_CFG, default_token_counter))


def test_holdout_checks(tmp_path, monkeypatch):
    v1 = tmp_path / "holdout_v1.json"
    v2 = tmp_path / "holdout_v2.json"
    v1.write_text(json.dumps([{"source": "s", "document_id": "legacy"}]), encoding="utf-8")
    monkeypatch.setattr(goldset, "DEFAULT_HOLDOUT", v1)
    spec = EvalSetSpec("v2", tmp_path / "gold.jsonl", v2, (tmp_path,))

    q = gq(id="q918")
    assert check_holdout([q], spec) == [f"holdout file missing: {v2}"]

    v2.write_text(json.dumps([{"source": "s", "document_id": "legacy"},
                              {"source": "s", "document_id": "d1"}]), encoding="utf-8")
    assert check_holdout([q], spec) == []

    v2.write_text(json.dumps([{"source": "s", "document_id": "d1"}]), encoding="utf-8")
    issues = check_holdout([q], spec)
    assert any("not a superset" in m for m in issues)

    v2.write_text(json.dumps([{"source": "s", "document_id": "legacy"}]), encoding="utf-8")
    issues = check_holdout([q], spec)
    assert any("q918" in m and "not in" in m for m in issues)
