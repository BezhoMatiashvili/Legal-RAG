from qdrant_client import models

from ingest.search import build_filter, rerank_points


def _conds(flt):
    """Map of payload key -> the condition object, for asserting on build_filter output."""
    return {c.key: c for c in (flt.must or [])}


def test_build_filter_none_when_empty():
    assert build_filter() is None


def test_build_filter_exact_keyword_fields():
    flt = build_filter(source="matsne", document_number="55", registration_code="RC-1",
                       document_id="6835981", court="supremecourt", document_type="legislation",
                       article_id="829", clause_id="829.1")
    c = _conds(flt)
    assert c["source"].match.value == "matsne"
    assert c["document_number"].match.value == "55"
    assert c["registration_code"].match.value == "RC-1"
    assert c["document_id"].match.value == "6835981"
    assert c["court"].match.value == "supremecourt"
    assert c["document_type"].match.value == "legislation"
    assert c["article_id"].match.value == "829"
    assert c["clause_id"].match.value == "829.1"


def test_build_filter_full_text_fields_use_matchtext():
    flt = build_filter(parties="წიკლაური", contains="გურიის პოლიცია")
    c = _conds(flt)
    assert isinstance(c["parties"].match, models.MatchText)
    assert c["parties"].match.text == "წიკლაური"
    assert isinstance(c["text"].match, models.MatchText)
    assert c["text"].match.text == "გურიის პოლიცია"


def test_build_filter_date_range():
    flt = build_filter(date_from="2026-04-01", date_to="2026-04-30")
    rng = _conds(flt)["date"].range  # pydantic coerces the RFC3339 strings to datetimes
    assert (rng.gte.year, rng.gte.month, rng.gte.day) == (2026, 4, 1)
    assert (rng.lte.year, rng.lte.month, rng.lte.day) == (2026, 4, 30)


def test_build_filter_as_of_uses_effective_interval_not_publication_date():
    flt = build_filter(as_of="2024-02-03")
    assert len(flt.must) == 2
    effective_from = flt.must[0]
    open_or_later = flt.must[1]
    assert effective_from.key == "effective_from"
    assert effective_from.range.lte.date().isoformat() == "2024-02-03"
    assert {condition.key for condition in open_or_later.should if hasattr(condition, "key")} == {
        "effective_to"
    }
    assert open_or_later.should[0].range.gt.date().isoformat() == "2024-02-03"
    assert open_or_later.should[1].is_empty.key == "effective_to"


def test_build_filter_status_is_exact_keyword():
    flt = build_filter(status="in_force")
    c = _conds(flt)
    assert isinstance(c["status"].match, models.MatchValue)
    assert c["status"].match.value == "in_force"


def test_build_filter_combined():
    flt = build_filter(source="matsne", document_number="55", date_from="2026-04-01")
    keys = {c.key for c in flt.must}
    assert keys == {"source", "document_number", "date"}


# --- reranking ---------------------------------------------------------------


class _Point:
    """Minimal stand-in for a Qdrant ScoredPoint (settable score + payload)."""

    def __init__(self, text, score=0.0):
        self.payload = {"text": text}
        self.score = score


class _FakeReranker:
    """Returns a preset relevance per text, keyed by substring, for deterministic tests."""

    def __init__(self, scores):
        self.scores = scores

    def score(self, query, texts):
        return [self.scores[t] for t in texts]


def test_rerank_points_reorders_by_relevance_and_overwrites_score():
    pts = [_Point("a", score=0.9), _Point("b", score=0.8), _Point("c", score=0.7)]
    reranker = _FakeReranker({"a": 0.1, "b": 0.95, "c": 0.5})
    out = rerank_points(reranker, "q", pts, top_k=3)
    assert [p.payload["text"] for p in out] == ["b", "c", "a"]  # reordered by rerank score
    assert out[0].score == 0.95                                  # score overwritten


def test_rerank_points_applies_min_score_gate():
    pts = [_Point("a"), _Point("b"), _Point("c")]
    reranker = _FakeReranker({"a": 0.9, "b": 0.2, "c": 0.05})
    out = rerank_points(reranker, "q", pts, top_k=10, min_score=0.3)
    assert [p.payload["text"] for p in out] == ["a"]            # b, c dropped below gate


def test_rerank_points_respects_top_k():
    pts = [_Point(t) for t in "abcde"]
    reranker = _FakeReranker({t: i / 10 for i, t in enumerate("abcde")})
    out = rerank_points(reranker, "q", pts, top_k=2)
    assert [p.payload["text"] for p in out] == ["e", "d"]
