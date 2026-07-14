"""Golden-set loading + the fail-loud guards: re-grounding, holdout, span-coverage lint.

Runs against the real snapshot v1 (fast — just reads the docs jsonl), which is the
invariant Part 2 must protect: every evidence quote still slices back exactly.
"""

import pytest

from ingest.chunking import default_token_counter
from eval import goldset
from eval.goldset import GoldQuery, Relevance

count = default_token_counter
CHUNK_CFG = dict(max_tokens=512, overlap=80, min_tokens=64)


class StubBodies:
    def __init__(self, mapping):
        self.mapping = mapping

    def body(self, source, document_id):
        return self.mapping[(source, document_id)]


def test_load_golden_set_shape():
    gold = goldset.load_golden_set()
    assert len(gold) == 103
    q = gold[0]
    assert q.id and q.query and q.query_type and q.relevance
    assert all(isinstance(r, Relevance) for r in q.relevance)


def test_holdout_is_51_docs():
    assert len(goldset.load_holdout()) == 51


@pytest.mark.snapshot
def test_reground_passes_on_all_103_real_spans():
    gold = goldset.load_golden_set()
    bodies = goldset.SnapshotBodies(needed=goldset.gold_docs(gold))
    assert goldset.reground(gold, bodies) == 103  # 103 spans, 0 drift


def test_enforce_holdout_passes_for_real_gold():
    gold = goldset.load_golden_set()
    goldset.enforce_holdout(gold, goldset.load_holdout())  # no raise


@pytest.mark.snapshot
def test_span_coverage_lint_passes_on_real_set():
    gold = goldset.load_golden_set()
    bodies = goldset.SnapshotBodies(needed=goldset.gold_docs(gold))
    # 103 spans all map to >=1 chunk (heading fallback covers the citation-in-title cases)
    assert goldset.lint_span_coverage(gold, bodies, count_tokens=count, **CHUNK_CFG) == 103


def test_reground_fails_loud_on_drift():
    q = GoldQuery(
        id="qX", query="q", query_type="keyword", query_language="ka",
        source="s", document_id="d", gold_source="s", gold_document_id="d",
        relevance=[Relevance("d", "მუხლი", 0, 5, 2)],
    )
    bodies = StubBodies({("s", "d"): "სხვა ტექსტი სულ განსხვავებული"})  # quote won't slice back
    with pytest.raises(ValueError, match="re-grounding drift"):
        goldset.reground([q], bodies)


def test_enforce_holdout_fails_on_missing_doc():
    q = GoldQuery(
        id="qX", query="q", query_type="keyword", query_language="ka",
        source="s", document_id="d", gold_source="s", gold_document_id="d",
        relevance=[Relevance("d", "x", 0, 1, 2)],
    )
    with pytest.raises(ValueError, match="missing from holdout"):
        goldset.enforce_holdout([q], holdout=set())


def test_lint_fails_loud_when_span_maps_to_nothing():
    # a trailing heading that introduces no body governs no chunk; a span pointing at it
    # overlaps neither a chunk body nor a governing heading → ∅ (a genuine annotation bug).
    body = "პირველი ტექსტი აქ.\n\n# ბოლო სათაური"
    h = body.index("ბოლო სათაური")
    q = GoldQuery(
        id="qEmpty", query="q", query_type="keyword", query_language="ka",
        source="s", document_id="d", gold_source="s", gold_document_id="d",
        relevance=[Relevance("d", "ბოლო სათაური", h, h + len("ბოლო სათაური"), 2)],
    )
    bodies = StubBodies({("s", "d"): body})
    with pytest.raises(ValueError, match="map to no chunk"):
        goldset.lint_span_coverage([q], bodies, count_tokens=count, **CHUNK_CFG)
