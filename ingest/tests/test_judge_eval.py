"""Offline unit tests for the L2 judge scorer (eval/judge_eval.py). No LLM, no I/O beyond tmp."""

import json

from eval.judge_eval import (
    JudgeVerdict,
    aggregate_verdicts,
    breakdown_verdicts,
    load_verdicts,
    merge_panel,
    verdict_from_dict,
)


def _v(qid, f, c, cp, ab=False, qt="legal_citation", lang="ka") -> JudgeVerdict:
    return JudgeVerdict(id=qid, faithful=f, correct=c, complete=cp, abstained=ab,
                        query_type=qt, language=lang)


def test_verdict_from_dict_lenient_coercion():
    v = verdict_from_dict({"id": "q1", "faithful": "yes", "correct": 1, "complete": True,
                           "abstained": "no", "query_type": "keyword", "language": "en"})
    assert v.id == "q1"
    assert v.faithful is True and v.correct is True and v.complete is True
    assert v.abstained is False
    assert v.query_type == "keyword" and v.language == "en"


def test_verdict_from_dict_falsey():
    v = verdict_from_dict({"id": "q2", "faithful": "false", "correct": 0, "complete": "no"})
    assert v.faithful is False and v.correct is False and v.complete is False


def test_aggregate_faithfulness_over_answered_only():
    verdicts = [
        _v("q1", True, True, True),
        _v("q2", False, False, True),
        _v("q3", True, True, True, ab=True),  # abstained → excluded from faithfulness denom
    ]
    agg = aggregate_verdicts(verdicts)
    assert agg.n == 3
    # answered = q1,q2 → faithfulness 1/2
    assert agg.faithfulness == 0.5
    assert agg.correctness == 0.5
    assert agg.completeness == 1.0
    assert abs(agg.abstain_rate - 1 / 3) < 1e-9


def test_aggregate_empty():
    agg = aggregate_verdicts([])
    assert agg.n == 0
    assert agg.faithfulness == 0.0 and agg.abstain_rate == 0.0


def test_merge_panel_majority_and_tie():
    # 3-judge panel: faithful 2/3 → True; correct 1/3 → False
    panels = [
        [_v("q1", True, True, True)],
        [_v("q1", True, False, True)],
        [_v("q1", False, False, False)],
    ]
    merged = merge_panel(panels)
    assert len(merged) == 1
    m = merged[0]
    assert m.faithful is True      # 2/3
    assert m.correct is False      # 1/3
    assert m.complete is True      # 2/3
    assert m.judge == "panel-of-3"

    # even panel tie 1-1 → conservative False
    tie = merge_panel([[_v("q9", True, True, True)], [_v("q9", False, False, False)]])
    assert tie[0].faithful is False and tie[0].correct is False


def test_load_verdicts_skips_comments(tmp_path):
    p = tmp_path / "verdicts.jsonl"
    p.write_text(
        "# a comment\n"
        + json.dumps({"id": "q1", "faithful": True, "correct": True, "complete": True}) + "\n"
        + "\n"
        + json.dumps({"id": "q2", "faithful": False, "correct": False, "complete": False}) + "\n",
        encoding="utf-8",
    )
    vs = load_verdicts(p)
    assert [v.id for v in vs] == ["q1", "q2"]
    assert vs[0].faithful is True and vs[1].faithful is False


def test_breakdown_by_query_type():
    verdicts = [
        _v("q1", True, True, True, qt="legal_citation"),
        _v("q2", False, False, False, qt="legal_citation"),
        _v("q3", True, True, True, qt="keyword"),
    ]
    bd = breakdown_verdicts(verdicts, "query_type")
    assert set(bd) == {"legal_citation", "keyword"}
    assert bd["legal_citation"]["faithfulness"] == 0.5
    assert bd["keyword"]["faithfulness"] == 1.0
