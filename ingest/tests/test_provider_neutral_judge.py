"""Provider-neutral judge provenance must not affect evaluation scores."""

from pathlib import Path

from eval.judge_eval import JudgeVerdict, aggregate_verdicts, verdict_from_dict


def test_missing_judge_label_defaults_to_unspecified() -> None:
    direct = JudgeVerdict(id="q1", faithful=True, correct=True, complete=True)
    parsed = verdict_from_dict(
        {"id": "q1", "faithful": True, "correct": True, "complete": True}
    )

    assert direct.judge == "unspecified"
    assert parsed.judge == "unspecified"


def test_explicit_judge_label_is_preserved() -> None:
    parsed = verdict_from_dict(
        {
            "id": "q1",
            "faithful": True,
            "correct": False,
            "complete": True,
            "judge": "manual-panel-v2",
        }
    )

    assert parsed.judge == "manual-panel-v2"


def test_judge_provenance_does_not_change_aggregate_scores() -> None:
    rows = [
        {"id": "q1", "faithful": True, "correct": True, "complete": True},
        {
            "id": "q2",
            "faithful": False,
            "correct": False,
            "complete": True,
            "abstained": False,
        },
        {
            "id": "q3",
            "faithful": True,
            "correct": True,
            "complete": True,
            "abstained": True,
        },
    ]
    unspecified = [verdict_from_dict(row) for row in rows]
    explicitly_labeled = [
        verdict_from_dict({**row, "judge": "approved-reviewer"}) for row in rows
    ]

    assert aggregate_verdicts(unspecified) == aggregate_verdicts(explicitly_labeled)


def test_evaluator_provenance_language_is_provider_neutral() -> None:
    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    for name in ("judge_eval.py", "dump_judge_batch.py", "answer_eval.py"):
        assert "Claude" not in (eval_dir / name).read_text(encoding="utf-8")
