"""L2 answer-faithfulness layer — verdict schema and reviewer aggregation.

The deterministic layer (``answer_eval.py``) measures retrieval-derived answer signals with
NO judge and hard-gates. This layer measures the faithfulness + correctness of a *composed*
answer, which needs an LLM judge. Per the approved plan §2b (user 2026-07-12: build the full
answer-quality eval, PII **UNMASKED**), ``dump_judge_batch.py`` produces a batch of
``(query, retrieved context, gold answer)`` triples, an approved reviewer emits one verdict
per item, and this module aggregates. Retrieved corpus text (incl. PII) is passed to the
reviewer unmasked — the same exposure already accepted at answer time.

2026 RAG-eval best practice applied here:
  * the deterministic floor (``answer_eval.py``) stays the hard gate; this judge is a logged,
    *calibrated* metric, **never the sole gate**;
  * **panel-of-N** (``merge_panel`` takes the majority per axis) blunts position / verbosity /
    self-enhancement bias;
  * calibrate the judge against a hand-checked subset before trusting the numbers.

A verdict grades three axes, each judged from ``query + retrieved context + gold answer``:
  * **faithful** — every claim an answer composed *from the retrieved context* would make is
    supported by that context (groundedness / no hallucination). THE #1 metric.
  * **correct**  — that answer matches the gold reference answer on the legal substance.
  * **complete** — the retrieved context is sufficient to fully answer (the judge's view of
    context-sufficiency, complementing the deterministic ``answerability@k``).
An ``abstained`` verdict (the context does not support any answer → the system SHOULD refuse)
is scored separately: on an abstain-appropriate item, abstaining is the correct outcome.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's verdict for one query (three boolean axes + optional abstain + note)."""

    id: str
    faithful: bool
    correct: bool
    complete: bool
    abstained: bool = False
    query_type: str = ""
    language: str = ""
    judge: str = "unspecified"
    note: str = ""


def _b(x: object) -> bool:
    """Coerce a JSON truthy/`"yes"`/`"true"`/1 value to bool (judge output is lenient)."""
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "faithful", "correct", "complete")
    return bool(x)


def verdict_from_dict(d: dict) -> JudgeVerdict:
    return JudgeVerdict(
        id=str(d["id"]),
        faithful=_b(d.get("faithful")),
        correct=_b(d.get("correct")),
        complete=_b(d.get("complete")),
        abstained=_b(d.get("abstained")),
        query_type=str(d.get("query_type", "")),
        language=str(d.get("language", "")),
        judge=str(d.get("judge", "unspecified")),
        note=str(d.get("note", "")),
    )


def load_verdicts(path: Path) -> list[JudgeVerdict]:
    """Load a verdicts JSONL (one judge). Blank lines and ``#`` comments are skipped."""
    out: list[JudgeVerdict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(verdict_from_dict(json.loads(line)))
    return out


def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def merge_panel(panels: Sequence[Sequence[JudgeVerdict]]) -> list[JudgeVerdict]:
    """Combine N judges into one verdict per id by MAJORITY vote on each axis.

    A panel reduces single-judge bias. Ties (even N, 50/50) resolve to ``False`` — the
    conservative choice for faithfulness/correctness (don't credit a contested pass).
    """
    by_id: dict[str, list[JudgeVerdict]] = {}
    order: list[str] = []
    for panel in panels:
        for v in panel:
            if v.id not in by_id:
                order.append(v.id)
            by_id.setdefault(v.id, []).append(v)

    def majority(vs: list[JudgeVerdict], attr: str) -> bool:
        yes = sum(1 for v in vs if getattr(v, attr))
        return yes * 2 > len(vs)  # strict majority; tie → False

    merged: list[JudgeVerdict] = []
    for qid in order:
        vs = by_id[qid]
        merged.append(
            JudgeVerdict(
                id=qid,
                faithful=majority(vs, "faithful"),
                correct=majority(vs, "correct"),
                complete=majority(vs, "complete"),
                abstained=majority(vs, "abstained"),
                query_type=vs[0].query_type,
                language=vs[0].language,
                judge=f"panel-of-{len(vs)}",
            )
        )
    return merged


@dataclass(frozen=True)
class JudgeAggregate:
    n: int
    faithfulness: float   # frac of answered items whose answer is grounded (THE headline)
    correctness: float    # frac of answered items matching the gold answer
    completeness: float   # frac of answered items with sufficient retrieved context
    abstain_rate: float   # frac of items the judge marked as should-abstain


def aggregate_verdicts(verdicts: Sequence[JudgeVerdict]) -> JudgeAggregate:
    """Rates over the verdicts. Faithfulness/correctness/completeness are computed over the
    ANSWERED items (an abstain is neither a hallucination nor an answer to grade)."""
    answered = [v for v in verdicts if not v.abstained]
    return JudgeAggregate(
        n=len(verdicts),
        faithfulness=_mean([1.0 if v.faithful else 0.0 for v in answered]),
        correctness=_mean([1.0 if v.correct else 0.0 for v in answered]),
        completeness=_mean([1.0 if v.complete else 0.0 for v in answered]),
        abstain_rate=_mean([1.0 if v.abstained else 0.0 for v in verdicts]),
    )


def breakdown_verdicts(verdicts: Sequence[JudgeVerdict], attr: str = "query_type") -> dict:
    groups: dict[str, list[JudgeVerdict]] = {}
    for v in verdicts:
        groups.setdefault(getattr(v, attr), []).append(v)
    return {g: aggregate_verdicts(vs).__dict__ for g, vs in sorted(groups.items())}
