"""Pure deterministic metrics for court-panel and disposition extraction.

This module deliberately knows nothing about snapshots, Qdrant, or command-line state.  It
scores already-grounded labels and predictions so unit tests can pin every denominator and
the runner can fail closed before publishing a trust attestation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DISPOSITION_VALUES = (
    "inadmissible",
    "not_considered",
    "upheld",
    "overturned",
    "overturned_remanded",
    "partially_overturned",
    "modified",
    "granted",
    "remanded",
    "terminated",
    "settled",
    "unknown",
)

TRUST_THRESHOLDS = {
    "disposition_accuracy_confident": 0.95,
    "disposition_coverage": 0.80,
    "judge_f1": 0.97,
}


@dataclass(frozen=True)
class ExtractionGold:
    source: str
    document_id: str
    gold_judges: tuple[str, ...]
    gold_reporting_judge: str | None
    gold_disposition: str
    gold_disposition_source: str

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.document_id


@dataclass(frozen=True)
class ExtractionPrediction:
    source: str
    document_id: str
    judges: tuple[str, ...]
    reporting_judge: str | None
    judge_confidence: str
    disposition: str
    disposition_source: str
    disposition_confidence: str
    disposition_mixed: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.document_id


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _unique_by_key(items: list[Any], *, label: str) -> dict[tuple[str, str], Any]:
    table: dict[tuple[str, str], Any] = {}
    for item in items:
        if item.key in table:
            raise ValueError(f"duplicate {label} identity: {item.key!r}")
        table[item.key] = item
    return table


def evaluate_extraction(
    gold: list[ExtractionGold], predictions: list[ExtractionPrediction]
) -> dict[str, Any]:
    """Return exact, stable metrics; missing or extra predictions are fatal."""

    gold_by_key = _unique_by_key(gold, label="gold")
    predictions_by_key = _unique_by_key(predictions, label="prediction")
    if not gold_by_key:
        raise ValueError("court extraction evaluation requires at least one gold record")
    missing = sorted(set(gold_by_key) - set(predictions_by_key))
    extra = sorted(set(predictions_by_key) - set(gold_by_key))
    if missing or extra:
        raise ValueError(f"prediction identity mismatch: missing={missing}, extra={extra}")

    judge_tp = judge_fp = judge_fn = 0
    exact_panels = reporting_correct = 0
    disposition_correct = disposition_source_correct = 0
    confident = confident_correct = 0
    confusion = {
        actual: {predicted: 0 for predicted in DISPOSITION_VALUES}
        for actual in DISPOSITION_VALUES
    }

    for key in sorted(gold_by_key):
        expected = gold_by_key[key]
        predicted = predictions_by_key[key]
        if expected.gold_disposition not in DISPOSITION_VALUES:
            raise ValueError(f"invalid gold disposition for {key!r}: {expected.gold_disposition!r}")
        if predicted.disposition not in DISPOSITION_VALUES:
            raise ValueError(f"invalid predicted disposition for {key!r}: {predicted.disposition!r}")
        if predicted.judge_confidence not in {"high", "low"}:
            raise ValueError(f"invalid judge confidence for {key!r}")
        if predicted.disposition_confidence not in {"high", "low"}:
            raise ValueError(f"invalid disposition confidence for {key!r}")

        expected_judges = set(expected.gold_judges)
        predicted_judges = set(predicted.judges)
        judge_tp += len(expected_judges & predicted_judges)
        judge_fp += len(predicted_judges - expected_judges)
        judge_fn += len(expected_judges - predicted_judges)
        exact_panels += predicted_judges == expected_judges
        reporting_correct += predicted.reporting_judge == expected.gold_reporting_judge

        is_correct = predicted.disposition == expected.gold_disposition
        disposition_correct += is_correct
        disposition_source_correct += (
            predicted.disposition_source == expected.gold_disposition_source
        )
        confusion[expected.gold_disposition][predicted.disposition] += 1
        if predicted.disposition_confidence == "high":
            confident += 1
            confident_correct += is_correct

    judge_precision = _ratio(judge_tp, judge_tp + judge_fp)
    judge_recall = _ratio(judge_tp, judge_tp + judge_fn)
    judge_f1 = _ratio(2 * judge_precision * judge_recall, judge_precision + judge_recall)
    count = len(gold_by_key)
    return {
        "n": count,
        "judges": {
            "true_positive": judge_tp,
            "false_positive": judge_fp,
            "false_negative": judge_fn,
            "precision": judge_precision,
            "recall": judge_recall,
            "f1": judge_f1,
            "exact_panel_match": _ratio(exact_panels, count),
            "reporting_judge_accuracy": _ratio(reporting_correct, count),
        },
        "disposition": {
            "accuracy": _ratio(disposition_correct, count),
            "source_accuracy": _ratio(disposition_source_correct, count),
            "coverage": _ratio(confident, count),
            "accuracy_confident": _ratio(confident_correct, confident),
            "n_confident": confident,
            "confusion_matrix": confusion,
        },
    }


def trust_bar_failures(
    metrics: dict[str, Any], thresholds: dict[str, float] = TRUST_THRESHOLDS
) -> tuple[str, ...]:
    """Return stable failure codes for the inclusive extraction trust gate."""

    checks = (
        (
            "disposition_accuracy_confident",
            float(metrics["disposition"]["accuracy_confident"]),
        ),
        ("disposition_coverage", float(metrics["disposition"]["coverage"])),
        ("judge_f1", float(metrics["judges"]["f1"])),
    )
    return tuple(
        name for name, value in checks if value < float(thresholds[name])
    )


def passes_trust_bar(
    metrics: dict[str, Any], thresholds: dict[str, float] = TRUST_THRESHOLDS
) -> bool:
    return not trust_bar_failures(metrics, thresholds)


__all__ = [
    "DISPOSITION_VALUES",
    "TRUST_THRESHOLDS",
    "ExtractionGold",
    "ExtractionPrediction",
    "evaluate_extraction",
    "passes_trust_bar",
    "trust_bar_failures",
]
