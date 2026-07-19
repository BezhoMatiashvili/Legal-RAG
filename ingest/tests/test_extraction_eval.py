import hashlib
import json
from pathlib import Path

import pytest

from eval.eval_court_extraction import (
    GoldsetValidationError,
    _canonical_body_for_eval,
    build_trust_attestation,
    load_extraction_goldset,
    run_court_extraction_eval,
)
from eval.extraction_eval import (
    DISPOSITION_VALUES,
    TRUST_THRESHOLDS,
    ExtractionGold,
    ExtractionPrediction,
    evaluate_extraction,
    passes_trust_bar,
    trust_bar_failures,
)
from eval.goldset import SnapshotBodies
from ingest.court_extraction_trust import (
    TRUST_SCHEMA_VERSION,
    evaluate_court_extraction_trust,
)


def _gold(document_id: str, judges=("ა. ალფა",), disposition="upheld"):
    return ExtractionGold(
        source="ecd",
        document_id=document_id,
        gold_judges=judges,
        gold_reporting_judge=None,
        gold_disposition=disposition,
        gold_disposition_source="body_operative",
    )


def _prediction(
    document_id: str,
    judges=("ა. ალფა",),
    disposition="upheld",
    confidence="high",
):
    return ExtractionPrediction(
        source="ecd",
        document_id=document_id,
        judges=judges,
        reporting_judge=None,
        judge_confidence="high",
        disposition=disposition,
        disposition_source="body_operative",
        disposition_confidence=confidence,
    )


def test_extraction_metrics_pin_micro_denominators_and_confusion_matrix():
    gold = [
        _gold("1", ("ა. ალფა", "ბ. ბეტა"), "upheld"),
        _gold("2", ("გ. გამა",), "overturned"),
    ]
    predictions = [
        _prediction("1", ("ა. ალფა", "დ. დელტა"), "upheld"),
        _prediction("2", ("გ. გამა",), "modified", confidence="low"),
    ]
    metrics = evaluate_extraction(gold, predictions)

    assert metrics["judges"] == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
        "exact_panel_match": 0.5,
        "reporting_judge_accuracy": 1.0,
    }
    assert metrics["disposition"]["accuracy"] == 0.5
    assert metrics["disposition"]["coverage"] == 0.5
    assert metrics["disposition"]["accuracy_confident"] == 1.0
    assert metrics["disposition"]["confusion_matrix"]["upheld"]["upheld"] == 1
    assert metrics["disposition"]["confusion_matrix"]["overturned"]["modified"] == 1
    assert tuple(metrics["disposition"]["confusion_matrix"]) == DISPOSITION_VALUES
    assert all(
        tuple(row) == DISPOSITION_VALUES
        for row in metrics["disposition"]["confusion_matrix"].values()
    )


def test_extraction_metrics_reject_identity_drift_and_bad_enum():
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate_extraction([_gold("1")], [_prediction("2")])
    with pytest.raises(ValueError, match="invalid predicted disposition"):
        evaluate_extraction([_gold("1")], [_prediction("1", disposition="other")])


def test_trust_gate_is_inclusive_and_no_confident_predictions_fail():
    metrics = {
        "judges": {"f1": TRUST_THRESHOLDS["judge_f1"]},
        "disposition": {
            "coverage": TRUST_THRESHOLDS["disposition_coverage"],
            "accuracy_confident": TRUST_THRESHOLDS["disposition_accuracy_confident"],
        },
    }
    assert passes_trust_bar(metrics)
    assert trust_bar_failures(metrics) == ()

    metrics["disposition"]["coverage"] = 0.0
    metrics["disposition"]["accuracy_confident"] = 0.0
    assert trust_bar_failures(metrics) == (
        "disposition_accuracy_confident",
        "disposition_coverage",
    )


def _gold_record(document_id="1"):
    return {
        "source": "ecd",
        "document_id": document_id,
        "body_sha256": "0" * 64,
        "gold_judges": ["ა. ალფა"],
        "gold_reporting_judge": None,
        "gold_disposition": "upheld",
        "gold_disposition_source": "body_operative",
        "operative_char_start": 1,
        "operative_char_end": 2,
        "operative_block_sha256": "1" * 64,
        "operative_header_sha256": "2" * 64,
        "tags": ["synthetic"],
    }


def test_goldset_loader_rejects_duplicate_json_keys_and_identities(tmp_path):
    duplicate_key = tmp_path / "duplicate-key.jsonl"
    duplicate_key.write_text(
        '{"source":"ecd","source":"ecd"}\n', encoding="utf-8"
    )
    with pytest.raises(GoldsetValidationError, match="duplicate JSON key"):
        load_extraction_goldset(duplicate_key, expected_source_counts=None)

    duplicate_id = tmp_path / "duplicate-id.jsonl"
    line = json.dumps(_gold_record(), ensure_ascii=False)
    duplicate_id.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(GoldsetValidationError, match="duplicate extraction label identity"):
        load_extraction_goldset(duplicate_id, expected_source_counts=None)


def test_snapshot_bodies_explicit_source_file_exposes_record(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {"document_id": "7", "body_markdown": "body", "result": "უცვლელად"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    bodies = SnapshotBodies(
        needed={("supremecourt", "7")},
        source_files={"supremecourt": path},
    )
    assert bodies.body("supremecourt", "7") == "body"
    assert bodies.record("supremecourt", "7")["result"] == "უცვლელად"


def test_runner_regrounds_operative_span_and_writes_separate_report(tmp_path):
    from ingest.court_extract import extract_disposition_from_body

    body = (
        "მოსამართლე: ლაშა ქოჩიაშვილი\n\n"
        "დ ა ა დ გ ი ნ ა:\n\n"
        "1. სააპელაციო საჩივარი არ დაკმაყოფილდეს.\n"
    )
    extraction = extract_disposition_from_body(body)
    assert extraction.operative_start is not None
    assert extraction.operative_end is not None
    ecd_path = tmp_path / "ecd.jsonl"
    ecd_path.write_text(
        json.dumps({"document_id": "1", "body_markdown": body}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record = _gold_record()
    record.update(
        {
            "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "gold_judges": ["ლ. ქოჩიაშვილი"],
            "operative_char_start": extraction.operative_start,
            "operative_char_end": extraction.operative_end,
            "operative_block_sha256": hashlib.sha256(
                body[extraction.operative_start : extraction.operative_end].encode()
            ).hexdigest(),
            "operative_header_sha256": hashlib.sha256(
                extraction.operative_header.encode()
            ).hexdigest(),
        }
    )
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path = tmp_path / "report.json"

    report = run_court_extraction_eval(
        goldset_path=gold_path,
        ecd_path=ecd_path,
        supremecourt_path=tmp_path / "unused.jsonl",
        output_path=output_path,
        expected_source_counts={"ecd": 1},
    )
    assert report["trusted"] is True
    assert report["metrics"]["judges"]["f1"] == 1.0
    assert output_path.is_file()

    record["operative_char_end"] -= 1
    gold_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(GoldsetValidationError, match="operative block slice-back drift"):
        run_court_extraction_eval(
            goldset_path=gold_path,
            ecd_path=ecd_path,
            supremecourt_path=tmp_path / "unused.jsonl",
            output_path=output_path,
            expected_source_counts={"ecd": 1},
        )


def test_supreme_eval_fails_when_raw_and_serving_coordinates_diverge():
    raw_body = "საქართველოს უზენაესი სასამართლო\x00" + " სამართალი" * 10
    raw = {
        "case_id": "7",
        "chamber": "სამოქალაქო საქმეთა პალატა",
        "body_markdown": raw_body,
    }
    with pytest.raises(GoldsetValidationError, match="differs from canonical serving text"):
        _canonical_body_for_eval(
            "supremecourt",
            raw,
            raw_body,
            identity=("supremecourt", "7:სამოქალაქო საქმეთა პალატა"),
        )


def _passing_report(extractor_path: Path):
    return {
        "schema_version": "court-extraction-eval/v1",
        "extractor_revision": "court-extract-v1",
        "extractor_source_sha256": hashlib.sha256(extractor_path.read_bytes()).hexdigest(),
        "eval_set_hash": "a" * 64,
        "thresholds": TRUST_THRESHOLDS,
        "metrics": {
            "judges": {"f1": 1.0},
            "disposition": {"accuracy_confident": 1.0, "coverage": 1.0},
        },
        "trusted": True,
        "failures": [],
        "per_document": [],
        "result_hash": "b" * 64,
        "timestamp": "ignored",
    }


def test_runtime_trust_loader_accepts_exact_attestation_and_fails_closed(tmp_path):
    import ingest.court_extract as court_extract

    extractor_path = Path(court_extract.__file__)
    artifact = build_trust_attestation(_passing_report(extractor_path))
    assert artifact["schema_version"] == TRUST_SCHEMA_VERSION
    trust_path = tmp_path / "trust.json"
    trust_path.write_text(json.dumps(artifact), encoding="utf-8")

    decision = evaluate_court_extraction_trust(
        collection_revision="court-extract-v1",
        path=trust_path,
        expected_eval_set_hash="a" * 64,
        extractor_path=extractor_path,
    )
    assert decision.trusted is True
    assert decision.reasons == ()

    artifact["metrics"]["judge_f1"] = 0.5
    trust_path.write_text(json.dumps(artifact), encoding="utf-8")
    decision = evaluate_court_extraction_trust(
        collection_revision="old-revision",
        path=trust_path,
        extractor_path=extractor_path,
    )
    assert decision.trusted is False
    assert "collection_extractor_revision_mismatch" in decision.reasons
    assert "metric_below_threshold:judge_f1" in decision.reasons

    missing = evaluate_court_extraction_trust(
        collection_revision="court-extract-v1", path=tmp_path / "missing.json"
    )
    assert missing.trusted is False
    assert missing.reasons == ("attestation_missing",)
