"""Run the deterministic court extraction accuracy gate over grounded labels.

Exit status 0 means the fixed trust bar passed, 1 means valid predictions missed a metric
threshold, and 2 means the gold data, body spans, source metadata, or extractor contract was
invalid.  The runner never calls a model or an external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest.artifacts import atomic_write_json
from ingest.court_extraction_trust import TRUST_SCHEMA_VERSION
from ingest.court_extraction_trust import EXPECTED_THRESHOLDS as RUNTIME_THRESHOLDS

from .extraction_eval import (
    DISPOSITION_VALUES,
    TRUST_THRESHOLDS,
    ExtractionGold,
    ExtractionPrediction,
    evaluate_extraction,
    trust_bar_failures,
)
from .goldset import DEFAULT_SNAPSHOT_DOCS, SnapshotBodies, eval_set_hash

EVAL_DIR = Path(__file__).resolve().parent
INGEST_ROOT = EVAL_DIR.parent
REPO_ROOT = INGEST_ROOT.parent
DEFAULT_GOLDSET = EVAL_DIR / "extraction_goldset.jsonl"
DEFAULT_ECD_DOCS = DEFAULT_SNAPSHOT_DOCS / "ecd.jsonl"
DEFAULT_SUPREMECOURT_DOCS = REPO_ROOT / "artifacts" / "supremecourt" / "latest" / "items.jsonl"
DEFAULT_STATE_DIR = INGEST_ROOT / ".state" / "court_extraction_eval"
REPORT_SCHEMA_VERSION = "court-extraction-eval/v1"
EXPECTED_SOURCE_COUNTS = {"ecd": 60, "supremecourt": 20}
_GOLD_KEYS = {
    "source",
    "document_id",
    "body_sha256",
    "gold_judges",
    "gold_reporting_judge",
    "gold_disposition",
    "gold_disposition_source",
    "operative_char_start",
    "operative_char_end",
    "operative_block_sha256",
    "operative_header_sha256",
    "tags",
}
_SHA256_LENGTH = 64


class GoldsetValidationError(ValueError):
    """The checked-in labels cannot be grounded against the frozen source bodies."""


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(_nfc(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(line: str, *, origin: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GoldsetValidationError(f"{origin}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise GoldsetValidationError(f"{origin}: non-finite JSON constant {value!r}")

    try:
        return json.loads(
            line,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise GoldsetValidationError(f"{origin}: invalid JSON: {exc}") from exc


def load_extraction_goldset(
    path: Path = DEFAULT_GOLDSET,
    *,
    expected_source_counts: dict[str, int] | None = EXPECTED_SOURCE_COUNTS,
) -> list[dict[str, Any]]:
    """Strictly parse labels and enforce the frozen source allocation."""

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GoldsetValidationError(f"cannot read extraction goldset: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = _strict_json(line, origin=f"{path}:{line_number}")
        if not isinstance(value, dict) or set(value) != _GOLD_KEYS:
            raise GoldsetValidationError(
                f"{path}:{line_number}: expected exact keys {sorted(_GOLD_KEYS)}"
            )
        source = value.get("source")
        document_id = value.get("document_id")
        if not isinstance(source, str) or not isinstance(document_id, str) or not document_id:
            raise GoldsetValidationError(f"{path}:{line_number}: invalid document identity")
        identity = (source, document_id)
        if identity in seen:
            raise GoldsetValidationError(f"duplicate extraction label identity: {identity!r}")
        seen.add(identity)
        records.append(value)

    if expected_source_counts is not None:
        observed = Counter(str(record["source"]) for record in records)
        if dict(observed) != expected_source_counts:
            raise GoldsetValidationError(
                f"goldset source counts must be {expected_source_counts}, observed {dict(observed)}"
            )
    elif not records:
        raise GoldsetValidationError("extraction goldset is empty")
    return records


def _scraped_result(record: dict[str, Any]) -> str | None:
    promoted = record.get("promoted")
    candidates = [
        promoted.get("result") if isinstance(promoted, dict) else None,
        record.get("result"),
        (record.get("extra") or {}).get("result")
        if isinstance(record.get("extra"), dict)
        else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _lint_hash(value: Any, *, field_name: str, identity: tuple[str, str]) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GoldsetValidationError(f"{identity!r}: invalid {field_name}")
    return value


def _canonical_body_for_eval(
    source: str,
    source_record: dict[str, Any],
    loaded_body: str,
    *,
    identity: tuple[str, str],
) -> str:
    """Prove raw Supreme labels use the serving payload's exact coordinate space."""

    if source != "supremecourt":
        # ECD labels are loaded from the immutable clean snapshot rather than a raw crawl.
        return loaded_body
    from ingest.pipeline import _prepare_doc_for_index
    from ingest.sources import normalize

    try:
        canonical_body = _prepare_doc_for_index(
            normalize("supremecourt", source_record)
        ).body_markdown
    except (KeyError, TypeError, ValueError) as exc:
        raise GoldsetValidationError(
            f"{identity!r}: Supreme Court record cannot enter canonical serving space: {exc}"
        ) from exc
    if loaded_body != canonical_body:
        raise GoldsetValidationError(
            f"{identity!r}: raw Supreme Court body differs from canonical serving text; "
            "regenerate spans from a canonical snapshot"
        )
    return canonical_body


def _validate_and_predict(
    records: list[dict[str, Any]], bodies: SnapshotBodies
) -> tuple[list[ExtractionGold], list[ExtractionPrediction], list[dict[str, Any]]]:
    try:
        from ingest.court_extract import (
            DISPOSITION_VALUES as EXTRACTOR_DISPOSITIONS,
            extract_disposition_from_body,
            extract_judges,
            disposition_from_scraped_result,
            normalize_judge_key,
        )
    except (ImportError, AttributeError) as exc:
        raise GoldsetValidationError(f"court extractor contract unavailable: {exc}") from exc

    if tuple(EXTRACTOR_DISPOSITIONS) != DISPOSITION_VALUES:
        raise GoldsetValidationError(
            "extractor disposition enum differs from the frozen evaluation enum"
        )
    if TRUST_THRESHOLDS != RUNTIME_THRESHOLDS:
        raise GoldsetValidationError("eval and runtime extraction thresholds differ")

    gold: list[ExtractionGold] = []
    predictions: list[ExtractionPrediction] = []
    rows: list[dict[str, Any]] = []
    for label in records:
        source = str(label["source"])
        document_id = str(label["document_id"])
        identity = (source, document_id)
        if source not in EXPECTED_SOURCE_COUNTS:
            raise GoldsetValidationError(f"unsupported extraction source: {source!r}")
        source_record = bodies.record(source, document_id)
        body = _canonical_body_for_eval(
            source,
            source_record,
            bodies.body(source, document_id),
            identity=identity,
        )
        if _lint_hash(label["body_sha256"], field_name="body_sha256", identity=identity) != (
            _text_sha256(body)
        ):
            raise GoldsetValidationError(f"{identity!r}: body hash drift")

        start = label["operative_char_start"]
        end = label["operative_char_end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(body)
        ):
            raise GoldsetValidationError(f"{identity!r}: invalid operative char span")
        block = body[start:end]
        block_hash = _lint_hash(
            label["operative_block_sha256"],
            field_name="operative_block_sha256",
            identity=identity,
        )
        if block_hash != _text_sha256(block):
            raise GoldsetValidationError(f"{identity!r}: operative block slice-back drift")

        # Ground the reviewed header hash directly from the frozen body slice before
        # consulting the extractor.  The label's span begins at the literal header and
        # the header occupies its first line; only presentation whitespace/the colon are
        # outside the hashed glyph sequence.
        header_line = block.splitlines()[0]
        grounded_header = header_line.rstrip(" :\t\u00a0")
        header_hash = _lint_hash(
            label["operative_header_sha256"],
            field_name="operative_header_sha256",
            identity=identity,
        )
        if not grounded_header or header_hash != _text_sha256(grounded_header):
            raise GoldsetValidationError(f"{identity!r}: operative header slice-back drift")

        body_disposition = extract_disposition_from_body(body)
        if body_disposition.operative_start != start or body_disposition.operative_end != end:
            raise GoldsetValidationError(
                f"{identity!r}: gold span is not the extractor's last operative block"
            )
        if body_disposition.operative_header != grounded_header:
            raise GoldsetValidationError(
                f"{identity!r}: extractor operative header differs from grounded gold"
            )

        gold_judges = label["gold_judges"]
        if (
            not isinstance(gold_judges, list)
            or any(not isinstance(judge, str) or not judge for judge in gold_judges)
            or gold_judges != sorted(set(gold_judges))
            or any(normalize_judge_key(judge) != judge for judge in gold_judges)
        ):
            raise GoldsetValidationError(
                f"{identity!r}: gold_judges must be sorted, unique normalized keys"
            )
        reporting = label["gold_reporting_judge"]
        if reporting is not None and (
            not isinstance(reporting, str)
            or reporting not in gold_judges
            or normalize_judge_key(reporting) != reporting
        ):
            raise GoldsetValidationError(f"{identity!r}: invalid gold reporting judge")
        gold_disposition = label["gold_disposition"]
        gold_source = label["gold_disposition_source"]
        if gold_disposition not in DISPOSITION_VALUES:
            raise GoldsetValidationError(f"{identity!r}: invalid gold disposition")
        if gold_source not in {"body_operative", "scraped_result"}:
            raise GoldsetValidationError(f"{identity!r}: invalid gold disposition source")
        tags = label["tags"]
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
            raise GoldsetValidationError(f"{identity!r}: tags must be non-empty strings")

        panel = extract_judges(body)
        chosen_disposition = body_disposition
        result_text = _scraped_result(source_record) if source == "supremecourt" else None
        if result_text is not None:
            scraped = disposition_from_scraped_result(result_text)
            if scraped.disposition != "unknown":
                chosen_disposition = scraped

        expected = ExtractionGold(
            source=source,
            document_id=document_id,
            gold_judges=tuple(gold_judges),
            gold_reporting_judge=reporting,
            gold_disposition=gold_disposition,
            gold_disposition_source=gold_source,
        )
        predicted = ExtractionPrediction(
            source=source,
            document_id=document_id,
            judges=tuple(panel.judges),
            reporting_judge=panel.reporting_judge,
            judge_confidence=panel.confidence,
            disposition=chosen_disposition.disposition,
            disposition_source=chosen_disposition.source,
            disposition_confidence=chosen_disposition.confidence,
            disposition_mixed=chosen_disposition.mixed,
        )
        gold.append(expected)
        predictions.append(predicted)
        rows.append(
            {
                "source": source,
                "document_id": document_id,
                "gold": {
                    "judges": list(expected.gold_judges),
                    "reporting_judge": expected.gold_reporting_judge,
                    "disposition": expected.gold_disposition,
                    "disposition_source": expected.gold_disposition_source,
                },
                "prediction": {
                    "judges": list(predicted.judges),
                    "reporting_judge": predicted.reporting_judge,
                    "judge_confidence": predicted.judge_confidence,
                    "disposition": predicted.disposition,
                    "disposition_source": predicted.disposition_source,
                    "disposition_confidence": predicted.disposition_confidence,
                    "disposition_mixed": predicted.disposition_mixed,
                },
            }
        )
    return gold, predictions, rows


def _result_hash(report_without_timestamp: dict[str, Any]) -> str:
    payload = json.dumps(
        report_without_timestamp,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_court_extraction_eval(
    *,
    goldset_path: Path = DEFAULT_GOLDSET,
    ecd_path: Path = DEFAULT_ECD_DOCS,
    supremecourt_path: Path = DEFAULT_SUPREMECOURT_DOCS,
    output_path: Path | None = None,
    expected_source_counts: dict[str, int] | None = EXPECTED_SOURCE_COUNTS,
) -> dict[str, Any]:
    """Ground labels, score the current extractor, and persist a deterministic report."""

    from ingest.court_extract import EXTRACTOR_REVISION

    records = load_extraction_goldset(
        goldset_path, expected_source_counts=expected_source_counts
    )
    needed = {(str(item["source"]), str(item["document_id"])) for item in records}
    bodies = SnapshotBodies(
        needed=needed,
        source_files={"ecd": Path(ecd_path), "supremecourt": Path(supremecourt_path)},
        document_id_fields={"supremecourt": ("case_id", "chamber")},
    )
    gold, predictions, rows = _validate_and_predict(records, bodies)
    metrics = evaluate_extraction(gold, predictions)
    failures = list(trust_bar_failures(metrics))
    deterministic = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_source_sha256": _file_sha256(
            Path(__file__).resolve().parents[1] / "ingest" / "court_extract.py"
        ),
        "eval_set_hash": eval_set_hash(goldset_path),
        "thresholds": TRUST_THRESHOLDS,
        "metrics": metrics,
        "trusted": not failures,
        "failures": failures,
        "per_document": rows,
    }
    report = {
        **deterministic,
        "result_hash": _result_hash(deterministic),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    destination = output_path
    if destination is None:
        destination = DEFAULT_STATE_DIR / (
            f"court_extraction_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        )
    atomic_write_json(Path(destination), report)
    return report


def build_trust_attestation(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a passing report to the only runtime-trusted artifact schema."""

    if report.get("schema_version") != REPORT_SCHEMA_VERSION or report.get("trusted") is not True:
        raise GoldsetValidationError("only a passing court extraction report can be attested")
    metrics = report["metrics"]
    gate_metrics = {
        "disposition_accuracy_confident": metrics["disposition"]["accuracy_confident"],
        "disposition_coverage": metrics["disposition"]["coverage"],
        "judge_f1": metrics["judges"]["f1"],
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in gate_metrics.values()
    ):
        raise GoldsetValidationError("report gate metrics are invalid")
    return {
        "schema_version": TRUST_SCHEMA_VERSION,
        "extractor_revision": report["extractor_revision"],
        "extractor_source_sha256": report["extractor_source_sha256"],
        "eval_set_hash": report["eval_set_hash"],
        "result_hash": report["result_hash"],
        "covered_sources": ["ecd", "supremecourt"],
        "thresholds": TRUST_THRESHOLDS,
        "metrics": gate_metrics,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--ecd", type=Path, default=DEFAULT_ECD_DOCS)
    parser.add_argument("--supremecourt", type=Path, default=DEFAULT_SUPREMECOURT_DOCS)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--attestation-out",
        type=Path,
        help="optional explicit output for a passing runtime trust attestation",
    )
    args = parser.parse_args(argv)
    try:
        report = run_court_extraction_eval(
            goldset_path=args.goldset,
            ecd_path=args.ecd,
            supremecourt_path=args.supremecourt,
            output_path=args.out,
        )
        if args.attestation_out is not None and report["trusted"]:
            atomic_write_json(args.attestation_out, build_trust_attestation(report))
    except (GoldsetValidationError, OSError, KeyError, ValueError) as exc:
        print(f"court extraction eval invalid: {exc}", file=sys.stderr)
        return 2
    summary = {
        "trusted": report["trusted"],
        "failures": report["failures"],
        "metrics": report["metrics"],
        "result_hash": report["result_hash"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["trusted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ECD_DOCS",
    "DEFAULT_GOLDSET",
    "DEFAULT_SUPREMECOURT_DOCS",
    "EXPECTED_SOURCE_COUNTS",
    "GoldsetValidationError",
    "build_trust_attestation",
    "load_extraction_goldset",
    "main",
    "run_court_extraction_eval",
]
