import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from scrapy import Request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.utils.pagination import (  # noqa: E402
    REPAIR_MANIFEST_FILENAME,
    PaginationReconciler,
    advertised_page_count,
    finalize_pagination_scope,
    handle_pagination_request_failure,
    parse_advertised_count,
)


def _codes(outcome):
    return {example["code"] for example in outcome.examples}


def test_advertised_count_parser_accepts_exact_decimal_values_only():
    assert parse_advertised_count(0) == 0
    assert parse_advertised_count(" 51 ") == 51
    for invalid in (True, -1, "-1", "+1", "1.5", 1.5, None):
        with pytest.raises(ValueError):
            parse_advertised_count(invalid)


def test_exact_multi_page_scope_finalizes_cleanly():
    tracker = PaginationReconciler("instance:1")
    tracker.observe_page(
        0,
        ["a", "b"],
        advertised_total=3,
        advertised_pages=2,
        page_number=1,
    )
    tracker.observe_page(
        2,
        ["c"],
        advertised_total=3,
        advertised_pages=2,
        page_number=2,
        terminal=True,
    )

    outcome = tracker.finalize()

    assert outcome.ok
    assert outcome.page_count == 2
    assert outcome.unique_identifier_count == 3
    assert outcome.duplicate_rate == 0
    assert outcome.examples == ()
    assert tracker.finalize() is outcome


def test_drift_early_empty_duplicates_and_cap_are_hard_failures():
    tracker = PaginationReconciler(
        "category:test",
        max_examples=20,
        max_duplicate_rate=0,
    )
    tracker.observe_page(
        0,
        ["a", "a"],
        advertised_total=4,
        advertised_pages=2,
        page_number=1,
    )
    tracker.observe_page(
        2,
        [],
        advertised_total=5,
        advertised_pages=3,
        page_number=2,
    )
    tracker.mark_cap(cursor=2, configured_cap=2)

    outcome = tracker.finalize()

    assert not outcome.ok
    assert {
        "advertised_total_drift",
        "advertised_pages_drift",
        "early_empty_page",
        "pagination_cap_reached",
        "scope_not_terminal",
        "advertised_total_mismatch",
        "advertised_page_count_mismatch",
        "duplicate_rate_exceeded",
    } <= _codes(outcome)
    assert outcome.duplicate_identifier_count == 1


def test_identifier_tracking_is_bounded_and_fails_closed():
    tracker = PaginationReconciler(
        "bounded",
        max_unique_ids=2,
        max_examples=2,
    )
    tracker.observe_page(
        0,
        ["a", "b", "c", None],
        advertised_total=4,
        advertised_pages=1,
        page_number=1,
        terminal=True,
    )

    outcome = tracker.finalize()

    assert not outcome.ok
    assert outcome.unique_identifier_count == 2
    assert outcome.missing_identifier_count == 1
    assert outcome.hard_failure_count >= 2
    assert len(outcome.examples) == 2
    assert "identifier_tracking_capacity_exceeded" in _codes(outcome)


def test_record_iteration_is_bounded_and_fails_closed():
    tracker = PaginationReconciler(
        "bounded-records",
        max_records=2,
    )
    tracker.observe_page(0, range(10), terminal=True)

    outcome = tracker.finalize()

    assert not outcome.ok
    assert outcome.record_count == 2
    assert dict(outcome.failure_counts)["record_tracking_capacity_exceeded"] == 1


def test_failure_appends_bounded_owner_only_repair_manifest(tmp_path):
    spider = SimpleNamespace(
        name="ecd",
        run_dir=tmp_path,
        record_quality_failure=MagicMock(),
    )
    tracker = PaginationReconciler("instance:1", max_examples=2)
    tracker.mark_failure("waf_or_non_json", cursor=0, detail="x" * 2_000)

    outcome = finalize_pagination_scope(
        spider,
        tracker,
        url="https://example.invalid/list",
    )
    repeated = finalize_pagination_scope(
        spider,
        tracker,
        url="https://example.invalid/list",
    )

    assert repeated is outcome
    spider.record_quality_failure.assert_called_once()
    path = tmp_path / REPAIR_MANIFEST_FILENAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "pagination_incomplete"
    assert record["reconciliation"]["scope"] == "instance:1"
    assert len(record["reconciliation"]["examples"]) <= 2
    assert len(record["reconciliation"]["examples"][0]["detail"]) == 300


def test_complete_scope_emits_no_failure_artifact(tmp_path):
    spider = SimpleNamespace(
        name="napr",
        run_dir=tmp_path,
        record_quality_failure=MagicMock(),
    )
    tracker = PaginationReconciler("catch_all")
    tracker.observe_page(
        0,
        [],
        advertised_total=0,
        advertised_pages=advertised_page_count(0, 50),
        page_number=1,
        terminal=True,
    )

    outcome = finalize_pagination_scope(
        spider,
        tracker,
        url="https://example.invalid/list",
    )

    assert outcome.ok
    spider.record_quality_failure.assert_not_called()
    assert not (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_exhausted_request_retries_are_bound_to_scope_and_repair_manifest(tmp_path):
    spider = SimpleNamespace(
        name="ecd",
        run_dir=tmp_path,
        request_failed=MagicMock(),
    )
    request = Request(
        "https://example.invalid/page",
        meta={
            "pagination_scope": "instance:2",
            "pagination_cursor": 50,
        },
    )
    failure = SimpleNamespace(request=request, value=RuntimeError("timed out"))

    handle_pagination_request_failure(spider, failure)

    spider.request_failed.assert_called_once_with(failure)
    outcome = spider._pagination_reconcilers["instance:2"].finalize()
    assert not outcome.ok
    assert "exhausted_retries" in _codes(outcome)
    repair = json.loads(
        (tmp_path / REPAIR_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert repair["reconciliation"]["scope"] == "instance:2"
