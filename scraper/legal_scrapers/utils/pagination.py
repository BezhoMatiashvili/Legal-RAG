"""Bounded pagination accounting and private repair-manifest reporting.

The reconciler is deliberately independent of Scrapy and I/O.  Spider adapters feed
it one listing page at a time, then the small integration helpers below persist only
bounded failure evidence when a scope cannot be proven complete.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

PAGINATION_SCHEMA_VERSION = 1
REPAIR_MANIFEST_FILENAME = "repair_manifest.jsonl"
DEFAULT_MAX_UNIQUE_IDS = 1_000_000
DEFAULT_MAX_RECORDS = 2_000_000
DEFAULT_MAX_PAGES = 20_000
DEFAULT_MAX_EXAMPLES = 20
DEFAULT_MAX_FAILURE_CODES = 64
DEFAULT_MAX_DUPLICATE_RATE = 0.0


def advertised_page_count(total: int, page_size: int) -> int:
    """Return requests needed to prove a total, including the empty-total probe."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be an integer >= 0")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page_size must be an integer >= 1")
    return max(1, math.ceil(total / page_size))


def parse_advertised_count(value: object) -> int:
    """Parse an API count without silently truncating floats or signed strings."""
    if isinstance(value, bool):
        raise ValueError("advertised count must be an integer >= 0")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if (
            not text
            or len(text) > 20
            or any(char not in "0123456789" for char in text)
        ):
            raise ValueError("advertised count must contain only decimal digits")
        result = int(text)
    else:
        raise ValueError("advertised count must be an integer or decimal string")
    if result < 0:
        raise ValueError("advertised count must be an integer >= 0")
    return result


@dataclass(frozen=True)
class PaginationOutcome:
    scope: str
    ok: bool
    page_count: int
    record_count: int
    unique_identifier_count: int
    duplicate_identifier_count: int
    duplicate_rate: float
    missing_identifier_count: int
    advertised_total_min: int | None
    advertised_total_max: int | None
    advertised_pages_min: int | None
    advertised_pages_max: int | None
    terminal_observed: bool
    hard_failure_count: int
    failure_counts: tuple[tuple[str, int], ...]
    examples: tuple[dict[str, Any], ...]
    schema_version: int = PAGINATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "ok": self.ok,
            "page_count": self.page_count,
            "record_count": self.record_count,
            "unique_identifier_count": self.unique_identifier_count,
            "duplicate_identifier_count": self.duplicate_identifier_count,
            "duplicate_rate": self.duplicate_rate,
            "missing_identifier_count": self.missing_identifier_count,
            "advertised_total_min": self.advertised_total_min,
            "advertised_total_max": self.advertised_total_max,
            "advertised_pages_min": self.advertised_pages_min,
            "advertised_pages_max": self.advertised_pages_max,
            "terminal_observed": self.terminal_observed,
            "hard_failure_count": self.hard_failure_count,
            "failure_counts": dict(self.failure_counts),
            "examples": [dict(example) for example in self.examples],
        }


class PaginationReconciler:
    """Account for one crawl scope while keeping memory and evidence bounded."""

    def __init__(
        self,
        scope: str,
        *,
        max_unique_ids: int = DEFAULT_MAX_UNIQUE_IDS,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        max_duplicate_rate: float = DEFAULT_MAX_DUPLICATE_RATE,
    ) -> None:
        if not isinstance(scope, str) or not scope or len(scope) > 512:
            raise ValueError("scope must be a non-empty string of at most 512 characters")
        for name, value in (
            ("max_unique_ids", max_unique_ids),
            ("max_records", max_records),
            ("max_pages", max_pages),
            ("max_examples", max_examples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if (
            isinstance(max_duplicate_rate, bool)
            or not isinstance(max_duplicate_rate, (int, float))
            or not math.isfinite(float(max_duplicate_rate))
            or not 0 <= float(max_duplicate_rate) <= 1
        ):
            raise ValueError("max_duplicate_rate must be between 0 and 1")

        self.scope = scope
        self.max_unique_ids = max_unique_ids
        self.max_records = max_records
        self.max_pages = max_pages
        self.max_examples = max_examples
        self.max_duplicate_rate = float(max_duplicate_rate)
        self._identifiers: set[str] = set()
        self._page_cursors: set[str] = set()
        self._record_count = 0
        self._duplicates = 0
        self._missing_identifiers = 0
        self._advertised_totals: set[int] = set()
        self._advertised_pages: set[int] = set()
        self._terminal_observed = False
        self._hard_failure_count = 0
        self._failure_counts: dict[str, int] = {}
        self._examples: list[dict[str, Any]] = []
        self._capacity_exceeded = False
        self._record_capacity_exceeded = False
        self._finalized: PaginationOutcome | None = None

    @property
    def advertised_pages_max(self) -> int | None:
        return max(self._advertised_pages) if self._advertised_pages else None

    @staticmethod
    def _cursor(value: object) -> str:
        text = str(value)
        return text[:128] if text else "<empty>"

    @staticmethod
    def _identifier(value: object) -> str | None:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None
        text = str(value).strip()
        if not text or len(text) > 2048:
            return None
        return text

    def _add_failure(self, code: str, **details: Any) -> None:
        normalized_code = str(code)[:128]
        self._hard_failure_count += 1
        count_key = normalized_code
        if (
            count_key not in self._failure_counts
            and len(self._failure_counts) >= DEFAULT_MAX_FAILURE_CODES
        ):
            count_key = "other_failure_codes"
        self._failure_counts[count_key] = self._failure_counts.get(count_key, 0) + 1
        if len(self._examples) >= self.max_examples:
            return
        bounded: dict[str, Any] = {"code": normalized_code}
        for key, value in details.items():
            if isinstance(value, str):
                bounded[str(key)[:64]] = value[:300]
            elif value is None or isinstance(value, (bool, int, float)):
                bounded[str(key)[:64]] = value
            else:
                bounded[str(key)[:64]] = str(value)[:300]
        self._examples.append(bounded)

    def _advertised_value(self, value: object, *, field: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self._add_failure(f"invalid_{field}", actual=value)
            return None
        return value

    def observe_page(
        self,
        cursor: object,
        record_ids: Iterable[object],
        *,
        advertised_total: object = None,
        advertised_pages: object = None,
        page_number: int | None = None,
        terminal: bool = False,
    ) -> None:
        if self._finalized is not None:
            raise RuntimeError("cannot observe a finalized pagination scope")
        cursor_text = self._cursor(cursor)
        if cursor_text in self._page_cursors:
            self._add_failure("duplicate_page_callback", cursor=cursor_text)
            return
        if len(self._page_cursors) >= self.max_pages:
            self._add_failure(
                "page_tracking_capacity_exceeded",
                cursor=cursor_text,
                configured_cap=self.max_pages,
            )
            return
        self._page_cursors.add(cursor_text)

        if page_number is not None and (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            self._add_failure("invalid_page_number", actual=page_number)
            page_number = None
        if not isinstance(terminal, bool):
            self._add_failure("invalid_terminal_flag", actual=terminal)
            terminal = False

        total = self._advertised_value(advertised_total, field="advertised_total")
        pages = self._advertised_value(advertised_pages, field="advertised_pages")
        if total is not None:
            self._advertised_totals.add(total)
            if len(self._advertised_totals) > 1:
                self._add_failure(
                    "advertised_total_drift",
                    cursor=cursor_text,
                    minimum=min(self._advertised_totals),
                    maximum=max(self._advertised_totals),
                )
        if pages is not None:
            self._advertised_pages.add(pages)
            if len(self._advertised_pages) > 1:
                self._add_failure(
                    "advertised_pages_drift",
                    cursor=cursor_text,
                    minimum=min(self._advertised_pages),
                    maximum=max(self._advertised_pages),
                )

        unique_before_page = len(self._identifiers)
        records_on_page = 0
        for raw_identifier in record_ids:
            if self._record_count >= self.max_records:
                if not self._record_capacity_exceeded:
                    self._record_capacity_exceeded = True
                    self._add_failure(
                        "record_tracking_capacity_exceeded",
                        configured_cap=self.max_records,
                    )
                break
            records_on_page += 1
            self._record_count += 1
            identifier = self._identifier(raw_identifier)
            if identifier is None:
                self._missing_identifiers += 1
                continue
            if identifier in self._identifiers:
                self._duplicates += 1
                continue
            if len(self._identifiers) >= self.max_unique_ids:
                if not self._capacity_exceeded:
                    self._capacity_exceeded = True
                    self._add_failure(
                        "identifier_tracking_capacity_exceeded",
                        configured_cap=self.max_unique_ids,
                    )
                continue
            self._identifiers.add(identifier)

        expected_more = total is not None and unique_before_page < total
        if pages is not None and page_number is not None and page_number < pages:
            expected_more = True
        if records_on_page == 0 and expected_more:
            self._add_failure(
                "early_empty_page",
                cursor=cursor_text,
                page_number=page_number,
                advertised_total=total,
                advertised_pages=pages,
            )
        self._terminal_observed = self._terminal_observed or bool(terminal)

    def mark_failure(self, kind: str, *, cursor: object, detail: str = "") -> None:
        if self._finalized is not None:
            raise RuntimeError("cannot fail a finalized pagination scope")
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a non-empty string")
        self._add_failure(kind, cursor=self._cursor(cursor), detail=str(detail)[:300])

    def mark_cap(self, *, cursor: object, configured_cap: int) -> None:
        self.mark_failure(
            "pagination_cap_reached",
            cursor=cursor,
            detail=f"configured_cap={configured_cap}",
        )

    def finalize(self) -> PaginationOutcome:
        if self._finalized is not None:
            return self._finalized
        if not self._page_cursors:
            self._add_failure("no_pages_observed")
        if not self._terminal_observed:
            self._add_failure("scope_not_terminal")
        if self._missing_identifiers:
            self._add_failure(
                "missing_identifiers",
                count=self._missing_identifiers,
            )

        if self._advertised_totals and not self._capacity_exceeded:
            expected_total = max(self._advertised_totals)
            if len(self._identifiers) != expected_total:
                self._add_failure(
                    "advertised_total_mismatch",
                    expected=expected_total,
                    actual=len(self._identifiers),
                )
        if self._advertised_pages:
            expected_pages = max(self._advertised_pages)
            if len(self._page_cursors) != expected_pages:
                self._add_failure(
                    "advertised_page_count_mismatch",
                    expected=expected_pages,
                    actual=len(self._page_cursors),
                )

        duplicate_rate = (
            self._duplicates / self._record_count if self._record_count else 0.0
        )
        if duplicate_rate > self.max_duplicate_rate:
            self._add_failure(
                "duplicate_rate_exceeded",
                duplicate_count=self._duplicates,
                record_count=self._record_count,
                duplicate_rate=duplicate_rate,
                threshold=self.max_duplicate_rate,
            )

        self._finalized = PaginationOutcome(
            scope=self.scope,
            ok=self._hard_failure_count == 0,
            page_count=len(self._page_cursors),
            record_count=self._record_count,
            unique_identifier_count=len(self._identifiers),
            duplicate_identifier_count=self._duplicates,
            duplicate_rate=duplicate_rate,
            missing_identifier_count=self._missing_identifiers,
            advertised_total_min=(
                min(self._advertised_totals) if self._advertised_totals else None
            ),
            advertised_total_max=(
                max(self._advertised_totals) if self._advertised_totals else None
            ),
            advertised_pages_min=(
                min(self._advertised_pages) if self._advertised_pages else None
            ),
            advertised_pages_max=(
                max(self._advertised_pages) if self._advertised_pages else None
            ),
            terminal_observed=self._terminal_observed,
            hard_failure_count=self._hard_failure_count,
            failure_counts=tuple(sorted(self._failure_counts.items())),
            examples=tuple(dict(example) for example in self._examples),
        )
        return self._finalized


def get_pagination_reconciler(
    spider: object,
    scope: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> PaginationReconciler:
    """Return a lazily-created scope tracker without requiring base-spider changes."""
    trackers = getattr(spider, "_pagination_reconcilers", None)
    if trackers is None:
        trackers = {}
        setattr(spider, "_pagination_reconcilers", trackers)
    tracker = trackers.get(scope)
    if tracker is None:
        tracker = PaginationReconciler(scope, max_pages=max_pages)
        trackers[scope] = tracker
    return tracker


def _append_private_jsonl(path: Path, record: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        payload = (
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written < 1:
                raise OSError("short write while appending repair manifest")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def finalize_pagination_scope(
    spider: object,
    reconciler: PaginationReconciler,
    *,
    url: str,
    quality_failure_recorded: bool = False,
) -> PaginationOutcome:
    """Finalize once, record hard incompleteness, and append bounded repair evidence."""
    outcome = reconciler.finalize()
    reported = getattr(spider, "_reported_pagination_scopes", None)
    if reported is None:
        reported = set()
        setattr(spider, "_reported_pagination_scopes", reported)
    if reconciler.scope in reported:
        return outcome
    if outcome.ok:
        reported.add(reconciler.scope)
        return outcome

    codes = [code for code, _count in outcome.failure_counts]
    context = {
        "scope": outcome.scope,
        "pages": outcome.page_count,
        "records": outcome.record_count,
        "unique_identifiers": outcome.unique_identifier_count,
        "hard_failures": outcome.hard_failure_count,
        "failure_codes": codes,
    }
    if not quality_failure_recorded:
        spider.record_quality_failure(
            "pagination_incomplete",
            url,
            detail=", ".join(codes)[:300],
            context=context,
        )

    run_dir = getattr(spider, "run_dir", None)
    if run_dir is not None:
        record = {
            "schema_version": PAGINATION_SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "kind": "pagination_incomplete",
            "source": str(getattr(spider, "name", "unknown"))[:128],
            "url": str(url)[:8192],
            "reconciliation": outcome.to_dict(),
        }
        _append_private_jsonl(
            Path(run_dir) / REPAIR_MANIFEST_FILENAME,
            record,
        )
    reported.add(reconciler.scope)
    return outcome


def handle_pagination_request_failure(spider: object, failure: object) -> None:
    """Attach exhausted transport retries to their scope, then use the base errback."""
    request = failure.request
    spider.request_failed(failure)
    scope = request.meta.get("pagination_scope")
    if not scope:
        return
    tracker = get_pagination_reconciler(spider, scope)
    tracker.mark_failure(
        "exhausted_retries",
        cursor=request.meta.get("pagination_cursor", request.url),
        detail=str(getattr(failure, "value", "request failed")),
    )
    finalize_pagination_scope(
        spider,
        tracker,
        url=request.url,
        quality_failure_recorded=True,
    )
