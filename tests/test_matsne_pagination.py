"""Synthetic Matsne listing reconciliation; no crawl or network required."""

from __future__ import annotations

import html
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

from scrapy import Request
from scrapy.http import HtmlResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.run import crawl_quality_issues  # noqa: E402
from legal_scrapers.spiders.matsne_spider import MatsneSpider  # noqa: E402
from legal_scrapers.utils.pagination import REPAIR_MANIFEST_FILENAME  # noqa: E402


class _Stats:
    def __init__(self):
        self.values = {}

    def inc_value(self, key, count=1):
        self.values[key] = self.values.get(key, 0) + count

    def get_stats(self):
        return dict(self.values)


def _spider(tmp_path: Path | None = None, *, doc_type: str = "all"):
    spider = MatsneSpider()
    spider.doc_type = doc_type
    spider.seen_request_urls = set()
    stats = _Stats()
    crawler = SimpleNamespace(
        stats=stats,
        spider=spider,
        spidercls=SimpleNamespace(name="matsne"),
    )
    spider.crawler = crawler
    if tmp_path is not None:
        spider.run_dir = tmp_path
    return spider, crawler


def _url(
    page: int,
    *,
    start: str = "01-01-2024",
    end: str = "31-01-2024",
    label: str = "",
    doc_type: str = "all",
) -> str:
    return (
        "https://matsne.gov.ge/ka/document/search"
        f"?publishing_date_fr%5Bdate%5D={start}"
        f"&publishing_date_to%5Bdate%5D={end}"
        f"&type={doc_type}&page={page}&limit=100"
        f"&label={label}&additional_status="
    )


def _response(
    url: str,
    identifiers: list[str | None],
    *,
    last_page: int | None = None,
    next_url: str | None = None,
    meta: dict | None = None,
) -> HtmlResponse:
    rows = []
    for identifier in identifiers:
        if identifier is None:
            rows.append('<li class="acts">missing link</li>')
        else:
            rows.append(
                '<li class="acts">'
                f'<a href="/ka/document/view/{identifier}">Act {identifier}</a>'
                "</li>"
            )
    pagination = []
    if last_page is not None:
        pagination.append(
            f'<a rel="last" href="{html.escape(_url(last_page), quote=True)}">ბოლო »</a>'
        )
    if next_url is not None:
        pagination.append(
            f'<a href="{html.escape(next_url, quote=True)}">შემდეგი »</a>'
        )
    body = (
        '<ul class="list-unstyled document-search-result-items">'
        + "".join(rows)
        + "</ul><ul class=\"pagination\">"
        + "".join(pagination)
        + "</ul>"
    )
    request = Request(url, meta=meta or {})
    return HtmlResponse(
        url=url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        request=request,
    )


def _listing_request(outputs):
    return next(request for request in outputs if request.callback.__name__ == "parse")


def _codes(outcome):
    return set(dict(outcome.failure_counts))


def test_scope_is_stable_across_pages_and_sensitive_to_filters():
    page_one = _url(1, label="code")
    page_two = _url(2, label="code")

    assert MatsneSpider.pagination_scope(page_one) == MatsneSpider.pagination_scope(
        page_two
    )
    assert MatsneSpider.pagination_scope(page_one) != MatsneSpider.pagination_scope(
        _url(1, label="decree")
    )


def test_exact_multi_page_listing_accounts_ids_before_document_dedup(tmp_path):
    spider, _crawler = _spider(tmp_path)
    spider.is_seen = lambda _fields: True
    first_url = _url(1)
    second_url = _url(2)

    first_outputs = list(
        spider.parse(
            _response(
                first_url,
                ["101", "102"],
                last_page=2,
                next_url=second_url,
            )
        )
    )
    next_request = _listing_request(first_outputs)
    second_outputs = list(
        spider.parse(
            _response(
                second_url,
                ["103"],
                last_page=2,
                meta=next_request.meta,
            )
        )
    )
    scope = MatsneSpider.pagination_scope(first_url)
    outcome = spider._pagination_reconcilers[scope].finalize()

    assert second_outputs == []
    assert outcome.ok
    assert outcome.page_count == 2
    assert outcome.record_count == 3
    assert outcome.unique_identifier_count == 3
    assert outcome.advertised_pages_min == 2
    assert not (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_early_empty_page_is_hard_failure_and_makes_combined_run_nonzero(tmp_path):
    spider, crawler = _spider(tmp_path)
    url = _url(1)

    assert list(spider.parse(_response(url, [], last_page=3))) == []
    scope = MatsneSpider.pagination_scope(url)
    outcome = spider._pagination_reconcilers[scope].finalize()

    assert not outcome.ok
    assert {
        "early_empty_page",
        "scope_not_terminal",
        "advertised_page_count_mismatch",
    } <= _codes(outcome)
    repair_path = tmp_path / REPAIR_MANIFEST_FILENAME
    assert stat.S_IMODE(repair_path.stat().st_mode) == 0o600
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    assert repair["reconciliation"]["scope"] == scope
    crawler.stats.values["finish_reason"] = "finished"
    assert any("completeness-affecting" in issue for issue in crawl_quality_issues([crawler]))


def test_waf_like_listing_is_repairable_and_never_green(tmp_path):
    spider, _crawler = _spider(tmp_path)
    url = _url(1)
    request = Request(url)
    response = HtmlResponse(
        url=url,
        body=b"<html><title>Access Denied</title><body>request blocked</body></html>",
        encoding="utf-8",
        request=request,
    )

    assert list(spider.parse(response)) == []
    outcome = spider._pagination_reconcilers[
        MatsneSpider.pagination_scope(url)
    ].finalize()

    assert not outcome.ok
    assert "waf_or_invalid_listing" in _codes(outcome)
    assert (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_exhausted_missing_page_uses_listing_scope_errback(tmp_path):
    spider, _crawler = _spider(tmp_path)
    url = _url(2)
    request = spider.build_request(url, callback=spider.parse)
    failure = SimpleNamespace(request=request, value=TimeoutError("retry exhausted"))

    assert request.errback == spider.pagination_request_failed
    spider.pagination_request_failed(failure)
    scope = MatsneSpider.pagination_scope(url)
    outcome = spider._pagination_reconcilers[scope].finalize()

    assert not outcome.ok
    assert "exhausted_retries" in _codes(outcome)
    assert (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_page_number_jump_is_incomplete_without_an_advertised_count(tmp_path):
    spider, _crawler = _spider(tmp_path)
    first_url = _url(1)
    third_url = _url(3)
    first_outputs = list(
        spider.parse(
            _response(
                first_url,
                ["151"],
                next_url=third_url,
            )
        )
    )
    jumped_request = _listing_request(first_outputs)

    list(
        spider.parse(
            _response(
                third_url,
                ["153"],
                meta=jumped_request.meta,
            )
        )
    )
    outcome = spider._pagination_reconcilers[
        MatsneSpider.pagination_scope(first_url)
    ].finalize()

    assert not outcome.ok
    assert {"page_cursor_mismatch", "non_contiguous_page_number"} <= _codes(
        outcome
    )


def test_duplicate_ids_and_advertised_page_drift_fail_scope(tmp_path):
    spider, _crawler = _spider(tmp_path)
    first_url = _url(1)
    second_url = _url(2)
    first_outputs = list(
        spider.parse(
            _response(
                first_url,
                ["201", "202"],
                last_page=2,
                next_url=second_url,
            )
        )
    )
    next_request = _listing_request(first_outputs)

    list(
        spider.parse(
            _response(
                second_url,
                ["202", "203"],
                last_page=3,
                meta=next_request.meta,
            )
        )
    )
    outcome = spider._pagination_reconcilers[
        MatsneSpider.pagination_scope(first_url)
    ].finalize()

    assert not outcome.ok
    assert {
        "advertised_pages_drift",
        "duplicate_rate_exceeded",
        "advertised_page_count_mismatch",
    } <= _codes(outcome)
    assert outcome.duplicate_identifier_count == 1


def test_wide_parent_is_explicitly_superseded_by_complete_monthly_scopes(tmp_path):
    spider, crawler = _spider(tmp_path)
    parent_url = _url(1, start="01-01-2024", end="31-12-2024")
    parent_scope = MatsneSpider.pagination_scope(parent_url)

    child_requests = list(
        spider.parse(_response(parent_url, ["ignored"], last_page=200))
    )

    assert len(child_requests) == 12
    assert parent_scope not in spider._pagination_reconcilers
    superseded = spider._pagination_superseded_scopes[parent_scope]
    assert superseded["reason"] == "monthly_split"
    assert superseded["advertised_pages"] == 200
    assert len(superseded["child_scopes"]) == 12
    assert crawler.stats.values["pagination/scopes_superseded"] == 1

    for child_request in child_requests:
        assert list(
            spider.parse(
                _response(
                    child_request.url,
                    [],
                    meta=child_request.meta,
                )
            )
        ) == []
    spider.spider_closed("finished")

    assert all(
        tracker.finalize().ok
        for tracker in spider._pagination_reconcilers.values()
    )
    assert crawler.stats.values.get("quality/failures", 0) == 0
    assert not (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_monthly_scope_above_hard_cap_stops_and_fails(tmp_path):
    spider, _crawler = _spider(tmp_path)
    spider.MAX_PAGES = 2
    first_url = _url(1)
    outputs = list(
        spider.parse(
            _response(
                first_url,
                ["301"],
                last_page=3,
                next_url=_url(2),
            )
        )
    )
    outcome = spider._pagination_reconcilers[
        MatsneSpider.pagination_scope(first_url)
    ].finalize()

    assert not any(request.callback == spider.parse for request in outputs)
    assert "pagination_cap_reached" in _codes(outcome)


def test_failed_main_listing_never_writes_completion_sentinel(tmp_path):
    spider, _crawler = _spider(tmp_path, doc_type="main")
    url = _url(1, doc_type="main")
    response = HtmlResponse(
        url=url,
        body=b"<html><title>Access Denied</title></html>",
        encoding="utf-8",
        request=Request(url),
    )

    assert list(spider.parse(response)) == []
    spider.spider_closed("finished")

    assert not (tmp_path / "main_listed_ids.txt.complete").exists()
