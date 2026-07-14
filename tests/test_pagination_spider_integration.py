import asyncio
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from scrapy import Request
from scrapy.http import HtmlResponse, TextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.spiders.ecd_spider import DOCUMENTS_URL, EcdSpider  # noqa: E402
from legal_scrapers.spiders.napr_spider import SEARCH_URL, NaprSpider  # noqa: E402
from legal_scrapers.spiders.tas_spider import GRID_URL, TasSpider  # noqa: E402
from legal_scrapers.spiders.tbappeal_spider import (  # noqa: E402
    BASE,
    CATEGORY_PATH,
    TbappealSpider,
)
from legal_scrapers.utils.pagination import (  # noqa: E402
    REPAIR_MANIFEST_FILENAME,
)


def _text_response(url, payload, meta):
    request = Request(url, meta=meta)
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return TextResponse(
        url=url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        request=request,
    )


def test_ecd_complete_instance_scope_finalizes_cleanly():
    spider = EcdSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider.record_quality_failure = MagicMock()
    response = _text_response(
        DOCUMENTS_URL,
        {
            "data": {
                "Total": 1,
                "Items": [
                    {
                        "Id": "ecd-1",
                        "InstanceId": 1,
                        "DecisionDocumentId": 10,
                    }
                ],
            }
        },
        {"instance_id": 1, "instance_name": "first", "skip": 0},
    )

    requests = list(spider.parse_list(response))
    outcome = spider._pagination_reconcilers["instance:1"].finalize()

    assert len(requests) == 1
    assert outcome.ok
    assert outcome.unique_identifier_count == 1
    spider.record_quality_failure.assert_not_called()


def test_ecd_record_callback_failure_is_reconciled_and_repairable(tmp_path):
    spider = EcdSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider.run_dir = tmp_path
    spider.record_quality_failure = MagicMock()
    spider.is_seen = MagicMock(side_effect=RuntimeError("dedup unavailable"))
    response = _text_response(
        DOCUMENTS_URL,
        {
            "data": {
                "Total": 1,
                "Items": [
                    {
                        "Id": "ecd-1",
                        "InstanceId": 1,
                        "DecisionDocumentId": 10,
                    }
                ],
            }
        },
        {"instance_id": 1, "instance_name": "first", "skip": 0},
    )

    assert list(spider.parse_list(response)) == []
    outcome = spider._pagination_reconcilers["instance:1"].finalize()

    assert not outcome.ok
    assert any(
        example["code"] == "callback_failure" for example in outcome.examples
    )
    spider.record_quality_failure.assert_called_once()
    assert (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_ecd_page_cap_stops_followup_and_fails_scope():
    spider = EcdSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider.MAX_PAGES = 1
    spider.record_quality_failure = MagicMock()
    response = _text_response(
        DOCUMENTS_URL,
        {
            "data": {
                "Total": 51,
                "Items": [
                    {
                        "Id": "ecd-1",
                        "InstanceId": 1,
                        "DecisionDocumentId": 10,
                    }
                ],
            }
        },
        {"instance_id": 1, "instance_name": "first", "skip": 0},
    )

    requests = list(spider.parse_list(response))
    outcome = spider._pagination_reconcilers["instance:1"].finalize()

    assert len(requests) == 1
    assert requests[0].url != DOCUMENTS_URL
    assert not outcome.ok
    assert "pagination_cap_reached" in dict(outcome.failure_counts)
    spider.record_quality_failure.assert_called_once()


def test_napr_complete_catch_all_scope_finalizes_cleanly():
    spider = NaprSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider.record_quality_failure = MagicMock()
    payload = json.dumps(
        {
            "data": [
                {
                    "LETTERS_ID": "napr-1",
                    "ABOUT": "decision",
                    "PDF": "/uploads/decision.pdf",
                }
            ],
            "total": "1",
        }
    )
    response = _text_response(
        SEARCH_URL,
        json.dumps(payload),
        {"from_n": 0, "dispute_category": None},
    )

    requests = list(spider.parse_list(response))
    outcome = spider._pagination_reconcilers["catch_all"].finalize()

    assert len(requests) == 1
    assert outcome.ok
    assert outcome.unique_identifier_count == 1
    spider.record_quality_failure.assert_not_called()


class _FakeTasPage:
    def __init__(self):
        self.closed = False

    async def wait_for_function(self, *_args, **_kwargs):
        return None

    async def evaluate(self, *_args, **_kwargs):
        return {"total": 1, "source": [{"documentId": 77}]}

    async def wait_for_timeout(self, *_args, **_kwargs):
        return None

    async def close(self):
        self.closed = True


class _FailingTasPage(_FakeTasPage):
    async def evaluate(self, *_args, **_kwargs):
        raise TimeoutError("DWR timeout")


def test_tas_complete_dwr_scope_finalizes_cleanly():
    spider = TasSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider.record_quality_failure = MagicMock()
    spider._fetch_detail = AsyncMock(return_value=None)
    page = _FakeTasPage()
    response = SimpleNamespace(
        url=GRID_URL,
        meta={"playwright_page": page},
    )

    async def consume():
        return [item async for item in spider.parse_docs(response)]

    assert asyncio.run(consume()) == []
    outcome = spider._pagination_reconcilers[spider.pagination_scope()].finalize()
    assert outcome.ok
    assert outcome.unique_identifier_count == 1
    assert page.closed
    spider.record_quality_failure.assert_not_called()


def test_tas_due_refresh_runs_by_id_outside_listing_discovery():
    spider = TasSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider._refresh_keys = {"42"}
    spider.crawler = SimpleNamespace(stats=MagicMock())
    spider.record_quality_failure = MagicMock()
    spider._fetch_detail = AsyncMock(side_effect=[{"document": {}}, None])
    spider.build_item = MagicMock(return_value={"refreshed": "42"})
    page = _FakeTasPage()
    response = SimpleNamespace(url=GRID_URL, meta={"playwright_page": page})

    async def consume():
        return [item async for item in spider.parse_docs(response)]

    assert asyncio.run(consume()) == [{"refreshed": "42"}]
    assert spider._fetch_detail.await_args_list[0].args[1] == "42"
    spider.build_item.assert_any_call(
        {
            "documentId": "42",
            "documentNo": None,
            "address": None,
            "registrationDate": None,
            "createDateStr": None,
        },
        {"document": {}},
    )


def test_tas_exhausted_list_retries_emit_repair_manifest(tmp_path):
    spider = TasSpider(start_date="2024-01-01", end_date="2024-01-31")
    spider.run_dir = tmp_path
    spider.record_quality_failure = MagicMock()
    page = _FailingTasPage()
    response = SimpleNamespace(
        url=GRID_URL,
        meta={"playwright_page": page},
    )

    async def consume():
        return [item async for item in spider.parse_docs(response)]

    assert asyncio.run(consume()) == []
    outcome = spider._pagination_reconcilers[spider.pagination_scope()].finalize()
    assert not outcome.ok
    assert any(
        example["code"] == "exhausted_retries" for example in outcome.examples
    )
    spider.record_quality_failure.assert_called_once()
    assert (tmp_path / REPAIR_MANIFEST_FILENAME).exists()


def test_tbappeal_explicit_last_page_finalizes_cleanly():
    spider = TbappealSpider(start_date="2018-01-01", end_date="2018-12-31")
    spider.record_quality_failure = MagicMock()
    request = Request(
        f"{BASE}{CATEGORY_PATH}?page=1",
        meta={"page": 1},
    )
    body = (
        '<a rel="last" href="?page=1">last</a>'
        '<div class="grid-post"><a href="/ka/news/ruling-1">open</a>'
        '<span class="date">05-02-2018</span><h4><a>Ruling</a></h4></div>'
    )
    response = HtmlResponse(
        url=request.url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        request=request,
    )

    requests = list(spider.parse_list(response))
    outcome = spider._pagination_reconcilers[
        spider.pagination_scope()
    ].finalize()

    assert len(requests) == 1
    assert outcome.ok
    assert outcome.unique_identifier_count == 1
    spider.record_quality_failure.assert_not_called()


def test_tbappeal_remembers_explicit_last_page_when_last_link_disappears():
    spider = TbappealSpider(start_date="2018-01-01", end_date="2018-12-31")
    spider.record_quality_failure = MagicMock()

    def response(page, body):
        request = Request(
            f"{BASE}{CATEGORY_PATH}?page={page}",
            meta={"page": page},
        )
        return HtmlResponse(
            url=request.url,
            body=body.encode("utf-8"),
            encoding="utf-8",
            request=request,
        )

    first = list(
        spider.parse_list(
            response(
                1,
                '<a rel="last" href="?page=2">last</a>'
                '<div class="grid-post"><a href="/ka/news/ruling-1">open</a>'
                '<span class="date">05-02-2018</span></div>',
            )
        )
    )
    second = list(
        spider.parse_list(
            response(
                2,
                '<div class="grid-post"><a href="/ka/news/ruling-2">open</a>'
                '<span class="date">06-02-2018</span></div>',
            )
        )
    )

    assert any(request.url.endswith("?page=2") for request in first)
    assert not any(request.url.endswith("?page=3") for request in second)
    outcome = spider._pagination_reconcilers[
        spider.pagination_scope()
    ].finalize()
    assert outcome.ok
    assert outcome.page_count == 2


def test_napr_waf_scope_appends_private_repair_manifest(tmp_path):
    spider = NaprSpider()
    spider.run_dir = tmp_path
    spider.record_quality_failure = MagicMock()
    response = _text_response(
        SEARCH_URL,
        "<html><title>Access Denied</title></html>",
        {"from_n": 0, "dispute_category": "blocked"},
    )

    assert list(spider.parse_list(response)) == []

    spider.record_quality_failure.assert_called_once()
    assert spider.record_quality_failure.call_args.args[0] == "non_json_response"
    repair_path = tmp_path / REPAIR_MANIFEST_FILENAME
    assert stat.S_IMODE(repair_path.stat().st_mode) == 0o600
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    assert repair["reconciliation"]["scope"] == "category:blocked"
    assert not repair["reconciliation"]["ok"]
