import asyncio
import sys
import unittest
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scrapy import Request
from scrapy.http import HtmlResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.items import MatsneItem  # noqa: E402
from legal_scrapers.spiders.matsne_spider import MatsneSpider, status_from_effective_dates  # noqa: E402


def _html(url, body, meta=None):
    req = Request(url=url, meta=meta or {})
    return HtmlResponse(url=url, body=body.encode("utf-8"), encoding="utf-8", request=req)


DETAIL_CONSOLIDATED = """
<html><body>
<div id="block-system-main">
  <table class="table-info">
    <tr><td>დოკუმენტის ნომერი</td><td>432</td></tr>
    <tr><td>ძალის დაკარგვის თარიღი</td><td>01/07/2003</td></tr>
    <tr><td><select id="publication-switcher">
      <option selected value="2">08/05/2003</option>
      <option value="1">28/05/1999</option>
    </select></td></tr>
  </table>
</div>
<div id="maindoc">Body of the repealed, consolidated act.</div>
</body></html>
"""

DETAIL_PLAIN = """
<html><body>
<div id="block-system-main">
  <table class="table-info">
    <tr><td>დოკუმენტის ნომერი</td><td>55</td></tr>
  </table>
</div>
<div id="maindoc">Body of a one-shot in-force act.</div>
</body></html>
"""


class SeedUrlLoadingTests(unittest.TestCase):
    def test_seed_urls_expands_bare_ids_and_dedups(self):
        spider = MatsneSpider()
        spider.seed_urls = "18070, 18070 https://matsne.gov.ge/ka/document/view/14944"
        urls = spider._load_seed_urls()
        self.assertEqual(
            urls,
            [
                "https://matsne.gov.ge/ka/document/view/18070",
                "https://matsne.gov.ge/ka/document/view/14944",
            ],
        )

    def test_no_seed_args_returns_empty(self):
        self.assertEqual(MatsneSpider()._load_seed_urls(), [])


class SeedStartModeTests(unittest.TestCase):
    def test_start_yields_detail_requests_for_seeds(self):
        spider = MatsneSpider()
        spider.seed_urls = "18070, https://matsne.gov.ge/ka/document/view/14944"

        async def collect():
            return [r async for r in spider.start()]

        reqs = asyncio.run(collect())
        self.assertEqual(len(reqs), 2)
        self.assertTrue(all(r.callback == spider.parse_document for r in reqs))
        urls = {r.url for r in reqs}
        self.assertEqual(
            urls,
            {
                "https://matsne.gov.ge/ka/document/view/18070",
                "https://matsne.gov.ge/ka/document/view/14944",
            },
        )
        # meta carries a MatsneItem seeded with the document_url parse_document needs.
        self.assertEqual(reqs[0].meta["item"]["document_url"], reqs[0].url)
        self.assertTrue(spider.deferred_batch_started)  # no phase-2 catch-all in seed mode


class ParseDocumentConsolidationTests(unittest.TestCase):
    def test_status_derivation_respects_future_transitions(self):
        as_of = date(2026, 7, 12)
        self.assertEqual(
            status_from_effective_dates("13/07/2026", None, as_of=as_of),
            "ასამოქმედებელი აქტები",
        )
        self.assertEqual(
            status_from_effective_dates("01/01/2020", "13/07/2026", as_of=as_of),
            "ძალაში მყოფი აქტები",
        )
        self.assertEqual(
            status_from_effective_dates("01/01/2020", "12/07/2026", as_of=as_of),
            "ძალადაკარგული აქტები",
        )

    def test_consolidated_and_status_derived_from_expiry(self):
        spider = MatsneSpider()
        item = MatsneItem()
        item["document_url"] = "https://matsne.gov.ge/ka/document/view/18070"
        out = list(spider.parse_document(_html(item["document_url"], DETAIL_CONSOLIDATED, {"item": item})))

        self.assertEqual(len(out), 1)
        doc = out[0]
        self.assertEqual(doc["document_id"], "18070")
        self.assertEqual(doc["language"], "ka")
        self.assertTrue(doc["is_consolidated"])
        self.assertEqual(doc["consolidated_count"], 2)
        self.assertEqual(doc["consolidated_dates"], ["08/05/2003", "28/05/1999"])
        # Seed item had no status; an expiry date is present → repealed.
        self.assertEqual(doc["status"], "ძალადაკარგული აქტები")

    def test_plain_doc_not_consolidated_and_in_force(self):
        spider = MatsneSpider()
        item = MatsneItem()
        item["document_url"] = "https://matsne.gov.ge/ka/document/view/55"
        doc = list(spider.parse_document(_html(item["document_url"], DETAIL_PLAIN, {"item": item})))[0]

        self.assertFalse(doc["is_consolidated"])
        self.assertEqual(doc["consolidated_count"], 0)
        self.assertEqual(doc["consolidated_dates"], [])
        self.assertEqual(doc["status"], "ძალაში მყოფი აქტები")  # no expiry → in force

    def test_existing_status_not_overwritten(self):
        spider = MatsneSpider()
        item = MatsneItem()
        item["document_url"] = "https://matsne.gov.ge/ka/document/view/18070"
        item["status"] = "ძალაში მყოფი აქტები"  # already set from the search listing panel
        doc = list(spider.parse_document(_html(item["document_url"], DETAIL_CONSOLIDATED, {"item": item})))[0]
        # Even though DETAIL_CONSOLIDATED carries an expiry date, the listing status wins.
        self.assertEqual(doc["status"], "ძალაში მყოფი აქტები")


class AdaptiveSplitTests(unittest.TestCase):
    def _search_url(self, fr, to):
        return (
            "https://matsne.gov.ge/ka/document/search"
            f"?publishing_date_fr%5Bdate%5D={fr}&publishing_date_to%5Bdate%5D={to}"
            "&type=all&page=1&limit=100&label=&additional_status="
        )

    def _split(self, spider, url, last_page):
        body = f'<ul class="pagination"><li><a href="?page={last_page}">ბოლო</a></li></ul>'
        resp = _html(url, body)
        return spider._split_requests(resp, parse_qs(urlparse(url).query))

    def test_deep_wide_window_splits_into_months(self):
        spider = MatsneSpider()
        spider.seen_request_urls = set()
        reqs = self._split(spider, self._search_url("01-01-2015", "31-12-2015"), 200)
        self.assertIsNotNone(reqs)
        self.assertEqual(len(reqs), 12)  # 12 monthly sub-windows
        self.assertTrue(all(r.callback == spider.parse for r in reqs))
        # each sub-request is a page-1 monthly search
        months = {parse_qs(urlparse(r.url).query)["publishing_date_fr[date]"][0][3:] for r in reqs}
        self.assertEqual(months, {f"{m:02d}-2015" for m in range(1, 13)})

    def test_live_last_link_with_arrow_suffix_still_splits(self):
        spider = MatsneSpider()
        spider.seen_request_urls = set()
        url = self._search_url("01-01-2015", "31-12-2015")
        body = '<ul class="pagination"><li><a href="?page=200">ბოლო »</a></li></ul>'
        response = _html(url, body)

        requests = spider._split_requests(response, parse_qs(urlparse(url).query))

        self.assertEqual(len(requests), 12)

    def test_shallow_window_does_not_split(self):
        spider = MatsneSpider()
        spider.seen_request_urls = set()
        self.assertIsNone(self._split(spider, self._search_url("01-01-2015", "31-12-2015"), 5))

    def test_already_monthly_window_does_not_split_even_if_deep(self):
        spider = MatsneSpider()
        spider.seen_request_urls = set()
        self.assertIsNone(self._split(spider, self._search_url("01-01-2015", "31-01-2015"), 200))


class CatchAllOnlyModeTests(unittest.TestCase):
    def test_catch_all_only_yields_one_pure_catchall_per_year(self):
        spider = MatsneSpider(start_date="2018-01-01", end_date="2020-12-31")
        spider.catch_all_only = "1"

        async def collect():
            return [r async for r in spider.start()]

        reqs = asyncio.run(collect())
        self.assertEqual(len(reqs), 3)  # 2018, 2019, 2020
        for r in reqs:
            q = parse_qs(urlparse(r.url).query, keep_blank_values=True)
            self.assertEqual(q.get("label", [""])[0], "")            # empty topic
            self.assertEqual(q.get("additional_status", [""])[0], "")  # empty status
            self.assertEqual(r.callback, spider.parse)
        self.assertTrue(spider.deferred_batch_started)  # no phase-2 transition

    def test_without_flag_runs_normal_phase1(self):
        spider = MatsneSpider(start_date="2020-01-01", end_date="2020-12-31")

        async def collect():
            return [r async for r in spider.start()]

        reqs = asyncio.run(collect())
        # normal phase 1 = non-empty topic×status cells → far more than 1/year
        self.assertGreater(len(reqs), 3)


if __name__ == "__main__":
    unittest.main()
