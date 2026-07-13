import asyncio
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from scrapy import Request
from scrapy.http import HtmlResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.items import MatsneItem  # noqa: E402
from legal_scrapers.spiders.matsne_spider import MatsneSpider  # noqa: E402
from legal_scrapers.utils.search_urls import (  # noqa: E402
    build_search_url,
    generate_start_url_batches,
)


def _html(url, body, meta=None):
    req = Request(url=url, meta=meta or {})
    return HtmlResponse(url=url, body=body.encode("utf-8"), encoding="utf-8", request=req)


LISTING = """
<html><body>
<ul class="list-unstyled document-search-result-items">
  <li class="acts"><a href="/ka/document/view/111">Act one</a></li>
  <li class="acts"><a href="/ka/document/view/222">Act two</a></li>
</ul>
</body></html>
"""

DETAIL_NO_SWITCHER = """
<html><body>
<div id="block-system-main">
  <table class="table-info">
    <tr><td>დოკუმენტის ნომერი</td><td>17</td></tr>
  </table>
</div>
<div id="maindoc">A never-amended base act: no publication switcher.</div>
</body></html>
"""


class BuildSearchUrlDocTypeTests(unittest.TestCase):
    # The exact URL the template produced before doc_type existed — the default
    # must stay byte-identical so every existing crawl mode is unaffected.
    LEGACY_URL = (
        "https://matsne.gov.ge/ka/document/search"
        "?publishing_date_fr%5Bdate%5D=01-01-2019&publishing_date_to%5Bdate%5D=31-12-2019"
        "&type=all&page=1&limit=100&label=კოდექსები&additional_status=ნორმატიული"
    )

    def test_default_is_byte_identical_to_legacy_url(self):
        url = build_search_url(date(2019, 1, 1), date(2019, 12, 31), "კოდექსები", "ნორმატიული")
        self.assertEqual(url, self.LEGACY_URL)

    def test_main_doc_type_sets_type_param(self):
        url = build_search_url(date(2019, 1, 1), date(2019, 12, 31), "", "", doc_type="main")
        q = parse_qs(urlparse(url).query, keep_blank_values=True)
        self.assertEqual(q["type"][0], "main")

    def test_unknown_doc_type_raises(self):
        with self.assertRaises(ValueError):
            build_search_url(date(2019, 1, 1), date(2019, 12, 31), "", "", doc_type="amendments")

    def test_generate_batches_threads_doc_type(self):
        first, deferred = generate_start_url_batches(
            date(2019, 1, 1), date(2019, 12, 31), doc_type="main"
        )
        for url in first + deferred:
            q = parse_qs(urlparse(url).query, keep_blank_values=True)
            self.assertEqual(q["type"][0], "main")


class SpiderDocTypeArgTests(unittest.TestCase):
    def test_default_doc_type_is_all(self):
        self.assertEqual(MatsneSpider()._doc_type(), "all")

    def test_main_doc_type_accepted(self):
        spider = MatsneSpider()
        spider.doc_type = "main"
        self.assertEqual(spider._doc_type(), "main")

    def test_bad_doc_type_fails_start_fast(self):
        spider = MatsneSpider()
        spider.doc_type = "consolidated"

        async def collect():
            return [r async for r in spider.start()]

        with self.assertRaisesRegex(ValueError, "doc_type"):
            asyncio.run(collect())

    def test_catch_all_only_urls_carry_main_type(self):
        spider = MatsneSpider(start_date="2019-01-01", end_date="2020-12-31")
        spider.catch_all_only = "1"
        spider.doc_type = "main"

        async def collect():
            return [r async for r in spider.start()]

        reqs = asyncio.run(collect())
        self.assertEqual(len(reqs), 2)  # 2019, 2020
        for r in reqs:
            q = parse_qs(urlparse(r.url).query, keep_blank_values=True)
            self.assertEqual(q["type"][0], "main")


class SplitPreservesDocTypeTests(unittest.TestCase):
    def _search_url(self, doc_type):
        return (
            "https://matsne.gov.ge/ka/document/search"
            "?publishing_date_fr%5Bdate%5D=01-01-2015&publishing_date_to%5Bdate%5D=31-12-2015"
            f"&type={doc_type}&page=1&limit=100&label=&additional_status="
        )

    def _split(self, url):
        spider = MatsneSpider()
        spider.seen_request_urls = set()
        body = '<ul class="pagination"><li><a href="?page=200">ბოლო</a></li></ul>'
        resp = _html(url, body)
        return spider._split_requests(resp, parse_qs(urlparse(url).query))

    def test_monthly_resplit_keeps_type_main(self):
        reqs = self._split(self._search_url("main"))
        self.assertEqual(len(reqs), 12)
        for r in reqs:
            q = parse_qs(urlparse(r.url).query, keep_blank_values=True)
            self.assertEqual(q["type"][0], "main")

    def test_monthly_resplit_keeps_type_all(self):
        reqs = self._split(self._search_url("all"))
        self.assertEqual(len(reqs), 12)
        for r in reqs:
            q = parse_qs(urlparse(r.url).query, keep_blank_values=True)
            self.assertEqual(q["type"][0], "all")


class MainListedIdsSidecarTests(unittest.TestCase):
    LISTING_URL = (
        "https://matsne.gov.ge/ka/document/search"
        "?publishing_date_fr%5Bdate%5D=01-01-2015&publishing_date_to%5Bdate%5D=31-12-2015"
        "&type=main&page=1&limit=100&label=&additional_status="
    )

    def _spider(self, run_dir):
        spider = MatsneSpider()
        spider.doc_type = "main"
        spider.seen_request_urls = set()
        spider.run_dir = Path(run_dir)
        return spider

    def test_listed_ids_recorded_even_for_seen_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            spider = self._spider(tmp)
            spider.is_seen = lambda mapping: True  # every doc already scraped
            spider.crawler = SimpleNamespace(stats=SimpleNamespace(inc_value=lambda key: None))
            out = list(spider.parse(_html(self.LISTING_URL, LISTING)))
            self.assertEqual(out, [])  # nothing fetched — all deduped
            ids = (Path(tmp) / "main_listed_ids.txt").read_text(encoding="utf-8").split()
            self.assertEqual(ids, ["111", "222"])

    def test_listed_ids_deduped_across_listings(self):
        with tempfile.TemporaryDirectory() as tmp:
            spider = self._spider(tmp)
            list(spider.parse(_html(self.LISTING_URL, LISTING)))
            list(spider.parse(_html(self.LISTING_URL + "&page=2", LISTING)))
            ids = (Path(tmp) / "main_listed_ids.txt").read_text(encoding="utf-8").split()
            self.assertEqual(ids, ["111", "222"])

    def test_all_mode_writes_no_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            spider = self._spider(tmp)
            spider.doc_type = "all"
            url = self.LISTING_URL.replace("type=main", "type=all")
            list(spider.parse(_html(url, LISTING)))
            self.assertFalse((Path(tmp) / "main_listed_ids.txt").exists())


class MainModeConsolidationStampTests(unittest.TestCase):
    def _parse_detail(self, main_listed):
        spider = MatsneSpider()
        item = MatsneItem()
        item["document_url"] = "https://matsne.gov.ge/ka/document/view/6937590"
        meta = {"item": item, "main_listed": main_listed}
        resp = _html(item["document_url"], DETAIL_NO_SWITCHER, meta)
        return list(spider.parse_document(resp))[0]

    def test_main_listed_meta_forces_is_consolidated_true_without_switcher(self):
        doc = self._parse_detail(main_listed=True)
        self.assertTrue(doc["is_consolidated"])
        self.assertEqual(doc["consolidated_count"], 0)  # count stays switcher-derived

    def test_without_main_listed_meta_switcher_semantics_hold(self):
        doc = self._parse_detail(main_listed=False)
        self.assertFalse(doc["is_consolidated"])

    def test_listing_detail_requests_carry_main_listed_meta(self):
        for doc_type, expected in (("main", True), ("all", False)):
            spider = MatsneSpider()
            spider.doc_type = doc_type
            spider.seen_request_urls = set()
            url = (
                "https://matsne.gov.ge/ka/document/search"
                "?publishing_date_fr%5Bdate%5D=01-01-2015&publishing_date_to%5Bdate%5D=31-12-2015"
                f"&type={doc_type}&page=1&limit=100&label=&additional_status="
            )
            reqs = [r for r in spider.parse(_html(url, LISTING)) if r.callback == spider.parse_document]
            self.assertEqual(len(reqs), 2)
            for r in reqs:
                self.assertEqual(r.meta["main_listed"], expected, doc_type)

    def test_seed_mode_with_doc_type_main_marks_seeds_listed(self):
        spider = MatsneSpider()
        spider.doc_type = "main"
        spider.seed_urls = "6937590"

        async def collect():
            return [r async for r in spider.start()]

        reqs = asyncio.run(collect())
        self.assertEqual(len(reqs), 1)
        self.assertTrue(reqs[0].meta["main_listed"])


class CompletionSentinelTests(unittest.TestCase):
    def _spider(self, run_dir, doc_type="main"):
        spider = MatsneSpider()
        spider.doc_type = doc_type
        spider.run_dir = Path(run_dir)
        spider._main_listed_ids = {"111", "222", "333"}
        return spider

    def test_finished_main_crawl_writes_sentinel_with_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._spider(tmp).spider_closed("finished")
            sentinel = Path(tmp) / "main_listed_ids.txt.complete"
            self.assertEqual(sentinel.read_text(encoding="utf-8").strip(), "3")

    def test_aborted_crawl_writes_no_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._spider(tmp).spider_closed("shutdown")
            self.assertFalse((Path(tmp) / "main_listed_ids.txt.complete").exists())

    def test_all_mode_writes_no_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._spider(tmp, doc_type="all").spider_closed("finished")
            self.assertFalse((Path(tmp) / "main_listed_ids.txt.complete").exists())


if __name__ == "__main__":
    unittest.main()
