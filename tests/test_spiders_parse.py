import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scrapy import Request
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse, TextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "matsne"))

from matsne.spiders.constcourt_spider import ConstcourtSpider, TEASER_MARKER  # noqa: E402
from matsne.spiders.matsne_spider import MatsneSpider  # noqa: E402
from matsne.spiders.napr_spider import NaprSpider  # noqa: E402
from matsne.spiders.supremecourt_spider import SupremecourtSpider  # noqa: E402


def _html(url, body, meta=None):
    req = Request(url=url, meta=meta or {})
    return HtmlResponse(url=url, body=body.encode("utf-8"), encoding="utf-8", request=req)


class MatsneTwoPhaseTests(unittest.TestCase):
    """Regression for A3: phase 2 is driven by spider_idle, not a per-callback counter."""

    def _spider(self):
        spider = MatsneSpider()
        spider.first_batch_urls = []
        spider.deferred_batch_urls = [
            "https://matsne.gov.ge/ka/document/search?a=1",
            "https://matsne.gov.ge/ka/document/search?a=2",
        ]
        spider.seen_request_urls = set()
        spider.deferred_batch_started = False
        spider.crawler = MagicMock()
        return spider

    def test_idle_schedules_deferred_batch_once(self):
        spider = self._spider()
        with self.assertRaises(DontCloseSpider):
            spider.spider_idle()
        self.assertEqual(spider.crawler.engine.crawl.call_count, 2)
        self.assertTrue(spider.deferred_batch_started)

        # A second idle must not re-schedule and must not block shutdown.
        spider.crawler.engine.crawl.reset_mock()
        self.assertIsNone(spider.spider_idle())
        spider.crawler.engine.crawl.assert_not_called()


class ConstcourtParseTests(unittest.TestCase):
    def test_teaser_body_triggers_docx_fetch(self):
        spider = ConstcourtSpider()
        body = (
            '<table><tr><td class="first-table-cell">დოკუმენტის ტიპი</td>'
            "<td>კონსტიტუციური სარჩელი</td></tr></table>"
            '<a href="/uploads/documents/abc.docx" download>doc</a>'
            f'<span class="legalactshowparagraph">{TEASER_MARKER} ...</span>'
        )
        resp = _html(
            "https://constcourt.ge/ka/judicial-acts?legal=1", body,
            {"legal_id": "1", "title": "T", "source_url": "https://constcourt.ge/ka/judicial-acts?legal=1"},
        )
        out = list(spider.parse_detail(resp))
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], Request)
        self.assertIn("/uploads/documents/abc.docx", out[0].url)

    def test_full_body_yields_item(self):
        spider = ConstcourtSpider()
        body = (
            '<table><tr><td class="first-table-cell">დოკუმენტის ტიპი</td>'
            "<td>განჩინება</td></tr></table>"
            '<span class="legalactshowparagraph">სრული ტექსტი აქ.</span>'
        )
        resp = _html(
            "https://constcourt.ge/ka/judicial-acts?legal=2", body,
            {"legal_id": "2", "title": "T2", "source_url": "https://constcourt.ge/ka/judicial-acts?legal=2"},
        )
        out = list(spider.parse_detail(resp))
        self.assertEqual(len(out), 1)
        item = out[0]
        self.assertEqual(item["legal_id"], "2")
        self.assertEqual(item["doc_type"], "განჩინება")
        self.assertIn("სრული ტექსტი", item["body_markdown"])


class SupremecourtParseTests(unittest.TestCase):
    def test_parse_list_extracts_labels_and_ids(self):
        spider = SupremecourtSpider(start_date="2024-01-01", end_date="2024-12-31")
        body = (
            '<div class="cases">'
            "<div><span>საქმის ნომერი:</span> ას-1</div>"
            "<div><span>თარიღი:</span> 2024-12-26</div>"
            "<span><span>დავის საგანი:</span> დავა</span>"
            "<div><span>შედეგი:</span> შედეგი1</div>"
            "<div><span>საჩივრის სახე:</span> საკასაციო</div>"
            '<a href="/ka/fullcase/73901/1">ნახვა</a>'
            "</div>"
        )
        req = Request(url="https://www.supremecourt.ge/ka/getCases?palata=1&page=1", meta={"palata": 1, "page": 1})
        resp = HtmlResponse(url=req.url, body=body.encode("utf-8"), encoding="utf-8", request=req)
        follows = [o for o in spider.parse_list(resp) if isinstance(o, Request) and "/fullcase/" in o.url]
        self.assertEqual(len(follows), 1)
        fields = follows[0].meta["fields"]
        self.assertEqual(fields["case_id"], "73901")
        self.assertEqual(fields["chamber"], "1")
        self.assertEqual(fields["case_number"], "ას-1")
        self.assertEqual(fields["date"], "2024-12-26")
        self.assertEqual(fields["appeal_type"], "საკასაციო")


class NaprParseTests(unittest.TestCase):
    def test_double_encoded_json_yields_pdf_follow(self):
        spider = NaprSpider(start_date="2024-01-01", end_date="2024-12-31")
        inner = {
            "data": [{
                "LETTERS_ID": "77", "RANDOMID": "r", "REGISTRATIONDATE": "2024-12-30 00:00:00",
                "SENDER": "s", "ABOUT": "a", "KANC_DATE": "2024-12-30 00:00:00", "KANC_NO": "9",
                "PDF": "/uploads/administrativeComplaints/file77.pdf",
            }],
            "total": "1",
        }
        body = json.dumps(json.dumps(inner))  # double-encoded, like the live endpoint
        req = Request(url="https://www.napr.gov.ge/legal_search", method="POST", meta={"from_n": 0})
        resp = TextResponse(url=req.url, body=body.encode("utf-8"), encoding="utf-8", request=req)
        reqs = [o for o in spider.parse_list(resp) if isinstance(o, Request)]
        self.assertTrue(any("file77.pdf" in r.url for r in reqs))


if __name__ == "__main__":
    unittest.main()
