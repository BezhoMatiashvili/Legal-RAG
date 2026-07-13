import json
import sys
import unittest
from datetime import date
from pathlib import Path

from scrapy import Request
from scrapy.http import TextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from legal_scrapers.spiders.constcourt_spider import ConstcourtSpider  # noqa: E402
from legal_scrapers.spiders.ecd_spider import TEXT_URL, EcdSpider  # noqa: E402
from legal_scrapers.spiders.napr_spider import NaprSpider  # noqa: E402
from legal_scrapers.spiders.supremecourt_spider import SupremecourtSpider  # noqa: E402
from legal_scrapers.spiders.tbappeal_spider import TbappealSpider  # noqa: E402


def json_response(url, payload, meta=None):
    request = Request(url=url, meta=meta or {})
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return TextResponse(url=url, body=body.encode("utf-8"), encoding="utf-8", request=request)


class SpiderConstructionTests(unittest.TestCase):
    """Every spider inherits BaseLegalSpider's date handling."""

    def test_all_spiders_share_date_args(self):
        for cls in (EcdSpider, ConstcourtSpider, NaprSpider, TbappealSpider, SupremecourtSpider):
            spider = cls(start_date="2024-01-01", end_date="2024-01-31")
            self.assertEqual(spider.scraping_start_date, date(2024, 1, 1))
            self.assertEqual(spider.scraping_end_date, date(2024, 1, 31))

    def test_supremecourt_overrides_robots(self):
        self.assertFalse(SupremecourtSpider.custom_settings["ROBOTSTXT_OBEY"])


class EcdParsingTests(unittest.TestCase):
    def test_parse_list_yields_detail_then_stops(self):
        spider = EcdSpider(start_date="2020-01-01", end_date="2020-12-31")
        record = {"InstanceId": 1, "DecisionDocumentId": 5861405, "Id": "1-5861405"}
        response = json_response(
            "https://ecd.court.ge/Decision/DecisionDocuments",
            {"success": True, "data": {"Total": 1, "Items": [record]}},
            meta={"instance_id": 1, "instance_name": "პირველი ინსტანცია", "skip": 0},
        )
        results = list(spider.parse_list(response))
        # One detail request, and no next page (Total == 1 < skip + PAGE_SIZE).
        self.assertEqual(len(results), 1)
        detail = results[0]
        self.assertEqual(detail.url, TEXT_URL)
        self.assertEqual(detail.method, "POST")
        self.assertEqual(json.loads(detail.body)["DecisionDocumentId"], 5861405)

    def test_parse_detail_builds_item(self):
        spider = EcdSpider()
        record = {
            "Id": "1-5861405",
            "DecisionDocumentId": 5861405,
            "InstanceId": 1,
            "CaseNo": "  330100119003015732  ",  # ItemLoader must strip this
            "CourtName": "თბილისის საქალაქო სასამართლო",
            "TypeName": "განაჩენი",
            "LitigationTypeName": ["126 (1) 1 "],  # list -> TakeFirst scalar (like matsne)
            "DecisionDate": "/Date(1588260918000)/",
        }
        response = json_response(
            TEXT_URL,
            {"success": True, "data": {"RawData": "line1\n\n\n\nline2"}},
            meta={"record": record, "instance_name": "პირველი ინსტანცია"},
        )
        item = next(iter(spider.parse_detail(response)))
        self.assertEqual(item["document_id"], "1-5861405")
        self.assertEqual(item["decision_date"], "2020-04-30")
        self.assertEqual(item["body_markdown"], "line1\n\nline2")
        # ItemLoader applied: whitespace trimmed, int preserved, list flattened via TakeFirst.
        self.assertEqual(item["case_no"], "330100119003015732")
        self.assertEqual(item["instance_id"], 1)
        self.assertEqual(item["litigation_type_name"], "126 (1) 1")


class TbappealWindowTests(unittest.TestCase):
    def test_in_window_filtering(self):
        spider = TbappealSpider(start_date="2018-01-01", end_date="2018-12-31")
        self.assertTrue(spider._in_window("05-02-2018"))
        self.assertFalse(spider._in_window("19-05-2017"))
        # Unparseable dates are kept rather than dropped.
        self.assertTrue(spider._in_window(""))


if __name__ == "__main__":
    unittest.main()
