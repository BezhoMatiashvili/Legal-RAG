import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scrapy import Request
from scrapy.http import TextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.spiders.napr_spider import SEARCH_URL, NaprSpider  # noqa: E402

WAF_HTML = "<html>\n<head>\n    <title>Access Denied</title>\n</head><body>...</body></html>"


def _response(body, meta):
    req = Request(url=SEARCH_URL, meta=meta)
    return TextResponse(url=SEARCH_URL, body=body.encode("utf-8"), encoding="utf-8", request=req)


class NaprWafResponseTests(unittest.TestCase):
    def test_waf_html_listing_is_recorded_as_quality_failure(self):
        spider = NaprSpider()
        counts = {}
        spider.crawler = types.SimpleNamespace(stats=types.SimpleNamespace(
            inc_value=lambda key: counts.__setitem__(key, counts.get(key, 0) + 1)
        ))
        meta = {"from_n": 0, "dispute_category": "სისტემური რეგისტრაცია (ირიგაციის არეალი)"}
        out = list(spider.parse_list(_response(WAF_HTML, meta)))
        self.assertEqual(out, [])
        self.assertEqual(counts["quality/failures"], 1)
        self.assertEqual(counts["quality/non_json_response"], 1)

    def test_valid_double_encoded_json_still_parses(self):
        spider = NaprSpider()
        spider.seen_request_urls = set()
        body = '"{\\"data\\": [], \\"total\\": 0}"'
        out = list(spider.parse_list(_response(body, {"from_n": 0, "dispute_category": None})))
        self.assertEqual(out, [])  # empty result set, but the parse path succeeds

    def test_pdf_parse_failure_records_quality_failure_and_emits_no_item(self):
        spider = NaprSpider()
        spider.record_quality_failure = MagicMock()
        request = Request(
            url="https://www.napr.gov.ge/bad.pdf",
            meta={"fields": {"document_id": "7", "title": "decision"}},
        )
        response = TextResponse(url=request.url, body=b"bad", request=request)

        with patch(
            "legal_scrapers.spiders.napr_spider.pdf_to_markdown",
            side_effect=ValueError("bad pdf"),
        ):
            out = list(spider.parse_pdf(response))

        self.assertEqual(out, [])
        spider.record_quality_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
