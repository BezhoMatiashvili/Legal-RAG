import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scrapy import Request
from scrapy.http import HtmlResponse, TextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.spiders.tbappeal_spider import TbappealSpider  # noqa: E402
from legal_scrapers.utils.documents import (  # noqa: E402
    PAGE_COORDINATE_REASON_EXACT_PDF_TEXT,
    ExtractionLimits,
    ExtractionResult,
    ExtractionStatus,
    PageBoundary,
    _pdf_text_result,
    docx_to_markdown,
    pdf_to_markdown,
)


def _html(body: str):
    request = Request(
        "https://tbappeal.court.ge/ka/news/example",
        meta={
            "source_url": "https://tbappeal.court.ge/ka/news/example",
            "slug": "example",
            "title": "Fallback title",
            "date": "05-02-2018",
            "featured_image_url": None,
        },
    )
    return HtmlResponse(
        url=request.url,
        body=body.encode(),
        encoding="utf-8",
        request=request,
    )


class TypedExtractionTests(unittest.TestCase):
    def test_pdf_page_boundaries_are_exact_offsets_in_emitted_markdown(self):
        result = _pdf_text_result(
            ["  პირველი  \nხაზი \n", "\n მეორე\t\n"],
            limits=ExtractionLimits(max_output_chars=1_000),
            truncated=False,
            mime="application/pdf",
        )

        self.assertEqual(result.status, ExtractionStatus.FULL_TEXT)
        self.assertEqual(result.text, "პირველი\nხაზი\n\nმეორე")
        self.assertEqual(
            result.page_coordinate_reason,
            PAGE_COORDINATE_REASON_EXACT_PDF_TEXT,
        )
        self.assertEqual(
            result.page_boundaries,
            (
                PageBoundary(page_number=1, char_start=0, char_end=12),
                PageBoundary(page_number=2, char_start=14, char_end=19),
            ),
        )
        for boundary, expected in zip(
            result.page_boundaries, ("პირველი\nხაზი", "მეორე"), strict=True
        ):
            self.assertEqual(
                result.text[boundary.char_start : boundary.char_end], expected
            )

    def test_pdf_rejects_wrong_mime_and_signature_before_parser(self):
        result = pdf_to_markdown(b"<html>blocked</html>", declared_mime="text/html")
        self.assertEqual(result.status, ExtractionStatus.MALFORMED)
        self.assertFalse(result.content_complete)

        result = pdf_to_markdown(b"not a pdf", declared_mime="application/pdf")
        self.assertEqual(result.status, ExtractionStatus.MALFORMED)
        self.assertIn("signature", result.detail)

    def test_docx_rejects_oversize_before_parser(self):
        limits = ExtractionLimits(max_input_bytes=3)
        result = docx_to_markdown(b"PK\x03\x04too-big", limits=limits)
        self.assertEqual(result.status, ExtractionStatus.RESOURCE_LIMITED)

    def test_valid_signature_delegates_to_isolated_worker(self):
        expected = ExtractionResult(
            text="full ruling",
            status=ExtractionStatus.FULL_TEXT,
            content_complete=True,
            detected_mime="application/pdf",
        )
        with patch(
            "legal_scrapers.utils.documents._run_isolated_extraction",
            return_value=expected,
        ) as isolated:
            result = pdf_to_markdown(
                b"%PDF-1.7\nbody", declared_mime="application/pdf; charset=binary"
            )
        self.assertEqual(result, expected)
        self.assertEqual(isolated.call_args.args[0], "pdf")


class TbappealRulingTests(unittest.TestCase):
    def _spider(self):
        spider = TbappealSpider(start_date="2018-01-01", end_date="2018-12-31")
        spider.record_quality_failure = MagicMock()
        return spider

    def test_detail_follows_pdf_and_keeps_article_as_summary(self):
        spider = self._spider()
        response = _html(
            '<h2 class="mb-5">Ruling title</h2>'
            '<div class="blog-details"><p>Article summary</p>'
            '<a href="/uploads/ruling.pdf">PDF</a></div>'
        )
        request = next(iter(spider.parse_detail(response)))
        self.assertIsInstance(request, Request)
        self.assertEqual(request.url, "https://tbappeal.court.ge/uploads/ruling.pdf")
        self.assertIn("Article summary", request.meta["fields"]["article_summary"])
        self.assertNotIn("body_markdown", request.meta["fields"])

    def test_full_pdf_becomes_body(self):
        spider = self._spider()
        request = Request(
            "https://tbappeal.court.ge/uploads/ruling.pdf",
            meta={
                "fields": {
                    "source_url": "u",
                    "slug": "s",
                    "title": "t",
                    "date": "05-02-2018",
                    "pdf_url": "p",
                    "source_binary_url": "p",
                    "article_summary": "summary",
                }
            },
        )
        response = TextResponse(url=request.url, body=b"%PDF-x", request=request)
        extracted = ExtractionResult(
            text="complete ruling text",
            status=ExtractionStatus.FULL_TEXT,
            content_complete=True,
            page_boundaries=(
                PageBoundary(page_number=1, char_start=0, char_end=20),
            ),
            page_coordinate_reason=PAGE_COORDINATE_REASON_EXACT_PDF_TEXT,
        )
        with patch(
            "legal_scrapers.spiders.tbappeal_spider.pdf_to_markdown",
            return_value=extracted,
        ):
            item = next(iter(spider.parse_pdf(response)))
        self.assertEqual(item["body_markdown"], "complete ruling text")
        self.assertEqual(item["article_summary"], "summary")
        self.assertEqual(item["content_kind"], "ruling_full_text")
        self.assertTrue(item["content_complete"])
        self.assertEqual(
            item["page_boundaries"],
            [{"page_number": 1, "char_start": 0, "char_end": 20}],
        )
        self.assertEqual(item["page_coordinate_reason"], "exact_pdf_text")

    def test_missing_or_bad_pdf_is_labeled_incomplete_and_retryable(self):
        spider = self._spider()
        item = next(iter(spider.parse_detail(_html('<div class="blog-details">summary</div>'))))
        self.assertEqual(item["content_kind"], "non_authoritative_summary")
        self.assertFalse(item["admissible"])
        self.assertEqual(item["source_authority"], "non_authoritative_summary")
        self.assertFalse(item["content_complete"])
        self.assertEqual(item["extraction_status"], "malformed")
        self.assertEqual(item["quarantine_reason"], "missing_ruling_pdf")
        spider.record_quality_failure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
