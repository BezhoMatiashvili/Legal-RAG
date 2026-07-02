import json
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs
from unittest.mock import MagicMock

from scrapy import Request
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse, TextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.spiders.constcourt_spider import ConstcourtSpider, TEASER_MARKER  # noqa: E402
from legal_scrapers.spiders.matsne_spider import MatsneSpider  # noqa: E402
from legal_scrapers.spiders.napr_spider import (  # noqa: E402
    DISPUTE_CATEGORIES,
    NaprSpider,
    decision_type_from_title,
)
from legal_scrapers.spiders.supremecourt_spider import SupremecourtSpider  # noqa: E402
from legal_scrapers.spiders.tas_spider import TasSpider  # noqa: E402


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
        self.assertEqual(fields["chamber"], "სამოქალაქო საქმეთა პალატა")
        self.assertEqual(fields["case_number"], "ას-1")
        self.assertEqual(fields["date"], "2024-12-26")
        self.assertEqual(fields["appeal_type"], "საკასაციო")

    def test_parse_detail_keeps_numeric_palata_for_download_url(self):
        spider = SupremecourtSpider(start_date="2024-01-01", end_date="2024-12-31")
        fields = {
            "case_id": "73901",
            "chamber": "სამოქალაქო საქმეთა პალატა",
            "case_number": "ას-1",
        }
        resp = _html(
            "https://www.supremecourt.ge/ka/fullcase/73901/1",
            '<div class="case-single" id="modalBody">სრული ტექსტი</div>',
            {"fields": fields, "palata": "1"},
        )

        item = next(iter(spider.parse_detail(resp)))

        self.assertEqual(item["chamber"], "სამოქალაქო საქმეთა პალატა")
        self.assertEqual(item["docx_url"], "https://www.supremecourt.ge/ka/download/73901/1")


class NaprParseTests(unittest.TestCase):
    def test_decision_type_from_title_extracts_parenthesized_detail_prefix(self):
        title = (
            "ა/ს დაკმაყოფილებაზე უარის თქმის შესახებ გადაწყვეტილება "
            "(რეგისტრაციაზე უარის თქმის შესახებ №1 გადაწყვეტილება)."
        )
        self.assertEqual(
            decision_type_from_title(title),
            "ა/ს დაკმაყოფილებაზე უარის თქმის შესახებ გადაწყვეტილება",
        )

    def test_decision_type_from_title_extracts_dash_detail_prefix(self):
        title = (
            "დაკმაყოფილებაზე უარის თქმის შესახებ გადაწყვეტილება - "
            "რეგისტრაციაზე უარის თქმის შესახებ №1 გადაწყვეტილება."
        )
        self.assertEqual(
            decision_type_from_title(title),
            "დაკმაყოფილებაზე უარის თქმის შესახებ გადაწყვეტილება",
        )

    def test_decision_type_from_title_collapses_whitespace(self):
        title = " ა/ს დაკმაყოფილების შესახებ  გადაწყვეტილება   (x)"
        self.assertEqual(
            decision_type_from_title(title),
            "ა/ს დაკმაყოფილების შესახებ გადაწყვეტილება",
        )

    def test_decision_type_from_title_preserves_administrative_complaint_variant(self):
        title = (
            "ადმინისტრაციული საჩივრის დაკმაყოფილებაზე უარის თქმის შესახებ "
            "გადაწყვეტილება (x)"
        )
        self.assertEqual(
            decision_type_from_title(title),
            "ადმინისტრაციული საჩივრის დაკმაყოფილებაზე უარის თქმის შესახებ გადაწყვეტილება",
        )

    def test_decision_type_from_title_returns_none_without_decision_word(self):
        self.assertIsNone(decision_type_from_title("სსიპ საჯარო რეესტრის წერილი"))
        self.assertIsNone(decision_type_from_title(None))

    def test_category_request_posts_ptag(self):
        spider = NaprSpider(start_date="2024-01-01", end_date="2024-12-31")
        category = DISPUTE_CATEGORIES[0]
        req = spider.request_page(from_n=0, dispute_category=category)
        payload = parse_qs(req.body.decode("utf-8"), keep_blank_values=True)
        self.assertEqual(payload["ptag"], [category])
        self.assertEqual(req.meta["dispute_category"], category)

    def test_idle_schedules_unfiltered_pass_once(self):
        spider = NaprSpider(start_date="2024-01-01", end_date="2024-12-31")
        spider.catch_all_started = False
        spider.crawler = MagicMock()
        with self.assertRaises(DontCloseSpider):
            spider.spider_idle()
        req = spider.crawler.engine.crawl.call_args.args[0]
        payload = parse_qs(req.body.decode("utf-8"), keep_blank_values=True)
        self.assertEqual(payload["ptag"], [""])
        self.assertIsNone(req.meta["dispute_category"])

        spider.crawler.engine.crawl.reset_mock()
        self.assertIsNone(spider.spider_idle())
        spider.crawler.engine.crawl.assert_not_called()

    def test_double_encoded_json_yields_pdf_follow(self):
        spider = NaprSpider(start_date="2024-01-01", end_date="2024-12-31")
        title = "ა/ს დაკმაყოფილების შესახებ გადაწყვეტილება (x)"
        inner = {
            "data": [{
                "LETTERS_ID": "77", "RANDOMID": "r", "REGISTRATIONDATE": "2024-12-30 00:00:00",
                "SENDER": "s", "ABOUT": title, "KANC_DATE": "2024-12-30 00:00:00", "KANC_NO": "9",
                "PDF": "/uploads/administrativeComplaints/file77.pdf",
            }],
            "total": "1",
        }
        body = json.dumps(json.dumps(inner))  # double-encoded, like the live endpoint
        req = Request(url="https://www.napr.gov.ge/legal_search", method="POST", meta={"from_n": 0})
        resp = TextResponse(url=req.url, body=body.encode("utf-8"), encoding="utf-8", request=req)
        reqs = [o for o in spider.parse_list(resp) if isinstance(o, Request)]
        self.assertTrue(any("file77.pdf" in r.url for r in reqs))
        self.assertEqual(
            reqs[0].meta["fields"]["decision_type_name"],
            "ა/ს დაკმაყოფილების შესახებ გადაწყვეტილება",
        )

    def test_category_phase_attaches_dispute_category(self):
        spider = NaprSpider(start_date="2024-01-01", end_date="2024-12-31")
        category = DISPUTE_CATEGORIES[0]
        inner = {
            "data": [{
                "LETTERS_ID": "77", "RANDOMID": "r", "REGISTRATIONDATE": "2024-12-30 00:00:00",
                "SENDER": "s", "ABOUT": "a", "KANC_DATE": "2024-12-30 00:00:00", "KANC_NO": "9",
                "PDF": "/uploads/administrativeComplaints/file77.pdf",
            }],
            "total": "1",
        }
        body = json.dumps(json.dumps(inner))
        req = Request(
            url="https://www.napr.gov.ge/legal_search",
            method="POST",
            meta={"from_n": 0, "dispute_category": category},
        )
        resp = TextResponse(url=req.url, body=body.encode("utf-8"), encoding="utf-8", request=req)
        pdf_req = next(o for o in spider.parse_list(resp) if isinstance(o, Request))
        self.assertEqual(pdf_req.meta["fields"]["dispute_category"], category)


def _tas_list_record():
    return {
        "documentId": 25755,
        "documentNo": "AR125755",
        "address": "; ქალაქი თბილისი , გლდანი , მიკრო/რაიონი I , კორპუსი 16 ",
        "registrationDate": "2012-02-24 12:21:34",
        "createDateStr": "24/02/2012",
        "cachedInfo": (
            "<documentCachedInfo><documentStatusName>-</documentStatusName>"
            "<categoryName>ანტრესოლის, კიბის, ვიტრინის</categoryName>"
            "<actionName>რეკონსტრუქცია</actionName>"
            "<caseId>4</caseId></documentCachedInfo>"
        ),
    }


def _tas_detail():
    return {
        "ok": True,
        "canSeeFinalResult": True,
        "openDate": "2012-03-05T20:00:00.000Z",
        "document": {
            "documentNo": "AR125755",
            "createDateStr": "24/02/2012",
            "deadLineDate": "2012-03-01T20:00:00.000Z",
            "documentStatusId": 1,
            "documentTypeId": 73058,
            "amountToPay": 0,
            "address": "; ქალაქი თბილისი , გლდანი , მიკრო/რაიონი I , კორპუსი 16 ",
            "responseText": (
                "<pre>ფასადზე I კლასის ფანჯრების შეცვლის თაობაზე</pre>\n"
                "<pre>ქ. თბილისის მერიის სსიპ თბილისის არქიტექტურის სამსახური "
                "ადასტურებს ფანჯრების შეცვლის შესაძლებლობას.</pre>"
            ),
        },
        "docAuthor": {
            "firstName": "თამარ",
            "lastName": "მგელაშვილი",
            "personalNo": "16001020967",
            "birhtDate": "1970-07-12T20:00:00.000Z",
            "address": "დუშეთი ს. მიგრიაულთა ",
            "email": "tamar@mail.ru",
            "phoneNumber": "555383887",
            "passSerialNumber": "ბ0752055",
            "personId": 171136,
        },
        "executorEmployee": {
            "firstName": "ეკა",
            "lastName": "კვირკველია",
            "personalNo": "01024022244",
            "email": "eka@gmail.com",
            "phoneNumber": "568321515",
            "employeeId": 525,
        },
        "mapInfos": [
            {
                "naprCadCode": "01.11.12.007.010.01.178",
                "naprAddress": "ქალაქი თბილისი , გლდანი , მიკრო/რაიონი I , კორპუსი 16 ",
                "naprArea": 17,
                "naprPurpose": "არასასოფლო სამეურნეო",
                "naprOwner": "თამარ   მგელაშვილი (P/N: 16001020967)",
                "naprCoowner": None,
                "naprRegNo": None,
            }
        ],
        "docValues": [
            {
                "documentFieldId": 2997,
                "clobValue": "გთხოვთ მომცეთ უფლება ჩემს კუთვნილ ბინაში ფანჯრების შეცვლის.",
                "stringValue": None,
                "dateValueStr": "",
            },
            {
                "documentFieldId": 2998,
                "clobValue": None,
                "stringValue": "C:\\fakepath\\foto.pdf",
                "dateValueStr": "",
            },
        ],
        "fieldsetPojos": [
            {"fields": [{"documentFieldId": 2997, "fieldLabel": "..მოთხოვნის ტექსტი"}]},
            {"fields": [{"documentFieldId": 2998, "fieldLabel": "..ფოტოსურათების pdf ფაილი "}]},
        ],
        "oldResponseMotions": [
            {
                "previousMotionId": 65192,
                "whenUserOpenedTheDoc": "2012-03-05T20:00:00.000Z",
                "motionDate": "2012-03-01T05:36:30.000Z",
                "documentStatusId": 3,
            }
        ],
        "attachedFiles": [{"fileName": "foto.pdf"}, {"fileName": "montaJi.pdf"}],
        "nomenklaturMarkup": (
            "ნომენკლატურა : <UL><LI><strong>I კლასი</strong></LI>"
            "<LI><strong>ანტრესოლის, კიბის</strong></LI></UL>"
        ),
    }


class TasDetailTests(unittest.TestCase):
    def test_enriched_item_maps_all_detail_fields(self):
        spider = TasSpider()
        item = spider.build_item(_tas_list_record(), _tas_detail())

        # Title-info block.
        self.assertEqual(item["document_no"], "AR125755")
        self.assertEqual(item["decision_status_id"], 1)
        self.assertEqual(item["decision"], "თანხმობა")
        self.assertEqual(item["decision_no"], 65192)
        self.assertEqual(item["applicant_name"], "თამარ მგელაშვილი")
        self.assertEqual(item["applicant_personal_no"], "16001020967")
        self.assertEqual(item["document_type_id"], 73058)
        self.assertTrue(item["can_see_final_result"])

        # Dates are converted to the Tbilisi (UTC+4) calendar day the site shows.
        self.assertEqual(item["deadline_date"], "2012-03-02")
        self.assertEqual(item["acquaint_date"], "2012-03-06")

        # Executor + cadastral.
        self.assertEqual(item["executor_name"], "ეკა კვირკველია")
        self.assertEqual(item["cad_code"], "01.11.12.007.010.01.178")
        self.assertEqual(item["land_area"], 17)
        self.assertEqual(item["land_purpose"], "არასასოფლო სამეურნეო")

        # Request text + structured lists.
        self.assertTrue(item["request_text"].startswith("გთხოვთ მომცეთ უფლება"))
        self.assertIn("foto.pdf", item["attachments"])
        self.assertEqual(len(item["parcels"]), 1)
        self.assertEqual(item["responses"][0]["decision_no"], 65192)
        labels = {field["label"] for field in item["form_fields"]}
        self.assertIn("..მოთხოვნის ტექსტი", labels)

        # Decision text: converted, no code fences from the <pre> wrappers.
        self.assertIn("ადასტურებს", item["response_markdown"])
        self.assertNotIn("```", item["response_markdown"])

        # Body weaves the title info + decision together.
        self.assertIn("**გადაწყვეტილება:** თანხმობა", item["body_markdown"])
        self.assertIn("AR125755", item["body_markdown"])
        self.assertIn("ადასტურებს", item["body_markdown"])

    def test_missing_detail_falls_back_to_list_only_item(self):
        spider = TasSpider()
        item = spider.build_item(_tas_list_record(), None)

        self.assertEqual(item["document_no"], "AR125755")
        self.assertIn("ანტრესოლის", item["nomenclature"])
        self.assertIn("**ნომენკლატურა:**", item["body_markdown"])
        # No detail-only fields when enrichment is skipped.
        self.assertNotIn("decision", item)
        self.assertNotIn("applicant_name", item)
        self.assertNotIn("parcels", item)

    def test_unsubmitted_draft_is_not_enriched(self):
        spider = TasSpider()
        detail = _tas_detail()
        detail["document"]["documentStatusId"] = 8  # draft, never submitted
        item = spider.build_item(_tas_list_record(), detail)

        self.assertEqual(item["document_no"], "AR125755")
        self.assertNotIn("decision", item)
        self.assertNotIn("applicant_name", item)


if __name__ == "__main__":
    unittest.main()
