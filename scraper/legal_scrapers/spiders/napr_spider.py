"""napr.gov.ge — National Agency of Public Registry, legal-practice decisions.

The list is a JSON POST API (``/legal_search``); the decision body itself is only
available as a PDF. So we read the metadata from JSON and extract the body text from
the linked PDF, giving every item a ``body_markdown`` like the matsne items.

Quirks handled here:
- ``/legal_search`` returns a *double-encoded* JSON string (see ``loads_maybe_double``).
- Date filtering uses ``fdate``/``tdate`` in ``dd/mm/yyyy``; pagination is offset-based
  via ``from_n``/``ret_count`` with a ``total`` count.
- The API does not return the "დავის ტიპი/კატეგორია" label in list records, so we
  first crawl each category via ``ptag`` and attach that category to those items; a
  deferred unfiltered pass then catches records not found through category filters.
- In theory the same first-filtered-then-catch-all pattern should capture "დავის
  საგანი" and "გადაწყვეტილების დასახელება/ტიპი" too. The live form is buggy as of
  2026-07-02: all three selects are collapsed into the same ``ptag`` parameter, and
  combined selections are concatenated without a separator, so those two fields are
  not reliable enough to scrape as distinct item fields.
- ``SENDER`` already arrives masked by the source (personal IDs redacted).

napr's robots.txt is an invalid IIS error page, which Scrapy treats as allow-all, so
no per-spider override is needed.
"""

from scrapy import signals
from scrapy.exceptions import DontCloseSpider
from scrapy.loader import ItemLoader

from ..items import NaprItem
from ..utils.dates import date_part, iso_to_slashed
from ..utils.documents import pdf_to_markdown
from ..utils.json_api import form_post, loads_maybe_double
from .base import BaseLegalSpider

BASE = "https://www.napr.gov.ge"
SEARCH_URL = f"{BASE}/legal_search"

DISPUTE_CATEGORIES = [
    "უძრავ ნივთებზე უფლებათა რეესტრი",
    "სამეწარმეო რეესტრი",
    "საჯარო–სამართლებრივი შეზღუდვისა და საგადასახადო გირავნობა/იპოთეკის რეესტრი",
    "სამისამართო რეესტრი",
    "მოძრავ ნივთებსა და არამატერიალურ ქონებრივ სიკეთეზე უფლებათა რეესტრი - გირავნობა/ლიზინგი",
    "პოლიტიკური პარტიების რეესტრი",
    "ეკონომიკურ საქმიანობათა რეესტრი",
    "ტექბიურო/არქივი/საინფორმაციო",
    "მოთხოვნა – სარეგისტრაციო წარმოების მიმდინარეობის/შეჩერების/შეწყვეტის/უარის შესახებ გადაწყვეტილების გაუქმება",
    "მოთხოვნა – რეგისტრაციის შესახებ გადაწყვეტილების გაუქმება/კანონიერების შესწავლა",
    "მოთხოვნა – რეგისტრაციის დავალება",
    "მოთხოვნა – ქმედების (მოქმედების/უმოქმედობის) კანონიერების შესწავლა",
    "მოთხოვნა – ქმედების (მოქმედების/უმოქმედობის) დავალება",
    "მოთხოვნა – ქმედების განხორციელება",
    "მოთხოვნა – რეგისტრაციის გაუქმების შესახებ გადაწყვეტილების კანონიერების შესწავლა",
    "სახელმწიფო პროექტი – სპორადული",
    "სახელმწიფო პროექტი – სისტემური (პილოტი)",
    "მიმდინარე რეგისტრაცია",
    "საინფორმაციო/რეალაქტი",
    "სისტემური რეგისტრაცია (ირიგაციის არეალი)",
]


class NaprSpider(BaseLegalSpider):
    name = "napr"
    DEDUP_KEY = ("document_id",)
    PAGE_SIZE = 50  # ``ret_count``

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_idle, signal=signals.spider_idle)
        return spider

    async def start(self):
        self.catch_all_started = False
        for dispute_category in DISPUTE_CATEGORIES:
            yield self.request_page(from_n=0, dispute_category=dispute_category)

    def spider_idle(self):
        """Start the unfiltered pass after category-filtered searches drain."""
        if self.catch_all_started:
            return
        self.catch_all_started = True
        self.crawler.engine.crawl(self.request_page(from_n=0))
        raise DontCloseSpider

    def request_page(self, from_n, dispute_category=None):
        payload = {
            "from_n": from_n,
            "ret_count": self.PAGE_SIZE,
            "fdate": iso_to_slashed(self.scraping_start_date),
            "tdate": iso_to_slashed(self.scraping_end_date),
            "psearch": "",
            "prandomid": "",
            "ptag": dispute_category or "",
        }
        return form_post(
            SEARCH_URL,
            payload,
            callback=self.parse_list,
            errback=self.request_failed,
            meta={"from_n": from_n, "dispute_category": dispute_category},
        )

    def parse_list(self, response):
        from_n = response.meta["from_n"]
        dispute_category = response.meta.get("dispute_category")
        data = loads_maybe_double(response.text)
        records = data.get("data") or []
        total = int(data.get("total") or 0)

        for record in records:
            fields = {
                "source_url": f"{BASE}/ka/legal-practice",
                "document_id": record.get("LETTERS_ID"),
                "app_no": record.get("RANDOMID"),
                "date": date_part(record.get("REGISTRATIONDATE")),
                "sender": record.get("SENDER"),
                "title": record.get("ABOUT"),
                "decision_date": date_part(record.get("KANC_DATE")),
                "decision_no": record.get("KANC_NO"),
            }
            if dispute_category:
                fields["dispute_category"] = dispute_category
            if self.is_seen(fields):
                self.crawler.stats.inc_value("dedup/skipped")
                continue
            pdf_path = record.get("PDF")
            if pdf_path:
                fields["pdf_url"] = response.urljoin(pdf_path)
                yield response.follow(
                    fields["pdf_url"],
                    callback=self.parse_pdf,
                    errback=self.request_failed,
                    meta={"fields": fields},
                )
            else:
                yield self.load_item(fields)

        next_from = from_n + self.PAGE_SIZE
        if next_from < total:
            yield self.request_page(next_from, dispute_category=dispute_category)

    def parse_pdf(self, response):
        fields = response.meta["fields"]
        try:
            fields["body_markdown"] = pdf_to_markdown(response.body)
        except Exception as exc:
            self.logger.warning("PDF parse failed for %s: %s", response.url, exc)
            fields["body_markdown"] = ""
        yield self.load_item(fields)

    @staticmethod
    def load_item(fields):
        loader = ItemLoader(item=NaprItem())
        for key, value in fields.items():
            loader.add_value(key, value)
        return loader.load_item()
