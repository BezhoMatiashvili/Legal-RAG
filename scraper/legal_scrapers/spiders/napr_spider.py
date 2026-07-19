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
  not reliable enough to scrape as distinct item fields. ``decision_type_name`` is
  instead derived from the leading title phrase when the title contains
  "გადაწყვეტილება".
- ``SENDER`` already arrives masked by the source (personal IDs redacted).

napr's robots.txt is an invalid IIS error page, which Scrapy treats as allow-all, so
no per-spider override is needed.
"""

from scrapy import signals
from scrapy.exceptions import DontCloseSpider
from scrapy.loader import ItemLoader

from ..items import NaprItem
from ..utils.dates import date_part, iso_to_slashed
from ..utils.documents import ExtractionStatus, pdf_to_markdown
from ..utils.json_api import form_post, loads_maybe_double
from ..utils.pagination import (
    advertised_page_count,
    finalize_pagination_scope,
    get_pagination_reconciler,
    handle_pagination_request_failure,
    parse_advertised_count,
)
from .base import BaseLegalSpider

BASE = "https://www.napr.gov.ge"
SEARCH_URL = f"{BASE}/legal_search"
DECISION_WORD = "გადაწყვეტილება"

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


def decision_type_from_title(title):
    if not isinstance(title, str):
        return None
    normalized = " ".join(title.split())
    decision_word_end = normalized.find(DECISION_WORD)
    if decision_word_end == -1:
        return None
    return normalized[: decision_word_end + len(DECISION_WORD)]


class NaprSpider(BaseLegalSpider):
    name = "napr"
    DEDUP_KEY = ("document_id",)
    PAGE_SIZE = 50  # ``ret_count``
    MAX_PAGES = 20_000

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

    @staticmethod
    def pagination_scope(dispute_category):
        return (
            f"category:{dispute_category}"
            if dispute_category
            else "catch_all"
        )

    def pagination_request_failed(self, failure):
        return handle_pagination_request_failure(self, failure)

    def request_page(self, from_n, dispute_category=None):
        scope = self.pagination_scope(dispute_category)
        get_pagination_reconciler(self, scope, max_pages=self.MAX_PAGES)
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
            errback=self.pagination_request_failed,
            meta={
                "from_n": from_n,
                "dispute_category": dispute_category,
                "pagination_scope": scope,
                "pagination_cursor": from_n,
                "dont_cache": True,
            },
        )

    def parse_list(self, response):
        from_n = response.meta["from_n"]
        dispute_category = response.meta.get("dispute_category")
        scope = response.meta.get("pagination_scope") or self.pagination_scope(
            dispute_category
        )
        tracker = get_pagination_reconciler(
            self,
            scope,
            max_pages=self.MAX_PAGES,
        )
        try:
            data = loads_maybe_double(response.text)
        except ValueError:  # includes json.JSONDecodeError
            # napr's WAF answers some queries (observed 2026-07-10: the category
            # "სისტემური რეგისტრაცია (ირიგაციის არეალი)") with HTTP 200 + an HTML
            # "Access Denied" page, which must not kill the crawl. Skipping is safe
            # for coverage: the deferred unfiltered pass (spider_idle catch-all)
            # still captures those records — only this category's label is lost.
            self.logger.warning(
                "non-JSON legal_search response (WAF block?) for category=%r "
                "from_n=%s — skipping this listing page; body starts: %.80r",
                dispute_category,
                from_n,
                response.text,
            )
            self.record_quality_failure(
                "non_json_response",
                response.url,
                detail=f"HTTP {response.status}; category={dispute_category!r}; from_n={from_n}",
            )
            tracker.mark_failure(
                "waf_or_non_json",
                cursor=from_n,
                detail=f"HTTP {response.status}",
            )
            finalize_pagination_scope(
                self,
                tracker,
                url=response.url,
                quality_failure_recorded=True,
            )
            return
        if not isinstance(data, dict):
            tracker.mark_failure(
                "callback_failure",
                cursor=from_n,
                detail="legal_search payload is not an object",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        records = data.get("data", [])
        if records is None:
            records = []
        if not isinstance(records, list):
            tracker.mark_failure(
                "callback_failure",
                cursor=from_n,
                detail="legal_search data is not a list",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        raw_total = data.get("total", 0)
        try:
            total = parse_advertised_count(raw_total)
        except ValueError:
            tracker.mark_failure(
                "callback_failure",
                cursor=from_n,
                detail=f"invalid advertised total: {raw_total!r}",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return

        next_from = from_n + self.PAGE_SIZE
        terminal = next_from >= total
        page_number = (from_n // self.PAGE_SIZE) + 1
        tracker.observe_page(
            from_n,
            [record.get("LETTERS_ID") if isinstance(record, dict) else None for record in records],
            advertised_total=total,
            advertised_pages=advertised_page_count(total, self.PAGE_SIZE),
            page_number=page_number,
            terminal=terminal,
        )
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                output = self._parse_listing_record(
                    response,
                    record,
                    dispute_category,
                )
            except Exception as exc:  # noqa: BLE001 - retain remaining listing rows
                tracker.mark_failure(
                    "callback_failure",
                    cursor=from_n,
                    detail=f"record processing failed: {exc}",
                )
                self.logger.warning("napr: record processing failed: %s", exc)
                continue
            if output is not None:
                yield output

        cap_reached = not terminal and page_number >= self.MAX_PAGES
        if cap_reached:
            tracker.mark_cap(cursor=from_n, configured_cap=self.MAX_PAGES)
            finalize_pagination_scope(self, tracker, url=response.url)
        elif terminal:
            finalize_pagination_scope(self, tracker, url=response.url)
        else:
            yield self.request_page(next_from, dispute_category=dispute_category)

    def _parse_listing_record(self, response, record, dispute_category):
        title = record.get("ABOUT")
        fields = {
            "source_url": f"{BASE}/ka/legal-practice",
            "document_id": record.get("LETTERS_ID"),
            "app_no": record.get("RANDOMID"),
            "date": date_part(record.get("REGISTRATIONDATE")),
            "sender": record.get("SENDER"),
            "title": title,
            "decision_type_name": decision_type_from_title(title),
            "decision_date": date_part(record.get("KANC_DATE")),
            "decision_no": record.get("KANC_NO"),
        }
        if dispute_category:
            fields["dispute_category"] = dispute_category
        if self.is_seen(fields):
            self.crawler.stats.inc_value("dedup/skipped")
            return None
        pdf_path = record.get("PDF")
        if pdf_path:
            fields["pdf_url"] = response.urljoin(pdf_path)
            return response.follow(
                fields["pdf_url"],
                callback=self.parse_pdf,
                errback=self.request_failed,
                meta={"fields": fields},
            )
        fields.update(
            {
                "body_markdown": "",
                "content_kind": "metadata_only",
                "content_complete": False,
                "extraction_status": ExtractionStatus.MALFORMED.value,
            }
        )
        self.record_quality_failure(
            "missing_document_pdf",
            response.url,
            detail="listing record has no PDF path",
            context={"document_id": fields.get("document_id")},
        )
        return self.load_item(fields)

    def parse_pdf(self, response):
        fields = response.meta["fields"]
        try:
            result = pdf_to_markdown(
                response.body,
                declared_mime=response.headers.get(b"Content-Type"),
            )
        except Exception as exc:
            self.logger.warning("PDF parse failed for %s: %s", response.url, exc)
            self.record_quality_failure(
                "document_parse_failed",
                response.url,
                detail=str(exc),
                context={"document_id": fields.get("document_id")},
            )
            return
        if result.status is not ExtractionStatus.FULL_TEXT:
            self.record_quality_failure(
                f"document_{result.status.value}",
                response.url,
                detail=result.detail,
                context={"document_id": fields.get("document_id")},
            )
            return
        fields["body_markdown"] = result.text
        fields["content_kind"] = result.content_kind
        fields["content_complete"] = result.content_complete
        fields["extraction_status"] = result.status.value
        fields["source_binary_url"] = response.url
        fields["page_boundaries"] = [
            boundary.as_dict() for boundary in result.page_boundaries
        ]
        fields["page_coordinate_reason"] = result.page_coordinate_reason
        yield self.load_item(fields)

    @staticmethod
    def load_item(fields):
        loader = ItemLoader(item=NaprItem())
        for key, value in fields.items():
            loader.add_value(key, value)
        return loader.load_item()
