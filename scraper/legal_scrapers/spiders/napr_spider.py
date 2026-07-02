"""napr.gov.ge — National Agency of Public Registry, legal-practice decisions.

The list is a JSON POST API (``/legal_search``); the decision body itself is only
available as a PDF. So we read the metadata from JSON and extract the body text from
the linked PDF, giving every item a ``body_markdown`` like the matsne items.

Quirks handled here:
- ``/legal_search`` returns a *double-encoded* JSON string (see ``loads_maybe_double``).
- Date filtering uses ``fdate``/``tdate`` in ``dd/mm/yyyy``; pagination is offset-based
  via ``from_n``/``ret_count`` with a ``total`` count.
- ``SENDER`` already arrives masked by the source (personal IDs redacted).

napr's robots.txt is an invalid IIS error page, which Scrapy treats as allow-all, so
no per-spider override is needed.
"""

from scrapy.loader import ItemLoader

from ..items import NaprItem
from ..utils.dates import date_part, iso_to_slashed
from ..utils.documents import pdf_to_markdown
from ..utils.json_api import form_post, loads_maybe_double
from .base import BaseLegalSpider

BASE = "https://www.napr.gov.ge"
SEARCH_URL = f"{BASE}/legal_search"


class NaprSpider(BaseLegalSpider):
    name = "napr"
    DEDUP_KEY = ("document_id",)
    PAGE_SIZE = 50  # ``ret_count``

    async def start(self):
        yield self.request_page(from_n=0)

    def request_page(self, from_n):
        payload = {
            "from_n": from_n,
            "ret_count": self.PAGE_SIZE,
            "fdate": iso_to_slashed(self.scraping_start_date),
            "tdate": iso_to_slashed(self.scraping_end_date),
            "psearch": "",
            "prandomid": "",
            "ptag": "",
        }
        return form_post(
            SEARCH_URL,
            payload,
            callback=self.parse_list,
            errback=self.request_failed,
            meta={"from_n": from_n},
        )

    def parse_list(self, response):
        from_n = response.meta["from_n"]
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
            yield self.request_page(next_from)

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
