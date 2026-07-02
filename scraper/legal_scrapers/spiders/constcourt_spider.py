"""constcourt.ge — Constitutional Court of Georgia judicial acts.

Server-rendered HTML. The list at ``/ka/judicial-acts`` supports server-side date
filtering (``dateFrom``/``dateTo`` in ``DD-MM-YYYY``) and ``page``/``quantity``
pagination; each item links to a detail page (``?legal=<ID>``) carrying a metadata
table and the act body as HTML, plus a downloadable DOCX.

Body handling: judgments/rulings carry their full text in the HTML body. Constitutional
*claims* show only a teaser there ("download the attached document for the full
version") — for those we fetch the DOCX and use its text as the body, so every item
gets a complete ``body_markdown`` like the matsne items.

Note: the visible dates are Georgian month names (e.g. "24 ივნისი 2026"); they are
stored verbatim since the range filtering happens server-side.
"""

from urllib.parse import parse_qs, urlencode, urlparse

from scrapy import Request
from scrapy.loader import ItemLoader

from ..items import ConstcourtItem
from ..utils.dates import iso_to_dotted
from ..utils.documents import docx_to_markdown
from ..utils.markdown import safe_html_to_markdown
from .base import BaseLegalSpider

BASE = "https://constcourt.ge"
LIST_URL = f"{BASE}/ka/judicial-acts"
MAX_PAGES = 5000  # safety cap against a server that never returns an empty page

# Phrase the site shows in place of a full body when only a DOCX has the complete text.
TEASER_MARKER = "სრული ვერსიის სანახავად"

# Georgian metadata-table labels -> ConstcourtItem field names.
METADATA_LABELS = {
    "დოკუმენტის ტიპი": "doc_type",
    "ნომერი": "number",
    "თარიღი": "date",
    "გამოქვეყნების თარიღი": "publication_date",
    "კოლეგია/პლენუმი": "college",
    "ავტორ(ებ)ი": "authors",
    "ავტორები": "authors",
    "ავტორი": "authors",
}


class ConstcourtSpider(BaseLegalSpider):
    name = "constcourt"
    DEDUP_KEY = ("legal_id",)
    PAGE_SIZE = 50  # ``quantity`` query parameter

    async def start(self):
        yield self.request_page(1)

    def request_page(self, page):
        query = {
            "quantity": self.PAGE_SIZE,
            "page": page,
            "dateFrom": iso_to_dotted(self.scraping_start_date),
            "dateTo": iso_to_dotted(self.scraping_end_date),
        }
        return Request(
            f"{LIST_URL}?{urlencode(query)}",
            callback=self.parse_list,
            errback=self.request_failed,
            meta={"page": page},
        )

    def parse_list(self, response):
        page = response.meta["page"]
        if response.status != 200:
            self.logger.warning("constcourt: page %s returned HTTP %s; stopping", page, response.status)
            return
        items = response.css("div.legal-act-info")

        for block in items:
            href = block.css("h5.legal-act-title a::attr(href)").get()
            if not href:
                continue
            title = block.css("h5.legal-act-title a::text").get(default="").strip()
            legal_id = parse_qs(urlparse(href).query).get("legal", [None])[0]
            if self.is_seen({"legal_id": legal_id}):
                self.crawler.stats.inc_value("dedup/skipped")
                continue
            yield response.follow(
                href,
                callback=self.parse_detail,
                errback=self.request_failed,
                meta={"legal_id": legal_id, "title": title, "source_url": response.urljoin(href)},
            )

        # Paginate until a page comes back empty (the page after the last full one
        # returns no acts), with a hard cap as a runaway guard.
        if items and page < MAX_PAGES:
            yield self.request_page(page + 1)
        elif items:
            self.logger.warning("constcourt: hit MAX_PAGES=%s cap; stopping pagination", MAX_PAGES)

    def parse_detail(self, response):
        data = {
            "source_url": response.meta["source_url"],
            "legal_id": response.meta["legal_id"],
            "title": response.meta["title"],
        }

        for cell in response.css("td.first-table-cell"):
            label = cell.xpath("normalize-space(.)").get()
            field = METADATA_LABELS.get(label)
            if field and field not in data:
                value = cell.xpath("normalize-space(following-sibling::td[1])").get()
                if value:
                    data[field] = value

        docx_href = response.css("a[href*='/uploads/documents/']::attr(href)").get()
        if docx_href:
            data["docx_url"] = response.urljoin(docx_href)

        body_html = response.css("span.legalactshowparagraph").get()
        data["body_markdown"] = safe_html_to_markdown(body_html, base_url=BASE, source_url=response.url)

        if data.get("docx_url") and TEASER_MARKER in data["body_markdown"]:
            yield response.follow(
                data["docx_url"],
                callback=self.parse_docx_body,
                errback=self.request_failed,
                meta={"data": data},
            )
        else:
            yield self.load_item(data)

    def parse_docx_body(self, response):
        data = response.meta["data"]
        try:
            data["body_markdown"] = docx_to_markdown(response.body) or data["body_markdown"]
        except Exception as exc:  # keep the HTML teaser body if DOCX parsing fails
            self.logger.warning("DOCX parse failed for %s: %s", response.url, exc)
        yield self.load_item(data)

    @staticmethod
    def load_item(data):
        loader = ItemLoader(item=ConstcourtItem())
        for key, value in data.items():
            loader.add_value(key, value)
        return loader.load_item()
