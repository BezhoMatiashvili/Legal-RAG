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
from ..utils.documents import ExtractionStatus, docx_to_markdown
from ..utils.markdown import safe_html_to_markdown
from ..utils.pagination import (
    finalize_pagination_scope,
    get_pagination_reconciler,
    handle_pagination_request_failure,
)
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

    @staticmethod
    def pagination_scope():
        return "judicial-acts"

    def pagination_request_failed(self, failure):
        return handle_pagination_request_failure(self, failure)

    def request_page(self, page, empty_retry=False):
        scope = self.pagination_scope()
        get_pagination_reconciler(self, scope, max_pages=MAX_PAGES)
        query = {
            "quantity": self.PAGE_SIZE,
            "page": page,
            "dateFrom": iso_to_dotted(self.scraping_start_date),
            "dateTo": iso_to_dotted(self.scraping_end_date),
        }
        return Request(
            f"{LIST_URL}?{urlencode(query)}",
            callback=self.parse_list,
            errback=self.pagination_request_failed,
            # The one-shot empty-page recheck must reach the network: otherwise Scrapy's
            # duplicate filter drops the identical URL and HTTP cache may replay the same
            # cached 200 WAF/empty response.
            dont_filter=empty_retry,
            meta={
                "page": page,
                "empty_retry": empty_retry,
                "dont_cache": empty_retry,
                "pagination_scope": scope,
                "pagination_cursor": page,
            },
        )

    @staticmethod
    def _looks_waf_blocked(response):
        sample = " ".join(response.xpath("//title//text() | //body//text()").getall())
        folded = sample.casefold()[:64_000]
        return any(
            marker in folded
            for marker in (
                "access denied",
                "request rejected",
                "forbidden",
                "captcha",
                "cloudflare",
                "web application firewall",
            )
        )

    def parse_list(self, response):
        page = response.meta["page"]
        scope = response.meta.get("pagination_scope") or self.pagination_scope()
        tracker = get_pagination_reconciler(self, scope, max_pages=MAX_PAGES)
        if response.status != 200:
            self.logger.warning("constcourt: page %s returned HTTP %s; stopping", page, response.status)
            tracker.mark_failure(
                "callback_failure", cursor=page, detail=f"HTTP {response.status}"
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        items = response.css("div.legal-act-info")

        if not items:
            # An empty page normally signals end-of-results, but a transient WAF 200-block also
            # returns 200-with-no-items — indistinguishable here. Re-check the SAME page once
            # before concluding, so a blip doesn't silently truncate coverage; a genuinely empty
            # end page just costs one extra fetch.
            if response.meta.get("empty_retry"):
                if self._looks_waf_blocked(response):
                    tracker.mark_failure(
                        "empty_waf_response",
                        cursor=page,
                        detail="uncached terminal retry matched a WAF/block page",
                    )
                else:
                    tracker.observe_page(
                        page,
                        [],
                        page_number=page,
                        terminal=True,
                    )
                finalize_pagination_scope(self, tracker, url=response.url)
            else:
                self.logger.info("constcourt: page %s empty — re-checking once (guards a transient "
                                 "200 block) before stopping", page)
                yield self.request_page(page, empty_retry=True)
            return

        identities = []
        for block in items:
            href = block.css("h5.legal-act-title a::attr(href)").get()
            identities.append(
                parse_qs(urlparse(href).query).get("legal", [None])[0]
                if href
                else None
            )
        tracker.observe_page(
            page,
            identities,
            page_number=page,
            terminal=False,
        )

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

        # Paginate until a page comes back empty (handled above, with a one-shot re-check),
        # with a hard cap as a runaway guard.
        if page < MAX_PAGES:
            yield self.request_page(page + 1)
        else:
            self.logger.warning("constcourt: hit MAX_PAGES=%s cap; stopping pagination", MAX_PAGES)
            tracker.mark_cap(cursor=page, configured_cap=MAX_PAGES)
            finalize_pagination_scope(self, tracker, url=response.url)

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
            data.update(
                {
                    "content_kind": "full_text",
                    "content_complete": True,
                    "extraction_status": ExtractionStatus.FULL_TEXT.value,
                }
            )
            yield self.load_item(data)

    def parse_docx_body(self, response):
        data = response.meta["data"]
        try:
            result = docx_to_markdown(
                response.body,
                declared_mime=response.headers.get(b"Content-Type"),
            )
        except Exception as exc:
            self.logger.warning("DOCX parse failed for %s: %s", response.url, exc)
            self.record_quality_failure(
                "document_parse_failed",
                response.url,
                detail=str(exc),
                context={"legal_id": data.get("legal_id")},
            )
            return
        if result.status is not ExtractionStatus.FULL_TEXT:
            self.record_quality_failure(
                f"document_{result.status.value}",
                response.url,
                detail=result.detail,
                context={"legal_id": data.get("legal_id")},
            )
            return
        data["body_markdown"] = result.text
        data["content_kind"] = result.content_kind
        data["content_complete"] = result.content_complete
        data["extraction_status"] = result.status.value
        data["source_binary_url"] = response.url
        yield self.load_item(data)

    @staticmethod
    def load_item(data):
        loader = ItemLoader(item=ConstcourtItem())
        for key, value in data.items():
            loader.add_value(key, value)
        return loader.load_item()
