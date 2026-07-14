"""tbappeal.court.ge — Tbilisi Court of Appeals "interesting decisions".

Server-rendered HTML (nginx). Published decisions live in the
``/ka/category/saintereso-gadatsqvetilebebi`` news category. Unlike the other
sources there is **no server-side date filter**, so we crawl the (small, ~7-page,
reverse-chronological) category and filter each post in-spider by its listed date,
deduping by detail-page slug. The full ruling is attached as a PDF (``pdf_url``) and
is the only complete ``body_markdown``. The news article is retained separately as
``article_summary``; a summary-only fallback is explicitly incomplete and retryable.

Note: only the bare ``tbappeal.court.ge`` host works — the ``www.`` variant does not
resolve, so all URLs use the apex domain.
"""

from urllib.parse import parse_qs, urlparse

from scrapy import Request
from scrapy.loader import ItemLoader

from ..items import TbappealItem
from ..utils.dates import parse_dotted
from ..utils.documents import ExtractionStatus, pdf_to_markdown
from ..utils.markdown import safe_html_to_markdown
from ..utils.pagination import (
    finalize_pagination_scope,
    get_pagination_reconciler,
    handle_pagination_request_failure,
)
from .base import BaseLegalSpider

BASE = "https://tbappeal.court.ge"
CATEGORY_PATH = "/ka/category/saintereso-gadatsqvetilebebi"
MAX_PAGES = 1000  # safety cap (category is ~7 pages today)


class TbappealSpider(BaseLegalSpider):
    name = "tbappeal"
    DEDUP_KEY = ("slug",)

    async def start(self):
        self.seen_slugs = set()
        yield self.request_page(1)

    @staticmethod
    def pagination_scope():
        return "interesting-decisions"

    def pagination_request_failed(self, failure):
        return handle_pagination_request_failure(self, failure)

    def request_page(self, page):
        scope = self.pagination_scope()
        get_pagination_reconciler(self, scope, max_pages=MAX_PAGES)
        return Request(
            f"{BASE}{CATEGORY_PATH}?page={page}",
            callback=self.parse_list,
            errback=self.pagination_request_failed,
            meta={
                "page": page,
                "pagination_scope": scope,
                "pagination_cursor": page,
            },
        )

    @staticmethod
    def _advertised_pages(response):
        """Read only an explicit last-page link; sliding page windows are not totals."""
        hrefs = response.css(
            "a[rel='last']::attr(href), "
            ".pagination a.last::attr(href), "
            ".pagination li.last a::attr(href)"
        ).getall()
        pages = []
        for href in hrefs:
            raw_page = parse_qs(urlparse(response.urljoin(href)).query).get("page")
            if not raw_page:
                continue
            try:
                page = int(raw_page[-1])
            except (TypeError, ValueError):
                continue
            if page > 0:
                pages.append(page)
        return max(pages) if pages else None

    def parse_list(self, response):
        page = response.meta["page"]
        scope = response.meta.get("pagination_scope") or self.pagination_scope()
        tracker = get_pagination_reconciler(
            self,
            scope,
            max_pages=MAX_PAGES,
        )
        if response.status != 200:
            self.logger.warning("tbappeal: page %s returned HTTP %s; stopping", page, response.status)
            tracker.mark_failure(
                "callback_failure",
                cursor=page,
                detail=f"HTTP {response.status}",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        posts = list(response.css("div.grid-post"))
        entries = []
        for post in posts:
            href = post.css("a::attr(href)").get()
            slug = href.rstrip("/").rsplit("/", 1)[-1] if href else None
            entries.append((post, href, slug))

        advertised_pages = self._advertised_pages(response)
        if advertised_pages is None:
            advertised_pages = tracker.advertised_pages_max
        terminal = not posts or (
            advertised_pages is not None and page >= advertised_pages
        )
        tracker.observe_page(
            page,
            [slug for _, _, slug in entries],
            advertised_pages=advertised_pages,
            page_number=page,
            terminal=terminal,
        )
        seen_slugs = getattr(self, "seen_slugs", None)
        if seen_slugs is None:
            seen_slugs = set()
            self.seen_slugs = seen_slugs
        for post, href, slug in entries:
            try:
                if not href:
                    continue
                if slug in seen_slugs:
                    continue
                # Cross-run dedup: skip documents already scraped in a previous run.
                if self.is_seen({"slug": slug}):
                    self.crawler.stats.inc_value("dedup/skipped")
                    continue

                date_str = (post.css("span.date::text").get() or "").strip()
                if not self._in_window(date_str):
                    continue

                seen_slugs.add(slug)
                image = post.css("img::attr(src)").get()
                detail_request = response.follow(
                    href,
                    callback=self.parse_detail,
                    errback=self.request_failed,
                    meta={
                        "slug": slug,
                        "date": date_str,
                        "title": (post.css("h4 a::text").get() or "").strip(),
                        "featured_image_url": (
                            response.urljoin(image) if image else None
                        ),
                        "source_url": response.urljoin(href),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - retain remaining listing rows
                tracker.mark_failure(
                    "callback_failure",
                    cursor=page,
                    detail=f"record processing failed: {exc}",
                )
                self.logger.warning("tbappeal: record processing failed: %s", exc)
                continue
            yield detail_request

        # The category is tiny (~7 pages); follow until a page returns no posts.
        if terminal:
            finalize_pagination_scope(self, tracker, url=response.url)
        elif page < MAX_PAGES:
            yield self.request_page(page + 1)
        else:
            self.logger.warning("tbappeal: hit MAX_PAGES=%s cap; stopping pagination", MAX_PAGES)
            tracker.mark_cap(cursor=page, configured_cap=MAX_PAGES)
            finalize_pagination_scope(self, tracker, url=response.url)

    def _in_window(self, date_str):
        parsed = parse_dotted(date_str)
        if parsed is None:
            return True  # keep undated/unparseable posts rather than silently dropping
        return self.scraping_start_date <= parsed <= self.scraping_end_date

    def parse_detail(self, response):
        body_html = response.css("div.blog-details").get()
        article_summary = safe_html_to_markdown(
            body_html, base_url=BASE, source_url=response.url
        )

        pdf_href = response.css(
            "div.blog-details a[href*='/uploads/']::attr(href)"
        ).re_first(r".*\.pdf")

        preferred_title = (response.css("h2.mb-5::text").get() or "").strip()
        data = {
            "source_url": response.meta["source_url"],
            "slug": response.meta["slug"],
            "title": preferred_title or response.meta["title"],
            "date": response.meta["date"],
            "featured_image_url": response.meta["featured_image_url"],
            "article_summary": article_summary,
        }
        if pdf_href:
            pdf_url = response.urljoin(pdf_href)
            data["pdf_url"] = pdf_url
            data["source_binary_url"] = pdf_url
            yield response.follow(
                pdf_url,
                callback=self.parse_pdf,
                errback=self.request_failed,
                meta={"fields": data},
            )
            return

        self.record_quality_failure(
            "missing_ruling_pdf",
            response.url,
            detail="article has no ruling PDF link",
            context={"slug": data["slug"]},
        )
        yield self._summary_fallback(data, ExtractionStatus.MALFORMED)

    def parse_pdf(self, response):
        data = response.meta["fields"]
        try:
            result = pdf_to_markdown(
                response.body,
                declared_mime=response.headers.get(b"Content-Type"),
            )
        except Exception as exc:
            self.record_quality_failure(
                "document_parse_failed",
                response.url,
                detail=str(exc),
                context={"slug": data.get("slug")},
            )
            yield self._summary_fallback(data, ExtractionStatus.MALFORMED)
            return

        if result.status is not ExtractionStatus.FULL_TEXT:
            self.record_quality_failure(
                f"document_{result.status.value}",
                response.url,
                detail=result.detail,
                context={"slug": data.get("slug")},
            )
            yield self._summary_fallback(data, result.status)
            return

        data.update(
            {
                "body_markdown": result.text,
                "content_kind": "ruling_full_text",
                "content_complete": True,
                "extraction_status": result.status.value,
            }
        )
        yield self.load_item(data)

    @classmethod
    def _summary_fallback(cls, data, status):
        data.update(
            {
                "body_markdown": data.get("article_summary") or "",
                "content_kind": "article_summary",
                "content_complete": False,
                "extraction_status": status.value,
            }
        )
        return cls.load_item(data)

    @staticmethod
    def load_item(data):
        loader = ItemLoader(item=TbappealItem())
        for key, value in data.items():
            loader.add_value(key, value)
        return loader.load_item()
