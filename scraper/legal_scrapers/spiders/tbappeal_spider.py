"""tbappeal.court.ge — Tbilisi Court of Appeals "interesting decisions".

Server-rendered HTML (nginx). Published decisions live in the
``/ka/category/saintereso-gadatsqvetilebebi`` news category. Unlike the other
sources there is **no server-side date filter**, so we crawl the (small, ~7-page,
reverse-chronological) category and filter each post in-spider by its listed date,
deduping by detail-page slug. The full ruling is attached as a PDF (``pdf_url``);
``body_markdown`` holds the post's article text.

Note: only the bare ``tbappeal.court.ge`` host works — the ``www.`` variant does not
resolve, so all URLs use the apex domain.
"""

from scrapy import Request
from scrapy.loader import ItemLoader

from ..items import TbappealItem
from ..utils.dates import parse_dotted
from ..utils.markdown import safe_html_to_markdown
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

    def request_page(self, page):
        return Request(
            f"{BASE}{CATEGORY_PATH}?page={page}",
            callback=self.parse_list,
            errback=self.request_failed,
            meta={"page": page},
        )

    def parse_list(self, response):
        page = response.meta["page"]
        if response.status != 200:
            self.logger.warning("tbappeal: page %s returned HTTP %s; stopping", page, response.status)
            return
        posts = response.css("div.grid-post")

        for post in posts:
            href = post.css("a::attr(href)").get()
            if not href:
                continue
            slug = href.rstrip("/").rsplit("/", 1)[-1]
            if slug in self.seen_slugs:
                continue
            # Cross-run dedup: skip documents already scraped in a previous run.
            if self.is_seen({"slug": slug}):
                self.crawler.stats.inc_value("dedup/skipped")
                continue

            date_str = (post.css("span.date::text").get() or "").strip()
            if not self._in_window(date_str):
                continue

            self.seen_slugs.add(slug)
            image = post.css("img::attr(src)").get()
            yield response.follow(
                href,
                callback=self.parse_detail,
                errback=self.request_failed,
                meta={
                    "slug": slug,
                    "date": date_str,
                    "title": (post.css("h4 a::text").get() or "").strip(),
                    "featured_image_url": response.urljoin(image) if image else None,
                    "source_url": response.urljoin(href),
                },
            )

        # The category is tiny (~7 pages); follow until a page returns no posts.
        if posts and page < MAX_PAGES:
            yield self.request_page(page + 1)
        elif posts:
            self.logger.warning("tbappeal: hit MAX_PAGES=%s cap; stopping pagination", MAX_PAGES)

    def _in_window(self, date_str):
        parsed = parse_dotted(date_str)
        if parsed is None:
            return True  # keep undated/unparseable posts rather than silently dropping
        return self.scraping_start_date <= parsed <= self.scraping_end_date

    def parse_detail(self, response):
        body_html = response.css("div.blog-details").get()
        body_markdown = safe_html_to_markdown(body_html, base_url=BASE, source_url=response.url)

        pdf_href = response.css(
            "div.blog-details a[href*='/uploads/']::attr(href)"
        ).re_first(r".*\.pdf")

        loader = ItemLoader(item=TbappealItem(), response=response)
        loader.add_value("source_url", response.meta["source_url"])
        loader.add_value("slug", response.meta["slug"])
        loader.add_css("title", "h2.mb-5::text")          # preferred title
        loader.add_value("title", response.meta["title"])  # fallback (TakeFirst)
        loader.add_value("date", response.meta["date"])
        loader.add_value("pdf_url", response.urljoin(pdf_href) if pdf_href else None)
        loader.add_value("featured_image_url", response.meta["featured_image_url"])
        loader.add_value("body_markdown", body_markdown)
        yield loader.load_item()
