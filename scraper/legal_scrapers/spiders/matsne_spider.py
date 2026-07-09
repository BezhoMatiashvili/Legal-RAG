import re
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scrapy
from scrapy import signals
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse
from scrapy.loader import ItemLoader

from ..items import MatsneItem
from ..utils.search_urls import (
    DATE_FORMAT,
    build_search_url,
    first_qs_value,
    generate_start_url_batches,
    sub_windows,
)
from .base import BaseLegalSpider


class MatsneSpider(BaseLegalSpider):
    name = "matsne"
    DEDUP_KEY = ("document_id",)

    # If a window's first results page advertises more than this many pages, the deep
    # "შემდეგი" chain is fragile — re-issue the window split monthly instead.
    MAX_SAFE_PAGES = 90

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_idle, signal=signals.spider_idle)
        return spider

    async def start(self):
        self.seen_request_urls = set()

        seed_urls = self._load_seed_urls()
        if seed_urls:
            # Seed mode: fetch a known list of document/view/{id} detail pages directly,
            # bypassing search discovery — the backfill path for docs a date-window sweep
            # can't reach. No phase-2 catch-all runs.
            self.first_batch_urls = self.deferred_batch_urls = []
            self.deferred_batch_started = True
            for url in seed_urls:
                document_id = urlparse(url).path.rstrip("/").split("/")[-1]
                if self.is_seen({"document_id": document_id}):
                    self.crawler.stats.inc_value("dedup/skipped")
                    continue
                item = MatsneItem()
                item["document_url"] = url
                request = self.build_request(
                    url, callback=self.parse_document, meta={"item": item}
                )
                if request:
                    yield request
            return

        if getattr(self, "catch_all_only", None):
            # Completeness shortcut: the empty-topic + empty-status catch-all listing returns
            # the WHOLE corpus, so the topic×status phase-1 pass is pure redundancy (it only
            # re-lists docs the catch-all already covers). Run just the catch-all, per yearly
            # window — same coverage, ~half the listing requests. No phase-2 transition.
            self.first_batch_urls = self.deferred_batch_urls = []
            self.deferred_batch_started = True
            for win_start, win_end in sub_windows(
                self.scraping_start_date, self.scraping_end_date, "yearly"
            ):
                request = self.build_request(
                    build_search_url(win_start, win_end, "", ""), callback=self.parse
                )
                if request:
                    yield request
            return

        self.first_batch_urls, self.deferred_batch_urls = generate_start_url_batches(
            self.scraping_start_date,
            self.scraping_end_date,
        )
        self.deferred_batch_started = False

        for request in self.start_phase(1):
            yield request

    def _load_seed_urls(self) -> list[str]:
        """Collect seed detail URLs from ``-a seed_ids_file=<path>`` and/or ``-a seed_urls=``.

        Each token may be a full ``document/view/{id}`` URL or a bare numeric id; bare ids
        are expanded to the canonical ka detail URL. Order-preserving de-dup.
        """
        tokens: list[str] = []
        seed_file = getattr(self, "seed_ids_file", None) or getattr(self, "seed_file", None)
        if seed_file:
            tokens.extend(Path(seed_file).read_text(encoding="utf-8").split())
        seed_urls = getattr(self, "seed_urls", None) or getattr(self, "seed_ids", None)
        if seed_urls:
            tokens.extend(re.split(r"[,\s]+", seed_urls))

        urls: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token.startswith("http"):
                url = token
            else:
                document_id = token.rstrip("/").split("/")[-1]
                url = f"https://matsne.gov.ge/ka/document/view/{document_id}"
            if url not in seen:
                seen.add(url)
                urls.append(url)
        return urls

    def spider_idle(self):
        """Start the deferred (catch-all) batch once phase 1 has fully drained.

        Driven by the ``spider_idle`` signal rather than a per-request counter, so an
        exception in any phase-1 callback can never strand the phase-2 transition (Scrapy
        does not route callback exceptions to errbacks). Fires only when the scheduler and
        downloader are empty, i.e. after every phase-1 request (incl. pagination/detail)
        has completed.
        """
        if self.deferred_batch_started:
            return
        self.deferred_batch_started = True
        scheduled = False
        for request in self.start_phase(2):
            self.crawler.engine.crawl(request)
            scheduled = True
        if scheduled:
            raise DontCloseSpider

    def start_phase(self, phase: int):
        urls = self.first_batch_urls if phase == 1 else self.deferred_batch_urls
        self.logger.info("Starting crawl phase %s with %s search URLs", phase, len(urls))
        for url in urls:
            request = self.build_request(url, callback=self.parse)
            if request:
                yield request

    def build_request(self, url: str, callback, **kwargs):
        if url in self.seen_request_urls:
            return None
        self.seen_request_urls.add(url)
        # dont_filter=True: dedup is owned by seen_request_urls, so Scrapy's
        # canonicalizing dupefilter can't silently drop a request we counted as new.
        return scrapy.Request(
            url=url,
            callback=callback,
            errback=self.request_failed,
            dont_filter=True,
            **kwargs,
        )

    def follow_request(self, response: HtmlResponse, url: str, callback, **kwargs):
        absolute_url = response.urljoin(url)
        if absolute_url in self.seen_request_urls:
            return None
        self.seen_request_urls.add(absolute_url)
        return response.follow(
            absolute_url,
            callback=callback,
            errback=self.request_failed,
            dont_filter=True,
            **kwargs,
        )

    def parse(self, response: HtmlResponse):
        parsed_url = parse_qs(urlparse(response.url).query)

        # On the first page of a window, if the listing advertises too many pages, re-issue
        # the window split into months instead of following the fragile deep next-page chain.
        if first_qs_value(parsed_url, "page") in (None, "1"):
            split_requests = self._split_requests(response, parsed_url)
            if split_requests:
                yield from split_requests
                return

        items = response.xpath("//ul[@class='list-unstyled document-search-result-items']/li")
        for item in items:
            loader = ItemLoader(item=MatsneItem(), selector=item)
            href = item.xpath(".//a/@href").get()
            if href:
                loader.add_value("document_url", response.urljoin(href))
            loader.add_xpath("status", ".//@class")
            loader.add_value("additional_status", first_qs_value(parsed_url, "additional_status"))
            loader.add_value("document_topic", first_qs_value(parsed_url, "label"))
            loaded_item = loader.load_item()

            href = loaded_item.get("document_url")
            if href:
                document_id = urlparse(href).path.split("/")[-1]
                if self.is_seen({"document_id": document_id}):
                    self.crawler.stats.inc_value("dedup/skipped")
                    continue
                request = self.follow_request(
                    response,
                    href,
                    callback=self.parse_document,
                    meta={"item": loaded_item},
                )
                if request:
                    yield request

        next_page_url = response.xpath(
            "//ul[contains(@class,'pagination')]//a[normalize-space()='შემდეგი']/@href"
        ).get()

        if next_page_url:
            request = self.follow_request(response, next_page_url, callback=self.parse)
            if request:
                yield request

    def _split_requests(self, response: HtmlResponse, parsed_url: dict):
        """Monthly re-split requests for an over-deep window, or None to page normally.

        Reads the advertised last page from the "ბოლო" (last) pagination link. If it
        exceeds ``MAX_SAFE_PAGES`` and the window is wider than a month, return one page-1
        request per monthly sub-window (whose own listings are shallow); the caller then
        skips this window's deep next-page chain. seen.sqlite dedup keeps re-listed docs
        from being re-fetched.
        """
        fr = first_qs_value(parsed_url, "publishing_date_fr[date]")
        to = first_qs_value(parsed_url, "publishing_date_to[date]")
        if not fr or not to:
            return None
        try:
            win_start = datetime.strptime(fr, DATE_FORMAT).date()
            win_end = datetime.strptime(to, DATE_FORMAT).date()
        except ValueError:
            return None
        if (win_end - win_start).days <= 40:  # already ~monthly — don't split further
            return None

        last_href = response.xpath(
            "//ul[contains(@class,'pagination')]//a[normalize-space()='ბოლო']/@href"
        ).get()
        if not last_href:
            return None
        last_page = int(first_qs_value(parse_qs(urlparse(last_href).query), "page") or "1")
        if last_page <= self.MAX_SAFE_PAGES:
            return None

        topic = first_qs_value(parsed_url, "label") or ""
        additional_status = first_qs_value(parsed_url, "additional_status") or ""
        requests = []
        for sub_start, sub_end in sub_windows(win_start, win_end, "monthly"):
            url = build_search_url(sub_start, sub_end, topic, additional_status)
            request = self.follow_request(response, url, callback=self.parse)
            if request:
                requests.append(request)
        return requests or None

    def parse_document(self, response: HtmlResponse):
        t_xpath = "//*[@id='block-system-main']//table[contains(@class,'table-info')]"
        item = response.meta["item"]
        loader = ItemLoader(item=item, response=response)

        path_parts = urlparse(item.get("document_url") or "").path.split("/")
        if len(path_parts) > 1 and path_parts[1]:
            loader.add_value("language", path_parts[1])
        if path_parts and path_parts[-1]:
            loader.add_value("document_id", path_parts[-1])
        loader.add_xpath("title", f"{t_xpath}//th/text()")

        loader.add_xpath('document_number', f"{t_xpath}//td[normalize-space()='დოკუმენტის ნომერი']/following-sibling::td[1]/text()")
        loader.add_xpath('document_recipient', f"{t_xpath}//td[normalize-space()='დოკუმენტის მიმღები']/following-sibling::td[1]/text()")
        loader.add_xpath('adoption_date', f"{t_xpath}//td[normalize-space()='მიღების თარიღი']/following-sibling::td[1]/text()")
        loader.add_xpath('document_type', f"{t_xpath}//td[normalize-space()='დოკუმენტის ტიპი']/following-sibling::td[1]/text()")
        loader.add_xpath('registration_code', f"{t_xpath}//td[normalize-space()='სარეგისტრაციო კოდი']/following-sibling::td[1]/text()")
        loader.add_xpath('publication_source', f"{t_xpath}//td[normalize-space()='გამოქვეყნების წყარო, თარიღი']/following-sibling::td[1]/text()")
        loader.add_xpath('publication_date', f"{t_xpath}//td[normalize-space()='გამოქვეყნების წყარო, თარიღი']/following-sibling::td[1]/text()")
        loader.add_xpath('entry_into_force_date', f"{t_xpath}//td[normalize-space()='ძალაში შესვლის თარიღი']/following-sibling::td[1]/text()")
        loader.add_xpath('expiry_date', f"{t_xpath}//td[normalize-space()='ძალის დაკარგვის თარიღი']/following-sibling::td[1]/text()")
        loader.add_xpath('consolidated_publications', f"{t_xpath}//*[@id='publication-switcher']/option")
        loader.add_xpath('body_markdown', "//*[@id='maindoc']")

        loaded = loader.load_item()

        # Consolidation: the #publication-switcher lists each consolidated version's date.
        # It is present only for acts that were amended/re-published, so ≥1 option ⇒
        # consolidated; an empty switcher (the ~77% majority) ⇒ a one-shot act.
        consolidated_dates = [
            text.strip()
            for text in response.xpath(
                "//*[@id='publication-switcher']/option/text()"
            ).getall()
            if text and text.strip()
        ]
        loaded["consolidated_dates"] = consolidated_dates
        loaded["consolidated_count"] = len(consolidated_dates)
        loaded["is_consolidated"] = len(consolidated_dates) >= 1

        # Seed-mode docs skip the search listing, so they carry no status panel; derive the
        # status from the expiry date (a repealed act shows ძალის დაკარგვის თარიღი) so the
        # ingest status filter (in_force / repealed) still works for them.
        if not loaded.get("status"):
            loaded["status"] = (
                "ძალადაკარგული აქტები" if loaded.get("expiry_date") else "ძალაში მყოფი აქტები"
            )

        yield loaded
