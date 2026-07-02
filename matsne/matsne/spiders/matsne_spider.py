from urllib.parse import parse_qs, urlparse

import scrapy
from scrapy import signals
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse
from scrapy.loader import ItemLoader

from ..items import MatsneItem
from ..utils.search_urls import first_qs_value, generate_start_url_batches
from .base import BaseLegalSpider


class MatsneSpider(BaseLegalSpider):
    name = "matsne"
    DEDUP_KEY = ("document_id",)

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_idle, signal=signals.spider_idle)
        return spider

    async def start(self):
        self.first_batch_urls, self.deferred_batch_urls = generate_start_url_batches(
            self.scraping_start_date,
            self.scraping_end_date,
        )
        self.seen_request_urls = set()
        self.deferred_batch_started = False

        for request in self.start_phase(1):
            yield request

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

        yield loader.load_item()
