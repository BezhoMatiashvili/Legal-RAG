import json
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scrapy
from scrapy.http import HtmlResponse
from scrapy.loader import ItemLoader

from ..items import MatsneItem
from ..utils.search_urls import first_qs_value, generate_start_url_batches


class QuotesSpider(scrapy.Spider):
    name = "matsne"
    DEFAULT_SCRAPING_START_DATE = date(2026, 6, 22)
    ARTIFACTS_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "matsne"
    date_format = "%d-%m-%Y"

    def __init__(self, start_date=None, end_date=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scraping_start_date = self.parse_date_arg(
            start_date,
            "start_date",
            self.DEFAULT_SCRAPING_START_DATE,
        )
        self.scraping_end_date = self.parse_date_arg(end_date, "end_date", date.today())

        if self.scraping_start_date > self.scraping_end_date:
            raise ValueError(
                "start_date must be on or before end_date "
                f"({self.scraping_start_date.isoformat()} > "
                f"{self.scraping_end_date.isoformat()})"
            )

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.configure_run_outputs(crawler.settings)
        return spider

    @classmethod
    def parse_date_arg(cls, value, argument_name: str, default: date) -> date:
        if value is None or value == "":
            return default

        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{argument_name} must use YYYY-MM-DD format, got {value!r}"
            ) from exc

    def configure_run_outputs(self, settings):
        self.started_at = datetime.now(UTC).replace(microsecond=0)
        self.run_id = self.build_run_id()
        self.run_dir = self.ARTIFACTS_ROOT / "runs" / self.run_id
        self.latest_dir = self.ARTIFACTS_ROOT / "latest"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

        self.items_path = self.run_dir / "items.jsonl"
        self.latest_items_path = self.latest_dir / "items.jsonl"
        self.log_path = self.run_dir / "spider.log"
        self.run_metadata_path = self.run_dir / "run.json"
        self.latest_metadata_path = self.latest_dir / "run.json"

        settings.set(
            "FEEDS",
            {
                str(self.items_path): self.feed_options(),
                str(self.latest_items_path): self.feed_options(),
            },
            priority="spider",
        )
        settings.set("LOG_FILE", str(self.log_path), priority="spider")

        self.write_run_metadata()

    def build_run_id(self) -> str:
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        base_run_id = (
            f"{timestamp}_start-{self.scraping_start_date.isoformat()}"
            f"_end-{self.scraping_end_date.isoformat()}"
        )
        run_id = base_run_id
        suffix = 2
        while (self.ARTIFACTS_ROOT / "runs" / run_id).exists():
            run_id = f"{base_run_id}-{suffix}"
            suffix += 1
        return run_id

    @staticmethod
    def feed_options() -> dict:
        return {
            "format": "jsonlines",
            "encoding": "utf8",
            "store_empty": False,
            "overwrite": True,
        }

    def write_run_metadata(self):
        metadata = {
            "run_id": self.run_id,
            "spider": self.name,
            "start_date": self.scraping_start_date.isoformat(),
            "end_date": self.scraping_end_date.isoformat(),
            "started_at": self.started_at.isoformat(),
            "items_path": str(self.items_path),
            "latest_items_path": str(self.latest_items_path),
            "log_path": str(self.log_path),
        }
        metadata_json = json.dumps(metadata, indent=2) + "\n"
        self.run_metadata_path.write_text(metadata_json, encoding="utf-8")
        self.latest_metadata_path.write_text(metadata_json, encoding="utf-8")

    async def start(self):
        self.first_batch_urls, self.deferred_batch_urls = generate_start_url_batches(
            self.scraping_start_date,
            self.scraping_end_date,
        )
        self.pending_phase_requests = {1: 0, 2: 0}
        self.deferred_batch_started = False
        self.seen_request_urls = set()

        for request in self.start_phase(1):
            yield request

    def start_phase(self, phase: int):
        urls = self.first_batch_urls if phase == 1 else self.deferred_batch_urls

        if phase == 2:
            self.deferred_batch_started = True

        self.logger.info("Starting crawl phase %s with %s search URLs", phase, len(urls))

        for url in urls:
            request = self.build_request(url, callback=self.parse, phase=phase)
            if request:
                yield request

    def build_request(self, url: str, callback, phase: int, **kwargs):
        if url in self.seen_request_urls:
            return None

        self.seen_request_urls.add(url)
        self.pending_phase_requests[phase] += 1
        meta = kwargs.pop("meta", {})
        meta["_crawl_phase"] = phase
        return scrapy.Request(
            url=url,
            callback=callback,
            errback=self.request_failed,
            meta=meta,
            **kwargs,
        )

    def follow_request(self, response: HtmlResponse, url: str, callback, phase: int, **kwargs):
        absolute_url = response.urljoin(url)
        if absolute_url in self.seen_request_urls:
            return None

        self.seen_request_urls.add(absolute_url)
        self.pending_phase_requests[phase] += 1
        meta = kwargs.pop("meta", {})
        meta["_crawl_phase"] = phase
        return response.follow(
            absolute_url,
            callback=callback,
            errback=self.request_failed,
            meta=meta,
            **kwargs,
        )

    def request_done(self, request_or_response):
        phase = request_or_response.meta["_crawl_phase"]
        self.pending_phase_requests[phase] -= 1

        if phase == 1 and self.pending_phase_requests[1] == 0 and not self.deferred_batch_started:
            yield from self.start_phase(2)

    def request_failed(self, failure):
        self.logger.warning("Request failed: %s", failure.request.url)
        yield from self.request_done(failure.request)

    def parse(self, response: HtmlResponse):
        phase = response.meta["_crawl_phase"]
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
                request = self.follow_request(
                    response,
                    href,
                    callback=self.parse_document,
                    phase=phase,
                    meta={"item": loaded_item},
                )
                if request:
                    yield request

        next_page_url = response.xpath(
            "//ul[contains(@class,'pagination')]//a[normalize-space()='შემდეგი']/@href"
        ).get()

        if next_page_url:
            request = self.follow_request(
                response,
                next_page_url,
                callback=self.parse,
                phase=phase,
            )
            if request:
                yield request

        yield from self.request_done(response)

    def parse_document(self, response: HtmlResponse):
        t_xpath = "//*[@id='block-system-main']//table[contains(@class,'table-info')]"
        loader = ItemLoader(item=response.meta["item"], response=response)

        splitted_url = urlparse(response.meta["item"].get('document_url')).path.split('/')
        loader.add_value('document_id', splitted_url[-1])
        loader.add_value('language', splitted_url[1])
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
        yield from self.request_done(response)
