import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scrapy
from scrapy import signals
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse
from scrapy.loader import ItemLoader

from ..items import MatsneItem
from ..utils.dates import parse_dotted
from ..utils.search_urls import (
    DATE_FORMAT,
    DOC_TYPES,
    build_search_url,
    first_qs_value,
    generate_start_url_batches,
    sub_windows,
)
from .base import BaseLegalSpider


def status_from_effective_dates(
    entry_into_force: str | None,
    expiry: str | None,
    *,
    as_of: date | None = None,
) -> str:
    """Derive the Georgian Matsne status at an explicit calendar date."""
    as_of = as_of or date.today()
    starts = parse_dotted(entry_into_force)
    ends = parse_dotted(expiry)
    if starts is not None and starts > as_of:
        return "ასამოქმედებელი აქტები"
    if ends is not None and ends <= as_of:
        return "ძალადაკარგული აქტები"
    return "ძალაში მყოფი აქტები"


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
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def _doc_type(self) -> str:
        """The ``-a doc_type=`` crawl mode: "all" (default) or "main".

        "main" targets matsne's ძირითადი (კონსოლიდირებული) filter (``type=main``) —
        the ~52k base normative acts that carry the current consolidated text. In that
        mode every listed document_id is also recorded to ``main_listed_ids.txt`` in
        the run dir (listing membership is the only authority for "main document";
        the detail page of a never-amended base act is indistinguishable from an
        amendment act's), and scraped items are stamped ``is_consolidated=True`` via
        per-request ``main_listed`` meta. When the crawl finishes cleanly a
        ``main_listed_ids.txt.complete`` sentinel (containing the id count) is written —
        consumers (scripts/reconcile_consolidated.py) treat sentinel-less files as
        partial enumerations.

        Seed mode contract: combining ``-a seed_ids_file=`` with ``-a doc_type=main``
        also stamps every seeded doc consolidated — only do that when the seed list
        came from a type=main enumeration (e.g. missing_consolidated_ids.txt).
        """
        doc_type = str(getattr(self, "doc_type", "") or "all")
        if doc_type not in DOC_TYPES:
            raise ValueError(f"doc_type must be one of {DOC_TYPES}, got {doc_type!r}")
        return doc_type

    def _record_main_listed_id(self, document_id: str) -> None:
        """Append a listing-enumerated id to ``<run_dir>/main_listed_ids.txt`` (dedup'd).

        Written BEFORE the seen-store skip, so the file enumerates the full type=main
        universe even when every doc is already scraped. Append-mode keeps a partial
        enumeration if the crawl dies. No-op in unit tests without run outputs.
        """
        if not document_id:
            return
        if not hasattr(self, "_main_listed_ids"):
            self._main_listed_ids: set[str] = set()
        if document_id in self._main_listed_ids:
            return
        self._main_listed_ids.add(document_id)
        run_dir = getattr(self, "run_dir", None)
        if run_dir is None:
            return
        with (run_dir / "main_listed_ids.txt").open("a", encoding="utf-8") as fh:
            fh.write(document_id + "\n")

    async def start(self):
        self.seen_request_urls = set()
        self._doc_type()  # fail fast on a bad -a doc_type= argument

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
                    url,
                    callback=self.parse_document,
                    # Seed contract (see _doc_type): doc_type=main declares the seed
                    # list to be main-universe ids, so seeds inherit the stamp.
                    meta={"item": item, "main_listed": self._doc_type() == "main"},
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
                    build_search_url(win_start, win_end, "", "", doc_type=self._doc_type()),
                    callback=self.parse,
                )
                if request:
                    yield request
            return

        self.first_batch_urls, self.deferred_batch_urls = generate_start_url_batches(
            self.scraping_start_date,
            self.scraping_end_date,
            doc_type=self._doc_type(),
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

    def spider_closed(self, reason: str):
        """On a CLEAN finish of a doc_type=main crawl, mark the enumeration complete.

        Writes ``<run_dir>/main_listed_ids.txt.complete`` containing the listed-id
        count. reconcile_consolidated.py refuses destructive passes on sidecars
        without a matching sentinel (an aborted crawl leaves a partial enumeration
        that must never be treated as the full type=main universe).
        """
        if reason != "finished":
            return
        if self._doc_type() != "main":
            return
        run_dir = getattr(self, "run_dir", None)
        if run_dir is None:
            return
        count = len(getattr(self, "_main_listed_ids", ()))
        (run_dir / "main_listed_ids.txt.complete").write_text(f"{count}\n", encoding="utf-8")

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
                if self._doc_type() == "main":
                    self._record_main_listed_id(document_id)
                if self.is_seen({"document_id": document_id}):
                    self.crawler.stats.inc_value("dedup/skipped")
                    continue
                request = self.follow_request(
                    response,
                    href,
                    callback=self.parse_document,
                    # main_listed = listing provenance for the consolidation stamp in
                    # parse_document (per-request, so mixed flows can't mis-stamp).
                    meta={"item": loaded_item, "main_listed": self._doc_type() == "main"},
                )
                if request:
                    yield request

        next_page_url = response.xpath(
            "//ul[contains(@class,'pagination')]//a[contains(normalize-space(.),'შემდეგი')]/@href"
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
            "//ul[contains(@class,'pagination')]//a[contains(normalize-space(.),'ბოლო')]/@href"
        ).get()
        if not last_href:
            return None
        last_page = int(first_qs_value(parse_qs(urlparse(last_href).query), "page") or "1")
        if last_page <= self.MAX_SAFE_PAGES:
            return None

        topic = first_qs_value(parsed_url, "label") or ""
        additional_status = first_qs_value(parsed_url, "additional_status") or ""
        # Recover the document-class filter from the page's own URL so a monthly
        # re-split of a type=main window keeps enumerating only main documents.
        doc_type = first_qs_value(parsed_url, "type") or "all"
        requests = []
        for sub_start, sub_end in sub_windows(win_start, win_end, "monthly"):
            url = build_search_url(sub_start, sub_end, topic, additional_status, doc_type)
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

        # Consolidation: the #publication-switcher lists each consolidated version's
        # date; it is present only for acts that were amended/re-published. But matsne's
        # own "consolidated document" class (type=main) is WIDER: it also contains base
        # acts never amended (no switcher) — their detail pages are indistinguishable
        # from amendment acts', so listing membership is the only authority. Switcher
        # presence therefore gives a sound lower bound: ≥1 option ⇒ definitely a main
        # (consolidated) document; a doc_type=main crawl stamps True by provenance.
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
        if response.meta.get("main_listed"):
            # Provenance stamp: this doc was reached through a type=main listing (or a
            # seed list declared main via -a doc_type=main), so it is a main
            # (consolidated) document by matsne's classifier, switcher or not.
            loaded["is_consolidated"] = True

        # Seed-mode docs skip the search listing, so they carry no status panel; derive the
        # status from effective/expiry dates as of today so future transitions are not applied
        # early and the ingest status filter remains temporally correct.
        if not loaded.get("status"):
            loaded["status"] = status_from_effective_dates(
                loaded.get("entry_into_force_date"),
                loaded.get("expiry_date"),
            )

        yield loaded
