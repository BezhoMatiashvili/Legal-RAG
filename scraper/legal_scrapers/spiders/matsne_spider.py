import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlparse

import scrapy
from scrapy import signals
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse
from scrapy.loader import ItemLoader

from ..items import MatsneItem
from ..utils.dates import parse_dotted
from ..utils.pagination import (
    finalize_pagination_scope,
    get_pagination_reconciler,
    handle_pagination_request_failure,
)
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
    # A monthly scope cannot be split further safely. Stop and emit repair evidence
    # before an unbounded or malicious paginator can keep the crawl alive forever.
    MAX_PAGES = 20_000
    _WAF_TITLE_MARKERS = (
        "access denied",
        "cloudflare",
        "captcha",
        "request blocked",
    )
    _WAF_BODY_MARKERS = ("cf-chl-", "challenge-platform", "g-recaptcha")

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

    @staticmethod
    def pagination_scope(url: str) -> str:
        """Return a deterministic search-window/filter identity, excluding only page."""
        parsed = urlparse(url)
        query = sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "page"
        )
        canonical = json.dumps(
            [parsed.netloc.lower(), parsed.path, query],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"search:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _page_number(parsed_url: dict) -> int:
        raw_pages = parsed_url.get("page")
        if raw_pages is None:
            raw_page = "1"
        elif not isinstance(raw_pages, list) or len(raw_pages) != 1:
            raise ValueError("Matsne listing URL must contain at most one page value")
        else:
            raw_page = raw_pages[0]
        if not raw_page.isdecimal() or int(raw_page) < 1:
            raise ValueError(f"invalid Matsne page number {raw_page!r}")
        return int(raw_page)

    @staticmethod
    def _advertised_last_page(response: HtmlResponse) -> int | None:
        """Read the largest explicit page link, including Matsne's ``ბოლო`` link."""
        pages: list[int] = []
        hrefs = response.xpath(
            "//ul[contains(@class,'pagination')]//a["
            "contains(normalize-space(.),'ბოლო') or @rel='last']/@href"
        ).getall()
        for href in hrefs:
            page_values = parse_qs(urlparse(href).query).get("page")
            if page_values is None:
                continue
            if len(page_values) != 1:
                raise ValueError("advertised Matsne link has duplicate page values")
            raw_page = page_values[0]
            if not raw_page.isdecimal() or int(raw_page) < 1:
                raise ValueError(f"invalid advertised Matsne page {raw_page!r}")
            pages.append(int(raw_page))
        return max(pages) if pages else None

    @classmethod
    def _looks_waf_blocked(cls, response: HtmlResponse) -> bool:
        title = " ".join(response.xpath("//title//text()").getall()).casefold()
        sample = response.text[:64_000].casefold()
        return any(marker in title for marker in cls._WAF_TITLE_MARKERS) or any(
            marker in sample for marker in cls._WAF_BODY_MARKERS
        )

    def _observe_page_number(self, scope: str, tracker, page_number: int) -> None:
        pages_by_scope = getattr(self, "_matsne_pagination_pages", None)
        if pages_by_scope is None:
            pages_by_scope = {}
            self._matsne_pagination_pages = pages_by_scope
        pages = pages_by_scope.setdefault(scope, set())
        if not pages and page_number != 1:
            tracker.mark_failure(
                "scope_did_not_start_at_page_one",
                cursor=page_number,
                detail=f"first_callback_page={page_number}",
            )
        elif pages and page_number != max(pages) + 1:
            tracker.mark_failure(
                "non_contiguous_page_number",
                cursor=page_number,
                detail=f"previous_max_page={max(pages)}",
            )
        pages.add(page_number)

    def _listing_request_options(self, url: str, callback, kwargs: dict):
        """Attach scope/cursor metadata and the reconciliation errback to listings."""
        options = dict(kwargs)
        if callback != self.parse:
            return self.request_failed, options
        meta = dict(options.pop("meta", {}))
        scope = meta.get("pagination_scope") or self.pagination_scope(url)
        raw_page = first_qs_value(parse_qs(urlparse(url).query), "page") or "1"
        meta.setdefault("pagination_scope", scope)
        meta.setdefault("pagination_cursor", raw_page)
        options["meta"] = meta
        get_pagination_reconciler(self, scope, max_pages=self.MAX_PAGES)
        scope_urls = getattr(self, "_pagination_scope_urls", None)
        if scope_urls is None:
            scope_urls = {}
            self._pagination_scope_urls = scope_urls
        scope_urls.setdefault(scope, url)
        return self.pagination_request_failed, options

    def pagination_request_failed(self, failure):
        return handle_pagination_request_failure(self, failure)

    def _supersede_pagination_scope(
        self,
        scope: str,
        child_requests: list[scrapy.Request],
        *,
        advertised_pages: int,
    ) -> None:
        """Retire a wide parent explicitly when monthly child scopes replace it."""
        trackers = getattr(self, "_pagination_reconcilers", {})
        trackers.pop(scope, None)
        getattr(self, "_pagination_scope_urls", {}).pop(scope, None)
        getattr(self, "_matsne_pagination_pages", {}).pop(scope, None)
        superseded = getattr(self, "_pagination_superseded_scopes", None)
        if superseded is None:
            superseded = {}
            self._pagination_superseded_scopes = superseded
        if scope in superseded:
            return
        superseded[scope] = {
            "reason": "monthly_split",
            "advertised_pages": advertised_pages,
            "child_scopes": tuple(
                sorted(
                    {
                        request.meta["pagination_scope"]
                        for request in child_requests
                        if request.meta.get("pagination_scope")
                    }
                )
            ),
        }
        crawler = getattr(self, "crawler", None)
        if crawler is not None:
            crawler.stats.inc_value("pagination/scopes_superseded")

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

        # Refresh selection is independent of the discovery window. These requests are
        # bounded by BaseLegalSpider and reserve their canonical URLs before listing
        # requests can schedule duplicates.
        for document_id in self.iter_refresh_keys():
            url = f"https://matsne.gov.ge/ka/document/view/{document_id}"
            refresh_context = self.dedup_refresh_context(document_id)
            known_is_consolidated = (
                refresh_context.get("is_consolidated")
                if refresh_context is not None
                else None
            )
            if known_is_consolidated is None:
                self.record_quality_failure(
                    "refresh_context_missing",
                    url,
                    detail=(
                        "direct Matsne refresh refused because the durable success "
                        "state has no consolidation classifier"
                    ),
                    context={"document_id": document_id},
                )
                continue
            item = MatsneItem(document_url=url)
            request = self.build_request(
                url,
                callback=self.parse_document,
                meta={
                    "item": item,
                    "refresh_due": True,
                    "known_is_consolidated": known_is_consolidated,
                },
            )
            if request:
                self.crawler.stats.inc_value("dedup/direct_refresh_scheduled")
                yield request

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
        pagination_ok = True
        if reason == "finished":
            scope_urls = getattr(self, "_pagination_scope_urls", {})
            outcomes = [
                finalize_pagination_scope(
                    self,
                    tracker,
                    url=scope_urls.get(
                        scope,
                        "https://matsne.gov.ge/ka/document/search",
                    ),
                )
                for scope, tracker in list(
                    getattr(self, "_pagination_reconcilers", {}).items()
                )
            ]
            pagination_ok = all(outcome.ok for outcome in outcomes)
        if reason != "finished" or not pagination_ok:
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
        errback, options = self._listing_request_options(url, callback, kwargs)
        # dont_filter=True: dedup is owned by seen_request_urls, so Scrapy's
        # canonicalizing dupefilter can't silently drop a request we counted as new.
        return scrapy.Request(
            url=url,
            callback=callback,
            errback=errback,
            dont_filter=True,
            **options,
        )

    def follow_request(self, response: HtmlResponse, url: str, callback, **kwargs):
        absolute_url = response.urljoin(url)
        if absolute_url in self.seen_request_urls:
            return None
        self.seen_request_urls.add(absolute_url)
        errback, options = self._listing_request_options(
            absolute_url,
            callback,
            kwargs,
        )
        return response.follow(
            absolute_url,
            callback=callback,
            errback=errback,
            dont_filter=True,
            **options,
        )

    def parse(self, response: HtmlResponse):
        parsed_url = parse_qs(urlparse(response.url).query)
        scope = response.meta.get("pagination_scope") or self.pagination_scope(
            response.url
        )
        tracker = get_pagination_reconciler(
            self,
            scope,
            max_pages=self.MAX_PAGES,
        )
        scope_urls = getattr(self, "_pagination_scope_urls", None)
        if scope_urls is None:
            scope_urls = {}
            self._pagination_scope_urls = scope_urls
        scope_urls.setdefault(scope, response.url)

        if self._looks_waf_blocked(response):
            tracker.mark_failure(
                "waf_or_invalid_listing",
                cursor=response.meta.get("pagination_cursor", response.url),
                detail=f"HTTP {response.status}; Matsne listing block marker detected",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return

        try:
            page_number = self._page_number(parsed_url)
            advertised_pages = self._advertised_last_page(response)
        except ValueError as exc:
            tracker.mark_failure(
                "callback_failure",
                cursor=response.meta.get("pagination_cursor", response.url),
                detail=str(exc),
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return

        # On the first page of a window, if the listing advertises too many pages, re-issue
        # the window split into months instead of following the fragile deep next-page chain.
        if page_number == 1:
            split_requests = self._split_requests(
                response,
                parsed_url,
                last_page=advertised_pages,
            )
            if split_requests:
                self._supersede_pagination_scope(
                    scope,
                    split_requests,
                    advertised_pages=advertised_pages or 1,
                )
                yield from split_requests
                return

        items = list(
            response.xpath(
                "//ul[contains(@class,'document-search-result-items')]/li"
            )
        )
        listing_rows = []
        for item in items:
            raw_href = item.xpath(".//a/@href").get()
            absolute_href = response.urljoin(raw_href) if raw_href else None
            document_id = None
            if absolute_href:
                document_id = urlparse(absolute_href).path.rstrip("/").split("/")[-1]
                document_id = document_id or None
            listing_rows.append((item, absolute_href, document_id))

        next_page_url = response.xpath(
            "//ul[contains(@class,'pagination')]//a[contains(normalize-space(.),'შემდეგი')]/@href"
        ).get()
        remembered_pages = tracker.advertised_pages_max
        expected_pages = (
            advertised_pages if advertised_pages is not None else remembered_pages
        )
        if expected_pages is not None and page_number > expected_pages:
            tracker.mark_failure(
                "page_number_exceeds_advertised_pages",
                cursor=page_number,
                detail=f"page={page_number}; advertised_pages={expected_pages}",
            )
        terminal = not next_page_url and (
            expected_pages is None or page_number >= expected_pages
        )
        expected_cursor = response.meta.get("pagination_cursor")
        if (
            expected_cursor is not None
            and str(expected_cursor).isdecimal()
            and int(expected_cursor) != page_number
        ):
            tracker.mark_failure(
                "page_cursor_mismatch",
                cursor=page_number,
                detail=f"request_cursor={expected_cursor}",
            )
        self._observe_page_number(scope, tracker, page_number)
        tracker.observe_page(
            page_number,
            [document_id for _item, _href, document_id in listing_rows],
            advertised_pages=advertised_pages,
            page_number=page_number,
            terminal=terminal,
        )

        for item, absolute_href, document_id in listing_rows:
            try:
                loader = ItemLoader(item=MatsneItem(), selector=item)
                if absolute_href:
                    loader.add_value("document_url", absolute_href)
                loader.add_xpath("status", ".//@class")
                loader.add_value(
                    "additional_status",
                    first_qs_value(parsed_url, "additional_status"),
                )
                loader.add_value(
                    "document_topic",
                    first_qs_value(parsed_url, "label"),
                )
                loaded_item = loader.load_item()
            except Exception as exc:  # noqa: BLE001 - retain remaining listing rows
                tracker.mark_failure(
                    "callback_failure",
                    cursor=page_number,
                    detail=f"listing row processing failed: {exc}",
                )
                self.logger.warning("matsne: listing row processing failed: %s", exc)
                continue

            href = loaded_item.get("document_url")
            if href and document_id:
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

        cap_reached = (
            expected_pages is not None and expected_pages > self.MAX_PAGES
        ) or (next_page_url is not None and page_number >= self.MAX_PAGES)
        if cap_reached:
            tracker.mark_cap(cursor=page_number, configured_cap=self.MAX_PAGES)
            finalize_pagination_scope(self, tracker, url=response.url)
        elif not next_page_url:
            finalize_pagination_scope(self, tracker, url=response.url)
        else:
            request = self.follow_request(
                response,
                next_page_url,
                callback=self.parse,
                meta={
                    "pagination_scope": scope,
                    "pagination_cursor": page_number + 1,
                },
            )
            if request is None:
                tracker.mark_failure(
                    "next_page_not_scheduled",
                    cursor=page_number,
                    detail=str(next_page_url)[:300],
                )
                finalize_pagination_scope(self, tracker, url=response.url)
            else:
                yield request

    def _split_requests(
        self,
        response: HtmlResponse,
        parsed_url: dict,
        *,
        last_page: int | None = None,
    ):
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

        if last_page is None:
            last_page = self._advertised_last_page(response)
        if last_page is None:
            return None
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
        loaded["is_consolidated"] = bool(consolidated_dates)
        if response.meta.get("main_listed"):
            # Provenance stamp: this doc was reached through a type=main listing (or a
            # seed list declared main via -a doc_type=main), so it is a main
            # (consolidated) document by matsne's classifier, switcher or not.
            loaded["is_consolidated"] = True
        elif not consolidated_dates and "known_is_consolidated" in response.meta:
            # A direct mutable-source refresh has no search-listing provenance.  Reuse
            # the last durably successful classifier when the current detail page has
            # no publication switcher; never turn an unknown legacy value into False.
            loaded["is_consolidated"] = bool(
                response.meta["known_is_consolidated"]
            )

        # Seed-mode docs skip the search listing, so they carry no status panel; derive the
        # status from effective/expiry dates as of today so future transitions are not applied
        # early and the ingest status filter remains temporally correct.
        if not loaded.get("status"):
            loaded["status"] = status_from_effective_dates(
                loaded.get("entry_into_force_date"),
                loaded.get("expiry_date"),
            )

        yield loaded
