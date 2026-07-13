"""supremecourt.ge — Supreme Court of Georgia cases.

The case list is delivered as HTML fragments by the ``/ka/getCases`` AJAX endpoint
(per chamber ``palata`` 0/1/2), with server-side date filtering
(``tarigiDan``/``tarigiMde`` in ``YYYY/MM/DD``) and ``page`` pagination (the visible
pager is JS-driven, so we just increment ``page`` until a page returns no cases).
Each case links to a full HTML page at ``/ka/fullcase/{id}/{palata}``; a DOCX is also
available at ``/ka/download/{id}/{palata}``.

ROBOTS: the site's robots.txt disallows ``/ka/getCases*``. Crawling it anyway is an
explicit, user-approved exception to the project's default ``ROBOTSTXT_OBEY=True``
policy (decision recorded 2026-06-29), so the override is scoped to this spider only.
"""

import re
from urllib.parse import urlencode

from scrapy import Request
from scrapy.loader import ItemLoader

from ..items import SupremecourtItem
from ..utils.dates import iso_to_year_slashed
from ..utils.markdown import safe_html_to_markdown
from .base import BaseLegalSpider

BASE = "https://www.supremecourt.ge"
GETCASES_URL = f"{BASE}/ka/getCases"
CHAMBER_NAMES = {
    "0": "ადმინისტრაციულ საქმეთა პალატა",
    "1": "სამოქალაქო საქმეთა პალატა",
    "2": "სისხლის სამართლის საქმეთა პალატა",
}
CHAMBERS = tuple(int(palata) for palata in CHAMBER_NAMES)
MAX_PAGES = 5000  # safety cap (the JS pager has no real "last page" marker)
_FULLCASE_RE = re.compile(r"/fullcase/(\d+)/(\d+)")

# Labels shown on each case card -> SupremecourtItem field names (colon stripped).
CASE_LABELS = {
    "საქმის ნომერი": "case_number",
    "თარიღი": "date",
    "დავის საგანი": "subject",
    "შედეგი": "result",
    "საჩივრის სახე": "appeal_type",
}


class SupremecourtSpider(BaseLegalSpider):
    name = "supremecourt"
    DEDUP_KEY = ("case_id", "chamber")
    custom_settings = {
        # User-approved exception (2026-06-29): /ka/getCases is the only case-list
        # endpoint and is disallowed by robots.txt. Override scoped to this spider.
        "ROBOTSTXT_OBEY": False,
        # This site rate-limits aggressively, so keep this spider slow and serial.
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": 8,
    }

    async def start(self):
        for palata in CHAMBERS:
            yield self.request_page(palata, 1)

    def request_page(self, palata, page):
        query = {
            "palata": palata,
            "page": page,
            "tarigiDan": iso_to_year_slashed(self.scraping_start_date),
            "tarigiMde": iso_to_year_slashed(self.scraping_end_date),
        }
        return Request(
            f"{GETCASES_URL}?{urlencode(query)}",
            callback=self.parse_list,
            errback=self.request_failed,
            headers={"X-Requested-With": "XMLHttpRequest"},
            meta={"palata": palata, "page": page},
        )

    def parse_list(self, response):
        palata = response.meta["palata"]
        page = response.meta["page"]
        if response.status != 200:
            self.logger.warning("supremecourt: palata %s page %s HTTP %s; stopping", palata, page, response.status)
            return
        cases = response.css("div.cases")

        for case in cases:
            href = case.css("a[href*='/fullcase/']::attr(href)").get()
            match = _FULLCASE_RE.search(href or "")
            if not match:
                continue
            # Per-case chamber from the fullcase URL. Deliberately NOT named `palata`: that name
            # holds the PAGE's chamber (response.meta) which the pagination below reuses for the
            # next-page request — shadowing it made pagination follow the LAST case's chamber and
            # silently switch chambers / truncate coverage.
            case_id, case_palata = match.group(1), match.group(2)

            fields = {"case_id": case_id, "chamber": CHAMBER_NAMES.get(case_palata, case_palata)}
            for child in case.xpath("./*"):
                label = "".join(child.xpath("./span//text()").getall()).strip().rstrip(":").strip()
                field = CASE_LABELS.get(label)
                if field:
                    value = "".join(child.xpath("./text()").getall()).strip()
                    if value:
                        fields[field] = value

            if self.is_seen(fields):
                self.crawler.stats.inc_value("dedup/skipped")
                continue

            yield response.follow(
                href,
                callback=self.parse_detail,
                errback=self.request_failed,
                meta={"fields": fields, "palata": case_palata},
            )

        if cases and page < MAX_PAGES:
            yield self.request_page(palata, page + 1)
        elif cases:
            self.logger.warning("supremecourt: palata %s hit MAX_PAGES=%s cap; stopping", palata, MAX_PAGES)

    def parse_detail(self, response):
        fields = response.meta["fields"]
        case_id = fields["case_id"]
        palata = response.meta["palata"]

        body_html = response.css("div.case-single#modalBody").get()
        body_markdown = safe_html_to_markdown(body_html, base_url=BASE, source_url=response.url)

        loader = ItemLoader(item=SupremecourtItem())
        loader.add_value("source_url", response.url)
        for key, value in fields.items():
            loader.add_value(key, value)
        loader.add_value("docx_url", f"{BASE}/ka/download/{case_id}/{palata}")
        loader.add_value("body_markdown", body_markdown)
        yield loader.load_item()
