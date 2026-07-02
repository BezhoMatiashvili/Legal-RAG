"""tas.ge / docs.tbilisi.gov.ge — Tbilisi Architecture Service public documents.

``tas.ge/?p=searchdocument`` only iframes the real app at
``docs.tbilisi.gov.ge/architect/publicInformation.html``, an ExtJS front end whose
data comes from a DWR RPC method, ``DocumentManager.getDocsForPublicInfo``. Raw HTTP
replay of DWR needs a browser-issued ``scriptSessionId``, so we drive it with a real
(headless) browser via scrapy-playwright.

Rather than scrape the fragile ExtJS grid DOM, we call the app's own data layer:
- set the search form's ``fromDate``/``toDate`` datefields to our window (the form
  serializes JS ``Date`` objects to the format the server accepts — passing date
  *strings* directly is rejected),
- read the fully-built query via ``form.getSearchObject()`` (adds ``applicationId=2``
  and the public ``docStatusIds`` set),
- page through ``getDocsForPublicInfo`` with ``start``/``limit`` against the ``total``
  in ``response.sources[0]``.

Each record is clean JSON plus a ``cachedInfo`` XML blob (nomenclature + status).
Applicant / architect / cadastral code are not in the public list payload, so they
are not captured by this list crawl.
"""

import re

import scrapy
from scrapy.loader import ItemLoader

from ..items import TasItem
from ..utils.dates import date_part
from .base import BaseLegalSpider

GRID_URL = "https://docs.tbilisi.gov.ge/architect/publicInformation.html"
DETAIL_URL = "https://docs.tbilisi.gov.ge/architect/public.html?docId={}"

# One round-trip per page fetches the search object and a page of records.
_FETCH_PAGE_JS = """
async ({y, m, d, y2, m2, d2, start, limit}) => {
    const form = Ext.ComponentQuery.query('nomenclaturesearchform')[0];
    form.getForm().findField('fromDate').setValue(new Date(y, m - 1, d));
    form.getForm().findField('toDate').setValue(new Date(y2, m2 - 1, d2));
    const sObj = form.getSearchObject();
    sObj.start = start;
    sObj.limit = limit;
    const r = await new Promise((resolve, reject) => {
        DocumentManager.getDocsForPublicInfo(sObj, resolve);
        setTimeout(() => reject('dwr-timeout'), 60000);
    });
    return {total: r && r.sources ? r.sources[0] : 0, source: r && r.source ? r.source : []};
}
"""

_READY_JS = (
    "typeof Ext !== 'undefined' && Ext.ComponentQuery && "
    "Ext.ComponentQuery.query('nomenclaturesearchform').length > 0 && "
    "typeof DocumentManager !== 'undefined' && !!DocumentManager.getDocsForPublicInfo"
)


def _xml_tag(xml, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", xml or "", re.S)
    return match.group(1).strip() if match else None


class TasSpider(BaseLegalSpider):
    name = "tas"
    DEDUP_KEY = ("document_id",)
    PAGE_SIZE = 50
    PAGE_DELAY_MS = 1500  # politeness between in-page DWR calls

    custom_settings = {
        # Scoped to this spider: the ExtJS/DWR app only renders under a real browser,
        # so route downloads through scrapy-playwright (the other spiders stay plain HTTP).
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True},
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 60000,
        "CONCURRENT_REQUESTS": 1,
    }

    async def start(self):
        yield scrapy.Request(
            GRID_URL,
            callback=self.parse_docs,
            meta={"playwright": True, "playwright_include_page": True},
        )

    async def parse_docs(self, response):
        page = response.meta["playwright_page"]
        s, e = self.scraping_start_date, self.scraping_end_date
        date_args = {
            "y": s.year, "m": s.month, "d": s.day,
            "y2": e.year, "m2": e.month, "d2": e.day,
        }
        try:
            await page.wait_for_function(_READY_JS, timeout=60000)

            start = 0
            total = None
            while True:
                data = await page.evaluate(_FETCH_PAGE_JS, {**date_args, "start": start, "limit": self.PAGE_SIZE})
                total = data["total"]
                records = data["source"]
                if not records:
                    break

                for record in records:
                    if self.is_seen({"document_id": record.get("documentId")}):
                        self.crawler.stats.inc_value("dedup/skipped")
                        continue
                    yield self.build_item(record)

                start += self.PAGE_SIZE
                if start >= total:
                    break
                await page.wait_for_timeout(self.PAGE_DELAY_MS)

            self.logger.info("tas: fetched up to %s of %s documents", min(start, total or 0), total)
        finally:
            await page.close()

    def build_item(self, record):
        cached = record.get("cachedInfo") or ""
        category = _xml_tag(cached, "categoryName")
        class_name = _xml_tag(cached, "className")
        action = _xml_tag(cached, "actionName")
        stadiums = _xml_tag(cached, "stadiums")
        nomenclature = " / ".join(p for p in (category, class_name, action) if p and p != "-")

        body_lines = []
        if nomenclature:
            body_lines.append(f"**ნომენკლატურა:** {nomenclature}")
        if stadiums and stadiums != "-":
            body_lines.append(f"**სტადია:** {stadiums}")
        if record.get("address"):
            body_lines.append(f"**მისამართი:** {record['address']}")

        doc_id = record.get("documentId")
        loader = ItemLoader(item=TasItem())
        loader.add_value("source_url", DETAIL_URL.format(doc_id))
        loader.add_value("document_id", doc_id)
        loader.add_value("document_no", record.get("documentNo"))
        loader.add_value("address", record.get("address"))
        loader.add_value("registration_date", date_part(record.get("registrationDate")))
        loader.add_value("create_date", record.get("createDateStr"))
        loader.add_value("status", _xml_tag(cached, "documentStatusName"))
        loader.add_value("nomenclature", nomenclature or None)
        loader.add_value("nomenclature_case_id", _xml_tag(cached, "caseId"))
        loader.add_value("body_markdown", "\n\n".join(body_lines))
        return loader.load_item()
