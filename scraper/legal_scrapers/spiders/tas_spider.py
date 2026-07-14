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

The list record is clean JSON plus a ``cachedInfo`` XML blob (nomenclature + status).
Each document is then enriched from the public detail payload the ``public.html`` page
loads: ``UserMethods.getUserDocumentLastMotion(docId)``. The grid page already exposes
``UserMethods`` (it loads both ``DocumentManager.js`` and ``UserMethods.js``), so we make
the same in-page DWR call for every document, no per-doc navigation. That descriptor
carries the applicant, cadastral/owner info, request text, attachments, the answers/
decision numbers, and the full decision text (``document.responseText``).
"""

import re
from datetime import datetime, timedelta, timezone

import scrapy
from scrapy.loader import ItemLoader

from ..items import TasItem
from ..utils.dates import date_part
from ..utils.markdown import safe_html_to_markdown
from ..utils.pagination import (
    advertised_page_count,
    finalize_pagination_scope,
    get_pagination_reconciler,
    handle_pagination_request_failure,
    parse_advertised_count,
)
from .base import BaseLegalSpider

BASE_URL = "https://docs.tbilisi.gov.ge"
GRID_URL = "https://docs.tbilisi.gov.ge/architect/publicInformation.html"
DETAIL_URL = "https://docs.tbilisi.gov.ge/architect/public.html?docId={}"

# Tbilisi is UTC+4 year-round (no DST). The DWR payload returns UTC timestamps, but
# the site displays local calendar dates, so we convert before taking the date part.
_TBILISI_TZ = timezone(timedelta(hours=4))

# Motion/document status id -> Georgian label, mirroring ``Architect.util.Doc.statuses``
# from ``architect/util/Doc.js``. Used to render the "გადაწყვეტილება" (decision) line the
# public page shows from ``document.documentStatusId``. Kept in sync with that source.
STATUSES = {
    1: "თანხმობა",
    2: "უარყოფა",
    3: "წაუკითხავი",
    4: "წაკითხული",
    5: "განხილვის პროცესში",
    6: "ხელახლა განხილვაზე",
    7: "შუალედური",
    8: "დრაფტი",
    9: "პასუხის რეჟიმში",
    10: "გადაგზავნილი",
    11: "განუხილველი",
    12: "პირობადადებული",
    13: "პასუხ გაგზავნილი",
    14: "კომისიაზე გატანილი",
    15: "კომისია დასრულებული",
    16: "კომისიაზე თანხმობა",
    17: "კომისიაზე უარყოფა",
    20: "კომისიაზე გატანილი",
    21: "უწყებებში გადაგზავნილი",
    22: "გადახდის მოლოდინში",
    25: "გადახდის მოლოდინში",
    26: "კულტურაში გაგზავნილი",
    27: "ძალადაკარგული",
    28: "დასრულებული",
    29: "ვიზიტის დანიშვნის მოლოდინში",
    30: "ექსკლუზიური მომსახურების მოლოდინში",
    32: "ექსკლუზიური მომსახურების გადახდის მოლოდინში",
    33: "ექსკლუზიური მომსახურების ვიზიტის მოლოდინში",
    34: "კომისიაზე შუალედური",
}

# ``documentStatusId == 8`` means the applicant never submitted the draft; the public
# page refuses to render it, so there is no detail to enrich with.
_DRAFT_STATUS_ID = 8

_FAKEPATH_RE = re.compile(r"^[a-zA-Z]:\\fakepath\\", re.I)

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

# One round-trip per document fetches the full public detail descriptor. We return only
# the subtrees we need and JSON round-trip the result so JS ``Date`` objects become ISO
# strings (deterministic for the Python side).
_FETCH_DETAIL_JS = """
async ({docId, timeout}) => {
    if (typeof UserMethods === 'undefined' || !UserMethods.getUserDocumentLastMotion) {
        return {ok: false, reason: 'no-usermethods'};
    }
    const descriptor = await new Promise((resolve, reject) => {
        UserMethods.getUserDocumentLastMotion(docId, resolve);
        setTimeout(() => reject('detail-timeout'), timeout);
    });
    if (!descriptor || !descriptor.isSuccess) {
        return {ok: false, reason: 'not-success'};
    }
    const w = descriptor.sources ? descriptor.sources[1] : null;
    if (!w) {
        return {ok: false, reason: 'no-wrapper'};
    }
    const motion = descriptor.source || null;
    const out = {
        ok: true,
        canSeeFinalResult: descriptor.sources ? descriptor.sources[0] : null,
        openDate: motion ? motion.whenUserOpenedTheDoc : null,
        document: w.document || null,
        docAuthor: w.docAuthor || null,
        executorEmployee: w.executorEmployee || null,
        mapInfos: w.mapInfos || [],
        docValues: w.docValues || [],
        fieldsetPojos: w.fieldsetPojos || [],
        oldResponseMotions: w.oldResponseMotions || [],
        attachedFiles: w.attachedFiles || [],
        nomenklaturMarkup: w.nomenklaturMarkup || null
    };
    return JSON.parse(JSON.stringify(out));
}
"""

_READY_JS = (
    "typeof Ext !== 'undefined' && Ext.ComponentQuery && "
    "Ext.ComponentQuery.query('nomenclaturesearchform').length > 0 && "
    "typeof DocumentManager !== 'undefined' && !!DocumentManager.getDocsForPublicInfo && "
    "typeof UserMethods !== 'undefined' && !!UserMethods.getUserDocumentLastMotion"
)


def _xml_tag(xml, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", xml or "", re.S)
    return match.group(1).strip() if match else None


def _strip(value):
    return value.strip() if isinstance(value, str) else value


def _clean_value(value):
    """File inputs arrive as ``C:\\fakepath\\foo.pdf``; keep just the base file name."""
    if isinstance(value, str) and _FAKEPATH_RE.search(value):
        return value.split("\\")[-1].strip()
    return _strip(value)


def _local_date(iso_value):
    """ISO UTC timestamp -> Tbilisi (UTC+4) calendar date ``YYYY-MM-DD``.

    Falls back to the plain date prefix when the value is not an ISO timestamp.
    """
    if not isinstance(iso_value, str) or not iso_value:
        return None
    try:
        parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    except ValueError:
        return date_part(iso_value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_TBILISI_TZ).date().isoformat()


def _slash(iso_date):
    """``2012-03-02`` -> ``02/03/2012`` (the d/m/Y form the site displays)."""
    if not iso_date:
        return ""
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return iso_date


def _field_labels(fieldset_pojos):
    """Map ``documentFieldId`` -> field label across all fieldsets."""
    labels = {}
    for fieldset in fieldset_pojos or []:
        for field in fieldset.get("fields") or []:
            field_id = field.get("documentFieldId")
            if field_id is not None:
                labels[field_id] = _strip(field.get("fieldLabel")) or None
    return labels


def _form_fields(doc_values, labels):
    """Labeled applicant-supplied values (free text and attached-file names)."""
    fields = []
    for value in doc_values or []:
        raw = value.get("clobValue")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raw = value.get("stringValue")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raw = value.get("dateValueStr") or None
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        fields.append(
            {
                "label": labels.get(value.get("documentFieldId")),
                "value": _clean_value(raw),
            }
        )
    return fields


def _request_text(doc_values, labels):
    """The applicant's free-text request (მოთხოვნის ტექსტი), else the first clob value."""
    fallback = None
    for value in doc_values or []:
        clob = value.get("clobValue")
        if not clob or not clob.strip():
            continue
        clob = clob.strip()
        label = labels.get(value.get("documentFieldId")) or ""
        if "მოთხოვნის ტექსტი" in label:
            return clob
        if fallback is None:
            fallback = clob
    return fallback


def _parcels(map_infos):
    """Cadastral/owner rows from the National Agency of Public Registry (naprXxx)."""
    parcels = []
    for info in map_infos or []:
        parcels.append(
            {
                "cad_code": _strip(info.get("naprCadCode")),
                "address": _strip(info.get("naprAddress")),
                "area": info.get("naprArea"),
                "purpose": _strip(info.get("naprPurpose")),
                "owner": _strip(info.get("naprOwner")),
                "coowner": _strip(info.get("naprCoowner")),
                "reg_no": _strip(info.get("naprRegNo")),
            }
        )
    return parcels


def _primary_parcel(parcels):
    """The first parcel carrying a cadastral code (the one the page headlines)."""
    for parcel in parcels:
        if parcel.get("cad_code"):
            return parcel
    return parcels[0] if parcels else None


def _responses(old_response_motions):
    """The "პასუხები" answers grid: each response's decision no + dates."""
    responses = []
    for motion in old_response_motions or []:
        responses.append(
            {
                "decision_no": motion.get("previousMotionId"),
                "acquaint_date": _local_date(motion.get("whenUserOpenedTheDoc")),
                "motion_date": _local_date(motion.get("motionDate")),
                "status_id": motion.get("documentStatusId"),
            }
        )
    return responses


def _response_to_markdown(html):
    """Decision text (``document.responseText``) -> Markdown.

    The body is plain paragraphs wrapped in ``<pre>`` (not code), so we swap those for
    blank lines before conversion to avoid emitting fenced code blocks.
    """
    if not html:
        return ""
    cleaned = re.sub(r"</?pre[^>]*>", "\n\n", html, flags=re.I)
    return safe_html_to_markdown(cleaned, base_url=BASE_URL)


def _nomenclature_full(markup):
    """``nomenklaturMarkup`` (HTML list) -> Markdown, without the leading label."""
    if not markup:
        return None
    text = safe_html_to_markdown(markup, base_url=BASE_URL)
    text = re.sub(r"^\s*ნომენკლატურა\s*:\s*", "", text)
    return text.strip() or None


def _full_name(person):
    if not person:
        return None
    name = f"{person.get('firstName') or ''} {person.get('lastName') or ''}".strip()
    return name or None


class TasSpider(BaseLegalSpider):
    name = "tas"
    DEDUP_KEY = ("document_id",)
    PAGE_SIZE = 50
    PAGE_DELAY_MS = 1500     # politeness between in-page list DWR calls
    DETAIL_DELAY_MS = 300    # politeness between per-document detail DWR calls
    DETAIL_TIMEOUT_MS = 60000
    MAX_PAGES = 20_000

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
        # The one navigational request only fetches the ExtJS shell page; all data comes
        # from in-page DWR calls. Serving that shell from the HTTP cache bypasses the
        # Playwright handler (no ``playwright_page`` in meta), so scope caching off here.
        "HTTPCACHE_ENABLED": False,
    }

    async def start(self):
        scope = self.pagination_scope()
        get_pagination_reconciler(self, scope, max_pages=self.MAX_PAGES)
        yield scrapy.Request(
            GRID_URL,
            callback=self.parse_docs,
            errback=self.pagination_request_failed,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "pagination_scope": scope,
                "pagination_cursor": 0,
            },
        )

    def pagination_scope(self):
        return (
            f"date:{self.scraping_start_date.isoformat()}:"
            f"{self.scraping_end_date.isoformat()}"
        )

    def pagination_request_failed(self, failure):
        return handle_pagination_request_failure(self, failure)

    async def parse_docs(self, response):
        page = response.meta["playwright_page"]
        scope = response.meta.get("pagination_scope") or self.pagination_scope()
        tracker = get_pagination_reconciler(
            self,
            scope,
            max_pages=self.MAX_PAGES,
        )
        s, e = self.scraping_start_date, self.scraping_end_date
        date_args = {
            "y": s.year, "m": s.month, "d": s.day,
            "y2": e.year, "m2": e.month, "d2": e.day,
        }
        try:
            await page.wait_for_function(_READY_JS, timeout=60000)

            # Due identities must refresh even when their original registration date is
            # outside the normal discovery window. The public detail RPC is keyed by ID,
            # so it is safe to issue this bounded slice before listing pagination.
            for doc_id in self.iter_refresh_keys():
                detail = await self._fetch_detail(page, doc_id)
                if detail is None:
                    continue
                document = detail.get("document") or {}
                record = {
                    "documentId": doc_id,
                    "documentNo": document.get("documentNo"),
                    "address": document.get("address"),
                    "registrationDate": document.get("registrationDate"),
                    "createDateStr": document.get("createDateStr"),
                }
                self.crawler.stats.inc_value("dedup/direct_refresh_scheduled")
                yield self.build_item(record, detail)
                await page.wait_for_timeout(self.DETAIL_DELAY_MS)

            start = 0
            total = None
            while True:
                page_number = (start // self.PAGE_SIZE) + 1
                if page_number > self.MAX_PAGES:
                    tracker.mark_cap(
                        cursor=start,
                        configured_cap=self.MAX_PAGES,
                    )
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break
                # Retry the list fetch on a transient DWR timeout instead of letting it propagate
                # out of parse_docs and abort the ENTIRE tas crawl (this loop drives all
                # pagination). A read-only page.evaluate is idempotent, so retrying is safe.
                data = None
                last_error = None
                for attempt in range(3):
                    try:
                        data = await page.evaluate(
                            _FETCH_PAGE_JS, {**date_args, "start": start, "limit": self.PAGE_SIZE})
                        break
                    except Exception as exc:  # noqa: BLE001 - transient list-fetch fault; retry
                        last_error = exc
                        self.logger.warning("tas: list fetch failed at start=%s (attempt %d/3): %s",
                                            start, attempt + 1, exc)
                        await page.wait_for_timeout(2000 * (attempt + 1))
                if data is None:
                    self.logger.error("tas: list fetch permanently failed at start=%s — stopping "
                                      "pagination (partial coverage this run)", start)
                    tracker.mark_failure(
                        "exhausted_retries",
                        cursor=start,
                        detail=str(last_error or "empty page result"),
                    )
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break
                if not isinstance(data, dict):
                    failure_kind = (
                        "waf_or_non_json" if isinstance(data, str) else "callback_failure"
                    )
                    tracker.mark_failure(
                        failure_kind,
                        cursor=start,
                        detail="DWR listing result is not an object",
                    )
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break
                raw_total = data.get("total")
                try:
                    total = parse_advertised_count(raw_total)
                except ValueError:
                    tracker.mark_failure(
                        "callback_failure",
                        cursor=start,
                        detail=f"invalid advertised total: {raw_total!r}",
                    )
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break
                records = data.get("source")
                if not isinstance(records, list):
                    tracker.mark_failure(
                        "callback_failure",
                        cursor=start,
                        detail="DWR listing source is not a list",
                    )
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break

                next_start = start + self.PAGE_SIZE
                terminal = not records or next_start >= total
                tracker.observe_page(
                    start,
                    [
                        record.get("documentId")
                        if isinstance(record, dict)
                        else None
                        for record in records
                    ],
                    advertised_total=total,
                    advertised_pages=advertised_page_count(total, self.PAGE_SIZE),
                    page_number=page_number,
                    terminal=terminal,
                )
                if not records:
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break

                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if self.is_seen({"document_id": record.get("documentId")}):
                        self.crawler.stats.inc_value("dedup/skipped")
                        continue
                    detail = await self._fetch_detail(page, record.get("documentId"))
                    if detail is None:
                        continue
                    yield self.build_item(record, detail)
                    await page.wait_for_timeout(self.DETAIL_DELAY_MS)

                start = next_start
                if terminal:
                    finalize_pagination_scope(self, tracker, url=response.url)
                    break
                await page.wait_for_timeout(self.PAGE_DELAY_MS)

            self.logger.info("tas: fetched up to %s of %s documents", min(start, total or 0), total)
        except Exception as exc:  # noqa: BLE001 - callback failures must leave repair evidence
            tracker.mark_failure(
                "callback_failure",
                cursor=locals().get("start", 0),
                detail=str(exc),
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            self.logger.error("tas: pagination callback failed: %s", exc)
        finally:
            await page.close()

    async def _fetch_detail(self, page, doc_id):
        """Fetch a document's public detail descriptor; ``None`` on any failure."""
        if doc_id is None:
            return None
        try:
            detail = await page.evaluate(
                _FETCH_DETAIL_JS, {"docId": doc_id, "timeout": self.DETAIL_TIMEOUT_MS}
            )
        except Exception as exc:  # noqa: BLE001 - enrichment must be best-effort
            self.logger.warning("tas: detail fetch failed for %s: %s", doc_id, exc)
            self.record_quality_failure(
                "detail_fetch_failed",
                DETAIL_URL.format(doc_id),
                detail=str(exc),
                context={"document_id": doc_id},
            )
            return None
        if not detail or not detail.get("ok"):
            self.logger.info(
                "tas: no detail for %s (%s)", doc_id, (detail or {}).get("reason")
            )
            self.record_quality_failure(
                "detail_fetch_failed",
                DETAIL_URL.format(doc_id),
                detail=str((detail or {}).get("reason") or "empty detail response"),
                context={"document_id": doc_id},
            )
            return None
        return detail

    def build_item(self, record, detail=None):
        cached = record.get("cachedInfo") or ""
        category = _xml_tag(cached, "categoryName")
        class_name = _xml_tag(cached, "className")
        action = _xml_tag(cached, "actionName")
        stadiums = _xml_tag(cached, "stadiums")
        nomenclature = " / ".join(p for p in (category, class_name, action) if p and p != "-")

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

        document = (detail or {}).get("document") or {}
        if not detail or document.get("documentStatusId") == _DRAFT_STATUS_ID:
            # List-only item: detail unavailable or the draft was never submitted.
            loader.add_value(
                "content_kind",
                "draft_metadata"
                if document.get("documentStatusId") == _DRAFT_STATUS_ID
                else "list_metadata",
            )
            loader.add_value("content_complete", False)
            loader.add_value("extraction_status", "malformed")
            loader.add_value(
                "body_markdown",
                self._list_body(nomenclature, stadiums, record.get("address")),
            )
            return loader.load_item()

        return self._enrich(loader, record, detail, nomenclature)

    def _enrich(self, loader, record, detail, nomenclature):
        document = detail.get("document") or {}
        author = detail.get("docAuthor") or {}
        executor = detail.get("executorEmployee") or {}
        labels = _field_labels(detail.get("fieldsetPojos"))

        parcels = _parcels(detail.get("mapInfos"))
        primary = _primary_parcel(parcels) or {}
        form_fields = _form_fields(detail.get("docValues"), labels)
        attachments = [
            name
            for name in (_strip(f.get("fileName")) for f in detail.get("attachedFiles") or [])
            if name
        ]
        responses = _responses(detail.get("oldResponseMotions"))

        status_id = document.get("documentStatusId")
        decision = STATUSES.get(status_id)
        decision_no = responses[0]["decision_no"] if responses else None
        deadline_iso = _local_date(document.get("deadLineDate"))
        acquaint_iso = _local_date(detail.get("openDate"))
        applicant_name = _full_name(author)
        request_text = _request_text(detail.get("docValues"), labels)
        response_markdown = _response_to_markdown(document.get("responseText"))
        nomenclature_full = _nomenclature_full(detail.get("nomenklaturMarkup"))

        # A populated detail descriptor is not itself proof that the actual decision
        # text was published. Keep request/metadata-only records retryable and out of
        # the serving corpus until responseText supplies the full decision body.
        has_full_decision = bool(response_markdown and response_markdown.strip())
        loader.add_value(
            "content_kind",
            "decision_full_text" if has_full_decision else "detail_metadata",
        )
        loader.add_value("content_complete", has_full_decision)
        loader.add_value(
            "extraction_status", "full_text" if has_full_decision else "malformed"
        )

        loader.add_value("document_type_id", document.get("documentTypeId"))
        loader.add_value("deadline_date", deadline_iso)
        loader.add_value("acquaint_date", acquaint_iso)
        loader.add_value("decision_status_id", status_id)
        loader.add_value("decision", decision)
        loader.add_value("decision_no", decision_no)
        loader.add_value("can_see_final_result", detail.get("canSeeFinalResult"))
        loader.add_value("amount_to_pay", document.get("amountToPay"))
        loader.add_value("response_markdown", response_markdown or None)

        loader.add_value("request_text", request_text)
        loader.add_value("nomenclature_full", nomenclature_full)

        loader.add_value("applicant_name", applicant_name)
        loader.add_value("applicant_first_name", author.get("firstName"))
        loader.add_value("applicant_last_name", author.get("lastName"))
        loader.add_value("applicant_personal_no", author.get("personalNo"))
        # docAuthor spells this field "birhtDate" upstream (sic).
        loader.add_value("applicant_birth_date", _local_date(author.get("birhtDate")))
        loader.add_value("applicant_address", author.get("address"))
        loader.add_value("applicant_email", author.get("email"))
        loader.add_value("applicant_phone", author.get("phoneNumber"))
        loader.add_value("applicant_passport", author.get("passSerialNumber"))
        loader.add_value("applicant_person_id", author.get("personId"))

        loader.add_value("executor_name", _full_name(executor))
        loader.add_value("executor_personal_no", executor.get("personalNo"))
        loader.add_value("executor_email", executor.get("email"))
        loader.add_value("executor_phone", executor.get("phoneNumber"))
        loader.add_value("executor_id", executor.get("employeeId"))

        loader.add_value("cad_code", primary.get("cad_code"))
        loader.add_value("land_area", primary.get("area"))
        loader.add_value("land_purpose", primary.get("purpose"))
        loader.add_value("owner", primary.get("owner"))
        loader.add_value("coowner", primary.get("coowner"))

        loader.add_value(
            "body_markdown",
            self._detail_body(
                document_no=record.get("documentNo"),
                create_date=record.get("createDateStr"),
                deadline_iso=deadline_iso,
                acquaint_iso=acquaint_iso,
                decision=decision,
                applicant_name=applicant_name,
                nomenclature=nomenclature,
                address=document.get("address") or record.get("address"),
                primary_parcel=primary,
                request_text=request_text,
                response_markdown=response_markdown,
            ),
        )

        item = loader.load_item()
        if form_fields:
            item["form_fields"] = form_fields
        if attachments:
            item["attachments"] = attachments
        if parcels:
            item["parcels"] = parcels
        if responses:
            item["responses"] = responses
        return item

    @staticmethod
    def _list_body(nomenclature, stadiums, address):
        body_lines = []
        if nomenclature:
            body_lines.append(f"**ნომენკლატურა:** {nomenclature}")
        if stadiums and stadiums != "-":
            body_lines.append(f"**სტადია:** {stadiums}")
        if address:
            body_lines.append(f"**მისამართი:** {address}")
        return "\n\n".join(body_lines)

    @staticmethod
    def _detail_body(*, document_no, create_date, deadline_iso, acquaint_iso, decision,
                     applicant_name, nomenclature, address, primary_parcel,
                     request_text, response_markdown):
        info = [
            ("განაცხადის ნომერი", document_no),
            ("განაცხადის თარიღი", create_date),
            ("პასუხის გაცემის ვადა", _slash(deadline_iso)),
            ("გაცნობის თარიღი", _slash(acquaint_iso)),
            ("გადაწყვეტილება", decision),
            ("განმცხადებელი", applicant_name),
        ]
        info_lines = [f"**{label}:** {value}" for label, value in info if value]

        meta_lines = []
        if nomenclature:
            meta_lines.append(f"**ნომენკლატურა:** {nomenclature}")
        if address:
            meta_lines.append(f"**მისამართი:** {address}")
        if primary_parcel and primary_parcel.get("cad_code"):
            meta_lines.append(f"**საკადასტრო კოდი:** {primary_parcel['cad_code']}")

        sections = []
        if info_lines:
            sections.append("\n".join(info_lines))
        if meta_lines:
            sections.append("\n".join(meta_lines))
        if request_text:
            sections.append(f"**მოთხოვნის ტექსტი:**\n{request_text}")
        if response_markdown:
            sections.append(f"**გადაწყვეტილება:**\n{response_markdown}")
        return "\n\n".join(section for section in sections if section.strip())
