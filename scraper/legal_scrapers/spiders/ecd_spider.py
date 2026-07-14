"""ecd.court.ge — Electronic Court Decisions of the Georgian common courts.

This is a pure JSON API (ASP.NET MVC + Kendo UI front end); there is no HTML to
scrape. Three POST endpoints drive everything:

- ``/Classifiers/Instances``       -> the three court instances (1/2/3)
- ``/Decision/DecisionDocuments``  -> a page of decisions for an instance + date range
- ``/Decision/DecisionDocumentText`` -> the full plain-text body of one decision

Date filtering (``DecisionDateFrom``/``DecisionDateTo`` in ISO ``YYYY-MM-DD``) and
offset pagination (``Skip``/``Take`` with a ``Total`` count) are native, so this maps
cleanly onto the project's ``start_date``/``end_date`` model.
"""

import json

from scrapy.loader import ItemLoader

from ..items import EcdItem
from ..utils.dates import dotnet_date_to_iso
from ..utils.json_api import json_post
from ..utils.pagination import (
    advertised_page_count,
    finalize_pagination_scope,
    get_pagination_reconciler,
    handle_pagination_request_failure,
    parse_advertised_count,
)
from ..utils.text import plain_text_to_markdown
from .base import BaseLegalSpider

BASE = "https://ecd.court.ge"
INSTANCES_URL = f"{BASE}/Classifiers/Instances"
DOCUMENTS_URL = f"{BASE}/Decision/DecisionDocuments"
TEXT_URL = f"{BASE}/Decision/DecisionDocumentText"


class EcdSpider(BaseLegalSpider):
    name = "ecd"
    DEDUP_KEY = ("decision_document_id",)
    PAGE_SIZE = 50
    MAX_PAGES = 20_000

    async def start(self):
        yield json_post(
            INSTANCES_URL,
            {},
            callback=self.parse_instances,
            errback=self.pagination_request_failed,
            meta={
                "pagination_scope": "instances",
                "pagination_cursor": 0,
            },
        )

    def pagination_request_failed(self, failure):
        return handle_pagination_request_failure(self, failure)

    def _json_data(self, response):
        """Parse a JSON API body; return None (logged) on a non-JSON 200 — a WAF/maintenance
        HTML page — so one bad response skips its callback instead of raising JSONDecodeError
        and killing the crawl / silently dropping the rest of an instance's pages."""
        try:
            return json.loads(response.text)
        except (ValueError, json.JSONDecodeError):
            self.logger.warning("ecd: non-JSON response from %s (%d bytes) — skipping",
                                response.url, len(response.text or ""))
            self.record_quality_failure(
                "non_json_response",
                response.url,
                detail=f"HTTP {response.status}; {len(response.text or '')} bytes",
            )
            return None

    def parse_instances(self, response):
        tracker = get_pagination_reconciler(self, "instances", max_pages=1)
        payload = self._json_data(response)
        if payload is None:
            tracker.mark_failure(
                "waf_or_non_json",
                cursor=0,
                detail=f"HTTP {response.status}",
            )
            finalize_pagination_scope(
                self,
                tracker,
                url=response.url,
                quality_failure_recorded=True,
            )
            return
        instances = payload.get("data", []) if isinstance(payload, dict) else None
        if not isinstance(instances, list):
            tracker.mark_failure(
                "callback_failure",
                cursor=0,
                detail="instances payload is not a list",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        tracker.observe_page(
            0,
            [instance.get("Id") if isinstance(instance, dict) else None for instance in instances],
            advertised_total=len(instances),
            advertised_pages=1,
            page_number=1,
            terminal=True,
        )
        finalize_pagination_scope(self, tracker, url=response.url)
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_id = instance.get("Id")
            if instance_id is None:
                continue
            yield self.request_page(instance_id, instance.get("Name"), skip=0)

    def request_page(self, instance_id, instance_name, skip):
        scope = f"instance:{instance_id}"
        get_pagination_reconciler(self, scope, max_pages=self.MAX_PAGES)
        payload = {
            "InstanceId": str(instance_id),
            "DecisionDateFrom": self.scraping_start_date.isoformat(),
            "DecisionDateTo": self.scraping_end_date.isoformat(),
            "Skip": skip,
            "Take": self.PAGE_SIZE,
        }
        return json_post(
            DOCUMENTS_URL,
            payload,
            callback=self.parse_list,
            errback=self.pagination_request_failed,
            meta={
                "instance_id": instance_id,
                "instance_name": instance_name,
                "skip": skip,
                "pagination_scope": scope,
                "pagination_cursor": skip,
                "dont_cache": True,
            },
        )

    def parse_list(self, response):
        instance_id = response.meta["instance_id"]
        instance_name = response.meta["instance_name"]
        skip = response.meta["skip"]
        scope = response.meta.get("pagination_scope") or f"instance:{instance_id}"
        tracker = get_pagination_reconciler(
            self,
            scope,
            max_pages=self.MAX_PAGES,
        )

        payload = self._json_data(response)
        if payload is None:
            tracker.mark_failure(
                "waf_or_non_json",
                cursor=skip,
                detail=f"HTTP {response.status}",
            )
            finalize_pagination_scope(
                self,
                tracker,
                url=response.url,
                quality_failure_recorded=True,
            )
            return
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            tracker.mark_failure(
                "callback_failure",
                cursor=skip,
                detail="listing payload data is not an object",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        raw_total = data.get("Total", 0)
        try:
            total = parse_advertised_count(raw_total)
        except ValueError:
            tracker.mark_failure(
                "callback_failure",
                cursor=skip,
                detail=f"invalid advertised total: {raw_total!r}",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return
        items = data.get("Items", [])
        if not isinstance(items, list):
            tracker.mark_failure(
                "callback_failure",
                cursor=skip,
                detail="listing Items is not a list",
            )
            finalize_pagination_scope(self, tracker, url=response.url)
            return

        next_skip = skip + self.PAGE_SIZE
        terminal = not items or next_skip >= total
        page_number = (skip // self.PAGE_SIZE) + 1
        tracker.observe_page(
            skip,
            [record.get("DecisionDocumentId") if isinstance(record, dict) else None for record in items],
            advertised_total=total,
            advertised_pages=advertised_page_count(total, self.PAGE_SIZE),
            page_number=page_number,
            terminal=terminal,
        )
        for record in items:
            if not isinstance(record, dict):
                continue
            try:
                iid = record.get("InstanceId")
                did = record.get("DecisionDocumentId")
                if iid is None or did is None:
                    self.logger.warning(
                        "ecd: skipping record with missing ids: %s",
                        record.get("Id"),
                    )
                    continue
                if self.is_seen({"decision_document_id": did}):
                    self.crawler.stats.inc_value("dedup/skipped")
                    continue
                yield json_post(
                    TEXT_URL,
                    {"InstanceId": iid, "DecisionDocumentId": did},
                    callback=self.parse_detail,
                    errback=self.request_failed,
                    meta={"record": record, "instance_name": instance_name},
                )
            except Exception as exc:  # noqa: BLE001 - retain the remaining listing
                tracker.mark_failure(
                    "callback_failure",
                    cursor=skip,
                    detail=f"record processing failed: {exc}",
                )
                self.logger.warning("ecd: record processing failed: %s", exc)
                continue

        cap_reached = not terminal and page_number >= self.MAX_PAGES
        if cap_reached:
            tracker.mark_cap(cursor=skip, configured_cap=self.MAX_PAGES)
            finalize_pagination_scope(self, tracker, url=response.url)
        elif terminal:
            finalize_pagination_scope(self, tracker, url=response.url)
        elif items:
            yield self.request_page(instance_id, instance_name, next_skip)

    def parse_detail(self, response):
        record = response.meta["record"]
        payload = self._json_data(response)
        if payload is None:
            return
        data = payload.get("data") or {}
        decision_document_id = record.get("DecisionDocumentId")

        loader = ItemLoader(item=EcdItem())
        loader.add_value("source_url", f"{BASE}/Decision#!?DecisionDocumentId={decision_document_id}")
        loader.add_value("document_id", record.get("Id"))
        loader.add_value("decision_document_id", decision_document_id)
        loader.add_value("instance_id", record.get("InstanceId"))
        loader.add_value("instance_name", response.meta.get("instance_name") or record.get("InstanceName"))
        loader.add_value("case_id", record.get("CaseId"))
        loader.add_value("case_no", record.get("CaseNo"))
        loader.add_value("court_name", record.get("CourtName"))
        loader.add_value("case_category_name", record.get("CaseCategoryName"))
        loader.add_value("decision_type_name", record.get("TypeName"))
        loader.add_value("litigation_type_name", record.get("LitigationTypeName"))
        loader.add_value("decision_date", dotnet_date_to_iso(record.get("DecisionDate")))
        loader.add_value("barcode", record.get("Barcode"))
        loader.add_value("body_markdown", plain_text_to_markdown(data.get("RawData")))
        yield loader.load_item()
