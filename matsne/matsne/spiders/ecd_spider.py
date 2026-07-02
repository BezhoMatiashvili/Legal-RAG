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

    async def start(self):
        yield json_post(INSTANCES_URL, {}, callback=self.parse_instances, errback=self.request_failed)

    def parse_instances(self, response):
        instances = json.loads(response.text).get("data", [])
        for instance in instances:
            instance_id = instance.get("Id")
            if instance_id is None:
                continue
            yield self.request_page(instance_id, instance.get("Name"), skip=0)

    def request_page(self, instance_id, instance_name, skip):
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
            errback=self.request_failed,
            meta={"instance_id": instance_id, "instance_name": instance_name, "skip": skip},
        )

    def parse_list(self, response):
        instance_id = response.meta["instance_id"]
        instance_name = response.meta["instance_name"]
        skip = response.meta["skip"]

        data = json.loads(response.text).get("data") or {}
        total = data.get("Total", 0)
        items = data.get("Items", [])

        # Yield the next page BEFORE the per-record loop so a single malformed record
        # cannot abort the callback and silently drop the rest of this instance's pages.
        next_skip = skip + self.PAGE_SIZE
        if items and next_skip < total:
            yield self.request_page(instance_id, instance_name, next_skip)

        for record in items:
            iid = record.get("InstanceId")
            did = record.get("DecisionDocumentId")
            if iid is None or did is None:
                self.logger.warning("ecd: skipping record with missing ids: %s", record.get("Id"))
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

    def parse_detail(self, response):
        record = response.meta["record"]
        data = json.loads(response.text).get("data") or {}
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
