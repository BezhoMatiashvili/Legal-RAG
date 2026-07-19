# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.exceptions import DropItem


class MatsnePipeline:
    def process_item(self, item, spider):
        return item


class DedupPipeline:
    """Cross-run + within-run deduplication safety net.

    Spiders already skip the detail request for documents whose identity is in
    the persistent seen-store (see ``BaseLegalSpider.is_seen``). This pipeline is
    the backstop: it drops any item whose identity is already seen (catching
    within-run duplicates and sources whose full key is only known at detail
    time), then stages its outcome. The durable-feed extension commits staged
    records only after every configured feed has closed successfully and fsynced.
    """

    def process_item(self, item, spider):
        key = getattr(spider, "dedup_key", None)
        if key is None:
            # Spider predates dedup support; pass everything through.
            return item
        if not getattr(spider, "dedup_enabled", False) and not getattr(
            spider, "evidence_crawl", False
        ):
            # Ordinary ``--no-dedup`` retains its historical pass-through behavior.
            # Evidence mode is the sole dedup-disabled mode that still reconciles and
            # suppresses duplicate identities within the current immutable run.
            return item

        identity = spider.dedup_key(item)
        if identity is None:
            if getattr(spider, "evidence_crawl", False):
                spider.record_evidence_identity_event("missing", None)
                spider.crawler.stats.inc_value("within_run/missing_identity_count")
                spider.record_quality_failure(
                    "missing_item_identity",
                    item.get("source_url") or item.get("document_url") or "",
                    detail="evidence-crawl item has no complete DEDUP_KEY",
                )
            return item

        spider.crawler.stats.inc_value("within_run/observed_identity_count")
        is_seen = getattr(spider, "is_seen", None)
        already_seen = (
            bool(is_seen(item))
            if callable(is_seen)
            else identity in getattr(spider, "_seen_keys", ())
        )
        if already_seen:
            if getattr(spider, "evidence_crawl", False) or identity in getattr(
                spider, "_within_run_keys", ()
            ):
                spider.record_evidence_identity_event("duplicate", identity)
                spider.crawler.stats.inc_value("within_run/duplicate_identity_count")
            spider.crawler.stats.inc_value("dedup/dropped")
            raise DropItem(f"already scraped: {identity}")

        if not spider.stage_seen(item):
            spider.record_evidence_identity_event("duplicate", identity)
            spider.crawler.stats.inc_value("within_run/duplicate_identity_count")
            spider.crawler.stats.inc_value("dedup/dropped")
            raise DropItem(f"already staged: {identity}")
        spider.record_evidence_identity_event("unique", identity)
        spider.crawler.stats.inc_value("within_run/unique_identity_count")
        staged = spider._staged_dedup_records.get(identity)
        if staged is not None and staged["outcome"] != "success":
            # Export the failure artifact for audit/quarantine, but leave it retryable.
            spider.crawler.stats.inc_value("dedup/staged_incomplete")
        return item


class SupremecourtDurablePipeline:
    """Persist a complete Supreme Court item before recording its identity as seen.

    Scrapy feed exporters run after item pipelines.  The generic pipeline therefore used to
    commit ``seen.sqlite`` first, leaving a ghost key when a timed crawl stopped between the
    pipeline and feed write.  The newest-first spider disables FEEDS and owns an fsync-backed
    journal; this pipeline is scoped to that spider through ``custom_settings`` and enforces
    the required order explicitly::

        validate full body -> append+fsync item -> mark seen -> settle its date window

    Other spiders continue to use :class:`DedupPipeline` unchanged.
    """

    def process_item(self, item, spider):
        identity = spider.dedup_key(item)
        if identity is None:
            spider.item_failed(
                None, "incomplete_identity", "item has no durable identity"
            )
            raise DropItem("Supreme Court item has incomplete identity")

        spider.crawler.stats.inc_value("within_run/observed_identity_count")
        is_seen = getattr(spider, "is_seen", None)
        already_seen = (
            bool(is_seen(item))
            if callable(is_seen)
            else identity in getattr(spider, "_seen_keys", ())
        )
        if already_seen:
            spider.crawler.stats.inc_value("within_run/duplicate_identity_count")
            spider.crawler.stats.inc_value("dedup/dropped")
            spider.item_failed(
                identity, "duplicate_after_detail", "identity became seen"
            )
            raise DropItem(f"already scraped: {identity}")

        body = item.get("body_markdown")
        if body is None or not str(body).strip():
            spider.crawler.stats.inc_value("dedup/not_marked_empty_body")
            spider.item_failed(
                identity, "empty_body", "nonempty #modalBody is required"
            )
            raise DropItem(f"empty Supreme Court body: {identity}")

        # persist_item flushes + fsyncs its append-only journal.  Never reorder these calls.
        spider.persist_item(dict(item), identity)
        try:
            spider.mark_seen(item)
        except (
            Exception
        ) as exc:  # pragma: no cover - sqlite/storage failure is operational
            # The full item is already durable and reconciliation will restore the missing key
            # next run.  Record the issue without throwing away this validated document.
            spider.record_quality_failure(
                "seen_write_failed",
                item.get("source_url") or "",
                detail=str(exc),
                context={"identity": identity},
            )
        spider.item_persisted(identity)
        spider.crawler.stats.inc_value("within_run/unique_identity_count")
        return item
