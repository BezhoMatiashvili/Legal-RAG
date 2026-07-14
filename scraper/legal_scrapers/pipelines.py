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

        identity = spider.dedup_key(item)
        if identity is None:
            # Dedup disabled or incomplete identity — never drop.
            return item

        if spider.is_seen(item):
            spider.crawler.stats.inc_value("dedup/dropped")
            raise DropItem(f"already scraped: {identity}")

        if not spider.stage_seen(item):
            spider.crawler.stats.inc_value("dedup/dropped")
            raise DropItem(f"already staged: {identity}")
        staged = spider._staged_dedup_records[identity]
        if staged["outcome"] != "success":
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

        if identity in spider._seen_keys:
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
        return item
