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
    time) and durably records every genuinely-new item as seen. Only items that
    pass through here reach the FEEDS export — so downstream ingest receives only
    new documents.
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

        if identity in spider._seen_keys:
            spider.crawler.stats.inc_value("dedup/dropped")
            raise DropItem(f"already scraped: {identity}")

        body = item.get("body_markdown")
        if body is not None and not str(body).strip():
            # Export the failure artifact for audit/quarantine, but never persist its identity
            # as successfully scraped: a later run must be allowed to fetch a repaired body.
            spider.crawler.stats.inc_value("dedup/not_marked_empty_body")
            return item

        spider.mark_seen(item)
        return item
