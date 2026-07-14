"""Offline tests for mutable-source refresh and feed-durable dedup commits."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import stat
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraper"))

from scrapy.settings import Settings  # noqa: E402

from legal_scrapers.extensions import (  # noqa: E402
    DurableDedupCommitExtension,
    RotatingSpiderLogExtension,
)
from legal_scrapers.pipelines import DedupPipeline  # noqa: E402
from legal_scrapers.spiders.base import BaseLegalSpider  # noqa: E402


NOW = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)


class _GenericSpider(BaseLegalSpider):
    name = "mutable"
    DEDUP_KEY = ("document_id",)


class _MatsneSpider(_GenericSpider):
    name = "matsne"


class _TasSpider(_GenericSpider):
    name = "tas"


class _Stats:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get_stats(self):
        return dict(self.values)

    def inc_value(self, key, count=1, **_kwargs):
        self.values[key] = self.values.get(key, 0) + count


def _make_spider(
    cls,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings_values=None,
):
    monkeypatch.setattr(cls, "ARTIFACTS_ROOT", root)
    spider = cls(start_date="2026-06-29", end_date="2026-07-13")
    spider.run_id = "run-test"
    spider._dedup_now = lambda: NOW
    settings = Settings({"DEDUP_ENABLED": True, **(settings_values or {})})
    spider.open_dedup_store(settings)
    crawler = types.SimpleNamespace(settings=settings, stats=_Stats(), spider=spider)
    spider.crawler = crawler
    return spider


def _legacy_store(root: Path, spider_name: str, rows) -> Path:
    directory = root / spider_name
    directory.mkdir(parents=True)
    path = directory / "seen.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE seen (key TEXT PRIMARY KEY, run_id TEXT, ts TEXT)"
    )
    connection.executemany("INSERT INTO seen VALUES (?, ?, ?)", rows)
    connection.commit()
    connection.close()
    return path


def _row(spider, key):
    return spider._dedup_conn.execute(
        "SELECT key, last_success, content_hash, refresh_deadline, outcome, "
        "content_kind, content_complete, source_binary_url FROM seen WHERE key = ?",
        (key,),
    ).fetchone()


def test_legacy_rows_migrate_due_and_rotate_oldest_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_store(
        tmp_path,
        _GenericSpider.name,
        [
            ("oldest", "legacy", "2025-01-01T00:00:00+00:00"),
            ("next", "legacy", "2025-01-02T00:00:00+00:00"),
        ],
    )
    spider = _make_spider(
        _GenericSpider,
        tmp_path,
        monkeypatch,
        settings_values={"DEDUP_REFRESH_LIMIT": 1},
    )

    assert spider._refresh_keys == {"oldest"}
    assert not spider.is_seen({"document_id": "oldest"})
    assert spider.is_seen({"document_id": "next"})
    migrated = _row(spider, "oldest")
    assert migrated[4:] == ("legacy_success", "legacy_unknown", 0, "")

    assert spider.stage_seen({"document_id": "oldest", "body_markdown": "updated"})
    spider.commit_staged_seen()
    spider._dedup_conn.close()

    reopened = _make_spider(
        _GenericSpider,
        tmp_path,
        monkeypatch,
        settings_values={"DEDUP_REFRESH_LIMIT": 1},
    )
    assert reopened._refresh_keys == {"next"}
    assert reopened.is_seen({"document_id": "oldest"})
    assert not reopened.is_seen({"document_id": "next"})


def test_refresh_cap_is_hard_2000_and_ties_rotate_by_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        (f"doc-{index:04d}", "legacy", "2025-01-01T00:00:00+00:00")
        for index in range(2_002)
    ]
    path = _legacy_store(tmp_path, _GenericSpider.name, rows)
    path.chmod(0o640)
    path.parent.chmod(0o750)

    spider = _make_spider(
        _GenericSpider,
        tmp_path,
        monkeypatch,
        settings_values={"DEDUP_REFRESH_LIMIT": 9_999},
    )

    assert len(spider._refresh_keys) == 2_000
    assert "doc-0000" in spider._refresh_keys
    assert "doc-1999" in spider._refresh_keys
    assert spider._seen_keys == set()
    assert spider.is_seen({"document_id": "doc-2000"})
    assert spider.is_seen({"document_id": "doc-2001"})
    # Existing user-owned artifact modes are preserved rather than normalized.
    assert path.stat().st_mode & 0o777 == 0o640
    assert path.parent.stat().st_mode & 0o777 == 0o750


@pytest.mark.parametrize(
    ("spider_cls", "item", "days"),
    (
        (_MatsneSpider, {"document_id": "m", "body_markdown": "law"}, 30),
        (
            _TasSpider,
            {
                "document_id": "t",
                "body_markdown": "decision",
                "decision_status_id": 4,
            },
            7,
        ),
        (
            _MatsneSpider,
            {"document_id": "p", "body_markdown": "proposal", "status": "draft"},
            1,
        ),
    ),
)
def test_refresh_cadence_by_source_and_pending_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spider_cls,
    item,
    days,
) -> None:
    spider = _make_spider(spider_cls, tmp_path / spider_cls.name, monkeypatch)
    assert spider.stage_seen(item)
    spider.commit_staged_seen()

    deadline = datetime.fromisoformat(_row(spider, item["document_id"])[3])
    assert deadline == NOW + timedelta(days=days)


def test_refresh_context_persists_nullable_consolidation_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_MatsneSpider, tmp_path / "matsne", monkeypatch)
    assert spider.stage_seen(
        {
            "document_id": "main-law",
            "body_markdown": "current consolidated law",
            "is_consolidated": True,
        }
    )
    spider.commit_staged_seen()

    assert spider.dedup_refresh_context("main-law") == {
        "is_consolidated": True,
    }


def test_legacy_refresh_context_remains_unknown_after_schema_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_store(
        tmp_path,
        _MatsneSpider.name,
        [("legacy-law", "legacy", "2025-01-01T00:00:00+00:00")],
    )
    spider = _make_spider(_MatsneSpider, tmp_path, monkeypatch)

    assert spider.dedup_refresh_context("legacy-law") == {
        "is_consolidated": None,
    }


def _feed_crawler(spider, feeds, stats):
    settings = Settings({"FEEDS": {str(path): {"format": "jsonlines"} for path in feeds}})
    crawler = types.SimpleNamespace(settings=settings, stats=_Stats(stats), spider=spider)
    spider.crawler = crawler
    return crawler


def test_success_commits_only_after_all_feeds_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    item = {
        "document_id": "complete",
        "body_markdown": "full ruling text",
        "content_kind": "ruling_full_text",
        "content_complete": True,
        "source_binary_url": "https://source.invalid/ruling.pdf",
    }
    assert DedupPipeline().process_item(item, spider) is item
    assert _row(spider, "complete") is None

    feeds = [tmp_path / "run.jsonl", tmp_path / "latest.jsonl"]
    for path in feeds:
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o600)
    crawler = _feed_crawler(
        spider,
        feeds,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    DurableDedupCommitExtension(crawler).feed_exporter_closed()

    stored = _row(spider, "complete")
    assert stored[1] == NOW.isoformat()
    assert stored[2] == hashlib.sha256(b"full ruling text").hexdigest()
    assert datetime.fromisoformat(stored[3]) == NOW + timedelta(days=30)
    assert stored[4:] == (
        "success",
        "ruling_full_text",
        1,
        "https://source.invalid/ruling.pdf",
    )
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in feeds)
    assert crawler.stats.values["dedup/committed_after_feeds"] == 1


def test_insecure_existing_feed_is_not_chmodded_or_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    feed = tmp_path / "existing.jsonl"
    feed.write_text("{}\n", encoding="utf-8")
    feed.chmod(0o644)
    crawler = _feed_crawler(
        spider,
        [feed],
        {"feedexport/success_count/FileFeedStorage": 1},
    )

    DurableDedupCommitExtension(crawler).feed_exporter_closed()

    assert feed.stat().st_mode & 0o777 == 0o644
    assert _row(spider, "retry") is None
    assert crawler.stats.values["dedup/feed_commit_refused"] == 1


def test_operational_spider_log_uses_bounded_private_rotation(tmp_path: Path) -> None:
    connected = []
    crawler = types.SimpleNamespace(
        signals=types.SimpleNamespace(
            connect=lambda callback, signal: connected.append((callback, signal))
        )
    )
    spider = types.SimpleNamespace(name="matsne", log_path=tmp_path / "spider.log")
    extension = RotatingSpiderLogExtension.from_crawler(crawler)

    extension.spider_opened(spider)
    handler = extension.handler
    assert handler.maxBytes == 50 * 1024 * 1024
    assert handler.backupCount == 10
    logging.getLogger("crawl-test").warning("bounded", extra={"spider": spider})
    handler.flush()
    assert stat.S_IMODE(spider.log_path.stat().st_mode) == 0o600
    assert "bounded" in spider.log_path.read_text(encoding="utf-8")

    extension.spider_closed(spider, "finished")
    assert extension.handler is None
    assert len(connected) == 2


def test_feed_failure_discards_staged_success_without_db_ghost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    item = {"document_id": "retry", "body_markdown": "complete"}
    DedupPipeline().process_item(item, spider)
    feeds = [tmp_path / "run.jsonl", tmp_path / "latest.jsonl"]
    for path in feeds:
        path.write_text("{}\n", encoding="utf-8")
    crawler = _feed_crawler(
        spider,
        feeds,
        {
            "feedexport/success_count/FileFeedStorage": 1,
            "feedexport/failed_count/FileFeedStorage": 1,
        },
    )

    DurableDedupCommitExtension(crawler).feed_exporter_closed()

    assert _row(spider, "retry") is None
    assert "retry" not in spider._seen_keys
    assert crawler.stats.values["dedup/feed_commit_refused"] == 1
    assert crawler.stats.values["quality/failures"] == 1
    assert crawler.stats.values["quality/durable_feed_commit_refused"] == 1


def test_fsync_failure_refuses_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    feeds = [tmp_path / "run.jsonl", tmp_path / "latest.jsonl"]
    for path in feeds:
        path.write_text("{}\n", encoding="utf-8")
    crawler = _feed_crawler(
        spider,
        feeds,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    extension = DurableDedupCommitExtension(crawler)
    monkeypatch.setattr(
        extension, "_fsync_local_feed", lambda _path: (_ for _ in ()).throw(OSError("disk"))
    )

    extension.feed_exporter_closed()

    assert _row(spider, "retry") is None
    assert "retry" not in spider._seen_keys
    assert crawler.stats.values["quality/failures"] == 1
    assert crawler.stats.values["quality/durable_feed_commit_refused"] == 1


def test_summary_only_outcome_is_durable_but_remains_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    item = {
        "document_id": "summary",
        "body_markdown": "article teaser",
        "content_kind": "article_summary",
        "content_complete": False,
        "source_binary_url": "https://source.invalid/ruling.pdf",
    }
    DedupPipeline().process_item(item, spider)
    assert "summary" not in spider._seen_keys

    feeds = [tmp_path / "run.jsonl", tmp_path / "latest.jsonl"]
    for path in feeds:
        path.write_text("{}\n", encoding="utf-8")
    crawler = _feed_crawler(
        spider,
        feeds,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    DurableDedupCommitExtension(crawler).feed_exporter_closed()

    stored = _row(spider, "summary")
    assert stored[1] is None
    assert stored[4:] == (
        "incomplete",
        "article_summary",
        0,
        "https://source.invalid/ruling.pdf",
    )
    spider._dedup_conn.close()
    reopened = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    assert not reopened.is_seen({"document_id": "summary"})
    assert reopened._refresh_keys == {"summary"}


def test_failed_refresh_retains_classifier_and_is_rescheduled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_MatsneSpider, tmp_path / "matsne", monkeypatch)
    complete = {
        "document_id": "old-main-law",
        "body_markdown": "previous complete law",
        "is_consolidated": True,
    }
    assert spider.stage_seen(complete)
    spider.commit_staged_seen()

    incomplete = {
        "document_id": "old-main-law",
        "body_markdown": "",
        "content_complete": False,
    }
    assert spider.stage_seen(incomplete)
    spider.commit_staged_seen()
    spider._dedup_conn.close()

    reopened = _make_spider(_MatsneSpider, tmp_path / "matsne", monkeypatch)
    assert reopened._refresh_keys == {"old-main-law"}
    assert reopened.dedup_refresh_context("old-main-law") == {
        "is_consolidated": True,
    }


def test_tas_list_only_fallback_never_becomes_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_TasSpider, tmp_path / "tas-store", monkeypatch)
    item = {
        "document_id": "draft",
        "body_markdown": "list metadata only",
        "status": "draft",
    }

    assert spider.stage_seen(item)
    assert spider._staged_dedup_records["draft"]["outcome"] == "incomplete"
    assert "draft" not in spider._seen_keys


def test_supreme_extension_path_remains_isolated() -> None:
    calls = []
    spider = types.SimpleNamespace(
        name="supremecourt",
        _staged_dedup_records={"x": {}},
        commit_staged_seen=lambda: calls.append("commit"),
        discard_staged_seen=lambda: calls.append("discard"),
    )
    crawler = types.SimpleNamespace(
        spider=spider,
        settings=Settings({"FEEDS": {}}),
        stats=_Stats(),
    )

    DurableDedupCommitExtension(crawler).feed_exporter_closed()

    assert calls == []
