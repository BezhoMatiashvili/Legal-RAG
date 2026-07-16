"""Offline tests for mutable-source refresh and feed-durable dedup commits."""

from __future__ import annotations

import hashlib
import json
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

from legal_scrapers import extensions as scraper_extensions  # noqa: E402
from legal_scrapers.completion import CompletionAlreadyFinalized  # noqa: E402
from legal_scrapers.extensions import (  # noqa: E402
    CompletionAttestationExtension,
    RotatingSpiderLogExtension,
)
from legal_scrapers.completion import (  # noqa: E402
    build_terminal_record,
    build_startup_record,
    evaluate_crawl_quality,
    failed_feed_durability,
    failed_source_validation,
    publish_startup_metadata,
    publish_terminal_record,
)
from legal_scrapers.pipelines import DedupPipeline  # noqa: E402
from legal_scrapers.spiders import base as base_spider_module  # noqa: E402
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


def _persisted_row(spider, key):
    connection = sqlite3.connect(spider.dedup_db_path)
    try:
        return connection.execute(
            "SELECT key, last_success, content_hash, refresh_deadline, outcome, "
            "content_kind, content_complete, source_binary_url FROM seen WHERE key = ?",
            (key,),
        ).fetchone()
    finally:
        connection.close()


def _full_persisted_row(spider, key):
    connection = sqlite3.connect(spider.dedup_db_path)
    try:
        return connection.execute(
            "SELECT key, run_id, ts, last_success, content_hash, refresh_deadline, "
            "outcome, content_kind, content_complete, source_binary_url, "
            "is_consolidated FROM seen WHERE key = ?",
            (key,),
        ).fetchone()
    finally:
        connection.close()


def _pending_rows(spider, run_id=None):
    connection = sqlite3.connect(spider.dedup_db_path)
    try:
        return connection.execute(
            "SELECT pending_run_id, key, run_id, outcome, content_hash "
            "FROM pending_seen WHERE pending_run_id = ? ORDER BY key",
            (run_id or spider.run_id,),
        ).fetchall()
    finally:
        connection.close()


def _feed_crawler(spider, root, stats, *, payload=b"{}\n"):
    source_root = root / "crawl-artifacts" / spider.name
    run_dir = source_root / "runs" / spider.run_id
    latest_dir = source_root / "latest"
    for directory in (source_root, source_root / "runs", run_dir, latest_dir):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
    spider.started_at = NOW
    spider.source_root = source_root
    spider.run_dir = run_dir
    spider.latest_dir = latest_dir
    spider.items_path = run_dir / "items.jsonl"
    spider.latest_items_path = latest_dir / "items.jsonl"
    spider.log_path = run_dir / "spider.log"
    spider.run_metadata_path = run_dir / "run.json"
    spider.latest_metadata_path = latest_dir / "run.json"
    for path in (spider.items_path, spider.latest_items_path):
        path.write_bytes(payload)
        path.chmod(0o600)
    startup = build_startup_record(spider)
    publish_startup_metadata(
        spider.run_metadata_path,
        spider.latest_metadata_path,
        startup,
    )
    settings = Settings(
        {
            "FEEDS": {
                str(spider.items_path): {"format": "jsonlines"},
                str(spider.latest_items_path): {"format": "jsonlines"},
            }
        }
    )
    crawler = types.SimpleNamespace(settings=settings, stats=_Stats(stats), spider=spider)
    spider.crawler = crawler
    return crawler


def _finish_attestation(crawler, *, order="feed-first"):
    extension = CompletionAttestationExtension(crawler)
    if order == "feed-first":
        extension.feed_exporter_closed()
        extension.spider_closed(crawler.spider, "finished")
    else:
        extension.spider_closed(crawler.spider, "finished")
        extension.feed_exporter_closed()
    return extension


@pytest.mark.parametrize("order", ["feed-first", "spider-first"])
def test_success_commits_only_after_all_feeds_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
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

    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    _finish_attestation(crawler, order=order)

    stored = _persisted_row(spider, "complete")
    assert stored[1] == NOW.isoformat()
    assert stored[2] == hashlib.sha256(b"full ruling text").hexdigest()
    assert datetime.fromisoformat(stored[3]) == NOW + timedelta(days=30)
    assert stored[4:] == (
        "success",
        "ruling_full_text",
        1,
        "https://source.invalid/ruling.pdf",
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in (spider.items_path, spider.latest_items_path)
    )
    assert crawler.stats.values["dedup/committed_after_feeds"] == 1
    assert spider._dedup_conn is None
    assert spider._pending_reversible_dedup_commit is None


def test_insecure_existing_feed_is_not_chmodded_or_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    feed = spider.latest_items_path
    feed.chmod(0o644)

    _finish_attestation(crawler)

    assert feed.stat().st_mode & 0o777 == 0o644
    assert _persisted_row(spider, "retry") is None
    assert crawler.stats.values["quality/feed_durability_failed"] == 1


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
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {
            "feedexport/success_count/FileFeedStorage": 1,
            "feedexport/failed_count/FileFeedStorage": 1,
        },
    )

    _finish_attestation(crawler)

    assert _persisted_row(spider, "retry") is None
    assert "retry" not in spider._seen_keys
    assert crawler.stats.values["quality/failures"] == 1
    assert crawler.stats.values["quality/feed_durability_failed"] == 1
    assert spider._dedup_conn is None


def test_fsync_failure_refuses_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    extension = CompletionAttestationExtension(crawler)

    def fail_feed_proof(_spider, _stats):
        raise OSError("disk")

    monkeypatch.setattr(extension, "_attest_feeds", fail_feed_proof)

    extension.spider_closed(spider, "finished")
    extension.feed_exporter_closed()

    assert _persisted_row(spider, "retry") is None
    assert "retry" not in spider._seen_keys
    assert crawler.stats.values["quality/failures"] == 1
    assert crawler.stats.values["quality/feed_durability_failed"] == 1


@pytest.mark.parametrize(
    ("failure_kind", "quality_key"),
    (
        ("spider_error", "spider_errors"),
        ("item_error", "item_errors"),
        ("quality_failure", "quality_failures"),
        ("abnormal_close", None),
    ),
)
def test_nonfeed_eligibility_failures_discard_staged_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    quality_key: str | None,
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    stats = {"feedexport/success_count/FileFeedStorage": 2}
    if failure_kind == "quality_failure":
        stats["quality/failures"] = 1
    crawler = _feed_crawler(spider, tmp_path, stats)
    extension = CompletionAttestationExtension(crawler)
    if failure_kind == "spider_error":
        extension.spider_error(None, None, spider)
    elif failure_kind == "item_error":
        extension.item_error(None, None, spider, None)
    reason = "shutdown" if failure_kind == "abnormal_close" else "finished"

    extension.feed_exporter_closed()
    extension.spider_closed(spider, reason)

    assert _persisted_row(spider, "retry") is None
    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "failure"
    assert terminal["quality_passed"] is False
    assert terminal["feeds_durable"] is False
    if quality_key is not None:
        assert terminal["quality"][quality_key] == 1
    assert spider._dedup_conn is None


def test_dedup_commit_error_discards_stage_and_publishes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )

    def fail_commit():
        raise sqlite3.OperationalError("simulated commit failure")

    monkeypatch.setattr(spider, "commit_staged_seen_reversible", fail_commit)
    _finish_attestation(crawler)

    assert _persisted_row(spider, "retry") is None
    assert spider._staged_dedup_records == {}
    assert crawler.stats.values["quality/dedup_commit_failed"] == 1
    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "failure"
    assert spider._dedup_conn is None


def test_ambiguous_prepare_commit_raise_leaves_no_active_seen_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    real_prepare = spider.commit_staged_seen_reversible

    def commit_then_raise():
        real_prepare()
        assert _full_persisted_row(spider, "retry") is None
        assert len(_pending_rows(spider)) == 1
        raise sqlite3.OperationalError("commit acknowledgement lost")

    monkeypatch.setattr(spider, "commit_staged_seen_reversible", commit_then_raise)
    _finish_attestation(crawler)

    assert _full_persisted_row(spider, "retry") is None
    assert _pending_rows(spider) == []
    assert crawler.stats.values["quality/dedup_commit_failed"] == 1
    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "failure"
    assert spider._dedup_conn is None


def test_late_quality_failure_discards_prepared_shadow_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    real_prepare = spider.commit_staged_seen_reversible

    def prepare_then_flip_quality():
        token = real_prepare()
        assert _full_persisted_row(spider, "retry") is None
        assert len(_pending_rows(spider)) == 1
        crawler.stats.inc_value("quality/failures")
        return token

    monkeypatch.setattr(
        spider,
        "commit_staged_seen_reversible",
        prepare_then_flip_quality,
    )
    _finish_attestation(crawler)

    assert _full_persisted_row(spider, "retry") is None
    assert _pending_rows(spider) == []
    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "failure"
    assert terminal["quality_passed"] is False
    assert spider._dedup_conn is None


def test_staged_records_without_reversible_api_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    monkeypatch.setattr(spider, "commit_staged_seen_reversible", None)

    _finish_attestation(crawler)

    assert _full_persisted_row(spider, "retry") is None
    assert spider._staged_dedup_records == {}
    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "failure"
    assert crawler.stats.values["quality/reversible_dedup_commit_unavailable"] == 1
    assert spider._dedup_conn is None


@pytest.mark.parametrize(
    "publication_error",
    [
        OSError("simulated publication failure"),
        CompletionAlreadyFinalized("simulated existing claim"),
    ],
    ids=["publication-error", "existing-claim"],
)
def test_publication_failure_compensates_new_seen_row_and_closes_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_error: Exception,
) -> None:
    spider = _make_spider(_GenericSpider, tmp_path / "store", monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    observed_prepared = []

    def fail_publication(_spider, _record):
        observed_prepared.append(
            (_full_persisted_row(spider, "retry"), _pending_rows(spider))
        )
        raise publication_error

    monkeypatch.setattr(scraper_extensions, "publish_terminal_record", fail_publication)
    _finish_attestation(crawler)

    assert observed_prepared
    assert observed_prepared[0][0] is None
    assert len(observed_prepared[0][1]) == 1
    assert _full_persisted_row(spider, "retry") is None
    assert _pending_rows(spider) == []
    assert spider._staged_dedup_records == {}
    assert "retry" not in spider._seen_keys
    assert spider._dedup_conn is None
    assert spider._pending_reversible_dedup_commit is None
    assert crawler.stats.values["dedup/compensated_after_publication_failure"] == 1
    startup = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert startup["outcome"] == "started"


def test_publication_failure_restores_exact_prior_seen_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    first = _make_spider(_GenericSpider, store, monkeypatch)
    assert first.stage_seen(
        {
            "document_id": "existing",
            "body_markdown": "prior body",
            "is_consolidated": True,
        }
    )
    first.commit_staged_seen()
    first._dedup_conn.execute(
        "UPDATE seen SET refresh_deadline = ? WHERE key = ?",
        ((NOW - timedelta(days=1)).isoformat(), "existing"),
    )
    first._dedup_conn.commit()
    first._dedup_conn.close()

    spider = _make_spider(_GenericSpider, store, monkeypatch)
    prior = _full_persisted_row(spider, "existing")
    assert "existing" in spider._refresh_keys
    assert DedupPipeline().process_item(
        {"document_id": "existing", "body_markdown": "replacement body"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )

    def fail_publication(_spider, _record):
        assert _full_persisted_row(spider, "existing") == prior
        pending = _pending_rows(spider)
        assert len(pending) == 1
        assert pending[0][1:4] == ("existing", spider.run_id, "success")
        raise OSError("simulated publication failure")

    monkeypatch.setattr(scraper_extensions, "publish_terminal_record", fail_publication)
    _finish_attestation(crawler)

    assert _full_persisted_row(spider, "existing") == prior
    assert spider._staged_dedup_records == {}
    assert "existing" in spider._refresh_keys
    assert not spider.is_seen({"document_id": "existing"})
    assert spider._dedup_conn is None


def test_persistent_pending_cleanup_failure_remains_inactive_and_reopens_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "crawl-artifacts"
    spider = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "retry", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )

    def fail_publication(_spider, _record):
        raise OSError("simulated publication failure")

    def fail_cleanup(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk remains read-only")

    monkeypatch.setattr(scraper_extensions, "publish_terminal_record", fail_publication)
    monkeypatch.setattr(spider, "discard_pending_staged_seen", fail_cleanup)
    monkeypatch.setattr(spider, "rollback_staged_seen_commit", fail_cleanup)
    _finish_attestation(crawler)

    assert _full_persisted_row(spider, "retry") is None
    assert len(_pending_rows(spider)) == 1
    assert spider._dedup_conn is None
    startup = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert startup["outcome"] == "started"

    reopened = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    assert _pending_rows(reopened) == []
    assert _full_persisted_row(reopened, "retry") is None
    assert not reopened.is_seen({"document_id": "retry"})


def test_publisher_raise_after_durable_success_leaves_pending_until_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "crawl-artifacts"
    spider = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "accepted", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    real_publish = scraper_extensions.publish_terminal_record

    def publish_then_raise(publishing_spider, record):
        real_publish(publishing_spider, record)
        assert _full_persisted_row(spider, "accepted") is None
        assert len(_pending_rows(spider)) == 1
        raise OSError("publisher acknowledgement lost")

    monkeypatch.setattr(
        scraper_extensions,
        "publish_terminal_record",
        publish_then_raise,
    )
    extension = _finish_attestation(crawler)

    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "success"
    assert _full_persisted_row(spider, "accepted") is None
    assert len(_pending_rows(spider)) == 1
    assert extension._terminal_success_verified is False
    assert spider._dedup_conn is None

    reopened = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    assert _pending_rows(reopened) == []
    assert _full_persisted_row(reopened, "accepted") is not None
    assert reopened.is_seen({"document_id": "accepted"})


def _leave_authorized_success_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    key: str,
) -> tuple[Path, _GenericSpider]:
    artifacts_root = tmp_path / "crawl-artifacts"
    spider = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    DedupPipeline().process_item(
        {"document_id": key, "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )

    def fail_promotion(_token):
        raise sqlite3.OperationalError("leave inactive for startup test")

    monkeypatch.setattr(spider, "accept_staged_seen_commit", fail_promotion)
    _finish_attestation(crawler)
    assert _full_persisted_row(spider, key) is None
    assert len(_pending_rows(spider)) == 1
    assert spider._dedup_conn is None
    return artifacts_root, spider


@pytest.mark.parametrize(
    "failure_site",
    ["success-verifier", "authorization-inspection"],
)
def test_reopen_preserves_authorized_pending_on_transient_proof_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    key = f"transient-{failure_site}"
    artifacts_root, _original = _leave_authorized_success_pending(
        tmp_path,
        monkeypatch,
        key=key,
    )
    if failure_site == "success-verifier":
        original_callable = base_spider_module.verify_terminal_record

        def transient_failure(*_args, **_kwargs):
            raise OSError("transient items/source proof failure")

        patched_name = "verify_terminal_record"
    else:
        original_callable = base_spider_module.terminal_authorization_exists

        def transient_failure(*_args, **_kwargs):
            raise OSError("transient authorization inspection failure")

        patched_name = "terminal_authorization_exists"
    monkeypatch.setattr(base_spider_module, patched_name, transient_failure)

    ambiguous = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    assert len(_pending_rows(ambiguous)) == 1
    assert _full_persisted_row(ambiguous, key) is None
    assert not ambiguous.is_seen({"document_id": key})
    ambiguous._dedup_conn.close()

    monkeypatch.setattr(base_spider_module, patched_name, original_callable)
    recovered = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    assert _pending_rows(recovered) == []
    assert _full_persisted_row(recovered, key) is not None
    assert recovered.is_seen({"document_id": key})


def test_reopen_discards_pending_for_strictly_authorized_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "crawl-artifacts"
    spider = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    key = "authorized-failure"
    DedupPipeline().process_item(
        {"document_id": key, "body_markdown": "complete"}, spider
    )
    spider.commit_staged_seen_reversible()
    assert len(_pending_rows(spider)) == 1
    _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    quality = evaluate_crawl_quality(
        {"quality/failures": 1},
        spider.name,
        "shutdown",
        spider_errors=1,
    )
    failure = build_terminal_record(
        spider,
        finish_reason="shutdown",
        quality=quality,
        feeds=failed_feed_durability(),
        source_validation=failed_source_validation(spider.name),
        completed_at=NOW + timedelta(hours=1),
        outcome="failure",
    )
    publish_terminal_record(spider, failure)
    spider._dedup_conn.close()

    reopened = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    assert _pending_rows(reopened) == []
    assert _full_persisted_row(reopened, key) is None
    assert not reopened.is_seen({"document_id": key})


def test_reopen_promotes_orphan_pending_only_for_exact_verified_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "crawl-artifacts"
    spider = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    DedupPipeline().process_item(
        {"document_id": "recovered", "body_markdown": "complete"}, spider
    )
    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )

    def fail_promotion(_token):
        raise sqlite3.OperationalError("process died before pending promotion")

    monkeypatch.setattr(spider, "accept_staged_seen_commit", fail_promotion)
    extension = _finish_attestation(crawler)

    terminal = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
    assert terminal["outcome"] == "success"
    assert extension._terminal_success_verified is True
    assert _full_persisted_row(spider, "recovered") is None
    assert len(_pending_rows(spider)) == 1
    assert spider._dedup_conn is None

    reopened = _make_spider(_GenericSpider, artifacts_root, monkeypatch)
    assert _pending_rows(reopened) == []
    assert _full_persisted_row(reopened, "recovered") is not None
    assert reopened.is_seen({"document_id": "recovered"})


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

    crawler = _feed_crawler(
        spider,
        tmp_path,
        {"feedexport/success_count/FileFeedStorage": 2},
    )
    _finish_attestation(crawler)

    stored = _persisted_row(spider, "summary")
    assert stored[1] is None
    assert stored[4:] == (
        "incomplete",
        "article_summary",
        0,
        "https://source.invalid/ruling.pdf",
    )
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

    extension = CompletionAttestationExtension(crawler)
    extension._discard_staged(spider, [])
    extension._commit_staged(spider, [])

    assert calls == []
