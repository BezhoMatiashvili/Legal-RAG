"""Focused tests for the create-only v3 source-evidence crawl mode."""

from __future__ import annotations

import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from scrapy.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers import completion  # noqa: E402
from legal_scrapers.spiders import base  # noqa: E402
from legal_scrapers.spiders.ecd_spider import EcdSpider  # noqa: E402
from legal_scrapers.spiders.matsne_spider import MatsneSpider  # noqa: E402
from legal_scrapers.spiders.supremecourt_spider import (  # noqa: E402
    SupremecourtSpider,
)


class _Stats:
    def __init__(self):
        self.values: dict[str, object] = {}

    def set_value(self, key, value):
        self.values[key] = value

    def inc_value(self, key, count=1):
        self.values[key] = int(self.values.get(key, 0)) + count

    def get_stats(self):
        return dict(self.values)


def _settings() -> Settings:
    return Settings(
        {
            "EVIDENCE_CRAWL_ENABLED": True,
            "EVIDENCE_CODE_REVISION": "abcdef0123456789",
            "EVIDENCE_CODE_IDENTITY_SHA256": "c" * 64,
            "HTTPCACHE_ENABLED": True,
            "DEDUP_ENABLED": True,
        }
    )


def _configured_evidence_spider(tmp_path, monkeypatch):
    evidence_root = tmp_path / "source-evidence" / base.EVIDENCE_SNAPSHOT_ID
    monkeypatch.setattr(base, "EVIDENCE_ARTIFACTS_ROOT", evidence_root)
    monkeypatch.setattr(completion, "_EVIDENCE_ARTIFACT_ROOT", evidence_root.absolute())
    spider = EcdSpider(start_date="1900-01-01", end_date="2026-07-15")
    settings = _settings()
    stats = _Stats()
    spider.crawler = SimpleNamespace(settings=settings, stats=stats)
    spider.configure_run_outputs(settings)
    return spider, settings, stats


def test_evidence_mode_forces_isolation_and_one_create_only_feed(
    tmp_path, monkeypatch
):
    spider, settings, stats = _configured_evidence_spider(tmp_path, monkeypatch)

    assert settings.getbool("HTTPCACHE_ENABLED") is False
    assert settings.getbool("DEDUP_ENABLED") is False
    assert spider.dedup_enabled is False
    assert spider._dedup_conn is None
    assert spider.latest_dir is None
    assert spider.latest_items_path is None
    assert spider.latest_metadata_path is None
    feeds = settings.getdict("FEEDS")
    assert list(feeds) == [os.fspath(spider.items_path)]
    assert feeds[os.fspath(spider.items_path)]["overwrite"] is False
    assert stats.values["evidence/enabled"] == 1
    assert not (evidence_root := spider.source_root).joinpath("seen.sqlite").exists()
    assert not (evidence_root / "latest").exists()


def test_evidence_terminal_binds_contract_and_verifies_without_latest(
    tmp_path, monkeypatch
):
    spider, settings, stats = _configured_evidence_spider(tmp_path, monkeypatch)
    spider.items_path.write_bytes(b'{"decision_document_id":"one"}\n')
    spider.items_path.chmod(0o600)
    spider.record_evidence_identity_event("unique", "one")
    spider.finalize_evidence_identity_journal()
    stats.set_value("within_run/observed_identity_count", 1)
    stats.set_value("within_run/unique_identity_count", 1)
    stats.set_value("feedexport/success_count/FileFeedStorage", 1)

    feeds = completion.attest_feed_outputs(
        settings.getdict("FEEDS"),
        stats.get_stats(),
        spider.items_path,
        None,
    )
    quality = completion.evaluate_crawl_quality(
        stats.get_stats(), spider.name, "finished"
    )
    record = completion.build_terminal_record(
        spider,
        finish_reason="finished",
        quality=quality,
        feeds=feeds,
        source_validation=completion.validate_ordinary_identity_source(
            spider,
            completion.hash_and_fsync_private_file(spider.items_path),
        ),
        completed_at=datetime.now(UTC).replace(microsecond=0),
    )

    contract = record["crawl_contract"]
    assert record["schema_version"] == 2
    assert record["latest_items_path"] is None
    assert contract["artifact_root"] == os.fspath(spider.source_root.parent)
    assert contract["http_cache_enabled"] is False
    assert contract["cross_run_dedup_enabled"] is False
    assert contract["shared_seen_sqlite_accessed"] is False
    assert contract["within_run_duplicates"] == {
        "observed": 1,
        "unique": 1,
        "duplicates": 0,
        "missing_identity": 0,
        "reconciled": True,
    }
    assert [row["role"] for row in contract["durable_outputs"]] == [
        "identity_journal",
        "run",
    ]

    result = completion.publish_terminal_record(spider, record)
    assert result.latest_path is None
    assert result.latest_updated is False
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source="ecd",
        expected_run_id=spider.run_id,
        expected_items_path=spider.items_path,
    ) == record


def test_evidence_contract_rejects_failure_signal_or_output_drift(
    tmp_path, monkeypatch
):
    spider, settings, stats = _configured_evidence_spider(tmp_path, monkeypatch)
    spider.items_path.write_bytes(b"")
    spider.items_path.chmod(0o600)
    spider.finalize_evidence_identity_journal()
    stats.set_value("feedexport/success_count/FileFeedStorage", 1)
    feeds = completion.attest_feed_outputs(
        settings.getdict("FEEDS"), stats.get_stats(), spider.items_path, None
    )
    record = completion.build_terminal_record(
        spider,
        finish_reason="finished",
        quality=completion.evaluate_crawl_quality(
            stats.get_stats(), spider.name, "finished"
        ),
        feeds=feeds,
        source_validation=completion.validate_ordinary_identity_source(
            spider,
            completion.hash_and_fsync_private_file(spider.items_path),
        ),
    )

    record["crawl_contract"]["failure_signals"]["feed_failures"] = 1
    with pytest.raises(completion.CompletionError, match="failure_signals drifted"):
        completion._validate_terminal_mapping(record, require_success=True)


def test_ordinary_evidence_source_cannot_create_a_second_run(tmp_path, monkeypatch):
    first, _settings_value, _stats = _configured_evidence_spider(
        tmp_path, monkeypatch
    )
    first.finalize_evidence_identity_journal()
    second = EcdSpider(start_date="1900-01-01", end_date="2026-07-15")
    settings = _settings()
    second.crawler = SimpleNamespace(settings=settings, stats=_Stats())

    with pytest.raises(FileExistsError, match="run exactly once"):
        second.configure_run_outputs(settings)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"doc_type": "main"}, "full/default"),
        ({"seed_ids_file": "subset.txt"}, "without seed"),
    ],
)
def test_matsne_evidence_rejects_nondefault_or_seeded_corpus(kwargs, message):
    spider = MatsneSpider(
        start_date="1900-01-01",
        end_date="2026-07-15",
        **kwargs,
    )
    settings = _settings()
    spider.crawler = SimpleNamespace(settings=settings, stats=_Stats())

    with pytest.raises(ValueError, match=message):
        spider.configure_run_outputs(settings)


def test_evidence_startup_rejects_wrong_interval_before_creating_artifacts():
    spider = EcdSpider(start_date="1900-01-02", end_date="2026-07-15")
    settings = _settings()
    spider.crawler = SimpleNamespace(settings=settings, stats=_Stats())

    with pytest.raises(ValueError, match="exact interval"):
        spider.configure_run_outputs(settings)


@pytest.mark.parametrize(
    ("initial_window_days", "runtime", "message"),
    [
        (8, 14_400, "initial_window_days=7"),
        (7, 3_600, "CLOSESPIDER_TIMEOUT=14400"),
    ],
)
def test_supreme_evidence_freezes_window_and_runtime(
    initial_window_days, runtime, message
):
    spider = SupremecourtSpider(
        start_date="1900-01-01",
        end_date="2026-07-15",
        initial_window_days=initial_window_days,
        parent_run_id="none",
    )
    settings = _settings()
    settings.set("CLOSESPIDER_TIMEOUT", runtime)
    spider.crawler = SimpleNamespace(settings=settings, stats=_Stats())

    with pytest.raises(ValueError, match=message):
        spider.configure_run_outputs(settings)


def test_evidence_run_id_collision_is_not_auto_suffixed(tmp_path):
    spider = EcdSpider(start_date="1900-01-01", end_date="2026-07-15")
    spider.started_at = datetime(2026, 7, 15, tzinfo=UTC)
    spider.source_root = tmp_path / "ecd"
    expected = spider.source_root / "runs" / (
        "20260715T000000Z_start-1900-01-01_end-2026-07-15"
    )
    expected.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        spider.build_run_id(create_only=True)


def test_evidence_constants_remain_frozen():
    assert base.EVIDENCE_START_DATE == date(1900, 1, 1)
    assert base.EVIDENCE_END_DATE == date(2026, 7, 15)
    assert base.EVIDENCE_SNAPSHOT_ID == "v3_512_attested_20260715_01"
