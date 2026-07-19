"""Focused filesystem and schema tests for crawl completion attestations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from legal_scrapers import completion  # noqa: E402


RUN_ID = "20200101T000000Z_start-2020-01-01_end-2020-01-02"
SUCCESS_STATS = {"feedexport/success_count/FileFeedStorage": 2}


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _private_file(path: Path, payload: bytes) -> Path:
    _private_directory(path.parent)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _spider(
    tmp_path: Path,
    *,
    run_id: str = RUN_ID,
    source: str = "ecd",
    items: bytes = b'{"document_id":"1"}\n',
) -> SimpleNamespace:
    source_root = _private_directory(tmp_path / "artifacts" / source)
    run_dir = _private_directory(source_root / "runs" / run_id)
    latest_dir = _private_directory(source_root / "latest")
    items_path = _private_file(run_dir / "items.jsonl", items)
    latest_items_path = _private_file(latest_dir / "items.jsonl", items)
    return SimpleNamespace(
        name=source,
        run_id=run_id,
        source_root=source_root,
        run_dir=run_dir,
        latest_dir=latest_dir,
        scraping_start_date=date(2020, 1, 1),
        scraping_end_date=date(2020, 1, 2),
        started_at=datetime(2020, 1, 1, tzinfo=UTC),
        items_path=items_path,
        latest_items_path=latest_items_path,
        log_path=run_dir / "spider.log",
        run_metadata_path=run_dir / "run.json",
        latest_metadata_path=latest_dir / "run.json",
    )


def _configured_feeds(spider: SimpleNamespace) -> dict[str, object]:
    # Reverse semantic order deliberately; attestation output must still be canonical.
    return {
        os.fspath(spider.items_path): {},
        os.fspath(spider.latest_items_path): {},
    }


def _quality(spider: SimpleNamespace) -> completion.QualityEvaluation:
    return completion.evaluate_crawl_quality({}, spider.name, "finished")


def _terminal_record(spider: SimpleNamespace) -> dict[str, object]:
    feeds = completion.attest_feed_outputs(
        _configured_feeds(spider),
        SUCCESS_STATS,
        spider.items_path,
        spider.latest_items_path,
    )
    return completion.build_terminal_record(
        spider,
        finish_reason="finished",
        quality=_quality(spider),
        feeds=feeds,
        source_validation=completion.generic_source_validation(),
        completed_at=datetime(2020, 1, 1, 1, tzinfo=UTC),
    )


def _publish_startup(spider: SimpleNamespace) -> dict[str, object]:
    record = completion.build_startup_record(spider)
    completion.publish_startup_metadata(
        spider.run_metadata_path,
        spider.latest_metadata_path,
        record,
    )
    return record


def test_startup_invalidates_latest_before_run_creation_failure(tmp_path, monkeypatch):
    spider = _spider(tmp_path)
    startup = completion.build_startup_record(spider)
    _private_file(spider.latest_metadata_path, b"older-success-record\n")

    def fail_run_create(*_args, **_kwargs):
        raise OSError("simulated run metadata failure")

    monkeypatch.setattr(completion, "atomic_create_private", fail_run_create)
    with pytest.raises(OSError, match="simulated run metadata failure"):
        completion.publish_startup_metadata(
            spider.run_metadata_path,
            spider.latest_metadata_path,
            startup,
        )

    assert spider.latest_metadata_path.read_bytes() == completion.canonical_json_bytes(
        startup
    )
    assert not spider.run_metadata_path.exists()


def test_startup_run_record_is_create_only(tmp_path):
    spider = _spider(tmp_path)
    startup = completion.build_startup_record(spider)
    sentinel = _private_file(spider.run_metadata_path, b"do-not-overwrite\n")

    with pytest.raises(FileExistsError, match="already exists"):
        completion.publish_startup_metadata(
            spider.run_metadata_path,
            spider.latest_metadata_path,
            startup,
        )

    assert sentinel.read_bytes() == b"do-not-overwrite\n"
    assert spider.latest_metadata_path.read_bytes() == completion.canonical_json_bytes(
        startup
    )


@pytest.mark.parametrize(
    "payload",
    [b"", b'{"document_id":"1"}\n'],
    ids=["empty", "nonempty"],
)
def test_feed_attestation_accepts_exact_identical_private_outputs(tmp_path, payload):
    spider = _spider(tmp_path, items=payload)

    proof = completion.attest_feed_outputs(
        _configured_feeds(spider),
        SUCCESS_STATS,
        spider.items_path,
        spider.latest_items_path,
    )

    expected_sha = hashlib.sha256(payload).hexdigest()
    assert proof.durable is True
    assert (proof.configured_count, proof.success_count, proof.failure_count) == (2, 2, 0)
    assert [row["role"] for row in proof.files] == ["latest", "run"]
    assert {row["size_bytes"] for row in proof.files} == {len(payload)}
    assert {row["sha256"] for row in proof.files} == {expected_sha}


def test_feed_attestation_rejects_missing_output(tmp_path):
    spider = _spider(tmp_path)
    spider.latest_items_path.unlink()

    with pytest.raises(completion.CompletionError, match="not materialized"):
        completion.attest_feed_outputs(
            _configured_feeds(spider),
            SUCCESS_STATS,
            spider.items_path,
            spider.latest_items_path,
        )


def test_feed_attestation_rejects_exporter_failure(tmp_path):
    spider = _spider(tmp_path)
    failed_stats = {
        "feedexport/success_count/FileFeedStorage": 1,
        "feedexport/failed_count/FileFeedStorage": 1,
    }

    with pytest.raises(completion.CompletionError, match="counts do not prove success"):
        completion.attest_feed_outputs(
            _configured_feeds(spider),
            failed_stats,
            spider.items_path,
            spider.latest_items_path,
        )


def test_feed_attestation_rejects_nonlocal_output(tmp_path):
    spider = _spider(tmp_path)

    with pytest.raises(completion.CompletionError, match="nonlocal"):
        completion.attest_feed_outputs(
            [os.fspath(spider.items_path), "s3://legal-crawls/latest.jsonl"],
            SUCCESS_STATS,
            spider.items_path,
            spider.latest_items_path,
        )


def test_feed_attestation_rejects_symlinked_output(tmp_path):
    spider = _spider(tmp_path)
    target = _private_file(tmp_path / "outside" / "items.jsonl", spider.items_path.read_bytes())
    spider.latest_items_path.unlink()
    spider.latest_items_path.symlink_to(target)

    with pytest.raises(completion.UnsafePathError, match="symlink"):
        completion.attest_feed_outputs(
            _configured_feeds(spider),
            SUCCESS_STATS,
            spider.items_path,
            spider.latest_items_path,
        )


def test_feed_attestation_rejects_insecure_output_mode(tmp_path):
    spider = _spider(tmp_path)
    spider.latest_items_path.chmod(0o644)

    with pytest.raises(completion.UnsafePathError, match="mode 0600"):
        completion.attest_feed_outputs(
            _configured_feeds(spider),
            SUCCESS_STATS,
            spider.items_path,
            spider.latest_items_path,
        )


def test_feed_attestation_fails_closed_when_file_fsync_fails(tmp_path, monkeypatch):
    spider = _spider(tmp_path)

    def fail_fsync(_descriptor):
        raise OSError("simulated feed fsync failure")

    monkeypatch.setattr(completion.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated feed fsync failure"):
        completion.attest_feed_outputs(
            _configured_feeds(spider),
            SUCCESS_STATS,
            spider.items_path,
            spider.latest_items_path,
        )


def test_atomic_create_fsync_failure_publishes_nothing_and_cleans_temp(
    tmp_path, monkeypatch
):
    parent = _private_directory(tmp_path / "publication")
    destination = parent / "run.json"
    real_fsync = completion.os.fsync

    def fail_regular_file_fsync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("simulated fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(completion.os, "fsync", fail_regular_file_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        completion.atomic_create_private(destination, b"payload\n")

    assert not os.path.lexists(destination)
    assert list(parent.iterdir()) == []


def test_terminal_record_is_canonical_and_deterministic(tmp_path):
    spider = _spider(tmp_path)

    first = _terminal_record(spider)
    second = _terminal_record(spider)
    payload = completion.canonical_json_bytes(first)

    assert first == second
    assert payload == completion.canonical_json_bytes(second)
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert json.loads(payload) == first
    assert first["outcome"] == "success"
    assert first["quality_passed"] is True
    assert first["feeds_durable"] is True
    assert first["failure_count"] == 0


def test_terminal_publication_is_single_claim_and_updates_matching_latest(tmp_path):
    spider = _spider(tmp_path)
    _publish_startup(spider)
    terminal = _terminal_record(spider)

    result = completion.publish_terminal_record(spider, terminal)

    assert result.latest_updated is True
    expected = completion.canonical_json_bytes(terminal)
    assert spider.run_metadata_path.read_bytes() == expected
    assert spider.latest_metadata_path.read_bytes() == expected
    candidate = (
        spider.run_metadata_path.parent / completion.TERMINAL_CANDIDATE_FILENAME
    )
    authorization = (
        spider.run_metadata_path.parent / completion.FINALIZATION_CLAIM_FILENAME
    )
    assert candidate.read_bytes() == expected
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o600
    authorization_record = json.loads(authorization.read_bytes())
    assert authorization_record == {
        "schema_version": 1,
        "state": "terminal_authorized",
        "source": spider.name,
        "run_id": spider.run_id,
        "candidate_filename": completion.TERMINAL_CANDIDATE_FILENAME,
        "terminal_size_bytes": len(expected),
        "terminal_sha256": hashlib.sha256(expected).hexdigest(),
    }
    assert authorization.read_bytes() == completion.canonical_json_bytes(
        authorization_record
    )
    assert stat.S_IMODE(authorization.stat().st_mode) == 0o600
    assert not os.path.lexists(
        spider.run_metadata_path.parent
        / completion.TERMINAL_RECOVERY_GUARD_FILENAME
    )
    with pytest.raises(completion.CompletionAlreadyFinalized):
        completion.publish_terminal_record(spider, terminal)
    assert spider.run_metadata_path.read_bytes() == expected


def test_terminal_prevalidation_failure_leaves_startup_without_claim(tmp_path):
    spider = _spider(tmp_path)
    startup = _publish_startup(spider)
    invalid = _terminal_record(spider)
    invalid["failure_count"] = 1
    claim = spider.run_metadata_path.parent / completion.FINALIZATION_CLAIM_FILENAME

    with pytest.raises(completion.CompletionError, match="inconsistent proof"):
        completion.publish_terminal_record(spider, invalid)

    assert spider.run_metadata_path.read_bytes() == completion.canonical_json_bytes(
        startup
    )
    assert not os.path.lexists(claim)


def test_terminal_publication_failure_after_claim_stays_startup_only(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    startup = _publish_startup(spider)
    terminal = _terminal_record(spider)
    claim = spider.run_metadata_path.parent / completion.FINALIZATION_CLAIM_FILENAME
    real_replace = completion.atomic_replace_terminal_private

    def fail_authoritative_replace(path, payload, startup_payload):
        if Path(path) == spider.run_metadata_path:
            raise OSError("simulated terminal publication failure")
        return real_replace(path, payload, startup_payload)

    monkeypatch.setattr(
        completion,
        "atomic_replace_terminal_private",
        fail_authoritative_replace,
    )
    with pytest.raises(
        completion.CompletionMaterializationPending,
        match="materialization is pending",
    ):
        completion.publish_terminal_record(spider, terminal)

    assert spider.run_metadata_path.read_bytes() == completion.canonical_json_bytes(
        startup
    )
    assert claim.is_file()
    candidate = (
        spider.run_metadata_path.parent / completion.TERMINAL_CANDIDATE_FILENAME
    )
    assert candidate.read_bytes() == completion.canonical_json_bytes(terminal)
    with pytest.raises(completion.CompletionError, match="does not exactly match"):
        completion.verify_terminal_record(spider.run_metadata_path)

    monkeypatch.setattr(
        completion,
        "atomic_replace_terminal_private",
        real_replace,
    )
    with completion.source_lock(spider.source_root):
        assert completion.recover_authorized_terminal(
            spider.run_metadata_path,
            expected_source=spider.name,
            expected_run_id=spider.run_id,
        )
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source=spider.name,
        expected_run_id=spider.run_id,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal
    with pytest.raises(completion.CompletionAlreadyFinalized):
        completion.publish_terminal_record(spider, terminal)


def test_terminal_parent_fsync_failure_restores_startup_then_wal_recovers(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    startup = _publish_startup(spider)
    terminal = _terminal_record(spider)
    startup_payload = completion.canonical_json_bytes(startup)
    terminal_payload = completion.canonical_json_bytes(terminal)
    claim = spider.run_metadata_path.parent / completion.FINALIZATION_CLAIM_FILENAME
    candidate = (
        spider.run_metadata_path.parent / completion.TERMINAL_CANDIDATE_FILENAME
    )
    real_fsync = completion.os.fsync
    failed = False

    def fail_once_after_terminal_rename(descriptor):
        nonlocal failed
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if (
            not failed
            and is_directory
            and spider.run_metadata_path.read_bytes() == terminal_payload
        ):
            failed = True
            raise OSError("simulated post-rename directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(completion.os, "fsync", fail_once_after_terminal_rename)
    with pytest.raises(completion.CompletionMaterializationPending):
        completion.publish_terminal_record(spider, terminal)

    assert failed is True
    assert spider.run_metadata_path.read_bytes() == startup_payload
    assert spider.latest_metadata_path.read_bytes() == startup_payload
    assert claim.is_file()
    assert candidate.read_bytes() == terminal_payload
    assert not any(
        path.name.endswith(".tmp") or path.name.endswith(".startup-backup")
        for path in spider.run_metadata_path.parent.iterdir()
    )
    with pytest.raises(completion.CompletionError, match="does not exactly match"):
        completion.verify_terminal_record(spider.run_metadata_path)
    with completion.source_lock(spider.source_root):
        assert completion.recover_authorized_terminal(
            spider.run_metadata_path,
            expected_source=spider.name,
            expected_run_id=spider.run_id,
        )
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal


def test_compound_commit_and_restore_fault_returns_success_for_exact_authorized_run(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    startup = _publish_startup(spider)
    terminal = _terminal_record(spider)
    startup_payload = completion.canonical_json_bytes(startup)
    terminal_payload = completion.canonical_json_bytes(terminal)
    real_replace = completion.os.replace
    real_fsync = completion.os.fsync
    terminal_renamed = False
    commit_fsync_failed = False
    refused_restores = 0

    def fail_both_restore_renames(source, destination, *args, **kwargs):
        nonlocal terminal_renamed, refused_restores
        if destination == spider.run_metadata_path.name:
            if terminal_renamed:
                refused_restores += 1
                raise OSError("simulated restore rename failure")
            terminal_renamed = True
        return real_replace(source, destination, *args, **kwargs)

    def fail_terminal_commit_fsync(descriptor):
        nonlocal commit_fsync_failed
        if (
            terminal_renamed
            and not commit_fsync_failed
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            commit_fsync_failed = True
            raise OSError("simulated terminal commit fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(completion.os, "replace", fail_both_restore_renames)
    monkeypatch.setattr(completion.os, "fsync", fail_terminal_commit_fsync)
    result = completion.publish_terminal_record(spider, terminal)

    assert commit_fsync_failed is True
    assert refused_restores >= 2
    assert spider.run_metadata_path.read_bytes() == terminal_payload
    assert spider.latest_metadata_path.read_bytes() == startup_payload
    assert result.latest_updated is False
    assert not list(spider.run_metadata_path.parent.glob(".*.startup-backup"))
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source=spider.name,
        expected_run_id=spider.run_id,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal


def test_exact_authorized_run_with_unstable_parent_fsync_reports_pending(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    _publish_startup(spider)
    terminal = _terminal_record(spider)
    terminal_payload = completion.canonical_json_bytes(terminal)
    real_replace = completion.os.replace
    real_fsync = completion.os.fsync
    terminal_renamed = False

    def refuse_both_startup_restores(source, destination, *args, **kwargs):
        nonlocal terminal_renamed
        if destination == spider.run_metadata_path.name:
            if terminal_renamed:
                raise OSError("persistent restore rename failure")
            terminal_renamed = True
        return real_replace(source, destination, *args, **kwargs)

    def refuse_terminal_parent_fsync(descriptor):
        if (
            stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and spider.run_metadata_path.read_bytes() == terminal_payload
        ):
            raise OSError("persistent terminal parent fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(completion.os, "replace", refuse_both_startup_restores)
    monkeypatch.setattr(completion.os, "fsync", refuse_terminal_parent_fsync)
    with pytest.raises(
        completion.CompletionMaterializationPending,
        match="durably confirmed",
    ):
        completion.publish_terminal_record(spider, terminal)

    assert spider.run_metadata_path.read_bytes() == terminal_payload
    with pytest.raises(OSError, match="persistent terminal parent fsync failure"):
        completion.verify_terminal_record(spider.run_metadata_path)

    monkeypatch.setattr(completion.os, "replace", real_replace)
    monkeypatch.setattr(completion.os, "fsync", real_fsync)
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source=spider.name,
        expected_run_id=spider.run_id,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal


def test_commit_and_restore_fsync_failures_leave_recoverable_authorization(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    startup = _publish_startup(spider)
    terminal = _terminal_record(spider)
    startup_payload = completion.canonical_json_bytes(startup)
    terminal_payload = completion.canonical_json_bytes(terminal)
    real_fsync = completion.os.fsync
    failed_terminal_commit = False
    failed_startup_restore = False

    def fail_commit_and_restore_fsync(descriptor):
        nonlocal failed_terminal_commit, failed_startup_restore
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            return real_fsync(descriptor)
        current = spider.run_metadata_path.read_bytes()
        if current == terminal_payload and not failed_terminal_commit:
            failed_terminal_commit = True
            raise OSError("simulated terminal commit fsync failure")
        if (
            failed_terminal_commit
            and current == startup_payload
            and not failed_startup_restore
        ):
            failed_startup_restore = True
            raise OSError("simulated startup restore fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(completion.os, "fsync", fail_commit_and_restore_fsync)
    with pytest.raises(completion.CompletionMaterializationPending):
        completion.publish_terminal_record(spider, terminal)

    assert failed_terminal_commit is True
    assert failed_startup_restore is True
    assert spider.run_metadata_path.read_bytes() == startup_payload
    assert spider.latest_metadata_path.read_bytes() == startup_payload
    assert (
        spider.run_metadata_path.parent / completion.FINALIZATION_CLAIM_FILENAME
    ).is_file()
    assert (
        spider.run_metadata_path.parent / completion.TERMINAL_CANDIDATE_FILENAME
    ).read_bytes() == terminal_payload
    with pytest.raises(completion.CompletionError, match="does not exactly match"):
        completion.verify_terminal_record(spider.run_metadata_path)
    with completion.source_lock(spider.source_root):
        assert completion.recover_authorized_terminal(
            spider.run_metadata_path,
            expected_source=spider.name,
            expected_run_id=spider.run_id,
        )
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal


def test_authorization_parent_fsync_failure_precedes_run_materialization(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    startup = _publish_startup(spider)
    terminal = _terminal_record(spider)
    startup_payload = completion.canonical_json_bytes(startup)
    claim = spider.run_metadata_path.parent / completion.FINALIZATION_CLAIM_FILENAME
    real_fsync = completion.os.fsync
    failed_authorization_fsync = False

    def fail_once_after_authorization_link(descriptor):
        nonlocal failed_authorization_fsync
        if (
            not failed_authorization_fsync
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and claim.is_file()
            and spider.run_metadata_path.read_bytes() == startup_payload
        ):
            failed_authorization_fsync = True
            raise OSError("simulated authorization directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(completion.os, "fsync", fail_once_after_authorization_link)
    with pytest.raises(OSError, match="authorization directory fsync failure"):
        completion.publish_terminal_record(spider, terminal)

    assert failed_authorization_fsync is True
    assert spider.run_metadata_path.read_bytes() == startup_payload
    assert spider.latest_metadata_path.read_bytes() == startup_payload
    assert claim.is_file()
    with pytest.raises(completion.CompletionError, match="does not exactly match"):
        completion.verify_terminal_record(spider.run_metadata_path)
    with completion.source_lock(spider.source_root):
        assert completion.recover_authorized_terminal(
            spider.run_metadata_path,
            expected_source=spider.name,
            expected_run_id=spider.run_id,
        )
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal


def test_source_lock_unlock_failure_does_not_mask_committed_publication(
    tmp_path, monkeypatch
):
    spider = _spider(tmp_path)
    _publish_startup(spider)
    terminal = _terminal_record(spider)
    real_flock = completion.fcntl.flock
    unlock_attempted = False

    def fail_unlock(descriptor, operation):
        nonlocal unlock_attempted
        if operation == completion.fcntl.LOCK_UN:
            unlock_attempted = True
            raise OSError("simulated lock cleanup failure")
        return real_flock(descriptor, operation)

    monkeypatch.setattr(completion.fcntl, "flock", fail_unlock)
    result = completion.publish_terminal_record(spider, terminal)

    assert unlock_attempted is True
    assert result.latest_updated is True
    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source=spider.name,
        expected_run_id=spider.run_id,
        expected_items_path=spider.items_path,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal


def test_terminal_failure_is_explicitly_nonqualifying_and_not_verifiable(tmp_path):
    spider = _spider(tmp_path)
    _publish_startup(spider)
    quality = completion.evaluate_crawl_quality(
        {"quality/failures": 1},
        spider.name,
        "shutdown",
        spider_errors=1,
        item_errors=1,
    )
    record = completion.build_terminal_record(
        spider,
        finish_reason="shutdown",
        quality=quality,
        feeds=completion.failed_feed_durability(),
        source_validation=completion.failed_source_validation(spider.name),
        completed_at=datetime(2020, 1, 1, 1, tzinfo=UTC),
        outcome="failure",
    )

    assert record["outcome"] == "failure"
    assert record["quality_passed"] is False
    assert record["feeds_durable"] is False
    assert record["failure_count"] >= 1
    assert record["quality"]["passed"] is False
    assert record["feed_outputs"]["durable"] is False
    assert record["source_validation"]["passed"] is False

    completion.publish_terminal_record(spider, record)
    with pytest.raises(completion.CompletionError, match="required terminal state"):
        completion.verify_terminal_record(spider.run_metadata_path)


def test_terminal_publication_does_not_clobber_newer_latest_attempt(tmp_path):
    first = _spider(tmp_path, run_id=RUN_ID)
    _publish_startup(first)
    newer = _spider(
        tmp_path,
        run_id="20200102T000000Z_start-2020-01-01_end-2020-01-02",
        items=first.items_path.read_bytes(),
    )
    newer_startup = _publish_startup(newer)
    latest_before = completion.canonical_json_bytes(newer_startup)

    result = completion.publish_terminal_record(first, _terminal_record(first))

    assert result.latest_updated is False
    assert first.run_metadata_path.read_bytes() != latest_before
    assert first.latest_metadata_path.read_bytes() == latest_before


def test_strict_verifier_rejects_startup_only_then_accepts_terminal(tmp_path):
    spider = _spider(tmp_path)
    _publish_startup(spider)

    with pytest.raises(completion.CompletionError, match="authorization|missing"):
        completion.verify_terminal_record(
            spider.run_metadata_path,
            expected_source=spider.name,
            expected_run_id=spider.run_id,
            expected_items_path=spider.items_path,
        )

    terminal = _terminal_record(spider)
    completion.publish_terminal_record(spider, terminal)
    verified = completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source=spider.name,
        expected_run_id=spider.run_id,
        expected_items_path=spider.items_path,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert verified == terminal

    spider.run_metadata_path.write_text(
        json.dumps(terminal, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    spider.run_metadata_path.chmod(0o600)
    with pytest.raises(completion.CompletionError, match="authorized candidate"):
        completion.verify_terminal_record(spider.run_metadata_path)


def test_strict_verifier_rejects_candidate_and_authorization_drift(tmp_path):
    spider = _spider(tmp_path)
    _publish_startup(spider)
    terminal = _terminal_record(spider)
    completion.publish_terminal_record(spider, terminal)
    candidate = spider.run_dir / completion.TERMINAL_CANDIDATE_FILENAME
    authorization = spider.run_dir / completion.FINALIZATION_CLAIM_FILENAME

    candidate.write_bytes(candidate.read_bytes() + b" ")
    candidate.chmod(0o600)
    with pytest.raises(completion.CompletionError, match="does not match"):
        completion.verify_terminal_record(spider.run_metadata_path)

    candidate.write_bytes(completion.canonical_json_bytes(terminal))
    candidate.chmod(0o600)
    authorization_record = json.loads(authorization.read_bytes())
    authorization_record["terminal_sha256"] = "0" * 64
    authorization.write_bytes(completion.canonical_json_bytes(authorization_record))
    authorization.chmod(0o600)
    with pytest.raises(completion.CompletionError, match="does not match"):
        completion.verify_terminal_record(spider.run_metadata_path)


def test_supreme_validator_report_must_include_explicit_empty_errors(tmp_path):
    spider = _spider(tmp_path, source="supremecourt")
    _private_file(spider.run_dir / "partial_manifest.json", b"{}\n")
    _private_file(spider.run_dir / "items.journal.jsonl", b"")
    items_proof = completion.hash_and_fsync_private_file(spider.items_path)
    journal_sha = hashlib.sha256(b"").hexdigest()

    def incomplete_report(run_dir):
        return {
            "schema_version": 1,
            "valid": True,
            "run_id": Path(run_dir).name,
            "run_dir": os.fspath(Path(run_dir).absolute()),
            "finish_reason": "closespider_timeout",
            "items_sha256": items_proof.sha256,
            "journal_sha256": journal_sha,
            "unresolved_failure_count": 0,
        }

    validator = SimpleNamespace(validate_run=incomplete_report)
    with pytest.raises(completion.CompletionError, match="contains errors"):
        completion.validate_supremecourt_source(
            spider.run_dir,
            "closespider_timeout",
            items_proof,
            validator=validator,
        )


def test_natural_supreme_finish_requires_frozen_lower_bound_and_exhausted_cursors(
    tmp_path,
):
    spider = _spider(tmp_path, source="supremecourt")
    _private_file(spider.run_dir / "partial_manifest.json", b"{}\n")
    _private_file(spider.run_dir / "items.journal.jsonl", b"")
    items_proof = completion.hash_and_fsync_private_file(spider.items_path)
    journal_sha = hashlib.sha256(b"").hexdigest()

    def report(run_dir):
        return {
            "schema_version": 1,
            "valid": True,
            "run_id": Path(run_dir).name,
            "run_dir": os.fspath(Path(run_dir).absolute()),
            "finish_reason": "finished",
            "items_sha256": items_proof.sha256,
            "journal_sha256": journal_sha,
            "unresolved_failure_count": 0,
            "errors": [],
            "lower_bound": "1900-01-02",
            "per_chamber_resume_cursors": {"a": None, "b": None, "c": None},
            "oldest_fully_completed_global_date_frontier": "1900-01-02",
        }

    validator = SimpleNamespace(validate_run=report)
    with pytest.raises(completion.CompletionError, match="exhaustion proof"):
        completion.validate_supremecourt_source(
            spider.run_dir,
            "finished",
            items_proof,
            validator=validator,
        )

    original_report = report

    def exhausted_report(run_dir):
        value = original_report(run_dir)
        value["lower_bound"] = "1900-01-01"
        value["per_chamber_resume_cursors"] = {
            "ადმინისტრაციულ საქმეთა პალატა": None,
            "სამოქალაქო საქმეთა პალატა": None,
            "სისხლის სამართლის საქმეთა პალატა": None,
        }
        value["oldest_fully_completed_global_date_frontier"] = "1900-01-01"
        return value

    accepted = completion.validate_supremecourt_source(
        spider.run_dir,
        "finished",
        items_proof,
        validator=SimpleNamespace(validate_run=exhausted_report),
    )
    assert accepted["passed"] is True
    assert accepted["finish_reason"] == "finished"


def test_strict_verifier_uses_run_scoped_items_after_latest_advances(tmp_path):
    spider = _spider(tmp_path)
    _publish_startup(spider)
    terminal = _terminal_record(spider)
    completion.publish_terminal_record(spider, terminal)

    newer = _spider(
        tmp_path,
        run_id="20200102T000000Z_start-2020-01-01_end-2020-01-02",
        items=b'{"document_id":"newer"}\n',
    )
    _publish_startup(newer)

    assert completion.verify_terminal_record(
        spider.run_metadata_path,
        expected_source=spider.name,
        expected_run_id=spider.run_id,
        expected_items_path=spider.items_path,
        now=datetime(2030, 1, 1, tzinfo=UTC),
    ) == terminal
