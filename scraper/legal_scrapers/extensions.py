"""Live terminal progress display for Scrapy crawls.

Every run in this project redirects its logs to a per-run ``LOG_FILE`` (see
``base.py``), which leaves the terminal silent for the whole crawl. With low
concurrency, a long ``DOWNLOAD_DELAY`` and AutoThrottle, that makes it hard to
tell whether scraping is progressing, stalling, or erroring.

``LiveProgressExtension`` fills that gap: it subscribes to crawl signals and
feeds ``crawler.stats`` snapshots into a single, process-global dashboard that
owns exactly one ``rich.Live``. For a single ``scrapy crawl`` the dashboard
draws a detailed panel; when several spiders run together (``python -m
legal_scrapers.run``) it draws one compact table with a row per spider. Routing every
spider through one ``Live`` is what makes the multi-spider view possible — two
concurrent ``Live`` instances on the same stdout would corrupt the terminal.

The live display is purely additive and read-only against the crawl. A separate
``CompletionAttestationExtension`` joins Scrapy's spider-close and
feed-exporter-close signals, proves durable outputs, commits generic staged dedup
outcomes, validates source-specific completion, and publishes a terminal record.
Totals are unknown up front (the ``matsne`` spider is
two-phase and ``spider_idle``-driven), so the display shows spinners plus
running counters and rates rather than a misleading "X% complete" bar.
"""

import logging
import os
import stat
import sys
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

from scrapy import signals
from scrapy.exceptions import NotConfigured

from .completion import (
    CompletionAlreadyFinalized,
    CompletionError,
    CompletionMaterializationPending,
    attest_feed_outputs,
    attest_materialized_outputs,
    build_terminal_record,
    evaluate_crawl_quality,
    failed_feed_durability,
    failed_source_validation,
    generic_source_validation,
    hash_and_fsync_private_file,
    publish_terminal_record,
    terminal_authorization_exists,
    validate_ordinary_identity_source,
    validate_supremecourt_source,
    verify_terminal_record,
)
from .utils.pagination import finalize_pagination_scope

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Candidate item keys to show as the "last item" label, in priority order.
# Covers every item type in items.py; the first present, non-empty value wins.
_TITLE_KEYS = (
    "title",
    "subject",
    "case_no",
    "case_number",
    "document_number",
    "document_no",
    "decision_no",
    "number",
    "app_no",
    "case_id",
    "slug",
    "document_id",
)

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MAX_TITLE_LEN = 48
_LOG_MAX_BYTES = 50 * 1024 * 1024
_LOG_BACKUPS = 10


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        descriptor = os.open(
            self.baseFilename,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        return os.fdopen(
            descriptor,
            "a",
            encoding=self.encoding,
            errors=self.errors,
        )


class _OnlySpider(logging.Filter):
    def __init__(self, spider):
        super().__init__()
        self.spider = spider

    def filter(self, record):
        bound = getattr(record, "spider", None)
        return bound is self.spider or (
            bound is not None
            and getattr(bound, "name", None) == getattr(self.spider, "name", None)
        )


class RotatingSpiderLogExtension:
    """Write each spider's own records to a bounded 50 MiB × 10 private log."""

    def __init__(self, crawler):
        self.crawler = crawler
        self.handler = None

    @classmethod
    def from_crawler(cls, crawler):
        extension = cls(crawler)
        crawler.signals.connect(extension.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.spider_closed, signal=signals.spider_closed)
        return extension

    def spider_opened(self, spider):
        path = Path(spider.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise PermissionError(
                    f"spider log is not a private regular file: {path}"
                )
        else:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            os.close(descriptor)
        handler = _PrivateRotatingFileHandler(
            path,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.addFilter(_OnlySpider(spider))
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
        self.handler = handler

    def spider_closed(self, spider, reason):
        if self.handler is None:
            return
        logging.getLogger().removeHandler(self.handler)
        self.handler.close()
        self.handler = None


class CompletionAttestationExtension:
    """Publish exactly one terminal attestation after spider and feeds close.

    Scrapy does not guarantee whether this extension observes ``spider_closed``
    before the feed exporter emits ``feed_exporter_closed``.  Both handlers only
    record state and call the same guarded join, so finalization is independent of
    signal order.  Callback and item-pipeline failures are counted from their own
    signals because only zero is acceptable for a successful attestation.
    """

    _MAX_OPERATIONAL_ERRORS = 32

    def __init__(self, crawler):
        self.crawler = crawler
        self._spider_closed_seen = False
        self._feed_exporter_closed_seen = False
        self._finish_reason = None
        self._spider = None
        self._spider_errors = 0
        self._item_errors = 0
        self._prepared = False
        self._preparation_errors: list[str] = []
        self._finalizing = False
        self._finalized = False
        self._staged_resolved = False
        self._dedup_commit_token = None
        self._dedup_prepare_attempted = False
        self._terminal_published = False
        self._terminal_success_verified = False
        self._preserve_pending_for_recovery = False

    @classmethod
    def from_crawler(cls, crawler):
        extension = cls(crawler)
        crawler.signals.connect(
            extension.spider_closed,
            signal=signals.spider_closed,
        )
        crawler.signals.connect(
            extension.feed_exporter_closed,
            signal=signals.feed_exporter_closed,
        )
        crawler.signals.connect(
            extension.spider_error,
            signal=signals.spider_error,
        )
        crawler.signals.connect(
            extension.item_error,
            signal=signals.item_error,
        )
        # The combined runner uses these exact observed counters with the same pure
        # quality evaluator after the process has drained.
        crawler.completion_attestation = extension
        return extension

    def spider_error(self, failure, response, spider):
        del failure, response, spider
        self._spider_errors += 1

    def item_error(self, item, response, spider, failure):
        del item, response, spider, failure
        self._item_errors += 1

    def spider_closed(self, spider, reason):
        if self._spider_closed_seen:
            self._try_finalize()
            return
        self._spider = spider
        self._finish_reason = reason
        self._prepare_completion(spider, reason)
        self._spider_closed_seen = True
        # Supreme Court intentionally disables FeedExporter and materializes both
        # outputs in prepare_completion(), so Scrapy may never emit its exporter-close
        # signal for that spider.  The custom durability proof below is its equivalent.
        if getattr(spider, "name", "") == "supremecourt":
            self._feed_exporter_closed_seen = True
        self._try_finalize()

    def feed_exporter_closed(self, **_kwargs):
        self._feed_exporter_closed_seen = True
        if self._spider is None:
            self._spider = getattr(self.crawler, "spider", None)
        self._try_finalize()

    def _bounded_error(self, errors: list[str], message: str) -> None:
        if len(errors) < self._MAX_OPERATIONAL_ERRORS:
            errors.append(str(message)[:500])

    def _record_operational_failure(
        self,
        spider,
        errors: list[str],
        *,
        kind: str,
        detail: object,
    ) -> None:
        message = f"{kind}: {detail}"
        self._bounded_error(errors, message)
        try:
            spider.record_quality_failure(
                kind,
                "completion://terminal-attestation",
                detail=str(detail)[:300],
            )
        except Exception as exc:  # noqa: BLE001 - retain the original failure
            self._bounded_error(
                errors,
                f"quality_failure_recording_failed: {type(exc).__name__}: {exc}",
            )

    def _prepare_completion(self, spider, reason) -> None:
        if self._prepared:
            return
        self._prepared = True
        hook = getattr(spider, "prepare_completion", None)
        if callable(hook):
            try:
                hook(reason)
            except Exception as exc:  # noqa: BLE001 - preparation must fail closed
                self._record_operational_failure(
                    spider,
                    self._preparation_errors,
                    kind="completion_preparation_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )

        identity_hook = getattr(spider, "finalize_evidence_identity_journal", None)
        if callable(identity_hook):
            try:
                identity_hook()
            except Exception as exc:  # noqa: BLE001 - identity proof must fail closed
                self._record_operational_failure(
                    spider,
                    self._preparation_errors,
                    kind="identity_journal_finalization_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )

        # Every registered scope must have an idempotent terminal outcome.  Most
        # adapters finalize on their last callback; this pass catches interrupted or
        # otherwise stranded scopes and records their repair evidence before quality
        # is evaluated.
        trackers = getattr(spider, "_pagination_reconcilers", {})
        scope_urls = getattr(spider, "_pagination_scope_urls", {})
        for scope, tracker in list(trackers.items()):
            if getattr(tracker, "_finalized", None) is not None:
                continue
            try:
                finalize_pagination_scope(
                    spider,
                    tracker,
                    url=scope_urls.get(scope, f"pagination://{scope}"),
                )
            except Exception as exc:  # noqa: BLE001 - reconciliation must fail closed
                self._record_operational_failure(
                    spider,
                    self._preparation_errors,
                    kind="pagination_finalization_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _try_finalize(self) -> None:
        if (
            self._finalized
            or self._finalizing
            or not self._spider_closed_seen
            or not self._feed_exporter_closed_seen
        ):
            return
        spider = self._spider
        if spider is None:
            return
        self._finalizing = True
        try:
            self._finalize(spider, str(self._finish_reason))
        except CompletionAlreadyFinalized as exc:
            self._preserve_pending_for_recovery = self._has_authorized_terminal(spider)
            if not self._preserve_pending_for_recovery:
                self._rollback_committed_stage(spider, [])
            spider.logger.error("completion attestation already finalized: %s", exc)
        except CompletionMaterializationPending as exc:
            # The durable WAL decision is reconstructible, but exception paths never
            # activate dedup.  Keep only the inactive pending batch; locked startup
            # recovery will materialize run.json and then promote it.
            self._preserve_pending_for_recovery = True
            spider.logger.error(
                "completion terminal is authorized and pending startup recovery: %s",
                exc,
            )
        except Exception as exc:  # noqa: BLE001 - leave startup-only on publication crash
            self._preserve_pending_for_recovery = self._has_authorized_terminal(spider)
            if not self._preserve_pending_for_recovery:
                self._rollback_committed_stage(spider, [])
            spider.logger.exception("completion attestation failed closed: %s", exc)
        finally:
            if (
                not self._terminal_success_verified
                and not self._preserve_pending_for_recovery
            ):
                self._rollback_committed_stage(spider, [])
            self._close_generic_dedup(spider)
            self._finalizing = False
            self._finalized = True

    @staticmethod
    def _has_published_success(spider) -> bool:
        try:
            verify_terminal_record(
                spider.run_metadata_path,
                expected_source=spider.name,
                expected_run_id=spider.run_id,
                expected_items_path=spider.items_path,
            )
        except Exception:  # noqa: BLE001 - only exact success authorizes promotion
            return False
        return True

    @staticmethod
    def _has_authorized_terminal(spider) -> bool:
        try:
            return terminal_authorization_exists(
                spider.run_metadata_path,
                expected_source=spider.name,
                expected_run_id=spider.run_id,
            )
        except Exception:  # noqa: BLE001 - ambiguity keeps only inactive rows
            # Only a definite, safely observed absence permits discard.  A malformed
            # or temporarily unreadable WAL may still encode an irrevocable decision;
            # retaining inactive shadow rows is always safer than false activation or
            # irreversible deletion.
            return True

    @staticmethod
    def _stats_mapping(crawler) -> dict:
        stats = getattr(crawler, "stats", None)
        return dict(stats.get_stats()) if stats is not None else {}

    def _attest_feeds(self, spider, stats: dict):
        if getattr(spider, "name", "") == "supremecourt":
            return attest_materialized_outputs(
                spider.items_path,
                spider.latest_items_path,
            )
        feeds = self.crawler.settings.getdict("FEEDS")
        return attest_feed_outputs(
            feeds,
            stats,
            spider.items_path,
            spider.latest_items_path,
        )

    def _discard_staged(self, spider, errors: list[str]) -> None:
        if self._staged_resolved or getattr(spider, "name", "") == "supremecourt":
            return
        discard = getattr(spider, "discard_staged_seen", None)
        if not callable(discard):
            self._staged_resolved = True
            return
        try:
            discard()
        except Exception as exc:  # noqa: BLE001 - preserve the primary refusal
            self._bounded_error(
                errors,
                f"dedup_discard_failed: {type(exc).__name__}: {exc}",
            )
        else:
            self._staged_resolved = True

    def _commit_staged(self, spider, errors: list[str]) -> None:
        if self._staged_resolved or getattr(spider, "name", "") == "supremecourt":
            return
        staged = getattr(spider, "_staged_dedup_records", None)
        if not staged:
            self._staged_resolved = True
            return
        commit = getattr(spider, "commit_staged_seen_reversible", None)
        rollback = getattr(spider, "rollback_staged_seen_commit", None)
        accept = getattr(spider, "accept_staged_seen_commit", None)
        discard_pending = getattr(spider, "discard_pending_staged_seen", None)
        if not all(
            callable(method)
            for method in (commit, rollback, accept, discard_pending)
        ):
            self._discard_staged(spider, errors)
            self._record_operational_failure(
                spider,
                errors,
                kind="reversible_dedup_commit_unavailable",
                detail="staged records cannot be committed before terminal publication",
            )
            return
        self._dedup_prepare_attempted = True
        try:
            token = commit()
        except Exception as exc:  # noqa: BLE001 - dedup commit must fail closed
            # A SQLite commit can succeed durably and still raise before returning.
            # Cleanup by run identity, not only by a token that may never arrive.
            self._rollback_committed_stage(spider, errors)
            self._record_operational_failure(
                spider,
                errors,
                kind="dedup_commit_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return
        if token is None:
            self._rollback_committed_stage(spider, errors)
            self._record_operational_failure(
                spider,
                errors,
                kind="dedup_commit_failed",
                detail="reversible dedup commit returned no compensation token",
            )
            return
        self._dedup_commit_token = token
        self._staged_resolved = True
        committed = int(getattr(token, "committed_count", 0))
        stats = getattr(self.crawler, "stats", None)
        if stats is not None:
            stats.inc_value("dedup/committed_after_feeds", committed)

    def _rollback_committed_stage(self, spider, errors: list[str]) -> None:
        """Discard pending rows without ever modifying active ``seen`` state."""
        if getattr(spider, "name", "") == "supremecourt":
            return
        token = self._dedup_commit_token or getattr(
            spider, "_pending_reversible_dedup_commit", None
        )
        if token is None and not self._dedup_prepare_attempted:
            self._discard_staged(spider, errors)
            return
        discarded = None
        failures: list[str] = []
        discard_pending = getattr(spider, "discard_pending_staged_seen", None)
        if callable(discard_pending):
            try:
                discarded = discard_pending(getattr(spider, "run_id", None))
            except Exception as exc:  # noqa: BLE001 - pending remains inactive
                failures.append(
                    f"dedup_pending_discard_failed: {type(exc).__name__}: {exc}"
                )
        else:
            failures.append("pending dedup discard API disappeared")

        # A token-aware delete is a useful second attempt if the run-id recovery API
        # itself failed.  Both paths only delete shadow rows.
        rollback = getattr(spider, "rollback_staged_seen_commit", None)
        if discarded is None and token is not None and callable(rollback):
            try:
                discarded = rollback(token)
            except Exception as exc:  # noqa: BLE001 - pending remains inactive
                failures.append(
                    f"dedup_pending_rollback_failed: {type(exc).__name__}: {exc}"
                )

        if discarded is not None:
            self._dedup_commit_token = None
            self._dedup_prepare_attempted = False
        else:
            for failure in failures:
                self._bounded_error(errors, failure)
            if failures:
                spider.logger.error(
                    "dedup pending cleanup failed; rows remain inactive: %s",
                    "; ".join(failures),
                )
        self._staged_resolved = False
        self._discard_staged(spider, errors)
        stats = getattr(self.crawler, "stats", None)
        if stats is not None and discarded is not None:
            stats.inc_value(
                "dedup/compensated_after_publication_failure",
                int(discarded or 0),
            )

    def _accept_committed_stage(self, spider) -> None:
        token = self._dedup_commit_token
        if token is None:
            return
        accept = getattr(spider, "accept_staged_seen_commit", None)
        if not callable(accept):
            raise RuntimeError("reversible dedup acceptance API disappeared")
        accept(token)
        self._dedup_commit_token = None
        self._dedup_prepare_attempted = False

    @staticmethod
    def _close_generic_dedup(spider) -> None:
        if getattr(spider, "name", "") == "supremecourt":
            return
        connection = getattr(spider, "_dedup_conn", None)
        if connection is None:
            return
        try:
            connection.close()
        except Exception as exc:  # noqa: BLE001 - terminal path must still unwind
            spider.logger.exception("closing generic dedup store failed: %s", exc)
        finally:
            spider._dedup_conn = None

    @staticmethod
    def _run_items_proof(feeds):
        run_rows = [row for row in feeds.files if row.get("role") == "run"]
        if len(run_rows) != 1:
            raise CompletionError("durable feeds do not contain one run-scoped item proof")
        return hash_and_fsync_private_file(run_rows[0]["path"])

    def _finalize(self, spider, reason: str) -> None:
        errors = list(self._preparation_errors)
        stats = self._stats_mapping(self.crawler)

        try:
            feeds = self._attest_feeds(spider, stats)
        except Exception as exc:  # noqa: BLE001 - all feed failures are terminal
            expected_count = 1 if getattr(spider, "evidence_crawl", False) else 2
            feeds = failed_feed_durability(
                configured_count=expected_count,
                expected_count=expected_count,
            )
            self._discard_staged(spider, errors)
            self._record_operational_failure(
                spider,
                errors,
                kind="feed_durability_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )

        try:
            if getattr(spider, "name", "") == "supremecourt":
                source_validation = validate_supremecourt_source(
                    spider.run_dir,
                    reason,
                    self._run_items_proof(feeds),
                )
            elif getattr(spider, "evidence_crawl", False):
                source_validation = validate_ordinary_identity_source(
                    spider,
                    self._run_items_proof(feeds),
                )
            else:
                source_validation = generic_source_validation()
        except Exception as exc:  # noqa: BLE001 - source proof must fail closed
            source_validation = failed_source_validation(
                kind=(
                    "supremecourt_partial_v1"
                    if getattr(spider, "name", "") == "supremecourt"
                    else "generic"
                )
            )
            self._record_operational_failure(
                spider,
                errors,
                kind="source_validation_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )

        # Staged generic successes remain reversible until every eligibility check
        # available at this point has passed.  Feed durability is a prerequisite,
        # while source/quality failures discard the stage instead of poisoning future
        # crawls with success keys from an ineligible attempt.
        preliminary_stats = self._stats_mapping(self.crawler)
        preliminary_quality = evaluate_crawl_quality(
            preliminary_stats,
            str(getattr(spider, "name", "")),
            reason,
            spider_errors=self._spider_errors,
            item_errors=self._item_errors,
            reconcilers=getattr(spider, "_pagination_reconcilers", {}),
        )
        if (
            not errors
            and feeds.durable
            and preliminary_quality.passed
            and source_validation.get("passed") is True
        ):
            self._commit_staged(spider, errors)
        else:
            self._discard_staged(spider, errors)

        # Re-read stats because preparation, feed, dedup, or source validation may
        # have recorded a quality failure above.
        stats = self._stats_mapping(self.crawler)
        quality = evaluate_crawl_quality(
            stats,
            str(getattr(spider, "name", "")),
            reason,
            spider_errors=self._spider_errors,
            item_errors=self._item_errors,
            reconcilers=getattr(spider, "_pagination_reconcilers", {}),
        )
        success = (
            not errors
            and quality.passed
            and feeds.durable
            and source_validation.get("passed") is True
        )
        if not success:
            # A late quality/stat change can invalidate a run after its shadow batch
            # was prepared.  Failure records never authorize pending promotion.
            self._rollback_committed_stage(spider, errors)
        record = build_terminal_record(
            spider,
            finish_reason=reason,
            quality=quality,
            feeds=feeds,
            source_validation=source_validation,
            outcome="success" if success else "failure",
            failure_count=0 if success else max(1, len(errors)),
        )
        publication = publish_terminal_record(spider, record)
        self._terminal_published = True
        if success:
            # The publisher's return can itself be ambiguous around filesystem
            # durability.  Reload the exact authoritative record (including recovery-
            # guard absence) before allowing SQLite activation.
            if not self._has_published_success(spider):
                raise CompletionError(
                    "published success could not be strictly reverified"
                )
            self._terminal_success_verified = True
            self._accept_committed_stage(spider)
            spider.logger.info(
                "completion attestation published for %s (latest_updated=%s)",
                spider.run_id,
                publication.latest_updated,
            )
        else:
            spider.logger.error(
                "crawl is ineligible; terminal failure attestation published for %s",
                spider.run_id,
            )


def extract_title(item) -> str | None:
    """Best-effort short label for a scraped item.

    Tries the well-known title-ish keys first, then falls back to the first
    short string field. Returns ``None`` if nothing suitable is found.
    """
    for key in _TITLE_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value.strip())

    for value in item.values():
        if isinstance(value, str) and value.strip() and len(value) <= 120:
            return _truncate(value.strip())

    return None


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _MAX_TITLE_LEN:
        return text[: _MAX_TITLE_LEN - 1] + "…"
    return text


@dataclass
class ProgressSnapshot:
    """Everything the renderers need, decoupled from Scrapy internals."""

    spider: str
    elapsed_s: float
    items: int
    requests: int
    responses: int
    total_items: int | None = None
    status_counts: dict = field(default_factory=dict)
    queue: int = 0
    errors: int = 0
    warnings: int = 0
    last_item: str | None = None
    done: bool = False
    finish_reason: str | None = None


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _items_per_min(items: int, elapsed_s: float) -> float:
    # Below ~1s the rate is dominated by startup noise (and would print an absurd
    # number); real crawls run for minutes, so suppress it until there's signal.
    if elapsed_s < 1.0:
        return 0.0
    return items * 60.0 / elapsed_s


def _format_status_counts(status_counts: dict) -> Text:
    if not status_counts:
        return Text("—", style="dim")
    text = Text()
    for code in sorted(status_counts):
        count = status_counts[code]
        if 200 <= code < 300:
            style = "green"
        elif 300 <= code < 400:
            style = "cyan"
        else:
            style = "red"
        if len(text):
            text.append("  ")
        text.append(f"{code}×{count}", style=style)
    return text


def _status_glyph(snap: ProgressSnapshot, spinner_frame: str) -> tuple[str, str, str]:
    """Return (glyph, glyph_style, status_word) for a snapshot."""
    if snap.finish_reason == "queued":
        return "·", "dim", "queued"
    if snap.done:
        if snap.finish_reason == "finished":
            return "✓", "bold green", "finished"
        return "■", "yellow", snap.finish_reason or "done"
    return spinner_frame, "cyan", "scraping"


def render_panel(snap: ProgressSnapshot, spinner_frame: str) -> Panel:
    """Detailed single-spider panel. Pure — no Scrapy/terminal dependencies."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="bold dim", no_wrap=True)
    grid.add_column()

    rate = _items_per_min(snap.items, snap.elapsed_s)
    items_text = Text(f"{snap.items}", style="bold green")
    items_text.append(f" this run   ({rate:.0f}/min)", style="dim")
    if snap.total_items is not None:
        items_text.append(f"   · {snap.total_items} total", style="bold blue")
    grid.add_row("items", items_text)
    grid.add_row("requests", f"{snap.requests} sent · {snap.responses} done")
    grid.add_row("status", _format_status_counts(snap.status_counts))
    grid.add_row("queue", Text(f"{snap.queue} pending", style="dim"))

    err_text = Text()
    err_text.append(f"{snap.warnings} ⚠", style="yellow" if snap.warnings else "dim")
    err_text.append("  ")
    err_text.append(f"{snap.errors} ✖", style="red" if snap.errors else "dim")
    grid.add_row("errors", err_text)

    if snap.last_item:
        grid.add_row("last", Text(snap.last_item, style="italic"))

    glyph, glyph_style, status_word = _status_glyph(snap, spinner_frame)

    title = Text()
    title.append(f" {snap.spider} ", style="bold")
    title.append(f"· {status_word} ", style="dim")

    subtitle = Text()
    subtitle.append(_format_elapsed(snap.elapsed_s), style="dim")
    subtitle.append(f" {glyph}", style=glyph_style)

    return Panel(
        grid,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="green"
        if snap.done and snap.finish_reason == "finished"
        else "cyan",
        padding=(0, 1),
    )


def render_table(snapshots, spinner_frame: str, elapsed_s: float):
    """Compact multi-spider view: one row per spider plus a totals footer. Pure."""
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)  # status glyph
    table.add_column(style="bold", no_wrap=True)  # spider name
    table.add_column(justify="right", no_wrap=True)  # run items
    table.add_column(justify="right", no_wrap=True)  # total items
    table.add_column(justify="right", style="dim", no_wrap=True)  # requests
    table.add_column(justify="right", no_wrap=True)  # errors
    table.add_column()  # status / state

    header = ("", "SPIDER", "RUN", "TOTAL", "REQS", "ERR", "STATUS")
    table.add_row(*(Text(h, style="bold dim") for h in header))

    total_items = 0
    grand_total_items = 0
    grand_total_known = False
    total_errors = 0
    for snap in snapshots:
        glyph, glyph_style, status_word = _status_glyph(snap, spinner_frame)
        total_items += snap.items
        if snap.total_items is not None:
            grand_total_items += snap.total_items
            grand_total_known = True
        total_errors += snap.errors
        if snap.finish_reason == "queued":
            state = Text("queued", style="dim")
        elif snap.done:
            state = Text(status_word, style=glyph_style)
        else:
            state = _format_status_counts(snap.status_counts)
        err_style = "red" if snap.errors else "dim"
        table.add_row(
            Text(glyph, style=glyph_style),
            Text(snap.spider),
            Text(str(snap.items), style="green" if snap.items else "dim"),
            Text(
                str(snap.total_items) if snap.total_items is not None else "—",
                style="blue" if snap.total_items is not None else "dim",
            ),
            str(snap.requests),
            Text(str(snap.errors), style=err_style),
            state,
        )

    footer = Text()
    footer.append("run ", style="bold dim")
    footer.append(f"{total_items} items", style="bold green")
    if grand_total_known:
        footer.append(" · total ", style="bold dim")
        footer.append(f"{grand_total_items} items", style="bold blue")
    footer.append(f" · {total_errors} errors", style="red" if total_errors else "dim")
    footer.append(f" · elapsed {_format_elapsed(elapsed_s)}", style="dim")

    return Panel(
        Group(table, Text(""), footer),
        title=Text(" scraping ", style="bold"),
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )


class _ProgressDashboard:
    """Process-global owner of the single ``rich.Live`` for all spiders.

    Extensions register themselves on ``spider_opened`` and the dashboard pulls
    each one's current snapshot every tick. With one active spider it renders the
    detailed panel; with several (or when the runner pre-``declare``s spiders) it
    renders the compact table.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.live = None
        self.loop = None
        self.exts = []  # registered LiveProgressExtension instances
        self.declared = []  # spider names pre-declared by the runner
        self.force_table = False
        self.started_monotonic = 0.0
        self.spinner_index = 0
        self._finished = False

    # --- runner-facing API -------------------------------------------------

    def declare(self, names):
        """Pre-register the spiders a multi-spider run will launch (shows the
        full table, including not-yet-opened spiders as ``queued``)."""
        self.declared = list(names)
        self.force_table = True

    def finish(self):
        """Stop the live display and print a per-spider summary. Idempotent.

        Called by the runner after ``process.start()`` returns (all crawls done);
        single-spider runs finish themselves via ``on_spider_closed``.
        """
        if self._finished:
            return
        self._finished = True
        if self.loop is not None and self.loop.running:
            self.loop.stop()
        self.loop = None
        if self.live is not None:
            try:
                self.live.update(self._render(), refresh=True)
            finally:
                self.live.stop()
                self.live = None
        for ext in self.exts:
            snap = ext.current_snapshot()
            total = (
                f" · {snap.total_items} total" if snap.total_items is not None else ""
            )
            print(
                f"✓ {snap.spider}: {snap.items} items this run{total}"
                f" · {snap.responses} responses"
                f" · {snap.errors} errors · {_format_elapsed(snap.elapsed_s)}"
                f" · {snap.finish_reason or 'done'}"
            )

    # --- extension-facing API ----------------------------------------------

    def register(self, ext):
        self.exts.append(ext)
        self._ensure_started()

    def on_spider_closed(self, ext):
        # In multi-spider mode the runner calls finish() once the reactor stops.
        # In single-spider mode there is no runner, so finish when all done.
        if not self.declared and all(e.final_snapshot is not None for e in self.exts):
            self.finish()

    # --- internals ---------------------------------------------------------

    def _multi(self) -> bool:
        return self.force_table or len(self.exts) > 1 or len(self.declared) > 1

    def _ensure_started(self):
        if self.live is not None or self._finished:
            return
        from rich.live import Live
        from twisted.internet import task

        self.started_monotonic = time.monotonic()
        try:
            self.live = Live(self._render(), auto_refresh=False, transient=False)
            self.live.start()
        except Exception:  # pragma: no cover - never break a crawl over the UI
            self.live = None
            return
        self.loop = task.LoopingCall(self._tick)
        self.loop.start(0.25, now=False)

    def _render(self):
        frame = _SPINNER_FRAMES[self.spinner_index]
        if not self._multi():
            if self.exts:
                return render_panel(self.exts[0].current_snapshot(), frame)
            return Text("starting…", style="dim")

        snaps = []
        opened = set()
        for ext in self.exts:
            snap = ext.current_snapshot()
            snaps.append(snap)
            opened.add(snap.spider)
        for name in self.declared:
            if name not in opened:
                snaps.append(
                    ProgressSnapshot(
                        spider=name,
                        elapsed_s=0.0,
                        items=0,
                        requests=0,
                        responses=0,
                        total_items=None,
                        finish_reason="queued",
                    )
                )
        elapsed = time.monotonic() - self.started_monotonic
        return render_table(snaps, frame, elapsed)

    def _tick(self):
        if self.live is None:
            return
        try:
            self.spinner_index = (self.spinner_index + 1) % len(_SPINNER_FRAMES)
            self.live.update(self._render(), refresh=True)
        except Exception:  # pragma: no cover - rendering must never crash a crawl
            pass


# Single process-wide instance shared by every crawler's extension.
_DASHBOARD = _ProgressDashboard()


def declare_spiders(names):
    """Runner hook: announce the spiders a multi-spider run will launch."""
    _DASHBOARD.declare(names)


def finish_dashboard():
    """Runner hook: tear down the live display after all crawls finish."""
    _DASHBOARD.finish()


class LiveProgressExtension:
    """Scrapy extension that feeds one spider's progress into the dashboard."""

    def __init__(self, crawler):
        self.crawler = crawler
        self.stats = crawler.stats
        self.spider = None
        self.started_monotonic = 0.0
        self.last_item_title: str | None = None
        self.final_snapshot: ProgressSnapshot | None = None

    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool("PROGRESS_DISPLAY_ENABLED", True):
            raise NotConfigured("PROGRESS_DISPLAY_ENABLED is False")
        # Don't draw a live display when stdout isn't a terminal (pipes, CI,
        # cron): the crawl then behaves exactly as before.
        if not sys.stdout.isatty():
            raise NotConfigured("stdout is not a TTY")

        ext = cls(crawler)
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        return ext

    # --- signal handlers (run on the Twisted reactor thread) ---------------

    def spider_opened(self, spider):
        self.spider = spider
        self.started_monotonic = time.monotonic()
        _DASHBOARD.register(self)

    def item_scraped(self, item, spider):
        title = extract_title(item)
        if title:
            self.last_item_title = title

    def spider_closed(self, spider, reason):
        self.final_snapshot = self._snapshot(spider, done=True, finish_reason=reason)
        _DASHBOARD.on_spider_closed(self)

    # --- dashboard hooks ---------------------------------------------------

    def current_snapshot(self) -> ProgressSnapshot:
        if self.final_snapshot is not None:
            return self.final_snapshot
        return self._snapshot(self.spider)

    def _snapshot(self, spider, *, done=False, finish_reason=None) -> ProgressSnapshot:
        get = self.stats.get_value
        status_counts = {}
        prefix = "downloader/response_status_count/"
        for key, value in self.stats.get_stats().items():
            if key.startswith(prefix):
                try:
                    status_counts[int(key[len(prefix) :])] = value
                except ValueError:
                    continue

        enqueued = get("scheduler/enqueued", 0) or 0
        dequeued = get("scheduler/dequeued", 0) or 0

        return ProgressSnapshot(
            spider=getattr(spider, "name", "spider"),
            elapsed_s=time.monotonic() - self.started_monotonic,
            items=get("item_scraped_count", 0) or 0,
            requests=get("downloader/request_count", 0) or 0,
            responses=get("downloader/response_count", 0) or 0,
            total_items=self._total_items(spider),
            status_counts=status_counts,
            queue=max(0, enqueued - dequeued),
            errors=get("log_count/ERROR", 0) or 0,
            warnings=get("log_count/WARNING", 0) or 0,
            last_item=self.last_item_title,
            done=done,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _total_items(spider) -> int | None:
        if not getattr(spider, "dedup_enabled", False):
            return None
        counter = getattr(spider, "dedup_seen_count", None)
        if not callable(counter):
            return None
        return counter()
