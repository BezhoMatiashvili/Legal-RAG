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
``DurableDedupCommitExtension`` waits for Scrapy's feed-exporter-closed signal,
fsyncs local feeds, and only then commits generic staged dedup outcomes. Totals
are unknown up front (the ``matsne`` spider is
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
from urllib.parse import unquote, urlparse

from scrapy import signals
from scrapy.exceptions import NotConfigured

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


class DurableDedupCommitExtension:
    """Commit generic dedup outcomes only after every feed is durably closed."""

    def __init__(self, crawler):
        self.crawler = crawler

    @classmethod
    def from_crawler(cls, crawler):
        extension = cls(crawler)
        crawler.signals.connect(
            extension.feed_exporter_closed, signal=signals.feed_exporter_closed
        )
        return extension

    @staticmethod
    def _local_feed_path(uri) -> Path | None:
        text = str(uri)
        parsed = urlparse(text)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if not parsed.scheme:
            return Path(text)
        return None

    @staticmethod
    def _fsync_local_feed(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"configured feed was not materialized: {path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError(
                f"configured feed is not owner-only; refusing to chmod existing file: {path}"
            )
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def feed_exporter_closed(self):
        spider = getattr(self.crawler, "spider", None)
        if spider is None or getattr(spider, "name", "") == "supremecourt":
            return
        staged = getattr(spider, "_staged_dedup_records", None)
        if not staged:
            return

        feeds = self.crawler.settings.getdict("FEEDS")
        stats = self.crawler.stats
        feed_stats = stats.get_stats()
        successes = sum(
            int(value or 0)
            for key, value in feed_stats.items()
            if key.startswith("feedexport/success_count/")
        )
        failures = sum(
            int(value or 0)
            for key, value in feed_stats.items()
            if key.startswith("feedexport/failed_count/")
        )
        if not feeds or failures or successes < len(feeds):
            spider.discard_staged_seen()
            stats.inc_value("dedup/feed_commit_refused")
            spider.record_quality_failure(
                "durable_feed_commit_refused",
                "feed-export://configured-outputs",
                detail=(
                    f"feeds={len(feeds)} successes={successes} failures={failures}"
                ),
            )
            spider.logger.error(
                "dedup: refusing staged commit; feeds=%d successes=%d failures=%d",
                len(feeds),
                successes,
                failures,
            )
            return

        try:
            for uri in feeds:
                path = self._local_feed_path(uri)
                if path is not None:
                    self._fsync_local_feed(path)
            committed = spider.commit_staged_seen()
        except Exception as exc:  # noqa: BLE001 - storage failure must fail closed
            spider.discard_staged_seen()
            stats.inc_value("dedup/feed_commit_refused")
            spider.record_quality_failure(
                "durable_feed_commit_refused",
                "feed-export://configured-outputs",
                detail=f"{type(exc).__name__}: {exc}",
            )
            spider.logger.error("dedup: staged commit failed closed: %s", exc)
            return
        stats.inc_value("dedup/committed_after_feeds", committed)


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
