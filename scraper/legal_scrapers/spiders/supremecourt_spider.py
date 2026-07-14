"""supremecourt.ge — newest-first, resumable Supreme Court decision crawl.

The official ``/ka/getCases`` AJAX response exposes an authoritative result total and up
to 30 decision cards.  A broad 1900→today pagination crawl starves chambers and reverses
card order under Scrapy's LIFO scheduler, so this spider coordinates all three chambers
through adaptive date windows instead:

* probe every chamber/window at the newest global end date before fetching any detail;
* split multi-day windows above 30 and expand the following window below 20;
* order the combined detail requests by decision date across chambers;
* fetch only ``/ka/fullcase/{id}/{palata}`` and require nonempty ``#modalBody`` HTML;
* fsync full items before ``seen.sqlite`` is updated, then atomically materialize a
  cumulative partial artifact and resume manifest on graceful close.

ROBOTS: ``/ka/getCases`` is disallowed by the site's robots.txt.  The scoped
``ROBOTSTXT_OBEY=False`` override remains the explicit user-approved exception recorded
2026-06-29.  DOCX/PDF fallbacks are deliberately not fetched in this timed workflow.
"""

from __future__ import annotations

import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import count
from pathlib import Path
from urllib.parse import urlencode

from scrapy import Request
from scrapy.loader import ItemLoader

from ..items import SupremecourtItem
from ..utils.dates import iso_to_year_slashed
from ..utils.markdown import safe_html_to_markdown
from .base import BaseLegalSpider

BASE = "https://www.supremecourt.ge"
GETCASES_URL = f"{BASE}/ka/getCases"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHAMBER_NAMES = {
    "0": "ადმინისტრაციულ საქმეთა პალატა",
    "1": "სამოქალაქო საქმეთა პალატა",
    "2": "სისხლის სამართლის საქმეთა პალატა",
}
CHAMBERS = tuple(int(palata) for palata in CHAMBER_NAMES)
PAGE_SIZE = 30
TARGET_MIN = 20
TARGET_MAX = 30
MAX_WINDOW_DAYS = 366
MAX_PAGES = 5000
MAX_FAILURE_EXAMPLES = 100
_FULLCASE_RE = re.compile(r"/fullcase/(\d+)/(\d+)")
_TOTAL_RE = re.compile(r"სულ\s+მოიძებნა\s+([\d\s,]+)\s+გადაწყვეტილება")

CASE_LABELS = {
    "საქმის ნომერი": "case_number",
    "თარიღი": "date",
    "დავის საგანი": "subject",
    "შედეგი": "result",
    "საჩივრის სახე": "appeal_type",
}


@dataclass
class DateWindow:
    """One chamber's inclusive date window and its durable completion state."""

    window_id: str
    chamber: int
    start: date
    end: date
    continues: bool = True
    parent_id: str | None = None
    status: str = "queued"
    authoritative_total: int | None = None
    expected_pages: int = 1
    pages_seen: set[int] = field(default_factory=set)
    listed_identities: set[str] = field(default_factory=set)
    known_identities: set[str] = field(default_factory=set)
    pending_identities: set[str] = field(default_factory=set)
    new_identities: set[str] = field(default_factory=set)
    cards: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    failure_count: int = 0

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


class NewestFirstPlanner:
    """Pure global priority queue + adaptive-window policy (independently testable)."""

    def __init__(self, lower_bound: date, initial_days: int = 7):
        self.lower_bound = lower_bound
        self.initial_days = max(1, initial_days)
        self.windows: dict[str, DateWindow] = {}
        self._heap: list[tuple[int, int, int, str]] = []
        self._sequence = count()
        self._ids = count(1)

    def _new_window(
        self,
        chamber: int,
        start: date,
        end: date,
        *,
        continues: bool = True,
        parent_id: str | None = None,
    ) -> DateWindow:
        window = DateWindow(
            window_id=f"w{next(self._ids):06d}",
            chamber=chamber,
            start=max(start, self.lower_bound),
            end=end,
            continues=continues,
            parent_id=parent_id,
        )
        self.windows[window.window_id] = window
        heapq.heappush(
            self._heap,
            (
                -window.end.toordinal(),
                window.chamber,
                next(self._sequence),
                window.window_id,
            ),
        )
        return window

    def seed(self, chambers: tuple[int, ...], frontier: date) -> None:
        self.seed_cursors({chamber: frontier for chamber in chambers})

    def seed_cursors(self, cursors: dict[int, date | None]) -> None:
        """Seed independent chamber cursors into the shared newest-first heap."""
        for chamber in sorted(cursors):
            frontier = cursors[chamber]
            if frontier is None or frontier < self.lower_bound:
                continue
            start = max(
                self.lower_bound,
                frontier - timedelta(days=self.initial_days - 1),
            )
            self._new_window(chamber, start, frontier)

    def peek_end(self) -> date | None:
        if not self._heap:
            return None
        return date.fromordinal(-self._heap[0][0])

    def pop_at_end(self, frontier_end: date) -> DateWindow | None:
        if self.peek_end() != frontier_end:
            return None
        *_, window_id = heapq.heappop(self._heap)
        return self.windows[window_id]

    def split(self, window: DateWindow) -> tuple[DateWindow, DateWindow]:
        if window.days <= 1:
            raise ValueError("a one-day window cannot be narrowed")
        older_days = window.days // 2
        newer_start = window.start + timedelta(days=older_days)
        newer = self._new_window(
            window.chamber,
            newer_start,
            window.end,
            continues=False,
            parent_id=window.window_id,
        )
        older = self._new_window(
            window.chamber,
            window.start,
            newer_start - timedelta(days=1),
            continues=window.continues,
            parent_id=window.window_id,
        )
        window.status = "split"
        return newer, older

    @staticmethod
    def adapted_days(window: DateWindow, total: int) -> int:
        if total >= TARGET_MIN:
            return window.days
        if total <= 0:
            return min(MAX_WINDOW_DAYS, max(window.days + 1, window.days * 2))
        estimate = math.ceil(window.days * 25 / total)
        return min(
            MAX_WINDOW_DAYS,
            max(window.days + 1, min(window.days * 2, estimate)),
        )

    def schedule_older(self, window: DateWindow, total: int) -> DateWindow | None:
        if not window.continues:
            return None
        next_end = window.start - timedelta(days=1)
        if next_end < self.lower_bound:
            return None
        days = self.adapted_days(window, total)
        next_start = max(self.lower_bound, next_end - timedelta(days=days - 1))
        return self._new_window(window.chamber, next_start, next_end)


def parse_authoritative_total(response) -> int | None:
    text = " ".join(response.xpath("//text()").getall())
    match = _TOTAL_RE.search(text)
    if not match:
        return None
    return int(re.sub(r"[^0-9]", "", match.group(1)))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        os.chmod(tmp, 0o600)
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class SupremecourtSpider(BaseLegalSpider):
    name = "supremecourt"
    DEDUP_KEY = ("case_id", "chamber")
    partial_by_design = True
    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": 8,
        "RANDOMIZE_DOWNLOAD_DELAY": False,
        "AUTOTHROTTLE_ENABLED": False,
        "HTTPCACHE_ENABLED": False,
        "RETRY_TIMES": 8,
        "DOWNLOADER_MIDDLEWARES": {
            "legal_scrapers.middlewares.RotateUserAgentMiddleware": 543,
            "legal_scrapers.middlewares.SupremecourtRetryAfterMiddleware": 560,
        },
        "ITEM_PIPELINES": {
            "legal_scrapers.pipelines.SupremecourtDurablePipeline": 100,
        },
    }

    def __init__(self, *args, initial_window_days=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_window_days = max(1, int(initial_window_days))
        self._planner = NewestFirstPlanner(
            self.scraping_start_date, self.initial_window_days
        )
        self._ready_window_ids: set[str] = set()
        self._detail_batch_active = False
        self._identity_to_window: dict[str, str] = {}
        self._scheduled_identities: set[str] = set()
        self._known_encountered: set[str] = set()
        self._new_identities: set[str] = set()
        self._existing_items: dict[str, dict] = {}
        self._cumulative_items: dict[str, dict] = {}
        self._completed_intervals: dict[int, list[tuple[date, date]]] = {
            chamber: [] for chamber in CHAMBERS
        }
        self._unresolved: list[dict] = []
        self._unresolved_count = 0
        self._finish_reason: str | None = None
        self._artifacts_finalized = False
        self._run_ready = False
        self._reconcile_counts = (0, 0, 0)
        self._chamber_start_cursors: dict[int, date | None] = {
            chamber: self.scraping_end_date for chamber in CHAMBERS
        }
        self._resume_parent: dict | None = None
        # Spider.closed runs on the spider_closed signal before Scrapy's CoreStats handler,
        # so CoreStats has not populated elapsed_time_seconds when our final manifest is
        # materialized. Keep an independent monotonic clock for the atomic final artifact.
        self._started_monotonic = time.monotonic()

    def configure_run_outputs(self, settings):
        start_cursors, resume_parent = self._discover_resume_state()
        super().configure_run_outputs(settings)
        # This spider's fsync journal and atomic materializer replace FeedExporter.  Keeping
        # FEEDS enabled would duplicate items and recreate the seen-before-feed race.
        settings.set("FEEDS", {}, priority="spider")
        self.journal_path = self.run_dir / "items.journal.jsonl"
        self.partial_manifest_path = self.run_dir / "partial_manifest.json"
        self.latest_manifest_path = self.latest_dir / "partial_manifest.json"

        self._existing_items = self._collect_existing_items()
        self._cumulative_items = dict(self._existing_items)
        self._reconcile_seen_store()
        self._chamber_start_cursors = start_cursors
        self._resume_parent = resume_parent
        self._planner.seed_cursors(self._chamber_start_cursors)
        self._run_ready = True

    async def start(self):
        # Scrapy installs the stats collector after spider construction. Reconciliation
        # runs during construction, so expose its already-computed counters only now.
        exported, ghosts, missing = self._reconcile_counts
        self.crawler.stats.set_value("dedup/reconciled_exported", exported)
        self.crawler.stats.set_value("dedup/ghosts_removed", ghosts)
        self.crawler.stats.set_value("dedup/exported_keys_added", missing)
        if not self._run_ready:
            # Direct construction is useful in unit tests; real crawls always pass through
            # configure_run_outputs/from_crawler.
            self._planner.seed_cursors(self._chamber_start_cursors)
        for request in self._advance_frontier():
            yield request

    @classmethod
    def _partial_validator(cls):
        """Load the stdlib-only offline gate from an execution-context-independent path."""
        # ARTIFACTS_ROOT is intentionally replaceable in tests and deployments, so it
        # cannot also be used to locate source code.  Anchor the validator to this module's
        # checked-out repository instead.
        path = REPOSITORY_ROOT / "ingest/scripts/validate_supremecourt_partial.py"
        module_name = "_georgian_legal_supremecourt_partial_validator"
        loaded = sys.modules.get(module_name)
        if loaded is not None and Path(getattr(loaded, "__file__", "")) == path:
            return loaded
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load Supreme Court resume validator from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _discover_resume_state(self) -> tuple[dict[int, date | None], dict | None]:
        """Return independently verified prior cursors, or a deliberate fresh seed.

        Only run-scoped final manifests participate; ``latest`` and live partial manifests
        have no final item digest and are ignored.  Once a newest final candidate exists it
        is validated before scope compatibility is considered, so corruption can never make
        the crawler silently fall back to an older or fresh frontier.
        """

        fresh = {chamber: self.scraping_end_date for chamber in CHAMBERS}
        runs_root = self.ARTIFACTS_ROOT / self.name / "runs"
        if not runs_root.is_dir():
            return fresh, None

        finalized: list[tuple[str, Path, bytes, dict]] = []
        for path in sorted(runs_root.glob("*/partial_manifest.json")):
            try:
                payload = path.read_bytes()
                value = json.loads(payload)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot inspect prior Supreme Court manifest {path}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"prior Supreme Court manifest is not an object: {path}")
            # Atomic live manifests have both fields null. Either non-null field is final
            # evidence (or a corrupt attempted finalization that must fail closed below).
            if value.get("finished_at") is None and value.get("items_sha256") is None:
                continue
            finalized.append((path.parent.name, path, payload, value))
        if not finalized:
            return fresh, None

        _run_name, manifest_path, payload, raw_manifest = max(
            finalized, key=lambda entry: entry[0]
        )
        validator = self._partial_validator()
        try:
            report = validator.validate_run(manifest_path.parent)
        except Exception as exc:
            raise RuntimeError(
                "newest finalized Supreme Court resume manifest failed validation: "
                f"{manifest_path}: {exc}"
            ) from exc

        # A different requested date scope intentionally starts fresh; using its cursor would
        # skip dates outside the parent ledger. The newest final candidate was still fully
        # validated above, and an older candidate is never substituted.
        if (
            raw_manifest.get("lower_bound") != self.scraping_start_date.isoformat()
            or raw_manifest.get("frontier_start_date")
            != self.scraping_end_date.isoformat()
        ):
            self.logger.info(
                "supremecourt: newest final run %s has a different date scope; "
                "starting fresh",
                manifest_path.parent.name,
            )
            return fresh, None

        official_to_chamber = {name: int(value) for value, name in CHAMBER_NAMES.items()}
        derived = report.get("per_chamber_resume_cursors")
        if not isinstance(derived, dict) or set(derived) != set(official_to_chamber):
            raise RuntimeError("validated parent report has no exact chamber cursor ledger")
        cursors: dict[int, date | None] = {}
        for official, chamber in official_to_chamber.items():
            raw_cursor = derived[official]
            if raw_cursor is None:
                cursors[chamber] = None
                continue
            try:
                cursor = date.fromisoformat(raw_cursor)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"validated parent returned invalid cursor {raw_cursor!r}"
                ) from exc
            if not self.scraping_start_date <= cursor <= self.scraping_end_date:
                raise RuntimeError(
                    f"validated parent cursor {cursor} lies outside requested bounds"
                )
            cursors[chamber] = cursor

        parent = {
            "run_id": manifest_path.parent.name,
            "manifest_file": str(manifest_path.absolute()),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "items_sha256": report["items_sha256"],
        }
        self.logger.info(
            "supremecourt: resuming from validated run %s at %s",
            manifest_path.parent.name,
            {CHAMBER_NAMES[str(chamber)]: value for chamber, value in cursors.items()},
        )
        return cursors, parent

    # ------------------------------------------------------------------ requests
    def request_page(self, palata, page):
        """Legacy broad-window request retained for existing direct parser tests."""
        query = {
            "palata": palata,
            "page": page,
            "tarigiDan": iso_to_year_slashed(self.scraping_start_date),
            "tarigiMde": iso_to_year_slashed(self.scraping_end_date),
        }
        return Request(
            f"{GETCASES_URL}?{urlencode(query)}",
            callback=self.parse_list,
            errback=self.request_failed,
            headers={"X-Requested-With": "XMLHttpRequest"},
            meta={
                "palata": palata,
                "page": page,
                "kind": "legacy_list",
                "dont_cache": True,
            },
            dont_filter=True,
        )

    def request_window(self, window: DateWindow, page: int = 1) -> Request:
        query = {
            "palata": window.chamber,
            "page": page,
            "tarigiDan": iso_to_year_slashed(window.start),
            "tarigiMde": iso_to_year_slashed(window.end),
        }
        return Request(
            f"{GETCASES_URL}?{urlencode(query)}",
            callback=self.parse_list,
            errback=self.request_failed,
            headers={"X-Requested-With": "XMLHttpRequest"},
            meta={
                "palata": window.chamber,
                "page": page,
                "kind": "window_list",
                "window_id": window.window_id,
                "dont_cache": True,
            },
            priority=window.end.toordinal() * 100_000 + 90_000,
            dont_filter=True,
        )

    def _advance_frontier(self) -> list[Request]:
        """Drive probes/details while proving no undiscovered newer decision can exist.

        A split chamber can leave a pending window ending 2026-07-10 while another chamber's
        already-complete broad window contains cards down to 2026-07-07.  Dispatching all of
        those cards immediately would still violate decision-level newest-first.  The safe
        cutoff is therefore the newest *real or virtual* unprobed window end.  A ready,
        continuing window has an adjacent older successor that cannot be queued until the
        current window is durably completed; treating that successor as a virtual probe keeps
        a wider peer window from emitting older cards in the meantime.  Only cards strictly
        newer than the cutoff may run.  Cards at/below it wait and are combined with the newly
        discovered cards after the next probe.
        """
        if self._detail_batch_active:
            return []

        while True:
            self._settle_ready_windows()
            next_probe_end = self._planner.peek_end()
            virtual_successor_ends = [
                window.start - timedelta(days=1)
                for window_id in self._ready_window_ids
                if (window := self._planner.windows[window_id]).continues
                and window.start > self._planner.lower_bound
            ]
            cutoff_candidates = [
                candidate
                for candidate in [next_probe_end, *virtual_successor_ends]
                if candidate is not None
            ]
            safe_cutoff = max(cutoff_candidates, default=None)
            safe_cards = []
            for window_id in self._ready_window_ids:
                window = self._planner.windows[window_id]
                for card in window.cards:
                    if card.get("dispatched"):
                        continue
                    if safe_cutoff is None or card["decision_date"] > safe_cutoff:
                        safe_cards.append(card)

            safe_cards.sort(
                key=lambda card: (
                    card["decision_date"],
                    -int(card["palata"]),
                    card["fields"]["case_id"],
                ),
                reverse=True,
            )
            requests: list[Request] = []
            for order, card in enumerate(safe_cards):
                card["dispatched"] = True
                identity = card["identity"]
                window = self._planner.windows[card["window_id"]]
                if identity in self._scheduled_identities or self.is_seen(
                    card["fields"]
                ):
                    window.known_identities.add(identity)
                    self._known_encountered.add(identity)
                    continue
                self._scheduled_identities.add(identity)
                self._identity_to_window[identity] = window.window_id
                window.pending_identities.add(identity)
                priority = card["decision_date"].toordinal() * 100_000 + (
                    80_000 - order
                )
                requests.append(
                    Request(
                        card["url"],
                        callback=self.parse_detail,
                        errback=self.request_failed,
                        meta={
                            "fields": card["fields"],
                            "palata": card["palata"],
                            "kind": "detail",
                            "window_id": window.window_id,
                            "identity": identity,
                            "dont_cache": True,
                        },
                        priority=priority,
                        dont_filter=True,
                    )
                )
            if requests:
                self._detail_batch_active = True
                return requests
            if safe_cards:
                # They were all known/duplicate and are now handled; settle and recalculate.
                continue

            if next_probe_end is not None:
                window = self._planner.pop_at_end(next_probe_end)
                if window is not None:
                    window.status = "probing"
                    return [self.request_window(window)]

            # No unprobed windows. Any remaining card is therefore safe (the next loop sees
            # next_pending_end=None); if none remain, the bounded corpus range is complete.
            remaining = any(
                not card.get("dispatched")
                for window_id in self._ready_window_ids
                for card in self._planner.windows[window_id].cards
            )
            if remaining:
                continue
            self._settle_ready_windows()
            return []

    # ------------------------------------------------------------------ list parse
    def parse_list(self, response):
        if "window_id" not in response.meta:
            yield from self._parse_legacy_list(response)
            return

        window = self._planner.windows[response.meta["window_id"]]
        page = int(response.meta["page"])
        if response.status != 200:
            self._window_failure(
                window, "list_http", f"HTTP {response.status}", response.url
            )
            self._mark_terminal_ready(window)
            yield from self._advance_frontier()
            return

        total = parse_authoritative_total(response)
        if total is None:
            retry = self._parse_retry(
                response.request, window, "authoritative total missing"
            )
            if retry is not None:
                yield retry
                return
            self._mark_terminal_ready(window)
            yield from self._advance_frontier()
            return

        if page == 1:
            window.authoritative_total = total
            if total > TARGET_MAX and window.days > 1:
                self._planner.split(window)
                yield from self._advance_frontier()
                return
            window.expected_pages = max(1, math.ceil(total / PAGE_SIZE))
            if window.expected_pages > MAX_PAGES:
                self._window_failure(
                    window,
                    "max_pages",
                    f"expected {window.expected_pages} pages exceeds {MAX_PAGES}",
                    response.url,
                )
                self._mark_terminal_ready(window)
                yield from self._advance_frontier()
                return
        elif total != window.authoritative_total:
            self._window_failure(
                window,
                "total_changed",
                f"page 1 total={window.authoritative_total}, page {page} total={total}",
                response.url,
            )

        parsed_dates: list[date] = []
        for card_index, case in enumerate(response.css("div.cases")):
            parsed = self._parse_case_card(case, expected_chamber=window.chamber)
            if parsed is None:
                self._window_failure(
                    window,
                    "invalid_card",
                    f"unparseable card on page {page}",
                    response.url,
                )
                continue
            fields, href, palata, decision_date = parsed
            identity = f"{fields['case_id']}:{fields['chamber']}"
            window.listed_identities.add(identity)
            parsed_dates.append(decision_date)
            if not (window.start <= decision_date <= window.end):
                self._window_failure(
                    window,
                    "date_outside_window",
                    f"{identity} date={decision_date} outside {window.start}..{window.end}",
                    response.url,
                )
                continue
            window.cards.append(
                {
                    "identity": identity,
                    "fields": fields,
                    "url": response.urljoin(href),
                    "palata": palata,
                    "decision_date": decision_date,
                    "window_id": window.window_id,
                    "page": page,
                    "card_index": card_index,
                    "dispatched": False,
                }
            )

        if parsed_dates != sorted(parsed_dates, reverse=True):
            self._window_failure(
                window, "date_order", f"page {page} is not newest-first", response.url
            )
        window.pages_seen.add(page)

        if page < window.expected_pages:
            yield self.request_window(window, page + 1)
            return

        expected = window.authoritative_total or 0
        if len(window.listed_identities) != expected:
            self._window_failure(
                window,
                "card_count_mismatch",
                f"authoritative total={expected}, unique cards={len(window.listed_identities)}",
                response.url,
            )
        self._mark_terminal_ready(window)
        yield from self._advance_frontier()

    def _parse_case_card(self, case, expected_chamber: int | None = None):
        href = case.css("a[href*='/fullcase/']::attr(href)").get()
        match = _FULLCASE_RE.search(href or "")
        if not match:
            return None
        case_id, case_palata = match.group(1), match.group(2)
        if case_palata not in CHAMBER_NAMES:
            return None
        if expected_chamber is not None and int(case_palata) != expected_chamber:
            return None

        fields = {"case_id": case_id, "chamber": CHAMBER_NAMES[case_palata]}
        for child in case.xpath("./*"):
            label = (
                "".join(child.xpath("./span//text()").getall())
                .strip()
                .rstrip(":")
                .strip()
            )
            target = CASE_LABELS.get(label)
            if target:
                value = "".join(child.xpath("./text()").getall()).strip()
                if value:
                    fields[target] = value
        try:
            decision_date = date.fromisoformat(fields.get("date", ""))
        except ValueError:
            return None
        return fields, href, case_palata, decision_date

    def _parse_legacy_list(self, response):
        """Compatibility parser for existing direct tests; production uses window metadata."""
        palata = response.meta["palata"]
        page = response.meta["page"]
        cases = response.css("div.cases")
        for case in cases:
            parsed = self._parse_case_card(case)
            if parsed is None:
                continue
            fields, href, case_palata, _ = parsed
            if self.is_seen(fields):
                continue
            yield response.follow(
                href,
                callback=self.parse_detail,
                errback=self.request_failed,
                meta={"fields": fields, "palata": case_palata, "kind": "legacy_detail"},
            )
        if cases and page < MAX_PAGES:
            yield self.request_page(palata, page + 1)

    def _parse_retry(self, request, window: DateWindow, detail: str) -> Request | None:
        attempts = int(request.meta.get("parse_retry_times", 0))
        if attempts < 2:
            retry = request.replace(dont_filter=True, priority=request.priority + 1)
            retry.meta["parse_retry_times"] = attempts + 1
            self.crawler.stats.inc_value("retry/parse")
            return retry
        self._window_failure(window, "list_parse", detail, request.url)
        return None

    def _mark_terminal_ready(self, window: DateWindow) -> None:
        window.status = "ready_for_details"
        self._ready_window_ids.add(window.window_id)

    # ---------------------------------------------------------------- detail/item
    def parse_detail(self, response):
        fields = response.meta["fields"]
        case_id = fields["case_id"]
        palata = str(response.meta["palata"])
        identity = response.meta.get("identity") or f"{case_id}:{fields['chamber']}"

        node = response.css("div.case-single#modalBody")
        body_html = node.get() if len(node) == 1 else None
        body_markdown = (
            safe_html_to_markdown(body_html, base_url=BASE, source_url=response.url)
            if body_html
            else ""
        )
        if not body_markdown or not body_markdown.strip():
            attempts = int(response.meta.get("detail_parse_retry_times", 0))
            if attempts < 2:
                retry = response.request.replace(
                    dont_filter=True, priority=response.request.priority + 1
                )
                retry.meta["detail_parse_retry_times"] = attempts + 1
                self.crawler.stats.inc_value("retry/detail_parse")
                yield retry
                return
            self.record_quality_failure(
                "empty_fullcase",
                response.url,
                detail="missing/empty unique div.case-single#modalBody after retries",
                context={"identity": identity},
            )
            if "window_id" in response.meta:
                self.item_failed(
                    identity, "empty_fullcase", "full decision body is empty"
                )
            return

        loader = ItemLoader(item=SupremecourtItem())
        loader.add_value("source_url", response.url)
        for key, value in fields.items():
            loader.add_value(key, value)
        loader.add_value("docx_url", f"{BASE}/ka/download/{case_id}/{palata}")
        loader.add_value("body_markdown", body_markdown)
        yield loader.load_item()

    def persist_item(self, item: dict, identity: str) -> None:
        """Append+fsync a complete item before the pipeline is allowed to mark it seen."""
        line = (
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.journal_path, flags, 0o600)
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode) & 0o077:
            os.close(descriptor)
            raise PermissionError(
                "Supreme Court journal is not an owner-only regular file; "
                "refusing to chmod existing evidence"
            )
        with os.fdopen(descriptor, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._cumulative_items[identity] = item
        self._new_identities.add(identity)
        self.crawler.stats.inc_value("supremecourt/new_items")

    def item_persisted(self, identity: str) -> None:
        window_id = self._identity_to_window.pop(identity, None)
        if window_id is None:
            return
        window = self._planner.windows[window_id]
        window.pending_identities.discard(identity)
        window.new_identities.add(identity)
        self._maybe_finish_detail_phase()

    def item_failed(self, identity: str | None, kind: str, detail: str) -> None:
        window_id = self._identity_to_window.pop(identity, None) if identity else None
        if window_id is None:
            return
        window = self._planner.windows[window_id]
        window.pending_identities.discard(identity)
        self._window_failure(window, kind, detail, "")
        self._maybe_finish_detail_phase()

    def _maybe_finish_detail_phase(self) -> None:
        if not self._detail_batch_active:
            return
        if any(
            self._planner.windows[window_id].pending_identities
            for window_id in self._ready_window_ids
        ):
            return
        self._detail_batch_active = False
        self._enqueue(self._advance_frontier())

    def _settle_ready_windows(self) -> None:
        settled = []
        for window_id in sorted(self._ready_window_ids):
            window = self._planner.windows[window_id]
            if window.pending_identities or any(
                not card.get("dispatched") for card in window.cards
            ):
                continue
            expected = window.authoritative_total or 0
            complete = (
                window.failure_count == 0
                and len(window.pages_seen) == window.expected_pages
                and len(window.listed_identities) == expected
                and not window.pending_identities
            )
            window.status = "completed" if complete else "incomplete"
            if complete:
                self._completed_intervals[window.chamber].append(
                    (window.start, window.end)
                )
            self._planner.schedule_older(window, expected)
            settled.append(window_id)

        for window_id in settled:
            self._ready_window_ids.discard(window_id)
        # Cursor/window completion is durable before any older request is returned to Scrapy.
        if settled and self._run_ready:
            self._write_manifest(final=False)

    def _enqueue(self, requests: list[Request]) -> None:
        engine = getattr(getattr(self, "crawler", None), "engine", None)
        if engine is None:
            return
        for request in requests:
            try:
                engine.crawl(request)
            except RuntimeError:
                # Expected when CloseSpider has already begun a graceful shutdown.
                return

    def request_failed(self, failure):
        request = failure.request
        super().request_failed(failure)
        kind = request.meta.get("kind")
        if kind == "detail":
            self.item_failed(
                request.meta.get("identity"),
                "detail_http",
                str(failure.value)[:300],
            )
            return None
        window_id = request.meta.get("window_id")
        if window_id:
            window = self._planner.windows[window_id]
            self._window_failure(
                window, "list_http", str(failure.value)[:300], request.url
            )
            self._mark_terminal_ready(window)
            self._enqueue(self._advance_frontier())
        return None

    def _window_failure(
        self, window: DateWindow, kind: str, detail: str, url: str
    ) -> None:
        record = {
            "window_id": window.window_id,
            "chamber": CHAMBER_NAMES[str(window.chamber)],
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "kind": kind,
            "detail": detail[:300],
            "url": url,
        }
        window.failure_count += 1
        if len(window.failures) < MAX_FAILURE_EXAMPLES:
            window.failures.append(record)
        self._unresolved_count += 1
        if len(self._unresolved) < MAX_FAILURE_EXAMPLES:
            self._unresolved.append(record)
        crawler = getattr(self, "crawler", None)
        if crawler is not None:
            crawler.stats.inc_value("quality/failures")
            crawler.stats.inc_value(f"quality/{kind}")

    # -------------------------------------------------------- reconciliation/artifacts
    @staticmethod
    def _identity(item: dict) -> str | None:
        case_id = str(item.get("case_id") or "").strip()
        chamber = str(item.get("chamber") or "").strip()
        if not case_id or chamber not in CHAMBER_NAMES.values():
            return None
        return f"{case_id}:{chamber}"

    def _collect_existing_items(self) -> dict[str, dict]:
        root = self.ARTIFACTS_ROOT / self.name
        paths = sorted(root.glob("runs/*/items.jsonl"))
        paths += sorted(root.glob("runs/*/items.journal.jsonl"))
        latest = root / "latest" / "items.jsonl"
        if latest.exists():
            paths.append(latest)
        items: dict[str, dict] = {}
        for path in paths:
            try:
                fh = path.open(encoding="utf-8")
            except OSError:
                continue
            with fh:
                for line in fh:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    identity = self._identity(item)
                    if identity and str(item.get("body_markdown") or "").strip():
                        items[identity] = item
        return items

    def _reconcile_seen_store(self) -> None:
        if not self.dedup_enabled or self._dedup_conn is None:
            return
        exported = set(self._existing_items)
        seen = set(self._seen_keys)
        ghosts = sorted(seen - exported)
        missing = sorted(exported - seen)
        if ghosts:
            self._dedup_conn.executemany(
                "DELETE FROM seen WHERE key = ?", ((x,) for x in ghosts)
            )
        if missing:
            timestamp = datetime.now(UTC).isoformat()
            self._dedup_conn.executemany(
                "INSERT OR IGNORE INTO seen (key, run_id, ts) VALUES (?, ?, ?)",
                ((x, "reconciled-export", timestamp) for x in missing),
            )
        self._dedup_conn.commit()
        self._seen_keys = set(exported)
        self._reconcile_counts = (len(exported), len(ghosts), len(missing))
        stats = getattr(getattr(self, "crawler", None), "stats", None)
        if stats is not None:
            stats.set_value("dedup/reconciled_exported", len(exported))
            stats.set_value("dedup/ghosts_removed", len(ghosts))
            stats.set_value("dedup/exported_keys_added", len(missing))
        if ghosts:
            self.logger.warning(
                "supremecourt: removed %d ghost seen key(s)", len(ghosts)
            )

    def _contiguous_cursor(self, chamber: int) -> tuple[date | None, date | None]:
        start_cursor = self._chamber_start_cursors[chamber]
        if start_cursor is None:
            return None, self.scraping_start_date
        cursor = start_cursor
        intervals = self._completed_intervals[chamber]
        while cursor >= self.scraping_start_date:
            covering = [start for start, end in intervals if start <= cursor <= end]
            if not covering:
                break
            cursor = min(covering) - timedelta(days=1)
        if cursor < self.scraping_start_date:
            return None, self.scraping_start_date
        if cursor == self.scraping_end_date and self._resume_parent is None:
            return cursor, None
        return cursor, cursor + timedelta(days=1)

    def _sorted_cumulative(self) -> list[dict]:
        def key(item):
            try:
                decision_date = date.fromisoformat(str(item.get("date") or ""))
            except ValueError:
                decision_date = date.min
            chamber = str(item.get("chamber") or "")
            return decision_date, chamber, str(item.get("case_id") or "")

        return sorted(self._cumulative_items.values(), key=key, reverse=True)

    def _materialize_items(self) -> tuple[int, str]:
        payload = b"".join(
            (
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
            for item in self._sorted_cumulative()
        )
        _atomic_write(self.items_path, payload)
        _atomic_write(self.latest_items_path, payload)
        return len(self._cumulative_items), hashlib.sha256(payload).hexdigest()

    def _manifest(self, *, final: bool, items_sha256: str | None = None) -> dict:
        crawler = getattr(self, "crawler", None)
        stats = getattr(crawler, "stats", None)
        stats_dict = stats.get_stats() if stats is not None else {}
        settings = getattr(crawler, "settings", None)
        configured_runtime = (
            settings.getint("CLOSESPIDER_TIMEOUT", 0) if settings else 0
        )
        elapsed_time_seconds = stats_dict.get("elapsed_time_seconds")
        if final and not isinstance(elapsed_time_seconds, (int, float)):
            elapsed_time_seconds = time.monotonic() - self._started_monotonic
        per_chamber = {}
        oldest_frontiers: list[date] = []
        resume_cursors = {}
        for chamber in CHAMBERS:
            official = CHAMBER_NAMES[str(chamber)]
            cumulative = [
                item
                for item in self._cumulative_items.values()
                if item.get("chamber") == official
            ]
            new = [
                identity
                for identity in self._new_identities
                if identity.endswith(f":{official}")
            ]
            known = [
                identity
                for identity in self._existing_items
                if identity.endswith(f":{official}")
            ]
            dates = []
            for item in cumulative:
                try:
                    dates.append(date.fromisoformat(str(item.get("date") or "")))
                except ValueError:
                    pass
            cursor, oldest = self._contiguous_cursor(chamber)
            resume_cursors[official] = cursor.isoformat() if cursor else None
            if oldest is not None:
                oldest_frontiers.append(oldest)
            per_chamber[official] = {
                "known_items": len(known),
                "new_items": len(new),
                "total_items": len(cumulative),
                "newest_date": max(dates).isoformat() if dates else None,
                "oldest_date": min(dates).isoformat() if dates else None,
                "resume_cursor": cursor.isoformat() if cursor else None,
            }

        global_oldest = (
            max(oldest_frontiers).isoformat()
            if len(oldest_frontiers) == len(CHAMBERS)
            else None
        )
        windows = []
        for window in self._planner.windows.values():
            if window.status in {"completed", "incomplete", "split"}:
                windows.append(
                    {
                        "id": window.window_id,
                        "chamber": CHAMBER_NAMES[str(window.chamber)],
                        "start": window.start.isoformat(),
                        "end": window.end.isoformat(),
                        "status": window.status,
                        "authoritative_total": window.authoritative_total,
                        "known": len(window.known_identities),
                        "new": len(window.new_identities),
                        "failures": window.failure_count,
                    }
                )

        return {
            "schema_version": 1,
            "run_id": getattr(self, "run_id", None),
            "started_at": getattr(self, "started_at", datetime.now(UTC)).isoformat(),
            "finished_at": datetime.now(UTC).isoformat() if final else None,
            "finish_reason": self._finish_reason if final else None,
            "partial_by_design": True,
            "date_order": "newest_first",
            "frontier_start_date": self.scraping_end_date.isoformat(),
            "lower_bound": self.scraping_start_date.isoformat(),
            "per_chamber_start_cursors": {
                CHAMBER_NAMES[str(chamber)]: (
                    self._chamber_start_cursors[chamber].isoformat()
                    if self._chamber_start_cursors[chamber] is not None
                    else None
                )
                for chamber in CHAMBERS
            },
            "resume_parent": self._resume_parent,
            "max_runtime_seconds": configured_runtime or None,
            "elapsed_time_seconds": elapsed_time_seconds if final else None,
            "items_file": str(getattr(self, "items_path", "items.jsonl")),
            "journal_file": str(getattr(self, "journal_path", "items.journal.jsonl")),
            "items_sha256": items_sha256,
            "known_items": len(self._existing_items),
            "new_items": len(self._new_identities),
            "total_items": len(self._cumulative_items),
            "known_encountered": len(self._known_encountered),
            "per_chamber": per_chamber,
            "oldest_fully_completed_global_date_frontier": global_oldest,
            "per_chamber_resume_cursors": resume_cursors,
            "completed_windows": windows,
            "retries": {
                "total": int(stats_dict.get("retry/count", 0)),
                "retry_after": int(stats_dict.get("retry_after/count", 0)),
                "parse": int(stats_dict.get("retry/parse", 0)),
                "detail_parse": int(stats_dict.get("retry/detail_parse", 0)),
            },
            "unresolved_failure_count": self._unresolved_count,
            "unresolved_failures_truncated": (
                self._unresolved_count > len(self._unresolved)
            ),
            "unresolved_failures": self._unresolved,
        }

    def _write_manifest(self, *, final: bool, items_sha256: str | None = None) -> None:
        manifest = self._manifest(final=final, items_sha256=items_sha256)
        payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        _atomic_write(self.partial_manifest_path, payload)
        _atomic_write(self.latest_manifest_path, payload)

    def closed(self, reason):
        if self._artifacts_finalized or not self._run_ready:
            return
        self._artifacts_finalized = True
        self._finish_reason = reason
        _, digest = self._materialize_items()
        self._write_manifest(final=True, items_sha256=digest)
        if getattr(self, "_dedup_conn", None) is not None:
            self._dedup_conn.close()
