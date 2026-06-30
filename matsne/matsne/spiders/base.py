"""Shared base spider: per-run artifact scaffolding + date-range arguments.

Every spider in this project writes one JSONL file per run under
``artifacts/<spider name>/runs/<run_id>/`` plus a ``latest/`` pointer, and accepts
``start_date``/``end_date`` arguments in ``YYYY-MM-DD`` form. That wiring originally
lived in the matsne spider; it is entirely generic, so it is factored out here for
every spider to inherit.
"""

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import scrapy


class BaseLegalSpider(scrapy.Spider):
    # Subclasses set ``name``. Override this to change the default crawl-window start
    # used when no ``start_date`` argument is given.
    DEFAULT_SCRAPING_START_DATE = date(2026, 6, 22)

    # Cross-run dedup identity: the item field name(s) that uniquely identify a
    # document, mirroring the downstream ingest ``id_fields`` so "already scraped"
    # means "already in the vector DB under this id". Subclasses override this; the
    # default ``None`` disables dedup for that spider.
    DEDUP_KEY: tuple[str, ...] | None = None

    # Safe defaults so is_seen/dedup_key are inert until ``open_dedup_store`` runs
    # (e.g. in unit tests that call parse callbacks without ``from_crawler``).
    dedup_enabled = False
    _seen_keys: set = frozenset()

    # base.py lives at <repo>/matsne/matsne/spiders/base.py, so parents[3] == <repo>.
    # Each spider writes under ARTIFACTS_ROOT / <name> / ... (see configure_run_outputs).
    ARTIFACTS_ROOT = Path(__file__).resolve().parents[3] / "artifacts"

    def __init__(self, start_date=None, end_date=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scraping_start_date = self.parse_date_arg(
            start_date,
            "start_date",
            self.DEFAULT_SCRAPING_START_DATE,
        )
        self.scraping_end_date = self.parse_date_arg(end_date, "end_date", date.today())

        if self.scraping_start_date > self.scraping_end_date:
            raise ValueError(
                "start_date must be on or before end_date "
                f"({self.scraping_start_date.isoformat()} > "
                f"{self.scraping_end_date.isoformat()})"
            )

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.configure_run_outputs(crawler.settings)
        return spider

    def request_failed(self, failure):
        """Default errback: surface transport-level failures in the spider log.

        matsne overrides this to also drive its two-phase bookkeeping; the other
        spiders use this log-only default. Wire it via ``errback=self.request_failed``.
        """
        self.logger.warning("Request failed: %s", failure.request.url)

    @classmethod
    def parse_date_arg(cls, value, argument_name: str, default: date) -> date:
        if value is None or value == "":
            return default

        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{argument_name} must use YYYY-MM-DD format, got {value!r}"
            ) from exc

    def configure_run_outputs(self, settings):
        self.started_at = datetime.now(UTC).replace(microsecond=0)
        self.run_id = self.build_run_id()
        self.run_dir = self.ARTIFACTS_ROOT / self.name / "runs" / self.run_id
        self.latest_dir = self.ARTIFACTS_ROOT / self.name / "latest"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

        self.items_path = self.run_dir / "items.jsonl"
        self.latest_items_path = self.latest_dir / "items.jsonl"
        self.log_path = self.run_dir / "spider.log"
        self.run_metadata_path = self.run_dir / "run.json"
        self.latest_metadata_path = self.latest_dir / "run.json"

        settings.set(
            "FEEDS",
            {
                str(self.items_path): self.feed_options(),
                str(self.latest_items_path): self.feed_options(),
            },
            priority="spider",
        )
        settings.set("LOG_FILE", str(self.log_path), priority="spider")

        self.open_dedup_store(settings)
        self.write_run_metadata()

    # --- cross-run deduplication -------------------------------------------

    def open_dedup_store(self, settings):
        """Open the per-spider SQLite seen-store and load known keys into memory.

        Dedup is keyed on ``DEDUP_KEY`` (the same identity ingest uses), so a
        document already scraped in a previous run is skipped before its detail
        page is fetched. Disabled when ``DEDUP_ENABLED`` is False or the spider
        sets no ``DEDUP_KEY`` — in which case ``is_seen`` is always False.
        """
        self.dedup_enabled = bool(self.DEDUP_KEY) and settings.getbool(
            "DEDUP_ENABLED", True
        )
        self._seen_keys: set[str] = set()
        self._dedup_conn = None
        if not self.dedup_enabled:
            return

        self.dedup_db_path = self.ARTIFACTS_ROOT / self.name / "seen.sqlite"
        self.dedup_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._dedup_conn = sqlite3.connect(str(self.dedup_db_path))
        self._dedup_conn.execute(
            "CREATE TABLE IF NOT EXISTS seen ("
            "key TEXT PRIMARY KEY, run_id TEXT, ts TEXT)"
        )
        self._dedup_conn.commit()
        self._seen_keys = {
            row[0] for row in self._dedup_conn.execute("SELECT key FROM seen")
        }
        self.logger.info(
            "dedup: loaded %d seen key(s) from %s",
            len(self._seen_keys),
            self.dedup_db_path,
        )

    def dedup_key(self, mapping) -> str | None:
        """Build the dedup identity from ``DEDUP_KEY`` fields of a dict/item.

        Returns ``None`` if dedup is off or any key field is missing/empty, so we
        fail open and never skip a document on incomplete data.
        """
        if not self.dedup_enabled:
            return None
        parts = []
        for field_name in self.DEDUP_KEY:
            value = mapping.get(field_name)
            if value is None or value == "":
                return None
            parts.append(str(value).strip())
        return ":".join(parts)

    def is_seen(self, mapping) -> bool:
        key = self.dedup_key(mapping)
        return key is not None and key in self._seen_keys

    def mark_seen(self, mapping) -> bool:
        """Record a document as scraped. Returns True if it was newly added."""
        key = self.dedup_key(mapping)
        if key is None or key in self._seen_keys:
            return False
        self._seen_keys.add(key)
        if self._dedup_conn is not None:
            self._dedup_conn.execute(
                "INSERT OR IGNORE INTO seen (key, run_id, ts) VALUES (?, ?, ?)",
                (key, self.run_id, datetime.now(UTC).isoformat()),
            )
            self._dedup_conn.commit()
        return True

    def build_run_id(self) -> str:
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        base_run_id = (
            f"{timestamp}_start-{self.scraping_start_date.isoformat()}"
            f"_end-{self.scraping_end_date.isoformat()}"
        )
        run_id = base_run_id
        suffix = 2
        while (self.ARTIFACTS_ROOT / self.name / "runs" / run_id).exists():
            run_id = f"{base_run_id}-{suffix}"
            suffix += 1
        return run_id

    @staticmethod
    def feed_options() -> dict:
        return {
            "format": "jsonlines",
            "encoding": "utf8",
            "store_empty": False,
            "overwrite": True,
        }

    def write_run_metadata(self):
        metadata = {
            "run_id": self.run_id,
            "spider": self.name,
            "start_date": self.scraping_start_date.isoformat(),
            "end_date": self.scraping_end_date.isoformat(),
            "started_at": self.started_at.isoformat(),
            "items_path": str(self.items_path),
            "latest_items_path": str(self.latest_items_path),
            "log_path": str(self.log_path),
        }
        metadata_json = json.dumps(metadata, indent=2) + "\n"
        self.run_metadata_path.write_text(metadata_json, encoding="utf-8")
        self.latest_metadata_path.write_text(metadata_json, encoding="utf-8")
