"""Shared base spider: per-run artifact scaffolding + date-range arguments.

Every spider in this project writes one JSONL file per run under
``artifacts/<spider name>/runs/<run_id>/`` plus a ``latest/`` pointer, and accepts
``start_date``/``end_date`` arguments in ``YYYY-MM-DD`` form. That wiring originally
lived in the matsne spider; it is entirely generic, so it is factored out here for
every spider to inherit.
"""

import hashlib
import json
import os
import sqlite3
from datetime import UTC, date, datetime, timedelta
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
    MAX_REFRESHES_PER_RUN = 2_000
    _SUCCESS_OUTCOMES = frozenset({"success", "legacy_success"})
    _PENDING_MARKERS = (
        "pending",
        "draft",
        "ასამოქმედებელ",
        "მოლოდინ",
        "პროექტ",
    )

    # Safe defaults so is_seen/dedup_key are inert until ``open_dedup_store`` runs
    # (e.g. in unit tests that call parse callbacks without ``from_crawler``).
    dedup_enabled = False
    _seen_keys: set = frozenset()

    # base.py lives at <repo>/scraper/legal_scrapers/spiders/base.py, so parents[3] == <repo>.
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
        request = failure.request
        context = {}
        fields = request.meta.get("fields")
        if isinstance(fields, dict) and self.DEDUP_KEY:
            context = {key: fields.get(key) for key in self.DEDUP_KEY if fields.get(key)}
        self.record_quality_failure(
            "request_failed",
            request.url,
            detail=str(failure.value)[:300],
            context=context,
        )

    def record_quality_failure(
        self,
        kind: str,
        url: str,
        *,
        detail: str = "",
        context: dict | None = None,
    ) -> None:
        """Record a completeness-affecting failure for the runner and repair tooling."""
        crawler = getattr(self, "crawler", None)
        if crawler is not None:
            crawler.stats.inc_value("quality/failures")
            crawler.stats.inc_value(f"quality/{kind}")
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "kind": kind,
            "url": url,
            "detail": detail[:300],
            "context": context or {},
        }
        run_dir = getattr(self, "run_dir", None)
        if run_dir is not None:
            with (run_dir / "failures.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        # Artifacts can contain public-record personal data and local crawl diagnostics.
        # Owner-only creation is process-wide so Scrapy's feed/log/cache writers inherit it.
        os.umask(0o077)
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
        self.open_dedup_store(settings)
        self.write_run_metadata()

    # --- cross-run deduplication -------------------------------------------

    def open_dedup_store(self, settings):
        """Open/migrate the per-spider success store and select bounded refreshes.

        Dedup is keyed on ``DEDUP_KEY`` (the same identity ingest uses), so a
        document already scraped in a previous run is skipped before its detail
        page is fetched. Disabled when ``DEDUP_ENABLED`` is False or the spider
        sets no ``DEDUP_KEY`` — in which case ``is_seen`` is always False.

        Legacy ``(key, run_id, ts)`` rows are retained as unverified prior successes
        and made immediately refresh-due. Generic sources release only the oldest
        bounded slice each run; Supreme Court keeps its journal-backed all-key model.
        """
        self.dedup_enabled = bool(self.DEDUP_KEY) and settings.getbool(
            "DEDUP_ENABLED", True
        )
        self._seen_keys: set[str] = set()
        self._refresh_keys: set[str] = set()
        self._staged_dedup_records: dict[str, dict] = {}
        self._dedup_conn = None
        if not self.dedup_enabled:
            return

        os.umask(0o077)
        self._dedup_refresh_default_days = max(
            1, settings.getint("DEDUP_REFRESH_DEFAULT_DAYS", 30)
        )
        self._dedup_refresh_tas_days = max(
            1, settings.getint("DEDUP_REFRESH_TAS_DAYS", 7)
        )
        self._dedup_refresh_pending_days = max(
            1, settings.getint("DEDUP_REFRESH_PENDING_DAYS", 1)
        )
        requested_limit = max(0, settings.getint("DEDUP_REFRESH_LIMIT", 2_000))
        self._dedup_refresh_limit = min(
            requested_limit, self.MAX_REFRESHES_PER_RUN
        )
        self.dedup_db_path = self.ARTIFACTS_ROOT / self.name / "seen.sqlite"
        self.dedup_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._dedup_conn = sqlite3.connect(str(self.dedup_db_path))
        self._dedup_conn.execute("PRAGMA synchronous=FULL")
        self._dedup_conn.execute(
            "CREATE TABLE IF NOT EXISTS seen ("
            "key TEXT PRIMARY KEY, run_id TEXT, ts TEXT, "
            "last_success TEXT, content_hash TEXT, refresh_deadline TEXT, "
            "outcome TEXT, content_kind TEXT, content_complete INTEGER, "
            "source_binary_url TEXT, is_consolidated INTEGER)"
        )
        self._migrate_seen_schema()
        self._load_seen_state()
        self.logger.info(
            "dedup: loaded %d active key(s), selected %d refresh(es) from %s",
            self.dedup_seen_count(),
            len(self._refresh_keys),
            self.dedup_db_path,
        )

    def _dedup_now(self) -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _parse_dedup_timestamp(value, fallback: datetime) -> datetime:
        if not value:
            return fallback
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _migrate_seen_schema(self) -> None:
        """Add success-state columns and make legacy rows safely refresh-due."""
        columns = {
            row[1] for row in self._dedup_conn.execute("PRAGMA table_info(seen)")
        }
        additions = {
            "last_success": "TEXT",
            "content_hash": "TEXT",
            "refresh_deadline": "TEXT",
            "outcome": "TEXT",
            "content_kind": "TEXT",
            "content_complete": "INTEGER",
            "source_binary_url": "TEXT",
            "is_consolidated": "INTEGER",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                self._dedup_conn.execute(
                    f"ALTER TABLE seen ADD COLUMN {name} {sql_type}"
                )

        now = self._dedup_now().isoformat()
        self._dedup_conn.execute(
            "UPDATE seen SET "
            "last_success = CASE WHEN ts IS NULL OR trim(ts) = '' THEN ? ELSE ts END, "
            "content_hash = '', refresh_deadline = ?, outcome = 'legacy_success', "
            "content_kind = 'legacy_unknown', content_complete = 0, "
            "source_binary_url = '' WHERE outcome IS NULL",
            (now, now),
        )
        self._dedup_conn.commit()

    def _load_seen_state(self) -> None:
        if self.name == "supremecourt":
            # Its fsync journal and reconciliation own retry/completeness semantics.
            self._seen_keys = {
                row[0] for row in self._dedup_conn.execute("SELECT key FROM seen")
            }
            return

        # The selected slice is bounded; ordinary membership stays in SQLite instead of
        # materializing the full corpus in Python. ISO-8601 UTC values sort chronologically.
        rows = self._dedup_conn.execute(
            "SELECT key FROM seen WHERE "
            "outcome IN ('success', 'legacy_success', 'incomplete') "
            "AND refresh_deadline <= ? "
            "ORDER BY refresh_deadline, last_success, key LIMIT ?",
            (self._dedup_now().isoformat(), self._dedup_refresh_limit),
        )
        self._refresh_keys = {row[0] for row in rows}

    def iter_refresh_keys(self):
        """Return the bounded, deterministic direct-refresh slice for this run."""
        return iter(sorted(getattr(self, "_refresh_keys", ())))

    def dedup_refresh_context(self, key: str) -> dict | None:
        """Return persisted source lineage needed to refresh *key* safely.

        A legacy row can be due without carrying every field required to reproduce
        its source classification.  Callers must treat a ``None`` field as unknown,
        rather than inventing a value during a listing-independent refresh.
        """
        if not self.dedup_enabled or self._dedup_conn is None:
            return None
        row = self._dedup_conn.execute(
            "SELECT is_consolidated FROM seen WHERE key = ? AND "
            "outcome IN ('success', 'legacy_success', 'incomplete')",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "is_consolidated": None if row[0] is None else bool(row[0]),
        }

    def dedup_seen_count(self) -> int:
        if not self.dedup_enabled or self._dedup_conn is None:
            return 0
        row = self._dedup_conn.execute(
            "SELECT COUNT(*) FROM seen WHERE outcome IN ('success', 'legacy_success')"
        ).fetchone()
        return int(row[0]) if row else 0

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
        if key is None:
            return False
        if key in self._seen_keys:
            return True
        if key in self._refresh_keys or self._dedup_conn is None:
            return False
        row = self._dedup_conn.execute(
            "SELECT outcome FROM seen WHERE key = ?", (key,)
        ).fetchone()
        return bool(row and row[0] in self._SUCCESS_OUTCOMES)

    @staticmethod
    def _explicitly_incomplete(value) -> bool:
        if value is False or value == 0:
            return True
        return isinstance(value, str) and value.strip().lower() in {
            "0",
            "false",
            "no",
        }

    def _is_pending_material(self, mapping) -> bool:
        if str(mapping.get("decision_status_id") or "").strip() == "8":
            return True
        text = " ".join(
            str(mapping.get(field) or "")
            for field in (
                "status",
                "additional_status",
                "decision",
                "decision_status",
                "content_kind",
            )
        ).casefold()
        return any(marker in text for marker in self._PENDING_MARKERS)

    def _refresh_days(self, mapping) -> int:
        if self._is_pending_material(mapping):
            return self._dedup_refresh_pending_days
        if self.name == "tas":
            return self._dedup_refresh_tas_days
        return self._dedup_refresh_default_days

    def _dedup_record(self, mapping, *, force_success: bool = False) -> dict:
        now = self._dedup_now()
        body = str(mapping.get("body_markdown") or "")
        content_kind = str(
            mapping.get("content_kind") or ("full_text" if body.strip() else "missing")
        )
        extraction_status = str(mapping.get("extraction_status") or "").lower()
        summary_only = "summary" in content_kind.lower() or content_kind.lower() in {
            "metadata_only",
            "missing",
        }
        extraction_incomplete = extraction_status in {
            "scanned_no_text",
            "truncated",
            "malformed",
            "resource_limited",
        }
        source_incomplete = self.name == "tas" and mapping.get(
            "decision_status_id"
        ) in (None, "")
        complete = force_success or (
            bool(body.strip())
            and not self._explicitly_incomplete(mapping.get("content_complete"))
            and not summary_only
            and not extraction_incomplete
            and not source_incomplete
        )
        outcome = "success" if complete else "incomplete"
        refresh_deadline = (
            now + timedelta(days=self._refresh_days(mapping)) if complete else now
        )
        raw_is_consolidated = mapping.get("is_consolidated")
        is_consolidated = (
            int(raw_is_consolidated)
            if isinstance(raw_is_consolidated, bool)
            or (
                isinstance(raw_is_consolidated, int)
                and raw_is_consolidated in (0, 1)
            )
            else None
        )
        return {
            "key": self.dedup_key(mapping),
            "run_id": getattr(self, "run_id", ""),
            "ts": now.isoformat(),
            "last_success": now.isoformat() if complete else None,
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "refresh_deadline": refresh_deadline.isoformat(),
            "outcome": outcome,
            "content_kind": content_kind,
            "content_complete": int(complete),
            "source_binary_url": str(mapping.get("source_binary_url") or ""),
            "is_consolidated": is_consolidated,
        }

    def stage_seen(self, mapping) -> bool:
        """Stage a generic item outcome; return False only for a staged duplicate."""
        key = self.dedup_key(mapping)
        if key is None:
            return True
        record = self._dedup_record(mapping)
        previous = self._staged_dedup_records.get(key)
        if previous is not None:
            if previous["outcome"] != "success" and record["outcome"] == "success":
                self._staged_dedup_records[key] = record
                self._seen_keys.add(key)
                return True
            return False
        self._staged_dedup_records[key] = record
        if record["outcome"] == "success":
            self._seen_keys.add(key)
        return True

    def _persist_dedup_records(self, records: list[dict]) -> None:
        if self._dedup_conn is None or not records:
            return
        sql = (
            "INSERT INTO seen (key, run_id, ts, last_success, content_hash, "
            "refresh_deadline, outcome, content_kind, content_complete, "
            "source_binary_url, is_consolidated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET run_id=excluded.run_id, ts=excluded.ts, "
            "last_success=excluded.last_success, content_hash=excluded.content_hash, "
            "refresh_deadline=excluded.refresh_deadline, outcome=excluded.outcome, "
            "content_kind=excluded.content_kind, "
            "content_complete=excluded.content_complete, "
            "source_binary_url=excluded.source_binary_url, "
            "is_consolidated=excluded.is_consolidated"
        )
        try:
            self._dedup_conn.execute("BEGIN IMMEDIATE")
            for original in records:
                record = dict(original)
                if record["outcome"] != "success":
                    prior = self._dedup_conn.execute(
                        "SELECT last_success, content_hash, is_consolidated "
                        "FROM seen WHERE key = ?",
                        (record["key"],),
                    ).fetchone()
                    if prior and prior[0]:
                        record["last_success"] = prior[0]
                        record["content_hash"] = prior[1]
                        if record["is_consolidated"] is None:
                            record["is_consolidated"] = prior[2]
                self._dedup_conn.execute(
                    sql,
                    (
                        record["key"],
                        record["run_id"],
                        record["ts"],
                        record["last_success"],
                        record["content_hash"],
                        record["refresh_deadline"],
                        record["outcome"],
                        record["content_kind"],
                        record["content_complete"],
                        record["source_binary_url"],
                        record["is_consolidated"],
                    ),
                )
            self._dedup_conn.commit()
        except Exception:
            self._dedup_conn.rollback()
            raise

    def commit_staged_seen(self) -> int:
        records = list(self._staged_dedup_records.values())
        self._persist_dedup_records(records)
        for record in records:
            self._refresh_keys.discard(record["key"])
        self._staged_dedup_records.clear()
        return len(records)

    def discard_staged_seen(self) -> None:
        for record in self._staged_dedup_records.values():
            if record["outcome"] == "success":
                self._seen_keys.discard(record["key"])
        self._staged_dedup_records.clear()

    def mark_seen(self, mapping) -> bool:
        """Immediately persist a success for custom durable pipelines.

        The generic pipeline uses :meth:`stage_seen`; Supreme Court intentionally calls
        this only after its journal append has been flushed and fsynced.
        """
        key = self.dedup_key(mapping)
        if key is None or self.is_seen(mapping):
            return False
        record = self._dedup_record(mapping, force_success=True)
        self._persist_dedup_records([record])
        self._seen_keys.add(key)
        self._refresh_keys.discard(key)
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
            # An explicit empty file is operational evidence. Without it, an empty/failed run
            # updates latest/run.json but leaves the previous latest/items.jsonl in place.
            "store_empty": True,
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
