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
import re
import sqlite3
import stat
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import scrapy

from ..completion import (
    atomic_create_private,
    build_startup_record,
    canonical_json_bytes,
    hash_and_fsync_private_file,
    publish_startup_metadata,
    recover_authorized_terminal,
    source_lock,
    terminal_authorization_exists,
    verify_authorized_terminal_record,
    verify_terminal_record,
)


EVIDENCE_SNAPSHOT_ID = "v3_512_attested_20260715_01"
EVIDENCE_START_DATE = date(1900, 1, 1)
EVIDENCE_END_DATE = date(2026, 7, 15)
EVIDENCE_ARTIFACTS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "ingest/.state/v3/source-evidence"
    / EVIDENCE_SNAPSHOT_ID
)
_EVIDENCE_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_EVIDENCE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DISCOVERY_LIMIT_NAMES = (
    "MAX_PAGES",
    "MAX_SAFE_PAGES",
    "PAGE_SIZE",
    "MAX_WINDOW_DAYS",
    "TARGET_MIN",
    "TARGET_MAX",
)
_EVIDENCE_IDENTITY_OUTCOMES = frozenset({"unique", "duplicate", "missing"})


@dataclass(eq=False, slots=True)
class ReversibleDedupCommit:
    """Handle for one durable but inactive pending seen-store batch.

    Pending rows never participate in :meth:`BaseLegalSpider.is_seen`.  The terminal
    attester holds this handle until exact successful completion evidence exists,
    then promotion makes the rows active in one SQLite transaction.
    """

    owner: object
    run_id: str
    committed_count: int
    active: bool = True


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
    _COMPLETION_SUCCESS = "success"
    _COMPLETION_FAILURE = "failure"
    _COMPLETION_ABSENT = "absent"
    _COMPLETION_AMBIGUOUS = "ambiguous"
    _PENDING_MARKERS = (
        "pending",
        "draft",
        "ასამოქმედებელ",
        "მოლოდინ",
        "პროექტ",
    )
    _DEDUP_COLUMNS = (
        "key",
        "run_id",
        "ts",
        "last_success",
        "content_hash",
        "refresh_deadline",
        "outcome",
        "content_kind",
        "content_complete",
        "source_binary_url",
        "is_consolidated",
    )
    _PENDING_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")

    # Safe defaults so is_seen/dedup_key are inert until ``open_dedup_store`` runs
    # (e.g. in unit tests that call parse callbacks without ``from_crawler``).
    dedup_enabled = False
    _seen_keys: set = frozenset()
    _pending_reversible_dedup_commit: ReversibleDedupCommit | None = None

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

    def record_quarantine(self, reason: str) -> None:
        """Count an explicitly persisted, non-admissible source record.

        Quarantine is not a crawl failure: the source row was discovered and durably
        represented, but downstream snapshot construction must route it away from the
        admitted corpus.  Transport/parser failures are recorded separately through
        :meth:`record_quality_failure`.
        """

        crawler = getattr(self, "crawler", None)
        stats = getattr(crawler, "stats", None)
        if stats is not None:
            stats.inc_value("quarantine/items")
            stats.inc_value(f"quarantine/{reason}")

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
        self.evidence_crawl = settings.getbool("EVIDENCE_CRAWL_ENABLED", False)
        if self.evidence_crawl:
            if (
                self.scraping_start_date != EVIDENCE_START_DATE
                or self.scraping_end_date != EVIDENCE_END_DATE
            ):
                raise ValueError(
                    "immutable evidence crawl requires the exact interval "
                    f"{EVIDENCE_START_DATE.isoformat()} through "
                    f"{EVIDENCE_END_DATE.isoformat()}"
                )
            revision = str(settings.get("EVIDENCE_CODE_REVISION") or "").strip()
            if _EVIDENCE_REVISION_RE.fullmatch(revision) is None:
                raise ValueError(
                    "immutable evidence crawl requires EVIDENCE_CODE_REVISION as "
                    "an immutable lowercase hexadecimal revision"
                )
            self.evidence_code_revision = revision
            code_identity = str(
                settings.get("EVIDENCE_CODE_IDENTITY_SHA256") or ""
            ).strip()
            if _EVIDENCE_SHA256_RE.fullmatch(code_identity) is None:
                raise ValueError(
                    "immutable evidence crawl requires "
                    "EVIDENCE_CODE_IDENTITY_SHA256 as a lowercase SHA-256 digest"
                )
            self.evidence_code_identity_sha256 = code_identity
            if self.name == "matsne":
                doc_type = getattr(self, "_doc_type", None)
                selected_doc_type = doc_type() if callable(doc_type) else "all"
                seed_arguments = {
                    name: getattr(self, name, None)
                    for name in ("seed_ids_file", "seed_file", "seed_urls")
                }
                if selected_doc_type != "all" or any(seed_arguments.values()):
                    raise ValueError(
                        "immutable Matsne evidence crawl requires full/default "
                        "doc_type=all mode without seed arguments"
                    )
            # These are hard safety properties, not caller-selectable preferences.
            settings.set("HTTPCACHE_ENABLED", False, priority="spider")
            settings.set("DEDUP_ENABLED", False, priority="spider")
            source_root = Path(os.path.abspath(EVIDENCE_ARTIFACTS_ROOT / self.name))
        else:
            source_root = Path(os.path.abspath(self.ARTIFACTS_ROOT / self.name))
        self.source_root = source_root
        # A per-source lock makes run-id selection and the two-record startup
        # publication one operation across concurrent crawler processes.  The startup
        # publisher deliberately replaces latest/run.json *before* it creates the
        # run-scoped record, so a crash can only leave nonqualifying ``started``
        # metadata -- never an older success masquerading as the current attempt.
        with source_lock(source_root):
            if self.evidence_crawl and self.name != "supremecourt":
                self._reject_existing_ordinary_evidence_runs(source_root)
            self.started_at = datetime.now(UTC).replace(microsecond=0)
            self.run_id = self.build_run_id(create_only=self.evidence_crawl)
            self.run_dir = source_root / "runs" / self.run_id
            self.latest_dir = None if self.evidence_crawl else source_root / "latest"

            self.items_path = self.run_dir / "items.jsonl"
            self.latest_items_path = (
                None if self.latest_dir is None else self.latest_dir / "items.jsonl"
            )
            self.log_path = self.run_dir / "spider.log"
            self.run_metadata_path = self.run_dir / "run.json"
            self.latest_metadata_path = (
                None if self.latest_dir is None else self.latest_dir / "run.json"
            )

            feed_options = self.feed_options()
            if self.evidence_crawl:
                feed_options["overwrite"] = False
            feeds = {str(self.items_path): feed_options}
            if self.latest_items_path is not None:
                feeds[str(self.latest_items_path)] = self.feed_options()
            settings.set("FEEDS", feeds, priority="spider")
            self.write_run_metadata()
            if self.evidence_crawl and self.name != "supremecourt":
                self._open_evidence_identity_journal()
        self.open_dedup_store(settings)
        if self.evidence_crawl:
            self.crawler.stats.set_value("evidence/enabled", 1)
            self.crawler.stats.set_value("within_run/observed_identity_count", 0)
            self.crawler.stats.set_value("within_run/unique_identity_count", 0)
            self.crawler.stats.set_value("within_run/duplicate_identity_count", 0)
            self.crawler.stats.set_value("within_run/missing_identity_count", 0)

    @staticmethod
    def _reject_existing_ordinary_evidence_runs(source_root: Path) -> None:
        """Enforce the release contract's one-and-only-one ordinary crawl attempt."""

        runs_dir = source_root / "runs"
        try:
            info = runs_dir.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"ordinary evidence runs path is unsafe: {runs_dir}")
        with os.scandir(runs_dir) as iterator:
            entries = sorted(entry.name for entry in iterator)
        if entries:
            raise FileExistsError(
                "immutable ordinary evidence source may be run exactly once; "
                f"existing entries for {source_root.name}: {entries[:8]}"
            )

    def _open_evidence_identity_journal(self) -> None:
        """Create the ordinary run's append-only identity decision journal."""

        path = self.run_dir / "identity.journal.jsonl"
        atomic_create_private(path, b"")
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
        ):
            os.close(descriptor)
            raise RuntimeError(f"ordinary identity journal is not private: {path}")
        self.evidence_identity_journal_path = path
        self._evidence_identity_journal_descriptor = descriptor
        self._evidence_identity_sequence = 0
        self._evidence_identity_journal_proof = None

    def record_evidence_identity_event(
        self, outcome: str, identity: str | None
    ) -> None:
        """Append one exact ordinary within-run identity decision before export/drop."""

        if not getattr(self, "evidence_crawl", False) or self.name == "supremecourt":
            return
        if outcome not in _EVIDENCE_IDENTITY_OUTCOMES:
            raise ValueError(f"invalid evidence identity outcome: {outcome!r}")
        if outcome == "missing":
            if identity is not None:
                raise ValueError("missing identity event cannot carry an identity")
            identity_sha256 = None
        else:
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{outcome} identity event requires an identity")
            identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        descriptor = getattr(self, "_evidence_identity_journal_descriptor", None)
        if descriptor is None:
            raise RuntimeError("ordinary evidence identity journal is not open")
        self._evidence_identity_sequence += 1
        payload = canonical_json_bytes(
            {
                "identity_sha256": identity_sha256,
                "outcome": outcome,
                "sequence": self._evidence_identity_sequence,
            }
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write to ordinary evidence identity journal")
            view = view[written:]

    def finalize_evidence_identity_journal(self):
        """Flush, close, and hash the ordinary identity journal exactly once."""

        path = getattr(self, "evidence_identity_journal_path", None)
        if path is None:
            return None
        descriptor = getattr(self, "_evidence_identity_journal_descriptor", None)
        if descriptor is not None:
            os.fsync(descriptor)
            os.close(descriptor)
            self._evidence_identity_journal_descriptor = None
        proof = hash_and_fsync_private_file(path)
        previous = getattr(self, "_evidence_identity_journal_proof", None)
        if previous is not None and previous != proof:
            raise RuntimeError("ordinary evidence identity journal changed after finalization")
        self._evidence_identity_journal_proof = proof
        return proof

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
        self._within_run_keys: set[str] = set()
        self._refresh_keys: set[str] = set()
        self._staged_dedup_records: dict[str, dict] = {}
        self._pending_reversible_dedup_commit: ReversibleDedupCommit | None = None
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
        self._ensure_pending_seen_schema()
        self._recover_all_authorized_terminals()
        self._reconcile_pending_seen()
        self._load_seen_state()
        self.logger.info(
            "dedup: loaded %d active key(s), selected %d refresh(es) from %s",
            self.dedup_seen_count(),
            len(self._refresh_keys),
            self.dedup_db_path,
        )

    def _ensure_pending_seen_schema(self) -> None:
        """Create the durable inactive stage used by completion publication."""
        self._dedup_conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_seen ("
            "pending_run_id TEXT NOT NULL, key TEXT NOT NULL, run_id TEXT, ts TEXT, "
            "last_success TEXT, content_hash TEXT, refresh_deadline TEXT, "
            "outcome TEXT, content_kind TEXT, content_complete INTEGER, "
            "source_binary_url TEXT, is_consolidated INTEGER, "
            "PRIMARY KEY (pending_run_id, key))"
        )
        expected = {"pending_run_id", *self._DEDUP_COLUMNS}
        actual = {
            row[1]
            for row in self._dedup_conn.execute("PRAGMA table_info(pending_seen)")
        }
        if actual != expected:
            raise RuntimeError(
                "pending_seen schema mismatch; refusing unsafe dedup activation"
            )
        self._dedup_conn.commit()

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

    def _dedup_source_root(self) -> Path:
        return Path(
            os.path.abspath(
                getattr(self, "source_root", self.ARTIFACTS_ROOT / self.name)
            )
        )

    def _has_exact_success_evidence(self, run_id: str) -> bool:
        """Return true only for a strictly verified run-scoped success record.

        Callers hold the per-source publication lock through both this check and any
        resulting SQLite promotion, so a cooperating publisher cannot change guard
        state between proof and activation.
        """
        return self._pending_completion_state(run_id) == self._COMPLETION_SUCCESS

    def _pending_completion_state(self, run_id: str) -> str:
        """Classify evidence without collapsing ambiguity into terminal failure.

        Only strict success may promote.  Only safely observed authorization absence
        or an exact, structurally valid authorized failure may delete inactive rows.
        Every inspection/fsync/source-validator error with possible authorization is
        ambiguous and therefore preserves the shadow batch for a later startup.
        """

        if self._PENDING_RUN_ID_RE.fullmatch(run_id) is None:
            return self._COMPLETION_ABSENT
        source_root = self._dedup_source_root()
        run_dir = source_root / "runs" / run_id
        run_path = run_dir / "run.json"
        try:
            authorized = terminal_authorization_exists(
                run_path,
                expected_source=self.name,
                expected_run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001 - ambiguity must retain pending
            self.logger.warning(
                "dedup: terminal authorization is ambiguous for %s: %s",
                run_id,
                exc,
            )
            return self._COMPLETION_AMBIGUOUS
        if not authorized:
            return self._COMPLETION_ABSENT
        try:
            terminal = verify_authorized_terminal_record(
                run_path,
                expected_source=self.name,
                expected_run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001 - exact outcome is still unknown
            self.logger.warning(
                "dedup: authorized terminal binding is ambiguous for %s: %s",
                run_id,
                exc,
            )
            return self._COMPLETION_AMBIGUOUS
        if terminal["outcome"] == "failure":
            return self._COMPLETION_FAILURE
        try:
            verify_terminal_record(
                run_path,
                expected_source=self.name,
                expected_run_id=run_id,
                expected_items_path=run_dir / "items.jsonl",
            )
        except Exception as exc:  # noqa: BLE001 - success proof may be transient
            self.logger.warning(
                "dedup: authorized success proof is temporarily unverified for %s: %s",
                run_id,
                exc,
            )
            return self._COMPLETION_AMBIGUOUS
        return self._COMPLETION_SUCCESS

    def _pending_records(self, run_id: str) -> list[dict]:
        columns = ", ".join(self._DEDUP_COLUMNS)
        rows = self._dedup_conn.execute(
            f"SELECT {columns} FROM pending_seen "  # noqa: S608
            "WHERE pending_run_id = ? ORDER BY key",
            (run_id,),
        ).fetchall()
        return [dict(zip(self._DEDUP_COLUMNS, row, strict=True)) for row in rows]

    def _delete_pending_run(self, run_id: str) -> int:
        cursor = self._dedup_conn.execute(
            "DELETE FROM pending_seen WHERE pending_run_id = ?",
            (run_id,),
        )
        return max(0, int(cursor.rowcount))

    def _promote_pending_run(self, run_id: str) -> list[dict]:
        """Atomically promote one exact-success batch into active seen rows."""
        try:
            self._dedup_conn.execute("BEGIN IMMEDIATE")
            records = self._pending_records(run_id)
            self._upsert_dedup_records(records)
            self._delete_pending_run(run_id)
            self._dedup_conn.commit()
            return records
        except Exception:
            self._dedup_conn.rollback()
            raise

    def _discard_pending_run(self, run_id: str) -> int:
        try:
            self._dedup_conn.execute("BEGIN IMMEDIATE")
            deleted = self._delete_pending_run(run_id)
            self._dedup_conn.commit()
            return deleted
        except Exception:
            self._dedup_conn.rollback()
            raise

    def _recover_all_authorized_terminals(self) -> None:
        """Recover every permanent WAL decision before loading active dedup state.

        Recovery is not limited to runs currently represented in ``pending_seen``:
        a prior process may have promoted rows and then crashed before a directory
        rename became durable.  Scanning exact private run directories ensures their
        authoritative ``run.json`` is reconstructed before those active rows load.
        """

        source_root = self._dedup_source_root()
        runs_root = source_root / "runs"
        if not runs_root.is_dir():
            return
        with source_lock(source_root):
            try:
                entries = sorted(os.scandir(runs_root), key=lambda entry: entry.name)
            except OSError as exc:
                self.logger.error("dedup: could not scan terminal recovery WAL: %s", exc)
                return
            for entry in entries:
                if (
                    self._PENDING_RUN_ID_RE.fullmatch(entry.name) is None
                    or not entry.is_dir(follow_symlinks=False)
                ):
                    continue
                try:
                    recovered = recover_authorized_terminal(
                        Path(entry.path) / "run.json",
                        expected_source=self.name,
                        expected_run_id=entry.name,
                    )
                except Exception as exc:  # noqa: BLE001 - never invent evidence
                    self.logger.error(
                        "dedup: authorized terminal recovery failed for %s: %s",
                        entry.name,
                        exc,
                    )
                    continue
                if recovered:
                    self.logger.info(
                        "dedup: materialized authorized terminal for run %s",
                        entry.name,
                    )

    def _reconcile_pending_seen(self) -> None:
        """Resolve crash-left pending batches without ever trusting them directly."""
        run_ids = [
            str(row[0])
            for row in self._dedup_conn.execute(
                "SELECT DISTINCT pending_run_id FROM pending_seen "
                "ORDER BY pending_run_id"
            )
        ]
        for run_id in run_ids:
            try:
                with source_lock(self._dedup_source_root()):
                    # This call deliberately occurs while the existing lock is held;
                    # recover_authorized_terminal must not recursively acquire it.
                    recover_authorized_terminal(
                        self._dedup_source_root() / "runs" / run_id / "run.json",
                        expected_source=self.name,
                        expected_run_id=run_id,
                    )
                    evidence_state = self._pending_completion_state(run_id)
                    if evidence_state == self._COMPLETION_SUCCESS:
                        promoted = self._promote_pending_run(run_id)
            except Exception as exc:  # noqa: BLE001 - pending remains inactive
                self.logger.error(
                    "dedup: could not reconcile pending batch %s: %s",
                    run_id,
                    exc,
                )
                continue
            if evidence_state == self._COMPLETION_SUCCESS:
                self.logger.info(
                    "dedup: recovered %d pending row(s) for successful run %s",
                    len(promoted),
                    run_id,
                )
                continue
            if evidence_state == self._COMPLETION_AMBIGUOUS:
                self.logger.warning(
                    "dedup: preserving inactive pending row(s) for ambiguous run %s",
                    run_id,
                )
                continue
            try:
                deleted = self._discard_pending_run(run_id)
            except Exception as exc:  # noqa: BLE001 - pending remains ignored
                self.logger.error(
                    "dedup: could not discard unproven pending batch %s: %s",
                    run_id,
                    exc,
                )
            else:
                self.logger.warning(
                    "dedup: discarded %d pending row(s) for %s run %s",
                    deleted,
                    evidence_state,
                    run_id,
                )

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
        if not self.DEDUP_KEY:
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
        if (
            self.dedup_enabled or getattr(self, "evidence_crawl", False)
        ) and key in getattr(self, "_within_run_keys", ()):
            return True
        if not self.dedup_enabled:
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
        track_within_run = self.dedup_enabled or getattr(
            self, "evidence_crawl", False
        )
        if not track_within_run:
            return True
        if key in self._within_run_keys:
            return False
        self._within_run_keys.add(key)
        if not self.dedup_enabled:
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

    def _upsert_dedup_records(self, records: list[dict]) -> None:
        """Upsert records inside the caller's active SQLite transaction."""
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
                tuple(record[column] for column in self._DEDUP_COLUMNS),
            )

    def _persist_dedup_records(self, records: list[dict]) -> None:
        if self._dedup_conn is None or not records:
            return
        try:
            self._dedup_conn.execute("BEGIN IMMEDIATE")
            self._upsert_dedup_records(records)
            self._dedup_conn.commit()
        except Exception:
            self._dedup_conn.rollback()
            raise

    def commit_staged_seen(self) -> int:
        """Irreversibly commit staged records for direct, non-attester callers."""
        if self._pending_reversible_dedup_commit is not None:
            raise RuntimeError("a reversible staged dedup commit is still pending")
        records = list(self._staged_dedup_records.values())
        self._persist_dedup_records(records)
        for record in records:
            self._refresh_keys.discard(record["key"])
            self._within_run_keys.discard(record["key"])
        self._staged_dedup_records.clear()
        return len(records)

    def commit_staged_seen_reversible(self) -> ReversibleDedupCommit:
        """Durably prepare staged records without making them visible to ``is_seen``.

        The batch is written only to ``pending_seen``.  A process crash or an
        ambiguous SQLite commit acknowledgement can therefore leave, at worst,
        inactive shadow rows.  Promotion into ``seen`` is a separate operation and
        requires a strictly verified successful completion record for this exact run.
        """
        if self._pending_reversible_dedup_commit is not None:
            raise RuntimeError("a reversible staged dedup commit is already pending")

        run_id = self._validated_pending_run_id(getattr(self, "run_id", ""))
        records = [dict(record) for record in self._staged_dedup_records.values()]
        if records and self._dedup_conn is None:
            raise RuntimeError("cannot prepare dedup without an open dedup store")
        for record in records:
            if record.get("run_id") != run_id:
                raise RuntimeError("staged dedup record has the wrong run identity")
        if self._dedup_conn is not None:
            self._write_pending_staged_records(run_id, records)

        token = ReversibleDedupCommit(
            owner=self,
            run_id=run_id,
            committed_count=len(records),
        )
        self._pending_reversible_dedup_commit = token
        # ``stage_seen`` temporarily remembers successful keys so duplicates within
        # one live crawl are suppressed.  Once the durable shadow batch exists, even
        # that process-local membership is removed: pending means inactive.
        for record in records:
            if record["outcome"] == "success":
                self._seen_keys.discard(record["key"])
        self._staged_dedup_records.clear()
        return token

    def _validated_pending_run_id(self, value: object) -> str:
        run_id = str(value)
        if self._PENDING_RUN_ID_RE.fullmatch(run_id) is None:
            raise RuntimeError("unsafe pending dedup run identity")
        return run_id

    def _write_pending_staged_records(
        self, run_id: str, records: list[dict]
    ) -> None:
        """Replace one run's inactive shadow batch in a single transaction."""
        if self._dedup_conn is None:
            raise RuntimeError("dedup connection is closed")
        columns = ", ".join(("pending_run_id", *self._DEDUP_COLUMNS))
        placeholders = ", ".join("?" for _column in range(1 + len(self._DEDUP_COLUMNS)))
        try:
            self._dedup_conn.execute("BEGIN IMMEDIATE")
            self._delete_pending_run(run_id)
            if records:
                self._dedup_conn.executemany(
                    f"INSERT INTO pending_seen ({columns}) "  # noqa: S608
                    f"VALUES ({placeholders})",
                    (
                        (run_id, *(record[column] for column in self._DEDUP_COLUMNS))
                        for record in records
                    ),
                )
            self._dedup_conn.commit()
        except Exception:
            self._dedup_conn.rollback()
            raise

    def _require_reversible_dedup_token(
        self, token: ReversibleDedupCommit
    ) -> None:
        if (
            not isinstance(token, ReversibleDedupCommit)
            or token.owner is not self
            or self._pending_reversible_dedup_commit is not token
            or not token.active
        ):
            raise RuntimeError("invalid or inactive reversible dedup commit token")

    def accept_staged_seen_commit(self, token: ReversibleDedupCommit) -> int:
        """Promote a shadow batch only after exact success evidence verifies."""
        self._require_reversible_dedup_token(token)
        if self._dedup_conn is None:
            raise RuntimeError("dedup connection closed before pending promotion")
        with source_lock(self._dedup_source_root()):
            if not self._has_exact_success_evidence(token.run_id):
                raise RuntimeError(
                    "cannot promote pending dedup without exact successful completion"
                )
            records = self._promote_pending_run(token.run_id)
        for record in records:
            self._refresh_keys.discard(record["key"])
            if record["outcome"] == "success":
                self._seen_keys.add(record["key"])
            else:
                self._seen_keys.discard(record["key"])
        token.active = False
        self._pending_reversible_dedup_commit = None
        return token.committed_count

    def rollback_staged_seen_commit(self, token: ReversibleDedupCommit) -> int:
        """Delete an inactive shadow batch; active ``seen`` rows are untouched."""
        self._require_reversible_dedup_token(token)
        if self._dedup_conn is None:
            raise RuntimeError("dedup connection closed before pending discard")
        pending_keys = {
            record["key"] for record in self._pending_records(token.run_id)
        }
        self._discard_pending_run(token.run_id)
        self._within_run_keys.difference_update(pending_keys)
        token.active = False
        self._pending_reversible_dedup_commit = None
        return token.committed_count

    def discard_pending_staged_seen(self, run_id: str | None = None) -> int:
        """Best-effort recovery API for a prepare whose commit acknowledgement failed.

        This method deliberately works without a token.  If SQLite committed the
        shadow batch and then raised before the caller received its handle, the run
        identity is enough to remove it.  Failure still leaves only ignored rows.
        """
        if self._dedup_conn is None:
            raise RuntimeError("dedup connection closed before pending discard")
        selected = self._validated_pending_run_id(
            getattr(self, "run_id", "") if run_id is None else run_id
        )
        pending_keys = {
            record["key"] for record in self._pending_records(selected)
        }
        deleted = self._discard_pending_run(selected)
        self._within_run_keys.difference_update(pending_keys)
        token = self._pending_reversible_dedup_commit
        if token is not None and token.run_id == selected:
            token.active = False
            self._pending_reversible_dedup_commit = None
        return deleted

    def pending_staged_seen_count(self, run_id: str | None = None) -> int:
        """Return the inactive shadow-row count for diagnostics and tests."""
        if self._dedup_conn is None:
            return 0
        selected = self._validated_pending_run_id(
            getattr(self, "run_id", "") if run_id is None else run_id
        )
        row = self._dedup_conn.execute(
            "SELECT COUNT(*) FROM pending_seen WHERE pending_run_id = ?",
            (selected,),
        ).fetchone()
        return int(row[0]) if row else 0

    def discard_staged_seen(self) -> None:
        # A durable pending batch, if any, is independent and remains invisible.
        # Clearing the process-local stage is therefore always safe, including when
        # deletion of pending rows failed and a later startup must reconcile them.
        for record in self._staged_dedup_records.values():
            if record["outcome"] == "success":
                self._seen_keys.discard(record["key"])
            self._within_run_keys.discard(record["key"])
        self._staged_dedup_records.clear()

    def mark_seen(self, mapping) -> bool:
        """Immediately persist a success for custom durable pipelines.

        The generic pipeline uses :meth:`stage_seen`; Supreme Court intentionally calls
        this only after its journal append has been flushed and fsynced.
        """
        key = self.dedup_key(mapping)
        if key is None or self.is_seen(mapping):
            return False
        if not self.dedup_enabled:
            if not getattr(self, "evidence_crawl", False):
                return False
            self._within_run_keys.add(key)
            return True
        record = self._dedup_record(mapping, force_success=True)
        self._persist_dedup_records([record])
        self._seen_keys.add(key)
        self._refresh_keys.discard(key)
        return True

    def build_run_id(self, *, create_only: bool = False) -> str:
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        base_run_id = (
            f"{timestamp}_start-{self.scraping_start_date.isoformat()}"
            f"_end-{self.scraping_end_date.isoformat()}"
        )
        run_id = base_run_id
        suffix = 2
        source_root = Path(
            os.path.abspath(getattr(self, "source_root", self.ARTIFACTS_ROOT / self.name))
        )
        if create_only and (source_root / "runs" / run_id).exists():
            raise FileExistsError(
                f"immutable evidence run destination already exists: "
                f"{source_root / 'runs' / run_id}"
            )
        while (source_root / "runs" / run_id).exists():
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
        """Invalidate latest and create this run's nonqualifying startup record."""
        metadata = build_startup_record(self)
        publish_startup_metadata(
            self.run_metadata_path,
            self.latest_metadata_path,
            metadata,
        )

    def evidence_source_arguments(self) -> dict[str, object]:
        """Return bounded, reportable spider arguments for the crawl contract."""
        arguments: dict[str, object] = {}
        if self.name == "matsne":
            doc_type = getattr(self, "_doc_type", None)
            arguments["doc_type"] = doc_type() if callable(doc_type) else "all"
            for name in ("seed_ids_file", "seed_file", "seed_urls"):
                value = getattr(self, name, None)
                arguments[name] = None if value in (None, "") else value
        if self.name == "supremecourt":
            arguments["initial_window_days"] = int(
                getattr(self, "initial_window_days", 7)
            )
            arguments["parent_run_id"] = getattr(self, "parent_run_id", None)
            crawler = getattr(self, "crawler", None)
            settings = getattr(crawler, "settings", None)
            arguments["max_runtime_seconds"] = (
                settings.getint("CLOSESPIDER_TIMEOUT", 0)
                if settings is not None
                else 0
            )
        return arguments

    def evidence_discovery_limits(self) -> dict[str, int]:
        """Return deterministic numeric discovery ceilings for attestation."""
        module = sys.modules.get(type(self).__module__)
        limits: dict[str, int] = {}
        for name in _DISCOVERY_LIMIT_NAMES:
            value = getattr(self, name, None)
            if value is None and module is not None:
                value = getattr(module, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                limits[name.lower()] = value
        return limits
