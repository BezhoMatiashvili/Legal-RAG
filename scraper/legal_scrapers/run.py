"""Run several spiders together in one process.

Usage (from the ``scraper/`` directory)::

    uv run python -m legal_scrapers.run --start-date 2026-06-01 --end-date 2026-06-30
    uv run python -m legal_scrapers.run --only ecd tbappeal --start-date 2026-06-20 --end-date 2026-06-22
    uv run python -m legal_scrapers.run --no-dedup        # force a full re-scrape

All spiders run concurrently in a single ``CrawlerProcess``. Each targets a
different domain, so per-domain politeness (DOWNLOAD_DELAY, AutoThrottle,
CONCURRENT_REQUESTS_PER_DOMAIN) is unchanged versus running them one at a time.
A single combined progress table is shown (see ``extensions.py``).

Single-spider runs still use the normal ``scrapy crawl <spider>`` command.
"""

import argparse
import os
import stat
import sys
from pathlib import Path

from .completion import evaluate_crawl_quality, verify_terminal_record
from .spiders.base import (
    EVIDENCE_ARTIFACTS_ROOT,
    EVIDENCE_END_DATE,
    EVIDENCE_START_DATE,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXED_RELEASE_BUNDLE = Path(
    "/secure/release-inputs/v3_512_candidate_20260715_01"
)
ORDINARY_EVIDENCE_SOURCE_ORDER = (
    "matsne",
    "ecd",
    "constcourt",
    "napr",
    "tas",
    "tbappeal",
)
ORDINARY_EVIDENCE_SOURCES = frozenset(ORDINARY_EVIDENCE_SOURCE_ORDER)

# Canonical order for the combined display. Only names actually present in the
# project's spider loader are launched, so this list can stay ahead of reality.
SPIDER_ORDER = [
    "matsne",
    "ecd",
    "constcourt",
    "napr",
    "supremecourt",
    "tas",
    "tbappeal",
]


def _bootstrap_project_dir() -> Path:
    """Make the Scrapy project importable + discoverable regardless of cwd.

    ``run.py`` lives at ``<repo>/scraper/legal_scrapers/run.py``; ``scrapy.cfg`` and the
    ``legal_scrapers`` package live one level up at ``<repo>/scraper``. ``scrapy crawl``
    relies on being run from there, so we replicate that here.
    """
    project_dir = Path(__file__).resolve().parent.parent  # <repo>/scraper
    os.chdir(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "legal_scrapers.settings")
    return project_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m legal_scrapers.run",
        description="Run all (or a subset of) the legal spiders together.",
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD; forwarded to every spider")
    parser.add_argument("--end-date", help="YYYY-MM-DD; forwarded to every spider")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SPIDER",
        help="Run only these spiders (default: all).",
    )
    parser.add_argument(
        "--evidence-crawl",
        action="store_true",
        help=(
            "use the fixed create-only v3 evidence root; forces the frozen interval, "
            "HTTP cache off, cross-run dedup off, and never writes latest"
        ),
    )
    parser.add_argument(
        "--code-revision",
        help="immutable lowercase hexadecimal code revision (required for evidence crawl)",
    )
    parser.add_argument(
        "--release-bundle",
        type=Path,
        help=(
            "fixed attested release bundle to revalidate before an evidence crawl "
            "can initialize Scrapy"
        ),
    )
    parser.add_argument(
        "--supreme-parent-run-id",
        help=(
            "exact validated Supreme Court parent run ID; use the literal 'none' "
            "only for the explicit first run"
        ),
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable cross-run dedup for this run (re-scrape everything).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress display.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=_positive_int,
        help=(
            "gracefully close after this many seconds (Scrapy CLOSESPIDER_TIMEOUT); "
            "in-flight requests/items are drained before final artifacts are written"
        ),
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_release_validator(repo_root: Path):
    """Load the offline release validator without importing Scrapy runtime code."""

    ingest_root = repo_root / "ingest"
    if str(ingest_root) not in sys.path:
        sys.path.insert(0, str(ingest_root))
    from ingest.release_inputs import validate_release_inputs

    return validate_release_inputs


def _validate_evidence_release_gate(
    args: argparse.Namespace,
    *,
    repo_root: Path = REPOSITORY_ROOT,
) -> tuple[str, str] | None:
    """Fail closed unless this invocation independently validates the fixed bundle."""

    if not args.evidence_crawl:
        if args.release_bundle is not None:
            raise SystemExit("--release-bundle is valid only with --evidence-crawl")
        return None

    if args.release_bundle is None:
        raise SystemExit(
            f"--evidence-crawl requires --release-bundle {FIXED_RELEASE_BUNDLE}"
        )
    bundle = args.release_bundle.expanduser().absolute()
    if bundle != FIXED_RELEASE_BUNDLE:
        raise SystemExit(
            "--evidence-crawl requires the exact fixed release bundle "
            f"{FIXED_RELEASE_BUNDLE}"
        )
    if not args.code_revision:
        raise SystemExit("--evidence-crawl requires --code-revision")

    try:
        validator = _load_release_validator(repo_root)
        validated = validator(bundle, repo_root=repo_root)
        repository_revision = validated.repository_revision
        code_identity_sha256 = validated.code_identity_sha256
    except Exception as exc:  # noqa: BLE001 - any validator failure keeps crawl disabled
        raise SystemExit(
            "evidence release bundle validation failed before crawler initialization: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if args.code_revision != repository_revision:
        raise SystemExit(
            "--code-revision does not match the independently validated release "
            "repository revision"
        )
    return repository_revision, code_identity_sha256


def _reject_existing_ordinary_evidence_destinations() -> None:
    """Preflight all six ordinary roots before any crawler can initialize."""

    existing: dict[str, list[str]] = {}
    for source in sorted(ORDINARY_EVIDENCE_SOURCES):
        runs_dir = (EVIDENCE_ARTIFACTS_ROOT / source / "runs").absolute()
        cursor = Path(runs_dir.anchor)
        missing = False
        for component in runs_dir.parts[1:]:
            cursor /= component
            try:
                info = cursor.lstat()
            except FileNotFoundError:
                missing = True
                break
            if stat.S_ISLNK(info.st_mode):
                raise SystemExit(
                    f"ordinary evidence destination contains a symlink: {cursor}"
                )
        if missing:
            continue
        if not runs_dir.is_dir():
            raise SystemExit(f"ordinary evidence runs path is not a directory: {runs_dir}")
        with os.scandir(runs_dir) as iterator:
            entries = sorted(entry.name for entry in iterator)
        if entries:
            existing[source] = entries[:8]
    if existing:
        detail = ", ".join(
            f"{source}={entries!r}" for source, entries in sorted(existing.items())
        )
        raise SystemExit(
            "the six ordinary immutable evidence sources may each be run exactly "
            f"once; existing run entries: {detail}"
        )


def _freeze_evidence_arguments(args: argparse.Namespace) -> None:
    """Validate the exact release crawl shape before Scrapy imports or mutation."""

    if not args.evidence_crawl:
        return
    requested = tuple(args.only or ())
    if requested not in {
        ORDINARY_EVIDENCE_SOURCE_ORDER,
        ("supremecourt",),
    }:
        raise SystemExit(
            "--evidence-crawl requires the six ordinary sources in frozen order "
            f"{ORDINARY_EVIDENCE_SOURCE_ORDER!r} or only supremecourt"
        )
    expected_start = EVIDENCE_START_DATE.isoformat()
    expected_end = EVIDENCE_END_DATE.isoformat()
    if args.start_date not in {None, expected_start} or args.end_date not in {
        None,
        expected_end,
    }:
        raise SystemExit(
            f"--evidence-crawl interval is frozen at {expected_start} through "
            f"{expected_end}"
        )
    args.start_date = expected_start
    args.end_date = expected_end
    if requested == ("supremecourt",):
        if not args.supreme_parent_run_id:
            raise SystemExit(
                "Supreme Court evidence crawl requires --supreme-parent-run-id "
                "(use 'none' only for the explicit first run)"
            )
        if args.max_runtime_seconds not in {None, 14400}:
            raise SystemExit("Supreme Court evidence runtime is fixed at 14400 seconds")
        args.max_runtime_seconds = 14400
    elif args.supreme_parent_run_id is not None:
        raise SystemExit("--supreme-parent-run-id is valid only for supremecourt")


def select_spiders(available, requested) -> list[str]:
    """Resolve the spider list in canonical order, validating any --only names."""
    available = set(available)
    if requested:
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise SystemExit(
                f"unknown spider(s): {', '.join(unknown)}. "
                f"available: {', '.join(sorted(available))}"
            )
        wanted = set(requested)
    else:
        wanted = available
    ordered = [name for name in SPIDER_ORDER if name in wanted]
    # Include any spiders not in SPIDER_ORDER (future additions) deterministically.
    ordered += sorted(wanted - set(ordered))
    return ordered


def crawl_quality_issues(crawlers) -> list[str]:
    """Return actionable reasons a combined run must not be treated as complete."""
    issues: list[str] = []
    for crawler in crawlers:
        stats = crawler.stats.get_stats()
        spider = getattr(crawler, "spider", None)
        name = getattr(spider, "name", crawler.spidercls.name)
        reason = stats.get("finish_reason")
        attester = getattr(crawler, "completion_attestation", None)
        evaluation = evaluate_crawl_quality(
            stats,
            name,
            reason,
            spider_errors=getattr(attester, "_spider_errors", 0),
            item_errors=getattr(attester, "_item_errors", 0),
            reconcilers=getattr(spider, "_pagination_reconcilers", {}),
        )
        issues.extend(evaluation.issues)
    return issues


def crawl_attestation_issues(crawlers) -> list[str]:
    """Strictly reload each exact successful terminal record and item binding."""

    issues: list[str] = []
    for crawler in crawlers:
        spider = getattr(crawler, "spider", None)
        name = getattr(spider, "name", crawler.spidercls.name)
        if spider is None:
            issues.append(f"{name}: spider never produced exact run evidence")
            continue
        run_id = getattr(spider, "run_id", None)
        record_path = getattr(spider, "run_metadata_path", None)
        items_path = getattr(spider, "items_path", None)
        if not run_id or record_path is None or items_path is None:
            issues.append(f"{name}: exact run-scoped completion paths are unavailable")
            continue
        try:
            verify_terminal_record(
                record_path,
                expected_source=name,
                expected_run_id=run_id,
                expected_items_path=items_path,
            )
        except Exception as exc:  # noqa: BLE001 - every verification failure exits 1
            issues.append(
                f"{name}: exact completion evidence rejected for {run_id}: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
    return issues


def main(argv=None) -> None:
    args = parse_args(argv)
    validated_code_identity = _validate_evidence_release_gate(args)
    _freeze_evidence_arguments(args)
    if (
        args.evidence_crawl
        and tuple(args.only or ()) == ORDINARY_EVIDENCE_SOURCE_ORDER
    ):
        _reject_existing_ordinary_evidence_destinations()
    _bootstrap_project_dir()

    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    from legal_scrapers.extensions import declare_spiders, finish_dashboard

    settings = get_project_settings()
    if args.evidence_crawl:
        if validated_code_identity is None:
            raise SystemExit("evidence release identity was not validated")
        validated_code_revision, validated_code_identity_sha256 = (
            validated_code_identity
        )
        settings.set("EVIDENCE_CRAWL_ENABLED", True, priority="cmdline")
        settings.set(
            "EVIDENCE_CODE_REVISION", validated_code_revision, priority="cmdline"
        )
        settings.set(
            "EVIDENCE_CODE_IDENTITY_SHA256",
            validated_code_identity_sha256,
            priority="cmdline",
        )
        settings.set("HTTPCACHE_ENABLED", False, priority="cmdline")
        settings.set("DEDUP_ENABLED", False, priority="cmdline")
    if args.no_dedup:
        settings.set("DEDUP_ENABLED", False, priority="cmdline")
    if args.no_progress:
        settings.set("PROGRESS_DISPLAY_ENABLED", False, priority="cmdline")
    if args.max_runtime_seconds:
        # CloseSpider asks the engine to close; Scrapy still drains the active request and
        # item pipeline before ``spider.closed`` atomically finalizes the partial artifact.
        settings.set(
            "CLOSESPIDER_TIMEOUT", args.max_runtime_seconds, priority="cmdline"
        )

    process = CrawlerProcess(settings)
    spiders = select_spiders(process.spider_loader.list(), args.only)
    if not spiders:
        raise SystemExit("no spiders to run")

    spider_kwargs = {}
    if args.start_date:
        spider_kwargs["start_date"] = args.start_date
    if args.end_date:
        spider_kwargs["end_date"] = args.end_date

    declare_spiders(spiders)
    crawlers = []
    for name in spiders:
        crawler = process.create_crawler(name)
        crawlers.append(crawler)
        kwargs = dict(spider_kwargs)
        if name == "supremecourt" and args.supreme_parent_run_id is not None:
            kwargs["parent_run_id"] = args.supreme_parent_run_id
        process.crawl(crawler, **kwargs)

    try:
        process.start()  # blocks until all crawls finish
    finally:
        finish_dashboard()
    issues = crawl_quality_issues(crawlers)
    issues.extend(crawl_attestation_issues(crawlers))
    if issues:
        print(
            "crawl did not satisfy the production completeness gate:", file=sys.stderr
        )
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
