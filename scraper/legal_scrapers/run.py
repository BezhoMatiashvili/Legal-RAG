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
import sys
from pathlib import Path

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
        "--no-dedup",
        action="store_true",
        help="Disable cross-run dedup for this run (re-scrape everything).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress display.",
    )
    return parser.parse_args(argv)


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


def main(argv=None) -> None:
    args = parse_args(argv)
    _bootstrap_project_dir()

    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    from legal_scrapers.extensions import declare_spiders, finish_dashboard

    settings = get_project_settings()
    if args.no_dedup:
        settings.set("DEDUP_ENABLED", False, priority="cmdline")
    if args.no_progress:
        settings.set("PROGRESS_DISPLAY_ENABLED", False, priority="cmdline")

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
    for name in spiders:
        process.crawl(name, **spider_kwargs)

    try:
        process.start()  # blocks until all crawls finish
    finally:
        finish_dashboard()


if __name__ == "__main__":
    main()
