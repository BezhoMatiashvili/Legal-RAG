#!/usr/bin/env python3
"""Run the scraper and the continuous ingest watcher together.

The scraper (``scraper/``) and the ingester (``ingest/``) live in separate, mutually
incompatible Python environments (Scrapy on >=3.14, torch/BGE-M3 on <3.14), so they
cannot share one process. This launcher spawns each in its own ``uv`` env as a
subprocess, forwards Ctrl-C/SIGTERM to both, and prefixes their output.

The watcher starts first and backfills everything already scraped (oldest->newest),
then ingests new documents as the scraper produces them. When the scrape finishes the
watcher keeps running as a daemon, waiting for future documents; press Ctrl-C to stop
it (it drains anything outstanding before exiting).

    python3 run_all.py --start-date 2026-06-01 --end-date 2026-06-30
    python3 run_all.py --only ecd tbappeal --start-date 2026-06-01 --end-date 2026-06-30
"""

import argparse
import signal
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
INGEST_DIR = REPO_ROOT / "ingest"
SCRAPER_DIR = REPO_ROOT / "scraper"

_print_lock = threading.Lock()
CORPUS_SOURCES = ("matsne", "ecd", "constcourt", "napr", "tas", "tbappeal")
SCRAPER_SOURCES = (*CORPUS_SOURCES[:4], "supremecourt", *CORPUS_SOURCES[4:])


def selected_ingest_sources(only: list[str] | None) -> list[str]:
    """Production-index sources selected by a scrape request, in canonical order."""
    if only is None:
        return list(CORPUS_SOURCES)
    unknown = sorted(set(only) - set(SCRAPER_SOURCES))
    if unknown:
        raise SystemExit(f"unknown spider(s): {', '.join(unknown)}")
    selected = [source for source in CORPUS_SOURCES if source in only]
    if not selected:
        raise SystemExit(
            "--only selected no production corpus source (supremecourt is intentionally excluded)"
        )
    return selected


def build_watch_command(args) -> list[str]:
    sources = selected_ingest_sources(args.only)
    command = ["uv", "run", "python", "-m", "ingest"]
    if args.collection:
        command += ["--collection", args.collection]
    command += [
        "watch",
        "--source",
        ",".join(sources),
        "--poll-interval",
        str(args.poll_interval),
    ]
    return command


def _pump(proc: subprocess.Popen, tag: str) -> None:
    """Forward a child's merged stdout/stderr to ours, one line at a time, prefixed."""
    assert proc.stdout is not None
    for line in proc.stdout:
        with _print_lock:
            sys.stdout.write(f"[{tag}] {line}")
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_all", description="Run the scraper and the ingest watcher together.")
    parser.add_argument("--start-date", help="YYYY-MM-DD (forwarded to the scraper)")
    parser.add_argument("--end-date", help="YYYY-MM-DD (forwarded to the scraper)")
    parser.add_argument("--only", nargs="+", metavar="SPIDER",
                        help="limit scraping to these spiders (default: all)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="disable cross-run dedup for this scrape (re-scrape everything)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="watcher idle poll interval in seconds (default 5)")
    parser.add_argument("--collection", help="override the Qdrant collection name (watcher)")
    args = parser.parse_args()

    watch_cmd = build_watch_command(args)

    scrape_cmd = ["uv", "run", "python", "-m", "legal_scrapers.run"]
    if args.start_date:
        scrape_cmd += ["--start-date", args.start_date]
    if args.end_date:
        scrape_cmd += ["--end-date", args.end_date]
    if args.only:
        scrape_cmd += ["--only", *args.only]
    if args.no_dedup:
        scrape_cmd += ["--no-dedup"]

    # start_new_session so the terminal's Ctrl-C reaches only this launcher; we forward
    # it explicitly, giving the watcher a chance to drain before exiting.
    common = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  text=True, bufsize=1, start_new_session=True)

    print(f"[run_all] starting watcher: {' '.join(watch_cmd)}  (cwd={INGEST_DIR})")
    watcher = subprocess.Popen(watch_cmd, cwd=str(INGEST_DIR), **common)
    print(f"[run_all] starting scraper: {' '.join(scrape_cmd)}  (cwd={SCRAPER_DIR})")
    scraper = subprocess.Popen(scrape_cmd, cwd=str(SCRAPER_DIR), **common)

    procs = {"ingest": watcher, "scrape": scraper}
    threads = [threading.Thread(target=_pump, args=(p, tag), daemon=True)
               for tag, p in procs.items()]
    for t in threads:
        t.start()

    stopping = threading.Event()

    def _forward(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        print(f"\n[run_all] received signal {signum}; stopping children...")
        for p in procs.values():
            if p.poll() is None:
                try:
                    p.send_signal(signum)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    # Wait for the scrape to finish; the watcher then keeps running as a daemon.
    scraper.wait()
    if scraper.returncode and watcher.poll() is None:
        # A failed crawl must not leave an idle writer running forever or later report success.
        watcher.send_signal(signal.SIGTERM)
    if not stopping.is_set():
        if scraper.returncode:
            print(f"[run_all] scrape failed with exit {scraper.returncode}; stopping watcher.")
        else:
            print("[run_all] scrape finished; watcher still running and waiting for new "
                  "documents. Press Ctrl-C to stop it.")

    # Block until the watcher exits (on Ctrl-C, which we forward above).
    watcher.wait()

    # Final cleanup in case anything is still alive.
    for p in procs.values():
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
    for p in procs.values():
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    for t in threads:
        t.join(timeout=2)

    sys.exit(scraper.returncode or watcher.returncode or 0)


if __name__ == "__main__":
    main()
