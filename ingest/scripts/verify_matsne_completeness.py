#!/usr/bin/env python
"""Prove how complete the matsne corpus is, and list what's still missing.

Three independent audits against the live site (read-only):

  A. Advertised count (N of M): N = distinct ids in the scraper's seen.sqlite; M = matsne's
     own claimed total, read from the last-page number of the widest unfiltered search.
  B. Reference-closure: for a sample of scraped ids, call the backReferences JSON API and
     collect every referenced doc id; any not in seen.sqlite is a concrete miss (this is how
     the firearms base law id 14944 was found from amendment 5118306).
  C. ID-enum ground truth (bounded): probe a random sample of ids across [1, ID_MAX];
     matsne returns an ~839-byte "Access Denied" body for a non-existent id (HTTP is always
     200), so existence is detected by body, not status. For sampled real docs, check
     membership in seen.sqlite to estimate the missed fraction. A full walk of ~6.9M ids is
     infeasible at a polite rate, so this is a statistical sample, not exhaustive.

Concrete missing ids (from B, and any real-but-unseen ids from C) are written to
``residual_missing_ids.txt`` for a targeted seed re-fetch:
    scrapy crawl matsne -a seed_ids_file=<...>/residual_missing_ids.txt -s HTTPCACHE_ENABLED=False

Usage (from ingest/):
    .venv/bin/python scripts/verify_matsne_completeness.py \
        [--ref-sample 500] [--id-sample 500] [--id-max 6900000] [--delay 1.0] [--out residual_missing_ids.txt]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from ingest.config import load_config  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
SEARCH_TMPL = (
    "https://matsne.gov.ge/ka/document/search"
    "?publishing_date_fr%5Bdate%5D={fr}&publishing_date_to%5Bdate%5D={to}"
    "&type=all&page={page}&limit=100&label=&additional_status="
)
BACKREF_TMPL = "https://matsne.gov.ge/ka/document/backReferences/{id}?part_id=DOCUMENT"
VIEW_TMPL = "https://matsne.gov.ge/ka/document/view/{id}"
DEFAULT_ID_MAX = 6_900_000

# ---- pure helpers (unit-tested) ------------------------------------------------


def parse_last_page(html: str) -> int:
    """Largest ``page=N`` in a results page's pagination = matsne's advertised last page."""
    pages = [int(m) for m in re.findall(r"[?&]page=(\d+)", html)]
    return max(pages) if pages else 1


def expected_total(last_page: int, tail_count: int, limit: int = 100) -> int:
    """Total docs a search advertises: full pages before the last, plus the last page's rows."""
    if last_page <= 0:
        return 0
    return (last_page - 1) * limit + tail_count


def is_absent_page(body: bytes | str) -> bool:
    """True if a document/view body is matsne's ~839-byte 'Access Denied' non-existent page."""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    if len(text) < 2000 and ("Access Denied" in text or "Oops" in text):
        return True
    return False


def residual_ids(referenced: set[str], have: set[str]) -> set[str]:
    """Referenced ids we don't have — the reference-closure miss set."""
    return {i for i in referenced if i and i not in have}


# ---- network -------------------------------------------------------------------


def fetch(url: str, *, timeout: float = 30.0, retries: int = 2) -> tuple[int, bytes]:
    """GET with a browser UA (matsne is behind Cloudflare). Returns (status, body)."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read() if exc.fp else b""
        except Exception as exc:  # noqa: BLE001 - transient network; retry then give up
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! fetch failed: {url} ({last_exc})")
    return 0, b""


def load_seen_ids(cfg) -> set[str]:
    db = cfg.artifacts_root / "matsne" / "seen.sqlite"
    if not db.exists():
        raise SystemExit(f"seen.sqlite not found at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT key FROM seen")}
    finally:
        conn.close()


def referenced_ids(doc_id: str, delay: float) -> set[str]:
    url = BACKREF_TMPL.format(id=doc_id)
    status, body = fetch(url)
    time.sleep(delay)
    if status != 200 or not body:
        return set()
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return set()
    out: set[str] = set()
    for entry in data if isinstance(data, list) else []:
        rid = entry.get("id") if isinstance(entry, dict) else None
        if rid:
            out.add(str(rid))
    return out


# ---- audits --------------------------------------------------------------------


def audit_advertised(have: set[str]) -> int:
    """Audit A — N (have) vs M (matsne's advertised total). Returns M (0 on failure)."""
    today = date.today().strftime("%d-%m-%Y")
    url = SEARCH_TMPL.format(fr="01-01-1900", to=today, page=1)
    status, body = fetch(url)
    if status != 200 or not body:
        print("  A. advertised-count: could not fetch the widest search page.")
        return 0
    html = body.decode("utf-8", "replace")
    last_page = parse_last_page(html)
    # Fetch the last page to count its rows for an exact tail (else assume a full page).
    tail = 100
    if last_page > 1:
        status2, body2 = fetch(SEARCH_TMPL.format(fr="01-01-1900", to=today, page=last_page))
        if status2 == 200 and body2:
            html2 = body2.decode("utf-8", "replace")
            n_links = len(set(re.findall(r"/ka/document/view/(\d+)", html2)))
            if 0 < n_links <= 100:
                tail = n_links
    m = expected_total(last_page, tail)
    n = len(have)
    pct = (100.0 * n / m) if m else 0.0
    print(f"  A. advertised-count: have N={n:,}  vs matsne M≈{m:,} (last_page={last_page})  → {pct:.1f}%")
    return m


def audit_reference_closure(have: set[str], sample: int, delay: float) -> set[str]:
    """Audit B — sample scraped ids, follow backReferences, return referenced-but-unseen ids."""
    ids = list(have)
    random.shuffle(ids)
    ids = ids[:sample]
    missing: set[str] = set()
    for i, doc_id in enumerate(ids, 1):
        missing |= residual_ids(referenced_ids(doc_id, delay), have)
        if i % 50 == 0:
            print(f"     ...{i}/{len(ids)} sampled, {len(missing)} referenced-but-missing so far")
    print(f"  B. reference-closure: sampled {len(ids)} ids → {len(missing)} referenced ids not in corpus")
    return missing


def audit_id_enum(have: set[str], sample: int, id_max: int, delay: float) -> tuple[set[str], int, int]:
    """Audit C — random id probes. Returns (real-but-unseen ids, n_real, n_probed)."""
    missing: set[str] = set()
    n_real = 0
    rng = random.Random()
    for i in range(1, sample + 1):
        doc_id = str(rng.randint(1, id_max))
        status, body = fetch(VIEW_TMPL.format(id=doc_id))
        time.sleep(delay)
        if status == 200 and body and not is_absent_page(body):
            n_real += 1
            if doc_id not in have:
                missing.add(doc_id)
        if i % 50 == 0:
            print(f"     ...{i}/{sample} probed, {n_real} real, {len(missing)} real-but-missing")
    frac = (100.0 * (n_real - len(missing)) / n_real) if n_real else 0.0
    print(
        f"  C. id-enum sample: probed {sample} ids → {n_real} real docs; "
        f"corpus holds {frac:.1f}% of them ({len(missing)} sampled real ids missing)"
    )
    return missing, n_real, sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-sample", type=int, default=500, help="ids to sample for backReferences (audit B)")
    ap.add_argument("--id-sample", type=int, default=500, help="random ids to probe (audit C)")
    ap.add_argument("--id-max", type=int, default=DEFAULT_ID_MAX, help="upper bound of the id space")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (politeness)")
    ap.add_argument("--out", default="residual_missing_ids.txt", help="where to write concrete missing ids")
    ap.add_argument("--skip-id-enum", action="store_true", help="skip the slow audit C")
    args = ap.parse_args()

    cfg = load_config()
    have = load_seen_ids(cfg)
    print(f"matsne corpus completeness audit — {len(have):,} scraped ids in seen.sqlite\n")

    audit_advertised(have)
    residual = audit_reference_closure(have, args.ref_sample, args.delay)
    if not args.skip_id_enum:
        id_missing, _, _ = audit_id_enum(have, args.id_sample, args.id_max, args.delay)
        residual |= id_missing

    out = Path(args.out)
    out.write_text("\n".join(sorted(residual, key=lambda x: int(x) if x.isdigit() else 0)) + "\n",
                   encoding="utf-8")
    print(f"\nResidual concrete missing ids: {len(residual)} → {out.resolve()}")
    if residual:
        print("Seed-fetch them with:")
        print(f"  scrapy crawl matsne -a seed_ids_file={out.resolve()} -s HTTPCACHE_ENABLED=False")


if __name__ == "__main__":
    main()
