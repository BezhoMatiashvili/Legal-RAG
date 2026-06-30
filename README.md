# Legal Scrapers

Scrapy spiders that collect Georgian legal documents (legislation + court
decisions). Each source is one spider; they all share the same scaffolding
(`BaseLegalSpider`): `start_date`/`end_date` arguments, polite throttling, and
per-run JSONL artifacts.

## Setup

```bash
uv sync
uv run playwright install chromium   # only needed for the `tas` spider
```

## Running a spider

All spiders run from the Scrapy project directory and share the same interface:

```bash
cd matsne
uv run scrapy crawl <spider> -a start_date=YYYY-MM-DD -a end_date=YYYY-MM-DD
```

- Dates use `YYYY-MM-DD`. If `end_date` is omitted it defaults to today; if
  `start_date` is omitted it defaults to each spider's `DEFAULT_SCRAPING_START_DATE`.
- Output is written under `artifacts/<spider>/`:
  - `runs/<run_id>/items.jsonl` — archived output for one run
  - `runs/<run_id>/spider.log`, `runs/<run_id>/run.json` — log + run metadata
  - `latest/items.jsonl`, `latest/run.json` — pointer to the most recent run
- `artifacts/` is git-ignored (local runtime data).

A live progress panel is shown automatically while a crawl runs (it auto-disables
when output is not a terminal). Disable it with `-s PROGRESS_DISPLAY_ENABLED=False`.

## Running all spiders together

To run every spider (or a subset) concurrently in one process with a single
combined progress table:

```bash
cd matsne
uv run python -m matsne.run --start-date YYYY-MM-DD --end-date YYYY-MM-DD
uv run python -m matsne.run --only ecd tbappeal --start-date 2026-06-01 --end-date 2026-06-30
uv run python -m matsne.run --no-dedup        # force a full re-scrape this run
```

Each spider targets a different domain, so per-domain politeness (delay,
AutoThrottle, per-domain concurrency) is identical to running them one at a time.
Each writes to its own `artifacts/<spider>/...` tree as usual.

## Scrape and ingest together

To crawl and embed into the RAG index at the same time, run the watcher alongside the
scraper. The repo-root launcher starts both with one command:

```bash
python3 run_all.py --start-date 2026-06-01 --end-date 2026-06-30
python3 run_all.py --only ecd tbappeal --start-date 2026-06-01 --end-date 2026-06-30
```

The watcher first backfills everything already scraped (oldest→newest), then ingests new
documents as the scrape produces them. When the scrape finishes the watcher keeps running
and waits for future documents; press **Ctrl-C** to stop it (it drains anything outstanding
first). Qdrant must be running (`cd ingest && docker compose up -d`).

Prefer two terminals for development (independent restarts):

```bash
# terminal A — continuous ingest
cd ingest && uv run python -m ingest watch --source all
# terminal B — scrape
cd matsne && uv run python -m matsne.run --start-date 2026-06-01 --end-date 2026-06-30
```

See `ingest/README.md` ("Continuous watch mode") for the watcher's flags and guarantees.

## Deduplication

Each spider remembers every document it has scraped in
`artifacts/<spider>/seen.sqlite` (keyed on the same identity the downstream
`ingest/` uses). On later runs an already-scraped document is skipped **before**
its detail page is fetched, so it is neither re-downloaded nor re-emitted — and
because each run's `latest/items.jsonl` then holds only new documents, `ingest`
stops re-embedding old ones too. The number skipped is reported in the run stats
as `dedup/skipped`.

- This is **skip-forever by identity**: a document already scraped is not
  re-fetched even if its content later changes on the source site.
- To force a full re-scrape (re-fetch and re-emit everything), pass
  `-s DEDUP_ENABLED=False` to `scrapy crawl` or `--no-dedup` to `python -m matsne.run`.
- To reset one spider's memory, delete its `artifacts/<spider>/seen.sqlite`.

## Spiders

| Spider | Source | Data | Date filter | Body |
|---|---|---|---|---|
| `matsne` | matsne.gov.ge | legislation (HTML) | server-side | HTML → Markdown |
| `ecd` | ecd.court.ge | common-court decisions (JSON API) | server-side | plain text |
| `constcourt` | constcourt.ge | Constitutional Court acts (HTML) | server-side | HTML → Markdown (+DOCX for claims) |
| `napr` | napr.gov.ge | Public Registry legal practice (JSON list) | server-side | PDF → text |
| `tbappeal` | tbappeal.court.ge | Court of Appeals "interesting decisions" (HTML) | **in-spider** | HTML → Markdown |
| `supremecourt` | supremecourt.ge | Supreme Court cases (HTML via AJAX) | server-side | HTML → Markdown (+DOCX URL) |
| `tas` | docs.tbilisi.gov.ge (via tas.ge) | Tbilisi Architecture Service docs (ExtJS/DWR) | server-side | metadata + nomenclature |

### Examples

```bash
cd matsne
uv run scrapy crawl ecd          -a start_date=2020-01-01 -a end_date=2020-01-31
uv run scrapy crawl constcourt   -a start_date=2026-06-01 -a end_date=2026-06-30
uv run scrapy crawl napr         -a start_date=2024-12-01 -a end_date=2024-12-31
uv run scrapy crawl tbappeal     -a start_date=2017-01-01 -a end_date=2026-12-31
uv run scrapy crawl supremecourt -a start_date=2024-12-01 -a end_date=2024-12-31
uv run scrapy crawl tas          -a start_date=2024-06-03 -a end_date=2024-06-03
```

### Per-spider notes

- **ecd** — pure JSON API (no HTML scraping). Covers all three court instances; the
  body is already plain text. The corpus is largely historical (richest around
  ~2019–2020), so recent windows can be empty.
- **constcourt** — judgments carry full text in HTML; constitutional *claims* show only
  a teaser there, so the spider downloads the attached DOCX for those to get the full
  body. Visible dates are Georgian month names (stored verbatim; filtering is
  server-side).
- **napr** — list metadata comes as JSON (double-encoded); the decision body is a PDF
  that the spider downloads and extracts to text. `sender` is already masked at source.
- **tbappeal** — the only source with **no server-side date filter**: it crawls the
  small (~7-page) reverse-chronological category, filters each post by its listed date,
  and dedupes by slug. The full ruling is linked as `pdf_url`. Uses the apex domain
  (`tbappeal.court.ge`); the `www.` host does not resolve.
- **supremecourt** — crawls `/ka/getCases`, which the site's **robots.txt disallows**.
  This spider deliberately overrides `ROBOTSTXT_OBEY=False` (scoped to itself) per an
  explicit project-owner decision (2026-06-29); the rest of the project keeps
  `ROBOTSTXT_OBEY=True`.
- **tas** — requires a headless browser (`scrapy-playwright` + Chromium). It drives the
  ExtJS app's own DWR data layer; the public corpus is very large (~500k docs), so use
  narrow date windows. Applicant/architect/cadastral fields are not in the public list
  payload and are not captured.

## Tests

```bash
uv run python -m unittest discover tests
```
