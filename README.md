# Legal Scrapers

## Matsne spider

Run the spider from the Scrapy project directory:

```bash
cd matsne
uv run scrapy crawl matsne -a start_date=2026-06-22
```

Use an explicit interval when you need a reproducible run:

```bash
cd matsne
uv run scrapy crawl matsne -a start_date=2026-06-22 -a end_date=2026-06-25
```

Dates use `YYYY-MM-DD`. If `end_date` is omitted, the spider uses today's date.
If `start_date` is omitted, it uses the default configured in
`matsne/matsne/spiders/matsne_spider.py`.

Generated results and logs are written under `artifacts/matsne/`:

- `runs/<run_id>/items.jsonl` is the archived output for one run.
- `runs/<run_id>/spider.log` is the archived log for one run.
- `runs/<run_id>/run.json` records the run's date interval and file paths.
- `latest/items.jsonl` and `latest/run.json` point to the most recent run for
  downstream scripts or quick inspection.

The `artifacts/` directory is ignored by git because scrape results and logs are
local runtime data.
