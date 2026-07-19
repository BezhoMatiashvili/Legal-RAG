# Scrapy settings for the legal_scrapers project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "legal_scrapers"

SPIDER_MODULES = ["legal_scrapers.spiders"]
NEWSPIDER_MODULE = "legal_scrapers.spiders"

ADDONS = {}


# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Concurrency and throttling settings.
# Moderate, still-polite rate for the live .gov.ge sites: ~2 req/s/domain
# steady-state. Every scraped document needs 2-3 sequential requests
# (list -> detail -> optional doc/PDF), so an aggressive per-request delay
# stalls item output entirely. AutoThrottle (below) still backs off on latency.
CONCURRENT_REQUESTS = 16
DOWNLOAD_DELAY = 1.5
CONCURRENT_REQUESTS_PER_DOMAIN = 3
RANDOMIZE_DOWNLOAD_DELAY = True

# Cap response size so an untrusted PDF/DOCX/HTML download can't OOM the crawler.
DOWNLOAD_MAXSIZE = 104857600   # 100 MB hard limit
DOWNLOAD_WARNSIZE = 33554432   # 32 MB warning

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "legal_scrapers.middlewares.MatsneSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
   "legal_scrapers.middlewares.RotateUserAgentMiddleware": 543
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
EXTENSIONS = {
    "legal_scrapers.extensions.CompletionAttestationExtension": 10,
    "legal_scrapers.extensions.RotatingSpiderLogExtension": 20,
    "legal_scrapers.extensions.LiveProgressExtension": 100,
}

# Live terminal progress panel. Auto-disables when stdout is not a TTY (pipes,
# CI, cron). Force off for any run with `-s PROGRESS_DISPLAY_ENABLED=False`.
PROGRESS_DISPLAY_ENABLED = True

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    "legal_scrapers.pipelines.DedupPipeline": 100,
}

# Cross-run deduplication. Each spider records scraped-document identities in
# artifacts/<spider>/seen.sqlite and skips already-seen documents (no detail
# fetch, no re-emit). Force a full re-scrape with `-s DEDUP_ENABLED=False`.
DEDUP_ENABLED = True
DEDUP_REFRESH_DEFAULT_DAYS = 30
DEDUP_REFRESH_TAS_DAYS = 7
DEDUP_REFRESH_PENDING_DAYS = 1
DEDUP_REFRESH_LIMIT = 2000

# Use the asyncio reactor process-wide. Required so scrapy-playwright (the tas
# spider) works under the single shared reactor when all spiders run together
# via `python -m legal_scrapers.run`; also hardens `scrapy crawl tas`.
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 1
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 10
# The average number of requests Scrapy should be sending in parallel to
# each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = True

# Enable and configure HTTP caching (disabled by default).
# NOTE: with the cache on, re-running the SAME date window serves cached list/detail
# pages for HTTPCACHE_EXPIRATION_SECS (good for idempotent re-runs, but it will NOT pick
# up docs published into that exact window in the meantime). Incremental runs with a
# moving end_date stay fresh. Override per run with `-s HTTPCACHE_ENABLED=False`.
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 604800
HTTPCACHE_DIR = "httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [401, 403, 429, 500, 502, 503, 504]
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
FEED_EXPORT_ENCODING = "utf-8"

LOG_LEVEL = "INFO"
