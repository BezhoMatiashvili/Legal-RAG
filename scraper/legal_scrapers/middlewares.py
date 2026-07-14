# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from scrapy import signals

from .utils.user_agents import generate_random_user_agent


class RotateUserAgentMiddleware:
    def process_request(self, request, spider):
        request.headers["User-Agent"] = generate_random_user_agent()


def _retry_after_seconds(
    value: str | bytes | None, *, now: datetime | None = None
) -> int:
    """Parse Retry-After delta-seconds or an HTTP date, with a polite one-second floor."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    value = (value or "").strip()
    if value.isdigit():
        return max(1, int(value))
    if value:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=UTC)
            seconds = int((target - (now or datetime.now(UTC))).total_seconds())
            return max(1, seconds)
        except TypeError, ValueError, OverflowError:
            pass
    return 8


class SupremecourtRetryAfterMiddleware:
    """Retry Supreme Court HTTP 429s after the server-requested delay.

    Scrapy's generic retry middleware includes 429 but retries immediately and ignores the
    response's ``Retry-After`` header.  This downloader middleware runs before it on the
    response path and returns a Deferred that yields a fresh, ``dont_filter`` request only
    after the delay.  Exhausted responses are passed onward with generic retry disabled so the
    spider errback records one durable unresolved failure instead of a second retry loop.
    """

    def process_response(self, request, response, spider):
        if spider.name != "supremecourt" or response.status != 429:
            return response

        retries = int(request.meta.get("supremecourt_retry_after_times", 0))
        max_retries = spider.crawler.settings.getint("RETRY_TIMES", 8)
        if retries >= max_retries:
            request.meta["dont_retry"] = True
            spider.crawler.stats.inc_value("retry_after/exhausted")
            return response

        cap = spider.crawler.settings.getint(
            "SUPREMECOURT_RETRY_AFTER_MAX_SECONDS", 600
        )
        delay = min(_retry_after_seconds(response.headers.get("Retry-After")), cap)
        retry = request.replace(dont_filter=True, priority=request.priority + 1)
        retry.meta["supremecourt_retry_after_times"] = retries + 1
        retry.meta["retry_times"] = retries + 1
        spider.crawler.stats.inc_value("retry_after/count")
        spider.crawler.stats.inc_value("retry/count")
        spider.crawler.stats.max_value("retry_after/max_seconds", delay)
        spider.logger.warning(
            "supremecourt: HTTP 429 for %s; honoring Retry-After=%ss (%s/%s)",
            request.url,
            delay,
            retries + 1,
            max_retries,
        )

        # Import the installed reactor lazily: importing it at module import time can select
        # the wrong reactor before Scrapy installs AsyncioSelectorReactor.
        from twisted.internet import reactor, task

        return task.deferLater(reactor, delay, lambda: retry)


class MatsneSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    async def process_start(self, start):
        # Called with an async iterator over the spider start() method or the
        # matching method of an earlier spider middleware.
        async for item_or_request in start:
            yield item_or_request

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class MatsneDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)
