import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from legal_scrapers.run import crawl_quality_issues, parse_args, select_spiders  # noqa: E402

ALL = ["matsne", "ecd", "constcourt", "napr", "supremecourt", "tas", "tbappeal"]


class SelectSpidersTests(unittest.TestCase):
    def test_default_is_all_in_canonical_order(self):
        # Available reported in arbitrary order -> returned in SPIDER_ORDER.
        self.assertEqual(select_spiders(sorted(ALL), None), ALL)

    def test_only_subset_keeps_canonical_order(self):
        self.assertEqual(
            select_spiders(ALL, ["tbappeal", "ecd"]), ["ecd", "tbappeal"]
        )

    def test_unknown_spider_raises(self):
        with self.assertRaises(SystemExit):
            select_spiders(ALL, ["nope"])

    def test_unordered_extra_spider_appended_deterministically(self):
        avail = ALL + ["zzz_future"]
        self.assertEqual(select_spiders(avail, None), ALL + ["zzz_future"])


class ParseArgsTests(unittest.TestCase):
    def test_dates_and_flags(self):
        args = parse_args(
            ["--start-date", "2026-06-01", "--end-date", "2026-06-30", "--no-dedup"]
        )
        self.assertEqual(args.start_date, "2026-06-01")
        self.assertEqual(args.end_date, "2026-06-30")
        self.assertTrue(args.no_dedup)
        self.assertFalse(args.no_progress)
        self.assertIsNone(args.only)

    def test_only_list(self):
        args = parse_args(["--only", "ecd", "tbappeal"])
        self.assertEqual(args.only, ["ecd", "tbappeal"])


class CrawlQualityGateTests(unittest.TestCase):
    @staticmethod
    def _crawler(name, **stats):
        return SimpleNamespace(
            spider=SimpleNamespace(name=name),
            spidercls=SimpleNamespace(name=name),
            stats=SimpleNamespace(get_stats=lambda: stats),
        )

    def test_clean_finished_crawl_has_no_issues(self):
        self.assertEqual(crawl_quality_issues([
            self._crawler("ecd", finish_reason="finished", item_scraped_count=3)
        ]), [])

    def test_quality_failures_and_callback_errors_fail_gate(self):
        issues = crawl_quality_issues([
            self._crawler(
                "ecd",
                finish_reason="finished",
                **{"quality/failures": 2, "spider_exceptions/ValueError": 1},
            )
        ])
        self.assertTrue(any("2 completeness" in issue for issue in issues))
        self.assertTrue(any("1 callback" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
