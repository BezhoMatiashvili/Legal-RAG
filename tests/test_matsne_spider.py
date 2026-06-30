import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scrapy.crawler import Crawler
from scrapy.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "matsne"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from matsne.spiders.matsne_spider import MatsneSpider  # noqa: E402


class MatsneSpiderDateTests(unittest.TestCase):
    def test_default_dates(self):
        spider = MatsneSpider()

        self.assertEqual(spider.scraping_start_date, date(2026, 6, 22))
        self.assertEqual(spider.scraping_end_date, date.today())

    def test_explicit_dates(self):
        spider = MatsneSpider(start_date="2026-06-22", end_date="2026-06-25")

        self.assertEqual(spider.scraping_start_date, date(2026, 6, 22))
        self.assertEqual(spider.scraping_end_date, date(2026, 6, 25))

    def test_invalid_date_format(self):
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            MatsneSpider(start_date="22-06-2026")

    def test_start_date_must_not_be_after_end_date(self):
        with self.assertRaisesRegex(ValueError, "start_date must be on or before end_date"):
            MatsneSpider(start_date="2026-06-26", end_date="2026-06-25")


class MatsneSpiderOutputTests(unittest.TestCase):
    def test_from_crawler_configures_run_outputs(self):
        original_artifacts_root = MatsneSpider.ARTIFACTS_ROOT

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                MatsneSpider.ARTIFACTS_ROOT = Path(tmp_dir) / "artifacts"
                crawler = Crawler(MatsneSpider, Settings())

                spider = MatsneSpider.from_crawler(
                    crawler,
                    start_date="2026-06-22",
                    end_date="2026-06-25",
                )

                feeds = crawler.settings.getdict("FEEDS")
                feed_paths = set(feeds)

                self.assertIn(str(spider.items_path), feed_paths)
                self.assertIn(str(spider.latest_items_path), feed_paths)
                self.assertEqual(crawler.settings.get("LOG_FILE"), str(spider.log_path))
                self.assertTrue(spider.run_metadata_path.exists())
                self.assertTrue(spider.latest_metadata_path.exists())

                metadata = json.loads(spider.run_metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["spider"], "matsne")
                self.assertEqual(metadata["start_date"], "2026-06-22")
                self.assertEqual(metadata["end_date"], "2026-06-25")
                self.assertEqual(metadata["items_path"], str(spider.items_path))
                self.assertEqual(metadata["log_path"], str(spider.log_path))
            finally:
                MatsneSpider.ARTIFACTS_ROOT = original_artifacts_root


if __name__ == "__main__":
    unittest.main()
