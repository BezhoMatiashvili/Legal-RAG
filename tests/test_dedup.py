import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from scrapy.exceptions import DropItem  # noqa: E402
from scrapy.settings import Settings  # noqa: E402

from legal_scrapers.pipelines import DedupPipeline  # noqa: E402
from legal_scrapers.spiders.base import BaseLegalSpider  # noqa: E402


class _Spider(BaseLegalSpider):
    name = "dedup_test"
    DEDUP_KEY = ("document_id",)


class _CompositeSpider(BaseLegalSpider):
    name = "dedup_test_composite"
    DEDUP_KEY = ("case_id", "chamber")


class _NoKeySpider(BaseLegalSpider):
    name = "dedup_test_nokey"
    DEDUP_KEY = None


def _make(cls, tmp, enabled=True):
    spider = cls(start_date=None, end_date=None)
    spider.run_id = "run-test"
    settings = Settings({"DEDUP_ENABLED": enabled})
    with mock.patch.object(BaseLegalSpider, "ARTIFACTS_ROOT", Path(tmp)):
        spider.open_dedup_store(settings)
    # A minimal crawler stub so the pipeline's stats.inc_value works.
    spider.crawler = types.SimpleNamespace(
        stats=types.SimpleNamespace(inc_value=lambda *a, **k: None)
    )
    return spider


class DedupKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = __import__("tempfile").TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

    def test_simple_key(self):
        sp = _make(_Spider, self.tmp)
        self.assertEqual(sp.dedup_key({"document_id": "1-58"}), "1-58")

    def test_composite_key_order(self):
        sp = _make(_CompositeSpider, self.tmp)
        self.assertEqual(
            sp.dedup_key({"case_id": "777", "chamber": "1"}), "777:1"
        )

    def test_missing_field_returns_none(self):
        sp = _make(_CompositeSpider, self.tmp)
        self.assertIsNone(sp.dedup_key({"case_id": "777"}))          # chamber missing
        self.assertIsNone(sp.dedup_key({"case_id": "777", "chamber": ""}))

    def test_values_are_stripped(self):
        sp = _make(_Spider, self.tmp)
        self.assertEqual(sp.dedup_key({"document_id": "  X  "}), "X")

    def test_disabled_spider_without_key(self):
        sp = _make(_NoKeySpider, self.tmp)
        self.assertFalse(sp.dedup_enabled)
        self.assertIsNone(sp.dedup_key({"document_id": "X"}))
        self.assertFalse(sp.is_seen({"document_id": "X"}))


class SeenStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = __import__("tempfile").TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

    def test_mark_then_seen(self):
        sp = _make(_Spider, self.tmp)
        self.assertFalse(sp.is_seen({"document_id": "A"}))
        self.assertTrue(sp.mark_seen({"document_id": "A"}))
        self.assertTrue(sp.is_seen({"document_id": "A"}))
        # Re-marking the same key is a no-op.
        self.assertFalse(sp.mark_seen({"document_id": "A"}))

    def test_persists_across_runs(self):
        sp1 = _make(_Spider, self.tmp)
        sp1.mark_seen({"document_id": "B"})
        sp1._dedup_conn.close()
        # A fresh spider of the same name reloads the persisted key.
        sp2 = _make(_Spider, self.tmp)
        self.assertTrue(sp2.is_seen({"document_id": "B"}))

    def test_disabled_records_nothing(self):
        sp = _make(_Spider, self.tmp, enabled=False)
        self.assertFalse(sp.dedup_enabled)
        self.assertFalse(sp.mark_seen({"document_id": "C"}))
        self.assertFalse(sp.is_seen({"document_id": "C"}))


class DedupPipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = __import__("tempfile").TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)
        self.pipe = DedupPipeline()

    def test_new_item_passes_and_is_recorded(self):
        sp = _make(_Spider, self.tmp)
        item = {"document_id": "A", "body_markdown": "x"}
        self.assertIs(self.pipe.process_item(item, sp), item)
        self.assertIn("A", sp._seen_keys)

    def test_duplicate_item_dropped(self):
        sp = _make(_Spider, self.tmp)
        self.pipe.process_item({"document_id": "A"}, sp)
        with self.assertRaises(DropItem):
            self.pipe.process_item({"document_id": "A"}, sp)

    def test_passthrough_when_disabled(self):
        sp = _make(_Spider, self.tmp, enabled=False)
        item1 = {"document_id": "A"}
        item2 = {"document_id": "A"}  # identical key
        self.assertIs(self.pipe.process_item(item1, sp), item1)
        # Even a second identical item passes when dedup is off (no DropItem).
        self.assertIs(self.pipe.process_item(item2, sp), item2)

    def test_incomplete_identity_never_dropped(self):
        sp = _make(_CompositeSpider, self.tmp)
        item = {"case_id": "777"}  # chamber missing -> key is None -> pass
        self.assertIs(self.pipe.process_item(item, sp), item)

    def test_empty_body_is_exported_but_not_persisted_as_seen(self):
        sp = _make(_Spider, self.tmp)
        item = {"document_id": "EMPTY", "body_markdown": ""}

        self.assertIs(self.pipe.process_item(item, sp), item)
        self.assertNotIn("EMPTY", sp._seen_keys)


if __name__ == "__main__":
    unittest.main()
