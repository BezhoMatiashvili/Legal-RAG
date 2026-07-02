import io
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "matsne"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from scrapy.exceptions import NotConfigured  # noqa: E402
from scrapy.utils.test import get_crawler  # noqa: E402

from matsne.extensions import (  # noqa: E402
    LiveProgressExtension,
    ProgressSnapshot,
    extract_title,
    render_panel,
    render_table,
)
from matsne.items import EcdItem, MatsneItem, SupremecourtItem, TasItem  # noqa: E402


def _render_to_text(snap: ProgressSnapshot, frame: str = "⠋") -> str:
    console = Console(file=io.StringIO(), force_terminal=True, width=70)
    console.print(render_panel(snap, frame))
    return console.file.getvalue()


class RenderPanelTests(unittest.TestCase):
    def _snapshot(self, **overrides) -> ProgressSnapshot:
        base = dict(
            spider="ecd",
            elapsed_s=134.0,  # 02:14
            items=128,
            requests=410,
            responses=402,
            status_counts={200: 398, 404: 3, 500: 1},
            queue=12,
            errors=0,
            warnings=1,
            last_item="სამოქალაქო საქმე №ა-1234",
        )
        base.update(overrides)
        return ProgressSnapshot(**base)

    def test_running_panel_shows_core_metrics(self):
        out = _render_to_text(self._snapshot())
        self.assertIn("ecd", out)
        self.assertIn("scraping", out)
        self.assertIn("128", out)          # items
        self.assertIn("57/min", out)       # 128 items over 134s ≈ 57/min
        self.assertIn("410 sent", out)
        self.assertIn("402 done", out)
        self.assertIn("200×398", out)
        self.assertIn("404×3", out)
        self.assertIn("500×1", out)
        self.assertIn("12 pending", out)
        self.assertIn("02:14", out)        # elapsed mm:ss
        self.assertIn("ა-1234", out)       # last item label

    def test_elapsed_includes_hours_when_long(self):
        out = _render_to_text(self._snapshot(elapsed_s=3725.0))  # 1:02:05
        self.assertIn("1:02:05", out)

    def test_done_panel_shows_finish_reason(self):
        out = _render_to_text(self._snapshot(done=True, finish_reason="finished"))
        self.assertIn("finished", out)
        self.assertNotIn("scraping", out)

    def test_empty_status_counts_renders_placeholder(self):
        out = _render_to_text(self._snapshot(status_counts={}))
        self.assertIn("status", out)  # row still present, no crash


class RenderTableTests(unittest.TestCase):
    def _snaps(self):
        return [
            ProgressSnapshot("matsne", 401.0, 842, 1200, 1190, {200: 1190},
                             done=True, finish_reason="finished"),
            ProgressSnapshot("ecd", 401.0, 128, 410, 402, {200: 401, 500: 1}, errors=1),
            ProgressSnapshot("tbappeal", 0.0, 0, 0, 0, finish_reason="queued"),
        ]

    def _render(self):
        console = Console(file=io.StringIO(), force_terminal=True, width=80)
        console.print(render_table(self._snaps(), "⠋", 401.0))
        return console.file.getvalue()

    def test_has_each_spider_row(self):
        out = self._render()
        for name in ("matsne", "ecd", "tbappeal"):
            self.assertIn(name, out)

    def test_shows_done_running_queued_states(self):
        out = self._render()
        self.assertIn("finished", out)  # matsne finished
        self.assertIn("queued", out)    # tbappeal not yet opened

    def test_totals_footer(self):
        out = self._render()
        self.assertIn("total", out)
        self.assertIn("970 items", out)   # 842 + 128 + 0
        self.assertIn("06:41", out)       # elapsed 401s


class ExtractTitleTests(unittest.TestCase):
    def test_prefers_title_field(self):
        item = MatsneItem()
        item["title"] = "  კანონი საქართველოს შრომის კოდექსი  "
        item["document_number"] = "123-XIIc"
        self.assertEqual(extract_title(item), "კანონი საქართველოს შრომის კოდექსი")

    def test_falls_back_to_case_no_for_ecd(self):
        item = EcdItem()
        item["case_no"] = "ა-1234-22"
        item["document_id"] = "1-5861405"
        self.assertEqual(extract_title(item), "ა-1234-22")

    def test_supremecourt_uses_subject_before_case_number(self):
        item = SupremecourtItem()
        item["case_number"] = "ას-100-2024"
        item["subject"] = " administrative dispute"
        self.assertEqual(extract_title(item), "administrative dispute")

    def test_truncates_long_values(self):
        item = TasItem()
        item["address"] = "x" * 200  # not a title key; falls through length guard
        item["document_no"] = "y" * 100  # title key, but very long -> truncated
        result = extract_title(item)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 48)
        self.assertTrue(result.endswith("…"))

    def test_returns_none_when_no_string_fields(self):
        item = EcdItem()
        self.assertIsNone(extract_title(item))


class FromCrawlerTests(unittest.TestCase):
    def test_disabled_setting_raises_not_configured(self):
        crawler = get_crawler(settings_dict={"PROGRESS_DISPLAY_ENABLED": False})
        with self.assertRaises(NotConfigured):
            LiveProgressExtension.from_crawler(crawler)

    def test_non_tty_raises_not_configured(self):
        crawler = get_crawler(settings_dict={"PROGRESS_DISPLAY_ENABLED": True})
        with mock.patch.object(sys.stdout, "isatty", return_value=False):
            with self.assertRaises(NotConfigured):
                LiveProgressExtension.from_crawler(crawler)

    def test_enabled_tty_builds_extension(self):
        crawler = get_crawler(settings_dict={"PROGRESS_DISPLAY_ENABLED": True})
        with mock.patch.object(sys.stdout, "isatty", return_value=True):
            ext = LiveProgressExtension.from_crawler(crawler)
        self.assertIsInstance(ext, LiveProgressExtension)
        self.assertIs(ext.crawler, crawler)


if __name__ == "__main__":
    unittest.main()
