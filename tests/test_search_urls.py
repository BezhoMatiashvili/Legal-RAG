import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.utils.search_urls import (  # noqa: E402
    ADDITIONAL_STATUSES,
    DATE_FORMAT,
    TOPICS,
    build_search_url,
    generate_start_url_batches,
    sub_windows,
)


class SubWindowsTests(unittest.TestCase):
    def _assert_contiguous_gapfree_clamped(self, windows, start, end):
        self.assertTrue(windows)
        self.assertEqual(windows[0][0], start)          # first clamps to start
        self.assertEqual(windows[-1][1], end)           # last clamps to end
        for (s, e) in windows:
            self.assertLessEqual(s, e)
        for (_, e), (s2, _) in zip(windows, windows[1:]):
            self.assertEqual(e + timedelta(days=1), s2)  # gap-free & non-overlapping

    def test_yearly_contiguous_gapfree_clamped(self):
        start, end = date(1998, 3, 15), date(2001, 2, 10)
        windows = sub_windows(start, end, "yearly")
        self._assert_contiguous_gapfree_clamped(windows, start, end)
        self.assertEqual(len(windows), 4)  # 1998,1999,2000,2001

    def test_monthly_full_year(self):
        windows = sub_windows(date(2015, 1, 1), date(2015, 12, 31), "monthly")
        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0], (date(2015, 1, 1), date(2015, 1, 31)))
        self.assertEqual(windows[-1], (date(2015, 12, 1), date(2015, 12, 31)))

    def test_monthly_clamps_partial_first_and_last(self):
        start, end = date(2015, 3, 15), date(2015, 5, 10)
        windows = sub_windows(start, end, "monthly")
        self._assert_contiguous_gapfree_clamped(windows, start, end)
        self.assertEqual(windows[0], (date(2015, 3, 15), date(2015, 3, 31)))

    def test_single_day(self):
        self.assertEqual(
            sub_windows(date(2020, 5, 5), date(2020, 5, 5), "yearly"),
            [(date(2020, 5, 5), date(2020, 5, 5))],
        )

    def test_start_after_end_is_empty(self):
        self.assertEqual(sub_windows(date(2021, 1, 1), date(2020, 1, 1), "yearly"), [])

    def test_unknown_granularity_raises(self):
        with self.assertRaises(ValueError):
            sub_windows(date(2020, 1, 1), date(2020, 2, 1), "weekly")


class GenerateStartUrlBatchesTests(unittest.TestCase):
    def test_bucket_rule_and_window_multiplication(self):
        start, end = date(2019, 1, 1), date(2020, 12, 31)
        first, deferred = generate_start_url_batches(start, end)

        n_windows = len(sub_windows(start, end, "yearly"))
        n_first = sum(1 for t in TOPICS for s in ADDITIONAL_STATUSES if t and s)
        n_deferred = sum(1 for t in TOPICS for s in ADDITIONAL_STATUSES if not t or not s)

        self.assertEqual(len(first), n_first * n_windows)
        self.assertEqual(len(deferred), n_deferred * n_windows)

        # Every deferred URL is a catch-all cell (empty label or empty status); no first is.
        for url in deferred:
            q = parse_qs(urlparse(url).query, keep_blank_values=True)
            self.assertTrue(q.get("label", [""])[0] == "" or q.get("additional_status", [""])[0] == "")
        for url in first:
            q = parse_qs(urlparse(url).query, keep_blank_values=True)
            self.assertNotEqual(q.get("label", [""])[0], "")
            self.assertNotEqual(q.get("additional_status", [""])[0], "")

    def test_urls_carry_the_window_date_bounds(self):
        first, deferred = generate_start_url_batches(date(2019, 1, 1), date(2019, 12, 31))
        for url in first + deferred:
            q = parse_qs(urlparse(url).query, keep_blank_values=True)
            self.assertEqual(q["publishing_date_fr[date]"][0], "01-01-2019")
            self.assertEqual(q["publishing_date_to[date]"][0], "31-12-2019")
            self.assertEqual(q["page"][0], "1")


class BuildSearchUrlTests(unittest.TestCase):
    def test_formats_dates_and_params(self):
        url = build_search_url(date(1994, 3, 15), date(1994, 12, 31), "კოდექსები", "ნორმატიული")
        q = parse_qs(urlparse(url).query, keep_blank_values=True)
        self.assertEqual(q["publishing_date_fr[date]"][0], date(1994, 3, 15).strftime(DATE_FORMAT))
        self.assertEqual(q["publishing_date_to[date]"][0], "31-12-1994")
        self.assertEqual(q["label"][0], "კოდექსები")
        self.assertEqual(q["additional_status"][0], "ნორმატიული")


if __name__ == "__main__":
    unittest.main()
