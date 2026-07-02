import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "matsne"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from matsne.run import parse_args, select_spiders  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
