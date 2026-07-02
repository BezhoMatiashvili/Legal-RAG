import sys
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from legal_scrapers.utils.dates import (  # noqa: E402
    date_part,
    dotnet_date_to_iso,
    iso_to_dotted,
    iso_to_slashed,
    iso_to_year_slashed,
    parse_dotted,
)
from legal_scrapers.utils.json_api import loads_maybe_double  # noqa: E402
from legal_scrapers.utils.text import plain_text_to_markdown  # noqa: E402


class DateConversionTests(unittest.TestCase):
    def test_format_converters(self):
        d = date(2024, 3, 9)
        self.assertEqual(iso_to_dotted(d), "09-03-2024")
        self.assertEqual(iso_to_slashed(d), "09/03/2024")
        self.assertEqual(iso_to_year_slashed(d), "2024/03/09")

    def test_dotnet_date_to_iso(self):
        # 1588260918000 ms -> 2020-04-30 (UTC)
        self.assertEqual(dotnet_date_to_iso("/Date(1588260918000)/"), "2020-04-30")
        self.assertEqual(dotnet_date_to_iso("/Date(1588260918000+0400)/"), "2020-04-30")
        self.assertIsNone(dotnet_date_to_iso(None))
        self.assertIsNone(dotnet_date_to_iso(""))
        self.assertIsNone(dotnet_date_to_iso("not a date"))

    def test_date_part(self):
        self.assertEqual(date_part("2024-06-03 08:00:07+00:00"), "2024-06-03")
        self.assertEqual(date_part("2024-12-27 00:00:00"), "2024-12-27")
        self.assertEqual(date_part("2024-06-03"), "2024-06-03")
        self.assertIsNone(date_part(None))

    def test_parse_dotted(self):
        self.assertEqual(parse_dotted("09-03-2024"), date(2024, 3, 9))
        self.assertEqual(parse_dotted("09/03/2024"), date(2024, 3, 9))
        self.assertEqual(parse_dotted("09.03.2024"), date(2024, 3, 9))
        self.assertIsNone(parse_dotted("garbage"))
        self.assertIsNone(parse_dotted(None))


class TextTests(unittest.TestCase):
    def test_plain_text_to_markdown_collapses_whitespace(self):
        self.assertEqual(plain_text_to_markdown("a\n\n\n\nb"), "a\n\nb")
        self.assertEqual(plain_text_to_markdown("  trailing   \nx"), "trailing\nx")
        self.assertEqual(plain_text_to_markdown(None), "")
        self.assertEqual(plain_text_to_markdown("\r\nwin\r\n"), "win")


class JsonApiTests(unittest.TestCase):
    def test_loads_maybe_double(self):
        # Single-encoded (ecd style)
        self.assertEqual(loads_maybe_double('{"a": 1}'), {"a": 1})
        # Double-encoded (napr style: a quoted JSON string)
        self.assertEqual(loads_maybe_double('"{\\"a\\": 1}"'), {"a": 1})


if __name__ == "__main__":
    unittest.main()
