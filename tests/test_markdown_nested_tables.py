import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from legal_scrapers.utils.markdown import html_to_markdown, safe_html_to_markdown  # noqa: E402


class NestedWrapperTableTests(unittest.TestCase):
    """Regression: matsne decrees nest wrapper tables inside wrapper tables.

    _transform_section_tables collects every table up front, then decompose()s outer
    wrappers — which also decomposes the nested wrappers still queued in the loop. A
    decomposed tag has attrs=None, so the next table.get('id') used to raise
    AttributeError, dropping the whole document. The loop must skip those stale tables.
    """

    def test_nested_wrapper_tables_do_not_crash(self):
        html = (
            '<div id="maindoc">'
            # Outer Title wrapper whose only text lives in an EMPTY nested wrapper cell →
            # the outer is decompose()d, taking the nested wrapper (still queued) with it.
            '<table id="DOCUMENT:1;ARTICLE:1;_Title"><tr><td>'
            '<table id="DOCUMENT:1;ARTICLE:2;_Title"><tr><td></td></tr></table>'
            "</td></tr></table>"
            '<table id="DOCUMENT:1;ARTICLE:3;_Content"><tr><td>ცოცხალი ტექსტი</td></tr></table>'
            "</div>"
        )
        md = html_to_markdown(html)  # must not raise
        self.assertIn("ცოცხალი ტექსტი", md)

    def test_deeply_nested_wrappers_preserve_inner_content(self):
        html = (
            '<div id="maindoc">'
            '<table id="DOCUMENT:1;ENCLOSURE:1;_Content"><tr><td>'
            '<table id="DOCUMENT:1;ENCLOSURE:1;ARTICLE:1;_Title"><tr><td>დანართის მუხლი</td></tr></table>'
            '<table id="DOCUMENT:1;ENCLOSURE:1;ARTICLE:1;_Content"><tr><td>დანართის ტექსტი</td></tr></table>'
            "</td></tr></table>"
            "</div>"
        )
        md = html_to_markdown(html)  # must not raise
        self.assertIn("დანართის მუხლი", md)
        self.assertIn("დანართის ტექსტი", md)

    def test_safe_wrapper_also_survives(self):
        html = (
            '<div id="maindoc">'
            '<table id="DOCUMENT:1;ARTICLE:1;_Title"><tr><td>'
            '<table id="DOCUMENT:1;ARTICLE:2;_Title"><tr><td></td></tr></table>'
            "</td></tr></table></div>"
        )
        self.assertIsInstance(safe_html_to_markdown(html), str)


if __name__ == "__main__":
    unittest.main()
