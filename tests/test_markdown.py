import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "matsne"))

from matsne.utils.markdown import html_to_markdown, safe_html_to_markdown  # noqa: E402


class HtmlToMarkdownTests(unittest.TestCase):
    def test_section_table_ids_become_heading_hierarchy(self):
        html = (
            '<div id="maindoc">'
            '<table id="DOCUMENT:1;HEADER:1;_Content"><tr><td>დოკუმენტის სათაური</td></tr></table>'
            '<table id="DOCUMENT:1;ARTICLE:1;_Title"><tr><td>მუხლი 1</td></tr></table>'
            '<table id="DOCUMENT:1;ARTICLE:1;_Content"><tr><td>მუხლის ტექსტი</td></tr></table>'
            "</div>"
        )
        md = html_to_markdown(html)
        self.assertIn("# დოკუმენტის სათაური", md)
        self.assertIn("## მუხლი 1", md)
        self.assertIn("მუხლის ტექსტი", md)

    def test_base_url_absolutizes_links_and_images(self):
        html = '<div id="maindoc"><p><a href="/ka/doc">link</a></p><p><img src="/img/x.png"></p></div>'
        md = html_to_markdown(html, base_url="https://constcourt.ge")
        self.assertIn("https://constcourt.ge/ka/doc", md)
        self.assertIn("https://constcourt.ge/img/x.png", md)

    def test_fragmented_bold_is_collapsed(self):
        md = html_to_markdown('<div id="maindoc"><p><b>ad</b><b>min</b></p></div>')
        self.assertNotIn("****", md)
        self.assertIn("admin", md)

    def test_colspan_table_expanded_without_crashing(self):
        html = (
            '<div id="maindoc"><table border="1">'
            "<tr><th>a</th><th>b</th></tr>"
            '<tr><td colspan="2">spanned</td></tr>'
            "</table></div>"
        )
        md = html_to_markdown(html)
        self.assertIn("spanned", md)


class SafeHtmlToMarkdownTests(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(safe_html_to_markdown(""), "")

    def test_non_string_is_swallowed(self):
        self.assertEqual(safe_html_to_markdown(123), "")  # type: ignore[arg-type]

    def test_valid_html_converts(self):
        self.assertIn("hi", safe_html_to_markdown("<p>hi</p>"))


if __name__ == "__main__":
    unittest.main()
