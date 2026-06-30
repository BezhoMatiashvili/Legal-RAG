"""HTML -> Markdown converter for matsne.gov.ge legal documents.

The page body (`#maindoc`) is built almost entirely out of nested tables. Every
logical section is wrapped in a table whose ``id`` encodes the document tree,
e.g. ``DOCUMENT:1;ENCLOSURE:1;CHAPTER:1;ARTICLE:1;_Title``. We use those ids to
rebuild a real heading hierarchy instead of guessing from CSS, then convert the
cleaned-up DOM with markdownify.

Section taxonomy seen in the corpus:
    HEADER (letterhead + document title), PREAMBLE, ARTICLE, FOOTER (signature),
    ENCLOSURE (დანართი / annex) -> HEADER / CHAPTER / ARTICLE / POINT / SUBPOINT.
"""

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

BASE_URL = "https://matsne.gov.ge"

_log = logging.getLogger(__name__)

# Section path segments that count as structural nesting (drive heading depth).
# POINT / SUBPOINT are numbered items *inside* a section's body, not headings.
STRUCTURAL = {
    "HEADER",
    "PREAMBLE",
    "ARTICLE",
    "FOOTER",
    "ENCLOSURE",
    "CHAPTER",
    "SECTION",
    "PART",
    "BOOK",
    "TITLE",
    "SUBSECTION",
    "PARAGRAPH",
}

# Tags / selectors that are pure UI chrome or duplicated content.
_CRUFT = [
    "style",
    "script",
    "noscript",
    "form",
    "textarea",
    "button",
    "input",
    ".btn-group",
    ".annotator-adder",
    ".document-anchor-panel",
    ".hide",
]

_WRAPPER_RE = re.compile(r"_(Title|Content)$")

_SUP_MAP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUB_MAP = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


class LegalConverter(MarkdownConverter):
    """markdownify converter that keeps super/subscripts and underline."""

    def convert_sup(self, el, text, parent_tags):
        return _script(text, _SUP_MAP, "sup")

    def convert_sub(self, el, text, parent_tags):
        return _script(text, _SUB_MAP, "sub")

    def convert_u(self, el, text, parent_tags):
        # Markdown has no underline; in legal text it usually marks amended
        # wording, so preserve it losslessly as inline HTML.
        return f"<u>{text}</u>" if text.strip() else text


def _script(text, table, tag):
    """Render super/subscript as Unicode when possible, else keep inline HTML."""
    stripped = text.strip()
    if not stripped:
        return ""
    converted = stripped.translate(table)
    if converted != stripped and not _has_untranslated(stripped, table):
        return converted
    return f"<{tag}>{stripped}</{tag}>"


def _has_untranslated(text, table):
    return any(ord(ch) not in table for ch in text)


def _strip_cruft(main):
    for selector in _CRUFT:
        for el in main.select(selector):
            el.decompose()
    # Navigation anchors: <a name="..."> with no visible text.
    for a in main.find_all("a"):
        if a.get("name") and not a.get_text(strip=True):
            a.decompose()


def _clean_links(main, base_url):
    for a in main.find_all("a"):
        href = a.get("href", "")
        if href.startswith("#") or not href:
            # In-page jump: keep the text, drop the link.
            text = a.get_text()
            a.replace_with(text) if text.strip() else a.decompose()
        elif href.startswith("/"):
            a["href"] = urljoin(base_url, href)


def _absolutize_images(main, base_url):
    for img in main.find_all("img"):
        src = img.get("src", "")
        if src.startswith("/"):
            img["src"] = urljoin(base_url, src)
        img.attrs.setdefault("alt", "")


def _section_path(table_id):
    """``DOCUMENT:1;ENCLOSURE:1;CHAPTER:1;_Title`` -> (['ENCLOSURE','CHAPTER'], 'Title')."""
    kind = _WRAPPER_RE.search(table_id).group(1)
    body = _WRAPPER_RE.sub("", table_id)
    segs = [re.sub(r":\d+$", "", s) for s in body.split(";") if s]
    segs = [s for s in segs if s and s != "DOCUMENT"]
    return segs, kind


def _heading_level(segments):
    structural = [s for s in segments if s in STRUCTURAL]
    depth = len(structural)
    # HEADER carries a section's own title, so it outranks its siblings
    # (an annex header should sit above the annex's chapters).
    if segments and segments[-1] == "HEADER":
        return max(1, min(depth, 6))
    return max(2, min(depth + 1, 6))


def _transform_section_tables(main, soup):
    """Replace every wrapper table with a heading or an unwrapped content block."""
    for table in main.find_all("table"):
        table_id = table.get("id", "")
        if not _WRAPPER_RE.search(table_id):
            continue
        segments, kind = _section_path(table_id)
        cell = table.find(["td", "th"])
        if cell is None:
            table.decompose()
            continue

        is_top_header = segments == ["HEADER"]
        if kind == "Content" and is_top_header:
            # The document's main title.
            _replace_with_heading(table, cell, 1, soup)
        elif kind == "Title" and is_top_header:
            # Letterhead (ministry / number / date): metadata, not a heading.
            _unwrap_into(table, cell, "div", soup)
        elif kind == "Title":
            _replace_with_heading(table, cell, _heading_level(segments), soup)
        else:
            _unwrap_into(table, cell, "div", soup)


def _replace_with_heading(table, cell, level, soup):
    text = cell.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    if not text:
        table.decompose()
        return
    heading = soup.new_tag(f"h{level}")
    heading.string = text
    table.replace_with(heading)


def _unwrap_into(table, cell, name, soup):
    block = soup.new_tag(name)
    for child in list(cell.contents):
        block.append(child.extract())
    table.replace_with(block)


def _is_data_table(table):
    """A real (data) table vs. a table used only for visual layout."""
    border = (table.get("border") or "").strip()
    if border and border not in ("0", "none"):
        return True
    if table.find(["th", "thead"]):
        return True
    rows_with_data = 0
    for tr in table.find_all("tr"):
        cells = [
            c
            for c in tr.find_all(["td", "th"], recursive=False)
            if c.get_text(strip=True)
        ]
        if len(cells) >= 2:
            rows_with_data += 1
    return rows_with_data >= 2


def _flatten_layout_tables(main):
    """Demote layout-only tables to plain blocks (innermost first)."""
    for table in reversed(main.find_all("table")):
        if _is_data_table(table):
            continue
        for el in table.find_all(["thead", "tbody", "tr", "td", "th"]):
            el.name = "div"
        table.name = "div"


def _expand_spans(table, soup):
    """Rebuild `table` as a rectangular grid, honoring colspan/rowspan.

    Budget/annex tables merge cells heavily; markdownify ignores the spans and
    emits ragged rows. We expand them so every row has the same number of cells:
    the top-left cell of a span keeps its content (inline formatting preserved)
    and the cells it covers become empty.
    """
    grid = []
    pending = {}  # column -> (remaining_rows, tag_name)
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        row, col, ci = [], 0, 0
        while ci < len(cells) or col in pending:
            if col in pending:
                rem, name = pending[col]
                row.append((name, None))
                if rem - 1 <= 0:
                    del pending[col]
                else:
                    pending[col] = (rem - 1, name)
                col += 1
                continue
            cell = cells[ci]
            ci += 1
            cspan = max(1, int(cell.get("colspan", 1) or 1))
            rspan = max(1, int(cell.get("rowspan", 1) or 1))
            for k in range(cspan):
                row.append((cell.name, cell if k == 0 else None))
                if rspan > 1:
                    pending[col] = (rspan - 1, cell.name)
                col += 1
        grid.append(row)
    if not grid:
        return
    width = max(len(r) for r in grid)

    new_table = soup.new_tag("table")
    for r in grid:
        tr = soup.new_tag("tr")
        for j in range(width):
            name, node = r[j] if j < len(r) else ("td", None)
            cell = soup.new_tag(name)
            if node is not None:
                for child in list(node.contents):
                    cell.append(child.extract())
            tr.append(cell)
        new_table.append(tr)
    table.replace_with(new_table)


def _normalize_data_tables(main, soup):
    """Flatten merged cells in data tables so Markdown renders aligned rows."""
    # Innermost first: a normalized inner table turns its parent into a
    # "contains a nested table" case, which we then skip rather than mangle.
    for table in reversed(main.find_all("table")):
        if not _is_data_table(table):
            continue
        if table.find("table"):  # nested data table (rare): leave alone
            continue
        if table.find(attrs={"colspan": True}) or table.find(attrs={"rowspan": True}):
            _expand_spans(table, soup)


def _strip_spans(main):
    for span in main.find_all("span"):
        span.unwrap()


def _collapse_emphasis(text):
    # matsne fragments a single bold phrase across many adjacent <b> runs;
    # markdownify (which already trims whitespace inside each) then emits things
    # like ``**a****b**`` or ``**a** **b**``. Stitch runs separated only by
    # whitespace back into one. Looping handles longer chains.
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\*\*(\s*)\*\*", r"\1", text)
    return text


def _tidy(text):
    text = _collapse_emphasis(text)
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank-line runs
    return text.strip()


def html_to_markdown(html: str, base_url: str = BASE_URL) -> str:
    """Convert a fragment of legal-document HTML to Markdown.

    ``base_url`` is used to absolutize root-relative links/images; it defaults to
    matsne but other spiders should pass their own site root.
    """
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("#maindoc") or soup

    _strip_cruft(main)
    _clean_links(main, base_url)
    _absolutize_images(main, base_url)
    _transform_section_tables(main, soup)
    _flatten_layout_tables(main)
    _normalize_data_tables(main, soup)
    _strip_spans(main)

    converter = LegalConverter(
        heading_style="ATX",
        bullets="-",
        escape_asterisks=False,
        escape_underscores=False,
    )
    return _tidy(converter.convert_soup(main))


def safe_html_to_markdown(html: str, base_url: str = BASE_URL, source_url: str | None = None) -> str:
    """``html_to_markdown`` that never raises: on a parser failure it logs and returns "".

    Detail-page bodies come from untrusted HTML; a single malformed document should not
    drop the whole item, so callers use this and still emit the item with an empty body.
    """
    if not html:
        return ""
    try:
        return html_to_markdown(html, base_url=base_url)
    except Exception as exc:  # noqa: BLE001 - body extraction must be best-effort
        _log.warning("html_to_markdown failed for %s: %s", source_url or "<unknown>", exc)
        return ""
