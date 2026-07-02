"""Plain-text -> Markdown normalization.

Some sources (ecd) return the full decision as already-plain text rather than HTML,
so ``utils.markdown.html_to_markdown`` does not apply. Here we only tidy whitespace
and keep the Georgian text intact.
"""

import re


def plain_text_to_markdown(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)  # strip trailing whitespace on each line
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip()
