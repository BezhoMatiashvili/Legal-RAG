"""Chunking against the REAL BGE-M3 subword tokenizer (not just a word counter).

Skipped automatically when transformers/the tokenizer aren't available (e.g. offline),
so the rest of the suite stays fast and hermetic.
"""

import pytest

from ingest.chunking import chunk_document


@pytest.fixture(scope="module")
def count_tokens():
    pytest.importorskip("transformers")
    from ingest.embedding import make_token_counter

    try:
        return make_token_counter("BAAI/bge-m3")
    except Exception as exc:  # noqa: BLE001 - no network / model cache
        pytest.skip(f"BGE-M3 tokenizer unavailable: {exc}")


def test_real_tokenizer_chunks_stay_within_budget(count_tokens):
    max_tokens = 128
    # Long Georgian-ish body with a long heading, to exercise the ctx-budget path.
    long_heading = "# " + "სათაური " * 40
    body = long_heading + "\n\n" + ("ეს არის გრძელი სამართლებრივი ტექსტი. " * 200)
    chunks = chunk_document(body, max_tokens=max_tokens, overlap=20, min_tokens=16, count_tokens=count_tokens)

    assert chunks
    # Allow a small slack for the prepended heading context + the min-floor merge.
    for c in chunks:
        assert count_tokens(c.text) <= max_tokens * 2
