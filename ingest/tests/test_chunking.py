from ingest.chunking import chunk_document


def test_empty_document_yields_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_short_document_is_one_chunk():
    chunks = chunk_document("just a short legal line.", max_tokens=50, overlap=5, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].heading_path == []
    assert "short legal line" in chunks[0].text


def test_heading_paths_and_context_prefix():
    md = (
        "# Title\n\npreamble paragraph.\n\n"
        "## Article 1\n\nbody of article one here.\n\n"
        "## Article 2\n\nbody of article two here.\n"
    )
    chunks = chunk_document(md, max_tokens=50, overlap=5, min_tokens=1)
    paths = [c.heading_path for c in chunks]
    assert ["Title"] in paths
    assert ["Title", "Article 1"] in paths
    assert ["Title", "Article 2"] in paths
    # Heading path is prepended to chunk text for context.
    assert any(c.text.startswith("Title > Article 1") for c in chunks)


def test_packing_respects_budget_and_overlaps():
    # 12 distinct 4-token sentences in one section.
    text = " ".join(f"u{i}a u{i}b u{i}c u{i}d." for i in range(12))
    max_tokens = 10
    chunks = chunk_document(text, max_tokens=max_tokens, overlap=4, min_tokens=1)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= max_tokens  # no chunk exceeds the budget

    # Overlap duplicates boundary content, so total tokens exceed the source length.
    doc_tokens = len(text.split())
    total = sum(len(c.text.split()) for c in chunks)
    assert total > doc_tokens

    # chunk indices are contiguous from 0.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_injected_token_counter_is_used():
    calls = {"n": 0}

    def counter(s):
        calls["n"] += 1
        return len(s.split())

    chunk_document("# H\n\none two three four five.", max_tokens=4, overlap=1,
                   min_tokens=1, count_tokens=counter)
    assert calls["n"] > 0
