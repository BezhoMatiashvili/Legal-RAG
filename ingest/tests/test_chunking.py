from ingest.chunking import _pack, build_embed_text, chunk_document


def test_empty_document_yields_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_short_document_is_one_chunk():
    chunks = chunk_document("just a short legal line.", max_tokens=50, overlap=5, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].heading_path == []
    assert "short legal line" in chunks[0].text


def test_heading_paths_and_clean_text():
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
    # Stored/display text is CLEAN — the heading path is NOT baked into it.
    a1 = next(c for c in chunks if c.heading_path == ["Title", "Article 1"])
    assert a1.text == "body of article one here."
    # Context is folded into the embedded text only, via build_embed_text.
    embed = build_embed_text(a1.text, title="Doc", document_type="law", heading_path=a1.heading_path)
    assert embed == "Doc > law > Title > Article 1\n\nbody of article one here."
    assert build_embed_text("bare", title=None, document_type=None, heading_path=None) == "bare"


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


def test_overlap_seed_never_pushes_full_atom_over_budget():
    text = "one two. " + " ".join(f"word{i}" for i in range(10)) + "."
    chunks = chunk_document(text, max_tokens=10, overlap=2, min_tokens=1)

    assert [chunk.token_count for chunk in chunks] == [2, 10]
    assert all(chunk.token_count <= 10 for chunk in chunks)


def test_small_tail_is_not_folded_when_combined_chunk_would_overflow():
    first = " ".join(f"first{i}" for i in range(9)) + "."
    text = first + " tail1 tail2."
    chunks = chunk_document(text, max_tokens=10, overlap=0, min_tokens=4)

    assert [chunk.token_count for chunk in chunks] == [9, 2]
    assert all(chunk.token_count <= 10 for chunk in chunks)


def test_adversarial_overlap_and_tail_pack_never_exceeds_budget():
    atoms = [
        ("a0.", 1, 0, 3),
        ("b0.", 1, 5, 8),
        ("c0 c1 c2 c3 c4 c5 c6 c7 c8.", 9, 10, 37),
    ]

    packed = _pack(atoms, max_tokens=10, overlap=3, min_tokens=3)

    assert [piece[1] for piece in packed] == [2, 10]
    assert all(piece[1] <= 10 for piece in packed)


def test_article_markers_start_new_sections_and_stay_in_body():
    md = (
        "შესავალი დებულება აქ.\n\n"
        "მუხლი 1. პირველი მუხლის შინაარსი.\n\n"
        "მუხლი 2. მეორე მუხლის შინაარსი აქ არის."
    )
    chunks = chunk_document(md, max_tokens=100, overlap=0, min_tokens=1)
    texts = [c.text for c in chunks]
    # each article is its own chunk, and the "მუხლი N" marker is kept in the body
    assert any(t.startswith("მუხლი 1.") for t in texts)
    assert any(t.startswith("მუხლი 2.") for t in texts)
    # the preamble is separated from article 1
    assert any(t.startswith("შესავალი") and "მუხლი 1" not in t for t in texts)
    # offsets still slice back exactly
    for c in chunks:
        assert md[c.char_start : c.char_end].strip()


def test_injected_token_counter_is_used():
    calls = {"n": 0}

    def counter(s):
        calls["n"] += 1
        return len(s.split())

    chunk_document("# H\n\none two three four five.", max_tokens=4, overlap=1,
                   min_tokens=1, count_tokens=counter)
    assert calls["n"] > 0
