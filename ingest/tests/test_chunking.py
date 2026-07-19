import hashlib

from ingest.chunking import _pack, _split_oversized_run, build_embed_text, chunk_document


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


def _subword_count(s: str) -> int:
    """Simulates a real subword tokenizer: ~1 token per 4 chars, never zero for non-empty."""
    return max(1, -(-len(s) // 4)) if s else 0


def test_unsplittable_run_over_budget_is_hard_split_not_raised():
    # A single whitespace-free run (e.g. an inline base64 data URI) long enough that even
    # one "word" blows the token budget under a real tokenizer. Must not raise or crash —
    # a data-quality edge case in one document must never kill ingestion of that document.
    blob = "data:image/jpeg;base64," + "A" * 2000
    text = f"preamble sentence here. {blob} trailing sentence after."
    chunks = chunk_document(text, max_tokens=50, overlap=0, min_tokens=1, count_tokens=_subword_count)

    assert chunks  # did not raise, produced real output
    assert all(c.token_count <= 50 for c in chunks)
    # The blob survives whole across its split pieces — nothing was silently dropped.
    reconstructed = "".join(
        c.text.replace("preamble sentence here.", "")
        .replace("trailing sentence after.", "")
        .strip()
        for c in chunks
    )
    assert blob in reconstructed or blob == reconstructed


def test_split_oversized_run_preserves_text_and_offsets():
    blob = "B" * 777
    pieces = _split_oversized_run(blob, start=100, max_tokens=50, count=_subword_count)

    assert all(tok <= 50 for _, tok, _, _ in pieces)
    # Offsets are contiguous and reconstruct the original blob exactly.
    assert "".join(p[0] for p in pieces) == blob
    assert pieces[0][2] == 100
    assert pieces[-1][3] == 100 + len(blob)
    for (_, _, _, end), (_, _, next_start, _) in zip(pieces, pieces[1:]):
        assert end == next_start


def test_split_oversized_run_bottoms_out_at_one_char():
    # A pathological counter that reports every non-empty string as over budget must not
    # infinite-loop — recursion bottoms out at a single character and returns it as-is.
    pieces = _split_oversized_run("xyz", start=0, max_tokens=0, count=lambda s: len(s) + 1)
    assert [p[0] for p in pieces] == ["x", "y", "z"]


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


def test_article_identity_and_exact_text_propagate_to_continuation_chunks():
    body = (
        "**თავი I**\n\n**მუხლი 12¹. სპეციალური წესი**\n\n1. "
        + " ".join(f"დებულება{i}." for i in range(45))
    )
    chunks = chunk_document(body, max_tokens=10, overlap=3, min_tokens=1)
    article_chunks = [chunk for chunk in chunks if chunk.article_id == "12¹"]
    assert len(article_chunks) > 2
    assert all(chunk.article_label in chunk.heading_path for chunk in article_chunks)
    assert all(chunk.article_start_chunk_index == article_chunks[0].chunk_index for chunk in article_chunks)
    assert all(chunk.parent_chunk_index == article_chunks[0].chunk_index for chunk in article_chunks)
    for chunk in chunks:
        assert chunk.text == body[chunk.char_start : chunk.char_end]
        assert chunk.canonical_text == chunk.text
        assert chunk.passage_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def test_token_bounded_overlap_keeps_tail_of_large_sentence_exactly():
    body = " ".join(f"სიტყვა{i}" for i in range(35)) + "."
    chunks = chunk_document(body, max_tokens=10, overlap=3, min_tokens=1)
    assert len(chunks) > 2
    assert any(chunks[i + 1].char_start < chunks[i].char_end for i in range(len(chunks) - 1))
    assert all(chunk.text == body[chunk.char_start : chunk.char_end] for chunk in chunks)


def test_chunks_derive_exact_physical_page_intersections():
    body = "one two three four\n\nfive six seven eight"
    page_two = body.index("five")
    chunks = chunk_document(
        body,
        max_tokens=6,
        overlap=2,
        min_tokens=1,
        page_boundaries=(
            {"page_number": 1, "char_start": 0, "char_end": page_two - 2},
            {"page_number": 2, "char_start": page_two, "char_end": len(body)},
        ),
        page_coordinate_reason="exact_pdf_text",
    )

    assert chunks
    assert all(chunk.page_start is not None and chunk.page_end is not None for chunk in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert {chunk.page_coordinate_reason for chunk in chunks} == {"exact_pdf_text"}


def test_non_paginated_chunks_keep_null_pages_with_explicit_reason():
    chunks = chunk_document("one two three", max_tokens=5, overlap=0, min_tokens=1)
    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(None, None)]
    assert chunks[0].page_coordinate_reason == "source_not_paginated"
