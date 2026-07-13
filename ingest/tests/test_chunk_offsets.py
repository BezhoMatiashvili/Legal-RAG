"""Offset-aware chunking: chunks must record where they came from in the source body.

The core invariant is per-atom exact reconstruction (offsets are tracked in the input's
coordinate space, never re-located on a joined/stripped/normalized string), which then
makes span→chunk mapping correct. Georgian text is used deliberately.
"""

from ingest.chunking import _atoms, _split_sections, chunk_document, default_token_counter

count = default_token_counter


def _ns(s: str) -> str:
    """Non-space content — offsets may include whitespace the atom text collapsed."""
    return "".join(s.split())


def test_split_sections_body_start_is_exact():
    text = "# სათაური\n\nპირველი აბზაცი.\n\n## თავი 1\n\nმეორე აბზაცი აქ.\n"
    for path, body, body_start in _split_sections(text):
        assert text[body_start : body_start + len(body)] == body


def test_atoms_reconstruct_exactly_from_body():
    body = (
        "პირველი წინადადება. მეორე წინადადება აქ.\n\n"
        "მესამე აბზაცი უფრო გრძელი შინაარსით და დამატებითი სიტყვებით."
    )
    for atom_text, tok, start, end in _atoms(body, max_tokens=6, count=count):
        assert _ns(atom_text) == _ns(body[start:end])
        assert 0 <= start <= end <= len(body)


def test_word_window_atoms_reconstruct():
    # one pathologically long sentence (no sentence punctuation) → word-window atoms
    body = " ".join(f"სიტყვა{i}" for i in range(40)) + "."
    atoms = _atoms(body, max_tokens=8, count=count)
    assert len(atoms) > 1
    for atom_text, tok, start, end in atoms:
        assert _ns(atom_text) == _ns(body[start:end])


def test_chunk_offsets_in_range_and_slice_back():
    md = (
        "# კანონი\n\n" + " ".join(f"დებულება{i} აქ არის." for i in range(60)) + "\n"
    )
    chunks = chunk_document(md, max_tokens=40, overlap=8, min_tokens=5)
    assert len(chunks) > 1
    for c in chunks:
        assert 0 <= c.char_start <= c.char_end <= len(md)
        # heading text lives in the prefix, not the span; the body region is non-empty
        assert md[c.char_start : c.char_end].strip()


def test_concrete_offsets():
    # No heading: single section starting at 0. Small budget forces multiple chunks.
    text = "აააა. ბბბბ. გგგგ. დდდდ."
    chunks = chunk_document(text, max_tokens=2, overlap=0, min_tokens=1)
    # first chunk starts at 0; last chunk ends at len (trailing '.' included in last atom)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)


def test_duplicate_paragraphs_get_monotonic_offsets():
    # identical paragraphs — a str.find-based mapper would collapse them; offsets must not.
    para = "იდენტური აბზაცი ტექსტით."  # 3 tokens
    text = f"{para}\n\n{para}\n\n{para}"
    # budget = 3 tokens → each identical paragraph lands in its own chunk
    chunks = chunk_document(text, max_tokens=3, overlap=0, min_tokens=1)
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
    # the three identical paragraphs occupy three distinct, increasing regions —
    # a str.find-based mapper would give all three the first paragraph's offset
    assert len({c.char_start for c in chunks}) == 3
    assert [text[c.char_start : c.char_end] for c in chunks] == [para, para, para]


def test_overlapping_chunks_have_overlapping_ranges():
    text = " ".join(f"ერთეული{i}." for i in range(30))
    chunks = chunk_document(text, max_tokens=10, overlap=5, min_tokens=1)
    assert len(chunks) >= 2
    # with overlap, consecutive chunks (same flat section) share source characters
    assert any(
        chunks[i + 1].char_start < chunks[i].char_end for i in range(len(chunks) - 1)
    )


def test_crlf_body_offsets_stay_exact():
    # \r survives hygiene; offsets are tracked in the input's own space, so they stay valid.
    text = "პირველი ხაზი.\r\nმეორე ხაზი.\r\n\r\nახალი აბზაცი აქ."
    chunks = chunk_document(text, max_tokens=6, overlap=0, min_tokens=1)
    for c in chunks:
        assert 0 <= c.char_start <= c.char_end <= len(text)
        assert _ns(c.text.split("\n\n")[-1]) in _ns(text[c.char_start : c.char_end]) or \
            _ns(text[c.char_start : c.char_end]) in _ns(c.text)


def test_empty_and_whitespace_bodies():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []
