"""Span→chunk mapping across the tricky cases: Georgian punctuation, clause boundaries,
mixed KA/EN, multi-span, overlap straddle, point spans, heading-governed spans."""

from dataclasses import dataclass

from ingest.chunking import chunk_document, default_token_counter
from eval.spanmap import graded_relevant_chunks, map_spans_to_chunks

count = default_token_counter
CFG = dict(max_tokens=8, overlap=0, min_tokens=1, count_tokens=count)


@dataclass(frozen=True)
class Rel:
    char_start: int
    char_end: int
    grade: int


def _map(body, spans, **over):
    cfg = {**CFG, **over}
    return map_spans_to_chunks(body, spans, **cfg)


def test_span_maps_to_its_paragraph_chunk():
    body = "პირველი აბზაცი აქ არის.\n\nმეორე აბზაცი სხვა ადგილას."
    p2 = body.index("მეორე")
    mapping = _map(body, [(p2, p2 + 5)])
    chunks = chunk_document(body, **CFG)
    # the mapped chunk actually contains "მეორე"
    assert mapping[0]
    assert any("მეორე" in chunks[ci].text for ci in mapping[0])


def test_georgian_sentence_punctuation_is_split_consistently():
    # oversized paragraph, Georgian '.' terminators → sentence atoms; mapping stays exact
    body = " ".join(f"წინადადება ნომერი {i}." for i in range(12))
    target = body.index("ნომერი 7")
    mapping = _map(body, [(target, target + len("ნომერი 7"))], max_tokens=6)
    chunks = chunk_document(body, max_tokens=6, overlap=0, min_tokens=1, count_tokens=count)
    assert mapping[0]
    assert any("ნომერი 7" in chunks[ci].text for ci in mapping[0])


def test_numbered_clause_boundaries():
    body = "1. პირველი პუნქტი აქ.\n\n2. მეორე პუნქტი აქ.\n\n3. მესამე პუნქტი."
    p = body.index("2. მეორე")
    mapping = _map(body, [(p, p + len("2. მეორე"))])
    chunks = chunk_document(body, **CFG)
    assert any("მეორე პუნქტი" in chunks[ci].text for ci in mapping[0])


def test_mixed_ka_en_span():
    body = "This section defines ტერმინი A.\n\nდანართი two follows here."
    p = body.index("ტერმინი")
    mapping = _map(body, [(p, p + len("ტერმინი A"))])
    chunks = chunk_document(body, **CFG)
    assert any("ტერმინი A" in chunks[ci].text for ci in mapping[0])


def test_multiple_spans_map_independently():
    body = " ".join(f"ერთეული {i} ტექსტი." for i in range(20))
    a = body.index("ერთეული 1 ")
    b = body.index("ერთეული 18")
    mapping = _map(body, [(a, a + 10), (b, b + 10)], max_tokens=6)
    assert mapping[0] and mapping[1]
    assert mapping[0] != mapping[1]  # far-apart spans hit different chunks


def test_span_straddling_overlap_maps_to_two_chunks():
    body = " ".join(f"ტოკ{i}." for i in range(30))
    chunks = chunk_document(body, max_tokens=6, overlap=4, min_tokens=1, count_tokens=count)
    # find an adjacent overlapping pair and a char inside the shared region
    pair = next(
        (i for i in range(len(chunks) - 1) if chunks[i + 1].char_start < chunks[i].char_end),
        None,
    )
    assert pair is not None
    mid = (chunks[pair + 1].char_start + chunks[pair].char_end) // 2
    mapping = map_spans_to_chunks(
        body, [(mid, mid + 1)], max_tokens=6, overlap=4, min_tokens=1, count_tokens=count
    )
    assert {pair, pair + 1} <= mapping[0]


def test_point_span_is_inclusive():
    body = "ერთი ორი სამი ოთხი ხუთი."
    chunks = chunk_document(body, **CFG)
    p = chunks[0].char_start  # exact left boundary of chunk 0
    mapping = _map(body, [(p, p)])
    assert 0 in mapping[0]


def test_span_at_document_end():
    body = "დასაწყისი აქ.\n\nდასასრული ტექსტი აქ ბოლოა."
    mapping = _map(body, [(len(body) - 5, len(body))])
    chunks = chunk_document(body, **CFG)
    assert mapping[0]
    assert any("ბოლოა" in chunks[ci].text for ci in mapping[0])


def test_span_inside_stripped_heading_maps_to_governed_chunks():
    # a legal_citation-style doc: the cited act title lives in the '#' heading, which the
    # chunker lifts into heading_path (folded into the EMBEDDED text, not the clean body).
    body = "# სათაური ციტატა\n\nსხეული ერთი აქ.\n\nსხეული ორი აქ."
    cited = body.index("ციტატა")
    mapping = _map(body, [(cited, cited + len("ციტატა"))])
    chunks = chunk_document(body, **CFG)
    assert mapping[0]  # not empty despite pointing at heading chars
    for ci in mapping[0]:
        # heading is the section context (→ prepended to the embedded text → retrievable)
        assert "სათაური ციტატა" in chunks[ci].heading_path


def test_empty_body_maps_to_nothing():
    assert map_spans_to_chunks("", [(0, 0)], **CFG) == {0: set()}


def test_graded_relevant_chunks_takes_max_grade():
    body = " ".join(f"სიტყვა {i} აქ." for i in range(20))
    a = body.index("სიტყვა 1 ")
    b = body.index("სიტყვა 15")
    graded = graded_relevant_chunks(
        body, [Rel(a, a + 8, 1), Rel(b, b + 8, 2)], max_tokens=6, overlap=0, min_tokens=1,
        count_tokens=count,
    )
    assert set(graded.values()) <= {1, 2}
    assert 2 in graded.values()
