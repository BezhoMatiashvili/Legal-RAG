"""Static EN→KA query-translation table for the cross-lingual eval knob (improvement I2).

BGE-M3 has a measured native-language retrieval bias — Georgian is listed among languages
that "only retrieve native documents" (BordIRLines, arXiv 2410.01171) — and our EN slice
confirms it (sparse nDCG 0.000). Variant A substitutes an authored Georgian legal-register
translation for each English golden query before embedding, so the sparse branch and the
reranker finally see Georgian tokens. The table is a checked-in artifact keyed by golden id
(``eval/query_translations_v1.json``); there is no runtime translation API anywhere.
"""

import hashlib
import json
from pathlib import Path

from ingest.search import detect_language

DEFAULT_TRANSLATIONS = Path(__file__).parent / "query_translations_v1.json"


def load_query_translations(path: Path, gold) -> tuple[dict[str, str], str]:
    """Load ``{original_query: georgian_query}`` plus its full content SHA-256.

    Fails loud (mirrors ``goldset.reground``) so the artifact can't silently drift:
    every entry's ``query`` must byte-match the golden query for its id, every source
    query must be English, every target must carry Georgian script. Georgian golden
    queries are untouched *by construction* — they simply have no entry.
    """
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    by_id = {q.id: q for q in gold}
    mapping: dict[str, str] = {}
    for qid, entry in data["translations"].items():
        q = by_id.get(qid)
        if q is None:
            raise ValueError(f"translations: unknown golden id {qid!r}")
        if entry["query"] != q.query:
            raise ValueError(f"translations: {qid} 'query' drifted from the golden set")
        if detect_language(q.query) != "en":
            raise ValueError(f"translations: {qid} source query is not English")
        if detect_language(entry["ka"]) != "ka":
            raise ValueError(f"translations: {qid} target carries no Georgian script")
        mapping[q.query] = entry["ka"]
    digest = hashlib.sha256(raw).hexdigest()
    return mapping, digest
