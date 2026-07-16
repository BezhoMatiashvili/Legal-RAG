"""Diagnose WHERE the confident-wrong hallucination comes from — and therefore which lever
actually fixes it — instead of guessing.

`answer_eval` measures `confident_wrong` (rank-1 score ≥ threshold but the WRONG doc, on
known-item queries): the demonstrated legal failure mode. This tool takes an answer_eval
result JSON's ``identity_failures`` (per-query: gold_doc, got_top1) and, for each failure,
asks which safeguard *could* have caught it:

  id_mismatch_deterministic  the query names a document-ID citation the top doc does NOT
                             carry  ->  a deterministic document_number check would ABSTAIN.
  id_ambiguity               the top doc DOES carry the cited id but isn't the gold doc
                             (same number, different act)  ->  a number check can't help.
  full_route_resolvable      no ID citation, but the query names a law/code whose alias
                             resolves (extract_citation mode='full')  ->  the TITLE/alias
                             route (CITATION_ROUTE=full) is the lever, not a number check.
  pure_semantic              neither fires — the query is a bare legal CONCEPT (no law
                             named)  ->  no identity check is possible; only reranker
                             quality + the client's advisory "does this answer the ask?"
                             judgment (already in the legal_search abstention contract) help.

Read: it tells you whether a deterministic identity guard is worth building (it is only if
``id_mismatch_deterministic`` is a large share) or is theater. Pure diagnostic — by-id
Qdrant retrieves + regex/alias, no embed, no rerank.

    RERANK_ENABLED=false python -m eval.diagnose_confident_wrong \
        --run ingest/.state/answer_eval/answer_eval_v2_rerank_ON_g.json --golden-set v2
"""

import argparse
import json
from pathlib import Path

from ingest.citations import extract_citation
from ingest.config import load_config
from ingest.qdrant_store import make_client, point_id

from . import goldset

BUCKETS = ("id_mismatch_deterministic", "id_ambiguity", "full_route_resolvable", "pure_semantic")


def _candidates(ref):
    """Every {key: value} identity dict the citation could match (mirrors citation_lookup)."""
    return [ref.filters, *ref.alternates]


def _carries_cited_id(ref, payload) -> bool:
    for cand in _candidates(ref):
        for key, val in cand.items():
            pv = payload.get(key)
            if pv is not None and str(pv).strip() == str(val).strip():
                return True
    return False


def classify(query: str, get_payload) -> str:
    """Which safeguard could catch this confident-wrong top-1 for this query?

    ``get_payload`` is a 0-arg callable fetching the top-1 doc payload — only invoked when a
    document-ID citation fires (so the no-citation majority costs no Qdrant retrieve).
    """
    ref_ids = extract_citation(query, mode="ids")
    if ref_ids is not None:
        return "id_ambiguity" if _carries_cited_id(ref_ids, get_payload()) else "id_mismatch_deterministic"
    # No document-ID citation. Would the alias/title route (mode='full') resolve the named law?
    return "full_route_resolvable" if extract_citation(query, mode="full") is not None else "pure_semantic"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the confident-wrong failure mode by fixability.")
    ap.add_argument("--run", type=Path, required=True, help="answer_eval result JSON (with identity_failures)")
    ap.add_argument("--golden-set", choices=sorted(goldset.EVAL_SETS), default="v2")
    ap.add_argument("--limit-samples", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    if cfg.generation_id:
        raise SystemExit(
            "diagnose_confident_wrong reads frozen v2 run artifacts with legacy point IDs; "
            "use the version-aware v3 release evaluator for immutable generations"
        )
    client = make_client(cfg)
    spec = goldset.EVAL_SETS[args.golden_set]
    meta = {}
    for line in open(spec.gold, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            d = json.loads(line)
            meta[d["id"]] = (d["query"], d.get("query_type", "?"))

    run = json.load(open(args.run, encoding="utf-8"))
    fails = run.get("identity_failures", [])

    def top_payload(source, doc):
        recs = client.retrieve(cfg.collection_name, ids=[point_id(source, doc, 0)], with_payload=True)
        return (recs[0].payload or {}) if recs else {}

    counts = {b: 0 for b in BUCKETS}
    by_type: dict[str, dict[str, int]] = {}
    samples: dict[str, list] = {b: [] for b in BUCKETS}
    for f in fails:
        query, qtype = meta.get(f["id"], ("", "?"))
        got = f["got_top1"]
        bucket = classify(query, lambda g=got: top_payload(g[0], g[1]))
        counts[bucket] += 1
        by_type.setdefault(qtype, {b: 0 for b in BUCKETS})[bucket] += 1
        if len(samples[bucket]) < args.limit_samples:
            samples[bucket].append((f["id"], qtype, query[:64], got))

    n = len(fails) or 1
    print(f"=== confident-wrong fixability — {len(fails)} identity-failures "
          f"({run.get('mode')} / cite={run.get('knobs', {}).get('citation_route')}) ===\n")
    print(f"  {'bucket':<28} {'count':>6}  share   lever")
    levers = {
        "id_mismatch_deterministic": "deterministic document_number guard",
        "id_ambiguity": "date/title disambiguation (number alone can't)",
        "full_route_resolvable": "CITATION_ROUTE=full (alias/title route)",
        "pure_semantic": "reranker quality + advisory abstention (already live)",
    }
    for b in BUCKETS:
        print(f"  {b:<28} {counts[b]:>6}  {counts[b]/n:>4.0%}   {levers[b]}")
    print("\n=== by query_type ===")
    for t, c in sorted(by_type.items()):
        print(f"  {t:<15} " + "  ".join(f"{b.split('_')[0]}:{c[b]}" for b in BUCKETS))
    for b in BUCKETS:
        if samples[b]:
            print(f"\n--- {b} (sample) ---")
            for qid, qtype, q, got in samples[b]:
                print(f"  {qid:<8} {qtype:<15} top1={got}  q={q!r}")


if __name__ == "__main__":
    main()
