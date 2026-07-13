#!/usr/bin/env python
"""Seeded stratified candidate sampler for golden_set_v2 (I5) — reproducible provenance.

Samples candidate GOLD DOCUMENTS from the corpus universe (a chunk-0 payload scroll),
stratified source × document_type × decade × status with per-source floors, EXCLUDING
every document already used by eval/tuning (v1 gold + v1 holdout + ids pinned in the I1
patch). Selection is purely metadata-driven — no retrieval mode is ever consulted, so the
sample cannot be biased toward what the current system already ranks well.

The pair plan then assigns each sampled doc 1–3 target query types by metadata
affordance (citation-capable → legal_citation; status/expiry-bearing → temporal; all →
natural_question / keyword / cross_lingual / paraphrase), up to the per-slice NEW-pair
targets (v2 total minus v1): citation 83, cross_lingual 78, natural 68, keyword 59,
paraphrase 49, temporal 60 = 397.

Usage (from ingest/):
    .venv/bin/python scripts/sample_goldset_candidates.py --universe docs_universe.jsonl \
        --out candidates.jsonl [--seed 20260710]

The universe file is produced by a read-only chunk-0 scroll (see I5 session notes);
candidates.jsonl carries doc metadata + assigned pair types, in priority order with
~1.5× headroom — authors consume it top-down and skip dropped docs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import goldset  # noqa: E402

DEFAULT_SEED = 20260710

# ids pinned as fixtures inside .improvements/i1_citation_route_full.patch (tuning artifact)
I1_PATCH_IDS = {
    ("matsne", "31702"), ("matsne", "102038"), ("matsne", "111390"), ("matsne", "240140000"),
    ("ecd", "330100116001532776"), ("ecd", "011746498351723"),
}

# New-pair targets per slice (spec table totals minus the frozen v1 counts).
NEW_PAIRS = {"legal_citation": 83, "cross_lingual": 78, "natural_question": 68,
             "keyword": 59, "paraphrase": 49, "temporal": 60}

# Candidate-doc allocation (≈1.5× the ~200 docs that 397 pairs at ~2/doc require).
# matsne is sub-stratified by document_type bucket; floors keep small sources present.
MATSNE_TYPE_BUCKETS = [
    ("consolidated", 30),          # is_consolidated=True base laws (citation/temporal anchors)
    ("საქართველოს კანონი", 20),
    ("საქართველოს მინისტრის ბრძანება", 20),
    ("საქართველოს მთავრობის დადგენილება", 15),
    ("საქართველოს მთავრობის განკარგულება", 10),
    ("საკრებულოს დადგენილება", 20),  # both municipal doc_type variants
    ("საქართველოს პრეზიდენტის ბრძანებულება", 10),
    ("საქართველოს პარლამენტის დადგენილება", 8),
    ("*other", 17),
]
SOURCE_ALLOC = {"matsne": 150, "napr": 45, "ecd": 45, "constcourt": 35, "tas": 22, "tbappeal": 13}


def matsne_bucket(d: dict) -> str:
    if d.get("is_consolidated"):
        return "consolidated"
    t = d.get("document_type") or ""
    for name, _ in MATSNE_TYPE_BUCKETS[1:-1]:
        if name == "საკრებულოს დადგენილება":
            if "საკრებულოს დადგენილება" in t:
                return name
        elif t == name:
            return name
    return "*other"


def spread_sample(rng: random.Random, docs: list[dict], n: int) -> list[dict]:
    """Sample n docs spread across decade × status cells, proportional with ≥1 floors."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for d in docs:
        cells[((d.get("date") or "")[:3], d.get("status") or "")].append(d)
    total = len(docs)
    if total <= n:
        return list(docs)
    picked: list[dict] = []
    quotas = {}
    for key, members in sorted(cells.items()):
        quotas[key] = max(1, round(n * len(members) / total)) if len(members) else 0
    # trim/expand to exactly n, largest cells absorb the rounding
    while sum(quotas.values()) > n:
        k = max(quotas, key=lambda k: (quotas[k], len(cells[k])))
        quotas[k] -= 1
    while sum(quotas.values()) < n:
        k = max(cells, key=lambda k: len(cells[k]) - quotas[k])
        quotas[k] += 1
    for key, members in sorted(cells.items()):
        q = min(quotas[key], len(members))
        picked.extend(rng.sample(sorted(members, key=lambda d: d["document_id"]), q))
    return picked


def assign_pairs(rng: random.Random, picked: list[dict]) -> list[dict]:
    """Assign each doc 1–3 planned query types until the per-slice targets are covered."""
    need = dict(NEW_PAIRS)
    for d in picked:
        d["planned_types"] = []

    def can_cite(d):
        return bool(d.get("document_number") or d.get("registration_code")) or \
            d["source"] in ("ecd", "tas", "constcourt", "napr")

    def can_temporal(d):
        return bool(d.get("status")) or bool(d.get("expiry_date")) or bool(d.get("in_force_date"))

    pools = {
        "legal_citation": [d for d in picked if can_cite(d)],
        "temporal": [d for d in picked if can_temporal(d)],
    }
    generic = ["cross_lingual", "natural_question", "keyword", "paraphrase"]
    for t in ("legal_citation", "temporal"):
        pool = pools[t]
        rng.shuffle(pool)
        for d in pool:
            if need[t] == 0:
                break
            if len(d["planned_types"]) >= 2:
                continue
            d["planned_types"].append(t)
            need[t] -= 1
    # generic types round-robin over the least-loaded docs for diversity
    order = sorted(picked, key=lambda d: (len(d["planned_types"]), rng.random()))
    gi = 0
    while any(need[t] for t in generic):
        progressed = False
        for d in order:
            if all(need[t] == 0 for t in generic):
                break
            if len(d["planned_types"]) >= 2:
                continue
            for _ in range(len(generic)):
                t = generic[gi % len(generic)]
                gi += 1
                if need[t] > 0 and t not in d["planned_types"]:
                    d["planned_types"].append(t)
                    need[t] -= 1
                    progressed = True
                    break
        if not progressed:  # everyone is at cap — allow a third pair on citation docs
            for d in order:
                if all(need[t] == 0 for t in generic):
                    break
                if len(d["planned_types"]) >= 3:
                    continue
                for t in generic:
                    if need[t] > 0 and t not in d["planned_types"]:
                        d["planned_types"].append(t)
                        need[t] -= 1
                        break
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--universe", required=True, metavar="PATH")
    ap.add_argument("--out", required=True, metavar="PATH")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    excluded: set[tuple[str, str]] = set(I1_PATCH_IDS)
    gold = goldset.load_golden_set()
    excluded |= {(q.gold_source, q.gold_document_id) for q in gold}
    excluded |= {(q.source, q.document_id) for q in gold}
    excluded |= goldset.load_holdout()

    by_source: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.universe, encoding="utf-8"):
        d = json.loads(line)
        if (d.get("source"), d.get("document_id")) in excluded:
            continue
        if not d.get("document_id"):
            continue
        by_source[d["source"]].append(d)

    picked: list[dict] = []
    for source, n in SOURCE_ALLOC.items():
        docs = by_source.get(source, [])
        if source == "matsne":
            buckets: dict[str, list[dict]] = defaultdict(list)
            for d in docs:
                buckets[matsne_bucket(d)].append(d)
            for bucket, bn in MATSNE_TYPE_BUCKETS:
                sub = spread_sample(rng, buckets.get(bucket, []), bn)
                for d in sub:
                    d["stratum"] = f"matsne/{bucket}"
                picked.extend(sub)
        else:
            sub = spread_sample(rng, docs, n)
            for d in sub:
                d["stratum"] = source
            picked.extend(sub)

    picked = assign_pairs(rng, picked)
    rng.shuffle(picked)  # consumption order should not correlate with strata

    with open(args.out, "w", encoding="utf-8") as f:
        for d in picked:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_pairs = Counter(t for d in picked for t in d["planned_types"])
    print(f"seed={args.seed} candidates={len(picked)} "
          f"(excluded {len(excluded)} contaminated doc-ids)")
    for t, n in sorted(n_pairs.items()):
        print(f"  planned {t:<17} {n:>3} / target {NEW_PAIRS[t]}")
    per_src = Counter(d["source"] for d in picked)
    print("  docs per source:", dict(sorted(per_src.items())))


if __name__ == "__main__":
    main()
