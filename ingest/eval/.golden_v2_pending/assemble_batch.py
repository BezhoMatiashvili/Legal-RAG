#!/usr/bin/env python
"""Assemble author outputs into golden-schema records with COMPUTED offsets (I5 Phase 3).

Author agents supply verbatim quotes; offsets are never trusted from agents — this script
locates each quote in the bundled body (exact bytes of the snapshot body_markdown),
computes char_start/char_end, re-slices to verify, and emits the batch jsonl. Pairs whose
quote does not locate are listed for rework, not guessed.

Usage: assemble_batch.py <batch_dir>   # reads <batch_dir>/raw/*.json (author outputs)
Writes <batch_dir>/batch.jsonl, <batch_dir>/translations.json, <batch_dir>/problems.json
"""
import json
import sys
import unicodedata
from pathlib import Path

S = Path(__file__).resolve().parent
BATCH = Path(sys.argv[1])
PLAN = {r["qid"]: r for r in map(json.loads, open(S / "pair_plan.jsonl", encoding="utf-8"))}

NOTE = ("generated from stratified-sampled doc (seed 20260710) + machine-validated "
        "(reground/coverage/near-dup/paraphrase/citation) + adversarially verified")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def body_path(source, document_id):
    return S / "docs" / f"{source}__{document_id.replace('/', '_')}.body.txt"


records, translations, problems = [], {}, []
seen_qids = set()
for f in sorted(BATCH.glob("raw/*.json")):
    for pair in json.loads(f.read_text(encoding="utf-8")):
        qid = pair["qid"]
        if qid in seen_qids:
            problems.append({"qid": qid, "file": f.name, "problem": "duplicate qid in outputs"})
            continue
        seen_qids.add(qid)
        plan = PLAN.get(qid)
        if plan is None:
            problems.append({"qid": qid, "file": f.name, "problem": "qid not in pair plan"})
            continue
        if pair.get("query_type") != plan["query_type"]:
            problems.append({"qid": qid, "file": f.name,
                             "problem": f"type {pair.get('query_type')} != planned {plan['query_type']}"})
            continue
        source, did = plan["source"], plan["document_id"]
        body = body_path(source, did).read_text(encoding="utf-8")
        quote = nfc(pair["evidence_quote"])
        start = body.find(quote)
        if start < 0:
            problems.append({"qid": qid, "file": f.name, "problem": "quote not found verbatim in body",
                             "quote_head": quote[:80]})
            continue
        end = start + len(quote)
        assert nfc(body[start:end]) == quote
        lang = pair.get("query_language") or ("en" if plan["query_type"] == "cross_lingual" else "ka")
        rec = {
            "id": qid, "query": pair["query"].strip(), "query_type": plan["query_type"],
            "query_language": lang, "source": source, "document_id": did,
            "gold": {"source": source, "document_id": did},
            "relevance": [{"document_id": did, "evidence_quote": body[start:end],
                           "char_start": start, "char_end": end, "grade": 2}],
            "answer": pair["answer"].strip(), "doc_title": pair["doc_title"].strip(),
            "note": NOTE,
        }
        records.append(rec)
        if plan["query_type"] == "cross_lingual":
            ka = (pair.get("ka_translation") or "").strip()
            if not ka:
                problems.append({"qid": qid, "file": f.name, "problem": "cross_lingual missing ka_translation"})
                records.pop()
                continue
            translations[qid] = {"query": rec["query"], "ka": ka}

missing = sorted(set(PLAN) & set()) # placeholder no-op
records.sort(key=lambda r: int(r["id"][1:]))
with open(BATCH / "batch.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(BATCH / "translations.json").write_text(
    json.dumps(translations, ensure_ascii=False, indent=1), encoding="utf-8")
(BATCH / "problems.json").write_text(
    json.dumps(problems, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"assembled {len(records)} records, {len(translations)} translations, {len(problems)} problems")
for p in problems:
    print("  PROBLEM:", p)
