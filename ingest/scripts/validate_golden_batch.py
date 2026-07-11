#!/usr/bin/env python
"""Machine-validate a golden-set candidate batch before adversarial review (I5).

Runs every DETERMINISTIC check a new golden pair must pass, so the human/agent
verification pass only spends judgment on semantics (does the span answer the query, is
the grade right, is the phrasing natural). Checks, all fail-listed per pair:

  (a) reground   — evidence_quote slices back byte-exactly (NFC) from the snapshot
                   body_markdown, via the REAL ``eval.goldset.reground`` over the eval
                   set's multi-root bodies (v1 + v2-delta);
  (b) coverage   — every span maps to ≥1 chunk under the live chunk config with the bge
                   tokenizer (exactly what ``--backend qdrant`` evaluates);
  (c) near-dup   — no query has >threshold token-Jaccard overlap with any existing
                   (v1 + v2-so-far + earlier-in-batch) query;
  (d) paraphrase — ``paraphrase`` queries share ZERO content words with their evidence
                   quote (stopwords stripped; digits count as content);
  (e) citation   — ``legal_citation`` queries' citation tokens (№-forms, digit runs, reg
                   codes) appear in the gold doc's snapshot metadata/body head; with
                   ``--qdrant`` also checked against the live chunk-0 payload;
  (f) consistency— schema fields, unique ids (incl. vs existing sets), gold doc listed in
                   the v2 holdout, v2 holdout ⊇ v1 holdout, query_language matches
                   ``detect_language``, grade ∈ {1,2,3}.

Usage (from ingest/):
    .venv/bin/python scripts/validate_golden_batch.py --batch /path/batch.jsonl
    .venv/bin/python scripts/validate_golden_batch.py --batch batch.jsonl --qdrant

Exit 0 iff the batch is machine-clean. No text leaves this box.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root

from eval import goldset  # noqa: E402
from ingest.config import load_config  # noqa: E402
from ingest.search import detect_language  # noqa: E402

# Small, script-local function-word lists — only to stop trivial overlaps (particles,
# copulas) from counting as "content". Digits are deliberately kept as content.
STOPWORDS_KA = frozenset(
    """და თუ ან რომ არის იყო არა არ ეს ის რა ვინ სად როდის როგორ რატომ თუმცა მაგრამ ასევე
    კი არა უნდა შეიძლება თუარა ამ იმ ერთი ორი მისი მათი ჩემი შენი ჩვენი თქვენი მე შენ ჩვენ
    თქვენ ისინი მას მან მათ ვის ვისი რის რისი აქ იქ ხოლო ანუ ე.ი. ეანუ როცა სადაც რომელიც
    რომლის რომელი რომლებიც იქნება იქნა იყოს არიან ყოფილა აქვს ჰქონდა ექნება მიერ შესახებ
    შემდეგ წინ ზე ში დან თან გან ვით მდე ისე ასე ან და ან""".split()
)
STOPWORDS_EN = frozenset(
    """a an the and or of to in on for with by at from as is are was were be been being
    it its this that these those which who whom whose what when where how why can could
    may might must shall should will would do does did done not no nor if then than so
    such about into under over between during before after above below up down out off
    again further once here there all any both each few more most other some own same""".split()
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# №-forms (№71, N1/2-345), slash/dash compound numbers (1გ/620-17), reg codes
# (010.090.000.05.001.016.312), long digit runs (ecd 18-digit ids), letter-prefixed ids
# (tas AR-…) — pragmatic union.
_CITATION_TOKEN_RE = re.compile(r"[№N#][\s]*[\w./–-]+|[A-Za-z]{1,3}-?\d{4,}|\d[\d./–-]{2,}[\w]*|\b\d{4,}\b")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tokens(text: str, *, content_only: bool = False) -> set[str]:
    toks = {t.casefold() for t in _WORD_RE.findall(_nfc(text))}
    if content_only:
        toks = {t for t in toks if t not in STOPWORDS_KA and t not in STOPWORDS_EN}
    return toks


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def citation_tokens(query: str) -> list[str]:
    """Digit-bearing citation tokens, normalized (№/whitespace stripped, dashes unified)."""
    out = []
    for m in _CITATION_TOKEN_RE.findall(_nfc(query)):
        t = m.lstrip("№N#").strip().replace("–", "-")
        if any(ch.isdigit() for ch in t):
            out.append(t)
    return out


def check_schema(batch: list, existing_ids: set[str]) -> list[str]:
    issues = []
    seen: set[str] = set()
    for q in batch:
        if q.id in existing_ids:
            issues.append(f"{q.id}: id collides with an existing golden-set id")
        if q.id in seen:
            issues.append(f"{q.id}: duplicate id within batch")
        seen.add(q.id)
        if not q.relevance:
            issues.append(f"{q.id}: no relevance spans")
        for rel in q.relevance:
            if rel.grade not in (1, 2, 3):
                issues.append(f"{q.id}: grade {rel.grade} outside 1..3")
            if not (0 <= rel.char_start < rel.char_end):
                issues.append(f"{q.id}: bad span [{rel.char_start}:{rel.char_end}]")
            if rel.document_id != q.gold_document_id:
                issues.append(f"{q.id}: relevance doc {rel.document_id} != gold doc {q.gold_document_id}")
        if (q.source, q.document_id) != (q.gold_source, q.gold_document_id):
            issues.append(f"{q.id}: top-level doc != gold doc (v1 recipe keeps them equal)")
        detected = detect_language(q.query)
        if q.query_language != detected:
            issues.append(f"{q.id}: query_language={q.query_language} but detect_language says {detected}")
        if not q.answer.strip():
            issues.append(f"{q.id}: empty answer")
        if not q.doc_title.strip():
            issues.append(f"{q.id}: empty doc_title")
    return issues


def check_reground(batch: list, bodies: goldset.SnapshotBodies) -> list[str]:
    issues = []
    for q in batch:
        try:
            goldset.reground([q], bodies)
        except (ValueError, KeyError) as e:
            issues.append(f"{q.id}: {e}")
    return issues


def check_span_coverage(batch: list, bodies, chunk_cfg: dict, count_tokens) -> list[str]:
    issues = []
    for q in batch:
        try:
            goldset.lint_span_coverage([q], bodies, count_tokens=count_tokens, **chunk_cfg)
        except (ValueError, KeyError) as e:
            issues.append(f"{q.id}: {e}")
    return issues


def check_near_dup(batch: list, existing: list[tuple[str, str]], threshold: float) -> list[str]:
    """existing = [(id, query)] from v1 + v2-so-far; batch also checked against itself."""
    issues = []
    pool = list(existing)
    for q in batch:
        qt = tokens(q.query)
        for other_id, other_query in pool:
            j = jaccard(qt, tokens(other_query))
            if j > threshold:
                issues.append(f"{q.id}: near-dup of {other_id} (jaccard={j:.2f})")
        pool.append((q.id, q.query))
    return issues


def check_paraphrase(batch: list, max_overlap: int = 0) -> list[str]:
    issues = []
    for q in batch:
        if q.query_type != "paraphrase":
            continue
        qt = tokens(q.query, content_only=True)
        for rel in q.relevance:
            shared = qt & tokens(rel.evidence_quote, content_only=True)
            if len(shared) > max_overlap:
                issues.append(f"{q.id}: paraphrase shares content words with evidence: {sorted(shared)[:8]}")
    return issues


def _doc_metadata(roots: Sequence[Path], needed: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """Full snapshot records (metadata + body head) for the needed docs, delta-aware."""
    by_source: dict[str, set[str]] = {}
    for source, did in needed:
        by_source.setdefault(source, set()).add(did)
    out: dict[tuple[str, str], dict] = {}
    for source, want in by_source.items():
        for root in reversed(list(roots)):  # primary applied last → wins on collision
            path = Path(root) / f"{source}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d["document_id"] in want:
                        out[(source, d["document_id"])] = d
    return out


def check_citations(batch: list, doc_meta: dict[tuple[str, str], dict],
                    live_payloads: dict[tuple[str, str], dict] | None = None) -> list[str]:
    issues = []
    for q in batch:
        if q.query_type != "legal_citation":
            continue
        cites = citation_tokens(q.query)
        if not cites:
            issues.append(f"{q.id}: legal_citation query has no extractable citation token")
            continue
        meta = doc_meta.get((q.gold_source, q.gold_document_id), {})
        if not meta:
            issues.append(f"{q.id}: gold doc not found in any snapshot root")
            continue
        hay = _nfc(" ".join(
            str(meta.get(k) or "")
            for k in ("document_number", "registration_code", "title", "doc_id", "document_id",
                      "date", "date_raw")
        ) + " " + str(meta.get("body_markdown", ""))[:800]).replace("–", "-")
        for tok in cites:
            if tok not in hay:
                issues.append(
                    f"{q.id}: citation token {tok!r} not in gold doc's "
                    f"document_number/registration_code/title/body-head — gold doc must BE the cited act"
                )
        if live_payloads is not None:
            pl = live_payloads.get((q.gold_source, q.gold_document_id))
            if pl is None:
                issues.append(f"{q.id}: gold doc missing from the live index")
            else:
                live_hay = _nfc(" ".join(
                    str(pl.get(k) or "") for k in ("document_number", "registration_code", "title")
                )).replace("–", "-")
                for tok in cites:
                    if tok not in live_hay and tok not in hay:
                        issues.append(f"{q.id}: citation token {tok!r} absent from live payload too")
    return issues


def check_holdout(batch: list, spec: goldset.EvalSetSpec) -> list[str]:
    issues = []
    try:
        holdout = goldset.load_holdout(spec.holdout)
    except FileNotFoundError:
        return [f"holdout file missing: {spec.holdout}"]
    v1_holdout = goldset.load_holdout(goldset.DEFAULT_HOLDOUT)
    if not v1_holdout <= holdout:
        issues.append(f"{spec.holdout.name} is not a superset of the frozen v1 holdout")
    for q in batch:
        if (q.gold_source, q.gold_document_id) not in holdout:
            issues.append(f"{q.id}: gold doc {q.gold_source}:{q.gold_document_id} not in {spec.holdout.name}")
    return issues


def fetch_live_payloads(batch: list, collection: str | None = None) -> dict[tuple[str, str], dict]:
    from ingest.qdrant_store import make_client, point_id

    cfg = load_config()
    client = make_client(cfg)
    coll = collection or cfg.collection_name
    docs = {(q.gold_source, q.gold_document_id) for q in batch}
    pid_map = {point_id(s, d, 0): (s, d) for s, d in docs}
    out: dict[tuple[str, str], dict] = {}
    ids = list(pid_map)
    for i in range(0, len(ids), 256):
        for pt in client.retrieve(collection_name=coll, ids=ids[i : i + 256],
                                  with_payload=["document_number", "registration_code", "title"]):
            out[pid_map[str(pt.id)]] = pt.payload or {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Machine-validate a golden-set candidate batch (I5).")
    ap.add_argument("--batch", required=True, metavar="PATH")
    ap.add_argument("--eval-set", choices=sorted(goldset.EVAL_SETS), default="v2")
    ap.add_argument("--existing", nargs="*", default=None,
                    help="golden files to dedup against (default: v1 + v2 if present)")
    ap.add_argument("--qdrant", action="store_true", help="also verify gold docs against the live index")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--near-dup-threshold", type=float, default=0.8)
    ap.add_argument("--paraphrase-max-overlap", type=int, default=0)
    args = ap.parse_args()

    spec = goldset.EVAL_SETS[args.eval_set]
    batch_path = Path(args.batch)
    batch = goldset.load_golden_set(batch_path)

    existing_paths = (
        [Path(p) for p in args.existing]
        if args.existing is not None
        else [p for p in (goldset.DEFAULT_GOLD, spec.gold) if p.exists() and p != batch_path]
    )
    existing: list[tuple[str, str]] = []
    for p in existing_paths:
        existing.extend((g.id, g.query) for g in goldset.load_golden_set(p))
    existing_ids = {i for i, _ in existing}

    cfg = load_config()
    chunk_cfg = {"max_tokens": cfg.chunk_tokens, "overlap": cfg.chunk_overlap,
                 "min_tokens": cfg.chunk_min_tokens}
    from ingest.embedding import make_token_counter

    count_tokens = make_token_counter(cfg.embed_model)

    needed = goldset.gold_docs(batch)
    bodies = goldset.SnapshotBodies(root=spec.roots[0], needed=needed, extra_roots=spec.roots[1:])
    doc_meta = _doc_metadata(spec.roots, needed)
    live = fetch_live_payloads(batch, args.collection) if args.qdrant else None

    report = {
        "schema": check_schema(batch, existing_ids),
        "reground": check_reground(batch, bodies),
        "span_coverage": check_span_coverage(batch, bodies, chunk_cfg, count_tokens),
        "near_dup": check_near_dup(batch, existing, args.near_dup_threshold),
        "paraphrase": check_paraphrase(batch, args.paraphrase_max_overlap),
        "citation": check_citations(batch, doc_meta, live),
        "holdout": check_holdout(batch, spec),
    }

    n_issues = sum(len(v) for v in report.values())
    for name, issues in report.items():
        status = "OK" if not issues else f"{len(issues)} issue(s)"
        print(f"[{name:>13}] {status}")
        for msg in issues:
            print(f"    - {msg}")
    if n_issues:
        print(f"\nFAILED: {n_issues} issue(s) across {len(batch)} pairs")
        raise SystemExit(1)
    print(f"\n{len(batch)} pairs machine-clean — hand off to adversarial verification")


if __name__ == "__main__":
    main()
