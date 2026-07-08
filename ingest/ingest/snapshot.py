"""Build a versioned, clean, deduplicated corpus snapshot (mission Phase 0 / Part 1).

Turns the raw scraped JSONL under ``artifacts/<source>/runs/*`` into a single versioned
snapshot that every downstream Part consumes instead of the raw files:

  * unions **all** runs per source (``latest/`` is a mirror of the newest run and is
    ignored — verified: ``latest ⊆ runs``), deduping by canonical ``doc_id`` keeping the
    newest run's version (runs processed newest→oldest, keep-first);
  * runs ``normalize()`` on every doc (malformed → counted, never fatal);
  * cleans each body (strip NUL/control chars, NFC) — personal data is retained;
  * classifies damage and routes empty/near-empty/mojibake docs to a **quarantine** file
    with a reason (never deleted — the raw is still in ``artifacts/``);
  * detects legal structure and records a stable ``content_hash`` per doc;
  * clusters exact dups + matsne amendment groups + (optional) MinHash near-dups — reported,
    never merged;
  * writes the clean snapshot, a quarantine file, profile/dedup/quarantine reports, and a
    ``manifest.json`` stamped with a **config hash** so every downstream artifact is
    traceable to the exact pipeline version.

Reports and logs carry only counts / coverage / doc-ids / cluster sizes — never body text
or personal-data values.
"""

import glob
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import dedup, hygiene, structure
from .config import REPO_ROOT, Config
from .sources import SOURCES, normalize

SNAPSHOT_PIPELINE_VERSION = "1"
SOURCES_PRESENT = ("matsne", "napr", "ecd", "constcourt", "tas", "tbappeal")
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "ingest" / "snapshots"


def _run_files_desc(source: str, artifacts_root: Path) -> list[tuple[str, str]]:
    """(run_label, path) for every run's items.jsonl, newest run first. latest/ is skipped
    because it duplicates the newest run."""
    out: list[tuple[str, str]] = []
    for path in glob.glob(str(artifacts_root / source / "runs" / "*" / "items.jsonl")):
        if os.path.getsize(path) == 0:
            continue
        run_label = Path(path).parent.name
        out.append((run_label, path))
    out.sort(key=lambda rp: rp[0], reverse=True)  # run_id starts with a sortable timestamp
    return out


def config_hash(cfg: Config) -> str:
    """Hash the parameters that define this snapshot so downstream artifacts are traceable."""
    material = {
        "pipeline_version": SNAPSHOT_PIPELINE_VERSION,
        # Bump when hygiene/structure *logic* changes (not just thresholds) so a rebuilt
        # snapshot is traceable to the exact code. r2: quarantine gate + mojibake ratio
        # measured on control-stripped text; article regex uses same-line whitespace.
        "logic_rev": "r2",
        "hygiene": {
            "near_empty_chars": hygiene.NEAR_EMPTY_CHARS,
            "mojibake_ratio": hygiene.MOJIBAKE_RATIO,
        },
        "dedup": {
            "shingle_k": dedup.SHINGLE_K,
            "minhash_perms": dedup.MINHASH_PERMS,
            "near_dup_threshold": dedup.NEAR_DUP_THRESHOLD,
            "large_amendment_cluster": dedup.LARGE_AMENDMENT_CLUSTER,
        },
        "chunk": {
            "tokens": cfg.chunk_tokens,
            "overlap": cfg.chunk_overlap,
            "min_tokens": cfg.chunk_min_tokens,
        },
        "embed_model": cfg.embed_model,
        "sources": sorted(SOURCES),
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class SourceStats:
    raw_lines: int = 0
    unique: int = 0
    malformed: int = 0
    clean: int = 0
    quarantined: Counter = None  # reason -> count
    nul_docs: int = 0
    control_docs: int = 0
    mojibake_docs: int = 0
    nfc_changed: int = 0
    char_lens: list = None
    structure: Counter = None  # marker -> docs
    kinds: Counter = None      # primary_kind -> docs
    doc_types: Counter = None
    exact: dict = None
    amendment: dict = None
    near: dict = None
    token_pctl: dict = None
    pii_fields: Counter = None

    def __post_init__(self):
        self.quarantined = Counter()
        self.char_lens = []
        self.structure = Counter()
        self.kinds = Counter()
        self.doc_types = Counter()
        self.pii_fields = Counter()


def _snapshot_record(source, doc, clean_body, chash, sinfo, run_label) -> dict:
    return {
        "snapshot_version": SNAPSHOT_PIPELINE_VERSION,
        "doc_id": f"{source}:{doc.document_id}",
        "source": source,
        "document_id": doc.document_id,
        "content_hash": chash,
        "title": doc.title,
        "date": doc.date,
        "date_raw": doc.date_raw,
        "language": doc.language,
        "document_type": doc.document_type,
        "court": doc.court,
        "source_url": doc.source_url,
        "document_number": doc.document_number,
        "registration_code": doc.registration_code,
        "parties": doc.parties,
        "status": doc.status,
        "status_raw": doc.status_raw,
        "in_force_date": doc.in_force_date,
        "expiry_date": doc.expiry_date,
        "promoted": doc.promoted,
        "structure": {
            "primary_kind": sinfo.primary_kind,
            "has_article": sinfo.has_article,
            "has_heading": sinfo.has_heading,
            "has_num_clause": sinfo.has_num_clause,
            "article_count": sinfo.article_count,
        },
        "body_char_len": len(clean_body),
        "source_run": run_label,        # provenance: raw is recoverable from artifacts/
        "body_markdown": clean_body,     # CLEANED text (raw preserved in artifacts/)
    }


def build_snapshot(
    cfg: Config,
    *,
    out_root: Path = DEFAULT_SNAPSHOT_ROOT,
    sources=SOURCES_PRESENT,
    limit: int | None = None,
    near_dup: bool = True,
    token_sample: int = 2000,
) -> dict:
    """Materialise snapshot v{SNAPSHOT_PIPELINE_VERSION}. Returns the manifest dict."""
    version = f"v{SNAPSHOT_PIPELINE_VERSION}"
    out_dir = out_root / version
    (out_dir / "docs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    quarantine_fp = open(out_dir / "quarantine.jsonl", "w", encoding="utf-8")

    stats: dict[str, SourceStats] = {}
    token_counter = _maybe_token_counter(cfg)

    for source in sources:
        st = SourceStats()
        seen: set[str] = set()
        exact_map: list[tuple[str, str]] = []       # (doc_id, content_hash)
        amend_map: list[tuple[str, str]] = []        # (doc_id, registration_code)
        docs_fp = open(out_dir / "docs" / f"{source}.jsonl", "w", encoding="utf-8")
        for run_label, path in _run_files_desc(source, cfg.artifacts_root):
            for line in open(path, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                st.raw_lines += 1
                try:
                    item = json.loads(line)
                    doc = normalize(source, item)
                except Exception:
                    st.malformed += 1
                    continue
                if doc.document_id in seen:
                    continue
                seen.add(doc.document_id)
                st.unique += 1
                if limit and st.unique > limit:
                    st.unique -= 1
                    break

                raw = doc.body_markdown
                report = hygiene.assess(raw)
                st.nul_docs += 1 if report.nul_chars else 0
                st.control_docs += 1 if report.control_chars else 0
                st.mojibake_docs += 1 if report.replacement_chars else 0
                st.nfc_changed += 1 if report.changed_by_nfc else 0
                st.doc_types[doc.document_type] += 1
                for k in doc.promoted:
                    st.pii_fields[k] += 1
                doc_id = f"{source}:{doc.document_id}"

                if not report.is_usable:
                    st.quarantined[report.quarantine_reason] += 1
                    quarantine_fp.write(json.dumps({
                        "doc_id": doc_id, "source": source, "document_id": doc.document_id,
                        "reason": report.quarantine_reason, "title": doc.title,
                        "source_run": run_label,
                        "damage": {"length": report.length, "meaningful": report.meaningful_chars,
                                   "nul": report.nul_chars, "control": report.control_chars,
                                   "replacement": report.replacement_chars},
                    }, ensure_ascii=False) + "\n")
                    continue

                clean_body = hygiene.clean_text(raw)
                chash = dedup.content_hash(clean_body)
                sinfo = structure.detect(clean_body)
                st.clean += 1
                st.char_lens.append(len(clean_body))
                st.kinds[sinfo.primary_kind] += 1
                for marker, present in (("article", sinfo.has_article), ("heading", sinfo.has_heading),
                                        ("num_clause", sinfo.has_num_clause), ("chapter", sinfo.has_chapter)):
                    if present:
                        st.structure[marker] += 1
                exact_map.append((doc_id, chash))
                if doc.registration_code:
                    amend_map.append((doc_id, doc.registration_code))
                docs_fp.write(json.dumps(
                    _snapshot_record(source, doc, clean_body, chash, sinfo, run_label),
                    ensure_ascii=False) + "\n")
            if limit and st.unique >= limit:
                break
        docs_fp.close()

        exact = dedup.cluster_by_key(exact_map, "exact")
        amend = dedup.cluster_by_key(amend_map, "amendment")
        st.exact = dedup.cluster_stats(exact)
        st.amendment = dedup.cluster_stats(amend)
        st.near = _near_dup_for_source(out_dir, source) if near_dup else {"skipped": True}
        st.token_pctl = _token_pctls(out_dir, source, token_counter, token_sample) if token_counter else None
        stats[source] = st
        print(f"  {source}: {st.clean} clean, {sum(st.quarantined.values())} quarantined, "
              f"{st.malformed} malformed  (exact-dups={st.exact['clusters']}, amend={st.amendment['clusters']})")

    quarantine_fp.close()
    manifest = _write_reports_and_manifest(cfg, out_dir, version, stats)
    return manifest


def _maybe_token_counter(cfg: Config):
    try:
        from .embedding import make_token_counter
        return make_token_counter(cfg.embed_model)
    except Exception:  # noqa: BLE001 — tokenizer/ML stack optional for the profile
        return None


def _token_pctls(out_dir, source, counter, sample):
    lens = []
    path = out_dir / "docs" / f"{source}.jsonl"
    for i, line in enumerate(open(path, encoding="utf-8")):
        if i >= sample:
            break
        rec = json.loads(line)
        lens.append(counter(rec["body_markdown"]))
    lens.sort()
    if not lens:
        return None
    p = lambda q: lens[min(len(lens) - 1, int(q / 100 * len(lens)))]  # noqa: E731
    return {"sample": len(lens), "p50": p(50), "p90": p(90), "p99": p(99), "max": lens[-1]}


def _near_dup_for_source(out_dir, source) -> dict:
    def rows():
        for line in open(out_dir / "docs" / f"{source}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            yield rec["doc_id"], rec["body_markdown"]
    res = dedup.near_dup_clusters(rows())
    if res.skipped:
        return {"skipped": True, "reason": res.reason}
    return {"skipped": False, **dedup.cluster_stats(res.clusters)}


def _pctl(vals, q):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q / 100 * len(s)))]


def _write_reports_and_manifest(cfg, out_dir, version, stats) -> dict:
    chash = config_hash(cfg)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    total_clean = sum(s.clean for s in stats.values())
    total_q = sum(sum(s.quarantined.values()) for s in stats.values())
    total_malformed = sum(s.malformed for s in stats.values())

    # --- profile.md ---
    prof = [f"# Corpus profile — snapshot {version}", "",
            f"Config hash: `{chash}` · created {created}", "",
            "| source | clean | quarantined | malformed | NUL | ctrl | mojibake | non-NFC | char p50/p90/p99/max |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--|"]
    for src, s in stats.items():
        prof.append(f"| {src} | {s.clean} | {sum(s.quarantined.values())} | {s.malformed} | "
                    f"{s.nul_docs} | {s.control_docs} | {s.mojibake_docs} | {s.nfc_changed} | "
                    f"{_pctl(s.char_lens,50)}/{_pctl(s.char_lens,90)}/{_pctl(s.char_lens,99)}/"
                    f"{max(s.char_lens) if s.char_lens else 0} |")
    prof += ["", "## Structure coverage (% of clean docs) & primary kind", ""]
    for src, s in stats.items():
        cov = ", ".join(f"{k} {100*v/max(1,s.clean):.0f}%" for k, v in s.structure.most_common())
        kinds = ", ".join(f"{k}={v}" for k, v in s.kinds.most_common())
        prof.append(f"- **{src}**: {cov or '—'}  · kinds: {kinds}")
        if s.token_pctl:
            tp = s.token_pctl
            prof.append(f"    - BGE-M3 tokens (sample {tp['sample']}): p50={tp['p50']} p90={tp['p90']} p99={tp['p99']} max={tp['max']}")
    prof += ["", "## PII field presence (counts only — values never exported)", ""]
    for src, s in stats.items():
        if s.pii_fields:
            prof.append(f"- **{src}**: " + ", ".join(f"{k}={v}" for k, v in s.pii_fields.most_common()))
    (out_dir / "reports" / "profile.md").write_text("\n".join(prof) + "\n", encoding="utf-8")

    # --- dedup.md ---
    ded = [f"# Dedup report — snapshot {version}", "",
           "Documents are reported, never deleted (prompt.md:95).", "",
           "| source | exact clusters (docs) | amendment clusters (docs, suspect) | near-dup |",
           "|---|--|--|--|"]
    for src, s in stats.items():
        near = "skipped" if s.near.get("skipped") else f"{s.near['clusters']} clusters ({s.near['docs_in_clusters']} docs)"
        ded.append(f"| {src} | {s.exact['clusters']} ({s.exact['docs_in_clusters']}) | "
                   f"{s.amendment['clusters']} ({s.amendment['docs_in_clusters']}, {s.amendment['suspect']} suspect) | {near} |")
    (out_dir / "reports" / "dedup.md").write_text("\n".join(ded) + "\n", encoding="utf-8")

    # --- quarantine.md ---
    quar = [f"# Quarantine report — snapshot {version}", "",
            "Quarantined docs are excluded from the clean set but preserved "
            "(`quarantine.jsonl` + raw in `artifacts/`).", "",
            "| source | " + " | ".join(sorted({r for s in stats.values() for r in s.quarantined})) + " |",
            "|---|" + "|".join("--" for _ in sorted({r for s in stats.values() for r in s.quarantined})) + "|"]
    reasons = sorted({r for s in stats.values() for r in s.quarantined})
    for src, s in stats.items():
        quar.append(f"| {src} | " + " | ".join(str(s.quarantined.get(r, 0)) for r in reasons) + " |")
    (out_dir / "reports" / "quarantine.md").write_text("\n".join(quar) + "\n", encoding="utf-8")

    # --- manifest.json ---
    manifest = {
        "snapshot_version": version,
        "pipeline_version": SNAPSHOT_PIPELINE_VERSION,
        "config_hash": chash,
        "created_at": created,
        "embed_model": cfg.embed_model,
        "totals": {"clean": total_clean, "quarantined": total_q, "malformed": total_malformed},
        "sources": {
            src: {
                "raw_lines": s.raw_lines, "unique": s.unique, "clean": s.clean,
                "malformed": s.malformed, "quarantined": dict(s.quarantined),
                "damage": {"nul": s.nul_docs, "control": s.control_docs,
                           "mojibake": s.mojibake_docs, "nfc_changed": s.nfc_changed},
                "dedup": {"exact": s.exact, "amendment": s.amendment, "near": s.near},
            }
            for src, s in stats.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSnapshot {version} written to {out_dir}")
    print(f"  clean={total_clean}  quarantined={total_q}  malformed={total_malformed}  config_hash={chash}")
    return manifest
