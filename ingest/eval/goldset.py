"""Load and validate the span-anchored golden eval set against the clean snapshot.

The golden set (`golden_set_v1.jsonl`) stores judgments as char spans into each document's
cleaned ``body_markdown`` — never chunk ids — so it survives re-chunking/re-embedding. On
load we:

  * **re-ground** every ``evidence_quote``: assert ``body[char_start:char_end] ==
    evidence_quote`` under NFC. Any drift means hygiene/normalization changed and every
    downstream score would be silently wrong — so we fail loudly with the offending ids.
  * **enforce the holdout**: every gold document must be in ``holdout_doc_ids.json`` (the
    anti-contamination set), split by document.
  * **lint span coverage**: every span must map to ≥1 chunk under the current config; a span
    that maps to none is an annotation bug to fix, not to score around.
"""

import hashlib
import json
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .spanmap import map_spans_to_chunks

EVAL_DIR = Path(__file__).resolve().parent
INGEST_ROOT = EVAL_DIR.parent
DEFAULT_GOLD = EVAL_DIR / "golden_set_v1.jsonl"
DEFAULT_HOLDOUT = EVAL_DIR / "holdout_doc_ids.json"
DEFAULT_SNAPSHOT_DOCS = INGEST_ROOT / "snapshots" / "v1" / "docs"
EVAL_SET_VERSION = "v1"  # default eval set; runs select others via EVAL_SETS

DEFAULT_GOLD_V2 = EVAL_DIR / "golden_set_v2.jsonl"
DEFAULT_HOLDOUT_V2 = EVAL_DIR / "holdout_doc_ids_v2.json"
V2_DELTA_DOCS = INGEST_ROOT / "snapshots" / "v2-delta" / "docs"


@dataclass(frozen=True)
class EvalSetSpec:
    """One eval-set version: golden file, holdout file, and snapshot roots (primary first).

    ``roots`` order is load-bearing: on a ``document_id`` collision the primary (first)
    root's body wins, so a delta snapshot can never shadow the frozen v1 bodies.
    """

    version: str
    gold: Path
    holdout: Path
    roots: tuple[Path, ...]


EVAL_SETS: dict[str, EvalSetSpec] = {
    "v1": EvalSetSpec("v1", DEFAULT_GOLD, DEFAULT_HOLDOUT, (DEFAULT_SNAPSHOT_DOCS,)),
    "v2": EvalSetSpec(
        "v2", DEFAULT_GOLD_V2, DEFAULT_HOLDOUT_V2, (DEFAULT_SNAPSHOT_DOCS, V2_DELTA_DOCS)
    ),
}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _evidence_group_id(record: dict) -> str | None:
    for field_name in ("evidence_group", "equivalence_group", "group_id"):
        if record.get(field_name) is not None:
            return str(record[field_name])
    return None


@dataclass(frozen=True)
class Relevance:
    document_id: str
    evidence_quote: str
    char_start: int
    char_end: int
    grade: int
    # Spans with the same explicit group id are alternative acceptable evidence.  With no
    # id, each annotation becomes its own required evidence unit in build_query_relevance.
    evidence_group: str | None = None
    required: bool = True
    source: str | None = None
    # v3 provenance.  Frozen v1/v2 records leave these unset, preserving their exact
    # historical interpretation while allowing the v3 adapter to retain canonical version
    # identity all the way to qrel construction.
    version_id: str | None = None
    lineage_family_id: str | None = None
    evidence_id: str | None = None
    article_id: str | None = None


@dataclass(frozen=True)
class GoldQuery:
    id: str
    query: str
    query_type: str
    query_language: str
    source: str
    document_id: str
    gold_source: str
    gold_document_id: str
    relevance: list[Relevance] = field(default_factory=list)
    answer: str = ""
    doc_title: str = ""
    # Bootstrap/permutation cluster.  v3 may provide an amendment/version-lineage family;
    # frozen v1/v2 default to the gold document family.
    cluster_id: str = ""
    # Optional v3 policy/slice metadata.  Defaults keep every v1/v2 constructor and loader
    # backward compatible.
    as_of: str | None = None
    split: str = ""
    risk_level: str = ""
    tags: tuple[str, ...] = ()
    expected_outcome: str = "answer"
    partition_family_ids: tuple[str, ...] = ()
    dataset_id: str = ""
    corpus_generation: str = ""
    gold_version_id: str | None = None


def load_golden_set(path: Path = DEFAULT_GOLD) -> list[GoldQuery]:
    """Parse the JSONL golden set, skipping the leading ``#`` comment lines."""
    out: list[GoldQuery] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            gold_record = r["gold"]
            rels = [
                Relevance(
                    document_id=x["document_id"],
                    evidence_quote=x["evidence_quote"],
                    char_start=x["char_start"],
                    char_end=x["char_end"],
                    grade=int(x["grade"]),
                    evidence_group=_evidence_group_id(x),
                    required=bool(x.get("required", True)),
                    source=x.get("source"),
                    version_id=x.get("version_id"),
                    lineage_family_id=x.get("lineage_family_id"),
                    evidence_id=x.get("evidence_id"),
                    article_id=x.get("article_id"),
                )
                for x in r.get("relevance", [])
            ]
            family = (
                r.get("cluster_id")
                or r.get("version_family")
                or r.get("document_family")
                or gold_record.get("version_family")
                or gold_record.get("document_family")
                or f"{gold_record['source']}:{gold_record['document_id']}"
            )
            out.append(
                GoldQuery(
                    id=r["id"],
                    query=r["query"],
                    query_type=r["query_type"],
                    query_language=r["query_language"],
                    source=r["source"],
                    document_id=r["document_id"],
                    gold_source=gold_record["source"],
                    gold_document_id=gold_record["document_id"],
                    relevance=rels,
                    answer=r.get("answer", ""),
                    doc_title=r.get("doc_title", ""),
                    cluster_id=str(family),
                    as_of=r.get("as_of"),
                    split=str(r.get("split", "")),
                    risk_level=str(r.get("risk_level", "")),
                    tags=tuple(str(tag) for tag in r.get("tags", ())),
                    expected_outcome=str(r.get("expected_outcome", "answer")),
                    partition_family_ids=tuple(
                        str(item) for item in r.get("partition_family_ids", ())
                    ),
                    dataset_id=str(r.get("dataset_id", "")),
                    corpus_generation=str(r.get("corpus_generation", "")),
                    gold_version_id=gold_record.get("version_id"),
                )
            )
    return out


def load_holdout(path: Path = DEFAULT_HOLDOUT) -> set[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(x["source"], x["document_id"]) for x in data}


def eval_set_hash(path: Path = DEFAULT_GOLD) -> str:
    """Short content hash of the eval-set file, for stamping every score."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# Module-level cache: (root, source) → {doc_id: body}. Shared across SnapshotBodies
# instances so the huge per-source jsonl (matsne is multi-GB) is read at most once.
_SOURCE_CACHE: dict[tuple[str, str], dict[str, str]] = {}


def gold_docs(gold: list[GoldQuery]) -> set[tuple[str, str]]:
    """All ``(source, document_id)`` pairs referenced by required evidence.

    Frozen v1/v2 use one source/document per query.  The general form matters for v3
    multi-document questions and prevents secondary evidence documents from being omitted
    from snapshot loading or holdout enforcement.
    """
    out: set[tuple[str, str]] = set()
    for q in gold:
        out.add((q.gold_source, q.gold_document_id))
        out.update((rel.source or q.gold_source, rel.document_id) for rel in q.relevance)
    return out


class SnapshotBodies:
    """Loader of cleaned ``body_markdown`` keyed by ``(source, document_id)``.

    Pass ``needed`` (the set of ``(source, document_id)`` the harness will request) to keep
    only those bodies in memory — the docs jsonl are multi-GB, so loading everything would
    cost gigabytes. ``needed=None`` loads a whole source (general use, not the eval path).

    ``extra_roots`` layers additive snapshot deltas (e.g. ``snapshots/v2-delta/docs``) under
    the primary root: all roots are consulted, the primary wins on a doc-id collision, and a
    source file may be absent from any root as long as at least one root has it. The cache
    stays keyed per (root, source), so a v1-only instance and a multi-root instance share
    the v1 entries.
    """

    def __init__(
        self,
        root: Path = DEFAULT_SNAPSHOT_DOCS,
        needed: set[tuple[str, str]] | None = None,
        *,
        extra_roots: Sequence[Path] = (),
    ):
        self.roots = [Path(root), *(Path(r) for r in extra_roots)]
        self.root = self.roots[0]
        self._needed: dict[str, set[str]] | None = None
        if needed is not None:
            self._needed = {}
            for source, document_id in needed:
                self._needed.setdefault(source, set()).add(document_id)

    def source_files(self, source: str) -> list[Path]:
        """Existing per-root docs files for ``source``, primary root first."""
        return [r / f"{source}.jsonl" for r in self.roots if (r / f"{source}.jsonl").exists()]

    def _merged_cache(self, source: str) -> dict[str, str]:
        merged: dict[str, str] = {}
        for root in reversed(self.roots):  # primary applied last → wins on collision
            merged.update(_SOURCE_CACHE.get((str(root), source), {}))
        return merged

    def _load_source(self, source: str) -> dict[str, str]:
        want = None if self._needed is None else self._needed.get(source, set())
        if len(self.roots) == 1:
            key = (str(self.root), source)
            cached = _SOURCE_CACHE.get(key, {})
            if want is not None and want <= set(cached):
                return cached  # everything we need is already loaded
            table = dict(cached)
            with open(self.root / f"{source}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if want is None or d["document_id"] in want:
                        table[d["document_id"]] = d["body_markdown"]
            _SOURCE_CACHE[key] = table
            return table

        merged = self._merged_cache(source)
        if want is not None and want <= set(merged):
            return merged  # everything we need is already loaded across the roots
        paths = [root / f"{source}.jsonl" for root in self.roots]
        if not any(p.exists() for p in paths):
            raise FileNotFoundError(f"no docs file for source {source!r} in any snapshot root: {paths}")
        for root, path in zip(self.roots, paths):
            if not path.exists():
                continue
            key = (str(root), source)
            table = dict(_SOURCE_CACHE.get(key, {}))
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if want is None or d["document_id"] in want:
                        table[d["document_id"]] = d["body_markdown"]
            _SOURCE_CACHE[key] = table
        return self._merged_cache(source)

    def body(self, source: str, document_id: str) -> str:
        try:
            return self._load_source(source)[document_id]
        except KeyError as e:
            raise KeyError(f"doc not in snapshot: {source}:{document_id}") from e


def relevance_body(bodies: object, source: str, relevance: Relevance) -> str:
    """Resolve the exact canonical version for v3, with a v1/v2-compatible fallback."""

    body_version = getattr(bodies, "body_version", None)
    if relevance.version_id is not None and callable(body_version):
        return body_version(source, relevance.document_id, relevance.version_id)
    body = getattr(bodies, "body")
    return body(source, relevance.document_id)


def reground(gold: list[GoldQuery], bodies: SnapshotBodies) -> int:
    """Assert every evidence span slices back exactly (NFC). Returns #spans checked.

    Raises ``ValueError`` naming the offending ids on any drift.
    """
    drift: list[str] = []
    checked = 0
    for q in gold:
        for rel in q.relevance:
            checked += 1
            source = rel.source or q.gold_source
            body = relevance_body(bodies, source, rel)
            got = _nfc(body[rel.char_start : rel.char_end])
            if got != _nfc(rel.evidence_quote):
                drift.append(f"{q.id}[{rel.char_start}:{rel.char_end}]")
    if drift:
        raise ValueError(
            f"re-grounding drift on {len(drift)} span(s) — hygiene/normalization changed: "
            + ", ".join(drift)
        )
    return checked


def enforce_holdout(gold: list[GoldQuery], holdout: set[tuple[str, str]]) -> None:
    """Every gold document must be in the reserved holdout set."""
    missing = sorted(gold_docs(gold) - holdout)
    if missing:
        raise ValueError(f"gold docs missing from holdout ({len(missing)}): {missing}")


def lint_span_coverage(
    gold: list[GoldQuery],
    bodies: SnapshotBodies,
    *,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    count_tokens: Callable[[str], int],
) -> int:
    """Every span must map to ≥1 chunk under the current config. Returns #spans linted.

    Raises ``ValueError`` naming the ids whose span maps to no chunk (fail loud, don't
    silently mis-score).
    """
    empty: list[str] = []
    linted = 0
    for q in gold:
        # group spans by their (gold) document so we chunk each body once
        by_doc: dict[tuple[str, str, str | None], list[Relevance]] = {}
        for rel in q.relevance:
            source = rel.source or q.gold_source
            by_doc.setdefault((source, rel.document_id, rel.version_id), []).append(rel)
        for (source, _document_id, _version_id), rels in by_doc.items():
            body = relevance_body(bodies, source, rels[0])
            mapping = map_spans_to_chunks(
                body,
                [(r.char_start, r.char_end) for r in rels],
                max_tokens=max_tokens, overlap=overlap, min_tokens=min_tokens,
                count_tokens=count_tokens,
            )
            for i, chunk_set in mapping.items():
                linted += 1
                if not chunk_set:
                    r = rels[i]
                    empty.append(f"{q.id}[{r.char_start}:{r.char_end}]")
    if empty:
        raise ValueError(
            f"{len(empty)} span(s) map to no chunk under the current config: "
            + ", ".join(empty)
        )
    return linted
