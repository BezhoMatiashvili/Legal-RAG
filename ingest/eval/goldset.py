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


@dataclass(frozen=True)
class Relevance:
    document_id: str
    evidence_quote: str
    char_start: int
    char_end: int
    grade: int


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


def load_golden_set(path: Path = DEFAULT_GOLD) -> list[GoldQuery]:
    """Parse the JSONL golden set, skipping the leading ``#`` comment lines."""
    out: list[GoldQuery] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            rels = [
                Relevance(
                    document_id=x["document_id"],
                    evidence_quote=x["evidence_quote"],
                    char_start=x["char_start"],
                    char_end=x["char_end"],
                    grade=int(x["grade"]),
                )
                for x in r.get("relevance", [])
            ]
            out.append(
                GoldQuery(
                    id=r["id"],
                    query=r["query"],
                    query_type=r["query_type"],
                    query_language=r["query_language"],
                    source=r["source"],
                    document_id=r["document_id"],
                    gold_source=r["gold"]["source"],
                    gold_document_id=r["gold"]["document_id"],
                    relevance=rels,
                    answer=r.get("answer", ""),
                    doc_title=r.get("doc_title", ""),
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
    """The ``(source, document_id)`` set the gold queries cite (also the eval-source set)."""
    return {(q.gold_source, q.gold_document_id) for q in gold}


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


def reground(gold: list[GoldQuery], bodies: SnapshotBodies) -> int:
    """Assert every evidence span slices back exactly (NFC). Returns #spans checked.

    Raises ``ValueError`` naming the offending ids on any drift.
    """
    drift: list[str] = []
    checked = 0
    for q in gold:
        for rel in q.relevance:
            checked += 1
            body = bodies.body(q.gold_source, rel.document_id)
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
    missing = sorted(
        {(q.gold_source, q.gold_document_id) for q in gold}
        - holdout
    )
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
        by_doc: dict[str, list[Relevance]] = {}
        for rel in q.relevance:
            by_doc.setdefault(rel.document_id, []).append(rel)
        for document_id, rels in by_doc.items():
            body = bodies.body(q.gold_source, document_id)
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
