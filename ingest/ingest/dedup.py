"""Corpus deduplication analysis — report clusters, never delete.

Three signals, matched to what the corpus actually contains (measured 2026-07-07 over the
full corpus):

  * **exact**      — identical cleaned bodies (verbatim re-publications; matsne ~63
                     clusters, constcourt ~4). Keyed on the body ``content_hash``.
  * **amendment**  — matsne acts sharing a ``registration_code`` are versions of one act
                     (~5k docs across ~87 codes). Linked as a version group, NOT merged.
                     Over-large groups are flagged as *suspect* (a shared code that is not a
                     true amendment lineage) so a human reviews them.
  * **near**       — near-identical bodies via MinHash/LSH over token shingles (formatting /
                     number variants that the exact hash misses). Uses ``datasketch``; if it
                     is unavailable the near-dup pass is reported as skipped, never faked.

Amended/duplicate documents are **kept**; this module only emits cluster reports so a human
can review them (prompt.md:95 — never auto-delete).
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

SHINGLE_K = 5                 # word-level shingle width for near-dup
MINHASH_PERMS = 128
NEAR_DUP_THRESHOLD = 0.85     # estimated Jaccard at/above which two docs are near-dups
LARGE_AMENDMENT_CLUSTER = 40  # amendment groups larger than this are flagged suspect


def content_hash(clean_body: str) -> str:
    """Stable SHA-256 of the *cleaned* body — the identity used for exact-dup + snapshot."""
    return hashlib.sha256(clean_body.encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class Cluster:
    kind: str                 # "exact" | "amendment" | "near"
    key: str                  # content hash / registration_code / representative id
    members: list[str]        # doc_ids in the cluster
    suspect: bool = False      # amendment: shared code that looks non-lineage

    @property
    def size(self) -> int:
        return len(self.members)


def cluster_by_key(pairs, kind: str) -> list[Cluster]:
    """Group ``(doc_id, key)`` pairs into clusters of size > 1 (skips falsy keys)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for doc_id, key in pairs:
        if key:
            groups[str(key)].append(doc_id)
    out = []
    for key, ids in groups.items():
        if len(ids) > 1:
            suspect = kind == "amendment" and len(ids) > LARGE_AMENDMENT_CLUSTER
            out.append(Cluster(kind=kind, key=key, members=sorted(ids), suspect=suspect))
    return out


def _shingles(clean_body: str, k: int = SHINGLE_K) -> set[str]:
    words = clean_body.split()
    if len(words) < k:
        return {clean_body.strip()} if clean_body.strip() else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


@dataclass
class NearDupResult:
    clusters: list[Cluster] = field(default_factory=list)
    skipped: bool = False
    reason: str | None = None


def near_dup_clusters(docs, *, threshold: float = NEAR_DUP_THRESHOLD) -> NearDupResult:
    """Cluster near-identical bodies with MinHash + LSH.

    ``docs`` is an iterable of ``(doc_id, clean_body)``. Returns clusters (size > 1) of
    documents whose estimated Jaccard similarity is ≥ ``threshold``. If ``datasketch`` is
    unavailable the pass is skipped (result flagged), never approximated incorrectly.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        return NearDupResult(skipped=True, reason="datasketch not installed")

    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_PERMS)
    sigs: dict[str, "MinHash"] = {}
    for doc_id, body in docs:
        sh = _shingles(body or "")
        if not sh:
            continue
        mh = MinHash(num_perm=MINHASH_PERMS)
        for s in sh:
            mh.update(s.encode("utf-8", "replace"))
        sigs[doc_id] = mh
        lsh.insert(doc_id, mh)

    # Union-find over LSH candidate pairs, verified by estimated Jaccard.
    parent: dict[str, str] = {d: d for d in sigs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for doc_id, mh in sigs.items():
        for cand in lsh.query(mh):
            if cand != doc_id and sigs[doc_id].jaccard(sigs[cand]) >= threshold:
                union(doc_id, cand)

    groups: dict[str, list[str]] = defaultdict(list)
    for d in sigs:
        groups[find(d)].append(d)
    clusters = [
        Cluster(kind="near", key=sorted(ids)[0], members=sorted(ids))
        for ids in groups.values()
        if len(ids) > 1
    ]
    return NearDupResult(clusters=clusters)


def cluster_stats(clusters: list[Cluster]) -> dict:
    """Summary counts for a report: #clusters, #docs covered, largest, #suspect."""
    if not clusters:
        return {"clusters": 0, "docs_in_clusters": 0, "largest": 0, "suspect": 0}
    sizes = [c.size for c in clusters]
    return {
        "clusters": len(clusters),
        "docs_in_clusters": sum(sizes),
        "largest": max(sizes),
        "suspect": sum(1 for c in clusters if c.suspect),
    }
