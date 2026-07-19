"""Full-corpus BM25 baseline — the mandatory neural-stack sanity floor, at 2.45M-chunk scale.

The reference :class:`eval.bm25.BM25Index` is the source of truth for the *scoring*, but it
keeps every chunk's token counts in Python objects and linear-scans them per query — at
2.45M chunks that is ~17 GB and minutes/query, and the harness' ``_ensure_bm25`` scroll is
just as impractical. This module builds the **identical** Okapi/Lucene BM25 index (same
``\\w+`` casefold tokenizer, same ``k1=1.5, b=0.75``, same non-negative Lucene IDF) but as a
compact CSR term→doc weight matrix, **streaming** so peak RAM stays ~3 GB, then persists it to
disk. Query time reads only the query terms' columns (mmap), so RAM stays tiny and each query
is milliseconds — letting the BM25 floor be measured over the *entire* corpus, the truest
apples-to-apples lexical comparison against the neural modes.

Scoring parity with :class:`eval.bm25.BM25Index` is asserted in ``tests/test_bm25_full.py``.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .bm25 import tokenize  # single source of truth for tokenization (\w+, casefold)

# Cache lives next to the eval package; git-ignored (see repo .gitignore).
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / ".bm25_full"

# BM25 hyper-parameters — must match eval.bm25.BM25Index defaults exactly.
_K1 = 1.5
_B = 0.75

_META = "meta.json"


class FullCorpusBM25:
    """A disk-backed, memory-mapped full-corpus Okapi BM25 index.

    The on-disk layout under ``index_dir``:
      - ``indptr.npy``  int64[V+1]  — CSR row pointers (one row per vocabulary term)
      - ``indices.npy`` int32[nnz]  — doc index of each nonzero (mmap at query time)
      - ``data.npy``    float32[nnz] — precomputed BM25 weight of (term, doc) (mmap)
      - ``terms.txt``   vocabulary, one token per line, line number == term id
      - ``keys.jsonl``  line j is a legacy ``[source, document_id, chunk_index]`` or
        canonical ``[source, document_id, version_id, chunk_index]`` identity
      - ``meta.json``   {k1, b, avgdl, N, V, nnz, collection}
    """

    def __init__(self, indptr, indices, data, vocab, keys, meta):
        self._indptr = indptr
        self._indices = indices  # may be an mmap
        self._data = data        # may be an mmap
        self._vocab = vocab      # token -> term id
        self._keys = keys        # legacy triples or canonical version-scoped quadruples
        self.meta = meta
        self._n_docs = int(meta["N"])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, client, collection, *, index_dir=DEFAULT_INDEX_DIR,
              batch: int = 8000, log=None) -> "FullCorpusBM25":
        """Two streaming passes over the collection → the persisted CSR index.

        Pass 1 builds the vocabulary, document frequencies, doc lengths and keys.
        Pass 2 fills preallocated CSR arrays with the BM25 weights. Peak RAM ~= the final
        matrix (nnz·8 B) plus the vocab — no full token list is ever held in memory.
        """
        index_dir = Path(index_dir)
        log = log or (lambda m: None)

        # ---- Pass 1: vocab, df, doc lengths, keys ------------------------------
        vocab: dict[str, int] = {}
        df: list[int] = []          # df[id] = number of docs containing term id
        doc_len: list[int] = []
        keys: list[tuple] = []
        t0 = time.perf_counter()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection, with_payload=True, with_vectors=False,
                limit=batch, offset=offset,
            )
            for pt in points:
                pl = pt.payload or {}
                toks = tokenize(pl.get("text") or "")
                seen: set[int] = set()
                for tok in toks:
                    i = vocab.get(tok)
                    if i is None:
                        i = len(vocab)
                        vocab[tok] = i
                        df.append(0)
                    if i not in seen:
                        seen.add(i)
                        df[i] += 1
                doc_len.append(len(toks))
                legacy_key = (
                    pl.get("source"),
                    pl.get("document_id"),
                    int(pl.get("chunk_index", -1)),
                )
                keys.append(
                    (
                        pl.get("source"),
                        pl.get("document_id"),
                        pl.get("version_id"),
                        int(pl.get("chunk_index", -1)),
                    )
                    if pl.get("version_id") is not None
                    else legacy_key
                )
            if len(keys) % 200_000 < batch:
                log(f"[bm25_full] pass1 scanned {len(keys):,} chunks "
                    f"| vocab={len(vocab):,} | {time.perf_counter() - t0:.0f}s")
            if offset is None:
                break

        n = len(keys)
        if n == 0:
            raise RuntimeError(f"collection {collection!r} is empty")
        v = len(vocab)
        df_arr = np.asarray(df, dtype=np.int64)
        dl_arr = np.asarray(doc_len, dtype=np.float64)
        avgdl = float(dl_arr.mean()) or 1.0  # all-empty corpus → avgdl 0; keep denom finite (weights are 0 anyway)
        # Lucene-style non-negative IDF — identical to eval.bm25.BM25Index.
        idf = np.log(1.0 + (n - df_arr + 0.5) / (df_arr + 0.5))  # float64[V]
        indptr = np.zeros(v + 1, dtype=np.int64)
        np.cumsum(df_arr, out=indptr[1:])
        nnz = int(indptr[-1])
        log(f"[bm25_full] pass1 done: N={n:,} V={v:,} nnz={nnz:,} avgdl={avgdl:.1f}")
        del df, df_arr  # free before allocating the matrix

        # ---- Pass 2: fill the CSR weight matrix --------------------------------
        indices = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float32)
        write = indptr[:-1].copy()  # per-term write cursor
        k1, b = _K1, _B
        t0 = time.perf_counter()
        offset = None
        j = 0
        while True:
            points, offset = client.scroll(
                collection_name=collection, with_payload=True, with_vectors=False,
                limit=batch, offset=offset,
            )
            for pt in points:
                pl = pt.payload or {}
                toks = tokenize(pl.get("text") or "")
                dl = doc_len[j]
                denom = k1 * (1.0 - b + b * dl / avgdl)
                for i, f in Counter(vocab[t] for t in toks).items():
                    w = idf[i] * (f * (k1 + 1.0)) / (f + denom)
                    pos = write[i]
                    indices[pos] = j
                    data[pos] = w
                    write[i] = pos + 1
                j += 1
            if j % 200_000 < batch:
                log(f"[bm25_full] pass2 weighted {j:,}/{n:,} chunks "
                    f"| {time.perf_counter() - t0:.0f}s")
            if offset is None:
                break
        if j != n:
            raise RuntimeError(
                f"scroll returned {j} docs in pass 2 but {n} in pass 1 — collection changed?")
        # Real check (not a bare assert — must survive `python -O`): the per-term write cursors
        # must land exactly on the df-derived segment ends, else the collection shifted mid-build.
        if not np.array_equal(write, indptr[1:]):
            raise RuntimeError("CSR fill did not match df layout (collection changed between passes?)")

        # ---- Persist ------------------------------------------------------------
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "indptr.npy", indptr)
        np.save(index_dir / "indices.npy", indices)
        np.save(index_dir / "data.npy", data)
        terms = [""] * v
        for tok, i in vocab.items():
            terms[i] = tok
        (index_dir / "terms.txt").write_text("\n".join(terms), encoding="utf-8")
        with (index_dir / "keys.jsonl").open("w", encoding="utf-8") as fh:
            for key in keys:
                fh.write(json.dumps(key, ensure_ascii=False) + "\n")
        meta = {"k1": k1, "b": b, "avgdl": avgdl, "N": n, "V": v, "nnz": nnz,
                "collection": collection}
        (index_dir / _META).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        log(f"[bm25_full] saved index to {index_dir}")
        return cls(indptr, indices, data, vocab, keys, meta)

    # ------------------------------------------------------------------- load
    @staticmethod
    def cache_exists(index_dir=DEFAULT_INDEX_DIR) -> bool:
        return (Path(index_dir) / _META).is_file()

    @classmethod
    def load(cls, index_dir=DEFAULT_INDEX_DIR, *, expect_collection=None) -> "FullCorpusBM25":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / _META).read_text(encoding="utf-8"))
        # Guard against a stale cache built for a different collection (else the bm25 floor
        # would be scored over a different corpus than the neural modes — silently wrong).
        if expect_collection is not None and meta.get("collection") != expect_collection:
            raise RuntimeError(
                f"BM25 cache at {index_dir} was built for collection {meta.get('collection')!r} "
                f"but the harness targets {expect_collection!r}; rebuild with "
                "`python -m eval.bm25_full build`.")
        indptr = np.load(index_dir / "indptr.npy")  # small, load fully
        indices = np.load(index_dir / "indices.npy", mmap_mode="r")
        data = np.load(index_dir / "data.npy", mmap_mode="r")
        terms = (index_dir / "terms.txt").read_text(encoding="utf-8").split("\n")
        vocab = {tok: i for i, tok in enumerate(terms)}
        keys: list[tuple] = []
        with (index_dir / "keys.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                raw_key = json.loads(line)
                if len(raw_key) == 3:
                    source, document_id, chunk_index = raw_key
                    keys.append((source, document_id, int(chunk_index)))
                elif len(raw_key) == 4:
                    source, document_id, version_id, chunk_index = raw_key
                    keys.append((source, document_id, version_id, int(chunk_index)))
                else:
                    raise RuntimeError(
                        f"corrupt bm25 cache key at {index_dir}: {raw_key!r}"
                    )
        if len(keys) != int(meta["N"]) or len(terms) != int(meta["V"]):
            raise RuntimeError(f"corrupt bm25 cache at {index_dir}")
        return cls(indptr, indices, data, vocab, keys, meta)

    @classmethod
    def build_or_load(cls, client, collection, *, index_dir=DEFAULT_INDEX_DIR, **kw):
        if cls.cache_exists(index_dir):
            return cls.load(index_dir)
        return cls.build(client, collection, index_dir=index_dir, **kw)

    # ------------------------------------------------------------------ query
    def search(self, query: str, k: int) -> list[tuple[tuple, float]]:
        """Return up to ``k`` ``(key, score)`` with score > 0, best first.

        ``key`` is a legacy triple or a canonical version-scoped quadruple, matching the
        collection generation from which this cache was built.
        Repeated query terms are summed (matching the reference, which iterates the raw
        token list), and only positive scores are returned.
        """
        if k <= 0:
            return []
        scores = np.zeros(self._n_docs, dtype=np.float32)
        matched = False
        for tok in tokenize(query):  # keep duplicates — matches the reference
            i = self._vocab.get(tok)
            if i is None:
                continue
            lo, hi = int(self._indptr[i]), int(self._indptr[i + 1])
            if hi > lo:
                scores[self._indices[lo:hi]] += self._data[lo:hi]
                matched = True
        if not matched:
            return []
        nz = np.nonzero(scores)[0]
        if nz.size == 0:
            return []
        # Rank by score desc, ties broken by ascending doc index — matches the reference
        # BM25Index's stable sort over scroll-order docs, so the two agree even at the top-k
        # truncation boundary when scores tie. (lexsort's last key is primary.)
        order = nz[np.lexsort((nz, -scores[nz]))][:k]
        return [(self._keys[int(j)], float(scores[j])) for j in order]


def _main(argv: list[str]) -> int:
    from ingest.config import load_config
    from qdrant_client import QdrantClient

    cmd = argv[0] if argv else "build"
    cfg = load_config()
    if cmd == "info":
        if not FullCorpusBM25.cache_exists():
            print(f"no cache at {DEFAULT_INDEX_DIR}")
            return 1
        idx = FullCorpusBM25.load()
        print(json.dumps(idx.meta, indent=2))
        return 0
    if cmd != "build":
        print(f"usage: python -m eval.bm25_full [build|info]   (got {cmd!r})")
        return 2
    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    t0 = time.perf_counter()
    FullCorpusBM25.build(client, cfg.collection_name, log=lambda m: print(m, flush=True))
    print(f"[bm25_full] total build {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
