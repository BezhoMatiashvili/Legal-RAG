"""Retrieval backends the harness drives, decoupled from the ML/Qdrant stack.

Every mode returns a ranked ``list[Hit]`` plus per-stage latencies (embed / search /
rerank, seconds). Two implementations:

  * :class:`FakeBackend` — deterministic, dependency-free (lexical vectors derived from the
    chunk text). Powers the unit tests and the STOP-gate end-to-end run with no torch/GPU.
  * :class:`QdrantBackend` — the real path over ``BGEM3Embedder`` + Qdrant + ``BGEReranker``,
    for the full-corpus baseline once the index exists (Part 3).
"""

import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .bm25 import BM25Index, tokenize
from .metrics import Hit

MODES = ("bm25", "dense", "sparse", "hybrid", "rerank", "routed")
_RRF_K = 60


@dataclass(frozen=True)
class ChunkRecord:
    source: str
    document_id: str
    chunk_index: int
    text: str


def _rrf_fuse(*rankings: Sequence) -> list:
    """Reciprocal-rank fusion of several ranked id lists → fused id list (best first)."""
    score: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            score[key] = score.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(score, key=lambda k: score[k], reverse=True)


class FakeBackend:
    """Deterministic lexical backend over an in-memory chunk corpus (tests + demo).

    Dense = idf-weighted, L2-normalised bag-of-words (cosine); sparse = raw term-frequency
    overlap; hybrid = RRF(dense, sparse); rerank = hybrid candidates re-scored by query↔text
    token Jaccard. Distinct enough that the modes produce genuinely different rankings.
    """

    def __init__(self, records: Sequence[ChunkRecord]):
        self.records = list(records)
        self.keys = [(r.source, r.document_id, r.chunk_index) for r in self.records]
        toks = [tokenize(r.text) for r in self.records]
        n = len(toks) or 1
        df: Counter = Counter()
        for t in toks:
            df.update(set(t))
        self._idf = {term: math.log(1.0 + n / (df[term])) for term in df}
        self._tf = [Counter(t) for t in toks]
        self._dense = [self._to_dense(tf) for tf in self._tf]  # normalised idf-weighted dicts
        self._tokset = [set(t) for t in toks]
        self.bm25 = BM25Index.from_pairs(
            [(k, r.text) for k, r in zip(self.keys, self.records)]
        )

    def _to_dense(self, tf: Counter) -> dict:
        vec = {term: tf[term] * self._idf.get(term, 0.0) for term in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def _hit(self, i: int, score: float) -> Hit:
        r = self.records[i]
        return Hit(r.source, r.document_id, r.chunk_index, float(score))

    def _dense_rank(self, q_dense: dict) -> list[int]:
        scored = [
            (i, sum(q_dense[t] * d.get(t, 0.0) for t in q_dense))
            for i, d in enumerate(self._dense)
        ]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored]

    def _sparse_rank(self, q_tf: Counter) -> list[int]:
        scored = []
        for i, tf in enumerate(self._tf):
            s = sum(q_tf[t] * tf.get(t, 0) for t in q_tf)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored]

    def search(self, query: str, mode: str, k: int) -> tuple[list[Hit], dict[str, float]]:
        lat = {"embed": 0.0, "search": 0.0, "rerank": 0.0}

        if mode == "bm25":
            t0 = time.perf_counter()
            ranked = self.bm25.search(query, k)  # [(key, score)]
            lat["search"] = time.perf_counter() - t0
            idx = {key: i for i, key in enumerate(self.keys)}
            return [self._hit(idx[key], sc) for key, sc in ranked], lat

        t0 = time.perf_counter()
        q_toks = tokenize(query)
        q_tf = Counter(q_toks)
        q_dense = self._to_dense(q_tf)
        lat["embed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if mode == "dense":
            order = self._dense_rank(q_dense)[:k]
        elif mode == "sparse":
            order = self._sparse_rank(q_tf)[:k]
        elif mode in ("hybrid", "rerank"):
            fused = _rrf_fuse(self._dense_rank(q_dense), self._sparse_rank(q_tf))
            order = fused[: max(k * 5, 50)] if mode == "rerank" else fused[:k]
        elif mode == "routed":
            # Mirror production routing: an English (cross-lingual) query drops the sparse
            # branch (BGE-M3 learned-sparse is lexical → EN tokens are noise); KA uses hybrid.
            from ingest.search import detect_language

            if detect_language(query) == "en":
                order = self._dense_rank(q_dense)[:k]
            else:
                order = _rrf_fuse(self._dense_rank(q_dense), self._sparse_rank(q_tf))[:k]
        else:
            raise ValueError(f"unknown mode: {mode}")
        lat["search"] = time.perf_counter() - t0

        if mode == "rerank":
            t0 = time.perf_counter()
            qset = set(q_toks)
            rescored = sorted(
                order,
                key=lambda i: len(qset & self._tokset[i]) / (len(qset | self._tokset[i]) or 1),
                reverse=True,
            )[:k]
            lat["rerank"] = time.perf_counter() - t0
            order = rescored

        # score field is informational for fake modes; ranking is what matters
        return [self._hit(i, 1.0 / (rank + 1)) for rank, i in enumerate(order)], lat


class QdrantBackend:
    """Real retrieval over BGE-M3 + Qdrant + the cross-encoder reranker (Part 3 onward).

    BM25 is built lazily by scrolling the collection's chunk texts, so it ranks the exact
    same chunk set the neural modes see.
    """

    def __init__(self, cfg, client, embedder, reranker=None, rerank_candidates: int | None = None,
                 *, fusion: str = "rrf", prefetch_limit: int | None = None,
                 hnsw_ef: int | None = None, rescore: bool | None = None,
                 sparse_weight: float | None = None, max_per_doc: int | None = None,
                 mmr_lambda: float | None = None, translations: dict[str, str] | None = None,
                 citation_route: str | None = None):
        self.cfg = cfg
        self.client = client
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates or cfg.rerank_candidates
        # CPU-tuning knobs (all measured through the harness; none re-embed):
        self.fusion = fusion                 # "rrf" | "dbsf" (score-based) native fusion
        self.prefetch_limit = prefetch_limit  # override the recall pool depth
        self.hnsw_ef = hnsw_ef               # HNSW search ef (recall/latency knee)
        self.rescore = rescore               # int8 quantization rescore on/off
        self.sparse_weight = sparse_weight   # manual weighted dense/sparse fusion (RRF is unweighted)
        self.max_per_doc = max_per_doc       # diversity: cap chunks per document_id
        self.mmr_lambda = mmr_lambda         # diversity: MMR trade-off (needs candidate vectors)
        # I2 cross-lingual knob: {original_query: authored KA legal-register translation}.
        # Substituted before encoding, so routing/sparse/rerank all see the Georgian text;
        # queries without an entry (all KA ones) are untouched by construction.
        self.translations = translations
        self.citation_route = citation_route  # I1: pin exact citation hits ("ids" | "full")
        self._bm25 = None  # BM25Index | FullCorpusBM25, built lazily in _ensure_bm25
        self._filter = None

    @property
    def _diversity_on(self) -> bool:
        return self.max_per_doc is not None or self.mmr_lambda is not None

    def _candidate(self, mode: str, k: int) -> int:
        """Depth of the recall pool to fetch before rerank/diversity/truncation."""
        if self.prefetch_limit is not None:
            return self.prefetch_limit
        if mode == "rerank":
            # Faithful rerank depth: rc=10 → pool 10 (the old max(.,k*5,50) floor masked 10/30).
            return max(self.rerank_candidates, k)
        base = max(k * 5, 50)
        if self._diversity_on:
            base = max(base, self.rerank_candidates, 100)  # diversity needs material to re-rank
        return base

    def _search_params(self, models):
        if self.hnsw_ef is None and self.rescore is None:
            return None
        quant = None
        if self.rescore is not None:
            quant = models.QuantizationSearchParams(rescore=bool(self.rescore))
        return models.SearchParams(hnsw_ef=self.hnsw_ef, quantization=quant)

    def _fusion_query(self, models):
        f = models.Fusion.DBSF if self.fusion == "dbsf" else models.Fusion.RRF
        return models.FusionQuery(fusion=f)

    def _manual_fusion(self, models, emb, candidate, sp, with_vectors):
        """Weighted dense/sparse fusion by min-max-normalised scores (RRF has no weight knob).

        ``score = (1-w)·dense_norm + w·sparse_norm``; w=0 is dense-only (== routed for EN),
        w=1 is sparse-only. Used for the cross-lingual sparse-weight sweep."""
        coll = self.cfg.collection_name
        w = self.sparse_weight
        dres = self.client.query_points(collection_name=coll, query=emb.dense, using="dense",
                                        limit=candidate, with_payload=True,
                                        with_vectors=with_vectors, search_params=sp)
        sres = self.client.query_points(
            collection_name=coll,
            query=models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
            using="sparse", limit=candidate, with_payload=True,
            with_vectors=with_vectors, search_params=sp)

        def norm(points):
            if not points:
                return {}
            ss = [p.score for p in points]
            lo, hi = min(ss), max(ss)
            rng = (hi - lo) or 1.0
            return {p.id: (p, (p.score - lo) / rng) for p in points}

        dn, sn = norm(dres.points), norm(sres.points)
        out = []
        for pid in set(dn) | set(sn):
            dp, spt = dn.get(pid), sn.get(pid)
            pt = (dp or spt)[0]
            pt.score = (1 - w) * (dp[1] if dp else 0.0) + w * (spt[1] if spt else 0.0)
            out.append(pt)
        out.sort(key=lambda p: p.score, reverse=True)
        return out

    def _points_to_hits(self, points) -> list[Hit]:
        hits = []
        for p in points:
            pl = p.payload or {}
            hits.append(
                Hit(pl.get("source"), pl.get("document_id"), pl.get("chunk_index", -1), float(p.score))
            )
        return hits

    def _ensure_bm25(self):
        """Return the BM25 index for the collection.

        Prefers the prebuilt, disk-backed full-corpus index (``eval/.bm25_full/``, built once
        via ``python -m eval.bm25_full build``) — mandatory at the 2.45M-chunk scale, where
        scrolling the whole collection into an in-memory ``BM25Index`` is ~17 GB and minutes
        per query. Falls back to the in-memory scroll only for small collections (the guard
        below refuses the impractical scroll on a large one)."""
        if self._bm25 is None:
            from .bm25_full import FullCorpusBM25

            if FullCorpusBM25.cache_exists():
                self._bm25 = FullCorpusBM25.load(expect_collection=self.cfg.collection_name)
                return self._bm25
            n = self.client.count(self.cfg.collection_name, exact=False).count
            if n > 200_000:
                raise RuntimeError(
                    f"BM25 over {n:,} chunks needs the prebuilt full-corpus index; run "
                    "`python -m eval.bm25_full build` first (writes eval/.bm25_full/)."
                )
            pairs = []
            offset = None
            while True:
                batch, offset = self.client.scroll(
                    collection_name=self.cfg.collection_name,
                    with_payload=True, with_vectors=False, limit=512, offset=offset,
                )
                for pt in batch:
                    pl = pt.payload or {}
                    key = (pl.get("source"), pl.get("document_id"), pl.get("chunk_index", -1))
                    pairs.append((key, pl.get("text") or ""))
                if offset is None:
                    break
            self._bm25 = BM25Index.from_pairs(pairs)
        return self._bm25

    def search(self, query: str, mode: str, k: int) -> tuple[list[Hit], dict[str, float]]:
        from qdrant_client import models

        if self.translations:
            query = self.translations.get(query, query)

        lat = {"embed": 0.0, "search": 0.0, "rerank": 0.0}

        if mode == "bm25":
            bm25 = self._ensure_bm25()
            t0 = time.perf_counter()
            ranked = bm25.search(query, k)
            lat["search"] = time.perf_counter() - t0
            return [Hit(s, d, c, sc) for (s, d, c), sc in ranked], lat

        t0 = time.perf_counter()
        emb = self.embedder.encode_query(query)
        lat["embed"] = time.perf_counter() - t0

        pinned = []
        if self.citation_route:
            from ingest.citations import citation_lookup, extract_citation

            t0 = time.perf_counter()
            ref = extract_citation(query, mode=self.citation_route)
            if ref is not None:
                pinned = citation_lookup(self.client, self.cfg.collection_name, emb.dense, ref)
            lat["search"] += time.perf_counter() - t0
        coll = self.cfg.collection_name
        sp = self._search_params(models)
        want_vec = self.mmr_lambda is not None
        deep = mode == "rerank" or self._diversity_on  # fetch a pool, not just top_k

        t0 = time.perf_counter()
        if mode == "dense":
            res = self.client.query_points(
                collection_name=coll, query=emb.dense, using="dense",
                limit=self._candidate(mode, k) if deep else k,
                with_payload=True, with_vectors=want_vec, search_params=sp,
            )
            points = res.points
        elif mode == "sparse":
            res = self.client.query_points(
                collection_name=coll,
                query=models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
                using="sparse", limit=self._candidate(mode, k) if deep else k,
                with_payload=True, with_vectors=want_vec, search_params=sp,
            )
            points = res.points
        else:  # hybrid / rerank / routed
            use_sparse = bool(emb.sparse.indices)
            if mode == "routed":
                # Mirror production routing: drop sparse for an English (cross-lingual) query.
                from ingest.search import detect_language

                use_sparse = use_sparse and detect_language(query) != "en"
            candidate = self._candidate(mode, k)
            if self.sparse_weight is not None and use_sparse:
                points = self._manual_fusion(models, emb, candidate, sp, want_vec)
            else:
                prefetch = [models.Prefetch(query=emb.dense, using="dense", limit=candidate, params=sp)]
                if use_sparse:
                    prefetch.append(models.Prefetch(
                        query=models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
                        using="sparse", limit=candidate, params=sp))
                fetch_n = candidate if deep else k
                res = self.client.query_points(
                    collection_name=coll, prefetch=prefetch, query=self._fusion_query(models),
                    limit=fetch_n, with_payload=True, with_vectors=want_vec,
                )
                points = res.points
        lat["search"] += time.perf_counter() - t0  # += : citation lookup time added above

        if mode == "rerank" and self.reranker is not None and points:
            from ingest.search import rerank_points

            t0 = time.perf_counter()
            # Rerank the whole pool; final top_k is chosen by diversity (or [:k]) below.
            points = rerank_points(self.reranker, query, points, top_k=len(points), min_score=None)
            lat["rerank"] = time.perf_counter() - t0

        if self._diversity_on:
            from ingest.search import diversify

            points = diversify(points, top_k=k, max_per_doc=self.max_per_doc,
                               mmr_lambda=self.mmr_lambda, query_vec=emb.dense)
        else:
            points = points[:k]

        if pinned:
            from ingest.citations import pin_points

            points = pin_points(pinned, points, k)

        return self._points_to_hits(points), lat
