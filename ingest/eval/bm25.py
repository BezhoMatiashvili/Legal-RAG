"""A small, dependency-free Okapi BM25 index — the mandatory neural-stack sanity floor.

Georgian (Mkhedruli) is caseless and space-separated, so a Unicode ``\\w+`` tokenizer with
``casefold`` (which lowercases the Latin used in English cross-lingual queries) is a fair,
transparent baseline. It ranks the **same chunk texts** the neural modes see, so the
comparison is apples-to-apples.
"""

import math
import re
from collections import Counter
from collections.abc import Hashable, Iterable

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


class BM25Index:
    """Okapi BM25 over a fixed set of documents identified by an opaque id."""

    def __init__(
        self,
        ids: list[Hashable],
        docs_tokens: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if len(ids) != len(docs_tokens):
            raise ValueError("ids and docs_tokens must be the same length")
        self.ids = list(ids)
        self.k1 = k1
        self.b = b
        self.doc_len = [len(t) for t in docs_tokens]
        self.n = len(docs_tokens)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(t) for t in docs_tokens]
        df: Counter = Counter()
        for tokens in self.tf:
            df.update(tokens.keys())
        # Lucene-style non-negative IDF: log(1 + (N - df + 0.5)/(df + 0.5)).
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    @classmethod
    def from_pairs(
        cls, pairs: Iterable[tuple[Hashable, str]], **kwargs
    ) -> "BM25Index":
        ids: list[Hashable] = []
        docs_tokens: list[list[str]] = []
        for doc_id, text in pairs:
            ids.append(doc_id)
            docs_tokens.append(tokenize(text or ""))
        return cls(ids, docs_tokens, **kwargs)

    def _score(self, q_terms: list[str], i: int) -> float:
        if not self.avgdl:
            return 0.0
        tf = self.tf[i]
        denom_norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            s += self.idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_norm)
        return s

    def search(self, query: str, k: int) -> list[tuple[Hashable, float]]:
        """Return up to ``k`` ``(id, score)`` with score > 0, best first."""
        q_terms = tokenize(query)
        scored = [
            (self.ids[i], sc)
            for i in range(self.n)
            if (sc := self._score(q_terms, i)) > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
