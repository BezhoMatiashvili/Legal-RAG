"""I5: the --golden-set flag's config_hash behavior.

v1 must contribute NOTHING to config_hash (every historical experiments.jsonl row keeps
its hash byte-stable); any other version folds in as a knob so v2 runs land as distinct,
forever-comparable config rows.
"""

from eval import explog
from eval.evaluate import eval_set_knob

BASE = {"mode": "hybrid", "relevance": "chunk", "top_k": 10, "tokenizer": "bge",
        "max_tokens": 512, "overlap": 80, "min_tokens": 64, "rerank_candidates": 80}


def test_v1_contributes_nothing():
    assert eval_set_knob("v1") == {}
    assert explog.config_hash(BASE) == explog.config_hash({**BASE, **eval_set_knob("v1")})


def test_v2_folds_in_and_changes_the_hash():
    assert eval_set_knob("v2") == {"eval_set": "v2"}
    assert explog.config_hash(BASE) != explog.config_hash({**BASE, **eval_set_knob("v2")})


def test_distinct_versions_hash_distinctly():
    h = {v: explog.config_hash({**BASE, **eval_set_knob(v)}) for v in ("v1", "v2", "v3")}
    assert len(set(h.values())) == 3
