import json

import pytest

from eval.goldset import GoldQuery, Relevance
from eval.v2_candidate_qrels import (
    EXPECTED_GOLDEN_SHA256,
    EXPECTED_HOLDOUT_SHA256,
    EXPECTED_TRANSLATIONS_SHA256,
    _create_private_json,
    _mapping_row,
    _validate_frozen_inputs,
    bind_gold_queries_to_candidate,
)
from eval import goldset


def _query(*, source="matsne", query_id="q1"):
    return GoldQuery(
        id=query_id,
        query="query",
        query_type="keyword",
        query_language="ka",
        source=source,
        document_id="doc",
        gold_source=source,
        gold_document_id="doc",
        relevance=[
            Relevance(
                document_id="doc",
                evidence_quote="bc",
                char_start=1,
                char_end=3,
                grade=2,
            )
        ],
    )


def _candidate(body="abcdef", **overrides):
    return {
        "source": "matsne",
        "document_id": "doc",
        "version_id": "v-candidate",
        "body_markdown": body,
        "source_authority": "primary_official",
        "content_complete": True,
        "admissible": True,
        **overrides,
    }


def test_exact_mapping_requires_one_primary_complete_byte_identical_body():
    query = _query()
    admitted = {("matsne", "doc"): [_candidate()]}
    row = _mapping_row(query, "abcdef", admitted, {})
    assert row["status"] == "mapped"
    assert row["version_id"] == "v-candidate"
    assert row["zero_score_failure"] is False
    assert len(row["candidate_body_sha256"]) == 64

    changed = _mapping_row(
        query,
        "abcdef",
        {("matsne", "doc"): [_candidate("abcdeX")]},
        {},
    )
    assert changed["failure_reason"] == "candidate_body_bytes_changed"
    assert changed["zero_score_failure"] is True

    ambiguous = _mapping_row(
        query,
        "abcdef",
        {("matsne", "doc"): [_candidate(), _candidate(version_id="v2")]},
        {},
    )
    assert ambiguous["failure_reason"] == "ambiguous_candidate_document"

    one_exact_version = _mapping_row(
        query,
        "abcdef",
        {
            ("matsne", "doc"): [
                _candidate(),
                _candidate("historical body", version_id="v-old"),
            ]
        },
        {},
    )
    assert one_exact_version["status"] == "mapped"
    assert one_exact_version["version_id"] == "v-candidate"

    inadmissible = _mapping_row(
        query,
        "abcdef",
        {("matsne", "doc"): [_candidate(admissible=False)]},
        {},
    )
    assert inadmissible["failure_reason"] == "candidate_not_admissible"


def test_missing_and_quarantined_are_explicit_and_never_fuzzy_reanchored():
    query = _query()
    missing = _mapping_row(query, "abcdef", {}, {})
    assert missing["failure_reason"] == "missing_candidate_document"

    quarantined = _mapping_row(
        query,
        "abcdef",
        {},
        {("matsne", "doc"): [{"reason": "summary_only"}]},
    )
    assert quarantined["failure_reason"] == "quarantined_candidate_document"


@pytest.mark.parametrize("source", ["tas", "tbappeal"])
def test_frozen_incomplete_sources_remain_zero_score_failures(source):
    query = _query(source=source)
    admitted = {
        (source, "doc"): [
            {
                **_candidate(),
                "source": source,
            }
        ]
    }
    row = _mapping_row(query, "abcdef", admitted, {})
    assert row["failure_reason"] == "frozen_incomplete_source_label"
    assert row["zero_score_failure"] is True


def test_binding_adds_exact_version_and_preserves_explicit_failures():
    queries = [_query(query_id="mapped"), _query(query_id="failed")]
    artifact = {
        "mappings": [
            {
                "query_id": "mapped",
                "status": "mapped",
                "version_id": "v-candidate",
            },
            {
                "query_id": "failed",
                "status": "failure",
                "failure_reason": "candidate_body_bytes_changed",
            },
        ]
    }
    bound, failures = bind_gold_queries_to_candidate(queries, artifact)
    assert bound[0].gold_version_id == "v-candidate"
    assert bound[0].relevance[0].version_id == "v-candidate"
    assert bound[1] == queries[1]
    assert failures == {"failed": "candidate_body_bytes_changed"}

    with pytest.raises(ValueError, match="order differs"):
        bind_gold_queries_to_candidate(list(reversed(queries)), artifact)


def test_qrel_json_destination_is_create_only(tmp_path):
    path = tmp_path / "qrels.json"
    _create_private_json(path, {"one": 1})
    assert json.loads(path.read_text()) == {"one": 1}
    with pytest.raises(FileExistsError, match="already exists"):
        _create_private_json(path, {"two": 2})
    assert json.loads(path.read_text()) == {"one": 1}


def test_frozen_v2_identity_and_fixed_source_counts_are_exact():
    queries, hashes = _validate_frozen_inputs(
        goldset.DEFAULT_GOLD_V2,
        goldset.EVAL_DIR / "query_translations_v2.json",
        goldset.DEFAULT_HOLDOUT_V2,
    )
    assert len(queries) == 337
    assert hashes == {
        "golden_set_sha256": EXPECTED_GOLDEN_SHA256,
        "translations_sha256": EXPECTED_TRANSLATIONS_SHA256,
        "holdout_sha256": EXPECTED_HOLDOUT_SHA256,
    }
