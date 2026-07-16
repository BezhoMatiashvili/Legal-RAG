"""v3-to-evaluator bridge preserves adjudicated evidence and policy semantics."""

from dataclasses import replace

import pytest

from eval.evaluate import build_query_relevance
from eval.metrics import Hit, query_score
from eval.v3_dataset import EvidenceEquivalenceGroup, Split, V3Dataset
from eval.v3_eval_adapter import V3CanonicalBodies, adapt_v3_dataset, build_v3_qrels
from ingest.chunking import default_token_counter
from test_v3_dataset import v3_fixture as _v3_fixture


CHUNK_CFG = {"max_tokens": 32, "overlap": 4, "min_tokens": 1}


@pytest.fixture
def adapter_fixture():
    return _v3_fixture.__wrapped__()


def test_adapter_separates_retrieval_qrels_from_clarify_and_abstain(adapter_fixture):
    dataset, _documents = adapter_fixture
    adapted = adapt_v3_dataset(dataset)

    assert [query.id for query in adapted.gold_queries] == [
        "q-lawyer-ka",
        "q-history-en",
    ]
    assert adapted.non_answerable_query_ids == ("q-unsupported", "q-unanswerable")
    assert adapted.expected_outcomes == {
        "q-lawyer-ka": "answer",
        "q-history-en": "answer",
        "q-unsupported": "clarify",
        "q-unanswerable": "abstain",
    }
    assert adapted.dataset_id == dataset.manifest.dataset_id
    assert adapted.corpus_generation == dataset.manifest.corpus_generation


def test_adapter_retains_version_temporal_slice_and_all_required_groups(adapter_fixture):
    dataset, documents = adapter_fixture
    adapted = adapt_v3_dataset(dataset, splits=(Split.DEV,))
    (query,) = adapted.gold_queries

    assert query.id == "q-history-en"
    assert query.as_of == "2019-07-15"
    assert query.split == "dev"
    assert query.risk_level == "medium"
    assert query.partition_family_ids == (
        "lineage-historical-law",
        "lineage-case-2018-42",
    )
    assert {item.version_id for item in query.relevance} == {
        "law-historical@2015",
        "case-2018-42@final",
    }
    assert {item.evidence_group for item in query.relevance} == {
        "historical-rule",
        "applying-case",
    }

    validated, qrels = build_v3_qrels(
        dataset,
        documents,
        chunk_config=CHUNK_CFG,
        count_tokens=default_token_counter,
        splits=(Split.DEV,),
    )
    assert validated.gold_queries == adapted.gold_queries
    assert qrels[query.id]["failure_reason"] is None
    assert set(qrels[query.id]["evidence_groups"]) == {
        "historical-rule",
        "applying-case",
    }
    assert all(len(key) == 4 for key in qrels[query.id]["chunk"])
    assert {key[2] for key in qrels[query.id]["chunk"]} == {
        "law-historical@2015",
        "case-2018-42@final",
    }
    assert all(len(key) == 3 for key in qrels[query.id]["doc"])
    assert {key[2] for key in qrels[query.id]["doc"]} == {
        "law-historical@2015",
        "case-2018-42@final",
    }

    wrong_version = Hit(
        source="matsne",
        document_id="law-historical",
        chunk_index=0,
        score=1.0,
        version_id="law-historical@current",
    )
    wrong_score = query_score(
        query.id,
        query.query_type,
        query.query_language,
        [wrong_version],
        qrels[query.id]["chunk"],
        "chunk",
        evidence_groups=qrels[query.id]["evidence_groups"],
    )
    assert wrong_score.success10 == 0.0
    assert wrong_score.document_identity1 == 0.0


def test_alternative_spans_remain_one_qrel_equivalence_group(adapter_fixture):
    dataset, documents = adapter_fixture
    question = dataset.questions[1]
    historical_group, case_group = question.evidence_groups
    alternative = replace(case_group.alternatives[0], evidence_id="ev-history-alt")
    historical_with_alternative = EvidenceEquivalenceGroup(
        group_id=historical_group.group_id,
        proposition=historical_group.proposition,
        required=True,
        alternatives=(*historical_group.alternatives, alternative),
    )
    changed_question = replace(
        question,
        evidence_groups=(historical_with_alternative, case_group),
    )
    changed = V3Dataset(
        manifest=dataset.manifest,
        questions=(changed_question,),
    )

    adapted = adapt_v3_dataset(changed)
    (query,) = adapted.gold_queries
    assert [item.evidence_group for item in query.relevance].count("historical-rule") == 2
    assert adapted.evidence_groups_by_query[query.id][0] is historical_with_alternative

    groups = build_query_relevance(
        adapted.gold_queries,
        V3CanonicalBodies(documents),
        CHUNK_CFG,
        default_token_counter,
    )[query.id]["evidence_groups"]
    assert set(groups) == {"historical-rule", "applying-case"}
    assert groups["applying-case"] < groups["historical-rule"]


def test_canonical_body_lookup_fails_closed_on_unversioned_ambiguity(adapter_fixture):
    _dataset, documents = adapter_fixture
    historical = documents[1]
    second_version = replace(
        historical,
        version_id="law-historical@2019",
        text="A second immutable version.",
    )
    bodies = V3CanonicalBodies((*documents, second_version))

    assert bodies.body_version(
        historical.source, historical.document_id, historical.version_id
    ) == historical.text
    with pytest.raises(ValueError, match="ambiguous canonical document version"):
        bodies.body(historical.source, historical.document_id)


def test_adapter_rejects_unknown_split(adapter_fixture):
    dataset, _documents = adapter_fixture
    with pytest.raises(ValueError, match="unknown v3 split"):
        adapt_v3_dataset(dataset, splits=("staging",))
