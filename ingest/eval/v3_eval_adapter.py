"""Lossless bridge from the adjudicated v3 schema to retrieval-evaluator inputs.

The legacy evaluator consumes :class:`GoldQuery` rows whose relevance annotations are
individual spans.  v3 instead makes an evidence *equivalence group* the scoring unit.  This
adapter emits one ``Relevance`` annotation per alternative span while retaining the same
``evidence_group`` id on every alternative; ``build_query_relevance`` therefore constructs
one qrel group, not one requirement per span.

Only adjudicated ``answer`` questions enter retrieval qrels.  ``clarify`` and ``abstain``
questions remain first-class policy cases so they can be evaluated by the selective-answer
gate without being mislabeled retrieval failures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .goldset import GoldQuery, Relevance
from .v3_dataset import (
    CanonicalDocument,
    EvidenceEquivalenceGroup,
    ExpectedOutcome,
    Split,
    V3Dataset,
    adjudicated_judgment,
    validate_dataset,
)


@dataclass(frozen=True)
class V3PolicyCase:
    """Adjudicated outcome and slice metadata for every selected v3 question."""

    id: str
    query: str
    query_language: str
    language_code: str
    expected_outcome: str
    answerable: bool
    split: str
    risk_level: str
    tags: tuple[str, ...]
    as_of: str | None
    identifiers: tuple[str, ...]
    partition_family_ids: tuple[str, ...]
    evidence_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class AdaptedV3Gold:
    """v3 retrieval rows plus the policy cases that must not be forced into qrels."""

    dataset_id: str
    corpus_generation: str
    gold_queries: tuple[GoldQuery, ...]
    policy_cases: tuple[V3PolicyCase, ...]
    # Original semantic groups are retained for audit/export; the GoldQuery annotations
    # are merely their chunk-mapping representation.
    evidence_groups_by_query: dict[str, tuple[EvidenceEquivalenceGroup, ...]]

    @property
    def expected_outcomes(self) -> dict[str, str]:
        return {case.id: case.expected_outcome for case in self.policy_cases}

    @property
    def non_answerable_query_ids(self) -> tuple[str, ...]:
        return tuple(
            case.id
            for case in self.policy_cases
            if case.expected_outcome != ExpectedOutcome.ANSWER.value
        )


class V3CanonicalBodies:
    """Exact canonical-body lookup suitable for v3 qrel construction and re-grounding."""

    def __init__(self, documents: Sequence[CanonicalDocument]):
        self._by_version: dict[tuple[str, str, str], str] = {}
        self._versions_by_document: dict[tuple[str, str], set[str]] = {}
        for document in documents:
            if document.key in self._by_version:
                raise ValueError(f"duplicate canonical document version: {document.key!r}")
            self._by_version[document.key] = document.text
            self._versions_by_document.setdefault(
                (document.source, document.document_id), set()
            ).add(document.version_id)

    def body_version(self, source: str, document_id: str, version_id: str) -> str:
        try:
            return self._by_version[(source, document_id, version_id)]
        except KeyError as exc:
            raise KeyError(
                f"canonical version not found: {source}:{document_id}@{version_id}"
            ) from exc

    def body(self, source: str, document_id: str) -> str:
        """Legacy lookup, allowed only when document identity resolves to one version."""

        versions = self._versions_by_document.get((source, document_id), set())
        if not versions:
            raise KeyError(f"canonical document not found: {source}:{document_id}")
        if len(versions) != 1:
            raise ValueError(
                f"ambiguous canonical document version: {source}:{document_id} has "
                f"{sorted(versions)}"
            )
        version_id = next(iter(versions))
        return self.body_version(source, document_id, version_id)


def _selected_splits(splits: Iterable[Split | str] | None) -> set[str] | None:
    if splits is None:
        return None
    selected = {
        split.value if isinstance(split, Split) else str(split)
        for split in splits
    }
    invalid = selected - {split.value for split in Split}
    if invalid:
        raise ValueError(f"unknown v3 split(s): {sorted(invalid)}")
    return selected


def _lineage_clusters(questions: Sequence[object]) -> dict[str, str]:
    """Build stable connected-component clusters for linked document/version families."""

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Lexicographic roots make the component deterministic regardless of input order.
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    for question in questions:
        families = tuple(dict.fromkeys(question.partition_family_ids))
        for family in families:
            find(family)
        for family in families[1:]:
            union(families[0], family)

    components: dict[str, list[str]] = {}
    for family in sorted(parent):
        components.setdefault(find(family), []).append(family)

    component_ids: dict[str, str] = {}
    for root, families in components.items():
        encoded = json.dumps(families, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        component_ids[root] = f"v3-family:{digest}"

    out: dict[str, str] = {}
    for question in questions:
        if question.partition_family_ids:
            out[question.question_id] = component_ids[find(question.partition_family_ids[0])]
        else:
            out[question.question_id] = f"v3-query:{question.question_id}"
    return out


def _query_type(tags: Sequence[object]) -> str:
    for tag in tags:
        value = tag.value
        if not value.startswith("risk_"):
            return value
    return "v3"


def adapt_v3_dataset(
    dataset: V3Dataset,
    *,
    splits: Iterable[Split | str] | None = None,
) -> AdaptedV3Gold:
    """Adapt adjudicated v3 records to GoldQuery rows without losing group semantics.

    Callers should run ``validate_dataset`` before this function.  The adapter still checks
    the invariants on which scoring depends and fails closed instead of guessing when a
    judgment references a missing, optional, or empty evidence group.
    """

    selected_splits = _selected_splits(splits)
    questions = tuple(
        question
        for question in dataset.questions
        if selected_splits is None or question.split.value in selected_splits
    )
    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("v3 adapter requires unique question ids")

    clusters = _lineage_clusters(questions)
    gold_queries: list[GoldQuery] = []
    policy_cases: list[V3PolicyCase] = []
    retained_groups: dict[str, tuple[EvidenceEquivalenceGroup, ...]] = {}

    for question in questions:
        judgment = adjudicated_judgment(question)
        selected_group_ids = tuple(judgment.evidence_group_ids)
        policy_cases.append(
            V3PolicyCase(
                id=question.question_id,
                query=question.question,
                query_language=question.language.value,
                language_code=question.language_code,
                expected_outcome=judgment.expected_outcome.value,
                answerable=judgment.answerable,
                split=question.split.value,
                risk_level=question.risk_level.value,
                tags=tuple(tag.value for tag in question.tags),
                as_of=question.as_of,
                identifiers=question.identifiers,
                partition_family_ids=question.partition_family_ids,
                evidence_group_ids=selected_group_ids,
            )
        )
        if judgment.expected_outcome is not ExpectedOutcome.ANSWER:
            continue
        if not judgment.answerable:
            raise ValueError(
                f"{question.question_id}: answer outcome must be marked answerable"
            )

        group_index = {group.group_id: group for group in question.evidence_groups}
        missing = sorted(set(selected_group_ids) - set(group_index))
        if missing:
            raise ValueError(
                f"{question.question_id}: judgment references missing evidence groups {missing}"
            )
        selected_groups = tuple(group_index[group_id] for group_id in selected_group_ids)
        if not selected_groups:
            raise ValueError(
                f"{question.question_id}: answer outcome has no required evidence groups"
            )
        invalid = [
            group.group_id
            for group in selected_groups
            if not group.required or not group.alternatives
        ]
        if invalid:
            raise ValueError(
                f"{question.question_id}: answer judgment selected non-required or empty "
                f"evidence groups {invalid}"
            )

        relevance = [
            Relevance(
                document_id=span.document_id,
                evidence_quote=span.quote,
                char_start=span.char_start,
                char_end=span.char_end,
                grade=2,
                evidence_group=group.group_id,
                required=True,
                source=span.source,
                version_id=span.version_id,
                lineage_family_id=span.lineage_family_id,
                evidence_id=span.evidence_id,
                article_id=span.article_id,
            )
            for group in selected_groups
            for span in group.alternatives
        ]
        primary = selected_groups[0].alternatives[0]
        retained_groups[question.question_id] = selected_groups
        gold_queries.append(
            GoldQuery(
                id=question.question_id,
                query=question.question,
                query_type=_query_type(question.tags),
                query_language=question.language.value,
                source=primary.source,
                document_id=primary.document_id,
                gold_source=primary.source,
                gold_document_id=primary.document_id,
                relevance=relevance,
                answer=question.reference_response,
                doc_title="; ".join(group.proposition for group in selected_groups),
                cluster_id=clusters[question.question_id],
                as_of=question.as_of,
                split=question.split.value,
                risk_level=question.risk_level.value,
                tags=tuple(tag.value for tag in question.tags),
                expected_outcome=judgment.expected_outcome.value,
                partition_family_ids=question.partition_family_ids,
                dataset_id=dataset.manifest.dataset_id,
                corpus_generation=dataset.manifest.corpus_generation,
                gold_version_id=primary.version_id,
            )
        )

    return AdaptedV3Gold(
        dataset_id=dataset.manifest.dataset_id,
        corpus_generation=dataset.manifest.corpus_generation,
        gold_queries=tuple(gold_queries),
        policy_cases=tuple(policy_cases),
        evidence_groups_by_query=retained_groups,
    )


def build_v3_qrels(
    dataset: V3Dataset,
    documents: Sequence[CanonicalDocument],
    *,
    chunk_config: Mapping[str, int],
    count_tokens: Callable[[str], int],
    splits: Iterable[Split | str] | None = None,
) -> tuple[AdaptedV3Gold, dict[str, dict[str, object]]]:
    """Validate v3 and construct evaluator qrels against exact canonical versions.

    This is the fail-closed production entry point.  ``adapt_v3_dataset`` remains useful
    on its own for policy evaluation and unit-level schema work, while this helper guarantees
    corpus anchoring before any retrieval score is computed.
    """

    validate_dataset(dataset, documents)
    adapted = adapt_v3_dataset(dataset, splits=splits)
    # Local import avoids coupling the standalone v3 schema module to the CLI harness.
    from .evaluate import build_query_relevance

    qrels = build_query_relevance(
        adapted.gold_queries,
        V3CanonicalBodies(documents),
        dict(chunk_config),
        count_tokens,
    )
    return adapted, qrels
