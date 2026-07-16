"""Hermetic contract tests for the accuracy-first presentation demo builder.

The tests in this module must never contact Qdrant, load a model, or access the
network.  All runtime behavior is exercised with injected, in-memory retrieval
fixtures and temporary presentation directories.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_presentation_demo.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_presentation_demo_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


EXPECTED_QUESTION_IDS = (
    "q138",
    "q169",
    "q208",
    "q211",
    "q229",
    "q257",
    "q271",
    "q276",
    "q283",
    "q285",
    "q291",
    "q293",
    "q312",
    "q329",
    "q343",
    "q121",
    "q191",
    "q003",
    "q024",
    "demo-broad-001",
)

EXPECTED_BUNDLE_FILES = {
    "README.md",
    "questions.jsonl",
    "questions.sha256",
    "run-1.json",
    "run-2.json",
    "scorecard.json",
    "scorecard.md",
    "demo-script.md",
    "slides-outline.md",
    "talk-track.md",
    "reviewer-checklist.md",
    "limitations.md",
}


def _write_frozen_fixture(tmp_path: Path) -> tuple[Path, tuple[Path, ...], Path]:
    """Write the complete fixed selection into a tiny, local-only corpus."""

    golden = tmp_path / "golden.jsonl"
    snapshot = tmp_path / "snapshot-v1"
    docs = snapshot / "docs"
    docs.mkdir(parents=True)
    by_source: dict[str, list[dict]] = {}
    golden_rows = []
    for question_id in EXPECTED_QUESTION_IDS[:-1]:
        source = (
            "tbappeal"
            if question_id == "q003"
            else "tas"
            if question_id == "q024"
            else "matsne"
        )
        document_id = f"target-{question_id}"
        number = f"N-{question_id}"
        quote = (
            "PRIVATE_Q024_QUOTE"
            if question_id == "q024"
            else "ნედლი\u009dუნიკოდი\u00a0q329"
            if question_id == "q329"
            else f"ზუსტი ციტატა {question_id}"
        )
        body = f"შესავალი {quote} დასასრული"
        start = body.index(quote)
        end = start + len(quote)
        query = f"დოკუმენტი №{number} რას ადგენს?"
        golden_rows.append(
            {
                "id": question_id,
                "query": query,
                "query_language": "ka",
                "gold": {"source": source, "document_id": document_id},
                "relevance": [
                    {
                        "document_id": document_id,
                        "evidence_quote": quote,
                        "char_start": start,
                        "char_end": end,
                        "grade": 2,
                    }
                ],
                "doc_title": f"სატესტო დოკუმენტი {question_id}",
            }
        )
        by_source.setdefault(source, []).append(
            {
                "snapshot_version": "fixture-v1",
                "source": source,
                "document_id": document_id,
                "document_number": number,
                "registration_code": None,
                "title": f"სატესტო დოკუმენტი {question_id}",
                "date": "2026-01-01",
                "status": (
                    "repealed"
                    if question_id in {"q229", "q276", "q191"}
                    else "in_force"
                ),
                "source_url": f"https://official.example/{source}/{document_id}",
                "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "body_markdown": body,
                "parties": ["PRIVATE_PARTY"] if question_id == "q024" else None,
            }
        )
    golden.write_text(
        "# tiny hermetic presentation fixture\n"
        + "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in golden_rows
        ),
        encoding="utf-8",
    )
    for source, records in by_source.items():
        (docs / f"{source}.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "sources": {
                    "tbappeal": {
                        "clean": 0,
                        "quarantined": {"incomplete_content": 1},
                    },
                    "tas": {
                        "clean": 0,
                        "quarantined": {"missing_source_attestation": 1},
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return golden, (snapshot,), preflight


class _FixtureRetriever:
    def __init__(self, rows, *, fail_question_id: str | None = None):
        self.rows = {row["question_id"]: row for row in rows}
        self.fail_question_id = fail_question_id
        self.requests = []

    def observe_collection(self):
        return {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 20,
            "indexed_vectors_count": 20,
            "segments_count": 1,
            "vector_names": ["dense"],
            "sparse_vector_names": ["bm25"],
        }

    def __call__(self, request):
        copied = json.loads(json.dumps(request, ensure_ascii=False))
        self.requests.append(copied)
        assert set(copied) == {"question_id", "category", "lookup_selector"}
        question = self.rows[copied["question_id"]]
        assert question["expected_document_id"] not in json.dumps(
            copied, ensure_ascii=False
        )
        assert copied["lookup_selector"] == question["lookup_selector"]
        if copied["question_id"] == self.fail_question_id:
            raise RuntimeError("retained fixture failure")
        if copied["category"] == "ambiguous_expected_clarification":
            return {
                "ordered_result_ids": ["matsne:ambiguous-a", "matsne:ambiguous-b"],
                "ordered_point_ids": ["point-a", "point-b"],
                "documents": {},
                "route_decision": "legacy_read_only_exact_selector_scroll",
                "truncated": True,
            }
        result_id = f"{question['expected_source']}:{question['expected_document_id']}"
        selector = question["lookup_selector"]
        return {
            "ordered_result_ids": [result_id],
            "ordered_point_ids": [f"point-{copied['question_id']}"],
            "documents": {
                result_id: {
                    "source": question["expected_source"],
                    "document_id": question["expected_document_id"],
                    selector["field"]: selector["value"],
                    "official_url": (
                        question["official_url"]
                    ),
                    "status": question.get("frozen_status"),
                    "content_hash": "a" * 64,
                }
            },
            "route_decision": "legacy_read_only_exact_selector_scroll",
            "truncated": False,
        }


def _synthetic_contract_result():
    return {
        "track": "synthetic_contract_fixture",
        "answer": None,
        "model_used": False,
        "fixture_is_legal_evidence": False,
        "checks": [{"name": "tamper_rejected", "passed": True}],
        "passed": True,
    }


def _clock(step: float):
    current = -step

    def tick():
        nonlocal current
        current += step
        return current

    return tick


def _write_bundle_inputs(tmp_path: Path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps(
            [
                {
                    "source": row["expected_source"],
                    "document_id": row["expected_document_id"],
                }
                for row in rows
                if row["question_id"] != builder.BROAD_ID
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snapshot_manifest = snapshot_roots[0] / "manifest.json"
    snapshot_manifest.write_text(
        json.dumps(
            {
                "snapshot_version": "fixture-v1",
                "config_hash": "f" * 64,
                "document_count": 19,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "golden_set": golden,
        "holdout": holdout,
        "snapshot_roots": snapshot_roots,
        "snapshot_manifests": (snapshot_manifest,),
        "preflight_manifest": preflight,
        "rows": rows,
    }


def test_predeclared_selection_order_and_bundle_inventory_are_frozen():
    assert tuple(builder.QUESTION_IDS) == EXPECTED_QUESTION_IDS
    assert tuple(builder.EXACT_IDS) == EXPECTED_QUESTION_IDS[:15]
    assert tuple(builder.AMBIGUOUS_IDS) == EXPECTED_QUESTION_IDS[15:17]
    assert tuple(builder.INCOMPLETE_IDS) == EXPECTED_QUESTION_IDS[17:19]
    assert tuple(builder.DISPLAY_IDS) == ("q169", "q291", "q329")
    assert set(builder.REQUIRED_BUNDLE_FILES) == EXPECTED_BUNDLE_FILES
    assert len(builder.REQUIRED_BUNDLE_FILES) == 12
    assert builder.LEGACY_NOT_AVAILABLE == "not_available_in_legacy_corpus"


def test_canonical_json_bytes_are_unicode_preserving_and_order_independent():
    left = {"question": "რა წერია კანონში?", "id": "q138"}
    right = {"id": "q138", "question": "რა წერია კანონში?"}

    encoded = builder.canonical_json_bytes(left)

    assert encoded == builder.canonical_json_bytes(right)
    assert "რა წერია კანონში?".encode("utf-8") in encoded


def test_frozen_loader_predeclares_selection_without_lookup_answer_leakage(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)

    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)

    builder.validate_question_set(rows)
    assert tuple(row["question_id"] for row in rows) == EXPECTED_QUESTION_IDS
    assert len(builder.sha256_canonical_jsonl(rows)) == 64
    for row in rows:
        selector = row["lookup_selector"]
        if row["question_id"] in (*builder.INCOMPLETE_IDS, builder.BROAD_ID):
            assert selector is None
            continue
        assert set(selector) == {"source", "field", "value"}
        assert selector["field"] in {"document_number", "registration_code"}
        assert selector["value"] != row["expected_document_id"]

    with pytest.raises(ValueError, match="order/identity"):
        builder.validate_question_set([*rows[1:], rows[0]])


def test_frozen_public_rows_enforce_quote_privacy_status_and_raw_unicode(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    indexed = {row["question_id"]: row for row in rows}

    quote_bearing = {
        row["question_id"]
        for row in rows
        if "quotation" in row.get("frozen_snapshot_audit", {})
    }
    assert quote_bearing == {"q169", "q291", "q329"}

    private_row = builder.canonical_json_bytes(indexed["q024"])
    assert b"PRIVATE_Q024_QUOTE" not in private_row
    assert b"PRIVATE_PARTY" not in private_row
    assert b"body_markdown" not in private_row
    assert b"parties" not in private_row

    assert indexed["q229"]["frozen_status"] == "repealed"
    assert indexed["q276"]["frozen_status"] == "repealed"
    assert indexed["q191"]["frozen_status"] == "repealed"

    raw_q329 = builder.canonical_json_bytes(indexed["q329"])
    assert "\u009d".encode("utf-8") in raw_q329
    assert "\u00a0".encode("utf-8") in raw_q329
    assert b"\\u009d" not in raw_q329
    assert b"\\u00a0" not in raw_q329


def test_frozen_loader_rejects_quote_offset_or_document_hash_tampering(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    golden_rows = [
        json.loads(line)
        for line in golden.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    golden_rows[0]["relevance"][0]["evidence_quote"] += " შეცვლილია"
    golden.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in golden_rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="offsets do not reconstruct"):
        builder.load_frozen_questions(golden, snapshot_roots, preflight)

    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path / "hash")
    matsne = snapshot_roots[0] / "docs" / "matsne.jsonl"
    documents = [json.loads(line) for line in matsne.read_text(encoding="utf-8").splitlines()]
    documents[0]["content_hash"] = "0" * 64
    matsne.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in documents
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        builder.load_frozen_questions(golden, snapshot_roots, preflight)


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1:6333",
        "http://[::1]:6333",
        "http://localhost:6333/",
    ),
)
def test_loopback_validator_accepts_only_explicit_local_http(url):
    assert builder.validate_loopback_url(url).startswith("http://")


@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:6333",
        "http://qdrant.example:6333",
        "http://127.0.0.1",
        "http://user:secret@127.0.0.1:6333",
        "http://127.0.0.1:6333/collections/legal",
        "http://127.0.0.1:6333?write=true",
    ),
)
def test_loopback_validator_rejects_external_or_ambiguous_destinations(url):
    with pytest.raises(builder.DemoSafetyError):
        builder.validate_loopback_url(url)


def test_output_is_create_only_outside_artifacts_and_write_approval_is_refused(
    tmp_path,
):
    existing = tmp_path / "already-finalized"
    existing.mkdir()
    marker = existing / "owned.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        builder.assert_safe_output(existing)
    assert marker.read_text(encoding="utf-8") == "preserve"

    with pytest.raises(builder.DemoSafetyError, match="artifacts"):
        builder.assert_safe_output(tmp_path / "artifacts" / "demo")

    builder.assert_read_only_environment({})
    with pytest.raises(builder.DemoSafetyError, match="write/spend-approved"):
        builder.assert_read_only_environment({"QDRANT_WRITE_APPROVED": "1"})


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    (
        ("PUT", "/collections/legal", {}),
        ("DELETE", "/collections/legal", None),
        ("POST", "/collections/legal", {}),
        ("POST", "/collections/legal/points", {"points": []}),
        ("POST", "/collections/legal/points/delete", {"points": []}),
        ("PATCH", "/collections/legal/points/payload", {}),
        ("GET", "/collections/legal/points/scroll", None),
        ("POST", "/collections/legal/points/scroll", None),
    ),
)
def test_qdrant_request_allowlist_refuses_every_write_shape_before_io(
    monkeypatch, method, path, payload
):
    def network_must_not_run(*_args, **_kwargs):
        raise AssertionError("network call occurred before read-only validation")

    monkeypatch.setattr(builder.urllib.request, "urlopen", network_must_not_run)

    with pytest.raises(builder.DemoSafetyError, match="only Qdrant"):
        builder.read_only_qdrant_request(
            "http://127.0.0.1:6333",
            method,
            path,
            payload=payload,
        )


def test_scroll_requests_only_safe_metadata_and_no_vectors(monkeypatch):
    calls = []

    def fake_request(base_url, method, path, **kwargs):
        calls.append((base_url, method, path, kwargs))
        return {"result": {"points": [], "next_page_offset": None}}

    monkeypatch.setattr(builder, "read_only_qdrant_request", fake_request)
    retriever = builder.ReadOnlyQdrantRetriever(
        "http://127.0.0.1:6333", "legal_fixture"
    )
    result = retriever._scroll(  # noqa: SLF001 - security boundary under test
        {"source": "matsne", "field": "document_number", "value": "9/2019"},
        stop_after_documents=None,
    )

    assert result["scan_exhausted"] is True
    assert len(calls) == 1
    _base, method, path, kwargs = calls[0]
    assert method == "POST"
    assert path == "/collections/legal_fixture/points/scroll"
    payload = kwargs["payload"]
    assert payload["with_vector"] is False
    assert {"text", "body", "body_markdown", "parties"}.isdisjoint(
        payload["with_payload"]
    )
    assert payload["filter"] == {
        "must": [
            {"key": "source", "match": {"value": "matsne"}},
            {"key": "document_number", "match": {"value": "9/2019"}},
        ]
    }


def test_stable_result_hash_excludes_observation_and_latency_only():
    result = {
        "question_id": "q138",
        "outcome": "answer",
        "ordered_result_ids": ["ecd:5909290"],
        "timings_ms": {"total": 1.25, "retrieval": 0.75},
        "latency_ms": 1.25,
        "observed_at": "2026-07-15T10:00:00Z",
        "collection_observed_at": "2026-07-15T10:00:00Z",
    }
    repeated = {
        **result,
        "timings_ms": {"total": 99.0, "retrieval": 95.0},
        "latency_ms": 99.0,
        "observed_at": "2026-07-15T10:01:00Z",
        "collection_observed_at": "2026-07-15T10:01:00Z",
    }

    first_hash = builder.stable_result_hash(result)
    assert first_hash == builder.stable_result_hash(repeated)
    assert len(first_hash) == 64

    changed_decision = {**repeated, "outcome": "failed"}
    assert builder.stable_result_hash(changed_decision) != first_hash


def test_two_runs_are_decision_identical_and_contract_fixture_is_separate(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    first_retriever = _FixtureRetriever(rows)
    second_retriever = _FixtureRetriever(rows)
    common = {
        "corpus_identity": {"snapshot": "fixture-v1"},
        "config_identity": {"route": "exact_selector_only"},
        "contract_runner": _synthetic_contract_result,
    }

    first = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, first_retriever),
        run_number=1,
        clock=_clock(0.001),
        **common,
    )
    second = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, second_retriever),
        run_number=2,
        clock=_clock(0.013),
        **common,
    )

    assert first["decision_hash"] == second["decision_hash"]
    assert first["record_hash"] != second["record_hash"]
    assert [row["result_hash"] for row in first["results"]] == [
        row["result_hash"] for row in second["results"]
    ]
    assert {row["latency_ms"] for row in first["results"]} == {1.0}
    assert {row["latency_ms"] for row in second["results"]} == {13.0}
    assert first["synthetic_contract"]["track"] == "synthetic_contract_fixture"
    assert first["synthetic_contract"]["fixture_is_legal_evidence"] is False
    assert first["synthetic_contract"]["answer"] is None
    assert all("synthetic_contract" not in row for row in first["results"])

    assert len(first_retriever.requests) == len(builder.EXACT_IDS) + len(
        builder.AMBIGUOUS_IDS
    )
    assert first_retriever.requests == second_retriever.requests


def test_success_clarification_and_abstention_rows_emit_no_model_answer(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    retriever = _FixtureRetriever(rows)

    run = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, retriever),
        run_number=1,
        contract_runner=_synthetic_contract_result,
    )
    results = {row["question_id"]: row for row in run["results"]}

    assert all(results[qid]["outcome"] == "retrieval_identity_found" for qid in builder.EXACT_IDS)
    assert all(results[qid]["correct_document_identity"] is True for qid in builder.EXACT_IDS)
    assert all(results[qid]["required_evidence_found"] is True for qid in builder.EXACT_IDS)
    assert all(results[qid]["outcome"] == "clarify" for qid in builder.AMBIGUOUS_IDS)
    assert all(results[qid]["reason"] == "non_unique_document_number" for qid in builder.AMBIGUOUS_IDS)
    assert all(results[qid]["outcome"] == "abstain" for qid in builder.INCOMPLETE_IDS)
    assert all(
        results[qid]["reason"] == "source_incomplete_or_unattested"
        for qid in builder.INCOMPLETE_IDS
    )
    assert results[builder.BROAD_ID]["outcome"] == "abstain"
    assert results[builder.BROAD_ID]["reason"] == "unsupported_broad_legal_advice_without_case_facts"
    assert all(row["answer"] is None for row in run["results"])
    assert all(
        row[field] == builder.LEGACY_UNAVAILABLE
        for row in run["results"]
        for field in (
            "evidence_id",
            "quotation",
            "passage_hash",
            "version_proof",
            "authority_proof",
            "completeness_proof",
        )
    )


@pytest.mark.parametrize(
    ("retrieved_ids", "documents", "expected_outcome", "expected_reason"),
    (
        ([], {}, "abstain", "exact_selector_returned_no_documents"),
        (
            ["matsne:different-document"],
            {"matsne:different-document": {"official_url": "https://official.example/x"}},
            "clarify",
            "exact_selector_not_unique_or_identity_mismatch",
        ),
        (
            ["matsne:target-q138"],
            {"matsne:target-q138": {}},
            "failed",
            "official_url_missing",
        ),
    ),
)
def test_exact_identity_path_fails_closed_on_missing_or_mismatched_evidence(
    tmp_path, retrieved_ids, documents, expected_outcome, expected_reason
):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    question = builder.load_frozen_questions(golden, snapshot_roots, preflight)[0]

    result = builder.evaluate_question(
        question,
        lambda _request: {
            "ordered_result_ids": retrieved_ids,
            "ordered_point_ids": [],
            "documents": documents,
            "route_decision": "legacy_read_only_exact_selector_scroll",
        },
    )

    assert result["outcome"] == expected_outcome
    assert result["reason"] == expected_reason
    assert result["answer"] is None
    assert result["correct_document_identity"] is False
    assert result["required_evidence_found"] is False


def test_executor_failure_is_retained_without_fallback_substitution(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    retriever = _FixtureRetriever(rows, fail_question_id="q257")

    run = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, retriever),
        run_number=1,
        contract_runner=_synthetic_contract_result,
    )
    results = {row["question_id"]: row for row in run["results"]}
    failed = results["q257"]

    assert len(run["results"]) == 20
    assert failed["outcome"] == "failed"
    assert failed["degraded"] is True
    assert failed["failure"] == "RuntimeError"
    assert "failure_detail" not in failed
    assert failed["ordered_result_ids"] == []
    assert failed["answer"] is None
    assert results["q271"]["outcome"] == "retrieval_identity_found"


def test_contract_fixture_failure_is_visible_but_not_mixed_into_retrieval(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    retriever = _FixtureRetriever(rows)

    def failed_contract():
        raise RuntimeError("private synthetic failure detail")

    run = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, retriever),
        run_number=1,
        contract_runner=failed_contract,
    )

    assert len(run["results"]) == 20
    assert all(row["outcome"] != "failed" for row in run["results"])
    assert run["synthetic_contract"] == {
        "track": "synthetic_contract_fixture",
        "passed": False,
        "status": "failed",
        "failure": "RuntimeError",
        "answer": None,
    }
    assert "private synthetic failure detail" not in json.dumps(run)


def test_scorecard_keeps_identity_integrity_abstention_and_legal_review_separate(
    tmp_path,
):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    first_retriever = _FixtureRetriever(rows, fail_question_id="q257")
    second_retriever = _FixtureRetriever(rows, fail_question_id="q257")
    common = {
        "corpus_identity": {"snapshot": "fixture-v1"},
        "config_identity": {"route": "exact_selector_only"},
        "contract_runner": _synthetic_contract_result,
    }
    first = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, first_retriever),
        run_number=1,
        **common,
    )
    second = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, second_retriever),
        run_number=2,
        **common,
    )

    scorecard = builder.build_scorecard(
        rows,
        first,
        second,
        question_set_hash=builder.sha256_canonical_jsonl(rows),
        test_evidence={
            "presentation_tests": {"status": "passed", "passed": 28, "failed": 0},
            "ingest_non_snapshot_tests": {
                "status": "interrupted",
                "summary": "existing async regression stalled",
            },
        },
    )

    assert scorecard["total_questions"] == 20
    assert scorecard["expected_answer_questions"] == 15
    assert scorecard["correct_document_identity"] == 14
    assert scorecard["required_evidence_found"] == 14
    assert scorecard["canonical_exact_quotation_validations"] == {
        "validated": 0,
        "status": builder.LEGACY_UNAVAILABLE,
        "explanation": "Legacy retrieval payloads are not canonical generation evidence.",
    }
    assert scorecard["frozen_snapshot_span_reconstruction"][
        "validated_expected_answer_rows"
    ] == 15
    assert scorecard["correct_abstention_or_clarification"] == {
        "ambiguous": 2,
        "incomplete_source": 2,
        "unsupported_broad_advice": 1,
    }
    assert scorecard["degraded_or_failed_queries"] == {
        "run_1": ["q257"],
        "run_2": ["q257"],
    }
    assert scorecard["deterministic_repeats"]["matching_questions"] == 20
    assert scorecard["synthetic_contract_fixture"][
        "excluded_from_measured_retrieval_results"
    ] is True
    assert scorecard["legal_correctness_review"] == "not yet independently adjudicated"
    assert scorecard["test_evidence"]["ingest_non_snapshot_tests"]["status"] == (
        "interrupted"
    )
    assert "existing async regression stalled" in builder._scorecard_markdown(scorecard)
    assert "accuracy" not in scorecard


def test_presentation_text_uses_approved_positioning_and_forbids_overclaiming(tmp_path):
    golden, snapshot_roots, preflight = _write_frozen_fixture(tmp_path)
    rows = builder.load_frozen_questions(golden, snapshot_roots, preflight)
    retriever = _FixtureRetriever(rows)
    run = builder.run_once(
        rows,
        execute=lambda question: builder.evaluate_question(question, retriever),
        run_number=1,
        contract_runner=_synthetic_contract_result,
    )
    scorecard = builder.build_scorecard(
        rows,
        run,
        run,
        question_set_hash=builder.sha256_canonical_jsonl(rows),
    )
    documents = "\n".join(
        (
            builder._readme(),
            builder._scorecard_markdown(scorecard),
            builder._demo_script(rows, run, scorecard),
            builder._slides_outline(scorecard),
            builder._talk_track(scorecard),
            builder._reviewer_checklist(),
            builder._limitations(),
        )
    )
    lowered = documents.casefold()

    assert builder.APPROVED_POSITIONING in documents
    assert builder.LEGAL_REVIEW_SENTENCE in documents
    assert "production severe-error certification is not yet complete" in lowered
    assert "not a blind benchmark" in lowered
    assert "full immutable corpus" in lowered
    assert "PRIVATE_Q024_QUOTE" not in documents
    assert "PRIVATE_PARTY" not in documents
    assert "100% accurate" not in lowered
    assert "below 1% severe errors" not in lowered
    assert "lawyer validated" not in lowered
    assert "production ready" not in lowered


def test_atomic_bundle_build_hashes_questions_before_execution_and_rehearses_offline(
    tmp_path, monkeypatch, capsys
):
    inputs = _write_bundle_inputs(tmp_path / "inputs")
    events = []

    class EventRetriever(_FixtureRetriever):
        def observe_collection(self):
            events.append("observe_collection")
            return super().observe_collection()

        def __call__(self, request):
            events.append(f"retrieve:{request['question_id']}")
            return super().__call__(request)

    retriever = EventRetriever(inputs["rows"])
    real_write_questions = builder._write_questions

    def tracked_write_questions(stage, rows):
        digest = real_write_questions(stage, rows)
        assert (stage / "questions.jsonl").is_file()
        assert (stage / "questions.sha256").is_file()
        events.append("questions_hashed")
        return digest

    real_run_once = builder.run_once

    def hermetic_run_once(*args, **kwargs):
        kwargs["contract_runner"] = _synthetic_contract_result
        return real_run_once(*args, **kwargs)

    monkeypatch.setattr(builder, "_write_questions", tracked_write_questions)
    monkeypatch.setattr(builder, "run_once", hermetic_run_once)
    for name in builder._WRITE_APPROVAL_ENVS:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "presentation" / "accuracy-first-demo"

    built = builder.build_bundle(
        **{key: value for key, value in inputs.items() if key != "rows"},
        output=output,
        qdrant_url="http://127.0.0.1:6333",
        collection="legal_fixture",
        retriever=retriever,
    )

    assert built == output.resolve()
    assert set(path.name for path in output.iterdir()) == EXPECTED_BUNDLE_FILES
    assert events[0] == "questions_hashed"
    assert events[1] == "observe_collection"
    assert events.index("questions_hashed") < min(
        index for index, event in enumerate(events) if event.startswith("retrieve:")
    )
    persisted_question_bytes = (output / "questions.jsonl").read_bytes()
    assert b"PRIVATE_Q024_QUOTE" not in persisted_question_bytes
    assert b"PRIVATE_PARTY" not in persisted_question_bytes
    assert "\u009d".encode("utf-8") in persisted_question_bytes
    assert "\u00a0".encode("utf-8") in persisted_question_bytes

    verification = builder.verify_bundle(output)
    assert verification["deterministic"] is True
    assert verification["total_questions"] == 20
    assert len(verification["question_set_sha256"]) == 64
    static_script = (output / "demo-script.md").read_text(encoding="utf-8")
    assert builder.rehearse_bundle(output) == static_script

    def network_must_not_run(*_args, **_kwargs):
        raise AssertionError("verify/rehearse attempted network access")

    monkeypatch.setattr(builder.urllib.request, "urlopen", network_must_not_run)
    assert builder.main(["verify", "--bundle", str(output)]) == 0
    verify_stdout = capsys.readouterr().out
    assert '"deterministic": true' in verify_stdout
    assert builder.main(["rehearse", "--bundle", str(output)]) == 0
    assert capsys.readouterr().out == static_script

    events.clear()
    with pytest.raises(FileExistsError, match="already exists"):
        builder.build_bundle(
            **{key: value for key, value in inputs.items() if key != "rows"},
            output=output,
            qdrant_url="http://127.0.0.1:6333",
            collection="legal_fixture",
            retriever=retriever,
        )
    assert events == []


def test_failed_final_validation_leaves_no_bundle_or_staging_directory(
    tmp_path, monkeypatch
):
    inputs = _write_bundle_inputs(tmp_path / "inputs")
    retriever = _FixtureRetriever(inputs["rows"])
    real_run_once = builder.run_once

    def hermetic_run_once(*args, **kwargs):
        kwargs["contract_runner"] = _synthetic_contract_result
        return real_run_once(*args, **kwargs)

    def reject_staged_bundle(_stage):
        raise ValueError("forced final validation failure")

    monkeypatch.setattr(builder, "run_once", hermetic_run_once)
    monkeypatch.setattr(builder, "validate_bundle", reject_staged_bundle)
    for name in builder._WRITE_APPROVAL_ENVS:
        monkeypatch.delenv(name, raising=False)
    parent = tmp_path / "presentation"
    output = parent / "accuracy-first-demo"

    with pytest.raises(ValueError, match="forced final validation failure"):
        builder.build_bundle(
            **{key: value for key, value in inputs.items() if key != "rows"},
            output=output,
            qdrant_url="http://127.0.0.1:6333",
            collection="legal_fixture",
            retriever=retriever,
        )

    assert not output.exists()
    assert list(parent.glob(".accuracy-first-demo.staging-*")) == []


def test_offline_verifier_detects_persisted_result_tampering(tmp_path, monkeypatch):
    inputs = _write_bundle_inputs(tmp_path / "inputs")
    retriever = _FixtureRetriever(inputs["rows"])
    real_run_once = builder.run_once

    def hermetic_run_once(*args, **kwargs):
        kwargs["contract_runner"] = _synthetic_contract_result
        return real_run_once(*args, **kwargs)

    monkeypatch.setattr(builder, "run_once", hermetic_run_once)
    for name in builder._WRITE_APPROVAL_ENVS:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "bundle"
    builder.build_bundle(
        **{key: value for key, value in inputs.items() if key != "rows"},
        output=output,
        qdrant_url="http://127.0.0.1:6333",
        collection="legal_fixture",
        retriever=retriever,
    )
    run_path = output / "run-1.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["results"][0]["outcome"] = "fabricated_success"
    run_path.write_text(
        json.dumps(run, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="result hash mismatch"):
        builder.verify_bundle(output)
