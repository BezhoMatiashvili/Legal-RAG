#!/usr/bin/env python3
"""Build and verify the offline, accuracy-first presentation demonstration.

This script is intentionally narrower than the production answer service.  It performs
only loopback Qdrant metadata/scroll reads, audits frozen local snapshot spans, and emits
an honest static bundle.  It never loads a model and has no Qdrant-writing code path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


EXACT_IDS = (
    "q138", "q169", "q208", "q211", "q229", "q257", "q271", "q276",
    "q283", "q285", "q291", "q293", "q312", "q329", "q343",
)
AMBIGUOUS_IDS = ("q121", "q191")
INCOMPLETE_IDS = ("q003", "q024")
BROAD_ID = "demo-broad-001"
QUESTION_IDS = EXACT_IDS + AMBIGUOUS_IDS + INCOMPLETE_IDS + (BROAD_ID,)
DISPLAY_IDS = ("q169", "q291", "q329")

REQUIRED_BUNDLE_FILES = (
    "README.md", "questions.jsonl", "questions.sha256", "run-1.json",
    "run-2.json", "scorecard.json", "scorecard.md", "demo-script.md",
    "slides-outline.md", "talk-track.md", "reviewer-checklist.md",
    "limitations.md",
)
LEGACY_UNAVAILABLE = "not_available_in_legacy_corpus"
# Alias retained because the approved plan used both spellings while tests were drafted.
LEGACY_NOT_AVAILABLE = LEGACY_UNAVAILABLE
APPROVED_POSITIONING = (
    "This is an accuracy-first pre-production Georgian legal RAG. It answers only when "
    "evidence, identity, version, quotation, and completeness checks pass; otherwise it "
    "clarifies or abstains."
)
LEGAL_REVIEW_SENTENCE = "Legal correctness review: not yet independently adjudicated."

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_DROP_FROM_STABLE_HASH = frozenset({
    "latency_ms", "observed_at", "started_at", "completed_at", "run_number",
    "record_hash", "result_hash", "timings_ms", "collection_observed_at",
})
_WRITE_APPROVAL_ENVS = (
    "QDRANT_WRITE_APPROVED", "QDRANT_RECREATE_APPROVED",
    "RUNPOD_EPHEMERAL_QDRANT", "RUNPOD_SPEND_APPROVED",
)
_PAYLOAD_FIELDS = (
    "source", "document_id", "document_number", "registration_code", "status",
    "date", "source_url", "official_url", "content_hash", "chunk_index",
)
_TEST_EVIDENCE_KEYS = (
    "presentation_tests", "legal_answer_evidence_abstention", "root_tests",
    "ingest_non_snapshot_tests", "ruff_root", "ruff_ingest", "symbol_map",
    "git_diff_check", "git_diff_cached_check",
)


class DemoSafetyError(RuntimeError):
    """Raised before an unsafe path, network destination, or method is used."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _without_private(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_private(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_without_private(item) for item in value]
    return value


def _stable_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_projection(item)
            for key, item in value.items()
            if str(key) not in _DROP_FROM_STABLE_HASH
        }
    if isinstance(value, (list, tuple)):
        return [_stable_projection(item) for item in value]
    return value


def stable_result_hash(result: Mapping[str, Any]) -> str:
    """Hash decision material while deliberately excluding timing/observation fields."""

    return sha256_bytes(canonical_json_bytes(_stable_projection(dict(result))))


def validate_loopback_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise DemoSafetyError("Qdrant URL must be unauthenticated loopback HTTP")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise DemoSafetyError("Qdrant URL must contain only scheme, loopback host, and port")
    host = parsed.hostname
    if not host:
        raise DemoSafetyError("Qdrant URL has no host")
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.casefold() == "localhost"
    if not is_loopback:
        raise DemoSafetyError("external network access is forbidden; use a loopback Qdrant URL")
    if parsed.port is None:
        raise DemoSafetyError("Qdrant URL must include an explicit port")
    return url.rstrip("/")


def validate_qdrant_url(url: str) -> str:
    """Compatibility alias for callers/tests using the earlier planned helper name."""

    return validate_loopback_url(url)


def assert_read_only_environment(environ: Mapping[str, str] | None = None) -> None:
    values = os.environ if environ is None else environ
    active = [name for name in _WRITE_APPROVAL_ENVS if values.get(name) == "1"]
    if active:
        raise DemoSafetyError("refusing a demo build in a write/spend-approved environment: " + ", ".join(active))


def assert_safe_output(path: Path) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    if "artifacts" in {part.casefold() for part in output.parts}:
        raise DemoSafetyError("presentation output under artifacts/ is forbidden")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"finalized bundle already exists: {output}")
    if output.name in ("", ".", ".."):
        raise DemoSafetyError("unsafe output directory")
    return output


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ).encode("utf-8") + b"\n")


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.rstrip() .encode("utf-8") + b"\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                rows.append(row)
    return rows


def _snapshot_documents(
    roots: Sequence[Path], needed: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    by_source: dict[str, set[str]] = {}
    for source, document_id in needed:
        by_source.setdefault(source, set()).add(document_id)
    found: dict[tuple[str, str], dict[str, Any]] = {}
    layers: dict[tuple[str, str], str] = {}
    # Earlier roots are authoritative and therefore applied last.
    for root in reversed(tuple(Path(root) for root in roots)):
        docs_root = root / "docs" if (root / "docs").is_dir() else root
        layer = root.name if docs_root != root else root.parent.name
        for source, wanted in by_source.items():
            path = docs_root / f"{source}.jsonl"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    document_id = str(record.get("document_id", ""))
                    key = (source, document_id)
                    if document_id in wanted:
                        found[key] = record
                        layers[key] = layer
    missing = sorted(needed - set(found))
    if missing:
        raise ValueError(f"selected documents absent from frozen snapshots: {missing}")
    return found, layers


def _selector_from_question(question: str, snapshot: Mapping[str, Any]) -> dict[str, str]:
    for field in ("registration_code", "document_number"):
        raw = snapshot.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if value and value in question:
            return {"field": field, "value": value}
        decorated = value.lstrip("№N#").strip()
        if decorated and (f"№{decorated}" in question or f"N{decorated}" in question):
            return {"field": field, "value": value}
    raise ValueError("no exact document number or registration code is literal in the frozen question")


def _manifest_source_evidence(manifest: Mapping[str, Any], source: str) -> dict[str, Any]:
    source_row = dict((manifest.get("sources") or {}).get(source) or {})
    quarantined = dict(source_row.get("quarantined") or {})
    if int(source_row.get("clean", -1)) != 0 or not quarantined:
        raise ValueError(f"preflight manifest does not prove {source} incomplete/unattested")
    return {
        "source": source,
        "clean_documents": 0,
        "quarantine_reasons": sorted(quarantined),
        "quarantined_documents": sum(int(value) for value in quarantined.values()),
    }


def load_frozen_questions(
    golden_set: Path,
    snapshot_roots: Sequence[Path],
    preflight_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Create the fixed public 20-row set without carrying document bodies or party data."""

    golden_path = Path(golden_set)
    selected = {row.get("id"): row for row in _load_jsonl(golden_path) if row.get("id") in QUESTION_IDS}
    missing = sorted(set(QUESTION_IDS[:-1]) - set(selected))
    if missing:
        raise ValueError(f"golden set is missing fixed questions: {missing}")
    needed = {
        (str(selected[qid]["gold"]["source"]), str(selected[qid]["gold"]["document_id"]))
        for qid in QUESTION_IDS[:-1]
    }
    documents, layers = _snapshot_documents(snapshot_roots, needed)
    preflight: Mapping[str, Any] = {}
    preflight_hash: str | None = None
    if preflight_manifest is not None:
        preflight_path = Path(preflight_manifest)
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight_hash = sha256_file(preflight_path)
    rows: list[dict[str, Any]] = []
    golden_hash = sha256_file(golden_path)
    for qid in QUESTION_IDS[:-1]:
        gold = selected[qid]
        source = str(gold["gold"]["source"])
        document_id = str(gold["gold"]["document_id"])
        snapshot = documents[(source, document_id)]
        body = snapshot.get("body_markdown")
        if not isinstance(body, str):
            raise ValueError(f"{qid}: frozen snapshot has no canonical body")
        content_hash = sha256_bytes(body.encode("utf-8", "replace"))
        if content_hash != snapshot.get("content_hash"):
            raise ValueError(f"{qid}: frozen document content hash mismatch")
        relevance = gold.get("relevance") or []
        if not relevance:
            raise ValueError(f"{qid}: no frozen evidence annotation")
        rel = relevance[0]
        start, end = int(rel["char_start"]), int(rel["char_end"])
        quote = str(rel["evidence_quote"])
        reconstructed = body[start:end]
        if reconstructed != quote:
            raise ValueError(f"{qid}: frozen quote offsets do not reconstruct exact bytes")
        if gold.get("query_language") != "ka":
            raise ValueError(f"{qid}: non-Georgian question is outside presentation scope")
        category = (
            "expected_answer_exact_identity" if qid in EXACT_IDS else
            "ambiguous_expected_clarification" if qid in AMBIGUOUS_IDS else
            "incomplete_source_expected_abstention"
        )
        expected = (
            "retrieval_identity_found" if qid in EXACT_IDS else
            "clarify" if qid in AMBIGUOUS_IDS else "abstain"
        )
        frozen_audit: dict[str, Any] = {
            "char_start": start,
            "char_end": end,
            "offset_unit": "unicode_codepoint",
            "quote_sha256": sha256_bytes(quote.encode("utf-8")),
            "quote_reconstructed": True,
            "exact_substring": True,
            "document_content_hash": content_hash,
            "document_content_hash_matches": True,
        }
        if qid in DISPLAY_IDS:
            frozen_audit["quotation"] = quote
        row: dict[str, Any] = {
            "schema_version": 1,
            "question_id": qid,
            "question": str(gold["query"]),
            "language": "ka",
            "category": category,
            "expected_outcome": expected,
            "expected_source": source,
            "expected_document_id": document_id,
            "expected_article": "not_annotated",
            "lookup_selector": (
                None if qid in INCOMPLETE_IDS else {
                    "source": source,
                    **_selector_from_question(str(gold["query"]), snapshot),
                }
            ),
            "lookup_selector_origin": "frozen_snapshot_metadata_and_question_literal",
            "publication_metadata_date": snapshot.get("date"),
            "frozen_status": snapshot.get("status"),
            "official_url": snapshot.get("official_url") or snapshot.get("source_url"),
            "snapshot_layer": layers[(source, document_id)],
            "frozen_snapshot_audit": frozen_audit,
            "evidence_provenance": {
                "kind": "frozen_golden_annotation_and_local_snapshot",
                "evaluation_version": "golden_set_v2",
                "golden_set_sha256": golden_hash,
                "official_source_url_present": bool(
                    snapshot.get("official_url") or snapshot.get("source_url")
                ),
            },
            "selection_rationale": (
                "Predeclared Georgian exact-identifier example with local frozen evidence."
                if qid in EXACT_IDS else
                "Predeclared non-unique number requiring clarification."
                if qid in AMBIGUOUS_IDS else
                "Predeclared source that the local v3 preflight quarantines as incomplete."
            ),
            "display_example": qid in (*DISPLAY_IDS, "q121", "q024"),
        }
        if qid in INCOMPLETE_IDS:
            if not preflight:
                raise ValueError("preflight manifest is required for incomplete-source questions")
            row["incomplete_source_evidence"] = {
                **_manifest_source_evidence(preflight, source),
                "preflight_manifest_sha256": preflight_hash,
            }
        rows.append(row)
    rows.append({
        "schema_version": 1,
        "question_id": BROAD_ID,
        "question": "როგორ მოვიგო ნებისმიერი სასამართლო დავა? მომეცი სრული იურიდიული რჩევა ჩემი საქმის დეტალების გარეშე.",
        "language": "ka",
        "category": "unsupported_broad_advice_expected_abstention",
        "expected_outcome": "abstain",
        "expected_source": None,
        "expected_document_id": None,
        "expected_article": "not_annotated",
        "lookup_selector": None,
        "evidence_provenance": {"kind": "authored_demo_scope_boundary", "evaluation_version": "presentation-v1"},
        "selection_rationale": "Predeclared broad-advice request outside the narrow demonstration contract.",
        "display_example": False,
    })
    validate_question_set(rows)
    return rows


def validate_question_set(rows: Sequence[Mapping[str, Any]]) -> None:
    if tuple(str(row.get("question_id")) for row in rows) != QUESTION_IDS:
        raise ValueError("question order/identity differs from the fixed presentation set")
    for row in rows:
        if row.get("language") != "ka":
            raise ValueError("live demonstration is Georgian-only")
        selector = row.get("lookup_selector")
        if row.get("question_id") in EXACT_IDS:
            official_url = row.get("official_url")
            if not isinstance(official_url, str) or not official_url.strip():
                raise ValueError(
                    f"{row.get('question_id')}: exact-identity question has no official URL"
                )
        if selector is not None:
            if set(selector) != {"source", "field", "value"}:
                raise ValueError(f"{row.get('question_id')}: malformed lookup selector")
            if selector["field"] not in {"document_number", "registration_code"}:
                raise ValueError("retrieval may not filter by expected document_id")
            if selector["field"] == "document_id":
                raise ValueError("expected document_id may only be used for scoring")


def read_only_qdrant_request(
    base_url: str,
    method: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Issue one allowlisted Qdrant read; POST is allowed only for the scroll API."""

    base = validate_loopback_url(base_url)
    normalized_method = method.upper()
    metadata_path = re.fullmatch(r"/collections/[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", path)
    scroll_path = re.fullmatch(
        r"/collections/[A-Za-z0-9][A-Za-z0-9_.-]{0,199}/points/scroll", path,
    )
    if not (
        (normalized_method == "GET" and metadata_path and payload is None)
        or (normalized_method == "POST" and scroll_path and payload is not None)
    ):
        raise DemoSafetyError("only Qdrant collection metadata GET and points/scroll POST are permitted")
    body = canonical_json_bytes(dict(payload)) if payload is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["api-key"] = api_key
    request = urllib.request.Request(
        base + path, data=body, headers=headers, method=normalized_method,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(64 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"loopback Qdrant read failed: {type(exc).__name__}") from exc
    if len(raw) > 64 * 1024 * 1024:
        raise RuntimeError("Qdrant response exceeded the 64 MiB safety limit")
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Qdrant returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Qdrant returned a non-object response")
    return decoded


def _safe_collection_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Qdrant collection metadata has no result object")
    config = result.get("config") if isinstance(result.get("config"), Mapping) else {}
    params = config.get("params") if isinstance(config.get("params"), Mapping) else {}
    vectors = params.get("vectors")
    vector_names = sorted(vectors) if isinstance(vectors, Mapping) else ["unnamed"] if vectors else []
    sparse = params.get("sparse_vectors")
    return {
        "status": str(result.get("status") or "unknown"),
        "optimizer_status": str(result.get("optimizer_status") or "unknown"),
        "points_count": result.get("points_count"),
        "indexed_vectors_count": result.get("indexed_vectors_count"),
        "segments_count": result.get("segments_count"),
        "vector_names": vector_names,
        "sparse_vector_names": sorted(sparse) if isinstance(sparse, Mapping) else [],
    }


class ReadOnlyQdrantRetriever:
    """Model-free exact selector lookup over the legacy local collection."""

    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
        page_size: int = 256,
    ) -> None:
        self.qdrant_url = validate_loopback_url(qdrant_url)
        if not _COLLECTION_RE.fullmatch(collection):
            raise DemoSafetyError("unsafe Qdrant collection name")
        self.collection = collection
        self.api_key = api_key
        self.timeout = timeout
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        self.page_size = page_size

    @property
    def _collection_path(self) -> str:
        return "/collections/" + urllib.parse.quote(self.collection, safe="._-")

    def observe_collection(self) -> dict[str, Any]:
        response = read_only_qdrant_request(
            self.qdrant_url, "GET", self._collection_path,
            api_key=self.api_key, timeout=self.timeout,
        )
        return _safe_collection_metadata(response)

    def _scroll(self, selector: Mapping[str, str], *, stop_after_documents: int | None) -> dict[str, Any]:
        offset: Any = None
        points: dict[str, dict[str, Any]] = {}
        document_order: list[str] = []
        documents: dict[str, dict[str, Any]] = {}
        exhausted = False
        pages = 0
        while True:
            request_payload: dict[str, Any] = {
                "filter": {"must": [
                    {"key": "source", "match": {"value": selector["source"]}},
                    {"key": selector["field"], "match": {"value": selector["value"]}},
                ]},
                "limit": self.page_size,
                "with_payload": list(_PAYLOAD_FIELDS),
                "with_vector": False,
            }
            if offset is not None:
                request_payload["offset"] = offset
            response = read_only_qdrant_request(
                self.qdrant_url, "POST", self._collection_path + "/points/scroll",
                payload=request_payload, api_key=self.api_key, timeout=self.timeout,
            )
            result = response.get("result")
            if not isinstance(result, Mapping) or not isinstance(result.get("points"), list):
                raise RuntimeError("Qdrant scroll response is malformed")
            pages += 1
            for point in result["points"]:
                if not isinstance(point, Mapping) or not isinstance(point.get("payload"), Mapping):
                    continue
                point_id = str(point.get("id") or "")
                payload = {key: point["payload"].get(key) for key in _PAYLOAD_FIELDS}
                source = str(payload.get("source") or "")
                document_id = str(payload.get("document_id") or "")
                if not point_id or not source or not document_id:
                    continue
                points[point_id] = payload
                result_id = f"{source}:{document_id}"
                if result_id not in documents:
                    documents[result_id] = payload
                    document_order.append(result_id)
            if stop_after_documents is not None and len(document_order) >= stop_after_documents:
                break
            next_offset = result.get("next_page_offset")
            if next_offset is None:
                exhausted = True
                break
            if next_offset == offset:
                raise RuntimeError("Qdrant scroll continuation did not advance")
            offset = next_offset
        retained = sorted(document_order)[:2] if stop_after_documents is not None else sorted(document_order)
        return {
            "ordered_result_ids": retained,
            "ordered_point_ids": sorted(points),
            "documents": {result_id: documents[result_id] for result_id in retained},
            "distinct_document_count_lower_bound": len(document_order),
            "scan_exhausted": exhausted,
            "truncated": len(document_order) > len(retained) or not exhausted,
            "pages_read": pages,
        }

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if set(request) != {"question_id", "category", "lookup_selector"}:
            raise DemoSafetyError("retriever boundary received scoring or evidence fields")
        selector = dict(request["lookup_selector"])
        if selector.get("field") not in {"document_number", "registration_code"}:
            raise DemoSafetyError("retriever selector must be an exact legal identifier")
        stop_after = 3 if request["category"] == "ambiguous_expected_clarification" else None
        raw_value = str(selector["value"])
        stripped = raw_value.lstrip("№N#").strip()
        values: list[str] = []
        for value in (raw_value, f"N{stripped}", f"№{stripped}"):
            if value and value not in values:
                values.append(value)
        attempts: list[dict[str, str]] = []
        scanned: dict[str, Any] | None = None
        selector_used: dict[str, str] | None = None
        for value in values:
            candidate = {**selector, "value": value}
            attempts.append(candidate)
            candidate_result = self._scroll(candidate, stop_after_documents=stop_after)
            if candidate_result["ordered_result_ids"]:
                scanned = candidate_result
                selector_used = candidate
                break
            scanned = candidate_result
        assert scanned is not None
        return {
            **scanned,
            "retrieval_filter": selector,
            "selector_attempts": attempts,
            "selector_used": selector_used,
            "route_decision": "legacy_read_only_exact_selector_scroll",
        }


def evaluate_question(
    question: Mapping[str, Any],
    retriever: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply policy and score identity after the anti-leakage retrieval boundary."""

    category = str(question["category"])
    base = _default_result(question)
    base.update({
        "route_decision": None, "reason": None, "degraded": False, "failure": None,
        "expected_article": question.get("expected_article", "not_annotated"),
        "official_url": None,
        "official_url_matches_frozen": False,
        "frozen_status": question.get("frozen_status"),
        "publication_metadata_date": question.get("publication_metadata_date"),
        "frozen_snapshot_audit": copy.deepcopy(question.get("frozen_snapshot_audit")),
    })
    if category == "unsupported_broad_advice_expected_abstention":
        base.update({
            "route_decision": "demo_scope_policy", "outcome": "abstain",
            "reason": "unsupported_broad_legal_advice_without_case_facts",
        })
        return base
    if category == "incomplete_source_expected_abstention":
        base.update({
            "route_decision": "frozen_v3_preflight_completeness_gate",
            "outcome": "abstain", "reason": "source_incomplete_or_unattested",
            "incomplete_source_evidence": copy.deepcopy(question.get("incomplete_source_evidence")),
        })
        return base
    request = {
        "question_id": question["question_id"],
        "category": category,
        "lookup_selector": copy.deepcopy(question["lookup_selector"]),
    }
    scanned = dict(retriever(request))
    base.update(scanned)
    if category == "ambiguous_expected_clarification":
        if len(scanned.get("ordered_result_ids", [])) >= 2:
            base.update({"outcome": "clarify", "reason": "non_unique_document_number"})
        else:
            base.update({
                "outcome": "failed", "reason": "predeclared_ambiguity_not_mechanically_observed",
                "degraded": True, "failure": "fewer_than_two_distinct_documents",
            })
        return base
    expected_result_id = f"{question['expected_source']}:{question['expected_document_id']}"
    result_ids = scanned.get("ordered_result_ids", [])
    only_expected = result_ids == [expected_result_id]
    if only_expected:
        documents = scanned.get("documents") or {}
        payload = documents.get(expected_result_id) or {}
        official_url = payload.get("official_url") or payload.get("source_url")
        if not isinstance(official_url, str) or not official_url.strip():
            base.update({
                "outcome": "failed", "reason": "official_url_missing", "degraded": True,
                "failure": "retrieved_identity_has_no_official_url",
            })
            return base
        if official_url != question.get("official_url"):
            base.update({
                "outcome": "failed", "reason": "official_url_mismatch", "degraded": True,
                "failure": "retrieved_official_url_differs_from_frozen_snapshot",
                "official_url": official_url,
            })
            return base
        base.update({
            "outcome": "retrieval_identity_found",
            "reason": "one_exact_selector_identity_matches_annotation",
            "required_evidence_found": True,
            "correct_document_identity": True,
            "official_url": official_url,
            "official_url_matches_frozen": True,
            "observed_document_number": payload.get("document_number"),
            "observed_registration_code": payload.get("registration_code"),
            "observed_status": payload.get("status"),
            "legacy_content_hash_observation": payload.get("content_hash"),
            "legacy_content_hash_verification": LEGACY_UNAVAILABLE,
        })
    elif not result_ids:
        base.update({"outcome": "abstain", "reason": "exact_selector_returned_no_documents"})
    else:
        base.update({"outcome": "clarify", "reason": "exact_selector_not_unique_or_identity_mismatch"})
    return base


def _default_result(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "category": question["category"],
        "expected_outcome": question["expected_outcome"],
        "outcome": "failed",
        "answer": None,
        "ordered_result_ids": [],
        "ordered_point_ids": [],
        "route_decision": "executor_failure",
        "reason": "executor_failure",
        "degraded": True,
        "failure": "unclassified_executor_failure",
        "required_evidence_found": False,
        "correct_document_identity": False,
        "evidence_id": LEGACY_UNAVAILABLE,
        "quotation": LEGACY_UNAVAILABLE,
        "passage_hash": LEGACY_UNAVAILABLE,
        "version_proof": LEGACY_UNAVAILABLE,
        "authority_proof": LEGACY_UNAVAILABLE,
        "completeness_proof": LEGACY_UNAVAILABLE,
        "official_url_matches_frozen": False,
        "legacy_content_hash_verification": LEGACY_UNAVAILABLE,
    }


def run_synthetic_contract_checks() -> dict[str, Any]:
    """Exercise the real schema-v2 canonical evidence validator without any model answer."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ingest_root = Path(__file__).resolve().parents[1]
    if str(ingest_root) not in sys.path:
        sys.path.insert(0, str(ingest_root))
    import dataclasses
    from types import SimpleNamespace

    from ingest.config import load_config
    from ingest.generation import CANONICAL_PAYLOAD_REVISION, GENERATION_SCHEMA_VERSION
    from ingest.legal_answer import (
        CanonicalEvidenceRepository, ClaimKind, DraftAnswer, DraftClaim, DraftQuotation,
        EvidenceContractError, canonical_evidence_from_point, parse_evidence_id,
        validate_draft,
    )

    generation = "synthetic-contract-v2"
    text = "SYNTHETIC CONTRACT FIXTURE — NOT LEGAL CONTENT"
    passage_hash = sha256_bytes(text.encode("utf-8"))
    start, end = 100, 100 + len(text)
    passage_material = f"fixture\0doc-1\0v1\0{start}\0{end}\0{passage_hash}"
    payload: dict[str, Any] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        "generation_id": generation,
        "source": "fixture",
        "source_authority": "official",
        "source_fingerprint": "a" * 64,
        "normalizer_revision": "synthetic-normalizer-v1",
        "chunker_revision": "synthetic-chunker-v1",
        "model_revision": "synthetic-contract-fixture-no-model-loaded",
        "document_id": "doc-1",
        "title": "Synthetic fixture",
        "document_number": "S-1",
        "registration_code": None,
        "document_type": "synthetic_fixture",
        "article_id": None,
        "clause_id": None,
        "subarticle": None,
        "chapter": None,
        "heading_path": [],
        "parent_id": None,
        "article_start_chunk_index": 0,
        "parent_chunk_index": 0,
        "court": None,
        "status": "fixture",
        "version_id": "v1",
        "supersedes": [],
        "effective_from": None,
        "effective_to": None,
        "repeal_date": None,
        "consolidation_status": "fixture",
        "version_lineage_status": "complete",
        "version_lineage_complete": True,
        "content_complete": True,
        "extraction_status": "full_text",
        "official_url": "https://invalid.example/fixture-not-accessed",
        "official_binary_url": None,
        "page_start": None,
        "page_end": None,
        "char_start": start,
        "char_end": end,
        "offset_unit": "unicode_codepoint",
        "canonical_text_exact": True,
        "content_hash": "c" * 64,
        "canonical_content_hash": "c" * 64,
        "passage_hash": passage_hash,
        "passage_id": "passage:" + sha256_bytes(passage_material.encode("utf-8")),
        "text": text,
        "token_count": len(text.split()),
        "chunk_index": 0,
        "freshness_sla_met": True,
    }
    cfg = dataclasses.replace(
        load_config(), generation_id=generation, embedding_revision=None,
    )
    point = SimpleNamespace(id="synthetic-point-1", score=1.0, payload=payload)
    checks: list[dict[str, Any]] = []
    try:
        evidence = canonical_evidence_from_point(cfg, point)
        locator = parse_evidence_id(evidence.evidence_id)
        ok = locator == {
            "generation_id": generation,
            "point_id": "synthetic-point-1",
            "passage_hash": passage_hash,
        }
        checks.append({"name": "evidence_id_binding", "passed": ok})
        pack = CanonicalEvidenceRepository(cfg, None).build_pack(
            (point,), retrieval_result_hash="d" * 64,
        )
        relative = text.index("CONTRACT FIXTURE")
        quote = "CONTRACT FIXTURE"
        draft = DraftAnswer(
            "SYNTHETIC VALIDATOR INPUT",
            (DraftClaim(
                "synthetic-claim", "SYNTHETIC CLAIM", ClaimKind.QUOTED_LAW,
                (evidence.evidence_id,),
                (DraftQuotation(
                    evidence.evidence_id, quote, start + relative,
                    start + relative + len(quote),
                ),),
            ),),
        )
        validated = validate_draft(draft, pack, as_of=None)
        quote_hash_ok = bool(
            validated.valid and validated.claims
            and validated.claims[0].quotations[0].quote_hash == sha256_bytes(quote.encode("utf-8"))
        )
        checks.append({"name": "exact_quote_offsets_and_quote_hash", "passed": quote_hash_ok})
        tampered = DraftAnswer(
            draft.answer_text,
            (DraftClaim(
                "synthetic-claim", "SYNTHETIC CLAIM", ClaimKind.QUOTED_LAW,
                (evidence.evidence_id,),
                (DraftQuotation(
                    evidence.evidence_id, "TAMPERED FIXTURE", start + relative,
                    start + relative + len(quote),
                ),),
            ),),
        )
        checks.append({
            "name": "tampered_quote_rejected",
            "passed": not validate_draft(tampered, pack, as_of=None).valid,
        })
    except Exception as exc:  # retained as a failed fixture, not silently substituted
        checks.extend([
            {"name": "evidence_id_binding", "passed": False, "failure": type(exc).__name__},
            {"name": "exact_quote_offsets_and_quote_hash", "passed": False, "failure": type(exc).__name__},
            {"name": "tampered_quote_rejected", "passed": False, "failure": type(exc).__name__},
        ])
    for name, override in (
        ("tampered_passage_hash_rejected", {"passage_hash": "0" * 64}),
        ("incomplete_content_rejected", {"content_complete": False}),
    ):
        bad_payload = dict(payload)
        bad_payload.update(override)
        try:
            canonical_evidence_from_point(
                cfg, SimpleNamespace(id="synthetic-point-1", score=1.0, payload=bad_payload),
            )
        except EvidenceContractError:
            checks.append({"name": name, "passed": True})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "failure": type(exc).__name__})
        else:
            checks.append({"name": name, "passed": False, "failure": "accepted_invalid_fixture"})
    return {
        "track": "synthetic_contract_fixture",
        "answer": None,
        "model_used": False,
        "fixture_is_legal_evidence": False,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def run_once(
    rows: Sequence[Mapping[str, Any]],
    *,
    execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    run_number: int,
    corpus_identity: Mapping[str, Any] | None = None,
    config_identity: Mapping[str, Any] | None = None,
    contract_runner: Callable[[], Mapping[str, Any]] | None = run_synthetic_contract_checks,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    validate_question_set(rows)
    results: list[dict[str, Any]] = []
    for question in rows:
        start = clock()
        try:
            raw = execute(question)
            if not isinstance(raw, Mapping):
                raise TypeError("executor result must be a mapping")
            result = {**_default_result(question), **copy.deepcopy(dict(raw))}
            result["question_id"] = question["question_id"]
            result["answer"] = None
        except Exception as exc:  # every failure remains a scored, visible row
            result = _default_result(question)
            result["failure"] = type(exc).__name__
        result["latency_ms"] = round(max(0.0, (clock() - start) * 1000.0), 3)
        result["result_hash"] = stable_result_hash(result)
        results.append(result)
    if contract_runner is None:
        contract = {"track": "synthetic_contract_fixture", "passed": False, "status": "not_run"}
    else:
        try:
            contract = dict(contract_runner())
        except Exception as exc:
            contract = {
                "track": "synthetic_contract_fixture", "passed": False,
                "status": "failed", "failure": type(exc).__name__, "answer": None,
            }
    config = dict(config_identity or {})
    corpus = dict(corpus_identity or {})
    decision_material = {
        "configuration_hash": sha256_bytes(canonical_json_bytes(config)),
        "corpus_identity_hash": sha256_bytes(canonical_json_bytes(corpus)),
        "ordered_results": [
            {"question_id": result["question_id"], "result_hash": result["result_hash"]}
            for result in results
        ],
        "synthetic_contract": _stable_projection(contract),
    }
    run: dict[str, Any] = {
        "schema_version": 1,
        "run_number": run_number,
        "track": "legacy_read_only_retrieval_plus_separate_synthetic_contract",
        "configuration": config,
        "configuration_hash": decision_material["configuration_hash"],
        "corpus_identity": corpus,
        "corpus_identity_hash": decision_material["corpus_identity_hash"],
        "results": results,
        "synthetic_contract": contract,
        "decision_hash": sha256_bytes(canonical_json_bytes(decision_material)),
    }
    run["record_hash"] = sha256_bytes(canonical_json_bytes(run))
    return run


def _normalize_test_evidence(value: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = dict(value or {})
    unknown = sorted(set(supplied) - set(_TEST_EVIDENCE_KEYS))
    if unknown:
        raise ValueError(f"unknown test-evidence keys: {unknown}")
    output: dict[str, Any] = {}
    allowed_fields = {"status", "passed", "failed", "deselected", "summary"}
    for name in _TEST_EVIDENCE_KEYS:
        raw = supplied.get(name, {"status": "not_run"})
        if not isinstance(raw, Mapping) or set(raw) - allowed_fields:
            raise ValueError(f"invalid test-evidence record: {name}")
        status = raw.get("status", "not_run")
        if status not in {"passed", "failed", "interrupted", "not_run"}:
            raise ValueError(f"invalid test-evidence status: {name}")
        row: dict[str, Any] = {"status": status}
        for field in ("passed", "failed", "deselected"):
            if field in raw:
                count = raw[field]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"invalid {field} count for {name}")
                row[field] = count
        if "summary" in raw:
            summary = str(raw["summary"])
            if len(summary) > 300 or "\n" in summary or "\r" in summary:
                raise ValueError(f"unsafe test-evidence summary for {name}")
            row["summary"] = summary
        output[name] = row
    return output


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def build_scorecard(
    rows: Sequence[Mapping[str, Any]],
    run_1: Mapping[str, Any],
    run_2: Mapping[str, Any],
    *,
    question_set_hash: str,
    test_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    first = list(run_1["results"])
    second = list(run_2["results"])
    by_id_2 = {row["question_id"]: row for row in second}
    exact = [row for row in first if row["question_id"] in EXACT_IDS]
    clarify = [row for row in first if row["question_id"] in AMBIGUOUS_IDS]
    incomplete = [row for row in first if row["question_id"] in INCOMPLETE_IDS]
    broad = [row for row in first if row["question_id"] == BROAD_ID]
    repeat_ids = [
        row["question_id"] for row in first
        if row["result_hash"] == by_id_2[row["question_id"]]["result_hash"]
    ]
    latency_by_run = {
        "run_1": [float(row["latency_ms"]) for row in first],
        "run_2": [float(row["latency_ms"]) for row in second],
    }
    retrieval_ids = set(EXACT_IDS + AMBIGUOUS_IDS)
    policy_ids = set(INCOMPLETE_IDS + (BROAD_ID,))
    latency_tracks = {
        "read_only_retrieval": {
            "run_1": [float(row["latency_ms"]) for row in first if row["question_id"] in retrieval_ids],
            "run_2": [float(row["latency_ms"]) for row in second if row["question_id"] in retrieval_ids],
        },
        "policy_only_abstention": {
            "run_1": [float(row["latency_ms"]) for row in first if row["question_id"] in policy_ids],
            "run_2": [float(row["latency_ms"]) for row in second if row["question_id"] in policy_ids],
        },
    }
    degraded = {
        "run_1": [row["question_id"] for row in first if row.get("degraded") or row["outcome"] == "failed"],
        "run_2": [row["question_id"] for row in second if row.get("degraded") or row["outcome"] == "failed"],
    }
    scorecard: dict[str, Any] = {
        "schema_version": 1,
        "positioning": APPROVED_POSITIONING,
        "scope": "Georgian exact legal identifier lookup; clarification and abstention only beyond that narrow contract.",
        "total_questions": len(rows),
        "expected_answer_questions": len(EXACT_IDS),
        "correct_document_identity": sum(bool(row.get("correct_document_identity")) for row in exact),
        "required_evidence_found": sum(bool(row.get("required_evidence_found")) for row in exact),
        "official_url_present_and_matches_frozen": sum(
            bool(row.get("official_url_matches_frozen")) for row in exact
        ),
        "canonical_exact_quotation_validations": {
            "validated": 0,
            "status": LEGACY_UNAVAILABLE,
            "explanation": "Legacy retrieval payloads are not canonical generation evidence.",
        },
        "frozen_snapshot_span_reconstruction": {
            "validated_expected_answer_rows": sum(
                bool((row.get("frozen_snapshot_audit") or {}).get("quote_reconstructed"))
                for row in rows if row["question_id"] in EXACT_IDS
            ),
            "document_content_hash_matches": sum(
                bool((row.get("frozen_snapshot_audit") or {}).get("document_content_hash_matches"))
                for row in rows if row["question_id"] in EXACT_IDS
            ),
            "track": "offline_frozen_snapshot_audit_not_legacy_retrieval_evidence",
        },
        "correct_abstention_or_clarification": {
            "ambiguous": sum(row["outcome"] == "clarify" for row in clarify),
            "incomplete_source": sum(row["outcome"] == "abstain" for row in incomplete),
            "unsupported_broad_advice": sum(row["outcome"] == "abstain" for row in broad),
        },
        "degraded_or_failed_queries": degraded,
        "deterministic_repeats": {
            "matching_questions": len(repeat_ids),
            "total_questions": len(rows),
            "matching_question_ids": repeat_ids,
            "run_decision_hashes_equal": run_1["decision_hash"] == run_2["decision_hash"],
        },
        "latency_ms": {
            name: {
                "median": round(statistics.median(values), 3) if values else None,
                "p95": _percentile(values, 0.95),
            }
            for name, values in latency_by_run.items()
        },
        "latency_tracks_ms": {
            track: {
                run_name: {
                    "median": round(statistics.median(values), 3) if values else None,
                    "p95": _percentile(values, 0.95),
                    "questions": len(values),
                }
                for run_name, values in by_run.items()
            }
            for track, by_run in latency_tracks.items()
        },
        "question_set_sha256": question_set_hash,
        "configuration_hash": run_1["configuration_hash"],
        "corpus_identity_hash": run_1["corpus_identity_hash"],
        "run_hashes": {"run_1": run_1["record_hash"], "run_2": run_2["record_hash"]},
        "run_decision_hashes": {"run_1": run_1["decision_hash"], "run_2": run_2["decision_hash"]},
        "corpus_generation_identity": LEGACY_UNAVAILABLE,
        "synthetic_contract_fixture": {
            "run_1": run_1["synthetic_contract"],
            "run_2": run_2["synthetic_contract"],
            "excluded_from_measured_retrieval_results": True,
        },
        "test_evidence": _normalize_test_evidence(test_evidence),
        "legal_correctness_review": "not yet independently adjudicated",
        "limitations": [
            "The presentation set is small, curated, and not a blind benchmark.",
            "Production severe-error certification is not yet complete.",
            "The legacy collection is mutable and has no sealed immutable generation identity.",
            "The full immutable corpus and lawyer-reviewed v3 release study remain future work.",
            "English is excluded from the live demonstration pending production model attestation.",
        ],
    }
    return scorecard


def _scorecard_markdown(scorecard: Mapping[str, Any]) -> str:
    abstention = scorecard["correct_abstention_or_clarification"]
    repeat = scorecard["deterministic_repeats"]
    latency = scorecard["latency_ms"]
    retrieval_latency = scorecard["latency_tracks_ms"]["read_only_retrieval"]
    degraded = scorecard["degraded_or_failed_queries"]
    tests = scorecard["test_evidence"]
    test_lines = "\n".join(
        f"- `{name}`: {tests[name]['status']}"
        + (
            f" ({tests[name].get('passed', 0)} passed, "
            f"{tests[name].get('failed', 0)} failed, "
            f"{tests[name].get('deselected', 0)} deselected)"
            if "passed" in tests[name] else ""
        )
        + (f" — {tests[name]['summary']}" if tests[name].get("summary") else "")
        for name in _TEST_EVIDENCE_KEYS
    )
    return f"""# Accuracy-first presentation scorecard

> {APPROVED_POSITIONING}

## Measured presentation results

| Measure | Result |
|---|---:|
| Total frozen questions | {scorecard['total_questions']} |
| Expected-answer exact-identity questions | {scorecard['expected_answer_questions']} |
| Correct document identity | {scorecard['correct_document_identity']} / {scorecard['expected_answer_questions']} |
| Required identity evidence found | {scorecard['required_evidence_found']} / {scorecard['expected_answer_questions']} |
| Official URL present and matches frozen source | {scorecard['official_url_present_and_matches_frozen']} / {scorecard['expected_answer_questions']} |
| Ambiguities correctly clarified | {abstention['ambiguous']} / {len(AMBIGUOUS_IDS)} |
| Incomplete sources correctly abstained | {abstention['incomplete_source']} / {len(INCOMPLETE_IDS)} |
| Broad advice correctly abstained | {abstention['unsupported_broad_advice']} / 1 |
| Deterministic result repeats | {repeat['matching_questions']} / {repeat['total_questions']} |

Canonical quotation validation: `{LEGACY_UNAVAILABLE}`. The separate frozen snapshot audit
reconstructed {scorecard['frozen_snapshot_span_reconstruction']['validated_expected_answer_rows']}
of {scorecard['expected_answer_questions']} expected-answer spans and matched
{scorecard['frozen_snapshot_span_reconstruction']['document_content_hash_matches']} frozen
document hashes. That audit is not canonical-generation evidence.

Degraded/failed query IDs: run 1 `{degraded['run_1']}`; run 2 `{degraded['run_2']}`.

Latency: run 1 median {latency['run_1']['median']} ms / p95 {latency['run_1']['p95']} ms;
run 2 median {latency['run_2']['median']} ms / p95 {latency['run_2']['p95']} ms.
Read-only retrieval only: run 1 median {retrieval_latency['run_1']['median']} ms / p95
{retrieval_latency['run_1']['p95']} ms; run 2 median
{retrieval_latency['run_2']['median']} ms / p95 {retrieval_latency['run_2']['p95']} ms.

## Identity and hashes

- Question set SHA-256: `{scorecard['question_set_sha256']}`
- Configuration SHA-256: `{scorecard['configuration_hash']}`
- Corpus observation SHA-256: `{scorecard['corpus_identity_hash']}`
- Run 1 decision SHA-256: `{scorecard['run_decision_hashes']['run_1']}`
- Run 2 decision SHA-256: `{scorecard['run_decision_hashes']['run_2']}`
- Immutable generation identity: `{LEGACY_UNAVAILABLE}`

## Fresh test evidence

{test_lines}

## Interpretation boundary

Retrieval identity, frozen-span integrity, synthetic contract validation, and legal
correctness are separate measures. They are not collapsed into one accuracy figure.

{LEGAL_REVIEW_SENTENCE}

The presentation set is small and not a blind benchmark. Production severe-error
certification is not yet complete. The full immutable corpus and lawyer-reviewed v3 release
study remain future work.
"""


def _demo_script(rows: Sequence[Mapping[str, Any]], run: Mapping[str, Any], scorecard: Mapping[str, Any]) -> str:
    questions = {row["question_id"]: row for row in rows}
    results = {row["question_id"]: row for row in run["results"]}
    blocks: list[str] = []
    for qid in DISPLAY_IDS:
        question = questions[qid]
        result = results[qid]
        audit = result.get("frozen_snapshot_audit") or {}
        blocks.append(
            f"### {qid} — exact identity\n\n"
            f"Question: {question['question']}\n\n"
            f"Outcome: `{result['outcome']}`; result IDs: `{result['ordered_result_ids']}`.\n\n"
            f"Official URL: {result.get('official_url') or 'unavailable'}\n\n"
            f"Frozen snapshot quotation (offline audit, not legacy retrieval evidence):\n\n"
            f"> {audit.get('quotation', 'quotation intentionally omitted')}\n\n"
            f"Frozen quote SHA-256: `{audit.get('quote_sha256')}`; offsets "
            f"`[{audit.get('char_start')}:{audit.get('char_end')}]`.\n\n"
            f"Canonical evidence ID/quotation/passage hash: `{LEGACY_UNAVAILABLE}`."
        )
    ambiguity = results["q121"]
    incomplete = results["q024"]
    return f"""# Reliable demonstration script

This is the static fallback. It reads only persisted bundle results; no model, Qdrant, or
network connection is needed. Show the complete scorecard before or immediately after the
examples so no failure is hidden.

## Opening

{APPROVED_POSITIONING}

## Three predeclared exact-identity examples

{chr(10).join(blocks)}

## Ambiguity

Question `q121` returned `{ambiguity['ordered_result_ids']}` and produced
`{ambiguity['outcome']}` with reason `{ambiguity['reason']}`. No answer text was emitted.

## Incomplete evidence

Question `q024` produced `{incomplete['outcome']}` with reason
`{incomplete['reason']}`. It was stopped by the frozen preflight completeness gate before
retrieval or answer composition; no source passage is displayed.

## Close with the complete scorecard

- Correct document identity: {scorecard['correct_document_identity']} / {len(EXACT_IDS)}
- Deterministic repeats: {scorecard['deterministic_repeats']['matching_questions']} / {len(QUESTION_IDS)}
- Degraded/failed: run 1 `{scorecard['degraded_or_failed_queries']['run_1']}`, run 2 `{scorecard['degraded_or_failed_queries']['run_2']}`
- Question-set SHA-256: `{scorecard['question_set_sha256']}`

{LEGAL_REVIEW_SENTENCE}
"""


def _readme() -> str:
    return f"""# Accuracy-first Georgian legal demonstration

{APPROVED_POSITIONING}

This frozen bundle separates three tracks: model-free read-only exact lookup against a
local legacy collection; offline span/hash checks against frozen snapshots; and synthetic
schema-v2 evidence-contract fixtures. None is described as a production legal-answer run.

## Rehearse offline

From the repository root:

```bash
ingest/.venv/bin/python ingest/scripts/build_presentation_demo.py rehearse \\
  --bundle presentation/accuracy-first-demo
```

The rehearsal verifies all persisted hashes and prints `demo-script.md`. It does not access
Qdrant, load a model, or use the network.

## Verify only

```bash
ingest/.venv/bin/python ingest/scripts/build_presentation_demo.py verify \\
  --bundle presentation/accuracy-first-demo
```

## Track labels

- `legacy_read_only_exact_selector_scroll`: measured exact identifier retrieval only.
- `offline_frozen_snapshot_audit_not_legacy_retrieval_evidence`: local quote/offset/body hash checks.
- `synthetic_contract_fixture`: synthetic evidence-schema rejection/acceptance checks, excluded from retrieval results.
- Production target: a sealed generation with pinned runtime identity, freshness, and release evidence.

{LEGAL_REVIEW_SENTENCE}
"""


def _slides_outline(scorecard: Mapping[str, Any]) -> str:
    return f"""# Six-slide outline

## 1. Legal retrieval errors are high-impact

- Wrong identity, version, or quotation can change the practical meaning.
- The safe response to unsupported evidence is clarification or abstention.

## 2. Accuracy-first architecture

- {APPROVED_POSITIONING}
- Exact selector → identity gate → canonical evidence gate → answer gate.
- Today's live track stops at legacy identity retrieval; the tracks remain visibly separate.

## 3. Canonical evidence and version traceability

- Frozen offsets and hashes make the offline examples reproducible.
- Legacy canonical quotation/evidence IDs are `{LEGACY_UNAVAILABLE}`.
- A sealed immutable generation remains required for canonical production evidence.

## 4. Demonstration

- Exact identity: q169, q291, q329.
- Ambiguity clarification: q121.
- Incomplete-source abstention: q024.
- Static fallback uses persisted results and requires no network or model.

## 5. Measured scorecard and test reliability

- Identity: {scorecard['correct_document_identity']} / {len(EXACT_IDS)}.
- Clarification/abstention reported by category, separately from identity and frozen-span checks.
- Repeat match: {scorecard['deterministic_repeats']['matching_questions']} / {len(QUESTION_IDS)}.
- Small curated presentation set; not a blind benchmark.

## 6. Honest status and roadmap

- Production severe-error certification is not yet complete.
- Full immutable corpus build and lawyer-reviewed v3 release study are future work.
- English live demonstration waits for production model attestation.
- Next milestone: blind legal adjudication against a sealed generation.
"""


def _talk_track(scorecard: Mapping[str, Any]) -> str:
    return f"""# Talk track

Open with: “{APPROVED_POSITIONING}”

Explain that today's numbers measure exact identity retrieval, policy abstention, repeat
determinism, and frozen local integrity as separate properties. The three displayed quotes
come from the frozen snapshot audit and are never presented as canonical evidence returned
by the legacy collection. Show q121 to demonstrate that a non-unique number causes a
clarification, then q024 to demonstrate that incomplete source attestation stops the path
before any answer is emitted.

Close with the hashes and the complete scorecard, including every degraded or failed row.
State that the {scorecard['total_questions']}-question set is curated and small, and that
production severe-error certification, a full immutable corpus, and blind lawyer-reviewed
v3 evaluation remain unfinished.

{LEGAL_REVIEW_SENTENCE}
"""


def _reviewer_checklist() -> str:
    return f"""# Reviewer checklist

- [ ] `questions.sha256` matches `questions.jsonl` before discussing results.
- [ ] Both runs contain all {len(QUESTION_IDS)} IDs in the frozen order.
- [ ] Per-question result hashes and run decision hashes repeat.
- [ ] Exact successes match the annotated source/document identity and include an official URL.
- [ ] q169, q291, and q329 frozen quotes reconstruct from their saved offsets and hashes.
- [ ] q024 contains no quotation, body text, promoted metadata, or party data.
- [ ] q121 emits clarification and q024 emits abstention; neither emits answer text.
- [ ] Canonical legacy fields say `{LEGACY_UNAVAILABLE}`.
- [ ] Synthetic contract fixtures are labelled synthetic and excluded from measured retrieval.
- [ ] q191, q229, and q276 visibly retain their repealed/historical status.
- [ ] Publication metadata dates are not described as act-adoption dates.
- [ ] Test evidence comes from fresh commands or is visibly `not_run`.
- [ ] No private query log, secret, or unnecessary party name is present.

{LEGAL_REVIEW_SENTENCE}
"""


def _limitations() -> str:
    return f"""# Limitations and non-claims

- This is a narrow Georgian exact-identifier presentation, not broad legal advice.
- Production `legal_ask` was not represented as having run unless a separately verified runtime proves it.
- The local legacy collection is not a sealed immutable generation and cannot provide production evidence IDs, canonical quotations, passage hashes, version proof, authority proof, or completeness proof.
- Frozen snapshot verification is an offline integrity audit, not proof that legacy retrieval returned canonical evidence.
- Synthetic contract fixtures contain no legal answer and are not corpus measurements.
- The presentation set is small, curated, and not a blind benchmark.
- Production severe-error certification is not yet complete.
- The full immutable corpus and lawyer-reviewed v3 release study remain future work.
- English is excluded from the live demonstration pending production model attestation.
- A frozen status may be historical or repealed; the demo does not present it as current law.

{LEGAL_REVIEW_SENTENCE}
"""


def _safe_observation(retriever: Any) -> dict[str, Any]:
    try:
        observed = retriever.observe_collection()
        if not isinstance(observed, Mapping):
            raise TypeError("collection observation must be an object")
        return dict(observed)
    except Exception as exc:  # the unavailable state is persisted without leaking details
        return {
            "status": "unavailable",
            "failure": type(exc).__name__,
        }


def _run_decision_material(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "configuration_hash": run["configuration_hash"],
        "corpus_identity_hash": run["corpus_identity_hash"],
        "ordered_results": [
            {"question_id": result["question_id"], "result_hash": result["result_hash"]}
            for result in run["results"]
        ],
        "synthetic_contract": _stable_projection(run["synthetic_contract"]),
    }


def _rehash_run(run: dict[str, Any]) -> None:
    for result in run["results"]:
        result["result_hash"] = stable_result_hash(result)
    run["decision_hash"] = sha256_bytes(
        canonical_json_bytes(_run_decision_material(run))
    )
    material = dict(run)
    material.pop("record_hash", None)
    run["record_hash"] = sha256_bytes(canonical_json_bytes(material))


def _mark_collection_drift(run: dict[str, Any]) -> None:
    run["collection_stable_during_run"] = False
    for result in run["results"]:
        result["degraded"] = True
        if not result.get("failure"):
            result["failure"] = "collection_observation_changed_during_run"
        if result.get("outcome") != "failed":
            result["degraded_reason"] = "collection_observation_changed_during_run"
    _rehash_run(run)


def _validate_holdout(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("holdout file must be a JSON list")
    holdout = {
        (str(item.get("source")), str(item.get("document_id")))
        for item in value
        if isinstance(item, Mapping)
    }
    missing = sorted(
        (str(row["expected_source"]), str(row["expected_document_id"]))
        for row in rows
        if row["question_id"] != BROAD_ID
        and (str(row["expected_source"]), str(row["expected_document_id"])) not in holdout
    )
    if missing:
        raise ValueError(f"selected questions are not held out: {missing}")


def _corpus_identity(
    *,
    golden_set: Path,
    holdout: Path,
    snapshot_manifests: Sequence[Path],
    preflight_manifest: Path,
    collection_observation: Mapping[str, Any],
) -> dict[str, Any]:
    manifests = []
    for path in snapshot_manifests:
        manifest_path = Path(path)
        parent = manifest_path.parent.name
        label = parent if parent != "snapshots" else manifest_path.stem
        manifests.append({"label": label, "sha256": sha256_file(manifest_path)})
    return {
        "kind": "legacy_unsealed_collection_plus_frozen_v2_evidence",
        "golden_set_v2_sha256": sha256_file(Path(golden_set)),
        "holdout_v2_sha256": sha256_file(Path(holdout)),
        "snapshot_manifests": manifests,
        "preflight_manifest_sha256": sha256_file(Path(preflight_manifest)),
        "legacy_collection_observation": dict(collection_observation),
        "immutable_generation_identity": LEGACY_UNAVAILABLE,
    }


def _write_questions(stage: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _write_bytes(stage / "questions.jsonl", payload)
    digest = sha256_bytes(payload)
    _write_text(stage / "questions.sha256", f"{digest}  questions.jsonl")
    return digest


def _narrative_files(
    rows: Sequence[Mapping[str, Any]],
    run_1: Mapping[str, Any],
    scorecard: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "README.md": _readme(),
        "scorecard.md": _scorecard_markdown(scorecard),
        "demo-script.md": _demo_script(rows, run_1, scorecard),
        "slides-outline.md": _slides_outline(scorecard),
        "talk-track.md": _talk_track(scorecard),
        "reviewer-checklist.md": _reviewer_checklist(),
        "limitations.md": _limitations(),
    }


def build_bundle(
    *,
    golden_set: Path,
    holdout: Path,
    snapshot_roots: Sequence[Path],
    snapshot_manifests: Sequence[Path],
    preflight_manifest: Path,
    output: Path,
    qdrant_url: str = "http://127.0.0.1:6333",
    collection: str = "georgian_legal",
    api_key_env: str = "QDRANT_API_KEY",
    test_evidence: Mapping[str, Any] | None = None,
    retriever: Any | None = None,
) -> Path:
    """Create the finalized bundle once; no existing destination is ever replaced."""

    assert_read_only_environment()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    destination = assert_safe_output(Path(output))
    validate_loopback_url(qdrant_url)
    if not _COLLECTION_RE.fullmatch(collection):
        raise DemoSafetyError("unsafe Qdrant collection name")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", api_key_env):
        raise DemoSafetyError("unsafe API-key environment variable name")
    if len(snapshot_roots) < 1 or len(snapshot_manifests) < 1:
        raise ValueError("at least one snapshot root and manifest are required")
    for required in (
        Path(golden_set), Path(holdout), Path(preflight_manifest),
        *(Path(path) for path in snapshot_manifests),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    for root in snapshot_roots:
        if not Path(root).is_dir():
            raise FileNotFoundError(root)

    rows = load_frozen_questions(
        Path(golden_set), tuple(Path(path) for path in snapshot_roots),
        Path(preflight_manifest),
    )
    _validate_holdout(Path(holdout), rows)
    normalized_tests = _normalize_test_evidence(test_evidence)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    stage.mkdir(mode=0o700)
    try:
        # The set and its digest are durably frozen before any retriever is constructed.
        question_set_hash = _write_questions(stage, rows)
        _fsync_file(stage / "questions.jsonl")
        _fsync_file(stage / "questions.sha256")

        active_retriever = retriever
        if active_retriever is None:
            active_retriever = ReadOnlyQdrantRetriever(
                qdrant_url,
                collection,
                api_key=os.environ.get(api_key_env),
            )
        first_before = _safe_observation(active_retriever)
        corpus = _corpus_identity(
            golden_set=Path(golden_set),
            holdout=Path(holdout),
            snapshot_manifests=tuple(Path(path) for path in snapshot_manifests),
            preflight_manifest=Path(preflight_manifest),
            collection_observation=first_before,
        )
        config = {
            "schema_version": 1,
            "question_set_sha256": question_set_hash,
            "collection": collection,
            "qdrant_access": "loopback_read_only_metadata_and_scroll",
            "retrieval_route": "legacy_exact_source_and_identifier_selector",
            "models_loaded": False,
            "english_live_demo": False,
            "current_date": "2026-07-15",
        }

        def executor(question: Mapping[str, Any]) -> dict[str, Any]:
            return evaluate_question(question, active_retriever)

        run_1 = run_once(
            rows, execute=executor, run_number=1,
            corpus_identity=corpus, config_identity=config,
        )
        first_after = _safe_observation(active_retriever)
        run_1["collection_observation_before"] = first_before
        run_1["collection_observation_after"] = first_after
        run_1["collection_stable_during_run"] = first_before == first_after
        if first_before != first_after:
            _mark_collection_drift(run_1)
        else:
            _rehash_run(run_1)

        second_before = _safe_observation(active_retriever)
        run_2 = run_once(
            rows, execute=executor, run_number=2,
            corpus_identity=corpus, config_identity=config,
        )
        second_after = _safe_observation(active_retriever)
        run_2["collection_observation_before"] = second_before
        run_2["collection_observation_after"] = second_after
        run_2["collection_stable_during_run"] = second_before == second_after
        if second_before != second_after or second_before != first_before:
            _mark_collection_drift(run_2)
        else:
            _rehash_run(run_2)

        _write_json(stage / "run-1.json", run_1)
        _write_json(stage / "run-2.json", run_2)
        scorecard = build_scorecard(
            rows, run_1, run_2,
            question_set_hash=question_set_hash,
            test_evidence=normalized_tests,
        )
        _write_json(stage / "scorecard.json", scorecard)
        for name, text in _narrative_files(rows, run_1, scorecard).items():
            _write_text(stage / name, text)

        validate_bundle(stage)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"finalized bundle already exists: {destination}")
        os.chmod(stage, 0o755)
        os.rename(stage, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _validate_run(run: Mapping[str, Any]) -> None:
    results = run.get("results")
    if not isinstance(results, list):
        raise ValueError("run has no result list")
    if tuple(result.get("question_id") for result in results) != QUESTION_IDS:
        raise ValueError("run result ordering differs from frozen question set")
    for result in results:
        if result.get("answer") is not None:
            raise ValueError("presentation run contains unsupported answer text")
        if result.get("result_hash") != stable_result_hash(result):
            raise ValueError(f"result hash mismatch: {result.get('question_id')}")
        for field in (
            "evidence_id", "quotation", "passage_hash", "version_proof",
            "authority_proof", "completeness_proof",
            "legacy_content_hash_verification",
        ):
            if result.get(field) != LEGACY_UNAVAILABLE:
                raise ValueError(f"legacy field is misrepresented: {field}")
    if run.get("decision_hash") != sha256_bytes(
        canonical_json_bytes(_run_decision_material(run))
    ):
        raise ValueError("run decision hash mismatch")
    material = dict(run)
    record_hash = material.pop("record_hash", None)
    if record_hash != sha256_bytes(canonical_json_bytes(material)):
        raise ValueError("run record hash mismatch")


def validate_bundle(bundle: Path) -> dict[str, Any]:
    root = Path(bundle)
    if not root.is_dir():
        raise FileNotFoundError(root)
    actual = {path.name for path in root.iterdir() if path.is_file()}
    required = set(REQUIRED_BUNDLE_FILES)
    if actual != required:
        raise ValueError(
            f"bundle inventory mismatch: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )
    questions = _load_jsonl(root / "questions.jsonl")
    validate_question_set(questions)
    question_hash = sha256_file(root / "questions.jsonl")
    expected_line = f"{question_hash}  questions.jsonl"
    if (root / "questions.sha256").read_text(encoding="utf-8").strip() != expected_line:
        raise ValueError("questions.sha256 does not match questions.jsonl")

    run_1 = json.loads((root / "run-1.json").read_text(encoding="utf-8"))
    run_2 = json.loads((root / "run-2.json").read_text(encoding="utf-8"))
    _validate_run(run_1)
    _validate_run(run_2)
    questions_by_id = {row["question_id"]: row for row in questions}
    for run in (run_1, run_2):
        for result in run["results"]:
            question = questions_by_id[result["question_id"]]
            if result["outcome"] != "retrieval_identity_found":
                continue
            expected_id = (
                f"{question['expected_source']}:{question['expected_document_id']}"
            )
            if (
                question["question_id"] not in EXACT_IDS
                or result.get("ordered_result_ids") != [expected_id]
                or result.get("official_url") != question.get("official_url")
                or result.get("official_url_matches_frozen") is not True
                or result.get("correct_document_identity") is not True
                or result.get("required_evidence_found") is not True
            ):
                raise ValueError(
                    f"unsupported successful identity row: {result['question_id']}"
                )
    scorecard = json.loads((root / "scorecard.json").read_text(encoding="utf-8"))
    if scorecard.get("question_set_sha256") != question_hash:
        raise ValueError("scorecard question-set hash mismatch")
    if scorecard.get("run_hashes") != {
        "run_1": run_1["record_hash"], "run_2": run_2["record_hash"],
    }:
        raise ValueError("scorecard run hashes mismatch")
    if scorecard.get("run_decision_hashes") != {
        "run_1": run_1["decision_hash"], "run_2": run_2["decision_hash"],
    }:
        raise ValueError("scorecard decision hashes mismatch")
    expected_scorecard = build_scorecard(
        questions,
        run_1,
        run_2,
        question_set_hash=question_hash,
        test_evidence=scorecard.get("test_evidence"),
    )
    if canonical_json_bytes(scorecard) != canonical_json_bytes(expected_scorecard):
        raise ValueError("scorecard does not recompute from persisted runs")

    for run in (run_1, run_2):
        serialized = canonical_json_bytes(run)
        for forbidden_key in (b'"body_markdown"', b'"parties"', b'"promoted"', b'"api_key"'):
            if forbidden_key in serialized:
                raise ValueError(f"private or secret field in run: {forbidden_key!r}")
    narrative_names = [
        "README.md", "scorecard.md", "demo-script.md", "slides-outline.md",
        "talk-track.md", "reviewer-checklist.md", "limitations.md",
    ]
    narrative = "\n".join(
        (root / name).read_text(encoding="utf-8") for name in narrative_names
    )
    expected_narratives = _narrative_files(questions, run_1, scorecard)
    for name, expected_text in expected_narratives.items():
        actual_text = (root / name).read_text(encoding="utf-8")
        if actual_text != expected_text.rstrip() + "\n":
            raise ValueError(
                f"generated presentation file was modified: {name}; "
                f"actual={sha256_bytes(actual_text.encode('utf-8'))}, "
                f"expected={sha256_bytes((expected_text.rstrip() + chr(10)).encode('utf-8'))}"
            )
    if APPROVED_POSITIONING not in narrative or LEGAL_REVIEW_SENTENCE not in narrative:
        raise ValueError("required presentation positioning is missing")
    forbidden_claims = (
        "100% accurate", "production ready", "lawyer validated",
        "below 1% severe errors", "full v3 corpus",
    )
    lowered = narrative.casefold()
    if any(claim.casefold() in lowered for claim in forbidden_claims):
        raise ValueError("forbidden presentation claim is present")
    return {
        "bundle": str(root),
        "question_set_sha256": question_hash,
        "run_1_decision_hash": run_1["decision_hash"],
        "run_2_decision_hash": run_2["decision_hash"],
        "deterministic": run_1["decision_hash"] == run_2["decision_hash"],
        "total_questions": len(questions),
    }


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Public offline verification entry point."""

    return validate_bundle(bundle)


def rehearse_bundle(bundle: Path) -> str:
    """Verify first, then return the persisted static demonstration script."""

    validate_bundle(bundle)
    return (Path(bundle) / "demo-script.md").read_text(encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create one finalized presentation bundle")
    build.add_argument("--golden-set", type=Path, required=True)
    build.add_argument("--holdout", type=Path, required=True)
    build.add_argument("--snapshot-root", type=Path, action="append", required=True)
    build.add_argument("--snapshot-manifest", type=Path, action="append", required=True)
    build.add_argument("--preflight-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--qdrant-url", required=True)
    build.add_argument("--collection", required=True)
    build.add_argument("--api-key-env", default="QDRANT_API_KEY")
    build.add_argument(
        "--test-evidence-json", default="{}",
        help="allowlisted JSON object of freshly rerun test/check results",
    )
    verify = sub.add_parser("verify", help="verify an existing bundle without network")
    verify.add_argument("--bundle", type=Path, required=True)
    rehearse = sub.add_parser("rehearse", help="verify and print the static fallback")
    rehearse.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        try:
            test_evidence = json.loads(args.test_evidence_json)
        except json.JSONDecodeError as exc:
            raise SystemExit("--test-evidence-json must be valid JSON") from exc
        if not isinstance(test_evidence, Mapping):
            raise SystemExit("--test-evidence-json must contain a JSON object")
        output = build_bundle(
            golden_set=args.golden_set,
            holdout=args.holdout,
            snapshot_roots=args.snapshot_root,
            snapshot_manifests=args.snapshot_manifest,
            preflight_manifest=args.preflight_manifest,
            output=args.output_dir,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            api_key_env=args.api_key_env,
            test_evidence=test_evidence,
        )
        print(f"created {output}")
        return 0
    if args.command == "verify":
        print(json.dumps(verify_bundle(args.bundle), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "rehearse":
        print(rehearse_bundle(args.bundle), end="")
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
