"""Hermetic preparation/publication tests; fake Qdrant is strictly read-only."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest import chunk_inventory
from ingest import generation_prepare as preparation
from ingest import qdrant_store as store
from ingest.chunking import chunk_document
from ingest.config import RETRIEVAL_FINGERPRINT_REVISION, load_config
from ingest.dedup import content_hash
from ingest.embed_job import snapshot_doc_to_canonical
from ingest.generation import CHECKSUM_FILENAME, GenerationFormatError, GenerationManifest
from ingest.generation_prepare import (
    GenerationPreparationError,
    PREPARATION_PROVENANCE_FILENAME,
    ValidatedPreparationInputs,
    load_prepared_generation,
    prepare_generation,
)
from ingest.generation_snapshot import (
    PROVENANCE_FILENAME,
    GenerationPublishError,
    publish_generation,
    source_state_sha256,
)
from ingest.pipeline import _document_state_hash
from ingest.promotion import SERVING_ALIAS, physical_collection_name
from ingest.snapshot import SNAPSHOT_PIPELINE_VERSION, SOURCES_PRESENT
from scripts import create_generation

GENERATION_ID = "gen-20260715-prepared"
SNAPSHOT_ID = "v3_512_attested_20260715"
LOCK_SHA = "1" * 64
SNAPSHOT_SHA = "2" * 64
CORPUS_SHA = "3" * 64
SOURCE_FINGERPRINT = "4" * 64
REVISION = "a" * 40
BODY = "მუხლი 1. ეს არის ოფიციალური სრული სამართლებრივი ტექსტი."
VERSION_ID = "derived:" + "5" * 64


def _cfg():
    return dataclasses.replace(
        load_config(),
        generation_id=GENERATION_ID,
        collection_name=physical_collection_name(GENERATION_ID),
        embedding_revision=REVISION,
        tokenizer_revision="b" * 40,
        reranker_revision="c" * 40,
        embed_header_v2=False,
    )


def _record() -> dict:
    return {
        "snapshot_version": SNAPSHOT_PIPELINE_VERSION,
        "snapshot_id": SNAPSHOT_ID,
        "doc_id": f"matsne:doc-1:{VERSION_ID}",
        "source": "matsne",
        "document_id": "doc-1",
        "content_hash": content_hash(BODY),
        "title": "კანონი",
        "date": "2026-07-15",
        "date_raw": "2026-07-15",
        "language": "ka",
        "document_type": "law",
        "court": None,
        "source_url": "https://example.invalid/doc-1",
        "source_binary_url": None,
        "document_number": "1",
        "registration_code": None,
        "parties": None,
        "status": "in_force",
        "status_raw": "in_force",
        "in_force_date": "2026-07-15",
        "expiry_date": None,
        "is_consolidated": True,
        "consolidated_count": 1,
        "content_kind": "full_text",
        "content_complete": True,
        "extraction_status": "full_text",
        "article_summary": None,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "normalizer_revision": "canonical-source-v2",
        "version_id": VERSION_ID,
        "version_id_kind": "derived",
        "supersedes": [],
        "effective_from": "2026-07-15",
        "effective_to": None,
        "repeal_date": None,
        "consolidation_status": "current",
        "version_lineage_status": "derived",
        "version_lineage_complete": False,
        "consolidated_dates": [],
        "official_url": "https://example.invalid/doc-1",
        "official_binary_url": None,
        "official_html_url": "https://example.invalid/doc-1",
        "official_pdf_url": None,
        "source_authority": "primary_official",
        "freshness_sla_met": True,
        "admissible": True,
        "page_boundaries": [],
        "page_coordinate_reason": "source_not_paginated",
        "promoted": {},
        "structure": {
            "primary_kind": "article",
            "has_article": True,
            "has_heading": False,
            "has_num_clause": False,
            "article_count": 1,
        },
        "body_char_len": len(BODY),
        "source_run": "run-matsne",
        "body_markdown": BODY,
    }


def _runs() -> list[dict]:
    return [
        {
            "source": source,
            "run_id": f"run-{source}",
            "items": {
                "path": f"{source}/runs/run-{source}/items.jsonl",
                "sha256": hashlib.sha256(f"items:{source}".encode()).hexdigest(),
                "size_bytes": 10,
            },
            "completion_record": {
                "path": f"{source}/runs/run-{source}/run.json",
                "sha256": hashlib.sha256(f"run:{source}".encode()).hexdigest(),
                "size_bytes": 20,
            },
            "completed_at": "2026-07-15T00:00:00Z",
            "success_verified": True,
        }
        for source in sorted(SOURCES_PRESENT)
    ]


def _snapshot(tmp_path: Path, *, raw_line: bytes | None = None) -> Path:
    root = tmp_path / SNAPSHOT_ID
    docs = root / "docs"
    docs.mkdir(parents=True)
    for source in SOURCES_PRESENT:
        value = b"" if source != "matsne" else (
            json.dumps(_record(), ensure_ascii=False) + "\n"
        ).encode()
        (docs / f"{source}.jsonl").write_bytes(value)
    cfg = _cfg()
    chunk_inventory.compute_structural_chunk_inventory(
        docs,
        sources=SOURCES_PRESENT,
        tokenizer_model=cfg.tokenizer_model,
        tokenizer_revision=cfg.tokenizer_revision,
        max_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        document_header=cfg.embed_header_v2,
        count_tokens=lambda _text: 10,
        output=root / chunk_inventory.CHUNK_INVENTORY_FILENAME,
    )
    if raw_line is not None:
        (docs / "matsne.jsonl").write_bytes(raw_line)
    return root


def _inputs(snapshot_root: Path) -> ValidatedPreparationInputs:
    files = [
        {
            "path": f"docs/{source}.jsonl",
            "sha256": hashlib.sha256(
                (snapshot_root / "docs" / f"{source}.jsonl").read_bytes()
            ).hexdigest(),
            "size_bytes": (snapshot_root / "docs" / f"{source}.jsonl").stat().st_size,
        }
        for source in sorted(SOURCES_PRESENT)
    ]
    inventory_path = snapshot_root / chunk_inventory.CHUNK_INVENTORY_FILENAME
    files.append(
        {
            "path": chunk_inventory.CHUNK_INVENTORY_FILENAME,
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "size_bytes": inventory_path.stat().st_size,
        }
    )
    cfg = _cfg()
    inventory_identity = chunk_inventory.chunk_inventory_identity(
        sources=SOURCES_PRESENT,
        tokenizer_model=cfg.tokenizer_model,
        tokenizer_revision=cfg.tokenizer_revision,
        max_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        document_header=cfg.embed_header_v2,
    )
    structural_inventory = {
        "schema_version": chunk_inventory.CHUNK_INVENTORY_SCHEMA_VERSION,
        "status": chunk_inventory.CHUNK_INVENTORY_STATUS_AVAILABLE,
        "reason": None,
        "format": chunk_inventory.CHUNK_INVENTORY_FORMAT,
        "path": chunk_inventory.CHUNK_INVENTORY_FILENAME,
        "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "size_bytes": inventory_path.stat().st_size,
        "identity": inventory_identity,
        "identity_sha256": chunk_inventory.unavailable_manifest_entry(
            inventory_identity
        )["identity_sha256"],
        "record_count": 2,
        "document_count": 1,
        "chunk_count": 1,
        "source_counts": {
            source: {
                "document_count": 1 if source == "matsne" else 0,
                "chunk_count": 1 if source == "matsne" else 0,
            }
            for source in SOURCES_PRESENT
        },
    }
    manifest = {
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_sha256": SNAPSHOT_SHA,
        "corpus_sha256": CORPUS_SHA,
        "build": {
            "sources": list(SOURCES_PRESENT),
            "embed_model": _cfg().embed_model,
            "embedding_revision": _cfg().embedding_revision,
            "tokenizer": {
                "model": _cfg().tokenizer_model,
                "revision": _cfg().tokenizer_revision,
            },
            "chunk": {
                "tokens": _cfg().chunk_tokens,
                "overlap": _cfg().chunk_overlap,
                "min_tokens": _cfg().chunk_min_tokens,
            },
            "document_header": cfg.embed_header_v2,
        },
        "structural_chunk_inventory": structural_inventory,
        "source_state_evidence": {"sha256": "e" * 64, "size_bytes": 100},
        "runs": _runs(),
        "files": files,
        "totals": {"clean": 1, "quarantined": 2, "malformed": 0},
    }
    runtime = {
        "status": "validated",
        "requirements_lock_sha256": LOCK_SHA,
    }
    collection_configuration = store.collection_configuration(_info(_cfg()))
    collection_configuration_sha = store.collection_configuration_sha256(
        collection_configuration
    )
    return ValidatedPreparationInputs(
        generation_id=GENERATION_ID,
        snapshot_root=snapshot_root,
        snapshot_manifest=manifest,
        physical_collection=physical_collection_name(GENERATION_ID),
        dependency_lock_sha256=LOCK_SHA,
        runtime_identity=runtime,
        runtime_identity_sha256=hashlib.sha256(
            json.dumps(runtime).encode()
        ).hexdigest(),
        image_digest="sha256:" + "6" * 64,
        code_identity={"git_sha": "d" * 40, "dirty_patch_sha256": None},
        embed_binding={
            "snapshot": {
                "snapshot_id": SNAPSHOT_ID,
                "snapshot_sha256": SNAPSHOT_SHA,
                "corpus_sha256": CORPUS_SHA,
                "structural_chunk_inventory": {
                    field: structural_inventory[field]
                    for field in (
                        "sha256",
                        "size_bytes",
                        "identity_sha256",
                        "record_count",
                        "document_count",
                        "chunk_count",
                    )
                },
            },
            "collection_configuration": {
                "sha256": collection_configuration_sha,
                "value": collection_configuration,
            }
        },
        embed_binding_sha256="7" * 64,
        vector_checksum_artifact_sha256="8" * 64,
        vector_probe_sha256="9" * 64,
    )


def _point(cfg, *, payload_overrides: dict | None = None):
    record = _record()
    doc = snapshot_doc_to_canonical(record, strict=True)
    chunks = chunk_document(
        BODY,
        max_tokens=cfg.chunk_tokens,
        overlap=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        count_tokens=lambda _text: 10,
        page_boundaries=doc.page_boundaries,
        page_coordinate_reason=doc.page_coordinate_reason,
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    payload = store.build_payload(
        doc,
        chunk,
        document_chunk_count=1,
        document_state_hash=_document_state_hash(cfg, doc=doc),
        cfg=cfg,
    )
    payload.update(payload_overrides or {})
    return SimpleNamespace(
        id=store.point_id("matsne", "doc-1", 0, version_id=VERSION_ID),
        payload=payload,
        vector={
            "dense": [0.0] * cfg.dense_dim,
            "sparse": SimpleNamespace(indices=[1, 3], values=[0.5, 0.25]),
        },
    )


def _info(cfg, points_count: int = 1):
    return SimpleNamespace(
        status="green",
        points_count=points_count,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    "dense": SimpleNamespace(size=cfg.dense_dim, distance="cosine")
                },
                sparse_vectors={"sparse": SimpleNamespace()},
            )
        ),
        payload_schema={},
    )


class _Client:
    def __init__(self, cfg, points=None, *, pages=None, counts=None):
        self.cfg = cfg
        self.points = [_point(cfg)] if points is None else points
        self.pages = pages
        self.counts = list(counts or [len(self.points), len(self.points)])
        self.scroll_calls = 0

    def get_collection(self, collection_name):
        assert collection_name == physical_collection_name(GENERATION_ID)
        return _info(self.cfg, self.counts.pop(0))

    def scroll(self, **kwargs):
        assert kwargs["collection_name"] == physical_collection_name(GENERATION_ID)
        assert kwargs["with_vectors"] is True
        if self.pages is not None:
            page = self.pages[self.scroll_calls]
            self.scroll_calls += 1
            return page
        return self.points, None


def _prepare(tmp_path: Path, *, client=None) -> tuple[Path, object]:
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path)
    client = client or _Client(cfg)
    output = tmp_path / "prepared"
    destination = prepare_generation(
        client,
        cfg,
        generation_id=GENERATION_ID,
        snapshot_root=snapshot_root,
        physical_collection=physical_collection_name(GENERATION_ID),
        dependency_lock=tmp_path / "unused.lock",
        runtime_identity=tmp_path / "unused-runtime.json",
        image_digest="sha256:" + "6" * 64,
        actor="pytest",
        run_id="prepare-run-1",
        output_dir=output,
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
        validated_inputs=_inputs(snapshot_root),
    )
    return destination, cfg


def test_prepare_scans_exact_physical_and_seals_attested_ledgers(tmp_path):
    destination, _cfg_value = _prepare(tmp_path)
    prepared = load_prepared_generation(destination)

    assert prepared.manifest.document_count == 1
    assert prepared.manifest.chunk_count == 1
    assert prepared.manifest.retrieval_fingerprint_revision == 2
    assert prepared.manifest.excluded_document_count == 0
    assert prepared.provenance["serving_collection"] == SERVING_ALIAS
    assert prepared.provenance["physical_collection"] == physical_collection_name(
        GENERATION_ID
    )
    assert prepared.provenance["queried_collection"] == physical_collection_name(
        GENERATION_ID
    )
    assert prepared.provenance["collection_access_kind"] == "direct_physical"
    assert {run.source for run in prepared.manifest.covered_runs} == set(
        SOURCES_PRESENT
    )
    assert [sample.chunk_index for sample in prepared.iter_samples()] == [0]
    assert prepared.collection_digest.point_count == prepared.manifest.chunk_count
    assert (
        prepared.collection_digest.collection_sha256
        == prepared.provenance["whole_collection_sha256"]
    )
    assert (
        prepared.collection_digest.collection_configuration_sha256
        == prepared.provenance["collection_configuration_sha256"]
    )
    assert prepared.collection_digest.vector_checksum_artifact_sha256 == "8" * 64
    assert prepared.collection_digest.vector_probe_sha256 == "9" * 64


def test_prepare_rejects_snapshot_embed_binding_chunk_inventory_mismatch(tmp_path):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path)
    inputs = _inputs(snapshot_root)
    binding = json.loads(json.dumps(inputs.embed_binding))
    binding["snapshot"]["structural_chunk_inventory"]["sha256"] = "0" * 64
    drifted = dataclasses.replace(inputs, embed_binding=binding)

    class NoCollectionAccess:
        def get_collection(self, _name):
            pytest.fail("inventory mismatch must abort before collection access")

    with pytest.raises(
        GenerationPreparationError,
        match="embed binding structural chunk inventory mismatch",
    ):
        prepare_generation(
            NoCollectionAccess(),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused-runtime.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-inventory-mismatch",
            output_dir=tmp_path / "prepared-mismatch",
            validated_inputs=drifted,
        )


def test_prepared_loader_rejects_chunk_inventory_provenance_tamper(tmp_path):
    destination, _cfg_value = _prepare(tmp_path)
    provenance_path = destination / PREPARATION_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["snapshot"]["structural_chunk_inventory"]["sha256"] = "0" * 64
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums_path = destination / CHECKSUM_FILENAME
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["files"][PREPARATION_PROVENANCE_FILENAME] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    checksums_path.write_text(
        json.dumps(checksums, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GenerationPreparationError,
        match="preparation provenance snapshot mismatch",
    ):
        load_prepared_generation(destination)


def test_prepare_rejects_point_identity_and_structural_revision_drift(tmp_path):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path)
    point = _point(cfg, payload_overrides={"chunker_revision": "old-revision"})
    with pytest.raises(GenerationPreparationError, match="chunker revision"):
        prepare_generation(
            _Client(cfg, [point]),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "prepared",
            validated_inputs=_inputs(snapshot_root),
        )


@pytest.mark.parametrize("tamper", ["boundary", "structure", "header"])
def test_prepare_rejects_exact_inventory_boundary_structure_or_header_drift(
    tmp_path, tamper
):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path)
    point = _point(cfg)
    if tamper == "structure":
        point.payload["article_label"] = "მუხლი 999. შეცვლილი"
    elif tamper == "header":
        point.payload["title"] = "შეცვლილი კანონი"
    else:
        shortened = BODY[:-1]
        passage_sha = hashlib.sha256(shortened.encode("utf-8")).hexdigest()
        point.payload.update(
            {
                "text": shortened,
                "char_end": len(shortened),
                "passage_hash": passage_sha,
                "passage_content_hash": passage_sha,
                "passage_id": "passage:"
                + hashlib.sha256(
                    (
                        f"matsne\0doc-1\0{VERSION_ID}\0{0}\0{len(shortened)}\0"
                        f"{passage_sha}"
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    with pytest.raises(GenerationPreparationError, match="identity mismatch"):
        prepare_generation(
            _Client(cfg, [point]),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id=f"prepare-{tamper}-drift",
            output_dir=tmp_path / "prepared",
            validated_inputs=_inputs(snapshot_root),
        )


def test_exact_join_rejects_swapped_per_document_counts_with_same_global_total(tmp_path):
    connection = preparation._create_ledger(tmp_path / "ledger.sqlite")
    expected_template = (
        "run",
        "source-id",
        "a" * 64,
        "b" * 64,
        "full_text",
        "full_text",
        None,
        None,
        "{}",
    )
    for document_id in ("doc-a", "doc-b"):
        connection.execute(
            "INSERT INTO expected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("matsne", document_id, "v1", *expected_template),
        )
    chunk_tail = (
        "c" * 64,
        1,
        1,
        1,
        0,
        1,
        "{}",
        "{}",
        "{}",
    )
    connection.execute(
        "INSERT INTO expected_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("matsne", "doc-a", "v1", 0, 1, *chunk_tail),
    )
    for index in range(2):
        connection.execute(
            "INSERT INTO expected_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("matsne", "doc-b", "v1", index, 2, *chunk_tail),
        )
    # Same global total (3), but A claims/contains two and B only one.
    for document_id, indices, declared in (
        ("doc-a", range(2), 2),
        ("doc-b", range(1), 1),
    ):
        for index in indices:
            connection.execute(
                "INSERT INTO observed VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "matsne",
                    document_id,
                    "v1",
                    index,
                    f"{document_id}-{index}",
                    "d" * 64,
                    declared,
                    "e" * 64,
                ),
            )
    with pytest.raises(GenerationPreparationError, match="chunk accounting mismatch"):
        preparation._validate_join(connection)
    connection.close()


def test_prepare_rejects_missing_vectors_and_ambiguous_collection_count(tmp_path):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path / "vectors")
    point = _point(cfg)
    point.vector = {"dense": point.vector["dense"]}
    with pytest.raises(GenerationPreparationError, match="dense/sparse named vectors"):
        prepare_generation(
            _Client(cfg, [point]),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "vectors-prepared",
            validated_inputs=_inputs(snapshot_root),
        )

    snapshot_root = _snapshot(tmp_path / "count")
    with pytest.raises(GenerationPreparationError, match="points_count is invalid"):
        prepare_generation(
            _Client(cfg, counts=[None]),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "count-prepared",
            validated_inputs=_inputs(snapshot_root),
        )


def test_prepare_rejects_duplicate_snapshot_json_before_collection_access(tmp_path):
    cfg = _cfg()
    record = json.dumps(_record(), ensure_ascii=False)
    duplicate = record[:-1] + ',"source":"matsne"}\n'
    snapshot_root = _snapshot(tmp_path, raw_line=duplicate.encode())

    class NoAccess(_Client):
        def get_collection(self, collection_name):
            raise AssertionError("collection accessed before strict snapshot conversion")

    with pytest.raises(GenerationPreparationError, match="duplicate JSON key"):
        prepare_generation(
            NoAccess(cfg),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "prepared",
            validated_inputs=_inputs(snapshot_root),
        )


def test_prepare_rejects_snapshot_build_config_drift_before_qdrant(tmp_path):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path)
    inputs = _inputs(snapshot_root)
    manifest = dict(inputs.snapshot_manifest)
    manifest["build"] = dict(manifest["build"])
    manifest["build"]["chunk"] = dict(manifest["build"]["chunk"])
    manifest["build"]["chunk"]["tokens"] = 513
    drifted = dataclasses.replace(inputs, snapshot_manifest=manifest)

    class NoAccess(_Client):
        def get_collection(self, collection_name):
            raise AssertionError("Qdrant accessed before snapshot/config drift rejection")

    with pytest.raises(GenerationPreparationError, match="configuration mismatch"):
        prepare_generation(
            NoAccess(cfg),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "prepared",
            validated_inputs=drifted,
        )


def test_prepare_rejects_empty_advancing_page_and_collection_drift(tmp_path):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path / "empty")
    with pytest.raises(GenerationPreparationError, match="empty page"):
        prepare_generation(
            _Client(cfg, pages=[([], "next")], counts=[1]),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "empty-prepared",
            validated_inputs=_inputs(snapshot_root),
        )

    snapshot_root = _snapshot(tmp_path / "drift")
    with pytest.raises(GenerationPreparationError, match="changed during"):
        prepare_generation(
            _Client(cfg, counts=[1, 2]),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=tmp_path / "drift-prepared",
            validated_inputs=_inputs(snapshot_root),
        )


def test_prepared_loader_rejects_checksum_tamper_and_extra_directory(tmp_path):
    destination, _cfg_value = _prepare(tmp_path)
    (destination / "extra").mkdir()
    with pytest.raises(GenerationPreparationError, match="unexplained filesystem"):
        load_prepared_generation(destination)

    (destination / "extra").rmdir()
    (destination / "documents.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(GenerationPreparationError, match="checksum mismatch"):
        load_prepared_generation(destination)


def test_prepared_loader_rejects_checksum_valid_placeholder_source_state(tmp_path):
    destination, _cfg_value = _prepare(tmp_path)
    placeholder = {
        "derived_from_collection": physical_collection_name(GENERATION_ID),
        "note": "not scraper attestation",
    }
    source_path = destination / "source_state.json"
    manifest_path = destination / "manifest.json"
    source_path.write_text(json.dumps(placeholder, sort_keys=True) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["state_sha256"] = source_state_sha256(placeholder)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    checksums_path = destination / CHECKSUM_FILENAME
    checksums = json.loads(checksums_path.read_text())
    checksums["files"]["source_state.json"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    checksums["files"]["manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

    with pytest.raises(GenerationPreparationError, match="source_state has invalid keys"):
        load_prepared_generation(destination)


def test_prepare_and_loader_reject_symlink_path_components(tmp_path):
    cfg = _cfg()
    snapshot_root = _snapshot(tmp_path / "snapshot")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(GenerationPreparationError, match="symlink path component"):
        prepare_generation(
            _Client(cfg),
            cfg,
            generation_id=GENERATION_ID,
            snapshot_root=snapshot_root,
            physical_collection=physical_collection_name(GENERATION_ID),
            dependency_lock=tmp_path / "unused.lock",
            runtime_identity=tmp_path / "unused.json",
            image_digest="sha256:" + "6" * 64,
            actor="pytest",
            run_id="prepare-run-1",
            output_dir=linked_parent / "prepared",
            validated_inputs=_inputs(snapshot_root),
        )

    destination, _cfg_value = _prepare(tmp_path / "published")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(destination, target_is_directory=True)
    with pytest.raises(GenerationPreparationError, match="symlink path component"):
        load_prepared_generation(linked_root)


def test_create_generation_accepts_only_sealed_prepared_directory(
    tmp_path, monkeypatch, capsys
):
    destination, cfg = _prepare(tmp_path)
    monkeypatch.setattr(create_generation, "load_config", lambda: cfg)
    output_root = tmp_path / "generations"

    assert create_generation.main(
        [
            "--generation",
            GENERATION_ID,
            "--prepared-dir",
            str(destination),
            "--output-root",
            str(output_root),
        ]
    ) == 0
    published = output_root / GENERATION_ID
    provenance = json.loads((published / PROVENANCE_FILENAME).read_text())
    evidence = provenance["preparation"]["evidence"]
    assert evidence["serving_collection"] == SERVING_ALIAS
    assert evidence["physical_collection"] == physical_collection_name(GENERATION_ID)
    assert provenance["preparation"]["prepared_checksums_sha256"] == hashlib.sha256(
        (destination / CHECKSUM_FILENAME).read_bytes()
    ).hexdigest()
    assert capsys.readouterr().out.strip() == str(published)

    with pytest.raises(SystemExit):
        create_generation._parser().parse_args(
            [
                "--generation",
                GENERATION_ID,
                "--manifest",
                str(destination / "manifest.json"),
                "--output-root",
                str(output_root),
            ]
        )


def test_publication_streams_checksum_bound_prepared_ledgers(tmp_path):
    destination, _cfg_value = _prepare(tmp_path)
    prepared = load_prepared_generation(destination)
    document_path = destination / "documents.jsonl"
    changed = json.loads(document_path.read_text())
    changed["content_hash"] = "f" * 64
    document_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    output_root = tmp_path / "generations"

    with pytest.raises(GenerationPublishError, match="checksum mismatch"):
        publish_generation(
            output_root,
            GENERATION_ID,
            prepared.manifest,
            prepared.iter_documents(),
            prepared.iter_samples(),
            prepared.source_state,
            collection_digest=prepared.collection_digest,
        )
    assert not (output_root / GENERATION_ID).exists()


def test_generation_manifest_rejects_earlier_retrieval_revision():
    data = {
        "schema_version": 2,
        "generation_id": GENERATION_ID,
        "document_count": 0,
        "indexed_document_count": 0,
        "excluded_document_count": 0,
        "chunk_count": 0,
        "sample_count": 0,
        "corpus": {"name": SNAPSHOT_ID, "snapshot_sha256": SNAPSHOT_SHA},
        "source": {"name": "source", "state_sha256": "7" * 64},
        "model": {
            "embedding_model": "embed",
            "embedding_revision": REVISION,
            "tokenizer_model": "tokenizer",
            "tokenizer_revision": "b" * 40,
            "reranker_model": "reranker",
            "reranker_revision": "c" * 40,
        },
        "vector_space": {
            "id": "8" * 64,
            "dense_name": "dense",
            "dense_dimension": 1,
            "distance": "cosine",
            "sparse_name": "sparse",
        },
        "chunking": {
            "fingerprint": "9" * 64,
            "max_tokens": 2,
            "overlap_tokens": 1,
            "document_header": False,
        },
        "covered_runs": [],
        "retrieval_fingerprint_revision": RETRIEVAL_FINGERPRINT_REVISION - 1,
        "retrieval_fingerprint": "a" * 64,
        "code": {"git_sha": "d" * 40, "dirty_patch_sha256": None},
        "dependency": {"lock_sha256": LOCK_SHA, "image_digest": None},
        "creation": {
            "created_at": "2026-07-15T00:00:00Z",
            "run_id": "run",
            "actor": "pytest",
        },
    }
    with pytest.raises(GenerationFormatError, match="retrieval_fingerprint_revision"):
        GenerationManifest.from_dict(data)
