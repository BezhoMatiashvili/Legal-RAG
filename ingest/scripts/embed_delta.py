#!/usr/bin/env python
"""Validate, size, and embed an explicit raw-item delta into Qdrant.

Raw ``items.jsonl`` records are normalized through the source registry, cleaned and
chunked through the same functions as the normal ingest pipeline, then embedded with
BGE-M3.  ``--dry-run`` performs the exact tokenizer/chunker pass without loading the
embedding model.  ``--manifest-out`` makes that result machine-readable so the RunPod
orchestrator can prove that the pod embedded precisely the host-validated document set.

Examples (from ``ingest/``)::

    .venv/bin/python scripts/embed_delta.py --source supremecourt \
        --items ../artifacts/supremecourt/runs/<run>/items.jsonl --dry-run --strict
    EMBED_DEVICE=cuda EMBED_USE_FP16=true .venv/bin/python scripts/embed_delta.py \
        --source supremecourt --items /workspace/delta_items/*.jsonl \
        --collection georgian_legal_delta_supremecourt_<run_id> --batch-size 256 --strict

The legacy Matsne ``--runs-since`` path and the per-source ``--items-dir`` path remain
available.  Point ids stay deterministic UUIDv5 values, so retrying a run is idempotent.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # ingest/ root -> import ingest

from ingest.config import load_config  # noqa: E402
from ingest.operational import (  # noqa: E402
    QDRANT_WRITE_APPROVAL_ENV,
    require_explicit_approval,
    require_run_scoped_delta_collection,
)
from ingest.sources import CanonicalDoc, SOURCES, normalize  # noqa: E402

MANIFEST_SCHEMA = 1
SUPREMECOURT_CHAMBERS = {
    "ადმინისტრაციულ საქმეთა პალატა",
    "სამოქალაქო საქმეთა პალატა",
    "სისხლის სამართლის საქმეთა პალატა",
}
_EXPECTED_FIELDS = (
    "source",
    "input_sha256",
    "document_ids_sha256",
    "documents",
    "chunks",
    "skipped",
)
_DEDUP_CACHE_KIB = 4 * 1024


class _SQLiteDocuments:
    """Temporary external-state map preserving Python dict replacement order.

    The prior ``dict[document_id, CanonicalDoc]`` retained every normalized body in
    memory.  A delta can contain many large revisions of the same documents, so keep
    only their current canonical representation in SQLite and stream final rows in
    first-seen order.  SQLite's page cache is explicitly bounded; the temporary
    directory and database are owner-only and are removed by :meth:`close`.
    """

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="legal-delta-dedup-")
        temporary_path = Path(self._temporary.name)
        temporary_path.chmod(0o700)
        self.storage_path = temporary_path / "documents.sqlite3"
        self._connection = sqlite3.connect(self.storage_path)
        self.storage_path.chmod(0o600)
        self._connection.executescript(
            f"""
            PRAGMA cache_size = -{_DEDUP_CACHE_KIB};
            PRAGMA mmap_size = 0;
            PRAGMA temp_store = FILE;
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                first_seen INTEGER NOT NULL UNIQUE,
                canonical_json TEXT NOT NULL,
                eligible INTEGER NOT NULL DEFAULT 1 CHECK (eligible IN (0, 1))
            );
            """
        )
        self._closed = False

    @staticmethod
    def _serialize(doc: CanonicalDoc) -> str:
        return json.dumps(
            dataclasses.asdict(doc),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def store_last(self, doc: CanonicalDoc, first_seen: int) -> None:
        """Insert a document or replace its value without moving its first position."""
        self._connection.execute(
            """
            INSERT INTO documents (document_id, first_seen, canonical_json)
            VALUES (?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                canonical_json = excluded.canonical_json
            """,
            (doc.document_id, first_seen, self._serialize(doc)),
        )

    def store_first(self, doc: CanonicalDoc, first_seen: int) -> bool:
        """Insert only a new identity, returning false for an existing identity."""
        cursor = self._connection.execute(
            """
            INSERT INTO documents (document_id, first_seen, canonical_json)
            VALUES (?, ?, ?)
            ON CONFLICT(document_id) DO NOTHING
            """,
            (doc.document_id, first_seen, self._serialize(doc)),
        )
        return cursor.rowcount == 1

    def finish(self) -> None:
        self._connection.commit()

    def exclude(self, document_ids: list[str]) -> None:
        """Exclude chunk-invalid documents from subsequent streaming passes."""
        self._connection.executemany(
            "UPDATE documents SET eligible = 0 WHERE document_id = ?",
            ((document_id,) for document_id in document_ids),
        )

    def items(self) -> Iterator[tuple[str, CanonicalDoc]]:
        """Yield canonical documents one at a time in original dict iteration order."""
        cursor = self._connection.execute(
            """
            SELECT document_id, canonical_json
            FROM documents
            WHERE eligible = 1
            ORDER BY first_seen
            """
        )
        try:
            for document_id, canonical_json in cursor:
                yield document_id, CanonicalDoc(**json.loads(canonical_json))
        finally:
            cursor.close()

    def __len__(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM documents WHERE eligible = 1"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.close()
        finally:
            self._temporary.cleanup()

    def __enter__(self) -> _SQLiteDocuments:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class _SQLiteDocumentCollection:
    """A re-iterable stream over per-source external document stores."""

    def __init__(self, stores: list[_SQLiteDocuments]) -> None:
        self._stores = stores
        self._closed = False

    def __iter__(self) -> Iterator[CanonicalDoc]:
        for store in self._stores:
            for _document_id, doc in store.items():
                yield doc

    def __len__(self) -> int:
        return sum(len(store) for store in self._stores)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for store in self._stores:
            store.close()

    def __enter__(self) -> _SQLiteDocumentCollection:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_paths(cfg, args) -> list[Path]:
    if args.items:
        return [Path(x) for x in args.items]
    runs_dir = cfg.artifacts_root / "matsne" / "runs"
    paths = sorted(runs_dir.glob("*/items.jsonl"))
    if args.runs_since:
        paths = [p for p in paths if p.parent.name >= args.runs_since]
    return paths


def _input_sha256(by_source: list[tuple[str, list[Path]]]) -> str:
    """Hash ordered source labels + exact file bytes, independent of absolute paths."""
    h = hashlib.sha256()
    for source, paths in by_source:
        source_bytes = source.encode("utf-8")
        h.update(len(source_bytes).to_bytes(8, "big"))
        h.update(source_bytes)
        for path in paths:
            data = Path(path).read_bytes()
            h.update(len(data).to_bytes(8, "big"))
            h.update(data)
    return h.hexdigest()


def _load_items(source: str, paths: list[Path]) -> tuple[_SQLiteDocuments, list[dict]]:
    """Normalize into bounded external dedup state and retain validation failures."""
    spec = SOURCES.get(source)
    if spec is None:
        raise ValueError(
            f"unknown source {source!r}; expected one of {sorted(SOURCES)}"
        )

    docs = _SQLiteDocuments()
    failures: list[dict] = []
    input_position = 0
    try:
        for path in paths:
            with Path(path).open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError("JSON value is not an object")
                        missing = [
                            field
                            for field in spec.id_fields
                            if item.get(field) in (None, "")
                        ]
                        if missing:
                            raise ValueError(
                                f"missing identity field(s): {', '.join(missing)}"
                            )
                        if (
                            source == "supremecourt"
                            and item.get("chamber") not in SUPREMECOURT_CHAMBERS
                        ):
                            raise ValueError(
                                "unofficial Supreme Court chamber: "
                                f"{item.get('chamber')!r}"
                            )
                        doc = normalize(source, item)
                    except Exception as exc:  # noqa: BLE001 - strict manifest records all
                        failures.append(
                            {
                                "source": source,
                                "file": Path(path).name,
                                "line": line_no,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue

                    if source == "supremecourt":
                        inserted = docs.store_first(doc, input_position)
                        if not inserted:
                            failures.append(
                                {
                                    "source": source,
                                    "file": Path(path).name,
                                    "line": line_no,
                                    "reason": (
                                        "duplicate Supreme Court identity: "
                                        f"{doc.document_id}"
                                    ),
                                }
                            )
                    else:
                        docs.store_last(doc, input_position)
                    input_position += 1
        docs.finish()
        return docs, failures
    except BaseException:
        docs.close()
        raise


def _analyze_docs(
    cfg, by_source: list[tuple[str, list[Path]]], count_tokens
) -> tuple[_SQLiteDocumentCollection, dict]:
    """Return a bounded document stream plus the exact input identity/chunk manifest."""
    from ingest.chunking import chunk_document
    from ingest.pipeline import _prepare_doc_for_index

    document_stores: list[_SQLiteDocuments] = []
    input_failures: list[dict] = []
    chunk_failures: list[dict] = []
    chunks_total = 0
    per_source: dict[str, dict[str, int]] = {}

    try:
        for source, paths in by_source:
            source_docs, failures = _load_items(source, paths)
            document_stores.append(source_docs)
            input_failures.extend(failures)
            source_chunks = 0
            source_good = 0
            excluded_ids: list[str] = []
            for document_id, doc in source_docs.items():
                try:
                    prepared = _prepare_doc_for_index(doc)
                    chunks = chunk_document(
                        prepared.body_markdown,
                        max_tokens=cfg.chunk_tokens,
                        overlap=cfg.chunk_overlap,
                        min_tokens=cfg.chunk_min_tokens,
                        count_tokens=count_tokens,
                    )
                    if not chunks:
                        raise ValueError("body produced no chunks")
                except Exception as exc:  # noqa: BLE001 - strict manifest records all
                    chunk_failures.append(
                        {
                            "source": source,
                            "document_id": document_id,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    excluded_ids.append(document_id)
                    continue
                source_good += 1
                source_chunks += len(chunks)
            source_docs.exclude(excluded_ids)
            source_docs.finish()
            chunks_total += source_chunks
            per_source[source] = {
                "documents": source_good,
                "chunks": source_chunks,
                "skipped": len(failures) + len(excluded_ids),
            }

        docs = _SQLiteDocumentCollection(document_stores)
        document_keys = sorted(f"{doc.source}\t{doc.document_id}" for doc in docs)
        ids_blob = "\n".join(document_keys).encode("utf-8")
        failures = [*input_failures, *chunk_failures]
        source_value = by_source[0][0] if len(by_source) == 1 else "multi"
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "kind": "delta_input",
            "source": source_value,
            "sources": per_source,
            "input_files": [
                Path(path).name for _, paths in by_source for path in paths
            ],
            "input_sha256": _input_sha256(by_source),
            "document_ids": document_keys,
            "document_ids_sha256": hashlib.sha256(ids_blob).hexdigest(),
            "documents": len(docs),
            "chunks": chunks_total,
            "skipped": len(failures),
            "failures": failures,
        }
        return docs, manifest
    except BaseException:
        for store in document_stores:
            store.close()
        raise


def validate_expected_manifest(expected: dict, actual: dict) -> None:
    """Fail when host and pod disagree on any input identity/count invariant."""
    errors = [
        f"{field}: expected {expected.get(field)!r}, got {actual.get(field)!r}"
        for field in _EXPECTED_FIELDS
        if expected.get(field) != actual.get(field)
    ]
    if expected.get("document_ids") != actual.get("document_ids"):
        errors.append("document_ids differ")
    if errors:
        raise RuntimeError("delta input manifest mismatch: " + "; ".join(errors))


def _embed_docs_exact(
    cfg,
    client,
    embedder,
    count_tokens,
    docs: Iterable[CanonicalDoc],
    *,
    batch_size: int,
) -> dict:
    """Embed docs while retaining exact successful identities and all failures."""
    from ingest import qdrant_store as store
    from ingest.pipeline import _build_doc_points

    pending = []
    embedded_ids: list[str] = []
    chunks = 0
    failures: list[dict] = []

    def flush() -> None:
        if pending:
            store.upsert_points(client, cfg.collection_name, pending, wait=True)
            pending.clear()

    for doc in docs:
        try:
            points, count = _build_doc_points(cfg, embedder, count_tokens, doc)
            if not points or count <= 0:
                raise ValueError("body produced no embedded chunks")
        except Exception as exc:  # noqa: BLE001 - report exact failed identity
            failures.append(
                {
                    "source": doc.source,
                    "document_id": doc.document_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        pending.extend(points)
        embedded_ids.append(f"{doc.source}\t{doc.document_id}")
        chunks += count
        if len(pending) >= batch_size:
            flush()
    flush()
    return {
        "document_ids": sorted(embedded_ids),
        "documents": len(embedded_ids),
        "chunks": chunks,
        "skipped": len(failures),
        "failures": failures,
    }


def _strict_manifest_check(manifest: dict) -> None:
    if manifest["documents"] <= 0:
        raise RuntimeError("delta contains no embeddable documents")
    if manifest["skipped"]:
        sample = manifest.get("failures", [])[:3]
        raise RuntimeError(
            f"delta validation skipped {manifest['skipped']} record(s): {sample}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", nargs="*", help="explicit items.jsonl path(s)")
    ap.add_argument(
        "--items-dir",
        help="directory of per-source files named <source>.jsonl (multi-source delta)",
    )
    ap.add_argument(
        "--source",
        default="matsne",
        help="normalization source (default: matsne; ignored for --items-dir)",
    )
    ap.add_argument("--runs-since", help="include Matsne run ids >= this value")
    ap.add_argument("--collection", help="target Qdrant collection")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--dry-run", action="store_true", help="tokenize/chunk/report; do not embed"
    )
    ap.add_argument(
        "--strict", action="store_true", help="fail on any malformed/skipped document"
    )
    ap.add_argument(
        "--recreate", action="store_true", help="recreate the target collection first"
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help=f"permit Qdrant writes; also requires {QDRANT_WRITE_APPROVAL_ENV}=1",
    )
    ap.add_argument(
        "--manifest-out", type=Path, help="atomically write input/embed JSON manifest"
    )
    ap.add_argument(
        "--expect-manifest",
        type=Path,
        help="require parity with a prior input manifest",
    )
    args = ap.parse_args()

    if not args.dry_run:
        require_explicit_approval(
            apply=args.apply,
            approval_env=QDRANT_WRITE_APPROVAL_ENV,
            operation="delta embedding Qdrant write",
        )
        if not args.collection:
            raise SystemExit("delta writes require an explicit --collection")
        require_run_scoped_delta_collection(args.collection)

    cfg = load_config()
    if args.collection:
        cfg = dataclasses.replace(cfg, collection_name=args.collection)

    if args.items_dir:
        by_source: list[tuple[str, list[Path]]] = [
            (path.stem, [path])
            for path in sorted(Path(args.items_dir).glob("*.jsonl"))
            if path.stat().st_size
        ]
        if not by_source:
            raise SystemExit(f"no non-empty <source>.jsonl in {args.items_dir}")
    else:
        paths = _resolve_paths(cfg, args)
        if not paths:
            raise SystemExit(
                "no items.jsonl found (pass --items, --items-dir, or --runs-since)"
            )
        missing = [
            str(path)
            for path in paths
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise SystemExit(f"missing/empty input file(s): {missing}")
        by_source = [(args.source, paths)]

    from ingest.embedding import make_token_counter

    count_tokens = make_token_counter(cfg.tokenizer_model, cfg.tokenizer_revision)
    docs, input_manifest = _analyze_docs(cfg, by_source, count_tokens)
    try:
        if args.expect_manifest:
            expected = json.loads(args.expect_manifest.read_text(encoding="utf-8"))
            validate_expected_manifest(expected, input_manifest)
        if args.strict:
            _strict_manifest_check(input_manifest)

        print(
            f"delta: {input_manifest['documents']} docs -> "
            f"{input_manifest['chunks']} chunks "
            f"({input_manifest['skipped']} skipped) {input_manifest['sources']} "
            f"-> {cfg.collection_name!r}"
        )
        if args.dry_run:
            if args.manifest_out:
                _write_json_atomic(args.manifest_out, input_manifest)
            return

        from ingest.embedding import BGEM3Embedder
        from ingest.qdrant_store import ensure_collection, make_client

        client = make_client(cfg)
        ensure_collection(
            client,
            cfg,
            recreate=args.recreate,
            apply=args.apply,
            allow_run_scoped_delta=True,
        )
        print(
            f"loading BGE-M3 (device={cfg.embed_device or 'cpu'}, "
            f"fp16={cfg.embed_use_fp16})..."
        )
        embedder = BGEM3Embedder(cfg)
        embedded = _embed_docs_exact(
            cfg, client, embedder, count_tokens, docs, batch_size=args.batch_size
        )
        info = client.get_collection(cfg.collection_name)
        report = {
            **input_manifest,
            "kind": "delta_embed",
            "collection": cfg.collection_name,
            "expected_documents": input_manifest["documents"],
            "expected_chunks": input_manifest["chunks"],
            "documents": embedded["documents"],
            "chunks": embedded["chunks"],
            "skipped": embedded["skipped"],
            "failures": embedded["failures"],
            "document_ids": embedded["document_ids"],
            "points_count": int(info.points_count),
        }
        report["document_ids_sha256"] = hashlib.sha256(
            "\n".join(report["document_ids"]).encode("utf-8")
        ).hexdigest()
        if args.manifest_out:
            _write_json_atomic(args.manifest_out, report)

        expected_for_embed = dict(input_manifest)
        validate_expected_manifest(expected_for_embed, report)
        if report["points_count"] != report["chunks"]:
            raise RuntimeError(
                f"collection points {report['points_count']} != "
                f"embedded chunks {report['chunks']}"
            )
        if args.strict:
            _strict_manifest_check(report)
        print(
            f"embedded {report['documents']} docs -> {report['chunks']} chunks "
            f"({report['skipped']} skipped) into {cfg.collection_name!r}; "
            f"collection has {report['points_count']} points."
        )
    finally:
        docs.close()


if __name__ == "__main__":
    main()
