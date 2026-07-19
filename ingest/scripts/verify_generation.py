#!/usr/bin/env python
"""Verify one immutable generation against an existing Qdrant collection.

Only read operations are used: collection metadata, an exact filtered count, and
a full scroll with payloads and vectors. The sole write is the owner-only sibling
"<generation>.verification.json" report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.collection_compatibility import (  # noqa: E402
    CollectionCompatibility,
    CompatibilityIssue,
    check_collection_compatibility,
)
from ingest.config import load_config  # noqa: E402
from ingest.generation import GenerationArtifacts, load_generation  # noqa: E402
from ingest.integrity import (  # noqa: E402
    VerificationOutcome,
    VerificationReport,
    load_structural_inventory_binding,
    verify_generation_artifacts,
    write_verification_report,
)
from ingest.promotion import physical_collection_name  # noqa: E402
from ingest.qdrant_store import (  # noqa: E402
    collection_configuration,
    collection_configuration_sha256,
    make_client,
)

DEFAULT_PAGE_SIZE = 256
DEFAULT_MAX_EXAMPLES = 20
_VERIFICATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _offline_token_counter(
    model_name: str,
    revision: str,
) -> Callable[[str], int]:
    """Load only the exact locally available tokenizer; verification never downloads."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=True,
    )

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def sibling_report_path(
    generation_dir: str | Path, verification_id: str | None = None
) -> Path:
    """Return the canonical report path outside the immutable generation."""
    directory = Path(generation_dir)
    if not directory.name:
        raise ValueError("generation_dir must name a generation directory")
    if verification_id is None:
        # Legacy callers can still locate historical fixed-name reports.  New verification
        # runs must always supply a run identity and never write this path.
        return directory.with_name(f"{directory.name}.verification.json")
    if not _VERIFICATION_ID_RE.fullmatch(verification_id):
        raise ValueError("verification_id is unsafe")
    return directory.with_name(
        f"{directory.name}.{verification_id}.verification.json"
    )


def stream_collection_points(
    client: Any,
    collection_name: str,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Iterator[Any]:
    """Yield one Qdrant scroll page at a time without corpus materialization."""
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size < 1:
        raise ValueError("page_size must be an integer >= 1")
    if page_size > 10_000:
        raise ValueError("page_size must be <= 10000")

    offset = None
    while True:
        result = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=page_size,
            with_payload=True,
            with_vectors=True,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Qdrant scroll returned an invalid page")
        points, next_offset = result
        if points is None:
            raise RuntimeError("Qdrant scroll returned no point iterable")
        yielded = 0
        for point in points:
            yielded += 1
            yield point
        if next_offset is None:
            return
        if not yielded:
            raise RuntimeError(
                "Qdrant scroll returned an empty page with a continuation"
            )
        if offset is not None and next_offset == offset:
            raise RuntimeError("Qdrant scroll continuation did not advance")
        offset = next_offset


def _append_issues(
    outcome: VerificationOutcome,
    issues: tuple[CompatibilityIssue, ...],
    *,
    max_examples: int,
) -> VerificationOutcome:
    compatibility_examples = []
    for issue in issues:
        if len(compatibility_examples) >= max_examples:
            break
        compatibility_examples.append(
            {
                "code": issue.code,
                "check": "collection_compatibility",
                **dict(issue.details),
            }
        )
    examples = (compatibility_examples + list(outcome.examples))[:max_examples]
    return VerificationOutcome(
        ok=outcome.ok and not issues,
        issue_count=outcome.issue_count + len(issues),
        examples=tuple(examples),
    )


def _merge_compatibility(
    report: VerificationReport,
    compatibility: CollectionCompatibility,
    *,
    max_examples: int,
) -> VerificationReport:
    stats = dict(report.stats)
    if compatibility.points_count is not None:
        stats["collection_points"] = compatibility.points_count
    if compatibility.identity_matched_points is not None:
        stats["identity_matched_points"] = compatibility.identity_matched_points
    return replace(
        report,
        stats=stats,
        coverage=_append_issues(
            report.coverage,
            compatibility.coverage_issues,
            max_examples=max_examples,
        ),
        integrity=_append_issues(
            report.integrity,
            compatibility.integrity_issues,
            max_examples=max_examples,
        ),
    )


def verify_loaded_generation(
    client: Any,
    artifacts: GenerationArtifacts,
    collection_name: str,
    *,
    snapshot_root: str | Path,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    verification_id: str | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> VerificationReport:
    """Run compatibility and streamed point verification for loaded artifacts."""
    expected_collection = physical_collection_name(artifacts.manifest.generation_id)
    if collection_name != expected_collection:
        raise ValueError(
            "generation verification requires the exact physical collection "
            f"{expected_collection!r}; got {collection_name!r}"
        )
    # Fail on snapshot/provenance drift before tokenizer loading or any Qdrant read.  The
    # artifact verifier repeats this binding after metadata checks, and the row iterator's
    # final digest/count checks close the remaining time-of-check/time-of-use window.
    load_structural_inventory_binding(artifacts, snapshot_root)
    token_counter = count_tokens or _offline_token_counter(
        artifacts.manifest.model.tokenizer_model,
        artifacts.manifest.model.tokenizer_revision,
    )
    compatibility = check_collection_compatibility(
        client,
        collection_name,
        artifacts.manifest,
    )
    info = client.get_collection(collection_name)
    config_sha256 = collection_configuration_sha256(
        collection_configuration(info)
    )
    report = verify_generation_artifacts(
        artifacts,
        stream_collection_points(
            client,
            collection_name,
            page_size=page_size,
        ),
        snapshot_root=snapshot_root,
        count_tokens=token_counter,
        max_examples=max_examples,
        physical_collection=collection_name,
        observed_collection_configuration_sha256=config_sha256,
        verification_id=verification_id,
    )
    return _merge_compatibility(
        report,
        compatibility,
        max_examples=max_examples,
    )


def verify_generation_directory(
    client: Any,
    generation_dir: str | Path,
    collection_name: str,
    *,
    snapshot_root: str | Path,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    verification_id: str,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[VerificationReport, Path]:
    """Load, verify, and atomically write the canonical sibling report."""
    artifacts = load_generation(generation_dir)
    report = verify_loaded_generation(
        client,
        artifacts,
        collection_name,
        snapshot_root=snapshot_root,
        page_size=page_size,
        max_examples=max_examples,
        verification_id=verification_id,
        count_tokens=count_tokens,
    )
    report_path = sibling_report_path(artifacts.root, verification_id)
    write_verification_report(report_path, report)
    return report, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="immutable generation directory containing manifest/checksum ledgers",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="exact physical generation collection; defaults to COLLECTION_NAME",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        required=True,
        help="independently supplied sealed snapshot root bound by generation provenance",
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument(
        "--verification-id",
        required=True,
        help="unique run identity used in the create-only sidecar filename",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifacts = load_generation(args.generation_dir)
        cfg = load_config()
        collection_name = args.collection or cfg.collection_name
        client = make_client(cfg)
        try:
            report = verify_loaded_generation(
                client,
                artifacts,
                collection_name,
                snapshot_root=args.snapshot_root,
                page_size=args.page_size,
                max_examples=args.max_examples,
                verification_id=args.verification_id,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        report_path = sibling_report_path(artifacts.root, args.verification_id)
        write_verification_report(report_path, report)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed on any uncertainty
        print(f"generation verification failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": report.ok,
                "generation_id": report.generation_id,
                "manifest_sha256": report.manifest_sha256,
                "physical_collection": collection_name,
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
