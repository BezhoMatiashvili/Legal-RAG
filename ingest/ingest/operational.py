"""Fail-closed guards shared by mutating operational entry points."""

from __future__ import annotations

from collections.abc import Mapping
import os
import re


RUNPOD_SPEND_APPROVAL_ENV = "RUNPOD_SPEND_APPROVED"
QDRANT_WRITE_APPROVAL_ENV = "QDRANT_WRITE_APPROVED"
QDRANT_RECREATE_APPROVAL_ENV = "QDRANT_RECREATE_APPROVED"
RUNPOD_EPHEMERAL_QDRANT_ENV = "RUNPOD_EPHEMERAL_QDRANT"

_RUN_SCOPED_DELTA = re.compile(
    r"^georgian_legal_delta_[A-Za-z0-9][A-Za-z0-9_-]{2,190}$"
)


def require_explicit_approval(
    *,
    apply: bool,
    approval_env: str,
    operation: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Require a CLI apply flag and a separately supplied environment approval."""

    environment = os.environ if environ is None else environ
    if not apply or environment.get(approval_env) != "1":
        raise SystemExit(
            f"{operation} requires both --apply and {approval_env}=1"
        )


def require_run_scoped_delta_collection(collection: str) -> None:
    """Reject stable/live names and accept only an explicit run-scoped staging name."""

    if not _RUN_SCOPED_DELTA.fullmatch(collection):
        raise SystemExit(
            "Qdrant delta writes require an explicit run-scoped collection named "
            "georgian_legal_delta_<source>_<run>; stable/live collection names are refused"
        )


def refuse_legacy_operation(operation: str) -> None:
    """Permanently disable in-place corpus mutations superseded by generations."""

    raise SystemExit(
        f"Legacy {operation} is disabled before configuration, subprocess, spend, or "
        "Qdrant access. Build and verify an immutable full-corpus generation, then use "
        "the guarded promotion workflow."
    )
