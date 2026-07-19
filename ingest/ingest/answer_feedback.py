"""Structured, local-only feedback for accuracy-first canary monitoring."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from .serving_rollout import ReleaseAuthorization, RolloutStage, ServingTuple


class FeedbackCategory(str, Enum):
    WRONG_AUTHORITY = "wrong_authority"
    WRONG_VERSION = "wrong_version"
    WRONG_PASSAGE = "wrong_passage"
    INCOMPLETE_ANSWER = "incomplete_answer"
    INVALID_CITATION = "invalid_citation"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"


class FeedbackSeverity(str, Enum):
    SEVERE = "severe"
    MATERIAL = "material"
    MINOR = "minor"


@dataclass(frozen=True)
class AnswerFeedback:
    trace_id: str
    category: FeedbackCategory
    severity: FeedbackSeverity
    serving: ServingTuple
    release_authorization: ReleaseAuthorization
    traffic_stage: RolloutStage
    question_language: str
    reviewer_id: str
    notes: str | None = None
    created_at: str | None = None

    def normalized(self) -> AnswerFeedback:
        self.serving.validate()
        if self.traffic_stage not in {
            RolloutStage.CANARY_5,
            RolloutStage.PRODUCTION,
        }:
            raise ValueError("feedback requires canary or production user traffic")
        if not isinstance(self.release_authorization, ReleaseAuthorization):
            raise ValueError("feedback requires release authorization")
        try:
            self.release_authorization.validate_for(
                self.serving,
                self.traffic_stage,
            )
        except ValueError as exc:
            raise ValueError(
                "feedback serving tuple does not match release authorization"
            ) from exc
        language = self.question_language.strip().lower()
        if language not in {"ka", "en"}:
            raise ValueError("feedback language must be supported Georgian or English")
        if len(self.trace_id) != 64 or not all(
            ch in "0123456789abcdef" for ch in self.trace_id
        ):
            raise ValueError("trace_id must be a 64-character lowercase hex digest")
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        return AnswerFeedback(
            trace_id=self.trace_id,
            category=self.category,
            severity=self.severity,
            serving=self.serving,
            release_authorization=self.release_authorization,
            traffic_stage=self.traffic_stage,
            question_language=language,
            reviewer_id=self.reviewer_id.strip(),
            notes=self.notes.strip() if self.notes else None,
            created_at=self.created_at
            or datetime.now(UTC).replace(microsecond=0).isoformat(),
        )


def append_feedback(record: AnswerFeedback, path: Path) -> None:
    """Append one owner-only JSONL record; never send legal feedback off-box."""

    normalized = record.normalized()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError("feedback log is not owner-only")
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    payload = asdict(normalized)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
