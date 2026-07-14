"""Offline acceptance gates for the timed Supreme Court partial artifact."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_supremecourt_partial as validator  # noqa: E402


ADMIN = "ადმინისტრაციულ საქმეთა პალატა"
CIVIL = "სამოქალაქო საქმეთა პალატა"
CRIMINAL = "სისხლის სამართლის საქმეთა პალატა"
CHAMBERS = (ADMIN, CIVIL, CRIMINAL)


def _item(case_id: str, chamber: str, decision_date: str, *, target=False) -> dict:
    palata = validator.CHAMBER_TO_PALATA[chamber]
    body = "საქართველოს უზენაესი სასამართლოს სრული გადაწყვეტილება. " * 8
    if target:
        body += f" პირველი ინსტანციის საქმის ნომერია {validator.TARGET_LITERAL}."
    return {
        "case_id": case_id,
        "case_number": f"ას-{case_id}-2026",
        "chamber": chamber,
        "date": decision_date,
        "source_url": f"{validator.BASE_URL}/ka/fullcase/{case_id}/{palata}",
        "body_markdown": body,
    }


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _derived_chambers(rows: list[dict], journal: list[dict]) -> dict:
    output = {}
    journal_ids = {(row["case_id"], row["chamber"]) for row in journal}
    for chamber in CHAMBERS:
        chamber_rows = [row for row in rows if row["chamber"] == chamber]
        dates = [row["date"] for row in chamber_rows]
        new = sum(
            (row["case_id"], row["chamber"]) in journal_ids for row in chamber_rows
        )
        output[chamber] = {
            "known_items": len(chamber_rows) - new,
            "new_items": new,
            "total_items": len(chamber_rows),
            "newest_date": max(dates) if dates else None,
            "oldest_date": min(dates) if dates else None,
            "resume_cursor": "2026-05-31",
        }
    return output


def _write_run(
    tmp_path: Path,
    *,
    run_id: str = "20260713T100000Z_start-1900-01-01_end-2026-07-13",
) -> tuple[Path, list[dict], list[dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = tmp_path / run_id
    run.mkdir(mode=0o700)
    rows = [
        _item("300", ADMIN, "2026-06-01"),
        _item("100", CIVIL, "2026-07-03"),
        _item("099", CIVIL, "2026-05-20"),
        _item("49251", CRIMINAL, "2026-07-06", target=True),
    ]
    rows.sort(
        key=lambda row: (row["date"], row["chamber"], row["case_id"]), reverse=True
    )
    journal_ids = {"49251", "100"}
    journal = [row for row in rows if row["case_id"] in journal_ids]

    items_path = run / "items.jsonl"
    journal_path = run / "items.journal.jsonl"
    items_payload = _jsonl_bytes(rows)
    items_path.write_bytes(items_payload)
    journal_path.write_bytes(_jsonl_bytes(journal))
    per_chamber = _derived_chambers(rows, journal)
    manifest = {
        "schema_version": 1,
        "run_id": run.name,
        "started_at": "2026-07-13T10:00:00+00:00",
        "finished_at": "2026-07-13T14:00:03+00:00",
        "finish_reason": "closespider_timeout",
        "partial_by_design": True,
        "date_order": "newest_first",
        "frontier_start_date": "2026-07-13",
        "lower_bound": "1900-01-01",
        "max_runtime_seconds": 14400,
        "elapsed_time_seconds": 14403.0,
        "items_file": str(items_path.absolute()),
        "journal_file": str(journal_path.absolute()),
        "items_sha256": hashlib.sha256(items_payload).hexdigest(),
        "known_items": len(rows) - len(journal),
        "new_items": len(journal),
        "total_items": len(rows),
        "known_encountered": 1,
        "per_chamber": per_chamber,
        "oldest_fully_completed_global_date_frontier": "2026-06-01",
        "per_chamber_resume_cursors": {
            chamber: values["resume_cursor"] for chamber, values in per_chamber.items()
        },
        "completed_windows": [
            {
                "id": "w000001",
                "chamber": ADMIN,
                "start": "2026-06-01",
                "end": "2026-07-13",
                "status": "completed",
                "authoritative_total": 1,
                "known": 1,
                "new": 0,
                "failures": 0,
            },
            {
                "id": "w000002",
                "chamber": CIVIL,
                "start": "2026-06-01",
                "end": "2026-07-13",
                "status": "completed",
                "authoritative_total": 1,
                "known": 0,
                "new": 1,
                "failures": 0,
            },
            {
                "id": "w000003",
                "chamber": CRIMINAL,
                "start": "2026-06-01",
                "end": "2026-07-13",
                "status": "completed",
                "authoritative_total": 1,
                "known": 0,
                "new": 1,
                "failures": 0,
            },
        ],
        "retries": {"total": 0, "retry_after": 0, "parse": 0, "detail_parse": 0},
        "unresolved_failure_count": 0,
        "unresolved_failures_truncated": False,
        "unresolved_failures": [],
    }
    manifest_path = run / "partial_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (items_path, journal_path, manifest_path):
        path.chmod(0o600)
    return run, rows, journal


def _manifest(run: Path) -> dict:
    return json.loads((run / "partial_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(run: Path, manifest: dict) -> None:
    path = run / "partial_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)


def _rewrite_items(run: Path, rows: list[dict], *, update_sha=True) -> None:
    payload = _jsonl_bytes(rows)
    path = run / "items.jsonl"
    path.write_bytes(payload)
    path.chmod(0o600)
    if update_sha:
        manifest = _manifest(run)
        manifest["items_sha256"] = hashlib.sha256(payload).hexdigest()
        _write_manifest(run, manifest)


def _rewrite_journal(run: Path, rows: list[dict]) -> None:
    path = run / "items.journal.jsonl"
    path.write_bytes(_jsonl_bytes(rows))
    path.chmod(0o600)


def _write_resume_child(runs: Path, parent: Path) -> Path:
    child, rows, _journal = _write_run(
        runs,
        run_id="20260713T150000Z_start-1900-01-01_end-2026-07-13",
    )
    _rewrite_journal(child, [])
    manifest = _manifest(child)
    manifest.update(
        {
            "started_at": "2026-07-13T15:00:00+00:00",
            "finished_at": "2026-07-13T19:00:03+00:00",
            "known_items": len(rows),
            "new_items": 0,
            "per_chamber_start_cursors": {
                chamber: "2026-05-31" for chamber in CHAMBERS
            },
            "resume_parent": {
                "run_id": parent.name,
                "manifest_file": str(
                    (parent / "partial_manifest.json").absolute()
                ),
                "manifest_sha256": hashlib.sha256(
                    (parent / "partial_manifest.json").read_bytes()
                ).hexdigest(),
                "items_sha256": hashlib.sha256(
                    (parent / "items.jsonl").read_bytes()
                ).hexdigest(),
            },
            "oldest_fully_completed_global_date_frontier": "2026-05-20",
            "per_chamber_resume_cursors": {
                chamber: "2026-05-19" for chamber in CHAMBERS
            },
            "completed_windows": [
                {
                    "id": f"resume-{index}",
                    "chamber": chamber,
                    "start": "2026-05-20",
                    "end": "2026-05-31",
                    "status": "completed",
                    "authoritative_total": 0,
                    "known": 0,
                    "new": 0,
                    "failures": 0,
                }
                for index, chamber in enumerate(CHAMBERS, 1)
            ],
        }
    )
    for chamber, values in manifest["per_chamber"].items():
        values["known_items"] = values["total_items"]
        values["new_items"] = 0
        values["resume_cursor"] = "2026-05-19"
    _write_manifest(child, manifest)
    return child


def test_valid_run_prints_machine_readable_exact_summary(tmp_path, capsys):
    run, _rows, _journal = _write_run(tmp_path)

    report = validator.validate_run(run)

    assert report["valid"] is True
    assert report["total_items"] == 4
    assert report["known_items"] == 2
    assert report["new_items"] == report["journal_items"] == 2
    assert report["target_body_hits"] == 1
    assert report["target_case_number_hits"] == 0
    assert report["per_chamber"][CRIMINAL]["new_items"] == 1
    assert report["elapsed_time_seconds"] == 14403.0
    assert report["oldest_fully_completed_global_date_frontier"] == "2026-06-01"
    assert report["per_chamber_resume_cursors"] == {
        chamber: "2026-05-31" for chamber in CHAMBERS
    }
    assert report["retries"] == {
        "total": 0,
        "retry_after": 0,
        "parse": 0,
        "detail_parse": 0,
    }
    assert report["unresolved_failure_count"] == 0
    assert (
        report["journal_sha256"]
        == hashlib.sha256((run / "items.journal.jsonl").read_bytes()).hexdigest()
    )

    assert validator.main(["--run-dir", str(run)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == report


def test_resume_chain_is_recursively_validated_and_uses_parent_derived_cursors(
    tmp_path,
):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(
        runs,
        run_id="20260713T100000Z_start-1900-01-01_end-2026-07-13",
    )
    child = _write_resume_child(runs, parent)

    report = validator.validate_run(child)

    assert report["valid"] is True
    assert report["per_chamber_start_cursors"] == {
        chamber: "2026-05-31" for chamber in CHAMBERS
    }
    assert report["per_chamber_resume_cursors"] == {
        chamber: "2026-05-19" for chamber in CHAMBERS
    }
    assert report["oldest_fully_completed_global_date_frontier"] == "2026-05-20"
    assert report["resume_parent"]["run_id"] == parent.name


def test_resume_chain_rejects_parent_byte_drift(tmp_path):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(runs)
    child = _write_resume_child(runs, parent)
    manifest = _manifest(child)
    manifest["resume_parent"]["manifest_sha256"] = "0" * 64
    _write_manifest(child, manifest)

    with pytest.raises(validator.ValidationError, match="parent bytes differ"):
        validator.validate_run(child)


def test_resume_chain_rejects_a_valid_parent_from_a_different_date_scope(tmp_path):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(runs)
    child = _write_resume_child(runs, parent)
    parent_manifest = _manifest(parent)
    parent_manifest["lower_bound"] = "1899-01-01"
    _write_manifest(parent, parent_manifest)
    child_manifest = _manifest(child)
    child_manifest["resume_parent"]["manifest_sha256"] = hashlib.sha256(
        (parent / "partial_manifest.json").read_bytes()
    ).hexdigest()
    _write_manifest(child, child_manifest)

    with pytest.raises(validator.ValidationError, match="date scope differs"):
        validator.validate_run(child)


def test_resume_chain_rejects_a_dropped_parent_identity(tmp_path):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(runs)
    child = _write_resume_child(runs, parent)
    child_rows = [
        json.loads(line)
        for line in (child / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    child_rows = [row for row in child_rows if row["chamber"] != ADMIN]
    _rewrite_items(child, child_rows)
    child_manifest = _manifest(child)
    child_manifest["known_items"] = child_manifest["total_items"] = len(child_rows)
    child_manifest["per_chamber"][ADMIN].update(
        {
            "known_items": 0,
            "new_items": 0,
            "total_items": 0,
            "newest_date": None,
            "oldest_date": None,
        }
    )
    _write_manifest(child, child_manifest)

    with pytest.raises(validator.ValidationError, match="dropped parent identity"):
        validator.validate_run(child)


def test_resume_chain_rejects_a_changed_parent_fingerprint(tmp_path):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(runs)
    child = _write_resume_child(runs, parent)
    child_rows = [
        json.loads(line)
        for line in (child / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    child_rows[0]["body_markdown"] += " შეცვლილი"
    _rewrite_items(child, child_rows)

    with pytest.raises(validator.ValidationError, match="changed parent identity"):
        validator.validate_run(child)


def test_resume_chain_rejects_paths_outside_the_sibling_runs_root(tmp_path):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(runs)
    child = _write_resume_child(runs, parent)
    manifest = _manifest(child)
    manifest["resume_parent"]["manifest_file"] = str(
        (tmp_path / "outside" / "partial_manifest.json").absolute()
    )
    _write_manifest(child, manifest)

    with pytest.raises(validator.ValidationError, match="exact contained"):
        validator.validate_run(child)


def test_resume_chain_rejects_self_cycles_and_cursor_drift(tmp_path):
    runs = tmp_path / "runs"
    parent, _rows, _journal = _write_run(runs)
    child = _write_resume_child(runs, parent)
    manifest = _manifest(child)
    manifest["per_chamber_start_cursors"][ADMIN] = "2026-05-30"
    _write_manifest(child, manifest)
    with pytest.raises(validator.ValidationError, match="parent's derived resume"):
        validator.validate_run(child)

    manifest = _manifest(child)
    manifest["per_chamber_start_cursors"][ADMIN] = "2026-05-31"
    manifest["resume_parent"]["run_id"] = child.name
    manifest["resume_parent"]["manifest_file"] = str(
        (child / "partial_manifest.json").absolute()
    )
    _write_manifest(child, manifest)
    with pytest.raises(validator.ValidationError, match="unsafe, missing, or cyclic"):
        validator.validate_run(child)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("finish_reason", "finished", "four-hour timeout"),
        ("max_runtime_seconds", 14399, "14400"),
        ("partial_by_design", False, "partial_by_design"),
        ("finished_at", None, "final ISO timestamp"),
        ("elapsed_time_seconds", None, "finite numeric duration"),
        ("elapsed_time_seconds", 14399.999, ">= 14400"),
    ],
)
def test_final_timeout_manifest_is_mandatory(tmp_path, field, bad_value, match):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest[field] = bad_value
    _write_manifest(run, manifest)

    with pytest.raises(validator.ValidationError, match=match):
        validator.validate_run(run)


def test_exact_items_sha_is_mandatory(tmp_path):
    run, rows, _journal = _write_run(tmp_path)
    changed = deepcopy(rows)
    changed[0]["body_markdown"] += " შეცვლილი"
    _rewrite_items(run, changed, update_sha=False)

    with pytest.raises(validator.ValidationError, match="items_sha256"):
        validator.validate_run(run)


def test_cumulative_identity_url_and_materialization_order_are_strict(tmp_path):
    run, rows, _journal = _write_run(tmp_path)
    reordered = deepcopy(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _rewrite_items(run, reordered)

    with pytest.raises(validator.ValidationError, match="crawler materialization"):
        validator.validate_run(run)

    run, rows, _journal = _write_run(tmp_path / "second")
    duplicate = deepcopy(rows)
    duplicate[-1] = deepcopy(duplicate[0])
    _rewrite_items(run, duplicate)
    with pytest.raises(validator.ValidationError, match="duplicate official identity"):
        validator.validate_run(run)

    run, rows, _journal = _write_run(tmp_path / "third")
    wrong_url = deepcopy(rows)
    wrong_url[0]["source_url"] += "?download=1"
    _rewrite_items(run, wrong_url)
    with pytest.raises(validator.ValidationError, match="exact official fullcase URL"):
        validator.validate_run(run)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"body_markdown": "   "}, "nonempty full text"),
        ({"content_complete": False}, "marks the record incomplete"),
        ({"content_kind": "metadata_only"}, "metadata-only content_kind"),
        ({"metadata_only": True}, "metadata_only=true"),
        ({"extraction_status": "malformed"}, "extraction_status"),
    ],
)
def test_full_text_has_no_metadata_only_markers(tmp_path, mutation, match):
    run, rows, _journal = _write_run(tmp_path)
    changed = deepcopy(rows)
    changed[0].update(mutation)
    _rewrite_items(run, changed)

    with pytest.raises(validator.ValidationError, match=match):
        validator.validate_run(run)


def test_journal_must_be_unique_exact_subset_count_and_date_order(tmp_path):
    run, _rows, journal = _write_run(tmp_path)
    _rewrite_journal(run, [journal[0], journal[0]])
    with pytest.raises(validator.ValidationError, match="duplicate official identity"):
        validator.validate_run(run)

    run, _rows, journal = _write_run(tmp_path / "second")
    _rewrite_journal(run, list(reversed(journal)))
    with pytest.raises(validator.ValidationError, match="dates must be non-increasing"):
        validator.validate_run(run)

    run, _rows, journal = _write_run(tmp_path / "third")
    changed = deepcopy(journal)
    changed[0]["body_markdown"] += " journal drift"
    _rewrite_journal(run, changed)
    with pytest.raises(validator.ValidationError, match="differs from items.jsonl"):
        validator.validate_run(run)


def test_manifest_and_per_chamber_counts_and_spans_are_exact(tmp_path):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["new_items"] += 1
    manifest["per_chamber"][CIVIL]["oldest_date"] = "2026-01-01"
    _write_manifest(run, manifest)

    with pytest.raises(validator.ValidationError) as captured:
        validator.validate_run(run)

    message = str(captured.value)
    assert "exact journal count" in message
    assert "oldest_date" in message


@pytest.mark.parametrize("bad_frontier", [None, "2026/06/01", "1899-12-31"])
def test_global_completed_frontier_is_nonnull_canonical_and_bounded(
    tmp_path, bad_frontier
):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["oldest_fully_completed_global_date_frontier"] = bad_frontier
    _write_manifest(run, manifest)

    with pytest.raises(
        validator.ValidationError,
        match="oldest_fully_completed_global_date_frontier",
    ):
        validator.validate_run(run)


def test_resume_cursors_and_global_frontier_are_derived_from_completed_windows(
    tmp_path,
):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["per_chamber"][ADMIN]["resume_cursor"] = "2026-05-30"
    manifest["per_chamber_resume_cursors"][ADMIN] = "2026-05-30"
    manifest["oldest_fully_completed_global_date_frontier"] = "2026-06-02"
    _write_manifest(run, manifest)

    with pytest.raises(validator.ValidationError) as captured:
        validator.validate_run(run)

    message = str(captured.value)
    assert "completed windows derive '2026-05-31'" in message
    assert "derive '2026-06-01'" in message


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda windows: windows[0].update(authoritative_total=2),
            "known \\+ new must equal authoritative_total",
        ),
        (
            lambda windows: windows[1].update(id=windows[0]["id"]),
            "duplicate window id",
        ),
        (
            lambda windows: windows.append(
                {
                    **windows[0],
                    "id": "w-overlap",
                    "start": "2026-06-15",
                    "end": "2026-06-30",
                    "authoritative_total": 0,
                    "known": 0,
                }
            ),
            "overlapping leaf intervals",
        ),
        (
            lambda windows: windows[0].update(status="incomplete", failures=0),
            "incomplete window must retain a failure",
        ),
    ],
)
def test_completed_window_evidence_is_structurally_trustworthy(
    tmp_path, mutation, match
):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    mutation(manifest["completed_windows"])
    _write_manifest(run, manifest)

    with pytest.raises(validator.ValidationError, match=match):
        validator.validate_run(run)


@pytest.mark.parametrize(
    "bad_retries",
    [
        {"total": 0, "retry_after": 0, "parse": 0},
        {
            "total": -1,
            "retry_after": 0,
            "parse": 0,
            "detail_parse": 0,
        },
        {
            "total": 0,
            "retry_after": 0,
            "parse": 0,
            "detail_parse": 0,
            "other": 0,
        },
    ],
)
def test_retry_accounting_is_exact_and_nonnegative(tmp_path, bad_retries):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["retries"] = bad_retries
    _write_manifest(run, manifest)

    with pytest.raises(validator.ValidationError, match="manifest.retries"):
        validator.validate_run(run)


def test_nonzero_unresolved_failures_are_allowed_when_truncation_is_consistent(
    tmp_path,
):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["unresolved_failure_count"] = 2
    manifest["unresolved_failures"] = [{"kind": "detail_http", "detail": "HTTP 503"}]
    manifest["unresolved_failures_truncated"] = True
    _write_manifest(run, manifest)

    report = validator.validate_run(run)

    assert report["valid"] is True
    assert report["unresolved_failure_count"] == 2


@pytest.mark.parametrize(
    ("count", "retained", "truncated"),
    [
        (2, [{"kind": "detail_http"}], False),
        (1, [{"kind": "detail_http"}], True),
        (0, "not-a-list", False),
        (0, [], "false"),
    ],
)
def test_unresolved_failure_count_list_and_truncation_must_agree(
    tmp_path, count, retained, truncated
):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["unresolved_failure_count"] = count
    manifest["unresolved_failures"] = retained
    manifest["unresolved_failures_truncated"] = truncated
    _write_manifest(run, manifest)

    with pytest.raises(validator.ValidationError, match="unresolved"):
        validator.validate_run(run)


def test_target_is_a_body_hit_never_a_supreme_case_number(tmp_path):
    run, rows, _journal = _write_run(tmp_path)
    changed = deepcopy(rows)
    target = next(
        row for row in changed if validator.TARGET_LITERAL in row["body_markdown"]
    )
    target["case_number"] = validator.TARGET_LITERAL
    _rewrite_items(run, changed)

    with pytest.raises(validator.ValidationError, match="first-instance number"):
        validator.validate_run(run)

    run, rows, _journal = _write_run(tmp_path / "second")
    changed = deepcopy(rows)
    for row in changed:
        row["body_markdown"] = row["body_markdown"].replace(
            validator.TARGET_LITERAL, ""
        )
    _rewrite_items(run, changed)
    _rewrite_journal(
        run,
        [row for row in changed if row["case_id"] in {"49251", "100"}],
    )
    with pytest.raises(validator.ValidationError, match="not found"):
        validator.validate_run(run)


def test_all_evidence_files_are_private_regular_and_not_symlinks(tmp_path):
    run, _rows, _journal = _write_run(tmp_path)
    (run / "items.jsonl").chmod(0o644)
    with pytest.raises(validator.ValidationError, match="owner-private"):
        validator.validate_run(run)

    run, _rows, _journal = _write_run(tmp_path / "second")
    journal = run / "items.journal.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(journal.read_bytes())
    outside.chmod(0o600)
    journal.unlink()
    os.symlink(outside, journal)
    with pytest.raises(validator.ValidationError, match="cannot safely open"):
        validator.validate_run(run)


def test_failure_cli_is_machine_readable_and_nonzero(tmp_path, capsys):
    run, _rows, _journal = _write_run(tmp_path)
    manifest = _manifest(run)
    manifest["partial_by_design"] = False
    _write_manifest(run, manifest)

    assert validator.main(["--run-dir", str(run)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert report["schema_version"] == 1
    assert any("partial_by_design" in error for error in report["errors"])
