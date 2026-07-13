import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from runpod_orchestrate_delta import stage_delta_items  # noqa: E402


def _mk_run(runs_dir: Path, run_id: str, lines: list[str] | None) -> Path:
    d = runs_dir / run_id
    d.mkdir(parents=True)
    p = d / "items.jsonl"
    if lines is not None:
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def test_stage_selects_since_and_orders_by_run_id(tmp_path):
    runs = tmp_path / "runs"
    _mk_run(runs, "20260709T080007Z_x", ['{"document_id":"1"}'])
    _mk_run(runs, "20260709T105827Z_x", ['{"document_id":"2"}'])
    _mk_run(runs, "20260708T000000Z_old", ['{"document_id":"0"}'])  # before `since` — excluded

    staged = stage_delta_items(runs, "20260709T080007Z", tmp_path / "stage")

    names = [p.name for p in staged]
    assert names == ["20260709T080007Z_x.jsonl", "20260709T105827Z_x.jsonl"]  # ascending run id
    assert all(p.exists() and p.stat().st_size > 0 for p in staged)


def test_stage_skips_missing_and_empty_items(tmp_path):
    runs = tmp_path / "runs"
    _mk_run(runs, "20260709T100000Z_ok", ['{"document_id":"7"}'])
    _mk_run(runs, "20260709T110000Z_empty", [])       # zero-byte items.jsonl — skipped
    (runs / "20260709T120000Z_noitems").mkdir(parents=True)  # run dir without items.jsonl

    staged = stage_delta_items(runs, "20260709T080007Z", tmp_path / "stage")

    assert [p.name for p in staged] == ["20260709T100000Z_ok.jsonl"]


def test_stage_explicit_items_override_since(tmp_path):
    runs = tmp_path / "runs"
    kept = _mk_run(runs, "20260709T100000Z_a", ['{"document_id":"1"}'])
    _mk_run(runs, "20260709T110000Z_b", ['{"document_id":"2"}'])

    staged = stage_delta_items(runs, None, tmp_path / "stage", explicit=[kept])

    assert [p.name for p in staged] == ["20260709T100000Z_a.jsonl"]
    assert staged[0].read_text(encoding="utf-8") == '{"document_id":"1"}\n'


def test_stage_restages_cleanly_on_rerun(tmp_path):
    runs = tmp_path / "runs"
    _mk_run(runs, "20260709T100000Z_a", ['{"document_id":"1"}'])
    stage = tmp_path / "stage"

    stage_delta_items(runs, None, stage)
    _mk_run(runs, "20260709T110000Z_b", ['{"document_id":"2"}'])
    staged = stage_delta_items(runs, None, stage)

    assert [p.name for p in staged] == [
        "20260709T100000Z_a.jsonl", "20260709T110000Z_b.jsonl"]
    assert sorted(p.name for p in stage.iterdir()) == [p.name for p in staged]
