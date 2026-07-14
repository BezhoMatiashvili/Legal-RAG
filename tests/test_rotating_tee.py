from __future__ import annotations

import importlib.util
import io
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ingest" / "scripts" / "rotating_tee.py"
SPEC = importlib.util.spec_from_file_location("rotating_tee", SCRIPT)
rotating_tee = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rotating_tee)


def test_mirrors_and_rotates_with_bounded_private_backups(tmp_path):
    log = tmp_path / "state" / "daily.log"
    mirrored = io.StringIO()

    rotating_tee.mirror_stream(
        log,
        io.StringIO("aaaa\nbbbb\ncccc\n"),
        mirrored,
        max_bytes=9,
        backups=2,
    )

    assert mirrored.getvalue() == "aaaa\nbbbb\ncccc\n"
    assert log.read_text(encoding="utf-8") == "cccc\n"
    assert (log.parent / "daily.log.1").read_text(encoding="utf-8") == "bbbb\n"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert not (log.parent / "daily.log.3").exists()


def test_refuses_insecure_existing_log_without_chmod(tmp_path):
    log = tmp_path / "daily.log"
    log.write_text("owner evidence\n", encoding="utf-8")
    log.chmod(0o644)

    try:
        rotating_tee.mirror_stream(log, io.StringIO("new\n"), io.StringIO())
    except PermissionError:
        pass
    else:
        raise AssertionError("insecure existing log should be refused")

    assert stat.S_IMODE(log.stat().st_mode) == 0o644
    assert log.read_text(encoding="utf-8") == "owner evidence\n"
