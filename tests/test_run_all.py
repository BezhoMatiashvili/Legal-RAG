import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_all import CORPUS_SOURCES, build_watch_command, selected_ingest_sources  # noqa: E402


def _args(**overrides):
    values = {"only": None, "collection": None, "poll_interval": 5.0}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_watcher_uses_only_production_corpus_sources():
    command = build_watch_command(_args())

    assert command[command.index("--source") + 1] == ",".join(CORPUS_SOURCES)
    assert "supremecourt" not in command


def test_only_subset_scopes_watcher_and_preserves_canonical_order():
    command = build_watch_command(_args(only=["tbappeal", "ecd"], collection="test"))

    assert command[command.index("--source") + 1] == "ecd,tbappeal"
    assert command[command.index("--collection") + 1] == "test"


def test_supremecourt_only_is_rejected_for_ingest_launcher():
    with pytest.raises(SystemExit, match="intentionally excluded"):
        selected_ingest_sources(["supremecourt"])


def test_unknown_spider_fails_before_children_start():
    with pytest.raises(SystemExit, match="unknown spider"):
        selected_ingest_sources(["not-a-spider"])
