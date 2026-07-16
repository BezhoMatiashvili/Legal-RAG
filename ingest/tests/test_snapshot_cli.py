"""Focused CLI contract tests for immutable snapshot publication."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import ingest.__main__ as cli


def test_snapshot_parser_requires_identity_and_output_root(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ingest", "snapshot", "--preflight"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "--snapshot-id" in error
    assert "--output-root" in error


def test_snapshot_parser_forwards_safe_build_arguments(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(cli, "_cmd_snapshot", received.append)
    output_root = tmp_path / ".state" / "v3" / "snapshots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "snapshot",
            "--snapshot-id",
            "v3_cli_preflight",
            "--output-root",
            str(output_root),
            "--source",
            "all",
            "--preflight",
            "--limit",
            "25",
            "--no-near-dup",
            "--token-sample",
            "0",
        ],
    )
    cli.main()
    assert len(received) == 1
    args = received[0]
    assert args.snapshot_id == "v3_cli_preflight"
    assert args.output_root == Path(output_root)
    assert args.source_state_evidence is None
    assert args.source == "all"
    assert args.preflight is True
    assert args.limit == 25
    assert args.no_near_dup is True
    assert args.token_sample == 0


def test_snapshot_command_rejects_partial_source_before_build(monkeypatch):
    monkeypatch.setattr(cli, "_resolved_cfg", lambda _args: object())
    args = type(
        "Args",
        (),
        {
            "source": "ecd",
            "snapshot_id": "v3_partial",
            "output_root": Path("unused"),
            "source_state_evidence": None,
            "preflight": True,
            "limit": None,
            "no_near_dup": True,
            "token_sample": 0,
        },
    )()
    with pytest.raises(SystemExit, match="all seven sources"):
        cli._cmd_snapshot(args)
