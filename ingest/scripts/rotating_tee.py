#!/usr/bin/env python3
"""Mirror stdin to stdout and a bounded owner-only operational log."""

from __future__ import annotations

import argparse
import logging
import os
import stat
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

MAX_BYTES = 50 * 1024 * 1024
BACKUPS = 10


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        descriptor = os.open(
            self.baseFilename,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        return os.fdopen(
            descriptor,
            "a",
            encoding=self.encoding,
            errors=self.errors,
        )


def _prepare_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        current = path.lstat()
    except FileNotFoundError:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.close(descriptor)
        return
    if not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode) & 0o077:
        raise PermissionError(
            f"refusing non-private or non-regular existing log without chmod: {path}"
        )


def mirror_stream(
    path: Path,
    source: TextIO,
    destination: TextIO,
    *,
    max_bytes: int = MAX_BYTES,
    backups: int = BACKUPS,
) -> None:
    if max_bytes < 1 or backups < 1:
        raise ValueError("rotation bounds must be positive")
    _prepare_log(path)
    logger = logging.getLogger(f"rotating-tee:{id(source)}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = _PrivateRotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        for line in source:
            destination.write(line)
            destination.flush()
            logger.info(line.rstrip("\n"))
    finally:
        handler.close()
        logger.handlers.clear()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    try:
        mirror_stream(args.log, sys.stdin, sys.stdout)
    except (OSError, ValueError) as exc:
        print(f"rotating log refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
