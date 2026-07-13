"""Live terminal progress panel for ingestion.

The embedding/upsert loop is otherwise silent-but-noisy: FlagEmbedding prints its
own ``pre tokenize`` / ``Inference Embeddings`` tqdm bars and ``httpx`` logs every
Qdrant POST at INFO, so the terminal scrolls fast with no sense of overall progress.

``IngestProgress`` replaces that with a single ``rich.Live`` dashboard — one row per
source (docs/total, chunks, skipped, rate, phase) plus a totals footer — modelled on
the scraper's live panel so the two stages look the same. It is purely additive: when
stdout is not a TTY (pipes, CI, cron) it disables itself and the pipeline falls back to
its plain tqdm bar, so non-interactive runs behave exactly as before.

It owns one ``rich.Live`` with ``auto_refresh`` and a time-based ``__rich__``, so the
spinner and elapsed clock keep animating even while the main thread is blocked inside a
multi-second ``model.encode`` call.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# (glyph, style, label) per phase. Active phases share the spinner glyph (filled in
# at render time); terminal phases get a fixed glyph.
_PHASE_STYLE = {
    "queued":    ("·", "dim", "queued"),
    "reading":   (None, "cyan", "reading"),
    "embedding": (None, "cyan", "embedding"),
    "upserting": (None, "cyan", "upserting"),
    "done":      ("✓", "bold green", "done"),
    "error":     ("■", "red", "error"),
}


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _rate_per_min(docs: int, elapsed_s: float) -> float:
    if elapsed_s < 1.0:
        return 0.0
    return docs * 60.0 / elapsed_s


def _quiet_background_noise() -> None:
    """Mute the per-request httpx logs and FlagEmbedding's own tqdm bars.

    Both would otherwise scribble over the live panel. Disabling tqdm globally is
    safe here: this is only called when our panel is active, and the pipeline's own
    fallback tqdm bar is only used when the panel is *off*.
    """
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from functools import partialmethod

        from tqdm.std import tqdm as _tqdm

        # tqdm.auto / tqdm.autonotebook resolve to this same class in a terminal, so
        # patching the base disable default silences FlagEmbedding's bars too.
        _tqdm.__init__ = partialmethod(_tqdm.__init__, disable=True)
    except Exception:  # pragma: no cover - never break ingest over a cosmetic patch
        pass


@dataclass
class _Row:
    source: str
    total: int = 0          # candidate docs (file lines, capped by --limit); 0 = unknown
    docs: int = 0           # docs successfully embedded + upserted
    chunks: int = 0
    skipped: int = 0
    phase: str = "queued"
    started: float | None = None
    finished_elapsed: float | None = None  # frozen elapsed once done

    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        if self.finished_elapsed is not None:
            return self.finished_elapsed
        return time.monotonic() - self.started


class IngestProgress:
    """Live per-source ingest dashboard. Use as a context manager.

    All mutators are no-ops when disabled (non-TTY), so the pipeline can call them
    unconditionally. Mutators run on the main thread; rendering runs on rich's
    refresh thread. State is plain ints/strs guarded by a lock — good enough under
    the GIL for a cosmetic display.
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled) and sys.stdout.isatty()
        self._rows: dict[str, _Row] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._live: Live | None = None

    # --- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "IngestProgress":
        if not self.enabled:
            return self
        _quiet_background_noise()
        self._start = time.monotonic()
        self._live = Live(self, auto_refresh=True, refresh_per_second=10, transient=False)
        self._live.start()
        return self

    def __exit__(self, *exc) -> bool:
        if self._live is not None:
            try:
                self._live.refresh()
            finally:
                self._live.stop()
                self._live = None
        return False

    # --- mutators (main thread) -------------------------------------------

    def add_source(self, source: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if source not in self._rows:
                self._rows[source] = _Row(source)
                self._order.append(source)

    def start_source(self, source: str, total: int = 0) -> None:
        if not self.enabled:
            return
        with self._lock:
            row = self._rows.setdefault(source, _Row(source))
            if source not in self._order:
                self._order.append(source)
            row.total = total
            row.started = time.monotonic()
            row.phase = "reading"

    def set_phase(self, source: str, phase: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            row = self._rows.get(source)
            if row is not None and row.phase not in ("done", "error"):
                row.phase = phase

    def update(self, source: str, *, docs: int, chunks: int, skipped: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            row = self._rows.get(source)
            if row is not None:
                row.docs, row.chunks, row.skipped = docs, chunks, skipped

    def finish_source(self, source: str, *, error: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            row = self._rows.get(source)
            if row is not None:
                row.phase = "error" if error else "done"
                row.finished_elapsed = row.elapsed()

    # --- rendering (rich refresh thread) ----------------------------------

    def __rich__(self) -> Panel:
        frame = _SPINNER_FRAMES[int(time.monotonic() * 10) % len(_SPINNER_FRAMES)]
        with self._lock:
            rows = [self._rows[name] for name in self._order]

        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True)                     # glyph
        table.add_column(style="bold", no_wrap=True)       # source
        table.add_column(justify="right", no_wrap=True)    # docs
        table.add_column(justify="right", style="dim", no_wrap=True)  # chunks
        table.add_column(justify="right", no_wrap=True)    # skipped
        table.add_column(justify="right", style="dim", no_wrap=True)  # rate
        table.add_column()                                 # phase

        header = ("", "SOURCE", "DOCS", "CHUNKS", "SKIP", "DOC/MIN", "PHASE")
        table.add_row(*(Text(h, style="bold dim") for h in header))

        total_docs = total_chunks = total_skipped = 0
        for row in rows:
            glyph, style, label = _PHASE_STYLE.get(row.phase, (None, "cyan", row.phase))
            if glyph is None:
                glyph = frame
            total_docs += row.docs
            total_chunks += row.chunks
            total_skipped += row.skipped

            if row.total:
                docs_cell = Text(f"{row.docs}", style="green" if row.docs else "dim")
                docs_cell.append(f"/{row.total}", style="dim")
            else:
                docs_cell = Text(str(row.docs), style="green" if row.docs else "dim")

            rate = _rate_per_min(row.docs, row.elapsed())
            skip_style = "yellow" if row.skipped else "dim"
            table.add_row(
                Text(glyph, style=style),
                Text(row.source),
                docs_cell,
                str(row.chunks),
                Text(str(row.skipped), style=skip_style),
                f"{rate:.0f}" if rate else "—",
                Text(label, style=style),
            )

        footer = Text()
        footer.append("total ", style="bold dim")
        footer.append(f"{total_docs} docs", style="bold green")
        footer.append(f" · {total_chunks} chunks", style="dim")
        footer.append(
            f" · {total_skipped} skipped",
            style="yellow" if total_skipped else "dim",
        )
        footer.append(
            f" · elapsed {_fmt_elapsed(time.monotonic() - self._start)}", style="dim"
        )

        return Panel(
            Group(table, Text(""), footer),
            title=Text(" ingesting → qdrant ", style="bold"),
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
        )
