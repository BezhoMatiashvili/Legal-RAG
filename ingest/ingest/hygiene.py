"""Corpus hygiene: repair scraped-body damage without touching content.

The corpus arrives with source-specific damage: napr PDF→text bodies embed NUL and other
control characters (~72% of docs), matsne HTML→Markdown carries `U+FFFD` mojibake in a few
hundred docs, and a handful of bodies are non-NFC. This module strips structural junk
(NUL / disallowed control chars), NFC-normalises Mkhedruli (idempotent), and classifies
damage so a document can be *quarantined with a reason* instead of silently dropped.

Personal data is never removed — only control characters and encoding artefacts. The
cleaning is applied to the text we embed/index; callers keep the raw body separately.
Pure stdlib: safe to import without the ML stack.
"""

import unicodedata
from dataclasses import dataclass

# Control characters to strip: C0 (0x00–0x1F) and C1 (0x80–0x9F), except the three
# whitespace controls we keep (tab, newline, carriage return). NUL is included here.
_KEEP = {0x09, 0x0A, 0x0D}
_CONTROL = set(range(0x00, 0x20)) | set(range(0x80, 0xA0))
_STRIP_TABLE = {c: None for c in (_CONTROL - _KEEP)}

REPLACEMENT_CHAR = "�"  # U+FFFD — the mojibake / lost-character marker

# Quarantine thresholds (tuned from the full-corpus profile, 2026-07-07).
NEAR_EMPTY_CHARS = 30    # < this many non-whitespace chars ⇒ not a usable document
MOJIBAKE_RATIO = 0.01    # ≥ 1% of chars are U+FFFD ⇒ text too corrupt to trust

# Quarantine reason codes (never delete — route to the quarantine file with one of these).
Q_EMPTY = "empty_body"
Q_NEAR_EMPTY = "near_empty"
Q_MOJIBAKE = "mojibake"


def strip_control(text: str) -> tuple[str, int]:
    """Remove NUL and disallowed C0/C1 control chars; return (cleaned, chars_removed)."""
    if not text:
        return text or "", 0
    cleaned = text.translate(_STRIP_TABLE)
    return cleaned, len(text) - len(cleaned)


def to_nfc(text: str) -> str:
    """Unicode NFC normalisation (idempotent: to_nfc(to_nfc(x)) == to_nfc(x))."""
    return unicodedata.normalize("NFC", text) if text else (text or "")


def clean_text(text: str) -> str:
    """Hygiene applied to indexed/embedded text: strip control chars, then NFC.

    Idempotent, and preserves all human-meaningful content (including personal data).
    """
    cleaned, _ = strip_control(text or "")
    return to_nfc(cleaned)


@dataclass(frozen=True)
class DamageReport:
    """What was wrong with a raw body, measured *before* cleaning."""

    length: int              # characters in the raw body
    meaningful_chars: int    # non-whitespace characters
    nul_chars: int           # NUL (U+0000) count
    control_chars: int       # all stripped C0/C1 controls (includes NUL)
    replacement_chars: int   # U+FFFD count
    replacement_ratio: float
    changed_by_nfc: bool
    quarantine_reason: str | None

    @property
    def is_usable(self) -> bool:
        return self.quarantine_reason is None


def assess(raw_body: str) -> DamageReport:
    """Classify a raw body's damage and decide whether it must be quarantined.

    NUL / control chars and non-NFC do NOT quarantine — those are repaired by
    ``clean_text``. Only an empty/near-empty body or overwhelming mojibake does.

    The empty/near-empty and mojibake-ratio decisions are measured on the
    *control-stripped* text — i.e. on what actually survives ``clean_text`` into the
    index — not on the raw body. Otherwise a body that is mostly NUL/control padding
    (common in napr PDF→text) would count that padding as "meaningful" and slip past
    the gate, only to be stripped to empty/near-empty at index time.
    """
    raw = raw_body or ""
    length = len(raw)
    stripped, control = strip_control(raw)      # what clean_text keeps (pre-NFC)
    meaningful = sum(1 for ch in stripped if not ch.isspace())
    nul = raw.count("\x00")
    repl = stripped.count(REPLACEMENT_CHAR)      # U+FFFD is not a control char, survives
    ratio = repl / len(stripped) if stripped else 0.0
    nfc_changed = to_nfc(stripped) != stripped

    reason: str | None = None
    if meaningful == 0:
        reason = Q_EMPTY
    elif meaningful < NEAR_EMPTY_CHARS:
        reason = Q_NEAR_EMPTY
    elif ratio >= MOJIBAKE_RATIO:
        reason = Q_MOJIBAKE

    return DamageReport(
        length=length,
        meaningful_chars=meaningful,
        nul_chars=nul,
        control_chars=control,
        replacement_chars=repl,
        replacement_ratio=ratio,
        changed_by_nfc=nfc_changed,
        quarantine_reason=reason,
    )
