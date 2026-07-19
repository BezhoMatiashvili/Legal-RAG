"""Persistent experiment log — every eval run is A/B-comparable forever.

Each run appends one JSON line: eval-set version + hash, a config hash (so identical
configs are recognisable across time), the mode, metrics, CIs, per-stage latency and
per-type breakdown, plus a fingerprint of the index it ran against. The config hash is a
stable digest of the retrieval-relevant knobs, mirroring ``snapshot.py``'s scheme.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = EVAL_DIR / "experiments.jsonl"

# Bump when scoring semantics change.  eval-r3 keeps legacy v1/v2 keys intact but makes
# canonical v3 document/chunk identities version-scoped and requires observed candidate
# recall for the strict accuracy track.
LOGIC_REV = "eval-r3"
SCHEMA_VERSION = "accuracy-eval/v2"


def config_hash(material: dict) -> str:
    """Stable full SHA-256 of the retrieval-relevant configuration."""
    payload = {"logic_rev": LOGIC_REV, **material}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run(record: dict, path: Path = DEFAULT_LOG) -> None:
    """Append one run record as a JSON line (created if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_log(path: Path = DEFAULT_LOG) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
