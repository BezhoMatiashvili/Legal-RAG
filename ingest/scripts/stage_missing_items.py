#!/usr/bin/env python
"""Stage the raw items of ONLY the not-yet-embedded docs, per source, for a GPU delta.

Reads `.state/embed_missing.txt` (``source\\tdocument_id`` lines from
verify_all_embedded.py), scans each source's run items, normalizes to derive the same
``document_id`` embed_delta uses, and writes the raw item JSON of the missing docs to
``<out>/<source>.jsonl`` (last run wins on duplicate ids). Tiny output (only the gaps),
so shipping it to a pod is cheap — unlike the 157k-line matsne run set.

    .venv/bin/python scripts/stage_missing_items.py --out .state/delta_stage
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.config import load_config  # noqa: E402
from ingest.sources import normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", default=".state/embed_missing.txt")
    ap.add_argument("--out", default=".state/delta_stage")
    args = ap.parse_args()

    cfg = load_config()
    want: dict[str, set[str]] = defaultdict(set)
    for line in Path(args.missing).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src, _, did = line.partition("\t")
        want[src].add(did)
    print("missing to stage:", {s: len(v) for s, v in want.items()})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grand = 0
    for src, ids in want.items():
        runs = sorted((cfg.artifacts_root / src / "runs").glob("*/items.jsonl"))
        picked: dict[str, str] = {}  # document_id -> raw json line (last run wins)
        scanned = 0
        for path in runs:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    scanned += 1
                    try:
                        item = json.loads(line)
                        doc = normalize(src, item)
                    except Exception:  # noqa: BLE001
                        continue
                    if doc.document_id in ids:
                        picked[doc.document_id] = line
        dst = out / f"{src}.jsonl"
        dst.write_text("\n".join(picked.values()) + ("\n" if picked else ""), encoding="utf-8")
        miss = ids - set(picked)
        grand += len(picked)
        print(f"  {src}: staged {len(picked)}/{len(ids)} (scanned {scanned} items)"
              + (f"  ⚠ {len(miss)} not found in runs: {sorted(miss)[:5]}" if miss else ""))
    print(f"staged {grand} docs total → {out}")


if __name__ == "__main__":
    main()
