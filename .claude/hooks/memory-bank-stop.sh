#!/usr/bin/env bash
# Stop-hook reminder (fail-open, never blocks): if Python source under ingest/ or
# scraper/ was modified recently (~this session) but no memory-bank/ file was, nudge
# the session to update the memory per memory-bank/INDEX.md "same-session rule".
set -u
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

WINDOW_MIN=45
newest_code=$(git status --porcelain=v1 2>/dev/null | awk '{print $NF}' \
  | grep -E '^(ingest|scraper)/.*\.py$' \
  | xargs -r stat -c %Y -- 2>/dev/null | sort -n | tail -1)
[ -z "${newest_code:-}" ] && exit 0

now=$(date +%s)
age_min=$(( (now - newest_code) / 60 ))
[ "$age_min" -gt "$WINDOW_MIN" ] && exit 0   # not touched this session (heuristic)

newest_mb=$(find memory-bank -name '*.md' -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
if [ -z "${newest_mb:-}" ] || [ "$newest_mb" -lt "$newest_code" ]; then
  echo "memory-bank reminder: ingest/scraper .py files changed recently but no memory-bank/*.md was updated — apply the same-session rule (memory-bank/INDEX.md) or state why not."
fi
exit 0
