# Georgia Legal Search — repo notes for Claude Code

**Multiple Claude Code sessions may run in this repo concurrently.** Before editing any
tracked file, read `coordination/README.md`, register yourself in
`coordination/sessions/`, and check other sessions' claimed files. Communicate via
`coordination/messages.md` (shell-append only). Watch `git status --short` for other
sessions' in-flight work — uncommitted changes you didn't make are NOT yours to revert,
"fix", or commit.

Read `HANDOFF.md` for project state and `improvement.md` for the gated retrieval
improvement queue. Standing rules: never commit unless the user asks; never pull/merge
`origin/dev` (divergent fork); never edit existing tests to make them pass.

## Project memory (auto-loaded)

@memory-bank/INDEX.md
@memory-bank/contracts.md

Before editing anything under `ingest/` or `scraper/`, run the pre-modification ritual
in the INDEX (open the matching `memory-bank/areas/` file + contract sections, grep for
callers). A change that touches any symbol named in memory-bank/ must update the
corresponding section in the SAME session; lint with
`python3 ingest/scripts/gen_code_map.py --check`. Standing exception (user-authorized
2026-07-09): sessions MAY commit **memory-bank/-only** changes without asking — one
commit per session, message prefix `memory:`.
