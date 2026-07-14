"""Generate memory-bank/generated/symbols.md and lint memory-bank/ references.

Static AST only (stdlib `ast`) — never imports project code, so it runs under any
Python >= 3.9 (system python3, either venv) without torch/scrapy installed.

Usage:
    python ingest/scripts/gen_code_map.py            # (re)write symbols.md
    python ingest/scripts/gen_code_map.py --check    # lint, exit 1 on problems
    python ingest/scripts/gen_code_map.py --check --quiet   # one-line verdict

--check verifies, without writing anything:
  1. generated/symbols.md matches a fresh regeneration (structure drift is loud);
  2. every `path/to/file.py:symbol` reference in hand-written memory-bank/*.md
     resolves to a real file containing that symbol (def/class/assignment);
  3. every relative markdown link (and heading anchor) in memory-bank/ resolves.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEMORY_BANK = REPO / "memory-bank"
GENERATED = MEMORY_BANK / "generated" / "symbols.md"

# Repo-relative roots to scan (directories walked recursively, files taken as-is).
SCAN_ROOTS = [
    "ingest/ingest",
    "ingest/eval",
    "ingest/scripts",
    "ingest/serverless",
    "scraper/legal_scrapers",
    "run_all.py",
]
SKIP_DIRS = {"__pycache__", ".venv", "node_modules"}

# Top-level package name -> repo-relative dir, for resolving internal imports.
PACKAGE_DIRS = {
    "ingest": "ingest/ingest",
    "eval": "ingest/eval",
    "legal_scrapers": "scraper/legal_scrapers",
}

HEADER = """\
# Symbol map (AUTO-GENERATED — do not edit)

Regenerate: `python3 ingest/scripts/gen_code_map.py` · Lint: `--check`.
One section per module: internal-import edges, then top-level classes (with methods)
and functions, each anchored `file.py:LINE`. Line numbers here are kept fresh by the
generator; hand-written memory-bank files must use `path.py:symbol` anchors instead.
"""


def iter_py_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        p = REPO / root
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(
                f
                for f in sorted(p.rglob("*.py"))
                if not any(part in SKIP_DIRS for part in f.parts)
            )
    return files


def resolve_import(module: str | None, level: int, importer: Path) -> str | None:
    """Map an import statement to a repo-relative .py path, if internal."""
    if level:  # relative import: anchor at the importing file's package dir
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        target = module.replace(".", "/") if module else ""
        cand = (base / target) if target else base
    else:
        if not module:
            return None
        top = module.split(".")[0]
        if top not in PACKAGE_DIRS:
            return None
        cand = REPO / PACKAGE_DIRS[top] / "/".join(module.split(".")[1:])
    for c in (cand.with_suffix(".py"), cand / "__init__.py"):
        if c.is_file():
            return c.relative_to(REPO).as_posix()
    return None


def module_section(path: Path) -> str:
    rel = path.relative_to(REPO).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # e.g. a py3.14-only file under an old parser
        return f"## {rel}\n*(unparseable: {exc.msg} at line {exc.lineno})*\n"
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if (t := resolve_import(a.name, 0, path)) and t != rel:
                    imports.add(t)
        elif isinstance(node, ast.ImportFrom):
            if t := resolve_import(node.module, node.level, path):
                if t != rel:
                    imports.add(t)
    lines = [f"## {rel}"]
    if imports:
        lines.append("imports: " + ", ".join(sorted(imports)))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            lines.append(f"- class **{node.name}**({bases}) `{rel}:{node.lineno}`")
            lines.extend(
                f"  - def {m.name} `{rel}:{m.lineno}`"
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"- def **{node.name}** `{rel}:{node.lineno}`")
    return "\n".join(lines) + "\n"


def generate() -> str:
    return HEADER + "\n" + "\n".join(module_section(f) for f in iter_py_files())


# --- linting -----------------------------------------------------------------

SYMBOL_REF = re.compile(r"\b([\w./-]+\.py):([A-Za-z_][\w.]*)")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)?(#[^)\s]*)?\)")


def slugify(heading: str) -> str:
    s = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"[\s]+", "-", s)


def symbol_defined(py_text: str, symbol: str) -> bool:
    for part in symbol.split("."):
        pat = rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(part)}\b|^\s*{re.escape(part)}\s*[:=]"
        if not re.search(pat, py_text):
            return False
    return True


def lint() -> list[str]:
    problems: list[str] = []
    if not GENERATED.is_file():
        problems.append(f"{GENERATED.relative_to(REPO)} missing — run the generator")
    elif GENERATED.read_text(encoding="utf-8") != generate():
        problems.append("generated/symbols.md is stale — rerun gen_code_map.py")

    md_files = [
        f
        for f in sorted(MEMORY_BANK.rglob("*.md"))
        if GENERATED not in (f,) and "generated" not in f.parts
    ]
    py_cache: dict[str, str | None] = {}
    for md in md_files:
        rel_md = md.relative_to(REPO).as_posix()
        text = md.read_text(encoding="utf-8")
        for m in SYMBOL_REF.finditer(text):
            path_s, symbol = m.groups()
            target = REPO / path_s
            if path_s not in py_cache:
                py_cache[path_s] = (
                    target.read_text(encoding="utf-8") if target.is_file() else None
                )
            src = py_cache[path_s]
            if src is None:
                problems.append(f"{rel_md}: dead file ref {path_s}:{symbol}")
            elif not symbol_defined(src, symbol):
                problems.append(f"{rel_md}: symbol not found {path_s}:{symbol}")
        for m in MD_LINK.finditer(text):
            href, anchor = m.groups()
            if href and re.match(r"^[a-z]+:", href):  # http:, mailto:, …
                continue
            target_md = (md.parent / href).resolve() if href else md
            if href and not target_md.exists():
                problems.append(f"{rel_md}: dead link {href}")
                continue
            if anchor and target_md.suffix == ".md" and target_md.exists():
                heads = re.findall(
                    r"(?m)^#{1,6}\s+(.*)$", target_md.read_text(encoding="utf-8")
                )
                if anchor[1:] not in {slugify(h) for h in heads}:
                    problems.append(f"{rel_md}: dead anchor {href or ''}{anchor}")
    return problems


def main(argv: list[str]) -> int:
    check, quiet = "--check" in argv, "--quiet" in argv
    if not check:
        GENERATED.parent.mkdir(parents=True, exist_ok=True)
        GENERATED.write_text(generate(), encoding="utf-8")
        print(f"wrote {GENERATED.relative_to(REPO)}")
        return 0
    problems = lint()
    if quiet:
        print(
            "memory refs: OK"
            if not problems
            else f"memory refs: {len(problems)} stale (python3 ingest/scripts/gen_code_map.py --check)"
        )
    else:
        for p in problems:
            print(p)
        print("memory refs: OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
