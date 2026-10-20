#!/usr/bin/env python3
"""One-shot initializer for a new PathSim toolbox created from this template.

Renames the placeholder package to your toolbox name, rewrites every
reference across the repository, strips the template banner from the README,
and finally deletes this script and TEMPLATE.md.

Usage:
    python init.py <name> [--description "..."] [--label "..."]

Example:
    python init.py rf --description "RF engineering toolbox for PathSim" --label RF

Arguments:
    name           short lowercase identifier, e.g. "rf", "chem", "vehicle".
                   Becomes package "pathsim_<name>", distribution
                   "pathsim-<name>", and docs slug "docs.pathsim.org/<name>".
    --description   one-line project description (pyproject.toml / README).
    --label         display capitalization for "PathSim-<label>"
                    (default: capitalized name; pass e.g. "RF" for acronyms).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()

# Directory names that are never walked or modified.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def iter_files() -> list[Path]:
    """Every file under ROOT, skipping build artefacts, .git and this script."""
    files = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.resolve() != SELF:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a new PathSim toolbox from this template."
    )
    parser.add_argument("name", help="short lowercase identifier, e.g. 'rf'")
    parser.add_argument("--description", default=None, help="one-line description")
    parser.add_argument("--label", default=None, help="display capitalization")
    args = parser.parse_args()

    name = args.name.strip()
    if not re.fullmatch(r"[a-z][a-z0-9]*", name):
        fail(f"name '{name}' must be lowercase letters/digits and start with a letter")

    label = args.label or name.capitalize()
    description = args.description or f"{label} toolbox for PathSim"

    pkg_dir = ROOT / "src" / "pathsim_toolbox"
    if not pkg_dir.is_dir():
        fail("src/pathsim_toolbox not found — already initialized?")

    # Ordered text replacements. The docs-slug rewrite runs first because it
    # is the only pattern containing a bare 'toolbox'.
    replacements = [
        ("docs.pathsim.org/toolbox", f"docs.pathsim.org/{name}"),
        ("pathsim_toolbox", f"pathsim_{name}"),
        ("pathsim-toolbox", f"pathsim-{name}"),
        ("PathSim-Toolbox", f"PathSim-{label}"),
        ("A toolbox for PathSim", description),
    ]

    # 1. Rename the package directory.
    new_pkg_dir = ROOT / "src" / f"pathsim_{name}"
    pkg_dir.rename(new_pkg_dir)
    print(f"renamed  src/pathsim_toolbox -> src/pathsim_{name}")

    # 2. Strip the template banner from the README.
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        text = re.sub(
            r"<!-- TEMPLATE:START -->.*?<!-- TEMPLATE:END -->\n*",
            "",
            text,
            flags=re.DOTALL,
        )
        readme.write_text(text, encoding="utf-8")
        print("updated  README.md (removed template banner)")

    # 3. Rewrite placeholders in every text file.
    changed = 0
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text = text
        for old, new in replacements:
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
    print(f"rewrote  {changed} file(s)")

    # 4. Remove template-only files (this script last).
    template_md = ROOT / "TEMPLATE.md"
    if template_md.exists():
        template_md.unlink()
        print("removed  TEMPLATE.md")
    SELF.unlink()
    print("removed  init.py")

    print()
    print(f"Done. 'pathsim-{name}' is initialized.")
    print("Next steps:")
    print("  - replace the FirstOrderLag example block with your own blocks")
    print("  - replace the example notebook in docs/source/examples/")
    print("  - register the toolbox in pathsim/docs (scripts/lib/config.py)")
    print("  - git add -A && git commit")


if __name__ == "__main__":
    main()
