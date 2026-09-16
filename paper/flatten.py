"""Flatten a multi-file manuscript into one .tex file for submission systems.

Inlines \\input{...} (section files) and \\inputrows{...} (generated table rows)
recursively, and rewrites figure paths so that every figure sits beside the
manuscript. The result compiles on its own with pdfLaTeX or tectonic.

    python paper/flatten.py --main <paper>/main.tex --out <dir>/manuscript.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

INPUT = re.compile(r"^[ \t]*\\input\{([^}]+)\}[ \t]*$", re.M)
ROWS = re.compile(r"\\inputrows\{([^}]+)\}")
GRAPHICS = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)figures/([^}]+)(\})")


def resolve(base: Path, name: str) -> Path:
    p = base / name
    return p if p.suffix else p.with_suffix(".tex")


def expand(text: str, base: Path, depth: int = 0) -> str:
    if depth > 8:
        raise RuntimeError("input nesting too deep")

    def sub_input(m: re.Match) -> str:
        path = resolve(base, m.group(1))
        return expand(path.read_text().rstrip("\n"), base, depth + 1)

    def sub_rows(m: re.Match) -> str:
        # Row files end with a newline that would close the final table row twice.
        return resolve(base, m.group(1)).read_text().rstrip("\n")

    text = INPUT.sub(sub_input, text)
    return ROWS.sub(sub_rows, text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-figure-dir", action="store_true",
                    help="keep figures/ in \\includegraphics paths instead of flattening them")
    args = ap.parse_args()
    main_path = Path(args.main).resolve()
    out = Path(args.out)
    text = expand(main_path.read_text(), main_path.parent)
    if not args.keep_figure_dir:
        text = GRAPHICS.sub(r"\1\2\3", text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"{out}: {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
