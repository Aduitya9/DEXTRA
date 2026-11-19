"""Make the generated booktabs tables fit an IEEEtran page.

The notebooks emit every table as a single-column `table` float. IEEEtran is two-column,
so a table with four or more columns overflows. This does two things, idempotently:

  1. converts such a table to the full-width `table*` float and adds \\footnotesize;
  2. wraps the tabular in a \\resizebox that only ever SHRINKS, so a table that is still
     too wide is scaled to the text width instead of running off the page.

The shrink-only idiom is the standard one: \\ifdim\\width>\\textwidth ... \\else\\width\\fi,
so a table narrower than the page is left at its natural size rather than blown up.
Numbers and captions are untouched.
"""
import pathlib
import re
import sys

BEGIN, END = "\\begin{table}", "\\end{table}"
BEGIN_WIDE, END_WIDE = "\\begin{table*}", "\\end{table*}"
SMALL = "\\footnotesize"
TAB_BEGIN, TAB_END = "\\begin{tabular}", "\\end{tabular}"
RESIZE = "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%"

directory = pathlib.Path(sys.argv[1])
for path in sorted(directory.glob("*.tex")):
    text = path.read_text(encoding="utf-8")
    spec = re.search(re.escape(TAB_BEGIN) + r"\{([^}]*)\}", text)
    ncol = len(spec.group(1).replace(" ", "")) if spec else 0
    if ncol < 4:
        print(f"  {path.name:30s} {ncol} cols -> fits one column, unchanged")
        continue

    changed = []
    if BEGIN_WIDE not in text:
        text = text.replace(BEGIN, BEGIN_WIDE).replace(END, END_WIDE)
        changed.append("table*")
    if SMALL not in text:
        text = text.replace(TAB_BEGIN, SMALL + "\n" + TAB_BEGIN, 1)
        changed.append("footnotesize")
    if "\\resizebox" not in text:
        text = text.replace(TAB_BEGIN, RESIZE + "\n" + TAB_BEGIN, 1)
        text = text.replace(TAB_END, TAB_END + "%\n}", 1)
        changed.append("resizebox")

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"  {path.name:30s} {ncol} cols -> {' + '.join(changed)}")
    else:
        print(f"  {path.name:30s} {ncol} cols -> already adapted")
