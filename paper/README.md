# The paper

A compile-ready IEEEtran draft with all fourteen generated assets wired in.

## Build it on Overleaf (recommended)

1. Go to Overleaf → **New Project → Upload Project**, and upload this whole `paper/`
   folder as a zip.
2. Menu → **Compiler: pdfLaTeX**, **Main document: main.tex**.
3. Recompile.

Overleaf has IEEEtran built in, so nothing needs installing.

## Build it locally

```
pdflatex main && pdflatex main
```

Twice, so the cross-references resolve.

## What is here

```
main.tex        the paper
figures/        7 vector PDFs  (fig1-fig5 from notebook 09, figA-figB from notebook 10)
tables/         7 booktabs tables (table1-table5 from notebook 09, table6-table7 from notebook 10)
main.pdf        the draft as it currently compiles
```

The tables are the files the notebooks emit, with one presentational change applied by
`adapt_tables.py`: any table with four or more columns is promoted to the full-width
`table*` float, set `\footnotesize`, and wrapped in a shrink-only `\resizebox` so it
cannot run off the page. **No number or caption was altered.** If you regenerate a table
from the notebooks, re-apply that adaptation or the wide ones will overflow.

## State of the draft

It compiles clean: exit 0, zero overfull boxes, every cross-reference and citation
resolved, **9 pages**.

Two things to deal with:

- **Length.** Most IEEE conferences allow 6 pages, with fees beyond that. Nine pages
  with seven figures is a draft, not a submission. The cheapest cuts are figures:
  `figA_city_aqi` and `figB_coverage` are both full-width and could become one, and
  `table2_scores` largely repeats `table4_noise_ceiling`.
- **The 13 `\todo{}` markers.** They render in bold in the PDF so none can survive to
  submission by accident. They cover the author block, Section II (Related Work, which
  is a scaffold), the nightly-anchor definition, and the CPCB permission statement.

Every reference in the bibliography must be verified before submission, and the
air-quality forecasting literature still has to be added.

## Where the numbers come from

Sections III and V take every figure from the generated tables, which came from
notebooks 08, 09 and 10. Nothing was retyped by hand. If a number changes, regenerate
the table and the paper follows.
