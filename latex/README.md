# LaTeX report → PDF (for NotebookLM)

`main.tex` is a concise, self-contained project summary intended to be compiled to PDF and
uploaded to **NotebookLM** as a source for generating a slide deck. It uses only standard
packages (geometry, booktabs, xcolor, titlesec, enumitem, hyperref).

## Easiest: Overleaf (no install)
1. Go to https://www.overleaf.com → **New Project → Blank Project**.
2. Replace the contents of `main.tex` with this folder's `main.tex`.
3. It compiles automatically → **Download PDF**.

## Local (if you have a TeX distribution)
```bash
cd latex
pdflatex main.tex
pdflatex main.tex      # run twice so hyperref/refs settle
```
Produces `main.pdf`. (On Windows, install MiKTeX or TeX Live first.)

The report is intentionally short (~3 pages) so the resulting slide deck stays concise.
