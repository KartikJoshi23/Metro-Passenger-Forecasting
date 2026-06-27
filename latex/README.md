# LaTeX documents → PDF

Two compilable documents (standard packages only: geometry, booktabs, xcolor, titlesec, enumitem,
hyperref):

| File | Purpose |
|---|---|
| `main.tex` | Concise project summary → PDF for **NotebookLM** (slide-deck source). Embeds 3 figures. |
| `presentation_guide.tex` | **Presentation playbook & Q&A** — 4-part talk plan, every concept/algorithm/metric explained in plain language, and a graded easy→advanced question bank. Text-only (no figures needed). |

## Required files
`main.tex` embeds three figures. Upload these alongside `main.tex` (they live in the **repo root**):
`demandsignal.png`, `modelcomparison.png`, `validationchart.png`.
(`\graphicspath` looks in both the current folder and the parent, so it works on Overleaf and locally.)

## Easiest: Overleaf (no install)
1. Go to https://www.overleaf.com → **New Project → Blank Project**.
2. Replace the contents of `main.tex` with this folder's `main.tex`.
3. **Upload** the three `.png` files from the repo root into the same project.
4. It compiles automatically → **Download PDF**.

## Local (if you have a TeX distribution)
```bash
cd latex
pdflatex main.tex
pdflatex main.tex      # run twice so hyperref/refs settle
```
Produces `main.pdf`. (On Windows, install MiKTeX or TeX Live first.)

The report is intentionally short (~3 pages) so the resulting slide deck stays concise.
