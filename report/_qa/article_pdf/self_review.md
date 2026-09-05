# Five-Problem Self-Review

After every substantive render, assume the figure is still wrong. List exactly
five concrete problems before deciding whether to rerender. Automated QA passing
is not enough.

## Render

- Figure: `article_pages.png` (15-page contact sheet)
- Script: XeLaTeX/Biber build of `md_reading_notes.tex`
- Output file inspected: `report/output/pdf/md_reading_notes.pdf`
- Inspection size / zoom: A4 pages rerendered at 120 dpi; contact sheet at 1250 x 1746 px
- Programmatic QA result: strict PASS after final rerender

## Pass 1

1. Scientific expression problem: method status needed to distinguish papers, preprints, and the DPA3 software release.
2. Geometric or data-coordinate correctness problem: the integrator table declared eight columns for seven fields and exceeded the text block.
3. Label, leader, legend, or panel-letter placement problem: long English model names and WHAM/MBAR produced unsafe line wrapping.
4. Readability or print-scale problem: the initial MLIP table width plus cell padding exceeded the page width.
5. Submission-quality or filename/output-discipline problem: the first compiled PDF contained unresolved citations before Biber and needed a clean final build.

Action taken: corrected table geometry, added safe breakpoints, ran Biber, rebuilt until references stabilized, rerendered all pages, and reran strict QA. Final log has no overfull boxes, missing references, or LaTeX errors.

## Stop Conditions

Stop only when one of these is true:

- The five listed problems are all fixed and no new blocking issue is visible.
- The remaining choice is a genuine scientific or manuscript-style decision for
  the user.
- The source data are ambiguous and choosing a representation would assert a fact
  not present in the data.
- The user explicitly accepts the current figure.
