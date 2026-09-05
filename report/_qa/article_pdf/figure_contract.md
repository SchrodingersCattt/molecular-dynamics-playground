# Figure Contract

Fill this before writing or revising a plotting script. Keep it near the figure
script or copy it into the working notes for the task.

## Output

- Canonical stem: `md_reading_notes` (delivery) and `article_pages` (QA contact sheet)
- Formats: LaTeX, BibTeX, PDF and QA PNG
- Final files: `md_reading_notes.tex`, `references.bib`, `output/pdf/md_reading_notes.pdf`
- Intermediate files with leading underscore: all page renders and checks under `_qa/` or `output/rendered/`
- Files that must be removed before delivery: TeX auxiliary files (`.aux`, `.bcf`, `.blg`, `.out`, `.run.xml`)

## Target Size

- Target context: single-column academic reading note
- Physical size: A4 portrait, 210 x 297 mm
- DPI or vector export rule: vector text/equations; Poppler QA rerender at 120 dpi
- Minimum text size at final size: 10 pt (11 pt body)

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| pages | Full reading note | 15 complete A4 pages, citations, equations, figures, bibliography | Uniform margins and isotropic scaling |

## Required Elements

- Data layers: prose, equations, tables, four existing MD figures, bibliography
- Highlights: navy/scarlet/lake hyperlinks only
- Labels: section headings, equation/figure/table numbers, page numbers
- Leaders/arrows: inherited only from QA-passed source figures
- Legends: inherited only from QA-passed source figures
- Scale bars / axis keys / compass keys: inherited from source figures
- Panel letters: subfigure labels where used
- Animation labels or frame timing, if any: not applicable

## Forbidden Changes

- Do not drop: caveats, citations, model publication status, or reference entries
- Do not reorder: the initial-value to integrator to PES to sampling narrative
- Do not crop away: prose, equations, captions, tables, bibliography or page numbers
- Do not smooth, normalize, or rescale without stating: embedded scientific figures
- Do not use non-proportional stretching: pages or embedded figures
- Do not replace raw/experimental content without user approval: SCF/MD numerical sources

## Style

- Font family: SimSun for Chinese, Latin Modern for Latin/math
- Font sizes: 11 pt body; no body/table text below 10 pt
- Line widths: inherited from source figures
- Marker sizes: inherited from source figures
- Palette and semantic color mapping: navy links, scarlet citations, lake URLs
- Background: white
- Shared axes/camera/viewport rules: embedded figure geometry unchanged

## QA Plan

- Programmatic helpers to call: Poppler, PDFInfo, Biber, `visualize-data --strict`
- Safety bands to check: all four edges on every rendered page and contact-sheet margins
- Label/leader panels to validate: source figure QA records plus PDF text extraction
- Raster/vector inspection command: `pdftoppm -png -r 120`
- Human self-review file: `self_review.md`
- Outstanding deviations log: none

## Delivery Gate

- [x] Canonical outputs exist with requested names.
- [x] No unwanted duplicate variants remain.
- [x] Programmatic QA passes or deviations are logged and approved.
- [x] Final image was inspected at intended physical size.
- [x] Five-problem self-review was completed after the latest render.
