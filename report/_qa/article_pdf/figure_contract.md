# Figure Contract

## Output

- Canonical article: md_reading_notes.tex
- Canonical PDF: output/pdf/md_reading_notes.pdf
- QA contact sheet: article_pages.png
- Final format: nine A4 portrait pages

## Typography and hierarchy

- Layout reference: SC-XRD manuscript template
- Body: 10 pt class setting; no text below 10 pt
- Chinese: Noto Serif CJK SC / Noto Sans CJK SC
- Latin: Liberation Serif
- Section titles: question-led, parallel in grammar and abstraction
- Subsection titles: object or method plus its physical or evidential role
- Header/footer: restrained running header and page number

## Required content

- Initial-value formulation
- Integrator geometry and Velocity Verlet
- Empirical, ab initio and machine-learning potential-energy surfaces
- Sampling and enhanced sampling
- Layered treatment of integration, model, ensemble and statistical errors
- Complete bibliography

## Forbidden content

- Appendix titled 最小数值自查清单
- A standalone subsection titled 与 Runge--Kutta 方法比较
- Empty process headings such as checklist or workflow
- Unresolved references, clipped text or stale page renders

## QA plan

- XeLaTeX/Biber compilation
- Poppler render at 144 dpi
- PDFInfo page-size and page-count check
- PDFFonts embedding check
- TeX log scan for overfull, underfull, undefined and error messages
- PDF text extraction
- Per-page edge-clearance analysis
- Contact-sheet and representative-page visual inspection

## Delivery gate

- [x] PDF builds without errors or box warnings
- [x] All fonts are embedded
- [x] Nine page renders match the final PDF
- [x] No clipping or overlap is visible
- [x] The title hierarchy follows the reference logic
- [x] The unwanted appendix is absent
