# Figure Contract - t-SNE and UMAP principle (08)

## Output
- Stem: `08_tsne_umap_principle`; PNG + MP4 (+ ppt/ MP4, GIF, poster).
- Static: 13.333 x 4.167 in at 300 dpi (4000 x 1250 px), text in slide points, for a 12.4 in wide slot.
- Video: 1920 x 600 (16:5), 24 fps, 16.5 s; PPT copy 1280 x 400, CRF 30.

## Panels
| Row | Stages (left to right, numbered discs, big arrows) |
|---|---|
| t-SNE | 1 10-D points (PCA view) -> 2 Gaussian neighbors p_ij -> 3 Student-t q_ij (heavy tail vs Gaussian) -> 4 KL(P||Q) falling -> 5 embedding |
| UMAP | 1 10-D points -> 2 fuzzy kNN graph -> 3 low-dim q_ij (edges attract / others repel) -> 4 cross-entropy falling -> 5 embedding, "more global layout" |

## Required
- Real optimization of the seeded toy (`generate_tsne_umap_toy.py`), both runs started from the PCA view; points in panel 5 follow the recorded trajectory; the loss dot is synchronized with it.
- Palette, fonts and card style from `common.py` / 05a (via `explainer_common.py`); Arial; no bold; text 14-18 pt.

## Forbidden
- Hand-placed embedding points, decorative colors outside the series palette, numbers on the slide, text below 14 pt.

## QA
- `common.render_video` every-frame QA (layout registry, pixel checks, vertical whitespace) must pass; contact sheets reviewed; static layout registry must pass.

## Round 10 revision (2026-10-17)
- Three parallel rows: PCA (10-D points → center, covariance → principal axes PC1/PC2 → variance explained → projection), t-SNE, UMAP. Stage 1 is the same oblique 2-D linear view of the 10-D toy cloud in every row.
- Canvas 338 × 125 mm, 144 dpi video (1920 × 710 px master; PowerPoint copy 1280 × 474, 24 fps, CRF 30); static 300 dpi. All text ≥14 pt at that physical size (titles 15 pt, labels 14–18 pt).
- Cards: light-gray fill, dark-gray (#505050) border; step discs and arrows dark gray when active, light gray when idle. All text black; color only on points, curves, bars and arrows.
- Duration 22.5 s (540 frames); every-frame QA.
