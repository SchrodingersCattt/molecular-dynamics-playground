# 08b_dimred_overview - figure contract (round 12: generic, method-agnostic)

- Purpose: one generic schematic of the dimensionality-reduction pipeline that PCA, t-SNE and UMAP share. No method
  names and no method-specific symbols; the presenter adds per-method notes in PowerPoint in the empty lower part of
  each card (below the faint dashed guide). 08 (3 x 5 per-method pipelines) stays as the detailed figure.
- One row, five cards, short generic titles with numbered discs:
  1 Data (point cloud + a star of many axes) -> 2 Relations (pairwise neighbour graph) -> 3 Objective (high-D vs
  low-D relation matrices, matched; labels "high-D" / "low-D") -> 4 Optimise (point trajectories + falling error curve)
  -> 5 Embedding (2-D clusters).
- Data: seeded toy from scripts/build_box/generate_tsne_umap_toy.py; relation matrices are Gaussian affinities
  (median-distance scale), ordered by cluster; one recorded neighbour-embedding run gives trajectory and error curve.
- Canvas 338 x 125 mm; static 300 dpi; video 1920 x 710 (ppt copy 1280 x 474, 24 fps, CRF 30), 10.6 s, every-frame QA.
- Style: light-grey cards, dark-grey borders, dark-grey numbered discs and arrows, black Arial text >= 14 pt
  (titles 18 pt); colour only on points, lines, curves and the relation matrices.
