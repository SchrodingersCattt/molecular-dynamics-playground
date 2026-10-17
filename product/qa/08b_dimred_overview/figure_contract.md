# 08b_dimred_overview - figure contract (round 13: generic, method-agnostic, compact)

- Purpose: one generic schematic of the dimensionality-reduction pipeline that PCA, t-SNE and UMAP share. No method
  names and no method-specific symbols. The presenter adds per-method text in PowerPoint OUTSIDE the cards; the cards
  hold only a title row and the picture (no reserved note area, no dashed guide). 08 (3 x 5 per-method pipelines)
  stays as the detailed figure.
- One row, five cards, short generic titles with numbered discs:
  1 Data (point cloud + a star of many axes) -> 2 Relations (pairwise neighbor graph) -> 3 Objective (high-D vs
  low-D relation matrices, matched; labels "high-D" / "low-D") -> 4 Optimize (point trajectories + falling error curve)
  -> 5 Low-D map (final low-dimensional layout with clusters).
- Title 5 is "Low-D map", not "Embedding" or "Projection". It names the result plainly, it matches the "low-D" label in
  card 3, and it fits any method: "Projection" suggests a linear map, and "Embedding" is jargon that can also mean
  the vectors themselves.
- Data: seeded toy from scripts/build_box/generate_tsne_umap_toy.py; relation matrices are Gaussian affinities
  (median-distance scale), ordered by cluster; one recorded neighbor-embedding run gives trajectory and error curve.
- Canvas 338 x 72 mm (millimeter layout: 4 mm outer margin, 10 mm gaps holding the arrows, 3 mm picture inset).
  Static 300 dpi (3992 x 850); video 1920 x 410 (ppt copy 1280 x 274, 24 fps, CRF 30), 10.6 s, 254 frames, every-frame QA.
- No clipping: every picture is framed with frame_to() (equal aspect, 2.5 mm margin around the data extent of all
  drawn items; card 5 frames the final cluster ellipses as well as the current points). check_08b_clipping.py
  verifies every frame: ink stays >= 2 mm inside each card's rounded border (result in _qa/clip_check.json).
- Style: light-gray cards, dark-gray borders, dark-gray numbered discs and arrows, black Arial text >= 14 pt
  (titles 18 pt); color only on points, lines, curves and the relation matrices. American spelling.
