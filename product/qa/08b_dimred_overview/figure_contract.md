# 08b_dimred_overview - figure contract

- Purpose: one-slide abstract overview of PCA / t-SNE / UMAP for a short talk; 08 (3 x 5 pipelines) stays as the detailed backup.
- One row of five cards = the shared abstraction: (1) high-dim data (one shared 10-D toy cloud) -> (2) structure
  (covariance ellipse | neighbour probabilities p_ij | fuzzy kNN graph) -> (3) objective (max variance | min KL(P||Q) |
  min cross-entropy) -> (4) solve (eigen-decomposition scree | gradient-descent loss | SGD loss) -> (5) 2-D embedding.
  Inside each card the three methods are stacked in bands aligned with the PCA / t-SNE / UMAP labels on the left.
- Data: same seeded toy as 08 (product/data/08_tsne_umap_toy.json).
- Canvas 338 x 125 mm; static 300 dpi; video 1920 x 710 (ppt copy 1280 x 474, 24 fps, CRF 30), 11.5 s, every-frame QA.
- Style: light-grey cards, dark-grey borders, dark-grey step discs and arrows, black Arial text >= 14 pt; colour only on points, curves, bars.
