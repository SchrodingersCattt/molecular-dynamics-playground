# Self-review of the round-8 explainers (rejected) and what 08/09 change

Round-8 files: `talk_tsne_umap_principle.png`, `talk_multitask_finetune.png` (PROGRESS_REPORT/_redraw_fig345/make_explainers.py).

## Palette
- The series palette is in `scripts/md_visuals/common.py`: INK #171717, DARK_GRAY #505050, MID_GRAY #8E8E8E, LINE_GRAY #C8C8C8,
  LIGHT_GRAY #EEEEEE, NAVY #183153, CRIMSON #A32035, GREEN #2F6B4F, R_BLUE #4F7CAC, V_PURPLE #8169AD, A_ORANGE #C8834A,
  ENERGY_TEAL #1F7A7A. The NNMD scenes (04/05/04a/05a) also use NNMD_ENERGY_TEAL #3F8A87, NNMD_ORANGE #CF9A6E and
  SIGNED_CMAP (#A9C97E, #D6E9B8, #FFFFFF, #DDD3EE, #AE9BD3).
- The 05a cards use pale tints with softened borders: STRUCTURE_BG #F7F9FA, INPUT_FILL #E7EEF6, PREDICTION_FILL #EEF6F5 with
  SOFT_TEAL #9CC6C2, LOSS_FILL #FFF9F4 with SOFT_ORANGE #E9BC96, and UPDATED_FILL #E5F0E8.
- Round 8 retyped a subset by hand and used it differently from the series:
  - flat grey cards (LIGHT_GRAY fill, MID_GRAY edge) in place of tinted cards;
  - the dark ENERGY_TEAL #1F7A7A, not the softer NNMD teal used in 05a;
  - V_PURPLE and R_BLUE as head and curve colours, although the series reserves them for the r/a/v integrator stages and categories;
  - no navy accent and no signed colormap.
- Next to the 05a and 04/05 slides, the result looked like a different deck.

## Fonts, line weights, layout
- Arial and 14 pt or larger were respected, but most labels sat at the 14 pt minimum. Each card carried two or three text lines plus formulas.
- Lines were 1.2–2.0 pt. The series uses 1.1 pt card borders, about 3 pt flow arrows and heavy accent strokes.
- Rounded cards of different sizes were packed into a 2 × 3 grid (t-SNE/UMAP), plus a five-column chain with loss formulas (multitask). Neither layout has a fixed slot grid like 04/05a.

## Why the flow was unclear
- t-SNE/UMAP: each row was "input | curves | output", but the steps of the algorithm were not separated.
  - p_ij, the kernel and the loss all sat in one card, and nothing numbered the order.
  - The arrows were 1.6 pt with small heads.
  - The points were small (s ≈ 70 px² at 200 dpi) and drawn by hand, so "projection → embedding" was asserted, not shown.
- Multitask: there was no focal element.
  - The descriptor, both heads, both losses, the gradient arcs and an extra "Each step: one head, 0.5 / 0.5" box all had the same weight.
  - The alternating gradient was two thin dashed arcs at the edge of the figure.
  - The loss formulas competed with the main idea.

## Resolution
- Both PNGs were saved at 200 dpi on a 338 mm canvas (2661 × 929 and 2661 × 881 px). On the 12.4 in slot that is about 215 px/in, so the thin lines and 14 pt text were softened by PowerPoint's resampling.
- The 08/09 statics are 4000 × 1250 px, 300 dpi at 13.33 in (≥ 320 px/in in the 12.4 in slot).

## 08/09 changes
- The palette and card style are imported from `common.py`; the 05a tints are copied verbatim into `explainer_common.py`. Text is 14–18 pt, regular weight.
- t-SNE/UMAP:
  - Numbered five-stage rows with big arrows: points → similarities → low-dim kernel → loss → embedding.
  - 24 large toy points from a real seeded optimisation.
  - The video reveals the stages in order and moves the points as the loss dot runs down.
- Multitask:
  - The shared DPA3 descriptor (three RepFlow strips) is the single focus, with a thick navy border.
  - Per step, one head lights and a thick gradient arrow runs in and fans into every RepFlow layer, which then updates.
  - Inputs and the step ribbon are grey; there are no loss formulas.
- Both videos are rendered by `common.render_video`, which does the every-frame QA and writes the contact sheets.

## Round 10
- Added a PCA row (linear baseline) parallel to t-SNE and UMAP; figure 338 × 125 mm, fonts ≥14 pt at that size.
- Unified card style (light-grey fill, dark-grey border, dark-grey discs); all text labels black.
- 09: same card style applied. Keeping it did not break the focus, because the descriptor stays the largest card with the heaviest border. Head identity now lives in the border colour rather than in a coloured fill or text.

## Round 11
- 08 kept; new 08b one-row overview (shared 5-stage abstraction, three methods stacked per card) is the default on slide 21.
- 09 redrawn from the verified DeePMD-kit training loop: serial stochastic alternation (one head per step), forward and backward both shown.

## Round 12
- 08b replaced by a generic, method-agnostic schematic (Data / Relations / Objective / Optimise / Embedding) with room
  under each card for the presenter's per-method notes; the stacked three-method version was rejected.
