# Figure Contract - multi-task fine-tuning (09)

## Output
- Stem: `09_multitask_finetune`; PNG + MP4 (+ ppt/ MP4, GIF, poster). Canvas as 08 (4000 x 1250 static, 1920 x 600 video, 14.5 s).

## Focus
- One focal element: the shared DPA3 descriptor (3 RepFlow layers, signed heat strips, navy border).
- Each step draws one head with p = 0.5 (model_prob 0.5 / 0.5): the head lights, its gradient arrow runs into the descriptor and fans into all three RepFlow layers, the activations update. The other head stays grey.
- Static: both gradient arrows ("step t", "step t + 1") ending in the descriptor grid.

## Source
- `product/data/09_multitask_config.json` (copied from a800 `exp7a_fold0/input.json`): DPA3 repflow nlayers 3; heads `deepems_vanilla` (E/F, pretrained) and `pems_vdet_kj` (V_det, random init); model_prob 0.5 / 0.5.
- Heat values and the shown head sequence are an illustrative seeded draw, not training logs.

## De-emphasised
- Inputs (DFT frames, ion clusters), forward arrows and the step ribbon in grey.

## QA
- Same as 08.

## Round 10 revision (2026-10-17)
- Canvas 338 × 105.6 mm (16:5, same width as 08); grey cards with dark-grey borders; all text black.
- Focus kept on the shared DPA3 descriptor by a heavier (3.2 pt) dark-grey border and size, not by fill colour; it flashes the active head's colour on each update. The active head is marked by a coloured border (a line) on a grey card; idle cards are white with a light border. Gradient arrows stay teal/orange.
