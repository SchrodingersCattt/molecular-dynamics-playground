# Figure Contract - multi-task fine-tuning (09)

## Output
- Stem: `09_multitask_finetune`; PNG + MP4 (+ ppt/ MP4, GIF, poster). Canvas as 08 (4000 x 1250 static, 1920 x 600 video, 14.5 s).

## Focus
- One focal element: the shared DPA3 descriptor (3 RepFlow layers, signed heat strips, navy border).
- Each step draws one head with p = 0.5 (model_prob 0.5 / 0.5): the head lights, its gradient arrow runs into the descriptor and fans into all three RepFlow layers, the activations update. The other head stays gray.
- Static: both gradient arrows ("step t", "step t + 1") ending in the descriptor grid.

## Source
- `product/data/09_multitask_config.json` (copied from a800 `exp7a_fold0/input.json`): DPA3 repflow nlayers 3; heads `deepems_vanilla` (E/F, pretrained) and `pems_vdet_kj` (V_det, random init); model_prob 0.5 / 0.5.
- Heat values and the shown head sequence are an illustrative seeded draw, not training logs.

## De-emphasized
- Inputs (DFT frames, ion clusters), forward arrows and the step ribbon in gray.

## QA
- Same as 08.

## Round 10 revision (2026-10-17)
- Canvas 338 × 105.6 mm (16:5, same width as 08); gray cards with dark-gray borders; all text black.
- Focus kept on the shared DPA3 descriptor by a heavier (3.2 pt) dark-gray border and size, not by fill color; it flashes the active head's color on each update. The active head is marked by a colored border (a line) on a gray card; idle cards are white with a light border. Gradient arrows stay teal/orange.

## Round 11 revision (2026-10-17): verified scheme, forward + backward
- Verified read-only on a800 (deepmd-kit 04e3517b, deepmd/pt/train/training.py Trainer.run().step(), l.698-740):
  each step samples ONE model_key with p = model_prob (0.5 / 0.5 in exp7a), runs that head's forward pass and loss,
  loss.backward(), clip_grad_norm_(1.0), optimizer.step(). Losses are not summed. Evidence in product/data/09_multitask_config.json.
- Two task lanes (DFT frames -> E/F head -> E^, F^ -> ener loss vs DFT E, F; PEMs clusters -> V_det head -> V^_det -> MAE loss vs label),
  both through the one shared DPA3 descriptor. Per sampled step the active lane shows forward (dark-gray arrows), loss,
  backward (head-color arrows from the loss back through the head into all three RepFlow layers), then the update
  (descriptor border flash + activations change). The other lane stays idle.
- Canvas 338 x 125 mm (same as 08/08b); 6 sampled steps, 3 s each, 20.5 s video; every-frame QA.
