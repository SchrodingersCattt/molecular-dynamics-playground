# 04a training story

`04a_training.mp4` is a 16 s, 1920 × 600, 24 fps teaching animation for one
neural-network potential training step. It uses the same state-0 64-water
periodic box and O126 local environment as the 04 Deep Potential story.

The story is organised as one causal path:

```text
structure → local environment → forward prediction → reference comparison
          → loss → backpropagated gradient → parameter update → new prediction
```

The left panel reuses the 04 geometry directly. The periodic box contains the
O126 cutoff circle, two guide lines, and a magnified local view from the same
snapshot. The panel is titled `Training dataset`; the former schematic footer
has been removed. The 0–2 s sample phase reveals the box first, then the guide
lines and magnifier; the complete composition remains visible for the rest of
the training step.

The middle panel shows the descriptor entering `G¹`, `G²`, and `fit`. The
scalar output node is merged with the prediction element in the comparison
panel, so one circular node carries the network output and its black label.
The grids use a deterministic mock value field and a shared `−1 … +1` colour
bar. During backpropagation the same colour mapping changes with the mock
gradient values, making the numerical change visible without adding `w₁/w₂/w₃`
parameter boxes.

The right panel is deliberately staged so that only the current teaching
quantity is prominent. It stays quiet during sample/zoom and the forward pass;
when the prediction is ready it shows three stacked elements: the DFT truth at
the top, the network prediction in the middle (aligned with the output path),
and the error at the bottom. The circular node and prediction card are one
merged element. A vertical orange double arrow links truth and prediction;
the bottom element carries `|delta_i|` and `L_i` without an additional axis or
comparison plot.
Text is black by default; colour is reserved for the small stage marker and
for graphical borders, arrows, and activation paths. Descriptor and hidden
layers remain grids, while the scalar atomic energy output is a circular node
with a black symbol and numeric label below it. Solid cross-panel paths start
at the local environment, connect to the DFT truth, and enter the descriptor.
During backpropagation the loss arrow returns to the merged output/prediction
node, then continues through fit, G², G¹, and the descriptor. Grid colour
changes carry the mock gradient/update values, so separate parameter cards are
not needed.
The per-atom values are a small teaching fixture in
`product/data/04a_training_demo.json`. The repository does not
contain a raw per-atom DFT decomposition, so these values remain labelled
`DFT reference · schematic` and are not presented as measured DFT atomic
energies or as DeepMD output.

The displayed relations are:

```text
delta_i = epsilon_hat_i - epsilon_ref_i
L_i = 1/2 delta_i^2
w' = w - eta * grad_w L
E_NN = sum_i epsilon_i
```

The timeline is sample/zoom (0–2 s), forward pass (2–4.5 s), comparison
(4.5–6.5 s), loss (6.5–8.5 s), backpropagation (8.5–11.5 s), weight update
(11.5–13.5 s), and the reduced-error forward pass (13.5–16 s).
