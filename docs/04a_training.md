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
snapshot. The 0–2 s sample phase reveals the box first, then the guide lines
and magnifier; the complete composition remains visible for the rest of the
training step.

The middle panel shows the descriptor entering `G¹`, `G²`, and `fit`. During
backpropagation an orange pulse travels from the predicted energy back through
the layers. Each layer is connected to its parameter card, which displays
`∂L/∂w₁`, `∂L/∂w₂`, or `∂L/∂w₃` while the parameter values remain fixed. During
the following update phase the cards change from old to new values with a
vertical arrow. This keeps gradient evaluation and parameter update visually
separate.

The right panel is deliberately staged so that only the current teaching
quantity is prominent. It stays quiet during sample/zoom and the forward pass;
when the prediction is ready it shows three stacked elements: the DFT truth at
the top, the network prediction in the middle (aligned with the output path),
and the error at the bottom. The prediction value remains visible in the
circular network output node, while the bottom element carries `|delta_i|` and
`L_i` without an additional axis or comparison plot.
Text is black by default; colour is reserved for the small stage marker and
for graphical borders, arrows, and activation paths. Descriptor and hidden
layers remain grids, while the scalar atomic energy output is a circular node
with a black symbol and numeric label below it. A cross-panel path starts at
the central O126 atom and enters the descriptor. During backpropagation the
loss arrow first returns to the output node, then continues through fit, G²,
G¹, and the descriptor. Parameter cards use shaded swatches for magnitude and
colour for sign, with an arrow between the pre-update and post-update swatch;
they do not print a separate numeric value for every weight.
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
