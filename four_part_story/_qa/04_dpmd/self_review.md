# Five-Problem Self-Review — 04 Deep Potential MD

## Pass 1

1. Scientific expression: the initial plan stopped at symbolic `E` and `F`, so it did not demonstrate that this specific water box had actually passed through a learned potential.
2. Geometry: the first render packed oversized atoms into the periodic box and made the 6 Å sphere too faint to read as a three-dimensional object.
3. Annotation: the first render placed full words inside small MD circles, covered pipeline text with node fills, and put the numerical inference label too close to the last node.
4. Readability: the first video layout reset the cutoff sphere at stage boundaries and used a stage-four heading wide enough to touch both panel edges.
5. Submission quality: the first static caption crossed both inter-panel safety bands, and the first complete video attempt was correctly rejected at frame 288.

Action taken: ran the 192-atom box with DeepMD-kit 3.1.3 in a time-limited Bohrium sandbox; retained and hashed the energy and force arrays; rebuilt the MD loop with symbols inside and labels outside; strengthened the true 3D sphere; separated sphere, outside fade, neighbor spokes, and force weights; shortened the stage heading; and split the center caption inside its panel.

## Pass 2

1. Scientific expression: no blocking issue remains; the same stable source IDs define the 64-water box, central atom 126, 83 exact minimum-image neighbors, atomic energies, total energy, and force vectors.
2. Geometry: no blocking issue remains; bond and neighbor views share one isotropic orthographic projection from `[1.55, -1.0, 0.62]`, and the cutoff is a projected 3D sphere rather than a decorative ellipse.
3. Annotation: no blocking issue remains; every label clears its node and panel edge, and all arrows are large enough to resolve at target size.
4. Readability: no blocking issue remains; the video uses 18 pt minimum text, keeps the structure large, and changes pale grey stages to black while navy, crimson, and green retain fixed meanings.
5. Submission quality: no blocking issue remains; A4 source/composite strict checks pass, MatterVis provenance is retained, and all 384 final video frames pass.

Action taken: accepted after inspecting the final A4 render plus bond, cutoff, neighbor, frame-288, and full-force frames at native output dimensions.

## Stop Condition

The five listed problems are fixed and no new blocking issue is visible.
