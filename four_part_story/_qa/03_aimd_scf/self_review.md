# Five-Problem Self-Review

## Render

- Figure: AIMD: electrons converge before nuclei move
- Script: four_part_story/render_aimd_scf.py
- Static inspected: four_part_story/figures/03_aimd_scf.png at full A4 landscape size
- Motion inspected: four_part_story/videos/03_aimd_scf.mp4 plus all contact sheets at 1920 × 1080
- Programmatic result: 360 / 360 video frames passed; final strict manifest check pending

## Pass 1

1. Scientific expression problem: The previous animation showed one electronic solve and did not establish that SCF is repeated at every ionic geometry.
2. Geometric or data-coordinate correctness problem: The molecular-plane camera made several real three-dimensional vectors project as circular caps instead of readable arrows.
3. Label, leader, legend, or panel-letter placement problem: The transition from SCF convergence to force, velocity half-kick and position drift was not explicit at phase boundaries.
4. Readability or print-scale problem: Fixed-size arrowheads consumed short physical vectors, while the old volumetric density made the molecular scene visually dirty.
5. Submission-quality or filename/output-discipline problem: The canonical video and QA report still described one ionic update rather than an auditable multi-step trajectory.

Action taken: Generated seven RHF/STO-3G geometries with every real SCF density frame, analytic gradients and six verified 0.5 fs Velocity Verlet updates. Replaced the density volume with a residual-driven sparse-to-fine 2D contour projection. Selected one fixed oblique camera from projected-vector readability, changed MatterVis vectors to proportional arrowheads, and separated SCF, pause, force, velocity and position phases.

## Pass 2

1. Scientific expression problem: Fixed; the first ten seconds show two complete SCF cycles and the last five seconds show four additional ionic updates.
2. Geometric or data-coordinate correctness problem: Fixed; density, atoms, two-colour bonds and atom-centred vectors share one camera and world coordinate system, with fixed vector scales preserving relative lengths.
3. Label, leader, legend, or panel-letter placement problem: Fixed; each ionic step and phase has one direct heading, the molecule remains outside both loops, and the 18 px panel safety bands pass.
4. Readability or print-scale problem: Fixed; coarse, intermediate and converged density states are visibly distinct, and olive, emerald and lake-blue arrows remain legible at 1920 × 1080.
5. Submission-quality or filename/output-discipline problem: Fixed; the canonical 15 s H.264 video contains 360 audited frames, and all source renders and contact sheets are kept under _qa.

Action taken: No new blocking issue was found in the final-size static render, the phase-boundary keyframes, the eight contact sheets or the decoded MP4. Proceed to the strict manifest gate and publish only if it passes.

## Stop Conditions

- All five listed problems are fixed and no new blocking issue is visible.
- Any remaining choice is a genuine scientific or manuscript-style decision for the user.
- The source data are ambiguous and choosing a representation would assert an unsupported fact.
- The user explicitly accepts the current figure.
