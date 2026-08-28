# Five-Problem Self-Review

## Render

- Figure: AIMD: electrons converge before nuclei move
- Script: four_part_story/render_aimd_scf.py
- Static inspected: four_part_story/figures/03_aimd_scf.png at full A4 landscape size
- Motion inspected: four_part_story/videos/03_aimd_scf.mp4 and the complete 1920 × 1080 contact sheet
- Programmatic result: 360 / 360 decoded frames passed at 24 fps; every registered font is 10–16 pt; zero clipping, overlap, colour or typography failures

## Pass 1

1. Scientific expression problem: Electronic-iteration text was mixed into the molecular panel, obscuring the nesting of SCF inside each simulation step.
2. Geometric or data-coordinate correctness problem: Removing the page header required all four panels to move upward without changing their reference proportions or the fixed MatterVis camera.
3. Label, leader, legend, or panel-letter placement problem: B needed one stable physical-action line and a fixed lower-left Simulation step label, while all iteration/action detail belonged in C.
4. Readability or print-scale problem: The earlier 18–36 pt hierarchy and heavy loop arrows conflicted with the requested compact 10–16 pt visual system.
5. Submission-quality or filename/output-discipline problem: The new header-free layout needed renewed decoded-frame QA, contact sheets and a matching review hash.

Action taken: Removed the page header and moved A/B/C/D upward at the reference proportions. B now carries only one physical-action line plus fixed lower-left `Simulation step NN`; C exclusively carries the real SCF energy trace, iteration number and active electronic action; D retains only the SCF schematic. Kept the fixed-camera MatterVis dimer, true density sequence, thinner atom-centred arrows and short true-coordinate position ghost.

## Pass 2

1. Scientific expression problem: Fixed; each simulation step has its own real energy-convergence trace in C, while B remains free of electronic-step text and density evolves from the computed RHF sequence using one stable eight-level contour set.
2. Geometric or data-coordinate correctness problem: Fixed; the molecule, density plane, two-colour bonds, vectors and ghost share one fixed MatterVis camera and world coordinate system, while both loops remain structure-free.
3. Label, leader, legend, or panel-letter placement problem: Fixed; A contains the active Verlet formula, B carries the physical action and Simulation step, C carries all electronic-step text and the energy trace, and D carries only the SCF loop.
4. Readability or print-scale problem: Fixed; all registered text is 10–16 pt, shared loop arrows are lighter, and fixed contour topology removes popping.
5. Submission-quality or filename/output-discipline problem: Fixed; the canonical 15-second H.264 video contains 360 audited frames, and generated density renders, contact sheets and provenance remain under the declared QA tree.

Action taken: Inspected the full-size static render, all phase-boundary keyframes, the complete contact sheet and the decoded MP4. No blocking issue remains; proceed to the strict manifest gate and publish only on pass.

## Stop Conditions

- All five listed problems are fixed and no new blocking issue is visible.
- Any remaining choice is a genuine scientific or manuscript-style decision for the user.
- The source data are ambiguous and choosing a representation would assert an unsupported fact.
- The user explicitly accepts the current figure.
