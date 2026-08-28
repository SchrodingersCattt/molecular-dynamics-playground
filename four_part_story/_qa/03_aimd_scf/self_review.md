# Five-Problem Self-Review

## Render

- Figure: AIMD: electrons converge before nuclei move
- Script: four_part_story/render_aimd_scf.py
- Static inspected: four_part_story/figures/03_aimd_scf.png at full A4 landscape size
- Motion inspected: four_part_story/videos/03_aimd_scf.mp4 and the complete 1920 × 1080 contact sheet
- Programmatic result: 360 / 360 decoded frames passed at 24 fps; zero clipping, overlap, colour or typography failures

## Pass 1

1. Scientific expression problem: The previous layout did not connect each ionic step to a quantitative SCF convergence trace, and changing the contour count made real density evolution read as flashing.
2. Geometric or data-coordinate correctness problem: The A/B/C/D hierarchy needed to match the reference proportions, and position drift required a true previous-geometry ghost without putting structures inside either loop.
3. Label, leader, legend, or panel-letter placement problem: The dimer panel used too much explanatory copy; the Verlet equations belonged inside A, while B needed only the current iteration and action.
4. Readability or print-scale problem: The acceleration caption violated the right safety band, and contour bands appeared discontinuously between cached SCF states.
5. Submission-quality or filename/output-discipline problem: The final video needed renewed decoded-frame QA, current contact sheets, and a review hash tied to the new static render.

Action taken: Rebuilt the composition as A: formula-bearing Velocity Verlet loop; B: fixed-camera MatterVis water dimer; C: real RHF energy error versus SCF iteration; D: separate SCF loop. Kept one black status line in B, used thinner atom-centred arrows, and added a short true-coordinate ghost during position drift. Doubled both detailed convergence pauses while accelerating SCF and later ionic phases so the total remains 15 seconds.

## Pass 2

1. Scientific expression problem: Fixed; each ionic step has its own real energy-convergence trace, and density evolves from the computed RHF sequence using one stable eight-level contour set with residual-driven blur.
2. Geometric or data-coordinate correctness problem: Fixed; the molecule, density plane, two-colour bonds, vectors and ghost share one fixed MatterVis camera and world coordinate system, while both loops remain structure-free.
3. Label, leader, legend, or panel-letter placement problem: Fixed; A contains the active Verlet formula, B carries one black iteration/action line, C carries the energy trace, and D carries only the SCF loop.
4. Readability or print-scale problem: Fixed; the caption is inset without shrinking the circle, arrows remain readable but lighter, and fixed contour topology removes popping. Adjacent SCF frames show a maximum mean luminance change of 2.35 / 255.
5. Submission-quality or filename/output-discipline problem: Fixed; the canonical 15-second H.264 video contains 360 audited frames, and generated density renders, contact sheets and provenance remain under the declared QA tree.

Action taken: Inspected the full-size static render, all phase-boundary keyframes, the complete contact sheet and the decoded MP4. No blocking issue remains; proceed to the strict manifest gate and publish only on pass.

## Stop Conditions

- All five listed problems are fixed and no new blocking issue is visible.
- Any remaining choice is a genuine scientific or manuscript-style decision for the user.
- The source data are ambiguous and choosing a representation would assert an unsupported fact.
- The user explicitly accepts the current figure.
