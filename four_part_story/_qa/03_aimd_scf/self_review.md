# Five-Problem Self-Review

After every substantive render, assume the figure is still wrong. List exactly
five concrete problems before deciding whether to rerender. Automated QA passing
is not enough.

## Render

- Figure: Ab initio MD: one force requires an SCF loop
- Script: `four_part_story/render_aimd_scf.py`
- Output file inspected: `four_part_story/figures/03_aimd_scf.png`
- Inspection size / zoom: full A4 landscape render plus 100% crop inspection of the dimer, density, arrows, and SCF labels
- Programmatic QA result: pending final strict check after the fixes below

## Pass 1

1. Scientific expression problem: The old concept exposed only six density samples and did not demonstrate the full 19-step RHF convergence history.
2. Geometric or data-coordinate correctness problem: The old dimer camera compressed both water geometries, and force arrows began inside the atomic spheres.
3. Label, leader, legend, or panel-letter placement problem: The former five-node SCF ring and numeric detail competed with the density and did not read as one compact loop.
4. Readability or print-scale problem: Dark red force arrows visually merged with the oxygen atoms and were too short at A4 scale.
5. Submission-quality or filename/output-discipline problem: Provisional test renders and stale projected-density panels were mixed with the canonical source assets.

Action taken: Recomputed all 19 real RHF density iterates on one grid; retained raw and residual-blurred display Cubes separately; switched molecule, density, and vector rendering to MatterVis; chose an asymmetric oblique camera; moved the SCF ring outside the molecule; simplified it to four nodes; and used larger, tail-offset, pale dusty-rose force arrows.

## Pass 2

Repeat after fixes and a fresh render.

1. Scientific expression problem: Fixed; the frame uses the converged member of a traceable 19-Cube sequence, and the metadata discloses residual-driven blur as display encoding only.
2. Geometric or data-coordinate correctness problem: Fixed; both H-O-H shapes are visible in a non-[111] view, while every force arrow is anchored to its calculated nuclear position by MatterVis.
3. Label, leader, legend, or panel-letter placement problem: Fixed; the four-node SCF loop sits clear of the density and structure, with no molecule inside either loop.
4. Readability or print-scale problem: Fixed; the force color is distinct from oxygen red and the offset shafts and arrowheads remain visible at A4 scale.
5. Submission-quality or filename/output-discipline problem: Fixed for the canonical static outputs and source panels; provisional test files are excluded and will be removed before the video commit.

Action taken: No new blocking issue is visible in the fresh static render. Proceed to strict manifest QA, then generate and inspect every 1920x1080 video frame before publication.

## Stop Conditions

Stop only when one of these is true:

- The five listed problems are all fixed and no new blocking issue is visible.
- The remaining choice is a genuine scientific or manuscript-style decision for
  the user.
- The source data are ambiguous and choosing a representation would assert a fact
  not present in the data.
- The user explicitly accepts the current figure.
