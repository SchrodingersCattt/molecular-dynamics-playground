# Five-Problem Self-Review — 02 AIMD SCF

## Render

- Figure: `02_aimd_scf`
- Script: `render_slide2_aimd_scf.py`
- Output inspected: original 1920 × 1080 PNG plus MP4 frames at 1.0, 7.0, 13.5, and 15.2 s
- Inspection size / zoom: original resolution and PowerPoint-equivalent full slide
- Programmatic QA result: all machine gates PASS except the review action before this record was completed

## Pass 1

1. Scientific expression problem: the first layout did not make the nesting of the SCF loop inside one force query sufficiently explicit.
2. Geometric problem: an equal-aspect square coordinate system compressed both loops into the middle of the 16:9 canvas.
3. Annotation problem: the inner convergence node and labels initially collided with the outer Verlet loop.
4. Readability problem: the SCF iteration readout initially covered part of the dimer geometry.
5. Submission-quality problem: the raw line-only density contours were too faint to survive PowerPoint projection.

Action taken: rebuilt the axes as a widescreen isotropic coordinate system, separated the two loops, added one explicit force-query connector, moved the numerical readout below the dimer, and added restrained grey density bands behind the real contour lines.

## Pass 2

1. Scientific expression problem: no blocking issue remains; the figure shows density, Fock build, orbital solve, density update, convergence test, and the retry edge.
2. Geometric problem: no blocking issue remains; dimer coordinates, density contours, bonds, and force origins share one isotropic projection.
3. Annotation problem: no labels, nodes, equations, or connector arrows overlap at 1920 × 1080.
4. Readability problem: all labels remain at least 17–18 pt and the active node is unambiguous in sampled video frames.
5. Submission-quality problem: PNG, SVG, and the independent 16 s H.264 MP4 use the canonical stem and pass dimension/codec checks.

Action taken: accepted after strict QA, source-clipping PASS, and original-resolution inspection of early, middle, pre-converged, and converged video states.

## Stop Condition

The five listed problems are fixed and no new blocking issue is visible.
