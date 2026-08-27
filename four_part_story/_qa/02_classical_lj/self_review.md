# Five-Problem Self-Review

After every substantive render, assume the figure is still wrong. List exactly
five concrete problems before deciding whether to rerender. Automated QA passing
is not enough.

## Render

- Figure: classical MD, TIP3P O--O Lennard--Jones term on a real water dimer
- Script: `four_part_story/render_classical_lj.py`
- Output inspected: `four_part_story/figures/02_classical_lj.png` and all video contact sheets
- Inspection size / zoom: 3508 x 2480 at 100%; representative 1920 x 1080 frames at 100%
- Programmatic QA target: static strict QA and 216/216 video-frame audits

## Pass 1

1. Scientific expression problem: the first draft showed a disconnected Ar pair and an LJ curve rather than the requested water-dimer force inside a real integration step.
2. Geometric or data-coordinate correctness problem: a square MatterVis canvas made the dimer too small, while the O--O guide was projected with a mismatched canvas aspect ratio.
3. Label, leader, legend, or panel-letter placement problem: radial displacement and velocity vectors collided with the O--O guide and O--H bonds.
4. Readability or print-scale problem: the physical arrows were hard to distinguish and the bottom TIP3P scope note sat below the required clearance.
5. Submission-quality or filename/output-discipline problem: the earlier MP4 and frame audit still referred to the superseded radial-motion render.

Action taken: replaced the disconnected case with a real H2O dimer; used the analytic TIP3P O--O LJ force; rendered displacement, force, and velocity with MatterVis native `vector_overlays`; widened the render; corrected the O--O projection; chose a real transverse VV motion; and raised the scope note.

## Pass 2

1. Scientific expression problem: no remaining blocking issue; the saved O--O distance feeds the LJ energy, analytic force, acceleration, and one verified rigid-water Velocity Verlet step.
2. Geometric or data-coordinate correctness problem: no remaining blocking issue; atoms, red/white half-bonds, and all physical arrows share the same fixed asymmetric MatterVis camera, while the O--O guide uses the matching source aspect ratio.
3. Label, leader, legend, or panel-letter placement problem: no remaining blocking issue; transverse native arrows remain clear of the O--O guide, O--H bonds, headings, and left abstract loop.
4. Readability or print-scale problem: no remaining blocking issue; A4 text is at least 10 pt, video text is at least 18 pt, and arrowheads remain conspicuous without numeric magnification labels.
5. Submission-quality or filename/output-discipline problem: no remaining blocking issue after regenerating the canonical PNG, SVG, MP4, MatterVis provenance, 216-frame audit, and all-frame contact sheets.

Action taken: stop after the fresh render and full-frame audit because all first-pass problems are fixed and no new blocking issue is visible.

## Stop Conditions

Stop only when one of these is true:

- The five listed problems are all fixed and no new blocking issue is visible.
- The remaining choice is a genuine scientific or manuscript-style decision for the user.
- The source data are ambiguous and choosing a representation would assert a fact not present in the data.
- The user explicitly accepts the current figure.
