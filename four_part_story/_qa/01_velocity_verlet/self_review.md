# Five-Problem Self-Review

After every substantive render, assume the figure is still wrong. List exactly
five concrete problems before deciding whether to rerender. Automated QA passing
is not enough.

## Render

- Figure: Velocity Verlet, abstract loop plus one real H2O step
- Script: `four_part_story/render_velocity_verlet.py`
- Output file inspected: `four_part_story/figures/01_velocity_verlet.png` and all video contact sheets
- Inspection size / zoom: 3508 x 2480 at 100%; representative 1920 x 1080 frames at 100%
- Programmatic QA result: static strict QA and 216/216 video-frame audits pass

## Pass 1

1. Scientific expression problem: the first redesign drew vectors after rendering, so the force and motion marks were not part of the molecular 3D scene.
2. Geometric or data-coordinate correctness problem: the first loop used axis-normalized circles that appeared elliptical on the A4 canvas.
3. Label, leader, legend, or panel-letter placement problem: oversized 2D arrowheads obscured atoms and visually detached from their physical anchors.
4. Readability or print-scale problem: the real molecule and its single-step motion were too small relative to the available right column.
5. Submission-quality or filename/output-discipline problem: the earlier video QA and review hash referred to a superseded render and did not document native vectors.

Action taken: moved every world-space vector to MatterVis native `vector_overlays`, used a physically compressed H2O starting geometry so the real restoring forces remain visible, rebuilt the circular loop in display geometry, enlarged the scene, and regenerated static/video QA artifacts.

## Pass 2

Repeat after fixes and a fresh render.

1. Scientific expression problem: no remaining blocking issue; the three displayed equations match the numerically verified Velocity Verlet update and the arrows come from its saved displacement, acceleration, and velocity arrays.
2. Geometric or data-coordinate correctness problem: no remaining blocking issue; atoms, two-colour bonds, and vectors share one fixed MatterVis orthographic camera with no 2D reprojection.
3. Label, leader, legend, or panel-letter placement problem: no remaining blocking issue; the molecule stays outside the loop, headings own separate whitespace, and no arrow collides with text.
4. Readability or print-scale problem: no remaining blocking issue; the header-free A/B/C/D layout keeps the molecule dominant and every static/video label stays within the common 10--16 pt range.
5. Submission-quality or filename/output-discipline problem: no remaining blocking issue; canonical PNG, SVG, MP4, source panels, MatterVis provenance, 216-frame audit, and five contact sheets are present.

Action taken: replaced the old two-column composition with the shared A/B/C/D geometry, moved the state summaries into C/D, fixed Simulation step at B lower-left, removed the page header, and stopped after the fresh render because no new blocking issue is visible.

## Stop Conditions

Stop only when one of these is true:

- The five listed problems are all fixed and no new blocking issue is visible.
- The remaining choice is a genuine scientific or manuscript-style decision for
  the user.
- The source data are ambiguous and choosing a representation would assert a fact
  not present in the data.
- The user explicitly accepts the current figure.
