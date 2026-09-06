# Five-Problem Self-Review

## Render

- Figure: Deep Potential molecular dynamics
- Script: `three_slide_story/render_slide3_deep_potential.py`
- Output file inspected: `three_slide_story/figures/03_deep_potential_md.png`
- Inspection size / zoom: original 1920 × 1080 plus five original-resolution video frames
- Programmatic QA result: preflight passed all gates except the intentionally stale review hash; final result recorded after Pass 2

## Pass 1

1. Scientific expression problem: the first render said “83 atoms inside,” although the mask contains 83 neighbors plus the central source atom.
2. Geometric or data-coordinate correctness problem: an early camera trial was close to a symmetric 111 view and flattened the distinction between cell depth and the cutoff sphere.
3. Label, leader, legend, or panel-letter placement problem: the cutoff label and the bottom neighbor caption needed to refer to the same exact 6.0 Å selection.
4. Readability or print-scale problem: MatterVis hydrogen atoms were nearly invisible against pure white in the independent QA render.
5. Submission-quality or filename/output-discipline problem: the failed Bohrium upload left a duplicate 56.9 MB model and copied input payloads under the QA directory.

Action taken: changed the caption to “83 neighbors,” selected the fixed asymmetric camera `[1.55, -1.0, 0.62]`, kept one `r_cut = 6.0 Å` definition throughout, used a light-grey MatterVis QA background while preserving the white final slide, and removed duplicate upload payloads after preserving the job recipe and failure record.

## Pass 2

Repeat after fixes and a fresh render.

1. Scientific expression problem: resolved — the figure now distinguishes 83 neighbors from the separately highlighted central atom and makes no equilibration or force-value claim.
2. Geometric or data-coordinate correctness problem: resolved — bonds, cell, sphere, and neighbors share one isotropic orthographic transform; selection uses 3D minimum-image distances, not projected distance.
3. Label, leader, legend, or panel-letter placement problem: resolved — the cutoff label is clear of atoms and the five-stage rail maps unambiguously to the active visual element.
4. Readability or print-scale problem: resolved — 17–39 pt typography remains readable at slide size; inactive content is visible but subordinate, and active content turns black.
5. Submission-quality or filename/output-discipline problem: resolved — canonical PNG/SVG/MP4 names are used; intermediate MatterVis, Bohrium, and extracted-frame evidence remains only under `_qa/03_dpmd/`.

Action taken: accepted after inspecting the final static render and all five video stages at 1920 × 1080, then reran strict QA and bound the review to the final PNG hash.

## Stop Conditions

Stop only when one of these is true:

- The five listed problems are all fixed and no new blocking issue is visible.
- The remaining choice is a genuine scientific or manuscript-style decision for
  the user.
- The source data are ambiguous and choosing a representation would assert a fact
  not present in the data.
- The user explicitly accepts the current figure.
