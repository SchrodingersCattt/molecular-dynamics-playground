# Five-Problem Self-Review

After every substantive render, assume the figure is still wrong. List exactly
five concrete problems before deciding whether to rerender. Automated QA passing
is not enough.

## Render

- Figure: Deep Potential MD, four-panel local-environment story on one retained 64-water case
- Script: four_part_story/render_dpmd.py
- Output file inspected: four_part_story/figures/04_deep_potential_md.png and all video contact sheets
- Inspection size / zoom: 3508 x 2480 at 100%; representative 1920 x 1080 frames at 100%
- Programmatic QA result: static strict QA and 384/384 video-frame audits

## Pass 1

1. Scientific expression problem: the preceding composition mixed the local-environment selection and learned-model evaluation in one wide panel, obscuring the causal sequence from cutoff to force.
2. Geometric or data-coordinate correctness problem: the preceding two-column slots did not preserve the same A/B/C/D geometry used by the AIMD reference composition.
3. Label, leader, legend, or panel-letter placement problem: the water-box action, neighbour summary, and model stages competed for the same visual hierarchy.
4. Readability or print-scale problem: the page header consumed vertical space and the older video typography exceeded the requested 16 pt ceiling.
5. Submission-quality or filename/output-discipline problem: the QA manifest still described a two-panel figure and omitted the separate C/D source panels.

Action taken: adopted the shared header-free A/B/C/D slots; kept the concrete water box in B with a top physical-action line and lower-left Simulation step; isolated the cutoff summary in C and learned model in D; exported all four exact source panels; and enforced 10--16 pt typography.

## Pass 2

Repeat after fixes and a fresh render.

1. Scientific expression problem: no remaining blocking issue in the composition; the same saved centre, minimum-image neighbour mask, DeepMD energy, and force arrays drive B, C, and D.
2. Geometric or data-coordinate correctness problem: no remaining blocking issue; the 6 Å sphere, 83-neighbour selection, periodic cell, and force origins use one retained 3D coordinate set.
3. Label, leader, legend, or panel-letter placement problem: no remaining blocking issue; B contains only the physical action and fixed simulation-step label, while C/D own the local and model abstractions.
4. Readability or print-scale problem: no remaining blocking issue; the header-free layout keeps the water box dominant and all static/video labels remain within 10--16 pt.
5. Submission-quality or filename/output-discipline problem: no remaining blocking issue after regenerating canonical PNG, SVG, MP4, four QA source panels, 384-frame audit, and all-frame contact sheets.

Action taken: stop after the fresh render and full-frame audit because all five first-pass problems are fixed and no new blocking layout issue is visible.

## Stop Conditions

Stop only when one of these is true:

- The five listed problems are all fixed and no new blocking issue is visible.
- The remaining choice is a genuine scientific or manuscript-style decision for
  the user.
- The source data are ambiguous and choosing a representation would assert a fact
  not present in the data.
- The user explicitly accepts the current figure.
