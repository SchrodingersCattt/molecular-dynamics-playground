# Five-Problem Self-Review

## Render

- Figure: A4 landscape comparison of harmonic-oscillator integrators
- Script: `integrator_animations/execute_integrator_notebook.py`
- Output file inspected: `integrator_animations/figures/integrator_comparison.png`
- Inspection size / zoom: native 3507 × 2481 px and fit-to-window
- Programmatic QA result: strict PASS; PDF re-rendered at 300 dpi and visually/numerically cross-checked

## Pass 1

1. Scientific expression problem: the old notebook labelled a hand-written fixed-step method as RK45 and overstated energy conservation; it also used an inexact time grid.
2. Geometric or data-coordinate correctness problem: unstable Explicit Euler traces could leave the shared viewport and later re-enter, drawing misleading vertical crossings.
3. Label, leader, legend, or panel-letter placement problem: the original dense method list competed with the plotted data and weakened the visual hierarchy.
4. Readability or print-scale problem: too many equally prominent traces made the phase-space geometry and long-time energy behavior hard to compare at A4 scale.
5. Submission-quality or filename/output-discipline problem: the two notebook filenames could diverge and the comparison figure lacked canonical, reproducible PNG/PDF outputs.

Action taken: corrected the method to classical RK4, used exact uniform time grids, rewrote the energy-conservation language, clipped divergent trajectories at their first viewport exit with explicit boundary markers, reduced the visible comparison to five representative methods, fixed a shared A4 layout, and generated both notebook copies and both output formats from one execution script.

## Pass 2

1. Scientific expression problem: verified Velocity Verlet second-order convergence (observed orders 2.004 and 2.001) and retained bounded-error rather than exact-conservation language.
2. Geometric or data-coordinate correctness problem: verified equal-aspect phase plots, shared diagnostic axes, and first-exit handling for divergent Explicit Euler trajectories.
3. Label, leader, legend, or panel-letter placement problem: verified the single legend sits in reserved top whitespace and does not overlap titles, axes, or data.
4. Readability or print-scale problem: verified all text is 10–14 pt, the five-method palette remains distinguishable, and the logarithmic energy scale is identical across rows.
5. Submission-quality or filename/output-discipline problem: verified canonical PNG/PDF names, byte-identical notebook copies, embedded output, and a single reproducible execution path.

Action taken: no further visual change was required after the second native-resolution inspection; proceeded to strict programmatic QA.

## Stop Conditions

The five first-pass problems are fixed and the second pass found no new blocking issue.
