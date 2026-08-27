# Five-Problem Self-Review — 03 AIMD and SCF

## Pass 1

1. Scientific expression: the earlier layout did not clearly multiply one MD force request by repeated SCF work.
2. Geometry: full words were squeezed into elliptical loop nodes, and the H2O dimer was not visually dominant.
3. Annotation: the first 16:9 preview overlapped the right-panel heading with the density label.
4. Readability: force and nuclear-displacement arrows appeared simultaneously in an early end-state draft.
5. Submission quality: the previous density treatment was a 2D contour rather than the requested true 3D isosurface.

Action taken: recomputed 19 RHF iterations and six three-dimensional density volumes; rebuilt the two loops with circular symbol nodes; enlarged the separate dimer scene; removed the redundant heading; and split the ending into force then motion phases.

## Pass 2

1. Scientific expression: no blocking issue remains; density, Fock build, orbital solve, density update, and convergence test repeat before a force is returned.
2. Geometry: no blocking issue remains; the two loops contain no structure, while density, atoms, bonds, forces, and motion share one fixed orthographic transform.
3. Annotation: no blocking issue remains; all labels clear their nodes and panel boundaries at original size.
4. Readability: no blocking issue remains; one active SCF node turns black, force arrows are red, and later motion arrows are green.
5. Submission quality: no blocking issue remains; strict A4 checks pass and all 408 video frames pass.

Action taken: accepted after inspecting request, early, middle, pre-converged, force, and nuclear-motion frames at 1920 × 1080.

## Stop Condition

The five listed problems are fixed and no new blocking issue is visible.
