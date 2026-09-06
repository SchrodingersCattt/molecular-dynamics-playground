# Five-Problem Self-Review — 01 Velocity Verlet

## Render

- Figure: `01_velocity_verlet`
- Script: `render_slide1_velocity_verlet.py`
- Output inspected: `figures/01_velocity_verlet.png` and MP4 frames at all five stages
- Inspection size / zoom: original 1920 × 1080 and PowerPoint-equivalent full slide
- Programmatic QA result: strict QA PASS; review hash PASS

## Pass 1

1. Scientific expression problem: the force-query step initially lacked a concise statement that the same integrator accepts any differentiable PES.
2. Geometric problem: the first molecule rendering was too small relative to the loop and hid its bond-level motion.
3. Annotation problem: the semantic color legend was initially too close to the final half-kick equation.
4. Readability problem: the original dashboard treatment used too much small text and split attention across boxes.
5. Submission-quality problem: the previous single 53 s montage could not be inserted as three independent PowerPoint videos.

Action taken: rebuilt the slide from scratch as one circular five-stage loop, enlarged the real H2O geometry, moved the legend into reserved whitespace, raised all type to at least 18 pt, and exported a standalone 12 s H.264 video.

## Pass 2

1. Scientific expression problem: no blocking issue remains; equations preserve the half-kick, drift, force-query, half-kick order.
2. Geometric problem: no blocking issue remains; atom-anchored arrows share the same isotropic projected coordinate system as the atoms.
3. Annotation problem: no blocking overlap or leader crossing remains at original resolution.
4. Readability problem: no blocking issue remains; the active stage is black while inactive stages stay light grey.
5. Submission-quality problem: no blocking issue remains; PNG, SVG, and standalone MP4 use one canonical stem.

Action taken: accepted after original-resolution visual inspection; strict QA and review-hash checks are the remaining machine gates.

## Stop Condition

The five listed problems are fixed and no new blocking issue is visible.
