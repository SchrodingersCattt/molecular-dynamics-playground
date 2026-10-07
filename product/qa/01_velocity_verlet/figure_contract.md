# Figure Contract — Velocity Verlet (plastic balls)

## Output

- Canonical stem: `01_velocity_verlet`
- Formats: PNG, SVG, MP4 (19.2 s, 24 fps)
- Final files: `figures/01_velocity_verlet.{png,svg}`, `videos/01_velocity_verlet.mp4`
- Source data: `data/vv_plastic_balls.npz/.json` (generator `scripts/build_box/generate_plastic_vv.py`)
- Intermediate files: `qa/01_velocity_verlet/source/plastic_balls_v1/` and `qa/01_velocity_verlet/_qa/`

## Scientific contract

- Nine identical hard balls (Lennard-Jones, `m = 39.948 amu`, `σ = 3.40 Å`, `ε = 0.060 eV`), `T = 100 K`, `dt = 80 fs`, no solvent.
- Initial velocities follow `velocity all create T seed dist gaussian mom yes rot no`: each component is one signed draw from a zero-mean Gaussian, the centre-of-mass velocity is subtracted, then one global factor rescales to `T`. The movie shows only the draw and assembly; dots and arrows use the final stored velocities, so the two correction steps are not animated.
- Every later state is one velocity-Verlet step (`v_{n+1/2}`, `r_{n+1}`, `v_{n+1}`) with analytic forces; force vs central difference < 1e-10 eV/Å, energy drift ≈ 3 %.
- Arrow lengths are display-scaled (declared in `qa/01_velocity_verlet/asset_manifest.json`); the half-step `Δv` is drawn at the true velocity scale.

## Panels (shared slots with 03/04)

| Panel | Role | Required content |
|---|---|---|
| A | abstract integrator | exact `r → a → v` loop with the active equation inside the circle |
| B | real space | MatterVis hard-plastic sphere pile filling the panel (no floor, drop lines or axis frame; camera roll chosen so the pile spans the wide panel), bottom-left `Simulation step` |
| R (right column) | velocity source | three stacked Gaussian histograms `v_x, v_y, v_z`; one purple dot per ball per histogram |

## Visual story

1. `Initial positions` — `r` lights up; all balls are present from the first frame, each with a blue centre marker (the position `r_i`).
2. `Sample each velocity component from a Gaussian` — `v` lights up; the dots appear on the three histograms.
3. `Three components → one velocity arrow` — each ball's three dots fly together and form one arrow; all arrows grow at once from a common centre above the histograms.
4. `Give every ball its velocity` — all arrows translate in parallel to their balls (small stagger only), then the scene switches to the MatterVis arrows.
5. `Force → acceleration`, `Update velocity` (tip-to-tail `Δv`), `Update position` (blue displacement arrows, centre markers, ghost of the old pose).
6. Six fast velocity-Verlet steps cycle `a → v → r`; each `r` third shows the blue displacement arrows before the balls move.

## Style

- White background; inactive loop light grey; the active stage uses the shared r/v/a colours (r blue `#2F6FB3`, v purple `#7A4FB8`, a orange `#E07A1F`).
- Only symbols, one stage title and the step label carry text; no numbers, no temperature line, no text boxes in the right column.
- Arrows are thin (shaft radius 0.06, small heads), shorter than before, and start at the ball surface; velocity purple, acceleration orange, displacement blue; a colour key sits at the bottom right of the scene. Arrow images switch without crossfade; only ball motion is interpolated.
- Arial, PPT points 14 / 16 / 18 / 24 when the video spans a 16:9 slide; no bold except vectors; static A4 landscape 3508 × 2480 px.

## QA Plan

- Keyframe contact sheet at all phase boundaries, layout registry validation on static and keyframes, decoded-frame sampled QA of the MP4 in `qa/01_velocity_verlet/_qa/sampled_frame_qa.json` (size, edge clipping, gutter ink, panel ink). The every-frame `--strict-video` audit keeps its 48 px bottom-whitespace gate, which the shared lower-left step label does not satisfy (same as 03b).

## Delivery Gate

- [x] Canonical outputs exist.
- [x] Static layout validation and sampled video QA pass.
- [x] Contact sheet reviewed at final size.
