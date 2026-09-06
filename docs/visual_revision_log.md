# Visual revision log

2026-09-04: Started a from-scratch visual pass in the isolated `md-visual` worktree. The four static figures and MP4s will be regenerated with a responsive composition, a shared three-stage rail, short labels, fixed orthographic cameras, and MatterVis-rendered atomistic bases/overlays. Source data and the report are intentionally untouched.

## Tooling note

`mat-vis` 0.0.4 is available locally. I ran `mat-vis inspect --json` on the VV, LJ, AIMD, and 64-water sources and a CPU `mat-vis render --json` smoke test. The custom compositions retain MatterVis CPU scene rendering for atoms, covalent bonds, cells, density meshes, and world-space vectors; Matplotlib is used only for paper-space labels/rail and fixed-grid 2D density contours.

2026-09-04: Added `responsive_story.py` (shared 3-stage rail, alpha-bbox crop of transparent MatterVis margins, fixed 3508×2480/1920×1080 slots), rewrote the VV and TIP3P O–O LJ compositors, and added responsive AIMD/DP wrappers. VV and LJ MatterVis caches use 1700×1180 source scenes with a fixed camera; the VV MP4 was regenerated at 1920×1080/24 fps/9 s. Static outputs for all four stories were regenerated and visually spot-checked. The AIMD wrapper keeps real fixed-grid `rho^k` contour assets and labels SCF residual/iteration state; the DP wrapper exposes O126, rc=6 Å, 83 atomic neighbours, MIC images, and a final `r′,v′` return phase. Existing source/report files were not changed.
