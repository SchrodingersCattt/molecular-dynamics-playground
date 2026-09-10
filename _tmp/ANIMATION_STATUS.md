# 03/04 animation revision

Progress snapshot pushed at the user's request before the final render completes.

- `render_dpmd_dynamic.py` is the canonical 04 entry point; `render_dpmd_dynamic_v2.py` is a compatibility alias.
- `deepmd_trace.py` independently replays the retained compressed frozen graph. All six states pass: D error 4.44e-16, maximum per-atom energy error 2.63e-14 eV, and maximum O126 force finite-difference error 7.88e-9 eV/angstrom.
- `dpmd_motion.py` renders 240 display frames through the public MatterVis CLI. The saved MD trajectory has six states and five complete Velocity-Verlet updates. Coordinate interpolation is display sampling during each drift, not extra simulated steps.
- The new renderer restores real text-overlap and canvas-boundary checks; the previous 04 QA report bypassed layout validation and is not evidence for this renderer.
- The committed 04 MP4 is the previous exported draft. The new continuous CLI animation is still being rendered and must pass `preview_current/report.json` and `acceptance.json` before replacing that MP4.
- 03 retains the completed single-geometry clarity fix and its 720-frame export.

Run from the project directory:

```powershell
python _tmp/render_dpmd_dynamic.py --preview-only
python _tmp/render_dpmd_dynamic.py
```

The development MatterVis checkout is detected as `../MatterVis`. The CLI environment may use optional dependencies installed only in `_tmp/.render-runtime`; this directory is excluded from Git. The CPU `general` renderer is used on Windows.

Frozen-model compression reference: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/lib/src/tabulate.cc
