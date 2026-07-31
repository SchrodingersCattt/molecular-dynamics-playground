# Molecular Dynamics Visualization

Educational animations explaining molecular dynamics from first principles.

## Outputs

| File | Description | Generator |
|------|-------------|-----------|
| `integrator_animations.mp4` | Newton → Euler → Verlet → Velocity Verlet → Leapfrog | Manim |
| `dft_md.mp4` | DFT-MD: SCF convergence, electron density, forces | matplotlib |
| `classical_md.mp4` | Classical FF MD: harmonic bond+angle, force decomposition | matplotlib |
| `deepmd_md.mp4` | DeePMD-kit MD: descriptor, per-atom energy, autograd forces | matplotlib |

## Setup

```bash
conda create -n md_vis python=3.10
conda activate md_vis

# System: ffmpeg + LaTeX (for Manim)
conda install -c conda-forge ffmpeg

# Python packages
pip install -r requirements.txt
```

## Part 1 — Manim Animation

```bash
# Render all 7 scenes as one video (high quality)
manim -pqh integrator_animations/integrators.py MDIntegrators

# Or render individual scenes
manim -pqh integrator_animations/integrators.py Scene1_NewtonSpring
manim -pqh integrator_animations/integrators.py Scene2_ManyBody
manim -pqh integrator_animations/integrators.py Scene3_NumericalPitfalls
manim -pqh integrator_animations/integrators.py Scene4_ExplicitEuler
manim -pqh integrator_animations/integrators.py Scene5_Verlet
manim -pqh integrator_animations/integrators.py Scene6_VelocityVerlet
manim -pqh integrator_animations/integrators.py Scene7_Leapfrog

# Output: media/videos/integrators/1080p60/MDIntegrators.mp4
```

## Part 2a — DFT-MD (PySCF)

```bash
python md_workflows/compute_dft_md.py
# Output: dft_md.mp4
```

Requires `pyscf`. If not installed, falls back to mock SCF data automatically.

## Part 2b — Classical Force Field MD

```bash
python md_workflows/compute_classical_md.py
# Output: classical_md.mp4
```

No external force field files needed — all parameters hardcoded (SPC/E-like).

## Part 2c — DeePMD-kit MD

```bash
# With a real .pb model:
python md_workflows/compute_deepmd_md.py --model /path/to/water.pb

# Without a model (uses mock DeePot network for demonstration):
python md_workflows/compute_deepmd_md.py
# Output: deepmd_md.mp4
```

## Scene Contents

### Part 1 Scenes

1. **Newton's 2nd Law & Spring-Mass** — `F=ma`, SHO analytic solution `x(t)=A cos(ωt+φ)`, energy conservation
2. **Many-Body Problem** — 2-body (solvable), 3-body (coupled ODEs, no closed form), N-body generalization
3. **Numerical Integration Pitfalls** — finite differences, energy drift with large Δt
4. **Explicit Euler** — Taylor expansion derivation, tangent-line geometry, energy growth
5. **Verlet** — forward+backward Taylor, `x(t+Δt) = 2x(t) - x(t-Δt) + a·Δt²`, symplectic
6. **Velocity Verlet** — 4-step algorithm on timeline, energy conservation
7. **Leapfrog** — staggered grid, x/v alternating, comparison table

### Part 2a: DFT-MD Panels

- **3D molecule view** with force arrows
- **SCF convergence** — |ΔE| vs. iteration (log scale), animated per SCF step
- **Electron density** — 2D contour in molecular plane, updated per SCF iteration
- **Force arrows** — from `pyscf.grad.RHF`
- **O-H distance** vs. MD time
- **Energy** (KE, PE, Total) vs. time

### Part 2b: Classical FF Panels

- **2D molecule** with live bond length + angle labels
- **V_bond = ½k(r-r₀)²** parabola with moving dot
- **V_angle = ½k(θ-θ₀)²** parabola with moving dot
- **Force decomposition** — bond forces (blue) + angle forces (orange) + total (black)
- **Geometry text panel** — live FF parameters and current geometry
- **Energy** (KE, PE, Total) vs. time with drift percentage

### Part 2c: DeePMD Panels

- **Molecule + cutoff sphere** — local environment of each atom (cycles O→H₁→H₂)
- **Descriptor bar chart** — `s_ij = f_cut(r)/r` per neighbor, with smooth cutoff inset
- **Per-atom energy** — ε_O, ε_H₁, ε_H₂ bars → E_total = Σ ε_i
- **NN architecture diagram** — input → embedding net → fitting net → ε_i
- **Force arrows** — F = −∂E/∂r (autograd)
- **O-H trajectory** and **energy vs. time**

## Force Field Parameters (Part 2b)

```
K_bond  = 1059.162 kcal/mol/Å²  = 45.93 eV/Å²   (SPC/E O-H)
r₀(O-H) = 1.012 Å
K_angle = 75.90 kcal/mol/rad²   = 3.29 eV/rad²   (SPC/E H-O-H)
θ₀(HOH) = 113.24°
```

## DeePMD Descriptor (Part 2c)

The smooth descriptor used:
```
s_ij = f_cut(r_ij) / r_ij
```
where `f_cut` is a smooth polynomial cutoff function that goes from 1 (r < r_inner)
to 0 (r > r_cut) with continuous derivatives.

The embedding network maps `s_ij → G_i` (invariant feature vector),
and the fitting network maps `G_i → ε_i` (per-atom energy).
Total energy `E = Σ_i ε_i`, forces via autograd `F_i = −∂E/∂r_i`.
